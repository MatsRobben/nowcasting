import torch
from torch.amp import autocast
import numpy as np
from typing import Union, Tuple

from nowcasting.models.ldcast.models.diffusion import plms
from nowcasting.models.ldcast.models.diffusion.diffusion import LatentDiffusion

class LDCast:
    """
    A wrapper for the diffusion-based probabilistic LDCast model using PLMS sampling.

    Loads the trained GenForecast model from a checkpoint and performs inference with
    a configurable number of diffusion iterations and ensemble members.
    """

    def __init__(self, config: dict,
                 use_amp: bool = True,):
        """
        Initializes the LDCast model.

        Expected config keys:
            config.model.checkpoint_path : str
                Path to the diffusion model checkpoint (.pt)
            config.model.num_diffusion_iters : int
                Number of diffusion sampling iterations
            config.model.ensemble_size : int
                Number of ensemble members to generate
        """
        checkpoint_path = config.model["checkpoint_path"]
        compile_model = config.model.pop("compile", False)
        self.num_diffusion_iters = config.model.get("num_diffusion_iters", 50)
        self.ensemble_size = config.model.get("ensemble_size", 32)

        # Add new config parameters
        self.ensemble_chunk_size = config.model.get("ensemble_chunk_size", 8)
        self.fast_sampling = config.model.get("fast_sampling", True)
        self.use_inference_mode = config.model.get("use_inference_mode", True)
        self.optimize_sampler = config.model.get("optimize_sampler", True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_amp = use_amp
        if self.use_amp and self.device == "cpu":
            print("Warning: Mixed precision (AMP) is only available on CUDA devices.")
            self.use_amp = False
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if self.device == 'cpu' and self.use_amp else torch.float16
        self.context_manager = (torch.inference_mode if self.use_inference_mode 
                                else torch.no_grad)()

        self.ldm = LatentDiffusion.load_from_checkpoint(checkpoint_path, config=config)
        self.ldm.to(self.device)

        # if compile_model:
        #     print("Compiling model with torch.compile...")
        #     self.ldm = torch.compile(self.ldm, mode="default")

        self.ldm.eval()
        self.sampler = plms.PLMSSampler(self.ldm)

    def _move_to_device(self, x: Union[torch.Tensor, list, tuple], device: torch.device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(self._move_to_device(v, device) for v in x)

    def to_mmh(self, img, norm_method='zscore', mean=0.03019706713265408, std=0.5392297631902654):
        """
        Convert normalized precipitation data to mm/h.

        Parameters:
            img (np.ndarray or torch.Tensor): Normalized input data.
            norm_method (str): 'zscore' or 'minmax'.
            mean (float): Mean for z-score normalization.
            std (float): Std for z-score normalization.

        Returns:
            img_mmh: Precipitation in mm/h (clipped to [0, 160]).
        """
        if norm_method == 'zscore':
            img = img * std + mean
            img_mmh = 10 ** img

        elif norm_method == 'minmax':
            # Undo minmax normalization to [0, 55] dBZ
            img = img * 55

            # Convert from dBZ to mm/h (in-place)
            if isinstance(img, np.ndarray):
                mask = img != 0
                img[mask] = ((10**(img[mask] / 10) - 1) / 200) ** (5 / 8)
            elif isinstance(img, torch.Tensor):
                mask = img != 0
                img[mask] = ((10**(img[mask] / 10) - 1) / 200) ** (5 / 8)
            else:
                raise TypeError("Input must be a numpy array or a torch tensor")

            img_mmh = img

        else:
            raise ValueError("norm_method must be 'zscore' or 'minmax'")

        # Clip final mm/h values to physical range
        if isinstance(img_mmh, torch.Tensor):
            img_mmh = torch.clamp(img_mmh, 0.0, 160.0)
        elif isinstance(img_mmh, np.ndarray):
            img_mmh = np.clip(img_mmh, 0.0, 160.0)

        return img_mmh

    def __call__(
        self, x: Union[torch.Tensor, list, tuple], y: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs probabilistic inference with the diffusion model.

        Parameters:
            x : model input (e.g., conditioning sequence)
            y : ground truth target (for comparison or evaluation)

        Returns:
            Tuple of:
            - y_true_processed: Ground truth tensor (NumPy) in original scale, trimmed to 18 frames.
            - y_hat_ensemble: Model predictions as NumPy array, shape (B, C, T, H, W, E)
        """
        x = self._move_to_device(x, self.device)
        y = y.to(self.device)

        batch_size = y.shape[0]

        gen_shape = (32, 5) + (y.shape[-2] // 4, y.shape[-1] // 4)
        ensemble_outputs = []

        for _ in range(self.ensemble_size):
            with self.context_manager, autocast(device_type=self.device, 
                                                dtype=self.amp_dtype, 
                                                enabled=self.use_amp):
                z_sample, _ = self.sampler.sample(
                    self.num_diffusion_iters,
                    batch_size,
                    gen_shape,
                    x,
                    progbar=False,
                )
                decoded = self.ldm.autoencoder.decode(z_sample)
                ensemble_outputs.append(decoded)

        y_hat_ensemble = torch.stack(ensemble_outputs, dim=-1)  # (B, C, T, H, W, E)
        if self.ensemble_size == 1:
            y_hat_ensemble = y_hat_ensemble.squeeze(-1)  # -> (B, C, T, H, W)

        # Convert dBz to mm/h for evaluation.
        y_hat_ensemble = self.to_mmh(y_hat_ensemble)
        y = self.to_mmh(y)

        return y[:, :, :18], y_hat_ensemble[:, :, :18]
