import torch
import numpy as np
from typing import Union, Tuple
from nowcasting.utils import data_prep
from nowcasting.models.ldcast.models.diffusion import plms
from nowcasting.models.ldcast.models.diffusion.diffusion import LatentDiffusion

class LDCast:
    """
    A wrapper for the diffusion-based probabilistic LDCast model using PLMS sampling.

    Loads the trained GenForecast model from a checkpoint and performs inference with
    a configurable number of diffusion iterations and ensemble members.
    """

    def __init__(self, config: dict):
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
        self.num_diffusion_iters = config.model.get("num_diffusion_iters", 50)
        self.ensemble_size = config.model.get("ensemble_size", 32)

        self.ldm = LatentDiffusion.load_from_checkpoint(checkpoint_path, config=config)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm.to(self.device)

        self.ldm.eval()
        self.sampler = plms.PLMSSampler(self.ldm)

    def _move_to_device(self, x: Union[torch.Tensor, list, tuple], device: torch.device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(self._move_to_device(v, device) for v in x)

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
        batch_size = y.shape[0]

        gen_shape = (32, 5) + (y.shape[-2] // 4, y.shape[-1] // 4)
        ensemble_outputs = []

        for _ in range(self.ensemble_size):
            with torch.no_grad():
                z_sample, _ = self.sampler.sample(
                    self.num_diffusion_iters,
                    batch_size,
                    gen_shape,
                    x,
                    progbar=False,
                )
                decoded = self.ldm.autoencoder.decode(z_sample)
                ensemble_outputs.append(decoded.cpu())

        y_hat_ensemble = torch.stack(ensemble_outputs, dim=-1).numpy()  # (B, C, T, H, W, E)
        if self.ensemble_size == 1:
            y_hat_ensemble = np.squeeze(y_hat_ensemble, axis=-1)  # -> (B, C, T, H, W)
        y_np = y.cpu().numpy()

        data_prep(y_hat_ensemble, convert_to_dbz=True, undo=True)
        data_prep(y_np, convert_to_dbz=True, undo=True)

        return y_np[:, :, :18], y_hat_ensemble[:, :, :18]
