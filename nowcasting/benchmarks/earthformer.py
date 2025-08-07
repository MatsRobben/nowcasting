import torch
import numpy as np

from nowcasting.models.earthformer.cuboid_transformer import CuboidTransformerModel
from nowcasting.utils import data_prep

class EarthformerModel:
    """
    A wrapper class for the EarthFormer model (implemented via CuboidTransformerModel)
    for precipitation nowcasting.

    This class handles the initialization, checkpoint loading, device management,
    and inference for the EarthFormer model, making it compatible with a broader
    evaluation or prediction pipeline. It also includes specific data
    preprocessing/postprocessing steps relevant to radar data.
    """
    def __init__(
        self, config: dict
    ):
        """
        Initializes the EarthFormerModel by instantiating the CuboidTransformerModel
        and loading a pre-trained checkpoint if specified.

        Parameters:
            config : dict
                A dictionary containing configuration parameters for the model.
                Expected keys include:
                - `model.checkpoint_path` (str, optional): Path to the PyTorch
                  checkpoint file (`.pt` or `.pth`) to load model weights from.
                - Other parameters specific to `CuboidTransformerModel`'s constructor
                  (passed via `**config.model`).
        """
        checkpoint_path = config.model.pop('checkpoint_path', None)
        compile = config.model.pop('compile', False)

        self.model = CuboidTransformerModel(**config.model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device))
            model_state_dict = {
                key[len('torch_nn_module.'):] : val
                for key, val in checkpoint["state_dict"].items()
                if key.startswith("torch_nn_module.") and key[len('torch_nn_module.'):] in self.model.state_dict()
            }
            self.model.load_state_dict(model_state_dict, strict=False)

        self.model.eval()
        self.model.to(self.device)

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

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass (inference) with the EarthFormer model.

        This method handles necessary tensor shape permutations, device transfer,
        inference in `no_grad` context, and post-processing of the output.

        Parameters:
            x : torch.Tensor
                The input radar data tensor. Expected shape: (Batch, Channels, Time, Height, Width).
                This will be permuted to (Batch, Time, Height, Width, Channels) for the model.
            y : torch.Tensor
                The target radar data tensor (passed through, but also post-processed).
                Expected shape: (Batch, Channels, Time, Height, Width).

        Returns:
            tuple[np.ndarray, np.ndarray]
                A tuple containing:
                - y_np: The processed target data as a NumPy array.
                - y_hat_np: The model's prediction as a NumPy array.
                Both outputs are in the original data scale (after `data_prep` undo).
        """
        # Shape (B, C, T, W, H) -> Earthformer Shape (B, T, W, H, C)
        x = x.permute(0, 2, 3, 4, 1)

        x = x.to(self.device)

        with torch.no_grad():
            y_hat = self.model(x)

        # Earthformer Shape (B, T, W, H, C) -> (B, C, T, W, H)
        y_hat = y_hat.permute(0, 4, 1, 2, 3)

        y_hat = y_hat.detach().cpu().numpy()

        # Convert dBz to mm/h for evaluation.
        y_hat = self.to_mmh(y_hat)
        y = self.to_mmh(y)

        return y, y_hat