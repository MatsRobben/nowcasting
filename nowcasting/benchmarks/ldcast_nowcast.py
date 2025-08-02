import torch
import numpy as np
from typing import Union, Tuple

from nowcasting.models.ldcast.models.nowcast.nowcast import AFNONowcastModule
from nowcasting.utils import data_prep

class LDCastNowcastNet:
    """
    A wrapper class for an AFNO-based nowcasting model, specifically designed
    to load and utilize models trained and checkpointed within the `ldcast` framework.

    This class provides functionality for loading a pre-trained model from a
    PyTorch Lightning checkpoint, managing device placement, and performing
    inference on radar data, including necessary pre- and post-processing steps.
    """
    def __init__(
        self, config: dict,
    ):
        """
        Initializes the LDCastNowcastNet by loading the AFNONowcastModule from a checkpoint.

        Parameters:
            config : dict
                A dictionary containing configuration parameters for the model.
                Expected keys within `config.model` include:
                - `checkpoint_path` (str, optional): Path to the PyTorch Lightning
                  checkpoint file (`.ckpt`) to load model weights from.
                - `compile` (bool, optional): Whether to compile the model using
                  `torch.compile` (requires PyTorch 2.0+ for performance optimization).
                  Defaults to False.
                - Other parameters specific to `AFNONowcastModule`'s constructor
                  (passed via the `config` object during `load_from_checkpoint`).
        """

        checkpoint_path = config.model.pop('checkpoint_path', None)
        compile = config.model.pop('compile', False)

        module = AFNONowcastModule.load_from_checkpoint(checkpoint_path, config=config)
        self.model = module.model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.eval()
        self.model.to(self.device)

    def _move_to_device(self, x: Union[torch.Tensor, list, tuple], device: torch.device):
        """
        Recursively moves PyTorch tensors within a nested structure (tensor, list, tuple)
        to the specified device.

        This helper function is crucial when the model's input `x` might not be a
        single tensor but a more complex data structure containing tensors.

        Parameters:
            x : Union[torch.Tensor, list, tuple]
                The input data, which can be a single tensor or a collection of tensors.
            device : torch.device
                The target device (e.g., 'cuda', 'cpu') to move the tensors to.

        Returns:
            Union[torch.Tensor, list, tuple]
                The input structure with all contained tensors moved to the specified device.
        """
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(self._move_to_device(v, device) for v in x)

    def __call__(self, x: Union[torch.Tensor, list, tuple], y: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass (inference) with the LDCast nowcasting model.

        This method handles moving input data to the correct device, performing
        the prediction, and post-processing both the prediction and target data.

        Parameters:
            x : Union[torch.Tensor, list, tuple]
                The input radar data. Expected to contain PyTorch tensors that
                the model can process. Typically: (Batch, Channels, Time, Height, Width).
            y : torch.Tensor
                The target radar data tensor. Expected shape: (Batch, Channels, Time, Height, Width).

        Returns:
            Tuple[np.ndarray, np.ndarray]
                A tuple containing:
                - y_processed: The processed target data as a NumPy array, trimmed to 18 timesteps.
                - y_hat_processed: The model's prediction as a NumPy array, trimmed to 18 timesteps.
                Both outputs are in the original data scale (after `data_prep` undo).
        """

        x = self._move_to_device(x, self.device)

        with torch.no_grad():
            y_hat = self.model(x)

        y_hat = y_hat.detach().cpu().numpy()

        # Convert dBz to mm/h for evaluation.
        data_prep(y_hat, convert_to_dbz=True, undo=True)
        data_prep(y, convert_to_dbz=True, undo=True)

        # Warning: this is hardcoded! LDCast has to predict in multiple of 4, so it was trained
        # to predict 20 frames, however, the other models only produce 18 fames.
        return y[:, :, :18], y_hat[:, :, :18]

