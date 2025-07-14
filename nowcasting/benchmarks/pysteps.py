# following https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html

from datetime import timedelta

import dask
import numpy as np
from pysteps import nowcasts
from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps.utils import transformation

class PySTEPSModel:
    """
    A wrapper class for integrating PySTEPS nowcasting methods into a
    broader evaluation or benchmarking framework.

    This class encapsulates the logic for running PySTEPS' STEPS (Stochastic
    Ensemble Precipitation Forecasting System) method, handling data
    transformations, motion estimation, and ensemble forecasting. It's designed
    to be initialized with a configuration object and then called like a model
    to make predictions on input radar data.

    References:
        - PySTEPS documentation: https://pysteps.readthedocs.io/en/stable/
        - Example: https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html
        - Code modified from original: https://github.com/MeteoSwiss/ldcast
    """
    def __init__(
        self, config
    ):
        """
        Initializes the PySTEPSModel with configuration parameters.

        Parameters:
            config : OmegaConf.DictConfig
                An OmegaConf configuration object containing model-specific
                parameters, typically under a 'model' key. Expected parameters include:
                - `model.transform_to_rainrate` (callable, optional): Function to
                  transform input data to rain rates before PySTEPS processing.
                - `model.transform_from_rainrate` (callable, optional): Function to
                  transform output rain rates back to original data format.
                - `model.future_timesteps` (int): Number of future timesteps to predict.
                - `model.ensemble_size` (int): Number of ensemble members for STEPS.
                - `model.km_per_pixel` (float): Spatial resolution of the input data
                  in kilometers per pixel.
        """
        self.transform_to_rainrate = config.model.get('transform_to_rainrate', None)
        self.transform_from_rainrate = config.model.get('transform_from_rainrate', None)
        self.nowcast_method = nowcasts.get_method("steps")
        self.future_timesteps = config.model.get('future_timesteps', 20)
        self.ensemble_size = config.model.get('ensemble_size', 32)
        self.km_per_pixel = config.model.get('km_per_pixel', 1.0)
        self.interval = timedelta(minutes=5)

    def zero_prediction(self, R: np.ndarray, zerovalue: float) -> np.ndarray:
        """
        Generates an array of zero predictions, used as a fallback when
        no precipitation is detected or PySTEPS encounters an error.

        Parameters:
            R : np.ndarray
                The input radar data array (used to infer spatial dimensions).
            zerovalue : float
                The 'zero' value in the transformed (dB) space.

        Returns:
            np.ndarray
                An array of shape (future_timesteps, height, width, ensemble_size)
                filled with `zerovalue`.
        """
        out_shape = (self.future_timesteps,) + R.shape[1:] + \
            (self.ensemble_size,)
        return np.full(out_shape, zerovalue, dtype=R.dtype)

    def predict_sample(self, x: np.ndarray, threshold: float = -10.0, zerovalue: float = -15.0) -> np.ndarray:
        """
        Performs a single PySTEPS nowcast for one input radar sequence.

        This method handles the full PySTEPS workflow:
        1. Optional transformation of input to rain rate.
        2. dB transformation of input.
        3. Motion field estimation using Lucas-Kanade.
        4. Running the STEPS nowcasting algorithm.
        5. Handling potential errors (e.g., all zeros input).
        6. Inverse dB transformation to get rain rates.
        7. Optional inverse transformation to original data format.

        Parameters:
            x : np.ndarray
                The input radar data for a single sample. Expected shape:
                (time_steps, height, width).
            threshold : float, default -10.0
                The dBZ threshold for precipitation. Values below this are
                considered non-precipitating.
            zerovalue : float, default -15.0
                The value used to represent "no precipitation" after dB transformation.

        Returns:
            np.ndarray
                The forecasted precipitation field, typically of shape
                (future_timesteps, height, width, ensemble_size).
        """
        if self.transform_to_rainrate is not None:
            R = self.transform_to_rainrate(x)
        else:
            R = x

        (R, _) = transformation.dB_transform(
            R, threshold=0.1, zerovalue=zerovalue
        )

        print("R stats:", R.shape, np.min(R), np.max(R), np.isnan(R).sum())

        R[~np.isfinite(R)] = zerovalue

        if (R == zerovalue).all():
            R_f = self.zero_prediction(R, zerovalue)
        else:
            V = dense_lucaskanade(R, verbose=False);

            print("Vector field stats:", V.shape, np.min(V), np.max(V), np.isnan(V).sum())
            try:
                R_f = self.nowcast_method(
                    R,
                    V,
                    self.future_timesteps,
                    n_ens_members=self.ensemble_size,
                    n_cascade_levels=6,
                    precip_thr=threshold,
                    kmperpixel=self.km_per_pixel,
                    timestep=self.interval.total_seconds()/60,
                    noise_method="nonparametric",
                    vel_pert_method="bps",
                    mask_method="incremental",
                    num_workers=self.future_timesteps,
                );
                R_f = R_f.transpose(1,2,3,0)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                zero_error = str(e).endswith("contains non-finite values") or \
                    str(e).startswith("zero-size array to reduction operation") or \
                    str(e).endswith("nonstationary AR(p) process") or \
                    str(e).endswith("Singular matrix")
                if zero_error:
                    # occasional PySTEPS errors that happen with little/no precip
                    # therefore returning all zeros makes sense
                    R_f = self.zero_prediction(R, zerovalue)
                else:
                    raise

        # Back-transform to rain rates
        R_f = transformation.dB_transform(
            R_f, threshold=threshold, inverse=True
        )[0]

        R_f[np.isnan(R_f)] = 0

        if self.transform_from_rainrate is not None:
            R_f = self.transform_from_rainrate(R_f)

        return R_f

    def __call__(self, x, y, parallel: bool = True):
        """
        Makes a prediction for a batch of input data. This method allows the
        PySTEPSModel to be used like a function (e.g., `model(x, y)`).

        It handles batch processing by applying `predict_sample` to each item
        in the batch, optionally using Dask for parallelization.

        Parameters:
            x : Any
                The input data. Expected to be a batch, potentially as a list, tuple,
                or NumPy array. The inner most array representing a single sample
                is expected to have shape (C, T_in, W, H), where C is channel (should be 1).
                The method will transpose it to PySTEPS compatible (T_in, W, H, C).
            y : Any
                The target data (passed through, not used by PySTEPS for prediction).
            parallel : bool, default True
                If True, use Dask to parallelize `predict_sample` calls across the batch.

        Returns:
            Tuple[Any, np.ndarray]
                A tuple containing:
                - y: The original target data (passed through unchanged).
                - y_hat: The predicted precipitation field, a NumPy array of shape
                  (Batch, Time, Height, Width, Channel=1, Ensemble_Size).
        """ 
        while isinstance(x, list) or isinstance(x, tuple):
            x = x[0]

        x = np.asarray(x)

        # Shape (B, C, T, W, H) -> PySteps Shape (B, T, W, H, C)
        x = x.transpose(0, 2, 3, 4, 1)

        print("Input stats:", x.shape, np.min(x), np.max(x), np.isnan(x).sum())

        pred = self.predict_sample
        if parallel:
            pred = dask.delayed(pred)
        y_hat = [
            pred(x[i,:,:,:,0]) 
            for i in range(x.shape[0])    
        ]
        if parallel:
            y_hat = dask.compute(y_hat, scheduler="threads", num_workers=len(y_hat))[0]
        y_hat = np.stack(y_hat, axis=0)

        print("y_hat stats:", x.shape, np.min(x), np.max(x), np.isnan(x).sum())

        # Shape (B, T, W, H, C) -> PySteps Shape (B, C, T, W, H)
        x = x.transpose(0, 4, 1, 2, 3)

        return y, y_hat