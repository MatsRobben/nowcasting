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
        self.method_name = config.model.get("method", "steps").lower()
        self.nowcast_method = nowcasts.get_method(self.method_name)
        # common
        self.future_timesteps = config.model.get("future_timesteps", 20)
        self.km_per_pixel = config.model.get("km_per_pixel", 1.0)
        self.interval = timedelta(minutes=5)

        # STEPS-specific
        self.ensemble_size = config.model.get("ensemble_size", 32)
        self.n_cascade_levels = config.model.get("n_cascade_levels", 6)
        self.noise_method = config.model.get("noise_method", "nonparametric")
        self.vel_pert_method = config.model.get("vel_pert_method", "bps")
        self.mask_method = config.model.get("mask_method", "incremental")

        # S‑PROG-specific
        self.ar_order = config.model.get("ar_order", 2)
        self.decomp_method = config.model.get("decomp_method", "fft")
        self.bandpass_filter = config.model.get("bandpass_filter", "gaussian")
        self.probmeth = config.model.get("probmatching_method", None)

        self.transform_to_rainrate = config.model.get("transform_to_rainrate", None)
        self.transform_from_rainrate = config.model.get("transform_from_rainrate", None)

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
                For STEPS: (future_timesteps, height, width, ensemble_size)
                For S-PROG: (future_timesteps, height, width, 1)
        """
        spatial_shape = R.shape[1:]

        if self.method_name == "sprog":
            out_shape = (self.future_timesteps,) + spatial_shape + (1,)
        else:  # steps or other ensemble-based methods
            out_shape = (self.future_timesteps,) + spatial_shape + (self.ensemble_size,)

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
            V = dense_lucaskanade(R, verbose=False)
            try:
                if self.method_name == "sprog":
                    try:
                        arr = self.nowcast_method(
                            R, V,
                            self.future_timesteps,
                            precip_thr=threshold,
                            n_cascade_levels=self.n_cascade_levels,
                            ar_order=self.ar_order,
                            decomp_method=self.decomp_method,
                            bandpass_filter_method=self.bandpass_filter,
                            probmatching_method=self.probmeth,
                            measure_time=False
                        )
                        R_f = arr[..., np.newaxis]
                    except IndexError as e:
                        if "out of bounds" in str(e):
                            # Known PySTEPS S-PROG bug with uniform fields
                            R_f = self.zero_prediction(R, zerovalue)
                        else:
                            raise
                else:  # steps
                    arr = self.nowcast_method(
                        R, V,
                        self.future_timesteps,
                        n_ens_members=self.ensemble_size,
                        n_cascade_levels=self.n_cascade_levels,
                        precip_thr=threshold,
                        kmperpixel=self.km_per_pixel,
                        timestep=self.interval.total_seconds()/60,
                        noise_method=self.noise_method,
                        vel_pert_method=self.vel_pert_method,
                        mask_method=self.mask_method,
                        probmatching_method=self.probmeth,
                        num_workers=self.future_timesteps
                    )
                    R_f = arr.transpose(1, 2, 3, 0)
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
        x = np.asarray(x)
        y = np.asarray(y)

        # Shape (B, C, T, W, H) -> PySteps Shape (B, T, W, H, C)
        x = x.transpose(0, 2, 3, 4, 1)

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

        # Shape (B, T, W, H, C) -> PySteps Shape (B, C, T, W, H)
        y_hat = y_hat.transpose(0, 4, 1, 2, 3)

        return y, y_hat