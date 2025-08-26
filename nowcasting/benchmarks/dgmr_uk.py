import gc

import numpy as np
import tensorflow as tf


class DGMRModel:
    """
    A wrapper class for loading and running the DeepMind Generative Model of Radar (DGMR)
    for precipitation nowcasting.

    This class facilitates using a pre-trained TensorFlow SavedModel of DGMR within
    a larger evaluation or prediction framework. It handles model loading, input
    preparation (including transformations and noise generation for stochastic forecasts),
    and output processing.

    Original Model Source:
    - Google DeepMind Research: https://github.com/google-deepmind/deepmind-research/tree/master/nowcasting

    Adaptation Source:
    - MeteoSwiss ldcast repository: https://github.com/MeteoSwiss/ldcast
      (Specifically, inspired by or derived from `eval_dgmr.py` in that project)
    """
    def __init__(
        self, config: dict,
        model_handle: str = "./models/dgmr_uk/256x256",
        future_timesteps: int = 18,
        multi_gpu: bool = True,
        transform_to_rainrate: callable = None,
        transform_from_rainrate: callable = None,
        calibrated: bool = False,
    ):
        """
        Initializes the DGMRModel by loading the TensorFlow SavedModel.

        Parameters:
            model_handle : str, default "./models/dgmr_uk/256x256"
                Path to the directory containing the TensorFlow SavedModel for DGMR.
            future_timesteps : int, default 18
                The number of future timesteps (forecast leadtimes) the model is
                expected to predict.
            multi_gpu : bool, default True
                If True and multiple GPUs are available, TensorFlow's MirroredStrategy
                will be used for distributed computation.
            transform_to_rainrate : callable, optional
                A function to apply to the input data before feeding it to the DGMR model.
                Typically used to transform raw radar units to rain rates (mm/h) if the
                model expects that input scale.
            transform_from_rainrate : callable, optional
                A function to apply to the model's output. Typically used to transform
                rain rates (mm/h) back to original radar units or desired output scale.
            calibrated : bool, default False
                If True, applies a calibration factor (multiplies latent noise 'z' by 2.0).
                This might be used to adjust the spread of the ensemble forecasts.
        """
        model_handle = config.model.get("model_handle", "./models/dgmr_uk/256x256")
        self.future_timesteps = config.model.get("future_timesteps", 18)
        multi_gpu = config.model.get("multi_gpu", True)
        self.calibrated = config.model.get("calibrated", False)

        # These may be None or actual callables injected via config
        self.transform_to_rainrate = config.model.get("transform_to_rainrate", None)
        self.transform_from_rainrate = config.model.get("transform_from_rainrate", None)

        if multi_gpu and len(tf.config.list_physical_devices('GPU')) > 1:
            # initialize multi-GPU strategy
            strategy = tf.distribute.MirroredStrategy()
        else: # use default strategy
            strategy = tf.distribute.get_strategy()
    
        with strategy.scope():
            module = tf.saved_model.load(model_handle)
        self.model = module.signatures['default']

        # Extract information about expected input shapes from the model's signature.
        # This helps in generating correct input tensors (like noise and past frames).
        input_signature = self.model.structured_input_signature[1]
        self.noise_dim = input_signature['z'].shape[1]
        self.past_timesteps = input_signature['labels$cond_frames'].shape[1]

    def __call__(self, x, y):
        """
        Generates a nowcast prediction for a batch of input radar data using DGMR.

        This method allows the DGMRModel instance to be called directly like a function.
        It preprocesses the input, generates latent noise, feeds data to the DGMR
        model, extracts the forecast, and post-processes the output.

        Parameters:
            x : The input radar data. Expected to be a batch, potentially as a list,
                tuple, or NumPy array. The internal logic handles converting to
                NumPy and transposing to the format DGMR expects.
                Assumed input shape: (Batch, Channels, Time_in, Height, Width).

            y : The target data (passed through unchanged, as DGMR is an autoregressive
                generative model and doesn't use `y` for its forward pass).

        Returns:
            A tuple containing:
            - y: The original target data (passed through).
            - y_hat: The forecasted precipitation field, a NumPy array of shape
                (Batch, Channels, Time_out, Height, Width, [Ensemble_Size if `create_ensemble`]).
                Note: This `__call__` method generates a single forecast. For ensembles,
                `create_ensemble` function should be used to call this multiple times.
        """       
        x = x.detach().numpy()
        y = y.detach().numpy()

        # Shape (B, C, T, W, H) -> Shape (B, T, W, H, C)
        x = x.transpose(0, 2, 3, 4, 1)

        if self.transform_to_rainrate is not None:
            x = self.transform_to_rainrate(x)        
        x = tf.convert_to_tensor(x)

        num_samples = x.shape[0]
        z = tf.random.normal(shape=(num_samples, self.noise_dim))
        if self.calibrated:
            z = z * 2.0

        onehot = tf.ones(shape=(num_samples, 1))
        inputs = {
            "z": z,
            "labels$onehot" : onehot,
            "labels$cond_frames" : x
        }
        y_hat = self.model(**inputs)['default']

        y_hat = y_hat[:,self.past_timesteps:self.past_timesteps+self.future_timesteps,...]

        y_hat = np.array(y_hat)

        # PySteps Shape (B, T, W, H, C) -> (B, C, T, W, H)
        y_hat = y_hat.transpose(0, 4, 1, 2, 3)

        if self.transform_from_rainrate is not None:
            y_hat = self.transform_from_rainrate(y_hat)

        return y, y_hat


def create_ensemble(
    dgmr,
    x,
    ensemble_size: int = 32,
) -> np.ndarray:
    """
    Generates an ensemble of DGMR forecasts for a single input sequence.

    This function repeatedly calls the `dgmr.__call__` method, which generates
    a single stochastic forecast. By collecting multiple such forecasts, an
    ensemble is created.

    Parameters:
        dgmr : DGMRModel
            An initialized DGMRModel instance.
        x : Any
            The input radar data for which to create an ensemble.
            This `x` should represent a single batch, typically (B, C, T_in, H, W).
        ensemble_size : int, default 32
            The number of ensemble members to generate.

    Returns:
        np.ndarray
            A NumPy array representing the ensemble forecast.
            Shape: (Batch, Channels, Time_out, Height, Width, Ensemble_Size).
    """
    y_pred = []
    for member in range(ensemble_size):
        print(f"Generating member {member+1}/{ensemble_size}")
        y_pred.append(dgmr(x))
    gc.collect()
    
    y_pred = np.stack(y_pred, axis=-1)
    return y_pred
