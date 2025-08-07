import importlib
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict, Any, Optional

from fire import Fire
from omegaconf import OmegaConf

from nowcasting.validation.metrics import Evaluator

def load_module(path: str):
    """
    Dynamically loads a Python module or an object (e.g., a class or function)
    from a given string path.

    This function attempts to import the path directly as a module. If that
    fails (e.g., it's a path to a class within a module), it splits the path
    and tries to import the module first, then retrieve the object by name.

    Parameters:
        path : str
            The string path to the module or object to load.
            Examples: "my_package.my_module", "my_package.my_module.MyClass".

    Returns:
        module or object
            The loaded Python module or the specified object (class, function).

    Raises:
        ImportError: If the module or object cannot be found or loaded.
    """
    try:
        # First, try to import it as a full module
        return importlib.import_module(path)
    except ModuleNotFoundError as e:
        # If it fails, split into module + object
        if "." not in path:
            raise ImportError(f"Invalid import path: '{path}' is not a module or object path.")

        module_path, object_name = path.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, object_name)
        except (ModuleNotFoundError, AttributeError) as e2:
            raise ImportError(
                f"Could not import '{object_name}' from module '{module_path}': {e2}"
            ) from e2

def evaluation(
    model,
    dataloader: torch.utils.data.DataLoader,
    model_name: str = "pysteps",
    eval_name: str = "val",
    num_samples: Optional[int] = None,
    leadtimes: List[int] = [5, 10, 15, 30, 60, 90],
    thresholds: List[float] = [0, 0.5, 1, 2, 5, 10, 30],
):
    """
    Evaluates a nowcasting model on a given dataset, accumulating and
    printing various verification metrics.

    This function iterates through the dataloader, feeds data to the model,
    and uses an `Evaluator` object to compute categorical, continuous, and FSS
    scores for specified leadtimes and thresholds.

    Parameters:
        model : 
            The nowcasting model to be evaluated. It should be a callable that
            takes `x` (input) and `y` (target) and returns `y_target`, `y_pred`.
        dataloader : torch.utils.data.DataLoader
            The DataLoader providing batches of input (`x`) and target (`y`) data.
        model_name : str, default "pysteps"
            A name identifying the current model being evaluated. Used for saving scores.
        eval_name : str, default "val"
            A name for the evaluation run (e.g., 'val', 'test', 'high_intensity').
            Used to organize saved scores on disk.
        num_samples : Optional[int], default None
            The total number of samples (time steps) to evaluate. If None,
            the entire dataloader is processed.
        leadtimes : List[int], default [5, 10, 15, 30, 60, 90]
            A list of lead times (in minutes) for which to compute metrics.
            The model's output is expected to have a time dimension that
            corresponds to these leadtimes (e.g., if leadtime 5 is the first,
            then array index `leadtime//5 - 1` is used).
        thresholds : List[float], default [0, 0.5, 1, 2, 5, 10, 30]
            A list of rainfall intensity thresholds (in mm/h) for categorical
            metrics and FSS.
    """
    if num_samples is None:
        num_batches = float('inf')
    else:
        num_batches = num_samples // dataloader.batch_size
    
    evaluator = Evaluator(
        nowcast_method=model_name,
        dir=eval_name,
        thresholds=thresholds,
        leadtimes=leadtimes,
        save_after_n_samples=dataloader.batch_size # Save scores after processing each batch
    )

    for idx, (x,y) in tqdm(enumerate(dataloader), desc="Verifying batches"):
        # Pass input (x) and target (y) through the model to get the forecast (y_hat).
        # The model's __call__ method is expected to return (y_target, y_predicted).
        # This is done to preform the same post-processing on y and y_hat.
        y, y_hat = model(x, y)

        # Handle ensemble dimension for models like DGMR UK which might output
        # forecasts as (Batch, Time, Channel, Height, Width, Ensemble_Size).
        if len(y_hat.shape) == 6:
            y_hat = np.mean(y_hat, axis=-1)
        
        # Convert PyTorch tensors to NumPy arrays if necessary for compatibility with Evaluator.
        if isinstance(y_hat, torch.Tensor): y_hat = y_hat.detach().numpy()
        if isinstance(y, torch.Tensor): y = y.detach().numpy()

        for y_target, y_pred in zip(y, y_hat):
            for i, leadtime in enumerate(leadtimes):
                # Extract the target and forecast radar images for the current leadtime.
                # The `leadtime//5 - 1` indexing assumes leadtimes are multiples of 5
                # and correspond to 0-indexed time steps (e.g., 5min -> index 0, 10min -> index 1).
                R_target = np.squeeze(y_target)[leadtime//5-1]
                R_forecast = np.squeeze(y_pred)[leadtime//5-1]
                
                evaluator.verify(R_target, R_forecast, leadtime=leadtime)

        if idx >= num_batches:
            break

    # After iterating through all batches, retrieve the final computed scores.
    cat_scores, cont_scores, fss_scores = evaluator.get_scores()
    print("\n--- Evaluation Results ---")
    print("Categorical scores:", cat_scores)
    print("Continuous scores:", cont_scores)
    print("Fss scores:", fss_scores)
    print("Evaluation completed.")

def setup_dataloader(
    module_path: str = "nowcasting.data.dataloader.RadarDataModule",
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Initializes and sets up a data module and returns its validation dataloader.

    This function dynamically loads a data module class (e.g., `RadarDataModule`),
    instantiates it with provided keyword arguments, sets it up, and then
    returns the validation dataloader.

    Parameters:
        module_path : str, default "nowcasting.data.dataloader.RadarDataModule"
            The string path to the data module class (e.g., "my_package.data.MyDataModule").
        **kwargs : Any
            Arbitrary keyword arguments that will be passed to the data module's
            constructor.

    Returns:
        torch.utils.data.DataLoader
            The validation dataloader ready for use in evaluation.
    """
    print(f"Setting up dataloader from: {module_path}")
    data_module = load_module(module_path)

    data_module = data_module(
        **kwargs
    )
    data_module.setup()
    print("Dataloader setup complete.")
    return data_module.val_dataloader()

def setup_model(
    module_path: str,
    config: OmegaConf
) -> torch.nn.Module:
    """
    Dynamically loads and initializes a nowcasting model based on the provided
    module path and configuration.

    The model's constructor is expected to accept the OmegaConf `config` object
    as its primary argument. It's assumed that the 'module_path' itself is not
    a parameter for the model's constructor and is removed from the config.model
    section before instantiation.

    Parameters:
        module_path : str
            The string path to the model class (e.g., "my_package.models.MyNowcastModel").
        config : OmegaConf
            The OmegaConf configuration object containing all necessary parameters
            for the model (typically under `config.model`).

    Returns:
        torch.nn.Module
            An initialized instance of the nowcasting model.
    """
    print(f"Setting up model from: {module_path}")
    model = load_module(module_path)

    # Remove 'module_path' from the config.model section before passing the config
    # to the model's constructor. This is only needed for training the model.
    if 'module_path' in config.model:
        config.model.pop('module_path')

    model = model(config)
    print("Model setup complete.")
    return model

def main(config: Optional[str] = None, **kwargs):
    """
    Main entry point for the evaluation script.

    This function handles loading the configuration, setting up the dataloader
    and model, and then initiating the evaluation process. It's designed to
    be executed via the 'fire' library, allowing configuration via a YAML file
    or command-line arguments.

    Parameters:
        config : Optional[str], default None
            Path to a YAML or JSON configuration file. If provided, settings
            from this file will be loaded.
        **kwargs : Any
            Arbitrary keyword arguments passed from the command line. These
            arguments will override or extend settings loaded from the config file.

    Usage from command line:
    `python evaluate.py --config="path/to/your_config.yaml"`
    """
    print("Starting main evaluation process.")
    
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)

    dataloader = setup_dataloader(**config.dataloader)

    model = setup_model(
        module_path=config.eval.pop('model_path'),
        config=config
    )
    
    evaluation(model, dataloader, **config.eval)

    print("Main evaluation process finished.")

if __name__ == "__main__":
    Fire(main)