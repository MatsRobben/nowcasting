import importlib
import numpy as np
import os
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# For tracking carbon emissions during training
from codecarbon import EmissionsTracker

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import torch

from fire import Fire
from omegaconf import OmegaConf

@rank_zero_only
def copy_config_to_log_dir(config_path: Optional[str], trainer: Trainer):
    """
    Copies the main configuration file to the TensorBoard log directory.

    This ensures that the exact configuration used for a training run is
    saved alongside the logs and checkpoints, making it easy to reproduce
    or inspect past experiments. This operation is performed only on the
    rank-zero process in a distributed training setup to avoid redundant copies.

    Parameters:
        config_path : Optional[str]
            The file path to the main configuration (e.g., a YAML file).
            If None or if the trainer has no logger, no action is taken.
        trainer : lightning.pytorch.Trainer
            The PyTorch Lightning Trainer instance, used to access its logger
            and determine the log directory.
    """
    if config_path and trainer.logger:
        tb_dir = trainer.logger.log_dir
        os.makedirs(tb_dir, exist_ok=True)
        shutil.copy(
            config_path,
            os.path.join(tb_dir, os.path.basename(config_path))
        )
        print(f"Copied config {config_path} → {tb_dir}")

def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """
    Finds the path to the latest `last.ckpt` checkpoint file in a given directory.

    It specifically looks for 'last.ckpt' or 'last-vX.ckpt' patterns,
    prioritizing higher version numbers for 'last-vX.ckpt'.

    Parameters:
        checkpoint_dir : pathlib.Path
            The directory to search for checkpoint files.

    Returns:
        Optional[pathlib.Path]
            The path to the most recent 'last.ckpt' file, or None if no
            such file is found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    # Regex to match 'last.ckpt' or 'last-vX.ckpt'
    pattern = re.compile(r"last(?:-v(\d+))?\.ckpt$")

    latest_file = None
    max_version = -1 # Initialize with -1 to handle 'last.ckpt' (version 0) correctly

    for file in checkpoint_dir.glob("last*.ckpt"):
        match = pattern.fullmatch(file.name)
        if match:
            version_str = match.group(1)
            # If 'version_str' is None (for 'last.ckpt'), default to version 0.
            # Otherwise, parse the version number.
            version = int(version_str) if version_str else 0 
            if version > max_version:
                max_version = version
                latest_file = file

    return latest_file

def load_version(version_path: str) -> Tuple[OmegaConf, TensorBoardLogger, Optional[Path]]:
    """
    Loads configuration, logger, and the latest checkpoint for a specific
    Lightning logging version.

    This function is useful for resuming training or evaluation from a
    previously run experiment by simply pointing to its version directory.
    It reconstructs the TensorBoard logger and finds the last saved checkpoint
    and the original configuration file.

    Parameters:
        version_path : str
            The path to a specific `version_X` directory within a TensorBoard
            experiment log directory.

    Returns:
        Tuple[OmegaConf, TensorBoardLogger, Optional[pathlib.Path]]
            - config: The OmegaConf object loaded from the original config file.
            - logger: The reconstructed TensorBoardLogger instance.
            - resume_checkpoint: The path to the latest checkpoint file to resume from,
                                 or None if not found.
    """
    version_path = Path(version_path)

    # Extract components from the version_path
    # Example: /logs/experiment_name/version_0
    version = int(version_path.name.replace("version_", ""))
    experiment_name = version_path.parent.name
    logger_dir = version_path.parent.parent

    # Reconstruct the TensorBoardLogger
    logger = TensorBoardLogger(logger_dir, name=experiment_name, version=version)

    # Find the latest checkpoint within this version's checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    resume_checkpoint = get_latest_checkpoint(checkpoint_dir)

    # Try to find the original config file used for this version.
    config_files = list(Path(logger.log_dir).glob("*.yaml"))
    config_file = next((f for f in config_files if f.name != "hparams.yaml" and f.stat().st_size > 0), None)

    print(f"Loaded config from: {config_file}")
    if config_file is not None:
        config = OmegaConf.load(config_file)
    else:
        # Fallback if no config file is found (though it should ideally exist)
        print("Warning: No config file found in the version directory. Using empty config.")
        config = OmegaConf.create()

    return config, logger, resume_checkpoint

def load_module(path: str):
    """
    Dynamically loads a Python module or an object (e.g., a class or function)
    from a given string path.

    This is a utility function used across the framework to load data modules,
    models, or other components specified by their string path in the config.

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

def setup_data(config: OmegaConf):
    """
    Initializes and sets up a PyTorch Lightning DataModule.

    The DataModule class path is expected under `config.module_path` (config.data -> config).
    All other parameters in `config` are passed directly to the DataModule's constructor.

    Parameters:
        config : OmegaConf
            An OmegaConf object containing data module configuration. Must include
            `module_path` and any other arguments required by the DataModule.

    Returns:
        An instantiated and set up PyTorch Lightning DataModule.
    """
    print(f"Setting up data module from: {config.module_path}")
    data_module = load_module(config.pop('module_path'))

    # Instantiate the data module with remaining config parameters
    data_module = data_module(**config)
    data_module.setup()
    print("Data module setup complete.")
    return data_module

def setup_model(config: OmegaConf):
    """
    Initializes a PyTorch Lightning model.

    If a checkpoint path is provided, only the model weights are loaded
    (without restoring trainer/optimizer state).
    Optionally compiles the model if 'compile' is set in the config.
    """
    print(f"Setting up model from: {config.model.module_path}")
    Model = load_module(config.model.pop("module_path"))

    # Pop flags from config
    compile_model = config.model.pop("compile", False)
    checkpoint_path = config.model.pop("checkpoint_path", None)
    initial_weights = config.model.pop("initial_weights", None)

    if checkpoint_path:
        print(f"Loading model weights from checkpoint: {checkpoint_path}")
        model = Model.load_from_checkpoint(checkpoint_path, config=config)
    else:
        model = Model(config)

    if initial_weights is not None:
            print(f"Loading weights from {initial_weights}...")

            weights = torch.load(initial_weights, map_location=model.device)

            filtered_weights = {
                k: v for k, v in weights.items()
                if not (k.startswith("autoencoder.") or k.startswith("context_encoder.autoencoder."))
            }

            model.load_state_dict(
                filtered_weights,
                strict=False
            )

    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    print("Model setup complete.")
    return model

def setup_trainer(
    config_trainer: Optional[OmegaConf] = None, 
    config_callbacks: Optional[OmegaConf] = None,
    config_logger: Optional[OmegaConf] = None,
    logger: Optional[TensorBoardLogger] = None
) -> Trainer:
    """
    Configures and returns a PyTorch Lightning Trainer instance.

    This function sets up the Trainer with parameters for training, logging,
    and various callbacks like ModelCheckpoint and EarlyStopping based on
    the provided configuration.

    Parameters:
        config_trainer : Optional[OmegaConf]
            OmegaConf object for Trainer arguments (e.g., 'max_epochs', 'devices').
        config_callbacks : Optional[OmegaConf]
            OmegaConf object for callback configurations. Expected keys:
            'checkpoint' (for ModelCheckpoint), 'early_stopping' (for EarlyStopping).
        config_logger : Optional[OmegaConf]
            OmegaConf object for TensorBoardLogger arguments (e.g., 'save_dir', 'name').
            Used if `logger` is not explicitly provided.
        logger : Optional[TensorBoardLogger], default None
            An already instantiated TensorBoardLogger. If provided, `config_logger`
            is ignored for logger instantiation.

    Returns:
        lightning.pytorch.Trainer
            A configured PyTorch Lightning Trainer instance.
    """
    print("Setting up PyTorch Lightning Trainer...")
    if logger is None:
        logger = TensorBoardLogger(**config_logger) if config_logger else None
    else:
        print(f"TensorBoard Logger Version: {logger.version}")
    
    callbacks = []
    if config_callbacks is not None:
        # Add additional callbacks here
        if 'checkpoint' in config_callbacks:
            checkpoint_callback = ModelCheckpoint(**config_callbacks['checkpoint'])
            callbacks.append(checkpoint_callback)

        if 'early_stopping' in config_callbacks:
            early_stopping_callback = EarlyStopping(**config_callbacks['early_stopping'])
            callbacks.append(early_stopping_callback)

    trainer = Trainer(
        **config_trainer,
        logger=logger,
        callbacks=callbacks,
    )

    return trainer

def main(config: Optional[str] = None, version_path: Optional[str] = None, **kwargs):
    """
    Main function to run the model training process.

    This function handles loading configurations, setting up data, model, and trainer,
    and then initiating the training (fit) process. It supports starting a new
    training run or resuming from a specific experiment version. Carbon emissions
    are tracked using CodeCarbon.

    Parameters:
        config : Optional[str], default None
            Path to a YAML/JSON configuration file for a new training run.
            If `version_path` is provided, this parameter is ignored.
        version_path : Optional[str], default None
            Path to a previously logged experiment's `version_X` directory.
            If provided, the script attempts to load the previous configuration,
            logger, and the latest checkpoint from this version to resume training.
        **kwargs : Any
            Arbitrary keyword arguments from the command line. These arguments
            will update or override settings loaded from the `config` file.
            Example: `python train.py --config my_config.yaml trainer.max_epochs=100`

    Usage Examples:
    1. Start a new training run:
       `python train.py --config=configs/my_model_config.yaml`

    2. Resume training from a specific experiment version:
       `python train.py --version_path=/path/to/lightning_logs/my_exp_name/version_X`
    """
    print("Starting main training process.")

    resume_checkpoint = None
    logger = None
    if version_path:
        # True resume
        config, logger, resume_checkpoint = load_version(version_path)
        config.model.pop("checkpoint_path", None)
        model = setup_model(config)  # fresh init, trainer restores
    else:
        config_path = config
        config = OmegaConf.load(config) if (config is not None) else {}
        config.update(kwargs)
        
        model = setup_model(config)

    data_module = setup_data(config.dataloader)

    trainer = setup_trainer(
                            config.trainer, 
                            config.callbacks, 
                            config.logger,
                            logger=logger
                            )

    # Copy the original config file to the TensorBoard log directory for record-keeping
    if not version_path:
        copy_config_to_log_dir(config_path, trainer)

    # Initialize CodeCarbon tracker to monitor carbon emissions
    os.makedirs("results/environment_metrics", exist_ok=True)
    tracker = EmissionsTracker(
            project_name='Earthformer',
            output_dir="results/environment_metrics",
            log_level="warning",
            allow_multiple_runs=True,
        )
    tracker.start()
    print("Initiating model training...")
    trainer.fit(model, data_module, ckpt_path=resume_checkpoint)
    print("Model training finished.")

    emissions = tracker.stop()
    print(f"Emissions for this run: {emissions} kg CO2")
    
if __name__ == "__main__":
    Fire(main)