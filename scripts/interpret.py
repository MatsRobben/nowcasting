import importlib
import os
import json
import csv
import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

from fire import Fire
from omegaconf import OmegaConf

from nowcasting.validation.interpretation import (
    PermutationImportance,
    csi_metric,
)

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
    config: OmegaConf,
) -> torch.nn.Module:
    """
    Dynamically loads and initializes a nowcasting model based on the provided
    module path and configuration, with an optional checkpoint.
    """
    module_path = config.model.pop('module_path')

    print(f"Setting up model from: {module_path}")
    model_module = load_module(module_path)

    if 'checkpoint_path' in config.model:
        checkpoint_path = config.model.pop('checkpoint_path')
    
    compile_model = config.model.pop("compile", False)
    
    # We'll use the check `hasattr(model_module, "load_from_checkpoint")`
    # to determine if it's a LightningModule that can be loaded from a checkpoint.
    if checkpoint_path and hasattr(model_module, "load_from_checkpoint"):
        model_instance = model_module.load_from_checkpoint(checkpoint_path, config=config)
        # Assuming the actual model is an attribute of the LightningModule
        if hasattr(model_instance, "torch_nn_module"):
            model = model_instance.torch_nn_module
        else:
            model = model_instance
    else:
        # For models that are not LightningModules
        model = model_module(config)

    if compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    print("Model setup complete.")
    return model

def create_permutation_importance_plots(results_data, config: OmegaConf):
    """
    Creates academic-quality plots for permutation importance results.
    """
    group_names = results_data["group_names"]
    metrics = results_data["metrics"]
    out_dir = results_data["out_dir"]
    run_id = results_data["run_id"]
    
    pi_config = config.get('interpret', {})
    baseline_metric = metrics[0]
    
    # Set style for academic plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Calculate percentages relative to baseline
    percentages = [(metric / baseline_metric) * 100 for metric in metrics]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Horizontal bar chart with percentages
    labels = ["Baseline"] + [f"Permute {g}" if isinstance(g, (int, str)) else f"Permute Group {i+1}" 
              for i, g in enumerate(group_names)]
    
    # Color mapping - baseline in one color, permuted in gradient
    colors = ['#2E8B57']  # Sea green for baseline
    cmap = plt.cm.Reds
    perm_colors = [cmap(0.3 + 0.7 * i / max(1, len(group_names) - 1)) for i in range(len(group_names))]
    colors.extend(perm_colors)
    
    bars = ax1.barh(range(len(percentages)), percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold', fontsize=10)
    
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel('Performance (% of Baseline)', fontsize=12, fontweight='bold')
    ax1.set_title('Permutation Importance: Relative Performance', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(percentages) * 1.15)
    
    # Add vertical line at 100%
    ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    ax1.text(100, len(labels)-0.5, '100%', rotation=90, ha='right', va='top', fontweight='bold')
    
    # Plot 2: Line plot with absolute scores and confidence-style markers
    x_pos = range(len(metrics))
    ax2.plot(x_pos, metrics, 'o-', markersize=8, linewidth=3, color='#1f77b4', alpha=0.8)
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(x_pos, metrics)):
        ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold', fontsize=9)
    
    # Fill area under curve for visual appeal
    ax2.fill_between(x_pos, metrics, alpha=0.2, color='#1f77b4')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('CSI Score', fontsize=12, fontweight='bold')
    ax2.set_title('Permutation Importance: Absolute Scores', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(min(metrics) * 0.95, max(metrics) * 1.05)
    
    # Add horizontal line at baseline for reference
    ax2.axhline(y=baseline_metric, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(len(x_pos)-1, baseline_metric, f'Baseline: {baseline_metric:.3f}', 
             ha='right', va='bottom', fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = os.path.join(out_dir, f"permutation_importance_academic_{run_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create additional standalone horizontal bar chart
    fig_bar, ax_bar = plt.subplots(figsize=(10, 8))
    
    # Calculate importance drop (100% - percentage)
    importance_drops = [100 - pct for pct in percentages[1:]]  # Skip baseline
    perm_labels = [f"Group {i+1}" if isinstance(g, list) else str(g) 
                   for i, g in enumerate(group_names)]
    
    # Sort by importance (highest drop first)
    sorted_data = sorted(zip(importance_drops, perm_labels), reverse=True)
    sorted_drops, sorted_labels = zip(*sorted_data) if sorted_data else ([], [])
    
    # Color gradient based on importance
    colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_drops)))
    
    bars = ax_bar.barh(range(len(sorted_drops)), sorted_drops, 
                       color=colors_gradient, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, drop) in enumerate(zip(bars, sorted_drops)):
        width = bar.get_width()
        ax_bar.text(width + max(sorted_drops) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{drop:.1f}%', ha='left', va='center', fontweight='bold', fontsize=11)
    
    ax_bar.set_yticks(range(len(sorted_labels)))
    ax_bar.set_yticklabels(sorted_labels, fontsize=12)
    ax_bar.set_xlabel('Performance Drop (%)', fontsize=12, fontweight='bold')
    ax_bar.set_title('Feature Importance Ranking\n(Performance Drop When Permuted)', 
                     fontsize=14, fontweight='bold', pad=20)
    ax_bar.grid(axis='x', alpha=0.3, linestyle='--')
    ax_bar.set_xlim(0, max(sorted_drops) * 1.15 if sorted_drops else 1)
    
    plt.tight_layout()
    
    # Save the importance ranking plot
    ranking_path = os.path.join(out_dir, f"permutation_importance_ranking_{run_id}.png")
    plt.savefig(ranking_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Academic plots saved to: {plot_path}")
    print(f"Importance ranking saved to: {ranking_path}")

def setup_permutation_importance(model: torch.nn.Module, config: OmegaConf) -> PermutationImportance:
    """
    Initializes the PermutationImportance object based on the config.
    """
    print("Setting up Permutation Importance...")
    pi_config = config.get('interpret', {})
    print(pi_config)
    
    # You may need to define the metric function here
    # or pass it from a predefined function.
    metric_func = csi_metric(**pi_config.metric)
    
    pi = PermutationImportance(
        model=model,
        metric=metric_func,
        device=pi_config.get('device', 'cuda'),
        use_amp=pi_config.get('use_amp', True),
        verbose=pi_config.get('verbose', True),
        model_type=pi_config.get('model_type')
    )
    print("Permutation Importance setup complete.")
    return pi

def run_permutation_importance(
    pi_runner,
    dataloader,
    config: OmegaConf
):
    """
    Runs the permutation importance analysis and saves results and metadata.
    Returns the results for plotting.
    """
    print("Running Permutation Importance analysis...")
    pi_config = config.get('interpret', {})
    groups = pi_config.get('groups', [[0]])
    max_batches = pi_config.get('max_batches', None)
    model_name = pi_config.get('model_name', "ef_rad")
    save_dir = pi_config.get('save_dir', "./results")

    # Ensure parent directory exists
    out_dir = os.path.join(save_dir, model_name, "permutation_importance")
    os.makedirs(out_dir, exist_ok=True)

    # --- Run analysis ---
    group_names, metrics = pi_runner.multi_pass(
        dataloader=dataloader,
        groups=groups,
        max_batches=max_batches
    )

    # --- Get output dirs ---
    def get_next_run_id(out_dir):
        existing = [f for f in os.listdir(out_dir) if f.startswith("results")]
        return len(existing) + 1

    run_id = get_next_run_id(out_dir)
    results_path = os.path.join(out_dir, f"results_run{run_id}.csv")
    metadata_path = os.path.join(out_dir, f"metadata_run{run_id}.json")

    # --- Save results (CSV) ---
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Group", "Metric"])
        writer.writerow([0, "no-permutation", metrics[0]])
        for i, (g, m) in enumerate(zip(group_names, metrics[1:]), start=1):
            writer.writerow([i, g, m])

    # --- Save metadata ---
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "groups": group_names,
        "metrics": metrics,
        "config": OmegaConf.to_container(config, resolve=True)
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Results saved to: {results_path}")
    print(f"Metadata saved to: {metadata_path}")
    print("Permutation Importance analysis finished.")
    
    return {
        "group_names": group_names,
        "metrics": metrics,
        "out_dir": out_dir,
        "run_id": run_id
    }

def main(config: Optional[str] = None, **kwargs):
    """
    Main entry point for the evaluation and interpretation script.
    
    This function loads a YAML configuration file and, based on the `run_mode`
    setting, either performs a standard model evaluation or a permutation
    importance analysis.
    """
    print("Starting main interpretation process.")
    
    config = OmegaConf.load(config) if (config is not None) else {}
    config.update(kwargs)

    dataloader = setup_dataloader(**config.dataloader)
    
    model = setup_model(
        config=config,
    )
    
    pi_runner = setup_permutation_importance(model, config)
    
    # Run the analysis and save results
    results_data = run_permutation_importance(pi_runner, dataloader, config)
    
    # Create the plots
    create_permutation_importance_plots(results_data, config)

    print("Main process finished.")

if __name__ == "__main__":
    Fire(main)