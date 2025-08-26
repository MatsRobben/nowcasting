import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import h5py
from typing import Dict, List, Optional, Union, Tuple, Any
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import warnings
warnings.filterwarnings('ignore')

# Set high DPI for crisp figures
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class ResearchPlotter:
    """Publication-quality plotter for precipitation nowcasting evaluation results."""
    
    def __init__(self, 
                 figsize_base: Tuple[float, float] = (12, 8),
                 style: str = 'nature',
                 color_palette: str = 'nature',
                 save_dir: str = './plots',
                 save_formats: List[str] = ['png', 'pdf', 'eps']):
        """
        Initialize the research plotter with publication-ready aesthetics.
        
        Args:
            figsize_base: Base figure size for single plots
            style: Plotting style ('nature', 'science', 'minimal')
            color_palette: Color palette ('nature', 'science', 'vibrant')
            save_dir: Directory to save plots
            save_formats: Formats to save plots in
        """
        self.figsize_base = figsize_base
        self.style = style
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.save_formats = save_formats
        
        # Initialize style and colors
        self._setup_style()
        self._setup_colors(color_palette)
        
        # Storage for loaded model results
        self.model_results: Dict[str, Dict] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Define metrics and their properties
        self.categorical_metrics = ['POD', 'FAR', 'CSI', 'BIAS']
        self.continuous_metrics = ['MSE', 'MAE', 'RMSE', 'Bias']
        self.spatial_metrics = ['FSS']
        
    def _setup_style(self):
        """Setup publication-quality matplotlib style."""
        if self.style == 'nature':
            plt.style.use('default')
            plt.rcParams.update({
                # Fonts
                'font.family': 'sans-serif',
                'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 11,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14,
                # Lines
                'lines.linewidth': 1.5,
                'lines.markersize': 5,
                # Axes
                'axes.linewidth': 0.8,
                'axes.edgecolor': 'black',
                'axes.labelcolor': 'black',
                # Ticks
                'xtick.direction': 'out',
                'ytick.direction': 'out',
                'xtick.color': 'black',
                'ytick.color': 'black',
                # Grid
                'grid.linewidth': 0.5,
                'grid.alpha': 0.4,
                # Figure
                'figure.facecolor': 'white',
                'figure.edgecolor': 'white',
                'savefig.facecolor': 'white',
                'savefig.edgecolor': 'white',
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'savefig.pad_inches': 0.05
            })
        elif self.style == 'science':
            plt.rcParams.update({
                'font.size': 8,
                'axes.titlesize': 10,
                'axes.labelsize': 9,
                'xtick.labelsize': 7,
                'ytick.labelsize': 7,
                'legend.fontsize': 7,
                'figure.titlesize': 12,
            })
        elif self.style == 'minimal':
            plt.rcParams.update({
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.grid': False,
            })
    
    def _setup_colors(self, palette: str):
        """Setup color palettes for different publication styles."""
        palettes = {
            'nature': ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', 
                      '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC'],
            'science': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
            'vibrant': ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00',
                       '#FFFF33', '#A65628', '#F781BF', '#999999']
        }
        
        self.colors = palettes.get(palette, palettes['nature'])
        self.primary_color = self.colors[0]
        self.secondary_color = self.colors[1]
        
        # Create custom colormaps
        self.diverging_cmap = LinearSegmentedColormap.from_list(
            'diverging', [self.colors[1], 'white', self.colors[0]], N=256
        )
        self.sequential_cmap = LinearSegmentedColormap.from_list(
            'sequential', ['white', self.colors[0]], N=256
        )
    
    def load_model_results(self, model_name: str, results_path: str, 
                          display_name: Optional[str] = None):
        """Load results for a model from JSON or HDF5 file."""
        results_path = Path(results_path)
        display_name = display_name or model_name.replace('_', ' ').title()
        
        if results_path.suffix == '.json':
            with open(results_path, 'r') as f:
                data = json.load(f)
                self.model_results[model_name] = data.get('scores', {})
                self.model_metadata[model_name] = data.get('metadata', {})
        
        elif results_path.suffix == '.h5':
            with h5py.File(results_path, 'r') as f:
                # Load metadata
                metadata = {}
                for key in f.attrs:
                    metadata[key] = f.attrs[key]
                self.model_metadata[model_name] = metadata
                
                # Load scores (simplified implementation)
                print(f"HDF5 loading not fully implemented. Use JSON files for complete functionality.")
        
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
        # Store display name
        self.model_metadata[model_name]['display_name'] = display_name
        print(f"Loaded results for {display_name}")

    def plot_roebber_diagram(self, leadtime: int = 30, threshold: float = 2.0,
                           figsize: Optional[Tuple[float, float]] = None,
                           save_name: str = 'roebber_diagram'):
        """
        Create a proper Roebber performance diagram (POD vs 1-FAR).
        
        The Roebber diagram plots Probability of Detection (POD) vs Success Ratio (1-FAR),
        with diagonal lines showing constant CSI values and bias lines.
        """
        if not self.model_results:
            raise ValueError("No model results loaded. Use load_model_results() first.")
        
        figsize = figsize or (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot CSI isolines (diagonal lines)
        csi_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for csi in csi_values:
            # CSI = POD / (POD + FAR + (1-POD))
            # Rearranging: FAR = POD/CSI - POD - (1-POD) = POD(1/CSI - 1) - (1-POD)
            pod_line = np.linspace(csi, 1, 100)
            sr_line = pod_line / (csi + pod_line * (1 - csi))  # Success Ratio = 1 - FAR
            ax.plot(sr_line, pod_line, 'k--', alpha=0.3, linewidth=0.8)
            # Add CSI labels
            if len(sr_line) > 50:
                ax.text(sr_line[50], pod_line[50], f'{csi:.1f}', fontsize=8, 
                       rotation=45, alpha=0.7, ha='center', va='bottom')
        
        # Plot bias lines
        bias_values = [0.5, 0.67, 1.0, 1.5, 2.0, 3.0]
        for bias in bias_values:
            # BIAS = (POD + FAR) / POD = (Hits + False Alarms) / (Hits + Misses)
            # FAR = BIAS * POD - POD = POD * (BIAS - 1)
            # Success Ratio = 1 - FAR = 1 - POD * (BIAS - 1) = 1 - POD*BIAS + POD
            if bias == 1.0:
                # Perfect bias line is vertical
                ax.axvline(x=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
                ax.text(1.02, 0.5, 'BIAS=1.0', rotation=90, fontsize=8, alpha=0.7, va='center')
            else:
                pod_line = np.linspace(0, 1, 100)
                if bias > 1:
                    # For bias > 1, SR decreases with POD
                    sr_line = (pod_line / bias)
                    valid_idx = sr_line <= 1
                    ax.plot(sr_line[valid_idx], pod_line[valid_idx], 'gray', linestyle=':', alpha=0.5, linewidth=1)
                    if np.any(valid_idx):
                        mid_idx = np.sum(valid_idx) // 2
                        if mid_idx > 0:
                            ax.text(sr_line[mid_idx], pod_line[mid_idx], f'BIAS={bias}', 
                                   fontsize=8, alpha=0.7, rotation=-45, ha='center', va='bottom')
                else:
                    # For bias < 1
                    sr_line = np.minimum(pod_line / bias, 1.0)
                    ax.plot(sr_line, pod_line, 'gray', linestyle=':', alpha=0.5, linewidth=1)
                    mid_idx = len(sr_line) // 2
                    ax.text(sr_line[mid_idx], pod_line[mid_idx], f'BIAS={bias}', 
                           fontsize=8, alpha=0.7, rotation=45, ha='center', va='bottom')
        
        # Plot model data points
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            color = self.colors[idx % len(self.colors)]
            
            # Extract POD and FAR for the specified leadtime and threshold
            for cat_result in results.get('categorical', []):
                if (cat_result['leadtime'] == leadtime and 
                    abs(cat_result['threshold'] - threshold) < 0.001):
                    pod = cat_result['POD']
                    far = cat_result['FAR']
                    success_ratio = 1 - far  # Success Ratio = 1 - FAR
                    
                    ax.scatter(success_ratio, pod, s=100, color=color, 
                             label=display_name, edgecolor='white', linewidth=1, 
                             alpha=0.8, zorder=5)
                    break
        
        # Styling
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Success Ratio (1 - FAR)')
        ax.set_ylabel('Probability of Detection (POD)')
        ax.set_title(f'Roebber Performance Diagram\n{leadtime}-min Lead Time, {threshold} mm/h Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, framealpha=0.8)
        
        # Add text annotations for CSI and BIAS
        ax.text(0.02, 0.98, 'CSI isolines (dashed)', transform=ax.transAxes, 
               fontsize=8, alpha=0.7, va='top')
        ax.text(0.02, 0.94, 'BIAS isolines (dotted)', transform=ax.transAxes, 
               fontsize=8, alpha=0.7, va='top')
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig

    def plot_fss_matrix(self, leadtimes: List[int] = [15, 30, 60],
                       thresholds: List[float] = [0.5, 2.0, 10.0],
                       scales: List[int] = [1, 2, 4, 8, 16, 32],
                       figsize: Optional[Tuple[float, float]] = None,
                       save_name: str = 'fss_matrix'):
        """
        Create a matrix of FSS plots showing performance across different scales,
        with subplots for different leadtime/threshold combinations.
        """
        if not self.model_results:
            raise ValueError("No model results loaded. Use load_model_results() first.")
        
        # Check if any model has FSS data
        has_fss = any('fss' in results for results in self.model_results.values())
        if not has_fss:
            print("No FSS data found in loaded models.")
            return None
        
        n_leadtimes = len(leadtimes)
        n_thresholds = len(thresholds)
        figsize = figsize or (5 * n_thresholds, 4 * n_leadtimes)
        
        fig, axes = plt.subplots(n_leadtimes, n_thresholds, figsize=figsize, 
                                sharex=True, sharey=True)
        
        # Handle single subplot case
        if n_leadtimes == 1 and n_thresholds == 1:
            axes = [[axes]]
        elif n_leadtimes == 1:
            axes = [axes]
        elif n_thresholds == 1:
            axes = [[ax] for ax in axes]
        
        for i, leadtime in enumerate(leadtimes):
            for j, threshold in enumerate(thresholds):
                ax = axes[i][j]
                
                # Plot FSS vs scale for each model
                for idx, (model_name, results) in enumerate(self.model_results.items()):
                    if 'fss' not in results:
                        continue
                        
                    display_name = self.model_metadata[model_name].get('display_name', model_name)
                    color = self.colors[idx % len(self.colors)]
                    
                    # Extract FSS data for this leadtime and threshold
                    scale_values = []
                    fss_values = []
                    
                    for fss_result in results['fss']:
                        if (fss_result['leadtime'] == leadtime and 
                            abs(fss_result['threshold'] - threshold) < 0.001):
                            scale_values.append(fss_result['scale'])
                            fss_values.append(fss_result['FSS'])
                    
                    if scale_values:
                        # Sort by scale
                        sorted_data = sorted(zip(scale_values, fss_values))
                        scale_values, fss_values = zip(*sorted_data)
                        
                        ax.plot(scale_values, fss_values, 'o-', color=color, 
                               linewidth=1.5, markersize=5, label=display_name,
                               alpha=0.9, markeredgecolor='white', markeredgewidth=0.5)
                
                # Add FSS = 0.5 reference line (useful forecast threshold)
                ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(0.02, 0.52, 'FSS=0.5', transform=ax.transAxes, fontsize=8, 
                       color='red', alpha=0.7)
                
                # Styling
                ax.set_xlim(min(scales) * 0.8, max(scales) * 1.2)
                ax.set_ylim(0, 1)
                ax.set_xscale('log')
                ax.set_xticks(scales)
                ax.set_xticklabels([str(s) for s in scales])
                ax.grid(True, alpha=0.3, which='both')
                ax.set_title(f'{leadtime} min, {threshold} mm/h', fontsize=10)
                
                # Add labels only to edge subplots
                if i == n_leadtimes - 1:  # Bottom row
                    ax.set_xlabel('Spatial Scale (pixels)')
                if j == 0:  # Left column
                    ax.set_ylabel('Fractions Skill Score (FSS)')
                
                # Add legend only to top-right subplot
                if i == 0 and j == n_thresholds - 1:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                             frameon=True, fancybox=True, shadow=True, framealpha=0.8)
        
        plt.suptitle('Fractions Skill Score vs Spatial Scale', fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0, 0.95, 0.96])
        self._save_figure(fig, save_name)
        return fig

    def plot_skill_curves(self, metrics: List[str] = None,
                        thresholds: List[float] = [0.5, 2.0, 10.0],
                        figsize: Optional[Tuple[float, float]] = None,
                        save_name: str = 'skill_curves'):
        """
        Plot skill scores as function of lead time for multiple models and thresholds.
        """
        if metrics is None:
            metrics = self.categorical_metrics
            
        if not self.model_results:
            raise ValueError("No model results loaded. Use load_model_results() first.")
        
        figsize = figsize or (15, 10)
        n_metrics = len(metrics)
        n_thresholds = len(thresholds)
        
        fig, axes = plt.subplots(n_metrics, n_thresholds, figsize=figsize,
                                sharex=True)
        if n_metrics == 1 and n_thresholds == 1:
            axes = [[axes]]
        elif n_metrics == 1:
            axes = [axes]
        elif n_thresholds == 1:
            axes = [[ax] for ax in axes]
        
        for i, metric in enumerate(metrics):
            for j, threshold in enumerate(thresholds):
                ax = axes[i][j]
                
                for idx, (model_name, results) in enumerate(self.model_results.items()):
                    display_name = self.model_metadata[model_name].get('display_name', model_name)
                    color = self.colors[idx % len(self.colors)]
                    
                    # Extract data for this metric and threshold
                    leadtimes, values = [], []
                    for cat_result in results.get('categorical', []):
                        if abs(cat_result['threshold'] - threshold) < 0.001:
                            leadtimes.append(cat_result['leadtime'])
                            values.append(cat_result[metric])
                    
                    if leadtimes:
                        # Sort by leadtime
                        sorted_data = sorted(zip(leadtimes, values))
                        leadtimes, values = zip(*sorted_data)
                        
                        ax.plot(leadtimes, values, 'o-', color=color, linewidth=1.5,
                               markersize=4, label=display_name, alpha=0.9,
                               markeredgecolor='white', markeredgewidth=0.5)
                
                # Add reference lines for some metrics
                if metric == 'BIAS':
                    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
                    ax.text(0.02, 1.05, 'Perfect BIAS=1.0', transform=ax.transData, 
                           fontsize=8, color='red', alpha=0.7)
                
                # Styling
                ax.set_xlabel('Lead Time (min)')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} - {threshold} mm/h', fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(left=0)
                
                # Set y-axis limits based on metric
                if metric == 'FAR':
                    ax.set_ylim(0, 1)
                elif metric in ['POD', 'CSI']:
                    ax.set_ylim(0, 1)
                elif metric == 'BIAS':
                    ax.set_ylim(0, 3)
                
                # Add legend to first subplot only
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, frameon=True, fancybox=True, 
                             shadow=True, framealpha=0.8)
        
        plt.suptitle('Categorical Skill Scores vs Lead Time', fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_figure(fig, save_name)
        return fig

    def plot_continuous_metrics(self, metrics: List[str] = None,
                              figsize: Optional[Tuple[float, float]] = None,
                              save_name: str = 'continuous_metrics'):
        """Plot continuous metrics vs lead time."""
        if metrics is None:
            metrics = self.continuous_metrics
            
        if not self.model_results:
            raise ValueError("No model results loaded. Use load_model_results() first.")
        
        figsize = figsize or (12, 8)
        n_metrics = len(metrics)
        
        # Create subplot layout
        cols = 2
        rows = (n_metrics + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            for model_idx, (model_name, results) in enumerate(self.model_results.items()):
                display_name = self.model_metadata[model_name].get('display_name', model_name)
                color = self.colors[model_idx % len(self.colors)]
                
                # Extract continuous metrics data
                leadtimes, values = [], []
                for cont_result in results.get('continuous', []):
                    leadtimes.append(cont_result['leadtime'])
                    values.append(cont_result[metric])
                
                if leadtimes:
                    sorted_data = sorted(zip(leadtimes, values))
                    leadtimes, values = zip(*sorted_data)
                    
                    ax.plot(leadtimes, values, 'o-', color=color, linewidth=1.5,
                           markersize=4, label=display_name, alpha=0.9,
                           markeredgecolor='white', markeredgewidth=0.5)
            
            # Add reference line for bias
            if metric == 'Bias':
                ax.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, linewidth=1)
                ax.text(0.02, 0.05, 'No Bias = 0', transform=ax.transAxes, 
                       fontsize=8, color='red', alpha=0.7)
            
            # Styling
            ax.set_xlabel('Lead Time (min)')
            ylabel = f'{metric}'
            if metric != 'Bias':
                ylabel += ' (mm/h)' if metric != 'MSE' else ' (mm²/h²)'
            ax.set_ylabel(ylabel)
            ax.set_title(f'{metric} vs Lead Time', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(left=0)
            
            if metric != 'Bias':
                ax.set_ylim(bottom=0)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        # Add unified legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=min(4, len(labels)), fontsize=8, frameon=True, 
                  fancybox=True, shadow=True, framealpha=0.8)
        
        plt.suptitle('Continuous Evaluation Metrics', fontsize=12, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        self._save_figure(fig, save_name)
        return fig

    def create_paper_figures(self, 
                           main_leadtime: int = 30,
                           main_threshold: float = 2.0,
                           leadtimes: List[int] = [15, 30, 60],
                           thresholds: List[float] = [0.5, 2.0, 10.0],
                           scales: List[int] = [1, 2, 4, 8, 16, 32],
                           save_prefix: str = 'paper'):
        """
        Create a focused set of paper figures without the excessive combinations.
        
        Args:
            main_leadtime: Primary leadtime for single-condition plots
            main_threshold: Primary threshold for single-condition plots
            leadtimes: List of lead times for multi-condition plots
            thresholds: List of thresholds for multi-condition plots
            scales: List of scales for FSS plots
            save_prefix: Prefix for saved figure files
        """
        print("Creating paper figures...")
        
        try:
            # 1. Single Roebber diagram for main conditions
            print(f"Creating Roebber diagram for {main_leadtime}min, {main_threshold}mmh...")
            self.plot_roebber_diagram(
                leadtime=main_leadtime, 
                threshold=main_threshold,
                save_name=f"{save_prefix}_roebber_{main_leadtime}min_{main_threshold}mmh"
            )
            
            # 2. FSS matrix for multiple conditions
            print("Creating FSS matrix...")
            self.plot_fss_matrix(
                leadtimes=leadtimes,
                thresholds=thresholds,
                scales=scales,
                save_name=f"{save_prefix}_fss_matrix"
            )
            
            # 3. Categorical skill curves
            print("Creating categorical skill curves...")
            self.plot_skill_curves(
                metrics=self.categorical_metrics,
                thresholds=thresholds,
                save_name=f"{save_prefix}_categorical_skills"
            )
            
            # 4. Continuous metrics
            print("Creating continuous metrics plot...")
            self.plot_continuous_metrics(
                metrics=self.continuous_metrics,
                save_name=f"{save_prefix}_continuous_metrics"
            )
            
            print("Paper figures created successfully!")
            
        except Exception as e:
            print(f"Error creating paper figures: {e}")
            raise

    def _save_figure(self, fig, name: str):
        """Save figure in specified formats."""
        for fmt in self.save_formats:
            filepath = self.save_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {filepath}")


def load_and_compare_models(model_configs: List[Dict[str, str]], 
                          plotter_config: Optional[Dict] = None) -> ResearchPlotter:
    """
    Convenient function to load multiple models and create plotter.
    
    Args:
        model_configs: List of dicts with 'name', 'path', and optional 'display_name'
        plotter_config: Configuration for the plotter
        
    Example:
        model_configs = [
            {'name': 'unet', 'path': 'results/unet/final_results.json', 'display_name': 'U-Net'},
            {'name': 'convlstm', 'path': 'results/convlstm/final_results.json', 'display_name': 'ConvLSTM'},
            {'name': 'transformer', 'path': 'results/transformer/final_results.json', 'display_name': 'Weather Transformer'}
        ]
    """
    plotter_config = plotter_config or {}
    plotter = ResearchPlotter(**plotter_config)
    
    for config in model_configs:
        plotter.load_model_results(
            model_name=config['name'],
            results_path=config['path'],
            display_name=config.get('display_name')
        )
    
    return plotter


if __name__ == "__main__":
    # Load and compare models
    # path = '/projects/prjs1634/nowcasting/results/scores/convective_events'
    path = '/projects/prjs1634/nowcasting/results/scores/evaluations'

    model_configs = [
        # EarthFormer
        # {'name': 'ef_rad_mse_old', 'path': f'{path}/ef_rad_mse_old/final_results.json', 'display_name': 'Earthformer (MSE)'},
        # {'name': 'ef_rad_balext_old', 'path': f'{path}/ef_rad_balext_old/final_results.json', 'display_name': 'Earthformer (Bal. Ext.)'},
        {'name': 'ef_rad', 'path': f'{path}/ef_rad/final_results.json', 'display_name': 'Earthformer'},
        {'name': 'ef_rad_sat_in', 'path': f'{path}/ef_rad_sat_in/final_results.json', 'display_name': '+ Satellite'},
        # {'name': 'ef_rad_sat_in_out', 'path': f'{path}/ef_rad_sat_in_out/final_results.json', 'display_name': 'EF Sat-In/Out'},
        {'name': 'ef_rad_har', 'path': f'{path}/ef_rad_har/final_results.json', 'display_name': '+ Harmonie'},
        {'name': 'ef_rad_aws', 'path': f'{path}/ef_rad_aws/final_results.json', 'display_name': '+ Ground Obs.'},
        # LDCast Nowcast
        # {'name': 'ldcast_nowcast', 'path': f'{path}/ldcast_nowcast/final_results.json', 'display_name': 'LNowcast'},
        # {'name': 'ldcast_nowcast_sat', 'path': f'{path}/ldcast_nowcast_sat/final_results.json', 'display_name': 'LDCast Nowcast Sat'},
        # {'name': 'ldcast_nowcast_har', 'path': f'{path}/ldcast_nowcast_har/final_results.json', 'display_name': 'LDCast Nowcast Har'},
        # {'name': 'ldcast_nowcast_har_t', 'path': f'{path}/ldcast_nowcast_har_t/final_results.json', 'display_name': 'LDCast Nowcast Har + Time Embedding'},
        # {'name': 'ldcast_nowcast_aws', 'path': f'{path}/ldcast_nowcast_aws/final_results.json', 'display_name': 'LDCast Nowcast AWS'},
        # Baselines
        {'name': 'sprog', 'path': f'{path}/sprog/final_results.json', 'display_name': 'S-PROG'},
        # {'name': 'pysteps', 'path': f'{path}/pysteps/final_results.json', 'display_name': 'STEPS'},
        # {'name': 'ldcast', 'path': f'{path}/ldcast/final_results.json', 'display_name': 'LDCast'},
    ]

    plotter = load_and_compare_models(model_configs, {
        'color_palette': 'nature',
        'save_formats': ['png', 'pdf', 'svg'],
        'save_dir': '/projects/prjs1634/nowcasting/results/figures/paper_figures_test_ef'
    })

    # plotter.create_appendix_figures()

    plotter.create_paper_figures(
        main_leadtime=30,
        main_threshold=2.0,
        leadtimes=[15, 30, 60],
        thresholds=[0.5, 2.0, 10.0],
        scales=[1, 2, 4, 8, 16, 32],
        save_prefix='paper'
    )