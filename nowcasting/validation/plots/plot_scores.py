import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json
import h5py
from typing import Dict, List, Optional, Union, Tuple, Any
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set high DPI for crisp figures
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class ResearchPlotter:
    """Publication-quality plotter for precipitation nowcasting evaluation results."""
    
    def __init__(self, 
                 figsize_base: Tuple[float, float] = (12, 8),
                 style: str = 'modern',
                 color_palette: str = 'custom',
                 save_dir: str = './plots',
                 save_formats: List[str] = ['png', 'pdf']):
        """
        Initialize the research plotter with modern aesthetics.
        
        Args:
            figsize_base: Base figure size for single plots
            style: Plotting style ('modern', 'minimal', 'academic')
            color_palette: Color palette ('custom', 'nature', 'science', 'vibrant')
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
        
    def _setup_style(self):
        """Setup modern matplotlib style."""
        if self.style == 'modern':
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                # Fonts
                'font.family': 'sans-serif',
                'font.sans-serif': ['Inter', 'Helvetica', 'Arial', 'DejaVu Sans'],
                'font.size': 14,               # base font size
                'axes.titlesize': 24,          # subplot titles
                'axes.labelsize': 22,          # x/y labels
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'legend.fontsize': 18,
                'figure.titlesize': 28,        # suptitle
                # Axes & spines
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': True,
                'axes.spines.bottom': True,
                'axes.linewidth': 1.2,
                'axes.edgecolor': '#2E2E2E',
                'axes.facecolor': '#FAFAFA',
                'figure.facecolor': 'white',
                # Grid
                'grid.alpha': 0.3,
                'grid.linewidth': 0.8,
                'grid.color': '#CCCCCC',
                # Lines & markers
                'lines.linewidth': 2.2,
                'lines.markersize': 7,
                # Patches
                'patch.linewidth': 0.8,
                # Text color
                'text.color': '#2E2E2E'
            })
        elif self.style == 'minimal':
            plt.rcParams.update({
                'font.family': 'sans-serif',
                'font.sans-serif': ['Source Sans Pro', 'Arial', 'DejaVu Sans'],
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.left': False,
                'axes.spines.bottom': False,
                'axes.grid': False,
                'axes.facecolor': 'white',
                'figure.facecolor': 'white'
            })
        elif self.style == 'academic':
            plt.rcParams.update({
                'font.family': 'serif',
                'font.serif': ['Times New Roman', 'DejaVu Serif'],
                'font.size': 11,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'legend.fontsize': 10
            })
    
    def _setup_colors(self, palette: str):
        """Setup color palettes for different journals/styles."""
        palettes = {
            'custom': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#5D737E', '#64A6BD', '#90A959'],
            'nature': ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2'],
            'science': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
            'vibrant': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8'],
            'earth': ['#8D5524', '#C68B59', '#D6A2E8', '#87CEEB', '#98FB98', '#F4A460', '#DDA0DD']
        }
        
        self.colors = palettes.get(palette, palettes['custom'])
        self.primary_color = self.colors[0]
        self.secondary_color = self.colors[1]
        
        # Create custom colormap
        self.custom_cmap = LinearSegmentedColormap.from_list(
            'custom', [self.colors[0], self.colors[2], self.colors[1]], N=256
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
                # Load scores (simplified - would need full implementation for HDF5)
                metadata = {}
                for key in f.attrs:
                    metadata[key] = f.attrs[key]
                self.model_metadata[model_name] = metadata
                
                # For HDF5, you'd need to implement score loading
                print(f"HDF5 loading not fully implemented. Use JSON files for now.")
                return
        
        else:
            raise ValueError("Unsupported file format. Use .json or .h5")
        
        # Store display name
        self.model_metadata[model_name]['display_name'] = display_name
        print(f"Loaded results for {display_name}")
    
    def plot_skill_scores_by_leadtime(self, metrics: List[str] = ['CSI', 'POD', 'FAR'],
                                        thresholds: List[float] = [0.5, 2.0, 10.0],
                                        figsize: Optional[Tuple[float, float]] = None,
                                        save_name: str = 'skill_scores_leadtime'):
        """Plot skill scores as function of lead time for multiple models and thresholds."""
        if not self.model_results:
            raise ValueError("No model results loaded. Use load_model_results() first.")
        
        figsize = figsize or (15, 15)
        n_metrics = len(metrics)
        n_thresholds = len(thresholds)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_metrics, n_thresholds, figure=fig, hspace=0.3, wspace=0.25)

        # Keep handles/labels for legend
        handles, labels = [], []
        
        for i, metric in enumerate(metrics):
            for j, threshold in enumerate(thresholds):
                ax = fig.add_subplot(gs[i, j])
                
                for idx, (model_name, results) in enumerate(self.model_results.items()):
                    display_name = self.model_metadata[model_name].get('display_name', model_name)
                    color = self.colors[idx % len(self.colors)]
                    
                    # Extract data for this metric and threshold
                    leadtimes, values = [], []
                    for cat_result in results.get('categorical', []):
                        if abs(cat_result['threshold'] - threshold) < 0.001:  # Float comparison
                            leadtimes.append(cat_result['leadtime'])
                            values.append(cat_result[metric])
                    
                    if leadtimes:
                        # Sort by leadtime
                        sorted_data = sorted(zip(leadtimes, values))
                        leadtimes, values = zip(*sorted_data)
                        
                        line, = ax.plot(leadtimes, values, 'o-', color=color, linewidth=2.5,
                                    markersize=7, label=display_name, alpha=0.9,
                                    markeredgecolor='white', markeredgewidth=1)
                        
                        # Only store handles/labels once
                        if (i == 0) and (j == 0):
                            handles.append(line)
                            labels.append(display_name)
                
                # Titles: only threshold
                ax.set_title(f'{threshold} mm/h', fontsize=13, fontweight='bold', pad=12)
                
                # Show only outer axis labels
                if i == n_metrics - 1:  # bottom row
                    ax.set_xlabel('Lead Time (minutes)', fontweight='medium')
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                
                if j == 0:  # first column
                    ax.set_ylabel(metric, fontweight='medium')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
                
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlim(left=0)
                
                # Set y-axis limits based on metric
                if metric == 'FAR':
                    ax.set_ylim(0, 1)
                elif metric in ['POD', 'CSI']:
                    ax.set_ylim(0, 1)
                elif metric == 'BIAS':
                    ax.set_ylim(0, 3)
                
                # Add subtle background
                ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], alpha=0.05, color=self.primary_color)
        
        # Global legend at the bottom
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                ncol=min(4, len(labels)), fontsize=9, frameon=True, 
                fancybox=True, shadow=True, framealpha=0.8)
        
        plt.suptitle('Categorical Skill Scores vs Lead Time', fontsize=18, fontweight='bold', y=0.97)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # leave space for legend & title
        self._save_figure(fig, save_name)
        return fig

    
    def plot_continuous_metrics(self, figsize: Optional[Tuple[float, float]] = None,
                                save_name: str = 'continuous_metrics'):
        """Plot continuous metrics (MSE, MAE, RMSE, Bias) vs lead time (clean style)."""
        figsize = figsize or (14, 11)
        
        metrics = ['MSE', 'MAE', 'RMSE', 'Bias']
        nrows, ncols = 2, 2

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                sharex=True, sharey=False)
        axes = axes.reshape(nrows, ncols)

        handles, labels = [], []

        for idx, metric in enumerate(metrics):
            i, j = divmod(idx, ncols)
            ax = axes[i, j]

            for model_idx, (model_name, results) in enumerate(self.model_results.items()):
                display_name = self.model_metadata[model_name].get('display_name', model_name)
                color = self.colors[model_idx % len(self.colors)]

                leadtimes, values = [], []
                for cont_result in results.get('continuous', []):
                    leadtimes.append(cont_result['leadtime'])
                    values.append(cont_result[metric])

                if leadtimes:
                    sorted_data = sorted(zip(leadtimes, values))
                    leadtimes, values = zip(*sorted_data)

                    line, = ax.plot(leadtimes, values, "o-", color=color, linewidth=2.5,
                                    markersize=6, label=display_name, alpha=0.9,
                                    markeredgecolor="white", markeredgewidth=1)

                    if idx == 0:  # store handles/labels only once
                        handles.append(line)
                        labels.append(display_name)

            # Titles: always show metric name
            ax.set_title(metric)

            # Y-labels: show for all subplots
            if metric == "Bias":
                ylabel = "Bias"
            elif "MSE" in metric:
                ylabel = f"{metric} (mm²/h²)"
            else:
                ylabel = f"{metric} (mm/h)"
            ax.set_ylabel(ylabel)

            # X-labels: only bottom row
            if i == nrows - 1:
                ax.set_xlabel("Lead Time (minutes)")

            ax.grid(True, alpha=0.4, linestyle="--")
            ax.set_xlim(left=0)
            if metric != "Bias":
                ax.set_ylim(bottom=0)

        # Normal legend inside top-right subplot
        top_right_ax = axes[0, -1]
        top_right_ax.legend(handles, labels, loc="lower right", frameon=True, fontsize=16)

        fig.suptitle("Continuous Evaluation Metrics", fontsize=28, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        self._save_figure(fig, save_name)
        return fig




    def plot_fss_heatmap(self, leadtime: int = 30, 
                        figsize: Optional[Tuple[float, float]] = None,
                        save_name: str = 'fss_heatmap'):
        """Create FSS heatmap comparing models across thresholds and scales."""
        figsize = figsize or (12, 8)
        
        # Prepare data for heatmap
        model_names = list(self.model_results.keys())
        if not model_names:
            raise ValueError("No model results loaded.")
        
        # Get thresholds and scales from first model
        first_model = list(self.model_results.values())[0]
        fss_data = first_model.get('fss', [])
        
        if not fss_data:
            print("No FSS data found in results.")
            return None
        
        # Filter for specific leadtime
        leadtime_data = [item for item in fss_data if item['leadtime'] == leadtime]
        
        if not leadtime_data:
            print(f"No FSS data found for leadtime {leadtime} minutes.")
            return None
        
        # Extract unique thresholds and scales
        thresholds = sorted(list(set([item['threshold'] for item in leadtime_data])))
        scales = sorted(list(set([item['scale'] for item in leadtime_data])))
        
        n_models = len(model_names)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for model_idx, (model_name, results) in enumerate(self.model_results.items()):
            ax = axes[model_idx]
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            
            # Create FSS matrix
            fss_matrix = np.full((len(thresholds), len(scales)), np.nan)
            
            for fss_item in results.get('fss', []):
                if fss_item['leadtime'] == leadtime:
                    thr_idx = thresholds.index(fss_item['threshold'])
                    scale_idx = scales.index(fss_item['scale'])
                    fss_matrix[thr_idx, scale_idx] = fss_item['FSS']
            
            # Create heatmap
            im = ax.imshow(fss_matrix, cmap=self.custom_cmap, aspect='auto',
                          vmin=0, vmax=1, origin='lower')
            
            # Customize ticks and labels
            ax.set_xticks(range(len(scales)))
            ax.set_yticks(range(len(thresholds)))
            ax.set_xticklabels([f'{s}px' for s in scales])
            ax.set_yticklabels([f'{t}mm/h' for t in thresholds])
            
            ax.set_xlabel('Spatial Scale', fontweight='medium')
            if model_idx == 0:
                ax.set_ylabel('Precipitation Threshold', fontweight='medium')
            
            ax.set_title(f'{display_name}\n(Lead Time: {leadtime} min)', 
                        fontsize=14, fontweight='bold', pad=15)
            
            # Add text annotations
            for i in range(len(thresholds)):
                for j in range(len(scales)):
                    if not np.isnan(fss_matrix[i, j]):
                        text_color = 'white' if fss_matrix[i, j] < 0.5 else 'black'
                        ax.text(j, i, f'{fss_matrix[i, j]:.2f}',
                               ha='center', va='center', fontsize=10, 
                               fontweight='bold', color=text_color)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=30)
        cbar.set_label('Fractions Skill Score', fontweight='medium', labelpad=15)
        
        plt.suptitle(f'FSS Comparison at {leadtime}-minute Lead Time', 
                    fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_performance_summary(self, figsize: Optional[Tuple[float, float]] = None,
                               save_name: str = 'performance_summary'):
        """Create a comprehensive performance summary dashboard."""
        figsize = figsize or (16, 12)
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. CSI vs Lead Time (main plot)
        ax_main = fig.add_subplot(gs[0, :2])
        threshold = 2.0  # Focus on 2mm/h threshold
        
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            color = self.colors[idx % len(self.colors)]
            
            leadtimes, csi_values = [], []
            for cat_result in results.get('categorical', []):
                if abs(cat_result['threshold'] - threshold) < 0.001:
                    leadtimes.append(cat_result['leadtime'])
                    csi_values.append(cat_result['CSI'])
            
            if leadtimes:
                sorted_data = sorted(zip(leadtimes, csi_values))
                leadtimes, csi_values = zip(*sorted_data)
                
                ax_main.fill_between(leadtimes, csi_values, alpha=0.3, color=color)
                ax_main.plot(leadtimes, csi_values, 'o-', color=color, linewidth=3,
                           markersize=8, label=display_name, markeredgecolor='white', 
                           markeredgewidth=2)
        
        ax_main.set_xlabel('Lead Time (minutes)', fontweight='medium')
        ax_main.set_ylabel('Critical Success Index', fontweight='medium')
        ax_main.set_title(f'CSI Performance at {threshold} mm/h', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(frameon=True, fancybox=True, shadow=True)
        
        # 2. RMSE comparison (top right)
        ax_rmse = fig.add_subplot(gs[0, 2])
        model_names = []
        rmse_30min = []
        
        for model_name, results in self.model_results.items():
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            model_names.append(display_name)
            
            # Get RMSE at 30 minutes
            rmse_val = None
            for cont_result in results.get('continuous', []):
                if cont_result['leadtime'] == 30:
                    rmse_val = cont_result['RMSE']
                    break
            rmse_30min.append(rmse_val or 0)
        
        bars = ax_rmse.bar(range(len(model_names)), rmse_30min, 
                          color=self.colors[:len(model_names)], alpha=0.8)
        ax_rmse.set_xticks(range(len(model_names)))
        ax_rmse.set_xticklabels(model_names, rotation=45, ha='right')
        ax_rmse.set_ylabel('RMSE (mm/h)', fontweight='medium')
        ax_rmse.set_title('RMSE at 30 min', fontsize=12, fontweight='bold')
        ax_rmse.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, rmse_30min):
            ax_rmse.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Skill scores radar chart (bottom left)
        ax_radar = fig.add_subplot(gs[1, 0], projection='polar')
        
        # Prepare radar chart data
        radar_metrics = ['CSI', 'POD', 'BIAS']  # Invert FAR for better visualization
        n_metrics = len(radar_metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            color = self.colors[idx % len(self.colors)]
            
            # Get values for 30min leadtime and 2mm/h threshold
            values = []
            for metric in radar_metrics:
                for cat_result in results.get('categorical', []):
                    if (cat_result['leadtime'] == 30 and 
                        abs(cat_result['threshold'] - 2.0) < 0.001):
                        if metric == 'BIAS':
                            # Normalize BIAS (closer to 1 is better)
                            bias_val = cat_result['BIAS']
                            normalized = 1 - abs(1 - bias_val) / 2  # Scale so 1->1, 0,2->0.5
                            values.append(max(0, normalized))
                        else:
                            values.append(cat_result[metric])
                        break
            
            if len(values) == n_metrics:
                values += values[:1]  # Complete the circle
                ax_radar.plot(angles, values, 'o-', color=color, linewidth=2, 
                            label=display_name, markersize=6)
                ax_radar.fill(angles, values, alpha=0.1, color=color)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(radar_metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Skill Scores at 30min\n(2mm/h threshold)', 
                          fontsize=12, fontweight='bold', pad=20)
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        # 4. Lead time performance summary (bottom center & right)
        ax_summary = fig.add_subplot(gs[1:, 1:])
        
        # Create performance matrix: models vs leadtimes (using CSI at 2mm/h)
        leadtime_set = set()
        for results in self.model_results.values():
            for cat_result in results.get('categorical', []):
                if abs(cat_result['threshold'] - 2.0) < 0.001:
                    leadtime_set.add(cat_result['leadtime'])
        
        leadtimes_sorted = sorted(list(leadtime_set))
        model_names_clean = [self.model_metadata[name].get('display_name', name) 
                            for name in self.model_results.keys()]
        
        perf_matrix = np.full((len(model_names_clean), len(leadtimes_sorted)), np.nan)
        
        for model_idx, (model_name, results) in enumerate(self.model_results.items()):
            for cat_result in results.get('categorical', []):
                if abs(cat_result['threshold'] - 2.0) < 0.001:
                    try:
                        lt_idx = leadtimes_sorted.index(cat_result['leadtime'])
                        perf_matrix[model_idx, lt_idx] = cat_result['CSI']
                    except ValueError:
                        continue
        
        # Create heatmap
        im = ax_summary.imshow(perf_matrix, cmap=self.custom_cmap, aspect='auto', 
                              vmin=0, vmax=1, origin='lower')
        
        ax_summary.set_xticks(range(len(leadtimes_sorted)))
        ax_summary.set_yticks(range(len(model_names_clean)))
        ax_summary.set_xticklabels([f'{lt}min' for lt in leadtimes_sorted])
        ax_summary.set_yticklabels(model_names_clean)
        ax_summary.set_xlabel('Lead Time', fontweight='medium')
        ax_summary.set_ylabel('Model', fontweight='medium')
        ax_summary.set_title('CSI Performance Matrix\n(2mm/h threshold)', 
                            fontsize=12, fontweight='bold')
        
        # Add text annotations
        for i in range(len(model_names_clean)):
            for j in range(len(leadtimes_sorted)):
                if not np.isnan(perf_matrix[i, j]):
                    text_color = 'white' if perf_matrix[i, j] < 0.5 else 'black'
                    ax_summary.text(j, i, f'{perf_matrix[i, j]:.2f}',
                                   ha='center', va='center', fontsize=9, 
                                   fontweight='bold', color=text_color)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax_summary, shrink=0.8)
        cbar.set_label('CSI Score', fontweight='medium')
        
        plt.suptitle('Precipitation Nowcasting Performance Summary', 
                    fontsize=18, fontweight='bold', y=0.95)
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def plot_crps_comparison(self, figsize: Optional[Tuple[float, float]] = None,
                           save_name: str = 'crps_comparison'):
        """Plot CRPS scores for ensemble models."""
        figsize = figsize or (10, 6)
        
        # Check if any models have CRPS data
        has_crps = any('crps' in results for results in self.model_results.values())
        if not has_crps:
            print("No CRPS data found in any model results.")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for idx, (model_name, results) in enumerate(self.model_results.items()):
            if 'crps' not in results:
                continue
                
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            color = self.colors[idx % len(self.colors)]
            
            leadtimes, crps_values = [], []
            for crps_result in results['crps']:
                leadtimes.append(crps_result['leadtime'])
                crps_values.append(crps_result['CRPS'])
            
            if leadtimes:
                sorted_data = sorted(zip(leadtimes, crps_values))
                leadtimes, crps_values = zip(*sorted_data)
                
                ax.fill_between(leadtimes, crps_values, alpha=0.2, color=color)
                ax.plot(leadtimes, crps_values, 'o-', color=color, linewidth=3,
                       markersize=8, label=display_name, markeredgecolor='white', 
                       markeredgewidth=1.5)
        
        ax.set_xlabel('Lead Time (minutes)', fontweight='medium')
        ax.set_ylabel('CRPS (mm/h)', fontweight='medium')
        ax.set_title('Continuous Ranked Probability Score', fontsize=16, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.9)
        
        plt.tight_layout()
        self._save_figure(fig, save_name)
        return fig
    
    def _save_figure(self, fig, name: str):
        """Save figure in specified formats."""
        for fmt in self.save_formats:
            filepath = self.save_dir / f"{name}.{fmt}"
            fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Saved: {filepath}")
    
    def create_publication_figure(self, figure_type: str = 'comprehensive',
                                figsize: Optional[Tuple[float, float]] = None,
                                save_name: str = 'publication_figure'):
        """Create a publication-ready figure with multiple panels."""
        if figure_type == 'comprehensive':
            return self.plot_performance_summary(figsize, save_name)
        elif figure_type == 'skill_scores':
            return self.plot_skill_scores_by_leadtime(figsize=figsize, save_name=save_name)
        elif figure_type == 'continuous':
            return self.plot_continuous_metrics(figsize, save_name)
        elif figure_type == 'fss':
            return self.plot_fss_heatmap(figsize=figsize, save_name=save_name)
        else:
            raise ValueError("figure_type must be 'comprehensive', 'skill_scores', 'continuous', or 'fss'")
    
    def export_data_table(self, leadtime: int = 30, threshold: float = 2.0,
                         save_name: str = 'performance_table'):
        """Export performance data as a formatted table for papers."""
        if not self.model_results:
            print("No model results loaded.")
            return None
        
        # Prepare data for table
        table_data = []
        
        for model_name, results in self.model_results.items():
            display_name = self.model_metadata[model_name].get('display_name', model_name)
            row = {'Model': display_name}
            
            # Get categorical metrics
            for cat_result in results.get('categorical', []):
                if (cat_result['leadtime'] == leadtime and 
                    abs(cat_result['threshold'] - threshold) < 0.001):
                    row.update({
                        'POD': f"{cat_result['POD']:.3f}",
                        'FAR': f"{cat_result['FAR']:.3f}",
                        'CSI': f"{cat_result['CSI']:.3f}",
                        'BIAS': f"{cat_result['BIAS']:.3f}"
                    })
                    break
            
            # Get continuous metrics
            for cont_result in results.get('continuous', []):
                if cont_result['leadtime'] == leadtime:
                    row.update({
                        'RMSE': f"{cont_result['RMSE']:.3f}",
                        'MAE': f"{cont_result['MAE']:.3f}",
                        'Bias': f"{cont_result['Bias']:.3f}"
                    })
                    break
            
            # Get CRPS if available
            for crps_result in results.get('crps', []):
                if crps_result['leadtime'] == leadtime:
                    row['CRPS'] = f"{crps_result['CRPS']:.3f}"
                    break
            
            table_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save as CSV and LaTeX
        csv_path = self.save_dir / f"{save_name}.csv"
        latex_path = self.save_dir / f"{save_name}.tex"
        
        df.to_csv(csv_path, index=False)
        
        # Create LaTeX table
        latex_table = df.to_latex(index=False, float_format=lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else str(x),
                                 caption=f'Model Performance Comparison at {leadtime}-minute lead time and {threshold} mm/h threshold',
                                 label=f'tab:performance_{leadtime}min_{threshold}mmh',
                                 position='htbp')
        
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        print(f"Table saved as CSV: {csv_path}")
        print(f"Table saved as LaTeX: {latex_path}")
        
        return df
    
    def generate_report(self, report_name: str = 'model_comparison_report'):
        """Generate a complete evaluation report with all plots and tables."""
        print(f"Generating comprehensive report: {report_name}")
        
        # Create all plots
        figures = {}
        
        print("Creating performance summary...")
        figures['summary'] = self.plot_performance_summary(save_name=f"{report_name}_summary")
        
        print("Creating skill scores plot...")
        figures['skill_scores'] = self.plot_skill_scores_by_leadtime(save_name=f"{report_name}_skill_scores")
        
        print("Creating continuous metrics plot...")
        figures['continuous'] = self.plot_continuous_metrics(save_name=f"{report_name}_continuous")
        
        # FSS plot if data available
        try:
            print("Creating FSS heatmap...")
            figures['fss'] = self.plot_fss_heatmap(save_name=f"{report_name}_fss")
        except:
            print("FSS data not available, skipping FSS plot.")
        
        # CRPS plot if data available
        try:
            print("Creating CRPS comparison...")
            figures['crps'] = self.plot_crps_comparison(save_name=f"{report_name}_crps")
        except:
            print("CRPS data not available, skipping CRPS plot.")
        
        # Generate tables
        print("Creating performance tables...")
        for leadtime in [15, 30, 60]:
            for threshold in [0.5, 2.0, 10.0]:
                try:
                    self.export_data_table(leadtime=leadtime, threshold=threshold,
                                         save_name=f"{report_name}_table_{leadtime}min_{threshold}mmh")
                except:
                    continue
        
        print(f"Report generation complete! Files saved in: {self.save_dir}")
        return figures

    def export_radar_models_table(self, thresholds=[1, 5, 10, 20], save_name="radar_models_table"):
        """Export table for radar-only models with CSI (avg over lead times) and RMSE."""
        radar_models = ["sprog", "ef_rad", "ldcast_nowcast"]

        table_data = []
        for model_name, results in self.model_results.items():
            if model_name not in radar_models:
                continue

            display_name = self.model_metadata[model_name].get("display_name", model_name)
            row = {"Model": display_name}

            # --- Average CSI for each threshold ---
            for thr in thresholds:
                csi_vals = [
                    cat["CSI"] for cat in results.get("categorical", [])
                    if abs(cat["threshold"] - thr) < 0.001
                ]
                if csi_vals:
                    row[f"CSI@{thr}"] = f"{np.mean(csi_vals):.3f}"
                else:
                    row[f"CSI@{thr}"] = "-"

            # --- Average RMSE ---
            rmse_vals = [cont["RMSE"] for cont in results.get("continuous", [])]
            if rmse_vals:
                row["RMSE"] = f"{np.mean(rmse_vals):.3f}"
            else:
                row["RMSE"] = "-"

            table_data.append(row)

        df = pd.DataFrame(table_data)

        # Save
        csv_path = self.save_dir / f"{save_name}.csv"
        latex_path = self.save_dir / f"{save_name}.tex"
        df.to_csv(csv_path, index=False)

        latex_table = df.to_latex(
            index=False,
            caption=f"Radar-only model comparison (mean CSI over lead times, thresholds {thresholds} mm/h, and mean RMSE)",
            label="tab:radar_models",
            position="htbp"
        )
        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"Table saved as CSV: {csv_path}")
        print(f"Table saved as LaTeX: {latex_path}")

        return df
        
    def plot_skill_scores_by_leadtime(self, metrics: List[str] = ['CSI', 'POD', 'FAR', 'BIAS'],
                                            thresholds: List[float] = [1.0, 5.0, 10.0],
                                            figsize: Optional[Tuple[float, float]] = None,
                                            save_name: str = 'skill_scores_leadtime'):
            """Plot skill scores as function of lead time for multiple models and thresholds."""
            if not self.model_results:
                raise ValueError("No model results loaded. Use load_model_results() first.")
            
            n_metrics = len(metrics)
            n_thresholds = len(thresholds)
            figsize = figsize or (5 * n_thresholds, 3.5 * n_metrics)

            fig, axes = plt.subplots(n_metrics, n_thresholds, figsize=figsize,
                                    sharex="col", sharey="row")
            
            # Ensure axes is always 2D
            if n_metrics == 1:
                axes = np.array([axes])
            if n_thresholds == 1:
                axes = axes[:, np.newaxis]

            handles, labels = [], []

            for i, metric in enumerate(metrics):
                for j, threshold in enumerate(thresholds):
                    ax = axes[i, j]

                    for idx, (model_name, results) in enumerate(self.model_results.items()):
                        display_name = self.model_metadata[model_name].get("display_name", model_name)
                        color = self.colors[idx % len(self.colors)]

                        leadtimes, values = [], []
                        for cat in results.get("categorical", []):
                            if abs(cat["threshold"] - threshold) < 1e-3:
                                leadtimes.append(cat["leadtime"])
                                values.append(cat[metric])

                        if leadtimes:
                            lt_sorted, val_sorted = zip(*sorted(zip(leadtimes, values)))
                            line, = ax.plot(lt_sorted, val_sorted, "o-", label=display_name,
                                            color=color, alpha=0.9)

                            # Store handles/labels once for legend
                            if (i == 0) and (j == 0):
                                handles.append(line)
                                labels.append(display_name)

                    # Titles only for top row
                    if i == 0:
                        ax.set_title(f"Threshold = {threshold} mm/h")

                    # Y-label only for first column
                    if j == 0:
                        ax.set_ylabel(metric)

                    # X-label only for bottom row
                    if i == n_metrics - 1:
                        ax.set_xlabel("Lead Time (minutes)")

                    ax.grid(True, alpha=0.4, linestyle="--")
                    ax.set_xlim(left=0)

                    # Metric-specific y limits
                    if metric in ["CSI", "POD", "FAR"]:
                        ax.set_ylim(0, 1)
                    elif metric == "BIAS":
                        ax.set_ylim(0, 2)

            # Normal legend: inside top-right subplot
            top_right_ax = axes[0, -1]
            top_right_ax.legend(loc="upper right", frameon=True, fontsize=14)

            fig.suptitle("Categorical Skill Scores vs Lead Time", fontsize=24, fontweight="bold")
            fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
            self._save_figure(fig, save_name)
            return fig

    def plot_models_csi(self, thresholds=[1.0, 5.0, 10.0],
                                figsize=(15, 7), save_name="models_csi_ln"):
        """
        Plot CSI vs lead time for radar-only and satellite-enhanced models.
        One subplot per threshold, same style as radar_models_csi.
        """
        model_groups = [
            # ('ef_rad_mse_old', 'Earthformer (MSE)'),
            # ('ef_rad_balext_old', 'Earthformer (Bal. Ext.)'),
            # ('sprog', 'S-PROG'),
            # ("ef_rad", "Earthformer (radar)"),
            # ("ef_rad_sat_in", "+ Satellite"),
            # ("ef_rad_har", "+ Harmonie"),
            # ("ef_rad_aws", "+ Ground Obs."),
            ("ldcast_nowcast", "LNowcaster (radar)"),
            ("ldcast_nowcast_sat", "+ Satellite"),
            ("ldcast_nowcast_har", "+ Harmonie"),
            ("ldcast_nowcast_aws", "+ Ground Obs.")
        ]

        n_thr = len(thresholds)
        fig, axes = plt.subplots(1, n_thr, figsize=figsize, sharey=True)

        if n_thr == 1:  # single threshold case
            axes = [axes]

        for ax, thr in zip(axes, thresholds):
            for idx, (model_name, display_name) in enumerate(model_groups):
                if model_name not in self.model_results:
                    continue

                results = self.model_results[model_name]
                color = self.colors[idx % len(self.colors)]

                leadtimes, csi_vals = [], []
                for cat in results.get("categorical", []):
                    if abs(cat["threshold"] - thr) < 1e-3:
                        leadtimes.append(cat["leadtime"])
                        csi_vals.append(cat["CSI"])

                if leadtimes:
                    lt_sorted, csi_sorted = zip(*sorted(zip(leadtimes, csi_vals)))
                    ax.plot(lt_sorted, csi_sorted, "o-", label=display_name, color=color, alpha=0.9)

            ax.set_xlabel("Lead Time (minutes)", fontweight="medium")
            ax.set_title(f"Threshold = {thr} mm/h", fontweight="bold")
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_xlim(left=0, right=95)
            ax.set_ylim(0, 1)

        axes[0].set_ylabel("Critical Success Index", fontweight="medium")
        axes[-1].legend(frameon=True)

        fig.suptitle("Radar vs Satellite-Enhanced Models: CSI vs Lead Time", fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_figure(fig, save_name)
        return fig
    
    def export_satellite_table(self, thresholds=[1, 5, 10, 20], save_name="performance_loss"):
        """
        Export a LaTeX table comparing radar-only and satellite-enhanced models.
        CSI thresholds first, RMSE last. Best value per metric is bolded.
        """
        import numpy as np
        import pandas as pd

        # Define model groups
        model_groups = [
            # ('ef_rad_mse_old', 'Earthformer (MSE)'),
            # ('ef_rad_balext_old', 'Earthformer (Bal. Ext.)'),
            ('sprog', 'S-PROG'),

            # ("ef_rad", "Earthformer (radar)"),
            # ("ef_rad_sat_in", "+ Satellite (input)"),
            # ("ef_rad_sat_in_out", "+ Satellite (joint)"),
            # ("ef_rad_har", "+ Harmonie (Forecasts) \n time embeding"),
            # ("ef_rad_aws", "+ Ground Obs."),
            # ("ldcast_nowcast", "LNowcaster (radar)"),
            # ("ldcast_nowcast_sat", "+ Satellite (pre-train)"),
            ("ldcast_nowcast_har", "LNowcaster Radar + Harmonie"),
            ("ldcast_nowcast_har_t", "LNowcaster Radar + Harmonie with time embeding"),
            # ("ldcast_nowcast_aws", "+ Ground Obs.")
        ]

        table_data = []

        for model_name, display_name in model_groups:
            results = self.model_results.get(model_name, {})
            row = {"Model": display_name}

            # --- CSI for each threshold ---
            for thr in thresholds:
                csi_vals = [cat["CSI"] for cat in results.get("categorical", []) if abs(cat["threshold"] - thr) < 0.001]
                row[f"CSI@{thr}"] = np.mean(csi_vals) if csi_vals else np.nan

            # --- RMSE ---
            rmse_vals = [cont["RMSE"] for cont in results.get("continuous", [])]
            row["RMSE"] = np.mean(rmse_vals) if rmse_vals else np.nan

            table_data.append(row)

        # Convert to DataFrame
        df = pd.DataFrame(table_data)

        # Bold best values per column
        for col in df.columns[1:]:
            max_val = df[col].max() if "CSI" in col else None
            min_val = df[col].min() if col == "RMSE" else None
            if "CSI" in col:
                df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == max_val else f"{x:.3f}")
            elif col == "RMSE":
                df[col] = df[col].apply(lambda x: f"\\textbf{{{x:.3f}}}" if x == min_val else f"{x:.3f}")

        # --- LaTeX table ---
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\caption{Impact of incorporating SEVIRI satellite data on nowcasting performance.}\n"
        latex_table += "\\label{tab:satellite_performance}\n"
        latex_table += "\\begin{tabular}{l|" + "c" * (len(thresholds) + 1) + "}\n"
        latex_table += "\\hline\n"
        latex_table += "Model / Configuration & " + " & ".join([f"CSI@{thr} mm/h" for thr in thresholds]) + " & RMSE (mm/h) \\\\\n"
        latex_table += "\\hline\n"

        for _, row in df.iterrows():
            latex_table += row["Model"] + " & " + " & ".join([row[f"CSI@{thr}"] for thr in thresholds] + [row["RMSE"]]) + " \\\\\n"

        latex_table += "\\hline\n\\end{tabular}\n\\end{table}"

        # Save CSV
        csv_path = self.save_dir / f"{save_name}.csv"
        df.to_csv(csv_path, index=False)

        # Save LaTeX
        latex_path = self.save_dir / f"{save_name}.tex"
        with open(latex_path, "w") as f:
            f.write(latex_table)

        print(f"Table saved as CSV: {csv_path}")
        print(f"Table saved as LaTeX: {latex_path}")
        return df



# Example usage and helper functions
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


def create_paper_figures(plotter: ResearchPlotter, paper_style: str = 'nature'):
    """Create figures optimized for specific journal styles."""
    
    journal_specs = {
        'nature': {
            'figsize_single': (8.9, 6),      # Nature single column
            'figsize_double': (18.3, 12),    # Nature double column
            'color_palette': 'nature',
            'style': 'modern'
        },
        'science': {
            'figsize_single': (8.5, 6),
            'figsize_double': (17.8, 12),
            'color_palette': 'science',
            'style': 'modern'
        },
        'ieee': {
            'figsize_single': (8.5, 6),
            'figsize_double': (17, 11),
            'color_palette': 'custom',
            'style': 'academic'
        }
    }
    
    specs = journal_specs.get(paper_style, journal_specs['nature'])
    
    # Update plotter settings
    plotter._setup_colors(specs['color_palette'])
    
    # Create paper-specific figures
    figures = {
        'main_figure': plotter.plot_performance_summary(
            figsize=specs['figsize_double'], 
            save_name=f'{paper_style}_main_figure'
        ),
        'supplementary_skills': plotter.plot_skill_scores_by_leadtime(
            figsize=specs['figsize_double'],
            save_name=f'{paper_style}_supplementary_skills'
        ),
        'supplementary_continuous': plotter.plot_continuous_metrics(
            figsize=specs['figsize_single'],
            save_name=f'{paper_style}_supplementary_continuous'
        )
    }
    
    return figures

if __name__ == "__main__":
    # Load and compare models
    # path = '/projects/prjs1634/nowcasting/results/scores/convective_events'
    path = '/projects/prjs1634/nowcasting/results/scores/evaluations'

    model_configs = [
        # EarthFormer
        # {'name': 'ef_rad_mse_old', 'path': f'{path}/ef_rad_mse_old/final_results.json', 'display_name': 'Earthformer (MSE)'},
        # {'name': 'ef_rad_balext_old', 'path': f'{path}/ef_rad_balext_old/final_results.json', 'display_name': 'Earthformer (Bal. Ext.)'},
        # {'name': 'ef_rad', 'path': f'{path}/ef_rad/final_results.json', 'display_name': 'Earthformer'},
        # {'name': 'ef_rad_sat_in', 'path': f'{path}/ef_rad_sat_in/final_results.json', 'display_name': '+ Satellite'},
        # {'name': 'ef_rad_sat_in_out', 'path': f'{path}/ef_rad_sat_in_out/final_results.json', 'display_name': 'EF Sat-In/Out'},
        # {'name': 'ef_rad_har', 'path': f'{path}/ef_rad_har/final_results.json', 'display_name': '+ Harmonie'},
        # {'name': 'ef_rad_aws', 'path': f'{path}/ef_rad_aws/final_results.json', 'display_name': '+ Ground Obs.'},
        # LDCast Nowcast
        {'name': 'ldcast_nowcast', 'path': f'{path}/ldcast_nowcast/final_results.json', 'display_name': 'LNowcast'},
        {'name': 'ldcast_nowcast_sat', 'path': f'{path}/ldcast_nowcast_sat/final_results.json', 'display_name': '+ Satellite'},
        {'name': 'ldcast_nowcast_har', 'path': f'{path}/ldcast_nowcast_har/final_results.json', 'display_name': '+ Harmonie'},
        # {'name': 'ldcast_nowcast_har_t', 'path': f'{path}/ldcast_nowcast_har_t/final_results.json', 'display_name': 'LDCast Nowcast Har + Time Embedding'},
        {'name': 'ldcast_nowcast_aws', 'path': f'{path}/ldcast_nowcast_aws/final_results.json', 'display_name': '+ Ground Obs.'},
        # Baselines
        {'name': 'sprog', 'path': f'{path}/sprog/final_results.json', 'display_name': 'S-PROG'},
        # {'name': 'pysteps', 'path': f'{path}/pysteps/final_results.json', 'display_name': 'STEPS'},
        # {'name': 'ldcast', 'path': f'{path}/ldcast/final_results.json', 'display_name': 'LDCast'},
    ]


    plotter = load_and_compare_models(model_configs, {
        'color_palette': 'nature',
        'save_formats': ['png', 'pdf', 'svg'],
        'save_dir': '/projects/prjs1634/nowcasting/results/figures/paper_figures'
    })

    # df_radar = plotter.export_radar_models_table()
    # plotter.plot_radar_models_csi()
    plotter.plot_models_csi()
    # print(df_radar)

    # df_sat = plotter.export_satellite_table()
    # print(df_sat)

    # plotter.plot_skill_scores_by_leadtime(save_name=f"skill_scores_ef")
    # plotter.plot_continuous_metrics(save_name="continuous_metrics_ln")


    # Create publication-ready figures
    # plotter.generate_report('nature_paper_2024')

    # # Or create specific journal figures
    # create_paper_figures(plotter, 'nature')