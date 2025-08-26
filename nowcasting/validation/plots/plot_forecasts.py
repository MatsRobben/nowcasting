import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib import pyplot, colors
import numpy as np
import pandas as pd
import torch
import zarr
from omegaconf import OmegaConf
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm
import warnings

@dataclass
class SampleMetadata:
    """Metadata for a saved sample"""
    sample_id: int
    intensity_score: float
    batch_idx: int
    original_batch_position: int
    timestamp: Optional[str] = None

import json
import numpy as np
import torch
import zarr
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class ForecastCollector:
    """
    Collects forecast samples during validation and saves high-intensity cases
    for later visualization. Optimized to write all samples at once during finalize.
    """
    
    def __init__(
        self,
        save_dir: Union[str, Path],
        model_name: str,
        data_info: Dict,
        intensity_threshold: float = 0.8,
        max_samples: int = 100,
        save_ensemble: bool = False,
        eval_subdir: str = "evaluations",
        enforce_registry_match: bool = True,
    ):
        """
        Initialize the forecast collector.
        
        Args:
            save_dir: Directory to save results
            model_name: Name identifier for this model
            data_info: Channel information with mean/std statistics
            intensity_threshold: Minimum intensity to save sample (quantile)
            max_samples: Maximum number of samples to save
            save_ensemble: If True, save full ensemble; if False, save ensemble mean
            enforce_registry_match: If True, only save samples whose indices
                                    match those in the global registry.
        """
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.data_info = data_info
        self.intensity_threshold = intensity_threshold
        self.max_samples = max_samples
        self.save_ensemble = save_ensemble
        self.eval_subdir = eval_subdir
        self.enforce_registry_match = enforce_registry_match # STORE THE NEW PARAMETER
        
        # Create directories
        self.model_dir = self.save_dir / self.eval_subdir / model_name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage for samples
        self.samples = {
            'targets': [],
            'predictions': [],
            'intensities': [],
            'indices': [],
        }
        self.batch_counter = 0
        
        # For consistency across models
        self.global_sample_registry = self.save_dir / self.eval_subdir / "sample_registry.json"
        self.selected_samples = self._load_or_create_sample_registry()

        # If enforcing registry match, convert the list of indices to a set for faster lookup
        if self.enforce_registry_match and self.selected_samples["selected_samples"]:
            self.registry_indices_set = set(tuple(i) for i in self.selected_samples["selected_samples"])
        else:
            self.registry_indices_set = None

    def _load_or_create_sample_registry(self) -> Dict:
        """Load existing sample registry or create a new one."""
        if self.global_sample_registry.exists():
            with open(self.global_sample_registry, 'r') as f:
                try:
                    registry = json.load(f)
                    # Check for "selected_samples" and "metadata" keys
                    if "selected_samples" in registry and "metadata" in registry:
                        return registry
                    else:
                        print(f"Warning: Existing registry file {self.global_sample_registry} has invalid format. Creating new.")
                except json.JSONDecodeError:
                    print(f"Warning: Existing registry file {self.global_sample_registry} is corrupted. Creating new.")
        return {"selected_samples": [], "metadata": {}}
    
    def _save_sample_registry(self):
        """Save the sample registry to disk."""
        with open(self.global_sample_registry, 'w') as f:
            json.dump(self.selected_samples, f, indent=2)

    def _to_numpy(self, array):
        """Convert torch.Tensor or np.ndarray to np.ndarray (on CPU)."""
        if isinstance(array, torch.Tensor):
            return array.detach().cpu().numpy()
        elif isinstance(array, np.ndarray):
            return array
        else:
            raise TypeError(f"Unsupported type {type(array)}")
    
    def _calculate_intensity(self, y_target):
        """
        Calculate intensity score for each sample in the batch.
        Works for torch.Tensor or np.ndarray.
        """
        if isinstance(y_target, torch.Tensor):
            # [B, ...] -> flatten across all but batch
            return y_target.flatten(start_dim=1).mean(dim=1)
        elif isinstance(y_target, np.ndarray):
            return y_target.reshape(y_target.shape[0], -1).mean(axis=1)
        else:
            raise TypeError(f"Unsupported type {type(y_target)}")
    
    def _should_save_sample(self, sample_id: str, intensity_score: float, sample_index: Optional[Tuple[int, int, int]] = None) -> bool:
        """Determine if a sample should be saved"""
        if self.enforce_registry_match:
            # Check if index is in the registry set
            if sample_index is not None and self.registry_indices_set is not None:
                return sample_index in self.registry_indices_set
            # If no registry is found, enforce_registry_match is meaningless, fall back to intensity
            return intensity_score >= self.intensity_threshold
        else:
            # Use registry if available (original logic)
            if "selected_samples" in self.selected_samples and self.selected_samples["selected_samples"]:
                return sample_id in self.selected_samples["selected_samples"]
            
            # Otherwise use intensity threshold
            return intensity_score >= self.intensity_threshold
    
    def process_batch(
        self,
        y_target: torch.Tensor,
        y_pred: torch.Tensor,
        sample_idx: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        """
        Process a batch of predictions, selecting high-intensity samples.
        
        Args:
            y_target: Target tensor [B, C, T, H, W]
            y_pred: Prediction tensor [B, C, T, H, W] or [B, C, T, H, W, E]
            sample_idx: Tuple of (t_idx, h_idx, w_idx), each shape [B]
        """
        # Skip only if max_samples is set and already full
        if self.max_samples is not None and len(self.samples['targets']) >= self.max_samples:
            return
            
        batch_size = y_target.shape[0]

        intensities = self._calculate_intensity(y_target)

        # Handle ensemble dimension
        if not self.save_ensemble and (
            (isinstance(y_pred, torch.Tensor) and y_pred.dim() == 6) or 
            (isinstance(y_pred, np.ndarray) and y_pred.ndim == 6)
        ):
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.mean(dim=-1)
            else:
                y_pred = y_pred.mean(axis=-1)
        
        # Move to CPU/numpy
        y_target = self._to_numpy(y_target)
        y_pred = self._to_numpy(y_pred)
        intensities = self._to_numpy(intensities)
        
        # If sample indices are provided, also move to numpy
        if sample_idx is not None:
            sample_idx_np = tuple(self._to_numpy(idx) for idx in sample_idx)
        else:
            sample_idx_np = None

        # Loop over samples
        for i in range(batch_size):
            if self.max_samples is not None and len(self.samples['targets']) >= self.max_samples:
                break
            
            # Get the index of the current sample
            current_idx = (int(sample_idx_np[0][i]), int(sample_idx_np[1][i]), int(sample_idx_np[2][i])) if sample_idx_np is not None else None
            
            # The original logic for `sample_id` is based on batch counter; this is for internal tracking.
            # The registry-based check will use the `current_idx`.
            sample_id = f"{self.batch_counter:06d}_{i:03d}"
            intensity_score = float(intensities[i])
            
            # Use the new check, passing the actual sample index
            if self._should_save_sample(sample_id, intensity_score, sample_index=current_idx):
                self.samples['targets'].append(y_target[i])
                self.samples['predictions'].append(y_pred[i])
                self.samples['intensities'].append(intensity_score)
                self.samples['indices'].append(current_idx) # Append the actual index
        
        self.batch_counter += 1

    def finalize(self):
        """Write all collected samples to disk at once."""
        if not self.samples['targets']:
            print("No samples collected.")
            return

        # NEW LOGIC: If a registry doesn't exist, create it from the first model's indices
        if not self.global_sample_registry.exists() or not self.selected_samples["selected_samples"]:
            self.selected_samples["selected_samples"] = [list(idx) for idx in self.samples['indices'] if idx is not None]
            self.selected_samples["metadata"]["model"] = self.model_name
            self._save_sample_registry()
            print(f"Created global sample registry with {len(self.samples['indices'])} indices.")
        
        zarr_path = self.model_dir / f"{self.model_name}.zarr"
        store = zarr.open_group(str(zarr_path), mode='w')
        n_samples = len(self.samples['targets'])
        
        target_shape = self.samples['targets'][0].shape
        prediction_shape = self.samples['predictions'][0].shape
        
        store.create_array('targets', shape=(n_samples, *target_shape), dtype=np.float16)
        store.create_array('predictions', shape=(n_samples, *prediction_shape), dtype=np.float16)
        store.create_array('intensities', shape=(n_samples,), dtype=np.float16)
        store.create_array('indices', shape=(n_samples, 3), dtype=np.int32)
        
        for i in range(n_samples):
            store['targets'][i] = np.asarray(self.samples['targets'][i], dtype=np.float16)
            store['predictions'][i] = np.asarray(self.samples['predictions'][i], dtype=np.float16)
            store['intensities'][i] = np.asarray(self.samples['intensities'][i], dtype=np.float16)
            store['indices'][i] = np.asarray(self.samples['indices'][i], dtype=np.int32)
        
        metadata_path = self.model_dir / "metadata.json"
        metadata_dict = {
            "model_name": self.model_name,
            "total_samples": n_samples,
            "data_info": self.data_info,
            "save_ensemble": self.save_ensemble,
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        print(f"Saved {n_samples} samples for model {self.model_name}")


# --- Pysteps Colormap Functions (to be added to ForecastVisualizer) ---
def _get_colorlist(units="mm/h", colorscale="pysteps"):
    if colorscale == "pysteps":
        redgrey_hex = "#%02x%02x%02x" % (156, 126, 148)
        color_list = [
            redgrey_hex,
            "#640064",
            "#AF00AF",
            "#DC00DC",
            "#3232C8",
            "#0064FF",
            "#009696",
            "#00C832",
            "#64FF00",
            "#96FF00",
            "#C8FF00",
            "#FFFF00",
            "#FFC800",
            "#FFA000",
            "#FF7D00",
            "#E11900",
        ]
        if units in ["mm/h", "mm"]:
            clevs = [
                0.08, 0.16, 0.25, 0.40, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100, 160
            ]
        elif units == "dBZ":
            clevs = np.arange(10, 65, 5)
        else:
            raise ValueError(f"Wrong units in get_colorlist: {units}")
    else:
        raise ValueError(f"Invalid colorscale {colorscale}")

    clevs_str = [f"{clev:.2f}" if 0.1 <= clev < 1 else str(int(clev)) if clev >= 1 and clev.is_integer() else f"{clev:.1f}" for clev in clevs]

    return color_list, clevs, clevs_str

def get_colormap(ptype, units="mm/h", colorscale="pysteps"):
    if ptype in ["intensity", "depth"]:
        color_list, clevs, clevs_str = _get_colorlist(units, colorscale)
        cmap = colors.LinearSegmentedColormap.from_list("cmap", color_list, len(clevs) - 1)
        cmap.set_over("darkred", 1)
        norm = colors.BoundaryNorm(clevs, cmap.N)
        cmap.set_bad("gray", alpha=0.5)
        cmap.set_under("none")
        return cmap, norm, clevs, clevs_str
    
    # Placeholder for other types if needed, but not used in this context
    cmap = pyplot.get_cmap("jet")
    norm = colors.Normalize()
    return cmap, norm, None, None

class ForecastVisualizer:
    """
    Visualize and compare forecasts from multiple models with geographic maps.
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing model results
        """
        self.results_dir = Path(results_dir)
        self.models_data = {}
        self.data_info = None
        self.start_time = pd.Timestamp("2020-01-01 00:00")
        self.interval = "5min"
        
        # Hardcoded geographic parameters from radar grid info
        self.geo_bounds = {
            'n_rows': 765,
            'n_cols': 700,
            'pixel_size_x': 1.0000013,  # km
            'pixel_size_y': -1.0000055,  # km (negative because y increases downward)
            'row_offset': 3649.9792,     # km from origin
            'col_offset': 0.0,           # km from origin
        }
        
        # Sterographic projection for Netherlands (from your projection_proj4_params)
        self.projection = ccrs.Stereographic(
            central_latitude=90,
            central_longitude=0,
            true_scale_latitude=60
        )

    def load_model_results(self, model_names: List[str], display_names: Optional[Dict[str, str]] = None):
        """
        Load results for specified models and assert index consistency.

        Args:
            model_names: List of model directory names
            display_names: Optional dict mapping model_names to display names for plots
        """
        self.display_names = display_names or {}

        for i, model_name in enumerate(model_names):
            model_dir = self.results_dir / model_name
            metadata_path = model_dir / "metadata.json"
            if not metadata_path.exists():
                warnings.warn(f"No metadata found for model {model_name}. Skipping.")
                continue

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            zarr_path = model_dir / f"{model_name}.zarr"
            if zarr_path.exists():
                store = zarr.open_group(str(zarr_path), mode='r')
                targets = store['targets'][:]
                predictions = store['predictions'][:]
                intensities = store['intensities'][:]
                indices = store['indices'][:]

                if i == 0:
                    self.first_model_indices = indices
                    print(f"✅ Loaded indices from first model '{model_name}' as reference.")
                else:
                    assert np.array_equal(indices, self.first_model_indices), \
                        f"❌ Index mismatch! Model '{model_name}' indices do not match."
                    print(f"✅ Successfully asserted index consistency for model '{model_name}'.")

                data = {
                    'targets': targets, 'predictions': predictions,
                    'intensities': intensities, 'indices': indices
                }
            else:
                warnings.warn(f"No data file found for model {model_name}. Skipping.")
                continue

            self.models_data[model_name] = {'data': data, 'metadata': metadata}
            if self.data_info is None:
                self.data_info = metadata['data_info']

    def _get_crop_extent(self, h_idx: int, w_idx: int, crop_size: int = 256):
        """
        Convert crop indices to geographic coordinates.
        
        Args:
            h_idx: Height index (scaled down by 32)
            w_idx: Width index (scaled down by 32)
            crop_size: Size of the crop in pixels (default 256)
            
        Returns:
            Tuple of (left, right, bottom, top) in projected coordinates (km)
        """
        # Convert indices to actual pixel positions (multiply by 32)
        h_pixel = h_idx * 32
        w_pixel = w_idx * 32
        
        # Calculate coordinates (in km)
        left = w_pixel * self.geo_bounds['pixel_size_x'] + self.geo_bounds['col_offset']
        right = (w_pixel + crop_size) * self.geo_bounds['pixel_size_x'] + self.geo_bounds['col_offset']
        
        # Note: y axis is inverted (pixel_size_y is negative)
        top = h_pixel * self.geo_bounds['pixel_size_y'] + self.geo_bounds['row_offset']
        bottom = (h_pixel + crop_size) * self.geo_bounds['pixel_size_y'] + self.geo_bounds['row_offset']
        
        return (left, right, bottom, top)

    def plot_model_comparison(
        self,
        sample_idx: int = 0,
        channel_idx: int = 0,
        selected_times: Optional[Union[int, List[int]]] = None,
        cmap: str = "viridis",
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Union[str, Path]] = None,
        add_map: bool = True  # New parameter to toggle map
    ):
        """
        Create comparison plot showing context + forecasts for all models.
        Now with optional geographic map underneath.
        """
        if not self.models_data:
            raise ValueError("No model data loaded. Call load_model_results first.")

        model_names = list(self.models_data.keys())
        first_model = model_names[0]

        n_models = len(model_names)

        # Get indices and calculate geographic extent
        if 'indices' in self.models_data[first_model]['data'] and sample_idx < len(self.models_data[first_model]['data']['indices']):
            indices = self.models_data[first_model]['data']['indices'][sample_idx]
            t_idx, h_idx, w_idx = indices
            timestamp = self.start_time + pd.to_timedelta(int(t_idx) * pd.Timedelta(self.interval))
            print(f"Showing sample at global indices: (t={t_idx}, h={h_idx}, w={w_idx})")
            print(f'Time: {timestamp}')
            
            # Calculate geographic extent for this crop
            if add_map:
                extent = self._get_crop_extent(h_idx, w_idx)
                print(f"Geographic extent: {extent}")

        target_data = self.models_data[first_model]['data']['targets'][sample_idx]

        if target_data.ndim == 4:
            n_channels, n_timesteps, height, width = target_data.shape
            target_channel = target_data[channel_idx]
        elif target_data.ndim == 3:
            n_channels, height, width = target_data.shape
            n_timesteps = 1
            target_channel = target_data[channel_idx:channel_idx + 1]
        else:
            raise ValueError(f"Unexpected data shape: {target_data.shape}")

        if isinstance(selected_times, int):
            selected_times = [selected_times]
        
        if selected_times is None:
            selected_times = [(i + 1) * 5 for i in range(n_timesteps)]

        time_indices = []
        valid_selected_times = []
        for t in selected_times:
            idx = (t // 5) - 1
            if idx >= 0 and idx < n_timesteps:
                time_indices.append(idx)
                valid_selected_times.append(t)
        if not time_indices:
            raise ValueError("No selected_times map to available prediction timesteps.")

        is_single_timestep = len(time_indices) == 1
        
        # Use the PySTEPS get_colormap with specified ptype and units
        cmap_obj, norm, clevs, clevs_str = get_colormap(ptype="intensity", units="mm/h", colorscale=cmap)
        
        im = None
        cbar_bottom = 0
        cbar_height = 0

        if is_single_timestep:
            n_rows = 1
            n_cols = n_models + 1
            figsize = (max(2.5 * n_cols, 6), 2.5)
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=figsize,
                subplot_kw={'projection': self.projection} if add_map else {},
                sharex=True, sharey=True,
                gridspec_kw={'hspace': 0.06, 'wspace': 0.06}
            )
            axes = np.array(axes)

            gt_ax = axes[0]
            if add_map:
                gt_ax.set_extent(extent, crs=self.projection)
                gt_ax.coastlines(resolution='10m', color='black', linewidth=0.5)
                gt_ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                # Add Netherlands province borders
                gt_ax.add_feature(cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces',
                    scale='10m',
                    facecolor='none',
                    edgecolor='gray',
                    linewidth=0.3
                ))
            
            im = gt_ax.imshow(target_channel[time_indices[0]], cmap=cmap_obj, norm=norm, 
                             transform=self.projection, extent=extent, alpha=0.8, origin='upper')
            gt_ax.set_title("Ground Truth", fontsize=11)
            gt_ax.set_xticks([])
            gt_ax.set_yticks([])
            
            for spine in gt_ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.0)
            
            for model_idx, model_name in enumerate(model_names):
                pred_data = self.models_data[model_name]['data']['predictions'][sample_idx]
                pred_channel = pred_data[channel_idx]
                
                ax_pred = axes[model_idx + 1]
                if add_map:
                    ax_pred.set_extent(extent, crs=self.projection)
                    ax_pred.coastlines(resolution='10m', color='black', linewidth=0.5)
                    ax_pred.add_feature(cfeature.BORDERS, linewidth=0.5)
                    ax_pred.add_feature(cfeature.NaturalEarthFeature(
                        category='cultural',
                        name='admin_1_states_provinces',
                        scale='10m',
                        facecolor='none',
                        edgecolor='gray',
                        linewidth=0.3
                    ))
                
                im = ax_pred.imshow(pred_channel[time_indices[0]], cmap=cmap_obj, norm=norm,
                                   transform=self.projection, extent=extent, alpha=0.8, origin='upper')
                
                display_name = self.display_names.get(model_name, model_name)
                ax_pred.set_title(display_name, fontsize=11)
                ax_pred.set_xticks([])
                ax_pred.set_yticks([])

                for spine in ax_pred.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.0)
            
            fig.text(0.11, 0.5, f"+{valid_selected_times[0]} min.",
                    ha='left', va='center', rotation=90, fontsize=11)
            
            # Calculate colorbar height for single timestep
            first_ax_pos = axes[0].get_position()
            cbar_bottom = first_ax_pos.y0
            cbar_height = first_ax_pos.height
                        
        else:
            # Similar modifications for multiple timesteps case
            n_rows = n_models + 1
            n_cols = len(time_indices)
            figsize = (max(2 * n_cols, 6), max(1.7 * n_rows, 4))
            
            # Create subplots with projection if map is enabled
            subplot_kw = {'projection': self.projection} if add_map else {}
            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=figsize,
                subplot_kw=subplot_kw,
                sharex=True, sharey=True,
                gridspec_kw={'hspace': 0.0, 'wspace': 0.06}
            )
            axes = np.array(axes).reshape((n_rows, n_cols))

            im = None
            for row_idx, ax_row in enumerate(axes):
                for col_idx, ax in enumerate(ax_row):
                    if add_map:
                        ax.set_extent(extent, crs=self.projection)
                        ax.coastlines(resolution='10m', color='black', linewidth=0.5)
                        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
                        ax.add_feature(cfeature.NaturalEarthFeature(
                            category='cultural',
                            name='admin_1_states_provinces',
                            scale='10m',
                            facecolor='none',
                            edgecolor='gray',
                            linewidth=0.3
                        ))
                    
                    if row_idx == 0:
                        if col_idx < target_channel.shape[0]:
                            im = ax.imshow(target_channel[time_indices[col_idx]], cmap=cmap_obj, norm=norm,
                                          transform=self.projection, extent=extent, alpha=0.8, origin='upper')
                        ax.set_title(f"+{valid_selected_times[col_idx]} min.", fontsize=10)
                        if col_idx == 0:
                            ax.set_ylabel("Ground Truth", fontsize=10, rotation=90, labelpad=16, va='center')
                    else:
                        model_name = model_names[row_idx - 1]
                        pred_data = self.models_data[model_name]['data']['predictions'][sample_idx]
                        pred_channel = pred_data[channel_idx]
                        if time_indices[col_idx] < pred_channel.shape[0]:
                            im = ax.imshow(pred_channel[time_indices[col_idx]], cmap=cmap_obj, norm=norm,
                                          transform=self.projection, extent=extent, alpha=0.8, origin='upper')
                        if col_idx == 0:
                            display_name = self.display_names.get(model_name, model_name)
                            ax.set_ylabel(display_name, fontsize=10, rotation=90, labelpad=16, va='center')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.0)
            
            fig.subplots_adjust(
                left=0.18, right=0.88, top=0.92, bottom=0.06
            )

            # Calculate colorbar height for multiple timesteps
            first_ax_pos = axes[0, 0].get_position()
            last_ax_pos = axes[-1, 0].get_position()
            cbar_bottom = last_ax_pos.y0
            cbar_height = first_ax_pos.y1 - last_ax_pos.y0

        # Add colorbar
        cbar_ax = fig.add_axes([0.905, cbar_bottom, 0.006, cbar_height])
        
        if im is not None:
            cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=clevs, extend="max")
            cbar.ax.set_yticklabels(clevs_str)
            cbar.set_label('Precipitation intensity (mm/h)')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

def flatten_omegaconf_list(l):
    """Recursively flatten a ListConfig or nested lists into a flat Python list."""
    flat_list = []
    for item in l:
        # OmegaConf Lists behave like sequences, so check recursively
        if OmegaConf.is_list(item) or isinstance(item, list):
            flat_list.extend(flatten_omegaconf_list(item))
        else:
            flat_list.append(item)
    return flat_list


def extract_data_info_from_config(config, use_input_vars=True):
    """
    Extract data_info from OmegaConf config structure.
    
    Args:
        config: OmegaConf config object with dataloader section
        use_input_vars: If True, use in_vars; if False, use out_vars
    
    Returns:
        dict: data_info dictionary with channel statistics
    """
    dataloader_config = config.dataloader
    var_list = dataloader_config.var_info.in_vars if use_input_vars else dataloader_config.var_info.out_vars
    transforms = dataloader_config.var_info.transforms

    # Flatten the var_list to handle nested ListConfigs
    flat_var_list = flatten_omegaconf_list(var_list)

    data_info = {}

    for var_name in flat_var_list:
        # Check if this variable has normalization transforms
        if var_name in transforms and 'normalize' in transforms[var_name]:
            norm_stats = transforms[var_name]['normalize']
            data_info[var_name] = {
                'mean': norm_stats['mean'],
                'std': norm_stats['std']
            }
        else:
            # Check for group-level transforms (e.g., 'radar')
            var_group = var_name.split('/')[0]
            if var_group in transforms:
                group_transforms = transforms[var_group]
                if 'default_rainrate' in group_transforms:
                    # For radar data
                    norm_stats = group_transforms['default_rainrate']
                    data_info[var_name] = {
                        'mean': norm_stats['mean'],
                        'std': norm_stats['std']
                    }
                elif 'normalize' in group_transforms:
                    # For other group-level normalizations
                    norm_stats = group_transforms['normalize']
                    data_info[var_name] = {
                        'mean': norm_stats['mean'],
                        'std': norm_stats['std']
                    }
                else:
                    print(f"Warning: No normalization stats found for {var_name}")
                    data_info[var_name] = {'mean': 0.0, 'std': 1.0}
            else:
                print(f"Warning: No normalization stats found for {var_name}")
                data_info[var_name] = {'mean': 0.0, 'std': 1.0}

    return data_info

# Usage example:
if __name__ == "__main__":
    # Extract data_info from config:
    """
    from omegaconf import OmegaConf
    
    # Load your config
    config = OmegaConf.load("path/to/your/config.yaml")
    
    # Extract data info for input variables (context + prediction input)
    data_info = extract_data_info_from_config(config, use_input_vars=True)
    
    # Or for output variables (what you're predicting)
    # data_info = extract_data_info_from_config(config, use_input_vars=False)
    """
    
    # During validation loop:
    """
    collector = ForecastCollector(
        save_dir="./forecast_results",
        model_name="model_v1",
        data_info=data_info,
        intensity_threshold=0.8,
        max_samples=100
    )
    
    for idx, (x, y) in tqdm(enumerate(dataloader), desc="Validating"):
        y, y_hat = model(x, y)
        collector.process_batch(y, y_hat)
        
        if idx >= num_batches:
            break
    
    collector.finalize()
    """
    
    # For visualization:
    # path = '/projects/prjs1634/nowcasting/results/scores/evaluations'
    path = '/projects/prjs1634/nowcasting/results/scores/convective_events'
    visualizer = ForecastVisualizer(path)

    model_names = [
        # "ef_rad",
        # "ef_rad_sat_in",
        # "ef_rad_sat_in_out",
        # "ef_rad_har",
        # "ef_rad_aws",
        "ldcast_nowcast",
        "ldcast_nowcast_sat",
        "ldcast_nowcast_har",
        "ldcast_nowcast_aws",
        # "ldcast",
        "sprog",
        # "pysteps"
        ]
    display_names = {
        # "ef_rad": "Earthformer",
        # "ef_rad_sat_in": "+ Satellite",
        # "ef_rad_sat_in_out": "+ Satellite (joint)",
        # "ef_rad_har": "+ Harmonie",
        # "ef_rad_aws": "+ Ground Obs.",
        "ldcast_nowcast": "LNowcaster",
        "ldcast_nowcast_sat": "+ Satellite",
        "ldcast_nowcast_har": "+ Harmonie",
        "ldcast_nowcast_aws": "+ Ground Obs.",
        # "ldcast": "LDCast",
        "sprog": "S-PROG",
        # "pysteps": "STEPS"
    }
    
    visualizer.load_model_results(model_names=model_names,
                                  display_names=display_names)
    
    save_path = f"/projects/prjs1634/nowcasting/results/figures/vis/model_comparison_full_ln.png"
    visualizer.plot_model_comparison(
        sample_idx=25,
        channel_idx=0,
        selected_times=[5, 10, 15, 30, 60, 90],
        # selected_times=[30],
        cmap="pysteps",
        save_path=save_path
    )

    # for i in range(42):
    #     save_path = f"/projects/prjs1634/nowcasting/results/figures/vis/convection/model_comparison_{i}.png"
    #     visualizer.plot_model_comparison(
    #         sample_idx=i,
    #         channel_idx=0,
    #         # selected_times=[5, 10, 15, 30, 60, 90],
    #         selected_times=[30],
    #         cmap="pysteps",
    #         save_path=save_path
    #     )
    
    # stats = visualizer.create_summary_statistics()
    # print(stats)