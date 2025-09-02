import os
import glob
import re
import tarfile
import io
from datetime import datetime
# import multiprocessing as mp

import numpy as np
from numba import jit, prange, njit
from typing import Dict, Tuple
import pandas as pd
import h5py
import zarr

# --- Configuration ---
BASE_DATA_DIR = "/nobackup_1/users/robben/full_dataset"
ZARR_OUTPUT_PATH = "/nobackup_1/users/robben/datasets/dataset.zarr"

START_TIME = pd.Timestamp("2020-01-01 00:00")
END_TIME = pd.Timestamp("2023-12-31 23:59")

CLUTTER_VAR_NAME = "clutter_score"
CLUTTER_DTYPE = np.uint16

source_configs = [
    {
        "name": "rtcor",
        "path": os.path.join(BASE_DATA_DIR, "radar/RTCOR"),
        "interval": 5,
        "fill_value": 0,
        "chunk_size": 4,
        "shard_size": 1024,
        "to_rainrate": True,       # Convert to mm/h
        "antialias": True,          # Apply antialiasing
        "compute_clutter": True,    # Compute clutter score
        "create_timemask": True,
        "create_max_intensity": True,  # Create max_intensity_grid
        "pad_value": 0,
        "vars": {
            'rtcor': {'dtype': np.float16, 'dims': (768, 704)},
            'max_intensity_grid': {'dtype': np.uint8, 'dims': (24, 22)},
        }
    },
    {
        "name": "rfcor",
        "path": os.path.join(BASE_DATA_DIR, "radar/RFCOR"),
        "interval": 5,
        "fill_value": 0,
        "chunk_size": 4,
        "shard_size": 1024,
        "to_rainrate": True,        # Convert to mm/h
        "antialias": False,
        "compute_clutter": False,
        "create_timemask": True,
        "create_max_intensity": False,
        "pad_value": 0,
        "vars": {
            'rfcor': {'dtype': np.float16, 'dims': (768, 704)},
        }
    },
    {
        "name": "eth",
        "path": os.path.join(BASE_DATA_DIR, "radar/ETH"),
        "interval": 5,
        "fill_value": 0,
        "chunk_size": 4,
        "shard_size": 1024,
        "to_rainrate": False,        # ETH is not rain rate
        "antialias": False,
        "compute_clutter": False,
        "create_timemask": True,
        "create_max_intensity": False,
        "pad_value": 0,
        "vars": {
            'eth': {'dtype': np.float16, 'dims': (768, 704)},
        }
    }
]


def create_time_index_map(start_time: pd.Timestamp, end_time: pd.Timestamp, interval_minutes: int) -> dict:
    # Generate the timestamps at the specified interval
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{interval_minutes}min')
    
    # Create a dictionary mapping timestamp to its index
    time_index_map = {timestamp: idx for idx, timestamp in enumerate(timestamps)}

    return time_index_map

def get_timestamp_from_filename(filename):
    """
    Extract the most relevant timestamp from a filename using multiple regex patterns.
    Tries in order: YYYYMMDDHHMMSS, YYYYMMDDHHMM, YYYYMMDD.
    Returns: pd.Timestamp or None
    """
    basename = os.path.basename(filename)

    patterns = [
        (r"_(\d{8}T\d{6})", "%Y%m%dT%H%M%S"), # Match e.g. _20200101T001500
        (r"_(\d{14})",       "%Y%m%d%H%M%S"), # Match e.g. _20200101001500
        (r"_(\d{12})",       "%Y%m%d%H%M"),   # Match e.g. _202001010015
        (r"_(\d{8})",        "%Y%m%d"),       # Match e.g. _20200101
    ]

    for regex, date_fmt in patterns:
        match = re.search(regex, basename)
        if match:
            try:
                dt = datetime.strptime(match.group(1), date_fmt)
                return pd.Timestamp(dt)
            except ValueError:
                continue
    return None

def filter_tar_files_by_time(source_dir, start_time, end_time):
    """
    Returns a list of tar files in `source_dir` that have a timestamp between start_time and end_time.
    Sorted chronologically by detected timestamp.
    """
    start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    end_time = end_time.replace(hour=23, minute=59, second=59, microsecond=999999)

    all_files = glob.glob(os.path.join(source_dir, '*.tar'))
    filtered_files = []

    for file in all_files:
        timestamp = get_timestamp_from_filename(file)
        if timestamp and start_time <= timestamp <= end_time:
            filtered_files.append((timestamp, file))

    # Sort by timestamp
    filtered_files.sort(key=lambda x: x[0])
    return [file for _, file in filtered_files]

def generate_members_tar(file_path_or_obj, is_fileobj=False):
    with tarfile.open(name=file_path_or_obj if not is_fileobj else None,
                        fileobj=file_path_or_obj if is_fileobj else None,  
                        mode='r:*') as tar:
        for member in sorted(tar.getmembers(), key=lambda member_info: member_info.name):
            tar_extracted = tar.extractfile(member)
            fileobj = io.BytesIO(tar_extracted.read())
            tar_extracted.close()

            if member.name.endswith('.tar'):
                yield from generate_members_tar(fileobj, is_fileobj=True)
            else:
                ts = get_timestamp_from_filename(member.name)
                yield fileobj, ts

@njit(parallel=True, fastmath=True)
def compute_intensity_grid_numba(image_mmh: np.ndarray, 
                                block_size: int = 32,
                                clip_max: float = 50.0,
                                percentile: float = 90) -> np.ndarray:
    """
    Numba-accelerated intensity grid calculation with zero-copy operations
    """
    H, W = image_mmh.shape
    grid_h = H // block_size
    grid_w = W // block_size
    intensity_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    
    # Precompute percentile index for 32x32 blocks (1024 elements)
    idx = (percentile/100) * 1023  # 0-based index for 1024 elements
    
    for i in prange(grid_h):
        for j in prange(grid_w):
            # Calculate block boundaries
            h_start = i * block_size
            w_start = j * block_size
            block = image_mmh[h_start:h_start+block_size, 
                             w_start:w_start+block_size]
            
            # Fast percentile calculation using partition
            flat_block = block.ravel()
            
            # Clip values inline to avoid extra allocations
            max_val = 0.0
            for k in range(1024):
                val = flat_block[k]
                if val > clip_max:
                    val = clip_max
                if val > max_val:
                    max_val = val
            
            # Store scaled value (approximate percentile for speed)
            intensity_grid[i, j] = np.uint8((max_val / clip_max) * 255)
    
    return intensity_grid

class Antialiasing:
    def __init__(self):
        # Precompute 1D kernel for separable convolution
        x = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        self.kernel_1d = np.exp(-2 * x**2).astype(np.float32)
        self.kernel_1d /= self.kernel_1d.sum()
        
        # Cache for edge correction factors
        self.edge_cache: Dict[Tuple[int, int], np.ndarray] = {}
        
    @staticmethod
    @jit(nopython=True, parallel=True, fastmath=True)
    def _separable_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Numba-accelerated separable convolution implementation."""
        # Horizontal pass
        smoothed_x = np.empty_like(image)
        rows, cols = image.shape
        kernel_size = len(kernel)
        half_k = kernel_size // 2
        
        for i in prange(rows):
            for j in range(cols):
                acc = 0.0
                for k in range(-half_k, half_k + 1):
                    idx = j + k
                    if 0 <= idx < cols:
                        acc += image[i, idx] * kernel[k + half_k]
                smoothed_x[i, j] = acc
        
        # Vertical pass
        smoothed_xy = np.empty_like(smoothed_x)
        for j in prange(cols):
            for i in range(rows):
                acc = 0.0
                for k in range(-half_k, half_k + 1):
                    idx = i + k
                    if 0 <= idx < rows:
                        acc += smoothed_x[idx, j] * kernel[k + half_k]
                smoothed_xy[i, j] = acc
        
        return smoothed_xy

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply optimized antialiasing with edge correction."""
        img = image.astype(np.float32, copy=False)
        img_shape = img.shape
        
        # Get or compute edge correction factors
        if img_shape not in self.edge_cache:
            ones = np.ones(img_shape, dtype=np.float32)
            sum_conv = self._separable_convolution(ones, self.kernel_1d)
            self.edge_cache[img_shape] = 1.0 / sum_conv
        
        # Perform convolution and apply correction
        smoothed = self._separable_convolution(img, self.kernel_1d)
        return smoothed * self.edge_cache[img_shape]

def resize_radar_image(original_image: np.ndarray, 
                       target_dims: tuple, 
                       pad_value: int = 0,
                       crop_mode: str = 'topleft') -> np.ndarray:
    """
    Resize radar image to target dimensions through padding or cropping.
    
    Args:
        original_image: 2D numpy array with shape (height, width)
        target_dims: Tuple of (target_height, target_width)
        pad_value: Value to use for padding (default: 0)
        crop_mode: 'center' or 'topleft' cropping (default: 'topleft')
    
    Returns:
        Resized 2D numpy array with specified dimensions
    
    Raises:
        ValueError: For invalid input dimensions or crop modes
    """
    # Validate inputs
    if len(original_image.shape) != 2:
        raise ValueError(f"Input image must be 2D, got {original_image.shape}")
    
    if not all(isinstance(d, int) and d > 0 for d in target_dims):
        raise ValueError("Target dimensions must be positive integers")
    
    current_rows, current_cols = original_image.shape
    target_rows, target_cols = target_dims
    
    # No resizing needed
    if (current_rows, current_cols) == target_dims:
        return original_image.copy()
    
    # Calculate needed adjustments
    row_diff = target_rows - current_rows
    col_diff = target_cols - current_cols
    
    # Handle padding
    if row_diff > 0 or col_diff > 0:
        pad_spec = (
            (0, max(row_diff, 0)),  # Vertical padding (bottom only)
            (0, max(col_diff, 0))   # Horizontal padding (right only)
        )
        return np.pad(original_image, pad_spec, mode='constant', constant_values=pad_value)
    
    # Handle cropping
    if crop_mode == 'center':
        row_start = (current_rows - target_rows) // 2
        col_start = (current_cols - target_cols) // 2
    elif crop_mode == 'topleft':
        row_start, col_start = 0, 0
    else:
        raise ValueError(f"Invalid crop_mode: {crop_mode}. Use 'center' or 'topleft'")
    
    return original_image[
        row_start:row_start + target_rows,
        col_start:col_start + target_cols
    ]

def clutter_pixels(rdr): 
    """Determine if an image has clutter based on gradient magnitude.""" 
    gx, gy = np.gradient(rdr) 
    grad = np.hypot(gx, gy) 

    grad_sum = np.sum(grad > 30)
    return grad_sum
    # return grad_sum > 50  # Discard if more than 50 pixels with gradient > 30

def process_h5_file(file_obj, config):
    """
    Process an HDF5 file-like object and return:
      - timestamp (extracted from file_obj.name)
      - image data
      - max_intensity (optional)
    """
    try:
        with h5py.File(file_obj, "r") as h5_file:
            try:
                raw_image = h5_file["image1/image_data"][:]
                mask_value = h5_file["image1/calibration"].attrs['calibration_out_of_image']
            except KeyError as e:
                print(f"Error: Required dataset or attribute not found in HDF5 file: {e}")
                return None, None
            except Exception as e:
                print(f"Error reading HDF5 file: {e}")
                return None, None
            
        radar_image = np.where(raw_image == mask_value, 0, raw_image)
        result = {}
        clutter_val = 0

        main_var_name = next((var for var in config["vars"] if var != "max_intensity_grid"), None)
        if not main_var_name:
            print("Error: No valid main variable found in config")
            return None, None
            
        target_dims = config["vars"][main_var_name]["dims"]
        current_dims = radar_image.shape

        if target_dims != current_dims:
            radar_image = resize_radar_image(
                original_image=radar_image,
                target_dims=target_dims,
                pad_value=config.get("pad_value", 0),
                crop_mode=config.get("crop_mode", "topleft"),
            )

        # Convert to rain rate if configured
        if config.get("to_rainrate", False):
            radar_image = radar_image.astype(np.float32) * 0.12

        # Compute clutter if configured
        if config.get("compute_clutter", False):
            clutter_val = clutter_pixels(radar_image)
        
        # Apply antialiasing if configured
        if config.get("antialias", False):
            try:
                global antialiaser
                radar_image = antialiaser.apply(radar_image)
            except Exception as e:
                print(f"Antialiasing failed: {e}")

        if config.get("create_max_intensity", False) and 'max_intensity_grid' in config['vars']:
            result['max_intensity_grid'] = compute_intensity_grid_numba(radar_image)

        dtype = config['vars'][main_var_name]['dtype']
        result[main_var_name] = radar_image.astype(dtype)
        
        return result, clutter_val

    except Exception as e:
        print(f"Error processing H5 file: {e}")
        return None, None

def process_source_sequentially(root, source_config, time_map):
    # Get shard size from config (after alignment fix)
    shard_size = source_config['shard_size']
    source_files = filter_tar_files_by_time(source_config['path'], START_TIME, END_TIME)

    if source_config.get("compute_clutter", False):
        clutter_scores = np.zeros(len(time_map), dtype=CLUTTER_DTYPE)
    else:
        clutter_scores = None

    buffers = {var: {} for var in source_config['vars'].keys()}
    for var in buffers:
        buffers[var] = {
            "shard_map": {},  # shard_index: (indices, data)
        }

    # Separate buffer for max_intensity_grid
    max_intensity_buffer = []
    max_intensity_indices = []

    processed_indices = set()

    preivous_idx = 0
    for file_path in source_files:        
        print(f"Processing {source_config['name']}: {file_path}")

        for fileobj, timestamp in generate_members_tar(file_path):
            index = time_map.get(timestamp)
            if index is None: continue

            result, clutter_val = process_h5_file(fileobj, source_config)
            if result is None:
                continue

            # Store max intensity data separately
            if 'max_intensity_grid' in result:
                max_intensity_buffer.append(result['max_intensity_grid'])
                max_intensity_indices.append(index)
                del result['max_intensity_grid']

            if source_config.get("compute_clutter", False):
                clutter_scores[index] = clutter_val
            
            shard_idx = index // shard_size
            shard_idx_previous = preivous_idx // shard_size

            for var, data in result.items():
                if var not in source_config['vars']: continue

                # Initialize shard buffer if needed
                if shard_idx not in buffers[var]["shard_map"]:
                    buffers[var]["shard_map"][shard_idx] = {"indices": [], "data": []}

                # Add to shard buffer
                buffers[var]["shard_map"][shard_idx]["indices"].append(index)
                buffers[var]["shard_map"][shard_idx]["data"].append(data)

                # Write the previous shard when moved on to the next
                if shard_idx_previous < shard_idx:
                    processed_indices.update(buffers[var]["shard_map"][shard_idx_previous]["indices"])
                    write_shard(root[var], buffers[var]["shard_map"][shard_idx_previous])

            preivous_idx = index
    
    # Process remaining partial shards
    for var in buffers:
        # Write any complete shards in shard_map
        for shard_idx, buffer in buffers[var]["shard_map"].items():
            if buffer["indices"]:
                # again, collect before write_shard clears
                processed_indices.update(buffer["indices"])
                write_shard(root[var], buffer)

    # Write max intensity data all at once
    if max_intensity_buffer and 'max_intensity_grid' in root:
        # Sort by index
        sort_order = np.argsort(max_intensity_indices)
        sorted_indices = np.array(max_intensity_indices)[sort_order]
        sorted_data = np.stack(max_intensity_buffer)[sort_order]
        
        # Write in one operation
        root['max_intensity_grid'][sorted_indices] = sorted_data

    # Update time mask
    if processed_indices:
        idx_arr = np.fromiter(processed_indices, dtype=int)
        root['time_mask'][idx_arr] = True

    # Write clutter vector
    if source_config.get("compute_clutter", False) and clutter_scores is not None:
        root[CLUTTER_VAR_NAME][:] = clutter_scores

def write_shard(dataset, buffer):
    """Write a full shard or partial shard to Zarr"""
    if not buffer["indices"]:
        return
        
    indices = np.array(buffer["indices"])
    batch_data = np.stack(buffer["data"])
    
    # Sort by index for contiguous writing
    sort_order = np.argsort(indices)
    sorted_indices = indices[sort_order]
    sorted_data = batch_data[sort_order]
    
    # Write in contiguous blocks where possible
    start_idx = 0
    while start_idx < len(sorted_indices):
        end_idx = start_idx + 1
        # Find contiguous block
        while (end_idx < len(sorted_indices)) and \
              (sorted_indices[end_idx] == sorted_indices[end_idx-1] + 1):
            end_idx += 1
            
        # Write contiguous block
        block_start = sorted_indices[start_idx]
        block_end = sorted_indices[end_idx-1] + 1
        dataset[block_start:block_end] = sorted_data[start_idx:end_idx]
        
        start_idx = end_idx

    buffer["indices"].clear()
    buffer["data"].clear()

def main():
    root = zarr.open_group(ZARR_OUTPUT_PATH, mode='a')
    group = root.create_group('radar')
    print(f"Zarr store initialized at {ZARR_OUTPUT_PATH}")

    time_map = create_time_index_map(START_TIME, END_TIME, source_configs[0]["interval"])

    # Create antialiaser if any config needs it
    global antialiaser
    if any(config.get("antialias", False) for config in source_configs):
        antialiaser = Antialiasing()

    # Add metadata attributes
    group.attrs['start_time'] = str(START_TIME)
    group.attrs['end_time'] = str(END_TIME)
    group.attrs['interval_minutes'] = source_configs[0]["interval"]
    group.attrs['creation_date'] = str(pd.Timestamp.now())

    for config in source_configs:
        for var_name, var_config in config["vars"].items():
            shape = (len(time_map), *var_config['dims'])
            
            if var_name == "max_intensity_grid":
                # Store entire time dimension as one chunk
                chunks = (len(time_map), *var_config['dims'])
            else:
                chunks = (config['chunk_size'], *var_config['dims'])
            
            # Calculate shard shape as multiple of chunk shape
            chunk_size = config['chunk_size']
            shard_size = config['shard_size']
            if shard_size % chunk_size != 0:
                # Round up to nearest multiple
                shard_size = ((shard_size // chunk_size) + 1) * chunk_size
                
            shards = (shard_size, *var_config['dims'])

            group.create_array(
                name=var_name, 
                shape=shape, 
                chunks=chunks,
                dtype=var_config['dtype']
            )
            # Store processing metadata
            group[var_name].attrs.update({
                'fill_value': config['fill_value'],
                'pad_value': config.get('pad_value', 0),
                'to_rainrate': config.get('to_rainrate', False),
                'antialias': config.get('antialias', False),
                'product': config['name']
            })

    if any(config.get("create_timemask", False) for config in source_configs):
        group.create_array(
            name='time_mask', 
            shape=(len(time_map),), 
            chunks=(len(time_map),),
            dtype=bool, 
        )
        group['time_mask'][...] = False  # Initialize to all missing

    if any(config.get("compute_clutter", False) for config in source_configs):
        group.create_array(
            name=CLUTTER_VAR_NAME,
            shape=(len(time_map),),
            chunks=(len(time_map),),  # full array as one chunk
            dtype=CLUTTER_DTYPE,
        )
        group[CLUTTER_VAR_NAME][...] = 0
    
    for config in source_configs:
        process_source_sequentially(group, config, time_map)

    print(root.tree())

if __name__ == '__main__':
    main()
