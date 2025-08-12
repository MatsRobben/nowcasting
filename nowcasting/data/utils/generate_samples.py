"""
Spatiotemporal Sample Weight Generator

This module slices a large time series array into train/val/test splits,
applies efficient 3D pooling, and returns time-indexed sample weights.

Author: Mats Robben
Date: 2025-07-12
"""

import pandas as pd
import numpy as np
import numba as nb
import os

# Avoid TBB-related warnings in Numba
os.environ["NUMBA_THREADING_LAYER"] = "omp"

def assign_splits(
    full_times: pd.DatetimeIndex,
    split_rules: dict[str, dict[str, list[int]]],
    default_key: str = None
) -> dict[str, np.ndarray]:
    """
    Assigns splits to each timestamp in `full_times` based on datetime attribute rules.

    This function categorizes timestamps into named splits (e.g., "train", "test", "val")
    based on a hierarchy of user-defined rules. Each rule consists of a split name and
    a dictionary of datetime attribute constraints (e.g., specific years, months, weekdays).
    Timestamps are assigned to the first split rule they match. Any timestamps that do not
    match any explicit rule are assigned to a designated default split.

    Parameters:
        full_times : pd.DatetimeIndex
            A Pandas DatetimeIndex containing the complete sequence of timestamps
            to be categorized into splits.
        split_rules : dict[str, dict[str, list[int]]]
            A dictionary where keys are split names (e.g., "train", "test") and
            values are dictionaries defining the attribute rules for that split.
            The inner dictionaries map datetime attributes (e.g., "year", "month",
            "weekday") to lists of allowed integer values for those attributes.
            Rules are applied in the order they appear in this dictionary.
        default_key : str, optional
            The name of the split to which timestamps are assigned if they do not match
            any of the provided `split_rules`. If `None`, the last key in `split_rules`
            is automatically used as the default fallback.

    Returns:
        dict[str, np.ndarray]
            A dictionary where keys are the names of the splits and values are
            NumPy arrays of integer indices. Each array contains the original
            indices from `full_times` that correspond to the timestamps assigned
            to that specific split.

    Supported datetime-attributes for `split_rules`:
        - "year"   (e.g., 2021, 2022, 2023)
        - "month"  (1-12)
        - "day"    (1-31)
        - "hour"   (0-23)
        - "minute" (0-59)
        - "second" (0-59)
        - "weekday" (0=Monday, 6=Sunday)

    Raises:
        KeyError: If a datetime attribute specified in `split_rules` is not supported.

    Usage Example:
    ```python
    full_times = pd.date_range("2020-01-01", "2023-12-31", freq="5min")
    split_rules = {
        "test": {"year": [2023]},
        "val":  {"month": [6, 11], "year": [2020, 2021, 2022]},
        "train": {} # Implicitly everything else
    }
    default_split_name = "train"
    splits = assign_splits(full_times, split_rules, default_split_name)

    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    ```
    """

    if default_key is None:
        default_key = list(split_rules.keys())[-1]

    labels = np.array([default_key] * len(full_times), dtype=object)
    attrs = {
        "year":    full_times.year,
        "month":   full_times.month,
        "day":     full_times.day,
        "hour":    full_times.hour,
        "minute":  full_times.minute,
        "second":  full_times.second,
        "weekday": full_times.weekday,
    }

    for name, selector in split_rules.items():
        if name == default_key:
            continue  # skip default; it fills in the rest
        mask = (labels == default_key)
        for key, allowed in selector.items():
            if key not in attrs:
                raise KeyError(f"Unsupported datetime property: {key}")
            mask &= np.isin(attrs[key], allowed)
        labels[mask] = name
    
    result = {
        split: np.nonzero(labels == split)[0]
        for split in set(labels)
    }
    return result

# Spatial pooling functions
@nb.njit(parallel=True, fastmath=True, nogil=True)
def spatial_max_pool(input_data: np.ndarray, output: np.ndarray, kh: int, kw: int, sh: int, sw: int):
    """
    Performs 2D spatial max pooling on 3D input data (time, height, width) in parallel.

    This Numba-jitted function applies a max pooling operation across the spatial
    dimensions (height and width) for each time slice. It's optimized for performance
    using Numba's just-in-time compilation, parallelization, and fast math.

    Parameters:
        input_data : np.ndarray
            A 3D NumPy array of shape (n_times, in_h, in_w) representing the input data.
        output : np.ndarray
            A pre-allocated 3D NumPy array of shape (n_times, out_h, out_w) to store
            the pooling results. Must have the same dtype as `input_data`.
        kh : int
            Kernel height. The height of the pooling window.
        kw : int
            Kernel width. The width of the pooling window.
        sh : int
            Stride height. The vertical step size of the pooling window.
        sw : int
            Stride width. The horizontal step size of the pooling window.
    """
    n_times, in_h, in_w = input_data.shape
    # Calculate output dimensions based on input size, kernel size, and stride.
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    
    # Outer loop parallelized across time dimension.
    for t in nb.prange(n_times):
        for i in range(out_h):
            h_start = i * sh # Starting row for the current pooling window
            for j in range(out_w):
                w_start = j * sw # Starting column for the current pooling window
                max_val = 0 # Initialize max value for the current window
                # Optimized inner loops

                # Iterate over the kernel window to find the maximum value.
                for di in range(kh):
                    for dj in range(kw):
                        val = input_data[t, h_start+di, w_start+dj]
                        max_val = max(max_val, val)
                output[t, i, j] = max_val

@nb.njit(parallel=True, fastmath=True, nogil=True)
def spatial_mean_pool(input_data: np.ndarray, output: np.ndarray, kh: int, kw: int, sh: int, sw: int):
    """
    Performs 2D spatial mean pooling on 3D input data (time, height, width) in parallel.

    This Numba-jitted function applies a mean pooling operation across the spatial
    dimensions (height and width) for each time slice. It's optimized for performance
    and uses float32 precision for intermediate sums to maintain accuracy. The `output`
    array will store the sum of values within each kernel window, which is later
    divided by the kernel size to get the mean.

    Parameters:
        input_data : np.ndarray
            A 3D NumPy array of shape (n_times, in_h, in_w) representing the input data.
        output : np.ndarray
            A pre-allocated 3D NumPy array of shape (n_times, out_h, out_w) with dtype
            `np.float32` to store the sum of values for each pooling window.
        kh : int
            Kernel height. The height of the pooling window.
        kw : int
            Kernel width. The width of the pooling window.
        sh : int
            Stride height. The vertical step size of the pooling window.
        sw : int
            Stride width. The horizontal step size of the pooling window.
    """
    n_times, in_h, in_w = input_data.shape
    # Calculate output dimensions based on input size, kernel size, and stride.
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    
    # Outer loop parallelized across time dimension.
    for t in nb.prange(n_times):
        for i in range(out_h):
            h_start = i * sh # Starting row for the current pooling window
            for j in range(out_w):
                w_start = j * sw # Starting column for the current pooling window
                total = 0.0  # Initialize sum for the current window (float for precision)
                
                # Iterate over the kernel window to sum values.
                for di in range(kh):
                    for dj in range(kw):
                        total += input_data[t, h_start+di, w_start+dj]
                output[t, i, j] = total  # Store sum for later processing

# Temporal pooling functions
@nb.njit(parallel=True, fastmath=True, nogil=True)
def temporal_max_pool(input_data: np.ndarray, output: np.ndarray, kt: int, st: int):
    """
    Performs 1D temporal max pooling on 3D input data (time, height, width) in parallel.

    This Numba-jitted function applies a max pooling operation along the time dimension
    for each spatial (height, width) location independently. It's optimized for performance
    using Numba's just-in-time compilation, parallelization, and fast math.

    Parameters:
        input_data : np.ndarray
            A 3D NumPy array of shape (n_times, h, w) representing the input data.
        output : np.ndarray
            A pre-allocated 3D NumPy array of shape (out_t, h, w) to store the
            pooling results. Must have the same dtype as `input_data`.
        kt : int
            Kernel time. The length of the pooling window along the time dimension.
        st : int
            Stride time. The step size of the pooling window along the time dimension.
    """
    n_times, h, w = input_data.shape
    # Calculate output time dimension based on input time, kernel time, and stride.
    out_t = (n_times - kt) // st + 1
    
    # Parallelize across all spatial locations (flattened for simpler prange).
    for spatial_idx in nb.prange(h * w):
        # Convert the flattened spatial index back to (h_idx, w_idx) coordinates.
        h_idx = spatial_idx // w
        w_idx = spatial_idx % w

        for t_out in range(out_t):
            t_start = t_out * st  # Starting time index for the current pooling window
            max_val = 0  # Initialize max value for the current window
            
            # Iterate over the temporal kernel window to find the maximum value.
            for dt in range(kt):
                val = input_data[t_start + dt, h_idx, w_idx]
                max_val = max(max_val, val)
            output[t_out, h_idx, w_idx] = max_val

@nb.njit(parallel=True, fastmath=True, nogil=True)
def temporal_mean_pool(input_data: np.ndarray, output: np.ndarray, kt: int, st: int):
    """
    Performs 1D temporal mean pooling on 3D input data (time, height, width) in parallel.

    This Numba-jitted function applies a mean pooling operation along the time dimension
    for each spatial (height, width) location independently. It's optimized for performance
    and uses float32 precision for intermediate sums to maintain accuracy. The `output`
    array will store the sum of values within each temporal kernel window, which is later
    divided by the kernel size to get the mean.

    Parameters:
        input_data : np.ndarray
            A 3D NumPy array of shape (n_times, h, w) representing the input data.
        output : np.ndarray
            A pre-allocated 3D NumPy array of shape (out_t, h, w) with dtype
            `np.float32` to store the sum of values for each pooling window.
        kt : int
            Kernel time. The length of the pooling window along the time dimension.
        st : int
            Stride time. The step size of the pooling window along the time dimension.
    """
    n_times, h, w = input_data.shape
    # Calculate output time dimension based on input time, kernel time, and stride.
    out_t = (n_times - kt) // st + 1
    
    # Parallelize across all spatial locations (flattened for simpler prange).
    for spatial_idx in nb.prange(h * w):
        h_idx = spatial_idx // w
        w_idx = spatial_idx % w
        for t_out in range(out_t):
            t_start = t_out * st  # Starting time index for the current pooling window
            total = 0.0  # Initialize sum for the current window (float for precision)
            for dt in range(kt):
                total += input_data[t_start + dt, h_idx, w_idx]
            output[t_out, h_idx, w_idx] = total  # Store the sum

def pool_3d(input_array: np.ndarray, kernel_size: tuple[int, int, int], stride: tuple[int, int, int] = (1, 1, 1), mode: str = 'max') -> np.ndarray:
    """
    Performs 3D pooling (max or mean) on a 3D NumPy array.

    This function orchestrates 3D pooling by applying spatial pooling followed by
    temporal pooling. It supports both 'max' and 'mean' pooling modes and handles
    data type conversions for precision in mean pooling.

    Parameters:
        input_array : np.ndarray
            A 3D NumPy array of shape (time, height, width) representing the data to be pooled.
        kernel_size : tuple[int, int, int]
            A tuple (temporal_k, height_k, width_k) specifying the dimensions of the
            pooling kernel along the time, height, and width axes, respectively.
        stride : tuple[int, int, int], optional
            A tuple (temporal_stride, height_stride, width_stride) specifying the
            stride (step size) of the pooling window along each dimension.
            Defaults to (1, 1, 1).
        mode : str, optional
            The pooling mode to apply. Can be 'max' for max pooling or 'mean' for mean pooling.
            Defaults to 'max'.

    Returns:
        np.ndarray
            A 3D NumPy array containing the pooled results, with reduced dimensions
            based on the kernel size and stride. For 'mean' pooling, the output
            dtype will be `np.float16` for memory efficiency after calculation.

    Raises:
        ValueError:
            - If `input_array` is not a 3D array.
            - If any dimension of `kernel_size` exceeds the corresponding dimension of `input_array`.
            - If any kernel or stride value is less than 1.
            - If `mode` is not 'max' or 'mean'.
    """

    if input_array.ndim != 3:
        raise ValueError("Input must be 3D array (time, height, width)")
    
    kt, kh, kw = kernel_size
    st, sh, sw = stride
    n_times, in_h, in_w = input_array.shape

    # Validate dimensions and parameters
    if kh > in_h or kw > in_w or kt > n_times:
        raise ValueError(
            f"Kernel size {kernel_size} exceeds input dimensions "
            f"({n_times}, {in_h}, {in_w})."
        )
    if any(s < 1 for s in (kt, kh, kw, st, sh, sw)):
        raise ValueError("Kernel and stride values must be ≥1.")
    if mode not in ['max', 'mean']:
        raise ValueError(f"Invalid pooling mode: '{mode}'. Must be 'max' or 'mean'.")

    # Calculate output dimensions
    out_h = (in_h - kh) // sh + 1
    out_w = (in_w - kw) // sw + 1
    out_t = (n_times - kt) // st + 1

    if mode == 'max':
        # For max pooling, intermediate and final outputs have the same dtype as input.
        spatial_out = np.empty((n_times, out_h, out_w), dtype=input_array.dtype)
        spatial_max_pool(input_array, spatial_out, kh, kw, sh, sw)
        
        final_out = np.empty((out_t, out_h, out_w), dtype=input_array.dtype)
        temporal_max_pool(spatial_out, final_out, kt, st)
        return final_out

    else:  # mode == 'mean'
        # For mean pooling, use float32 for intermediate sums to preserve precision.
        spatial_out = np.empty((n_times, out_h, out_w), dtype=np.float32)
        spatial_mean_pool(input_array, spatial_out, kh, kw, sh, sw)
        
        temp_out = np.empty((out_t, out_h, out_w), dtype=np.float32)
        temporal_mean_pool(spatial_out, temp_out, kt, st)
        
        # Calculate the final mean by dividing the sum by the total number of elements in the kernel.
        kernel_product = kt * kh * kw
        mean_values = temp_out / kernel_product

        # Convert the final mean values to float16 for memory efficiency if desired.
        return mean_values.astype(np.float16)

def generate_weighted_samples(
    sample_array: np.ndarray,
    indices: np.ndarray,
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    agg_method: str = 'mean_pool',
    center_crop: bool = False
    ) -> pd.DataFrame:
    """
    Generates a Pandas DataFrame of weighted samples by applying 3D pooling
    to contiguous blocks of data identified by `indices`.

    This function processes segments of `sample_array` corresponding to `indices`,
    performing 3D pooling (max or mean) to aggregate values. It's designed to
    extract "samples" or "patches" from a larger data cube, associating them
    with their time, height, and width indices and an aggregated "weight" (value).

    Parameters:
        sample_array : np.ndarray
            The full 3D NumPy array (time, height, width) from which samples are drawn.
        indices : np.ndarray
            A 1D NumPy array of integer indices representing the time steps to consider.
            These indices are expected to be from the `full_times` array used
            with `assign_splits`. The function will process contiguous blocks of
            indices within this array.
        kernel : tuple[int, int, int]
            A tuple (temporal_k, height_k, width_k) defining the pooling kernel size.
        stride : tuple[int, int, int]
            A tuple (temporal_stride, height_stride, width_stride) defining the
            pooling stride.
        agg_method : str, optional
            The aggregation method to use for pooling. Can be 'max_pool' or 'mean_pool'.
            Defaults to 'mean_pool'.
        center_crop : bool, optional
            If `True`, before pooling, a spatial center crop of `sample_array`
            matching the `kernel`'s spatial dimensions (h_k, w_k) is performed
            on the `block_data`. If `False`, the entire spatial extent is used.
            Defaults to `False`.

    Returns:
        pd.DataFrame
            A Pandas DataFrame with columns:
            - 't_idx' (np.uint32): The starting time index of the pooled window.
            - 'h_idx' (np.uint8): The starting height index of the pooled window.
            - 'w_idx' (np.uint8): The starting width index of the pooled window.
            - 'weight' (np.uint8 for max_pool, np.float16 for mean_pool): The
              aggregated value (max or mean) of the pooled window.
            If `indices` is empty or no valid blocks are found, an empty DataFrame
            with the specified columns is returned.
    """

    if len(indices) == 0:
        # Return an empty DataFrame with correct columns and dtypes if no indices are provided.
        return pd.DataFrame(columns=['t_idx', 'h_idx', 'w_idx', 'weight']).astype({
            't_idx': np.uint32, 'h_idx': np.uint8, 'w_idx': np.uint8, 'weight': np.float16
        })

    t_k, h_k, w_k = kernel
    t_s, h_s, w_s = stride
    
    # Identify contiguous blocks of indices to ensure pooling is applied to continuous time series.
    # `np.diff(indices) > 1` finds gaps, and `np.where(...)[0] + 1` gets the split points.
    splits = np.where(np.diff(indices) > 1)[0] + 1
    contiguous_blocks = np.split(indices, splits)

    all_t_idx = []
    all_h_idx = []
    all_w_idx = []
    all_weights = []
    
    for block in contiguous_blocks:
        # A block must be at least as long as the temporal kernel to be valid for pooling.
        if len(block) < t_k:
            continue

        if center_crop:
            # Determine the spatial crop boundaries for center_crop.
            h, w = sample_array.shape[-2:] # Get height and width of the full array
            crop_h, crop_w = kernel[-2:] # Spatial kernel dimensions
            h0 = h//2 - crop_h//2
            w0 = w//2 - crop_w//2
            h1 = h0 + crop_h
            w1 = w0 + crop_w

            block_data = sample_array[block, h0:h1, w0:w1]
            h_offset = h0 # Store the offset for correct spatial index calculation
            w_offset = w0
        else:
            h_offset = 0
            w_offset = 0
            block_data = sample_array[block, :, :]
        
        if agg_method == 'max_pool':
            pooled_weights = pool_3d(block_data, kernel, stride, 'max')
            weight_dtype = np.uint8 # Max pooling typically returns values of the input type uint8, as the input is stored in that format
        elif agg_method == 'mean_pool':
            pooled_weights = pool_3d(block_data, kernel, stride, 'mean')
            weight_dtype = np.float16 # Mean pooling outputs float16 as per pool_3d
        
        if pooled_weights.size == 0:
            continue
            
        out_t, out_h, out_w = pooled_weights.shape

        # Calculate the original time, height, and width indices corresponding to the pooled outputs.
        t_starts = block[0] + np.arange(out_t) * t_s
        h_starts = h_offset + np.arange(out_h) * h_s
        w_starts = w_offset + np.arange(out_w) * w_s

        # Create a meshgrid to get all combinations of (t_idx, h_idx, w_idx) for the pooled data.
        t_coords, h_coords, w_coords = np.meshgrid(
            t_starts, h_starts, w_starts, indexing='ij'
        )
        
        all_t_idx.append(t_coords.ravel())
        all_h_idx.append(h_coords.ravel())
        all_w_idx.append(w_coords.ravel())
        all_weights.append(pooled_weights.ravel())

    if not all_t_idx:
        # If no samples were generated (e.g., all blocks too short), return an empty DataFrame.
        return pd.DataFrame(columns=['t_idx', 'h_idx', 'w_idx', 'weight']).astype({
            't_idx': np.uint32, 'h_idx': np.uint8, 'w_idx': np.uint8, 'weight': np.float16
        })
    
    # Concatenate results from all blocks.
    all_t_idx = np.concatenate(all_t_idx)
    all_h_idx = np.concatenate(all_h_idx)
    all_w_idx = np.concatenate(all_w_idx)
    all_weights = np.concatenate(all_weights)
    
    df = pd.DataFrame({
        't_idx': all_t_idx,
        'h_idx': all_h_idx,
        'w_idx': all_w_idx,
        'weight': all_weights,
    })
    
    # Cast columns to specified dtypes for memory efficiency and consistency.
    df['t_idx'] = df['t_idx'].astype(np.uint32)
    df['h_idx'] = df['h_idx'].astype(np.uint8)
    df['w_idx'] = df['w_idx'].astype(np.uint8)
    df['weight'] = df['weight'].astype(weight_dtype)

    return df


def get_sample_dfs(
    sample_array: np.ndarray,
    kernel: tuple[int, int, int],
    stride: tuple[int, int, int],
    timestamps: pd.DatetimeIndex,
    split_rules: dict[str, dict[str, list[int]]],
    methods: dict[str, dict[str, any]],
    time_mask: np.ndarray = None
) -> dict[str, pd.DataFrame]:
    """
    Generates a dictionary of Pandas DataFrames, where each DataFrame contains
    weighted samples for a specific data split (e.g., 'train', 'val', 'test').

    This function integrates the time-based splitting logic with the 3D pooling
    mechanism. It first assigns timestamps to different splits based on `split_rules`,
    then for each split, it generates weighted samples by applying 3D pooling
    to the `sample_array` for the relevant time indices.

    Parameters:
        sample_array : np.ndarray
            The full 3D NumPy array (time, height, width) of the data.
        kernel : tuple[int, int, int]
            A tuple (temporal_k, height_k, width_k) defining the pooling kernel size.
        stride : tuple[int, int, int]
            A tuple (temporal_stride, height_stride, width_stride) defining the
            pooling stride.
        timestamps : pd.DatetimeIndex
            The complete Pandas DatetimeIndex corresponding to the time dimension
            of `sample_array`. Used by `assign_splits` to categorize time periods.
        split_rules : dict[str, dict[str, list[int]]]
            A dictionary defining the rules for assigning timestamps to different
            data splits. Passed directly to `assign_splits`.
        methods : dict[str, dict[str, any]]
            A dictionary configuring the pooling method for each split. Keys are
            split names (e.g., 'train', 'val'), and values are dictionaries with
            pooling parameters:
            - 'agg': ('max_pool' or 'mean_pool') - The aggregation method.
            - 'center_crop': (bool) - Whether to apply a spatial center crop.
            If a split name is not found, 'mean_pool' is used as the default for 'agg'.
        time_mask : np.ndarray, optional
            A 1D boolean NumPy array, of the same length as `timestamps`, indicating
            valid time steps (True) or missing/invalid time steps (False). If provided,
            only valid time steps within each split will be considered for sample generation.
            Defaults to `None`, meaning all time steps are considered valid.

    Returns:
        dict[str, pd.DataFrame]
            A dictionary where keys are split names (e.g., "train", "val", "test")
            and values are Pandas DataFrames generated by `generate_weighted_samples`
            for that specific split. Each DataFrame contains 't_idx', 'h_idx', 'w_idx',
            and 'weight' columns.
    """
    # First, assign timestamps to their respective splits based on the defined rules.
    splits = assign_splits(timestamps, split_rules)

    dfs = {}
    for split_name, indices in splits.items():
        # If a time_mask is provided, filter the indices to include only valid timestamps.
        if time_mask is not None:
            # Ensure the indices are within the bounds of time_mask
            valid_indices_mask_for_split = (indices >= 0) & (indices < len(time_mask))
            indices = indices[valid_indices_mask_for_split]

            # Apply the time_mask to the current split's indices
            valid_mask_from_time_mask = time_mask[indices]
            indices = indices[valid_mask_from_time_mask]
        
        # Get specific pooling method configurations for the current split.
        # Default to 'mean_pool' and no center crop if not specified for a split.
        split_methods = methods.get(split_name, {'agg': 'mean_pool', 'center_crop': False})
        agg_method = split_methods.get('agg', 'mean_pool')
        center_crop = split_methods.get('center_crop', False)
        
        # Generate weighted samples for the current split using the specified parameters.
        dfs[split_name] = generate_weighted_samples(
                            sample_array=sample_array, 
                            indices=indices, 
                            kernel=kernel, stride=stride, 
                            agg_method=agg_method,
                            center_crop=center_crop
                     )

    return dfs


if __name__ == "__main__":
    import zarr

    # Example Usage:

    # 1) Define your full time axis. This represents the entire duration of your dataset.
    full_times = pd.date_range("2020-01-01 00:00", "2023-12-31 23:59", freq="5min")

    # 2) Define your split rules in priority order.
    # The first rule that a timestamp matches will assign it to that split.
    split_rules = {
        "test": {"year": [2023]},  # All of 2023 goes to the 'test' split.
        "val":  {"month": [6, 11]}, # June and November of any year (that's not 2023) go to 'val'.
        "train": {} # All remaining timestamps go to 'train'. This acts as the default.
    }

    # 3) Load your sample array (e.g., radar intensity data) using Zarr.
    # Replace this with the actual path to your Zarr dataset.
    root = zarr.open_group('/projects/prjs1634/nowcasting/data/dataset.zarr', mode='r')
    sample_array = root['radar/max_intensity_grid'][:]

    def create_time_mask(root, num_timesteps):
        "Helper function, this functionality is usually handeled in the datamodule."

        global_mask = np.ones(num_timesteps, dtype=bool)
        
        # Mask the missing data
        for group_name in ['radar', 'harmonie', 'sat_l2', 'sat_l1p5', 'aws']:
            if group_name not in root:
                continue

            group = root[group_name]
            factor = group.attrs['interval_minutes'] // 5
            
            if 'time_mask' not in group:
                continue

            group_mask = group['time_mask'][:]
            
            global_mask &= np.repeat(group_mask, factor)

        # Masked based on clutter
        clutter = root['radar']['clutter_score'][:]
        clutter_mask = clutter < 50
        global_mask &= clutter_mask

        return global_mask

    # 4) Create a time mask to exclude invalid or missing data points.
    # This example uses a custom function to create the mask based on Zarr group properties.
    time_mask = create_time_mask(root, len(sample_array))

    # 5) Define the pooling kernel and stride.
    kernel = (22, 8, 8) # (Temporal Kernel, Height Kernel, Width Kernel)
    stride = (1, 1, 1) # (Temporal Stride, Height Stride, Width Stride)

    # 6) Configure aggregation methods and other options for each split.
    methods = {
        'train': {'agg': 'mean_pool'},
        'val': {'agg': 'mean_pool', 'center_crop': True}, # 'val' split will use mean pooling and center crop
        'test': {'agg': 'mean_pool'}
    }

    # 7) Call the main function to get the split DataFrames.
    print("Generating split DataFrames...")
    dfs = get_sample_dfs(sample_array=sample_array,
                 kernel=kernel,
                 stride=stride,
                 timestamps=full_times,
                 split_rules=split_rules,
                 methods=methods,
                 time_mask=time_mask
                 )
    
    # 8) Define the output directory for saving the Parquet files.
    output_dir = "/projects/prjs1634/nowcasting/data/tables"

    # 9) Save each DataFrame.
    file_format = 'csv'
    for name, df in dfs.items():
        df = df[df['weight'] > (0.01 * (255/50))]
        
        # Define file path
        file_path = os.path.join(output_dir, f"{name}.{file_format}")

        # Save in the specified format
        if file_format == 'parquet':
            df.to_parquet(file_path)
        elif file_format == 'csv':
            df.to_csv(file_path, index=False)
        else:
            raise ValueError("Unsupported file format. Use 'parquet' or 'csv'.")
        print(name)
        print(df)