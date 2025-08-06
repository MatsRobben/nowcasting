"""
Transforms for nowcasting data

This module contains several transforms that can be applied to the loaded data.

Author: Mats Robben
Date: 2025-07-12
"""

import numpy as np
import torch

def get_transforms(transform_name: str, params: list):
    """
    Factory function to retrieve and initialize a specific data transformation.

    This function acts as a dispatcher, returning a configured transform function
    based on `transform_name` and `params`. It uses a `match` statement for
    clear mapping of transform names to their respective creation functions.

    Parameters:
        transform_name : str
            The string identifier for the desired transform (e.g., 'default_rainrate',
            'dbz_normalization').
        params : list or dict
            A list or dictionary of parameters to pass to the transform's
            initialization function. The specific type and content depend on the
            chosen transform.

    Returns:
        Callable or None
            A callable transformation function configured with the provided parameters,
            or `None` if `transform_name` does not correspond to a known transform.
    """

    # Use a dictionary to store transform creation functions, providing more flexibility
    # and extensibility than a direct match statement for complex parameter handling
    # or future additions. The `**params` or `*params` is handled by the individual
    # transform functions.
    transform_factories = {
        'default_rainrate': default_rainrate_transform,
        'dbz_normalization': dbz_normalization_transform,
        'normalize': normalize_transform,
        'resize': resize_transform,
        'dropout': dropout_transform,
        'aws_interpolation': aws_transform,
    }

    transform_func = transform_factories.get(transform_name)
    if transform_func:
        # Check if params is a list or dict for appropriate unpacking
        if isinstance(params, dict):
            return transform_func(**params)
        elif isinstance(params, (list, tuple)):
            # Assuming if it's a list/tuple, it needs to be unpacked as positional arguments
            return transform_func(*params)
        else:
            print(f"Warning: Unexpected parameter type for {transform_name}. Trying to call directly.")
            return transform_func(params) # Attempt to pass directly if not a dict/list
    else:
        print(f'Error: The transform "{transform_name}" does not exist or is not registered.')
        return None
        
def combine(transforms: list[callable]):
    """
    Combines multiple transformation functions into a single sequential transform.

    This higher-order function takes a list of transform functions and returns
    a new function that applies each transform in the list sequentially to the input data.
    The output of one transform becomes the input for the next.

    Parameters:
        transforms : list[callable]
            A list of callable transformation functions. Each function should accept
            the output of the previous function as its input.

    Returns:
        Callable
            A single transformation function that, when called with `raw_data`,
            applies all transforms in the `transforms` list in order.
    """
    def transform(raw_data):
        """
        Applies each transformation in the sequence to the raw data.
        """
        for t in transforms:
            raw_data = t(raw_data)
        return raw_data
    
    return transform
        
def set_transforms(transform_dict: dict):
    """
    Initializes and sets transformation pipelines for each variable in a dictionary.

    This function iterates through a dictionary where keys are variable names
    and values define the transformations to be applied to that variable.
    If a transformation definition is a dictionary (implying a sequence of transforms),
    it constructs a combined transformation pipeline using `get_transforms` and `combine`.
    Callable transformation definitions are left unchanged.

    Parameters:
        transform_dict : dict
            A dictionary where keys are variable names (e.g., 'radar', 'temperature')
            and values are either:
            - A callable transformation function (already initialized).
            - A dictionary where keys are `transform_name` strings and values are
              parameter dictionaries or lists for `get_transforms`. This defines
              a sequence of transforms for the variable.

    Returns:
        None
            The `transform_dict` is modified in place, with transformation definitions
            replaced by initialized, callable transform functions or combined pipelines.
    """
    for var, transform_definition in transform_dict.items():
        if callable(transform_definition):
            # If the transform definition is already a callable, skip it.
            continue

        transforms = []
        # Ensure transform_definition is a dictionary before iterating, if not, it's malformed.
        if isinstance(transform_definition, dict):
            for transform_name, params in transform_definition.items():
                # Append each initialized transform to the list.
                transform = get_transforms(transform_name, params)
                if transform:
                    transforms.append(transform)
        else:
            print(f"Warning: Transform definition for variable '{var}' is not a callable or a dictionary. Skipping.")
            continue # Skip malformed entries

        # Combine all individual transforms into a single callable pipeline.
        transform_dict[var] = combine(transforms)

def aws_transform(target_shape: tuple[int, int], method: str = 'dense', radius: int = 3, power: int = 2, search_radius: int = 5, downscale: int = 8):
    """
    Creates an AWS (Automatic Weather Station) data interpolation transform function.

    This transform handles sparse station measurements and interpolates them onto
    a regular grid. It supports two main methods: 'sparse' (drawing disks around
    station locations) and 'dense' (Inverse Distance Weighting - IDW interpolation).

    Parameters:
        target_shape : tuple[int, int]
            The desired (height, width) of the output grid.
        method : str, optional
            The interpolation method to use.
            - 'sparse': Fills a disk of `radius` around each station with its value.
            - 'dense': Uses Inverse Distance Weighting (IDW) interpolation.
            Defaults to 'dense'.
        radius : int, optional
            Only applicable for `method='sparse'`. The radius of the disk (in grid units)
            to fill around each station. Defaults to 3.
        power : int, optional
            Only applicable for `method='dense'`. The power parameter for IDW. Higher
            values give more weight to closer points. Defaults to 2.
        search_radius : int, optional
            Only applicable for `method='dense'`. The number of nearest stations
            to consider for interpolation. Defaults to 5.
        downscale : int, optional
            Only applicable for `method='dense'`. A factor by which to downscale the
            grid before IDW interpolation and then upscale the result. This can
            improve performance for very large grids. Defaults to 8.

    Returns:
        Callable
            A function that takes `(values, locations)` as input and returns
            the interpolated grid(s).
            - `values`: 1D or 2D array of station measurements (num_stations) or (num_images, num_stations).
            - `locations`: 2D array of station coordinates (num_stations, 2) in grid units.

    Raises:
        ValueError: If an invalid `method` is provided.
    """
    from scipy.spatial import KDTree
    from scipy.ndimage import zoom

    grid_height, grid_width = target_shape
    
    # Precompute grid points for dense method interpolation to avoid recomputing per call.
    if method == 'dense':
        width_down = grid_width // downscale
        height_down = grid_height // downscale
        
        # Create grid points for interpolation
        x_grid, y_grid = np.meshgrid(
            np.linspace(0, width_down - 1, width_down),
            np.linspace(0, height_down - 1, height_down)
        )
        grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    def transform(input_data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        Applies the AWS interpolation to the input station data.
        
        Args:
            input_data: A tuple containing:
                - values: 1D (for a single time step) or 2D (for multiple time steps)
                  NumPy array of station measurements. Shape: (num_stations,) or (num_images, num_stations).
                - locations: 2D NumPy array of station coordinates (num_stations, 2),
                  where each row is (x_coordinate, y_coordinate) in grid units.
                
        Returns:
            np.ndarray:
                The interpolated grid(s).
                - Shape (grid_height, grid_width) for a single input image.
                - Shape (num_images, grid_height, grid_width) for multiple input images.
                Returns an array of zeros if no station data is available.
        """
        values, locations = input_data
        single_image = values.ndim == 1
        
        if single_image:
            # Convert 1D values to 2D for consistent processing (add a time dimension).
            values = values[np.newaxis, :]
        
        num_images = values.shape[0]
        
        if method == 'sparse':
            # Initialize the result array with zeros.
            if single_image:
                result = np.zeros((grid_height, grid_width), dtype=np.float32)
            else:
                result = np.zeros((num_images, grid_height, grid_width), dtype=np.float32)
                
            if values.size == 0:
                return result
                
            # Extract integer coordinates for direct indexing
            x_coords = locations[:, 0].astype(int)
            y_coords = locations[:, 1].astype(int)
            
            # Precompute disk masks for each station.
            # This avoids repeated calculations inside the time loop for static locations.
            disk_masks = []
            for xi, yi in zip(x_coords, y_coords):
                # Ensure coordinates are within grid bounds before creating mask.
                if 0 <= yi < grid_height and 0 <= xi < grid_width:
                    # Define the bounding box for the disk.
                    y_range = np.clip(np.arange(yi-radius, yi+radius+1), 0, grid_height-1)
                    x_range = np.clip(np.arange(xi-radius, xi+radius+1), 0, grid_width-1)
                    yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
                    # Calculate distances from the station center and apply radius mask.
                    mask = (xx - xi)**2 + (yy - yi)**2 <= radius**2
                    disk_masks.append((yy[mask], xx[mask]))
                else:
                    disk_masks.append(None) # Mark as invalid station if out of bounds
            
            # Apply precomputed disk masks for each image.
            for t in range(num_images):
                img = result[t] if not single_image else result
                for i, mask_coords in enumerate(disk_masks):
                    if mask_coords is not None:
                        # Assign station value to all pixels within its disk.
                        img[mask_coords] = values[t, i]
            
            return result
            
        elif method == 'dense':
            # Dense method - IDW interpolation

            # Handle empty input gracefully.
            if values.size == 0:
                if single_image:
                    return np.zeros((grid_height, grid_width), dtype=np.float32)
                else:
                    return np.zeros((num_images, grid_height, grid_width), dtype=np.float32)
                    
            # Scale station coordinates to the downscaled grid.
            scaled_locs = locations / downscale
            num_stations = scaled_locs.shape[0]
            
            # Initialize output array
            if single_image:
                result = np.zeros((grid_height, grid_width), dtype=np.float32)
            else:
                result = np.zeros((num_images, grid_height, grid_width), dtype=np.float32)
            
            # Process each image (time step).
            for t in range(num_images):
                # Combine scaled locations with current image's values.
                stations = np.column_stack((scaled_locs, values[t]))
                
                # Create a KDTree for efficient nearest neighbor search.
                # `k` ensures we don't request more neighbors than available stations.
                k = min(search_radius, num_stations)
                tree = KDTree(stations[:, :2])

                # Query the KDTree for distances and indices of nearest neighbors for each grid point.
                distances, indices = tree.query(grid_points, k=k)
                
                # If k=1, `tree.query` might return 1D arrays for distances/indices.
                # Ensure they are 2D for consistent operations.
                if k == 1:
                    distances = distances[:, None]
                    indices = indices[:, None]
                
                # Calculate Inverse Distance Weighting. Add a small epsilon to avoid division by zero.
                weights = 1 / (distances**power + 1e-6)

                # Calculate weighted sum of values and sum of weights.
                # `stations[indices, 2]` retrieves the values for the nearest stations.
                weighted_sum = np.sum(stations[indices, 2] * weights, axis=1)
                sum_weights = np.sum(weights, axis=1)

                # Compute the IDW interpolated values.
                idw_values = weighted_sum / sum_weights
                
                # Reshape the 1D interpolated values back to the downscaled grid.
                idw_grid = idw_values.reshape((height_down, width_down))
                
                # Upscale the interpolated grid to the target shape if necessary.
                if downscale > 1:
                    # Using order=0 (nearest neighbor) for upscaling is common to preserve original values.
                    idw_grid = zoom(idw_grid, downscale, order=0)
                
                # Store the result in the appropriate slice of the output array.
                if single_image:
                    result = idw_grid
                else:
                    result[t] = idw_grid
            
            return result
            
        else:
            raise ValueError(f"Invalid AWS method: {method}. Must be 'sparse' or 'dense'.")
    
    return transform

def default_rainrate_transform(threshold: float = 0.1, fill_value: float = 0.02, mean: float = 0.0, std: float = 1.0):
    """
    Creates a transformation function for default rain rate data.

    This transform applies several steps:
    1. Sets values below a `threshold` to a `fill_value`.
    2. Applies a `log10` transformation.
    3. Standardizes the data (Z-score normalization) using provided `mean` and `std`.

    Parameters:
        threshold : float, optional
            Values below this threshold will be replaced by `fill_value`. Defaults to 0.1.
        fill_value : float, optional
            The value to replace data below the `threshold` with. Defaults to 0.02.
        mean : float, optional
            The mean value for standardization. Defaults to 0.
        std : float, optional
            The standard deviation for standardization. Defaults to 1.

    Returns:
        Callable
            A function that takes a NumPy array as input and returns the transformed array.
    """
    def transform(data: np.ndarray) -> np.ndarray:
        """
        Applies the default rain rate transformation to the input data.

        Args:
            data : np.ndarray
                The input rain rate data (e.g., radar reflectivity or precipitation).

        Returns:
            np.ndarray
                The transformed data.
        """
        # Apply thresholding and fill value.
        data[data < threshold] = fill_value
        # Apply logarithmic transformation (base 10).
        data = np.log10(data)

        # Apply Z-score normalization.
        data -= mean
        data /= std

        return data
    return transform

def dbz_normalization_transform(convert_to_dbz: bool = True, undo: bool = False, norm_method: str = 'minmax'):
    """
    Creates a transformation function for rain rate data, specifically handling dBZ conversion
    and normalization. It provides both forward and inverse transformations.

    The dBZ conversion uses a specific Z-R relationship (Z = 200 * R^1.6), and the
    normalization supports min-max scaling or a tanh-like min-max scaling.

    Parameters:
        convert_to_dbz : bool, optional
            If `True`, converts input from rain rate (mm/hr) to dBZ during forward pass
            and back during inverse pass. If `False`, skips dBZ conversion and only
            applies normalization. Defaults to `True`.
        undo : bool, optional
            If `True`, undoes the processing. If convert_to_dbz is set to True, it undoes the conversion. 
            In addition, it undoes the selected norm_method. Defaults to `False`.
        norm_method : str, optional
            The normalization method to apply:
            - 'minmax': Standard min-max scaling to [0, 1] or [-1, 1].
            - 'minmax_tanh': A variation suitable for tanh activation, scaling to [-1, 1].
            - `None`: No normalization is applied (only dBZ conversion if `convert_to_dbz` is `True`).
            Defaults to 'minmax'.

    Returns:
        Callable
            A function `transform(x)` that applies the transformation.
            `x` can be a NumPy array or a PyTorch tensor.
    """
    # Precomputed constants for clipping and normalization ranges.
    # These are fixed parameters based on typical radar reflectivity ranges.
    MIN = 0.0 # Minimum value after dBZ conversion or for rain rate
    MAX = 55.0 if convert_to_dbz else 100.0 # Maximum value after dBZ conversion (55 dBZ) or for rain rate (100 mm/hr)
    HALF_RANGE = MAX / 2.0
    RANGE = MAX - MIN
    
    # Precompute normalization constants based on the chosen method.
    center = 0.0
    scale = 1.0
    if norm_method == 'minmax_tanh':
        # For tanh-like normalization, center around the mid-point and scale by half the range.
        center = MIN + HALF_RANGE
        scale = HALF_RANGE
    elif norm_method == 'minmax':
        # For standard min-max normalization, center at MIN and scale by the full range.
        center = MIN
        scale = RANGE
    
    def transform(x):
        """
        Applies or reverses the rain rate / dBZ transformation and normalization.
        
        Args:
            x : np.ndarray or torch.Tensor
                The input data to be transformed.
            undo : bool, optional
                If `True`, applies the inverse transformation. Defaults to `False`.
                
        Returns:
            np.ndarray or torch.Tensor
                The transformed or inverse-transformed data.
        """
        # Ensure data is float32 for calculations to maintain precision.
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
        elif isinstance(x, torch.Tensor):
            x = x.to(torch.float32)

        if not undo:
            # --- Forward transformation ---
            if convert_to_dbz:
                # Convert rain rate to dBZ (Z = 200 * R^1.6)
                # The formula is 10 * log10(200 * R^1.6 + 1) to handle R=0 gracefully.
                if isinstance(x, np.ndarray):
                    # Create a mask for non-zero values to avoid log(0).
                    mask = x != 0
                    x_copy = x.copy() # Work on a copy to avoid modifying original array in-place until assignment
                    x_copy[mask] = 10 * np.log10(200 * x[mask]**(1.6) + 1)
                    x = x_copy
                elif isinstance(x, torch.Tensor):
                    mask = x != 0
                    x_clone = x.clone()
                    x_clone[mask] = 10 * torch.log10(200 * x[mask]**(1.6) + 1)
                    x = x_clone
            
            # Apply clipping to defined MIN/MAX range.
            if isinstance(x, np.ndarray):
                x = np.clip(x, MIN, MAX)
            elif isinstance(x, torch.Tensor):
                x = torch.clamp(x, MIN, MAX)
            
            # Apply normalization if specified.
            if norm_method in ['minmax', 'minmax_tanh']:
                x = (x - center) / scale
                
            return x
            
        else:
            # --- Inverse transformation ---
            # Inverse normalization.
            if norm_method in ['minmax', 'minmax_tanh']:
                x = x * scale + center
                
            # Convert back from dBZ to rain rate.
            if convert_to_dbz:
                # Inverse of Z = 200 * R^1.6 + 1, so R = ((10^(Z/10) - 1) / 200) ^ (1/1.6)
                if isinstance(x, np.ndarray):
                    mask = x != 0
                    x_copy = x.copy()
                    # Ensure positive argument to log10 and power. Add a small epsilon if needed for robustness.
                    # The formula is simplified assuming x is already in dBZ scale.
                    # Reversing `10 * log10(200 * R**(1.6) + 1)`:
                    # x/10 = log10(200 * R**1.6 + 1)
                    # 10^(x/10) = 200 * R**1.6 + 1
                    # (10^(x/10) - 1) = 200 * R**1.6
                    # (10^(x/10) - 1) / 200 = R**1.6
                    # R = ((10^(x/10) - 1) / 200) ** (1/1.6)
                    x_copy[mask] = ((10**(x[mask]/10) - 1) / 200) ** (1/1.6)
                    x = x_copy
                elif isinstance(x, torch.Tensor):
                    mask = x != 0
                    x_clone = x.clone()
                    x_clone[mask] = ((10**(x[mask]/10) - 1) / 200) ** (1/1.6)
                    x = x_clone
                    
            return x
    
    return transform


def normalize_transform(mean: list | np.ndarray, std: list | np.ndarray, aws: bool = False):
    """
    Creates a Z-score normalization transform function for per-channel data.

    This transform normalizes data by subtracting the `mean` and dividing by the `std`
    for each channel. It's optimized for performance using NumPy broadcasting.

    Parameters:
        mean : list or np.ndarray
            A list or 1D NumPy array of mean values, one for each channel.
        std : list or np.ndarray
            A list or 1D NumPy array of standard deviation values, one for each channel.
        aws : bool, optional
            If `True`, assumes the input data is from AWS (Automatic Weather Stations)
            and is typically 2D (channels, stations) where normalization should be
            applied per channel across stations. If `False`, assumes 3D data
            (channels, height, width) and normalizes per channel across spatial dimensions.
            Defaults to `False`.

    Returns:
        Callable
            A function that takes a NumPy array (or a tuple for AWS data)
            and returns its normalized version.
    """
    # Convert mean and std to NumPy arrays and reshape for broadcasting.
    # Reshaping ensures that mean/std align correctly with the channel dimension (C, H, W) or (C, N_STATIONS).
    if aws:
        # For AWS, data is often (C, num_stations), so reshape to (C, 1) to broadcast across stations.
        mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(-1, 1) 
    else:
        # For image-like data (C, H, W), reshape to (C, 1, 1) to broadcast across H and W.
        mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(-1, 1, 1)

    def transform(data: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Applies Z-score normalization to the input data.

        Args:
            data : np.ndarray or tuple[np.ndarray, np.ndarray]
                The input data. If `aws` is True, it can be a tuple `(data_values, locations)`.
                Otherwise, it's expected to be a NumPy array.

        Returns:
            np.ndarray or tuple[np.ndarray, np.ndarray]
                The normalized data. If input was a tuple, output is also a tuple.
        """
        if isinstance(data, tuple):
            # If it's AWS data, normalize the values part and return with locations.
            data_values, locations = data
            return ((data_values - mean) / std, locations)
        
        # Standard case: normalize the array directly.
        return (data - mean) / std 

    return transform

def resize_transform(target_shape: tuple[int, int] = None, scale: float = None, method: str = 'nearest'):
    """
    Creates an OpenCV-based image resizing transform function.

    This transform resizes 2D (H, W) or 3D (C, H, W) NumPy arrays to a specified
    `target_shape` or by a `scale` factor. It wraps OpenCV's `cv2.resize` function
    and handles channel dimension transposition for compatibility.

    Parameters:
        target_shape : tuple[int, int], optional
            The desired output (height, width) of the resized image.
            Provide either `target_shape` or `scale`, but not both.
        scale : float, optional
            A floating-point factor by which to scale the height and width.
            E.g., 0.5 for halving, 2.0 for doubling.
            Provide either `target_shape` or `scale`, but not both.
        method : str, optional
            The interpolation method to use during resizing:
            - 'nearest': cv2.INTER_NEAREST (Fastest, blocky for upscaling)
            - 'bilinear': cv2.INTER_LINEAR (Good balance of speed and quality)
            - 'area': cv2.INTER_AREA (Recommended for downsampling)
            - 'lanczos': cv2.INTER_LANCZOS4 (High quality, slowest)
            Defaults to 'nearest'.

    Returns:
        Callable
            A function that takes a NumPy array (C, H, W) and returns the resized array.

    Raises:
        ValueError: If both `target_shape` and `scale` are provided, or neither.
                    If an unsupported `method` is provided.
    """
    import cv2

    if (target_shape is None) == (scale is None):
        raise ValueError("Provide either `target_shape` or `scale`, but not both.")

    # Map friendly method names to OpenCV interpolation constants.
    method_map = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    interp = method_map.get(method)
    if interp is None:
        raise ValueError(f"Unsupported interpolation method: '{method}'. "
                         f"Choose from {list(method_map.keys())}.")

    def transform(data: np.ndarray) -> np.ndarray:
        # Ensure data is (C, H, W)
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # -> (1, H, W)

        C, H, W = data.shape
        arr = np.transpose(data, (1, 2, 0))  # -> (H, W, C)
        orig_dtype = arr.dtype
        if arr.dtype == np.float16:
            arr = arr.astype(np.float32)

        # Determine new size
        if target_shape is not None:
            newH, newW = target_shape
        else:
            newH = int(H * scale)
            newW = int(W * scale)

        # Perform resize; might drop the last dim if C == 1
        resized = cv2.resize(arr, (newW, newH), interpolation=interp)

        # If OpenCV returned 2D, reintroduce channel axis
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]  # -> (newH, newW, 1)

        # Now resized is always (H', W', C)
        resized = np.transpose(resized, (2, 0, 1))  # -> (C, H', W')

        # Back to original dtype if needed
        if orig_dtype == np.float16:
            resized = resized.astype(np.float16)

        return resized

    return transform

def dropout_transform(p: float = 0.5):
    """
    Creates a dropout transformation function.

    This transform randomly (with probability `p`) replaces the entire input array
    with an array of zeros of the same shape and data type. It reuses a zero buffer
    for performance, avoiding repeated memory allocations.

    Parameters:
        p : float, optional
            The probability (between 0.0 and 1.0) of dropping out the input data
            (i.e., replacing it with zeros). Defaults to 0.5.

    Returns:
        Callable
            A function that takes a NumPy array as input and returns either the
            original array or an array of zeros, depending on the dropout probability.
    """
    # Variables to store the zero buffer and its properties for reuse.
    zero_buf = None
    last_shape = None
    last_dtype = None
    
    def transform(data: np.ndarray) -> np.ndarray:
        """
        Applies dropout to the input data.

        Args:
            data : np.ndarray
                The input NumPy array.

        Returns:
            np.ndarray
                Either the original `data` array or a zero-filled array of the
                same shape and type, based on the dropout probability `p`.
        """
        nonlocal zero_buf, last_shape, last_dtype
        
        # Randomly decide whether to apply dropout.
        if np.random.random() < p:
            # Check if the zero buffer needs to be reallocated (e.g., if input shape or dtype changes).
            if (zero_buf is None or data.shape != last_shape or 
                data.dtype != last_dtype):
                # Allocate a new zero buffer matching the current input data's shape and dtype.
                zero_buf = np.zeros_like(data)
                last_shape = data.shape
                last_dtype = data.dtype
            return zero_buf
        return data
    
    return transform
    

if __name__ == "__main__": 
    import time

    # Set input shape for synthetic data (e.g., 22 time steps, 256x256 spatial resolution).
    shape = (22, 256, 256)
    print("=" * 60)
    print(f"Benchmarking transformations with input shape: {shape}")
    print("=" * 60)

    # Generate synthetic data for benchmarking. Using float16 to simulate common
    # data types for meteorological data, which also tests type handling in transforms.
    raw_data = np.random.gamma(shape=2.0, scale=0.5, size=shape).astype(np.float16)

    # --- AWS specific setup ---
    # Define target grid for AWS interpolation
    aws_target_height, aws_target_width = 768, 704
    num_aws_stations = 50 # Number of synthetic AWS stations
    
    # Generate realistic-ish AWS station locations within the specified grid
    # x-coordinates between 0 and 704, y-coordinates between 0 and 768
    aws_locations = np.column_stack([
        np.random.randint(0, aws_target_width, num_aws_stations),
        np.random.randint(0, aws_target_height, num_aws_stations)
    ]).astype(np.float32)

    # Generate AWS values for each time step (22 time steps matching raw_data's first dim)
    # The values can be anything, here we use random floats.
    aws_values = np.random.uniform(0, 100, size=(shape[0], num_aws_stations)).astype(np.float32)

    # Combine into the tuple format expected by aws_transform
    aws_raw_data = (aws_values, aws_locations)
    # --- End AWS specific setup ---

    # Prepare different transform pipelines.
    # Each entry in 'methods' defines a named transformation.
    methods = {
        'Default Rainrate Transform': get_transforms('default_rainrate', {}),
        'dBZ Normalization Transform': get_transforms('dbz_normalization', {"convert_to_dbz": True}),
        'Simple Normalize Transform': get_transforms('normalize', {"mean": [0.5], "std": [1.0]}),
        'Resize 128x128 Nearest': get_transforms('resize', {"target_shape": (128, 128), "method": "nearest"}),
        'Dropout (p=0.5)': get_transforms('dropout', {"p": 0.5}),
        'AWS Interpolation Dense': get_transforms('aws_interpolation', {'target_shape': (aws_target_height, aws_target_width), 'method': 'dense', 'downscale': 4}), # Added aws_interpolation
        'AWS Interpolation Sparse': get_transforms('aws_interpolation', {'target_shape': (aws_target_height, aws_target_width), 'method': 'sparse', 'radius': 5}), # Added aws_interpolation
    }

    # Perform a warm-up run for each transform.
    # This ensures Numba functions are compiled and any initial setup is done,
    # leading to more accurate benchmark times.
    print("\nPerforming warm-up runs...")
    for name, transform in methods.items():
        if transform: # Only warm up if transform was successfully created
            try:
                # Provide appropriate dummy data for each transform type during warm-up
                if name in ['Default Rainrate Transform', 'dBZ Normalization Transform', 'Simple Normalize Transform', 'Resize 128x128 Nearest', 'Dropout (p=0.5)']:
                    _ = transform(raw_data)
                elif 'AWS Interpolation' in name:
                    _ = transform(aws_raw_data) # Use the specially prepared AWS data
                elif 'Regrid Transform' in name:
                    # Regrid needs source path to initialize, cannot easily dummy it here without a real Zarr
                    print(f"Skipping warm-up for '{name}' (requires valid Zarr source).")
                    continue
                print(f"- Warm-up complete for: {name}")
            except Exception as e:
                print(f"- Warm-up failed for {name}: {e}")
        else:
            print(f"- Transform '{name}' not created, skipping warm-up.")


    # Benchmark each method.
    print("\nStarting benchmark (100 runs per transform)...")
    results = {}
    for name, transform in methods.items():
        if not transform:
            print(f"Skipping benchmark for '{name}' as transform was not created.")
            continue
        
        times = []
        # Run multiple times to get a stable measurement, then take the median.
        for _ in range(100): 
            start = time.perf_counter()
            try:
                if name in ['Default Rainrate Transform', 'dBZ Normalization Transform', 'Simple Normalize Transform', 'Resize 128x128 Nearest', 'Dropout (p=0.5)']:
                    _ = transform(raw_data)
                elif 'AWS Interpolation' in name:
                    _ = transform(aws_raw_data) # Use the specially prepared AWS data
                elif 'Regrid Transform' in name:
                    # Cannot reliably benchmark regrid without a real Zarr source.
                    print(f"Skipping benchmark for '{name}' (requires valid Zarr source for runtime).")
                    break # Exit inner loop for this transform
            except Exception as e:
                print(f"Error during benchmark for {name}: {e}")
                break # Exit inner loop if an error occurs
            times.append(time.perf_counter() - start)
        
        if times: # Only calculate if there were successful runs
            median_time = np.median(times) * 1000  # Convert to milliseconds
            results[name] = median_time
            print(f"{name:<30} | Median time: {median_time:.3f} ms")

    # Print a performance comparison, sorted by speed.
    print("\n" + "=" * 40)
    print("Performance Comparison (Sorted by Speed):")
    print("-" * 40)
    if results:
        for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
            print(f"{name:<30} | {time_ms:.3f} ms")
    else:
        print("No benchmark results to display.")
    print("=" * 40)