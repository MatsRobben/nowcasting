import torch
from torch.amp import autocast

import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Union

def csi_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: Union[float, List[float]] = 0.5,
    time_idx: Union[int, List[int], None] = None,
    reduce: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Compute Critical Success Index (CSI) for precipitation nowcasting.
    
    Args:
        pred: Model predictions (B, C, T, H, W)
        target: Ground truth (same shape as pred)
        threshold: Threshold value(s) in mm/h. Can be:
                  - Single float (e.g., 0.5)
                  - List of floats (e.g., [0.1, 1.0, 5.0])
        time_idx: Time index/indices to evaluate. Can be:
                  - None (all timesteps)
                  - Single int (e.g., 12 for 60min)
                  - List of ints (e.g., [6, 12, 18])
        reduce: How to aggregate results:
               - 'mean' : average across thresholds/timesteps
               - 'none' : return raw results (T_thresholds × T_times)
    
    Returns:
        CSI score(s). Shape depends on `reduce`:
        - float if reduce='mean'
        - tensor (n_thresholds, n_times) if reduce='none'
    """
    # Convert inputs to lists for uniform processing
    thresholds = [threshold] if isinstance(threshold, float) else threshold
    time_indices = (
        [time_idx] if isinstance(time_idx, int) 
        else list(range(pred.shape[2])) if time_idx is None 
        else time_idx
    )
    
    # Initialize results tensor
    results = torch.zeros(len(thresholds), len(time_indices))
    
    # Compute CSI for each threshold/time combination
    for i, thresh in enumerate(thresholds):
        for j, t in enumerate(time_indices):
            # Slice the required timestep
            pred_t = pred[:, :, t, :, :]
            target_t = target[:, :, t, :, :]
            
            # Binarize
            pred_bin = (pred_t > thresh).float()
            target_bin = (target_t > thresh).float()
            
            # Confusion matrix
            tp = (pred_bin * target_bin).sum()
            fp = (pred_bin * (1 - target_bin)).sum()
            fn = ((1 - pred_bin) * target_bin).sum()
            
            # CSI calculation
            eps = 1e-8
            results[i, j] = tp / (tp + fp + fn + eps)
    
    return results.mean() if reduce == 'mean' else results



def to_mmh(img, norm_method='zscore', mean=0.03019706713265408, std=0.5392297631902654):
    """
    Convert normalized precipitation data to mm/h.

    Parameters:
        img (np.ndarray or torch.Tensor): Normalized input data.
        norm_method (str): 'zscore' or 'minmax'.
        mean (float): Mean for z-score normalization.
        std (float): Std for z-score normalization.

    Returns:
        img_mmh: Precipitation in mm/h (clipped to [0, 160]).
    """
    if norm_method == 'zscore':
        img = img * std + mean
        img_mmh = 10 ** img

    elif norm_method == 'minmax':
        # Undo minmax normalization to [0, 55] dBZ
        img = img * 55

        # Convert from dBZ to mm/h (in-place)
        if isinstance(img, np.ndarray):
            mask = img != 0
            img[mask] = ((10**(img[mask] / 10) - 1) / 200) ** (5 / 8)
        elif isinstance(img, torch.Tensor):
            mask = img != 0
            img[mask] = ((10**(img[mask] / 10) - 1) / 200) ** (5 / 8)
        else:
            raise TypeError("Input must be a numpy array or a torch tensor")

        img_mmh = img

    else:
        raise ValueError("norm_method must be 'zscore' or 'minmax'")

    # Clip final mm/h values to physical range
    if isinstance(img_mmh, torch.Tensor):
        img_mmh = torch.clamp(img_mmh, 0.0, 160.0)
    elif isinstance(img_mmh, np.ndarray):
        img_mmh = np.clip(img_mmh, 0.0, 160.0)

    return img_mmh


class PermutationImportance:
    def __init__(
        self,
        model: torch.nn.Module,
        metric: Callable[[torch.Tensor, torch.Tensor], float],
        device: str = "cuda",
        use_amp: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            model: Trained PyTorch model
            metric: Evaluation metric function (y_pred, y_true) -> float (higher = better)
            device: Device for computation
            verbose: Show progress bar
        """
        self.model = model.to(device).eval()
        self.metric = metric
        self.device = device
        self.verbose = verbose

        self.use_amp = use_amp
        if self.use_amp and self.device == "cpu":
            print("Warning: Mixed precision (AMP) is only available on CUDA devices.")
            self.use_amp = False

    def _permute_tensor(
        self, 
        tensor: torch.Tensor, 
        channel_indices: List[int]
    ) -> torch.Tensor:
        """Permute selected channels across spatial dimensions"""
        # print(channel_indices)

        tensor_perm = tensor.clone()
        for idx in channel_indices:
            # Get all timesteps for this channel
            # tensor shape: (B, T, H, W, C)
            c_data = tensor_perm[:, :, :, :, idx]
            c_shape = c_data.shape
            
            # Flatten spatial dimensions for each sample and timestep
            flat_data = c_data.reshape(c_shape[0], c_shape[1], -1)  # (B, T, H*W)
            
            # Permute each sample and timestep independently
            for b in range(flat_data.shape[0]):
                for t in range(flat_data.shape[1]):
                    perm_idx = torch.randperm(flat_data.shape[2])
                    flat_data[b, t] = flat_data[b, t, perm_idx]
            
            # Reshape and insert back
            tensor_perm[:, :, :, :, idx] = flat_data.reshape(c_shape)
        return tensor_perm

    def multi_pass(
        self,
        dataloader: torch.utils.data.DataLoader,
        groups: List[List[int]],
        max_batches: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Args:
            dataloader: Test dataloader yielding (input_tensor, target_tensor)
            groups: List of channel index groups to permute [[0], [1], [2,3], ...]
            max_batches: Limit batches for efficiency
        
        Returns:
            ordered_groups: Group names in permutation order
            metrics: Metric values after each permutation
        """
        # Baseline metric (no permutation)
        baseline_metric = self._evaluate(dataloader, [], max_batches)
        metrics = [baseline_metric]
        ordered_groups = []
        unpermuted = groups.copy()
        
        # Multi-pass algorithm
        for _ in tqdm(range(len(groups)), disable=not self.verbose):
            candidate_metrics = []

            print(metrics)
            print(ordered_groups)
            print(unpermuted)
            for candidate in unpermuted:
                candidate_perm = ordered_groups + [candidate]
                metric_val = self._evaluate(dataloader, candidate_perm, max_batches)
                candidate_metrics.append(metric_val)
            
            # Find candidate causing largest performance drop
            worst_idx = np.argmin(candidate_metrics)
            worst_group = unpermuted[worst_idx]
            
            # Update lists
            ordered_groups.append(worst_group)
            unpermuted.remove(worst_group)
            metrics.append(candidate_metrics[worst_idx])
        
        # Convert groups to readable names
        ordered_group_names = [f"Channels_{g}" for g in ordered_groups]
        return ordered_group_names, metrics

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        perm_groups: List[List[int]],
        max_batches: int = None
    ) -> float:
        """Evaluate model with given permutation groups applied"""
        total_metric, total_samples = 0, 0
        batch_iter = dataloader if not self.verbose else tqdm(dataloader)

        # Determine the data type for autocast on CPU
        amp_dtype = torch.bfloat16 if self.device == 'cpu' and self.use_amp else torch.float16
        
        for i, (inputs, targets) in enumerate(batch_iter):
            if max_batches and i >= max_batches:
                break

            # Shape (B, C, T, W, H) -> Earthformer Shape (B, T, W, H, C)
            inputs = inputs.permute(0, 2, 3, 4, 1).to(self.device)
            targets = targets.permute(0, 2, 3, 4, 1).to(self.device)
                
            # Apply permutations
            if perm_groups:
                inputs = self._permute_tensor(inputs, 
                    channel_indices=[idx for group in perm_groups for idx in group]
                )
            
            # Forward pass
            with torch.no_grad(), autocast(device_type=self.device, dtype=amp_dtype, enabled=self.use_amp):
                preds = self.model(inputs)

                preds = to_mmh(preds)
                targets = to_mmh(targets)

                metric_val = self.metric(preds, targets)
            
            # Accumulate
            total_metric += metric_val * targets.shape[0]
            total_samples += targets.shape[0]
        
        return total_metric / total_samples


if __name__ == "__main__":
    from nowcasting.models.earthformer.earthformer import CuboidSEVIRPLModule
    from omegaconf import OmegaConf
    import matplotlib.pyplot as plt

    config_dict = {
        "model": {
            "input_shape": [4, 256, 256, 10],
            "target_shape": [18, 256, 256, 1],
            "base_units": 128,
            "block_units": None,
            "scale_alpha": 1.0,

            "enc_depth": [1, 1],
            "dec_depth": [1, 1],
            "enc_use_inter_ffn": True,
            "dec_use_inter_ffn": True,
            "dec_hierarchical_pos_embed": False,

            "downsample": 2,
            "downsample_type": "patch_merge",
            "upsample_type": "upsample",

            "num_global_vectors": 8,
            "use_dec_self_global": False,
            "dec_self_update_global": True,
            "use_dec_cross_global": False,
            "use_global_vector_ffn": False,
            "use_global_self_attn": True,
            "separate_global_qkv": True,
            "global_dim_ratio": 1,

            "enc_attn_patterns": "axial",
            "dec_self_attn_patterns": "axial",
            "dec_cross_attn_patterns": "cross_1x1",
            "dec_cross_last_n_frames": None,

            "attn_drop": 0.1,
            "proj_drop": 0.1,
            "ffn_drop": 0.1,
            "num_heads": 4,

            "ffn_activation": "gelu",
            "gated_ffn": False,
            "norm_layer": "layer_norm",
            "padding_type": "zeros",
            "pos_embed_type": "t+h+w",
            "use_relative_pos": True,
            "self_attn_use_final_proj": True,
            "dec_use_first_self_attn": False,

            "z_init_method": "zeros",
            "checkpoint_level": 0,

            "initial_downsample_type": "stack_conv",
            "initial_downsample_activation": "leaky",
            "initial_downsample_stack_conv_num_layers": 3,
            "initial_downsample_stack_conv_dim_list": [16, 64, 128],
            "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
            "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],

            "attn_linear_init_mode": "0",
            "ffn_linear_init_mode": "0",
            "conv_init_mode": "0",
            "down_up_linear_init_mode": "0",
            "norm_init_mode": "0"
        },
        "loss": {
            "loss_name": "balanced",
            "normalization": "zscore",
            "mean_log_rain": 0.03019706713265408,
            "std_log_rain": 0.5392297631902654,
            "max_weight_r": 30,
            "weight_intensity": 1.0,
            "extended": True
        }
    }

    config = OmegaConf.create(config_dict)
    checkpoint_path = "/projects/prjs1634/nowcasting/results/tb_logs/earthformer/version_12/checkpoints/epoch=58-step=59000.ckpt"
    module = CuboidSEVIRPLModule.load_from_checkpoint(checkpoint_path, cfg=config)
    model = module.torch_nn_module

    from nowcasting.data.dataloader_zarrv3 import NowcastingDataModule

    path = '/projects/prjs1634/nowcasting/data/dataset.zarr'

    var_info = {
        "sample_var": "radar/max_intensity_grid",
        "latlon": False,
        "in_vars": [
            "radar/rtcor",
            "sat_l1p5/IR_016",
            "sat_l1p5/IR_039",
            "sat_l1p5/WV_062",
            "sat_l1p5/WV_073",
            "sat_l1p5/IR_087",
            "sat_l1p5/IR_097",
            "sat_l1p5/IR_108",
            "sat_l1p5/IR_120",
            "sat_l1p5/IR_134"
        ],
        "out_vars": [
            "radar/rtcor"
            "sat_l1p5/IR_016",
            "sat_l1p5/IR_039",
            "sat_l1p5/WV_062",
            "sat_l1p5/WV_073",
            "sat_l1p5/IR_087",
            "sat_l1p5/IR_097",
            "sat_l1p5/IR_108",
            "sat_l1p5/IR_120",
            "sat_l1p5/IR_134"
        ],
        "transforms": {
            "radar": {
                "default_rainrate": {
                    "mean": 0.03019706713265408,
                    "std": 0.5392297631902654
                }
            },
            "sat_l1p5/WV_062": {
                "normalize": {
                    "mean": 230.6474,
                    "std": 4.9752
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_108": {
                "normalize": {
                    "mean": 266.7156,
                    "std": 16.5736
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_016": {
                "normalize": {
                    "mean": 6.8788,
                    "std": 10.0196
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_039": {
                "normalize": {
                    "mean": 269.7683,
                    "std": 14.6116
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/WV_073": {
                "normalize": {
                    "mean": 246.0418,
                    "std": 8.5848
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_087": {
                "normalize": {
                    "mean": 264.6952,
                    "std": 15.4881
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_097": {
                "normalize": {
                    "mean": 241.9831,
                    "std": 9.0873
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_120": {
                "normalize": {
                    "mean": 265.4512,
                    "std": 16.6885
                },
                "resize": {
                    "scale": 2
                }
            },
            "sat_l1p5/IR_134": {
                "normalize": {
                    "mean": 247.9466,
                    "std": 10.4477
                },
                "resize": {
                    "scale": 2
                }
            }
        }
    }

    split_info = {
        'split_rules': {
            "test": {"year": [2023]}, 
            "val":  {"month": [6, 11]},
            "train": {} 
        },
        'apply_missing_masks': ['radar', 'harmonie', 'sat_l2', 'sat_l1p5', 'aws'],
        'clutter_threshold': 50,
    }   


    sample_info = {
        'threshold': 0.01, 
        'methods': {
            'train': {'agg': 'mean_pool'},
            'val': {'agg': 'mean_pool', 'center_crop': True},
            'test': {'agg': 'mean_pool'}
        }
    }

    # --- Instantiate and Setup DataModule ---
    print("Initializing NowcastingDataModule...")
    data_module = NowcastingDataModule(
        path=path,
        var_info=var_info,
        sample_info=sample_info,
        split_info=split_info,
        context_len=4,       # 4 time steps for input
        forecast_len=18,     # 18 time steps for output
        include_timestamps=False, # Include relative timestamps in context
        img_size=(8,8),      # Spatial patch size: 8 blocks x 8 blocks
        stride=(12,1,1),      # Sample generation stride (t, h, w)
        batch_size=32,        # Batch size for DataLoaders
        num_workers=16,       # Number of CPU workers for data loading
    )
    print("Setting up DataModule (loading metadata, generating samples, applying transforms)...")
    data_module.setup() # This is a crucial step that prepares the datasets
    dataloader = data_module.val_dataloader()

    pi = PermutationImportance(model, csi_metric, device="cuda")

    groups = [
        [0],       # Radar only
        [1],       # IR1 only
        [2],       # IR2 only 
        [3],       # IR3 only
        [4],
        # [5],
        # [6],
        # [7],
        # [8],
        # [9],
        # [1, 2, 3], # All IR
        # [0, 1, 2, 3] # All channels
    ]

    # Run multi-pass permutation
    group_names, metrics = pi.multi_pass(
        dataloader, 
        groups,
        max_batches=None  # Use subset for speed
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(metrics, 'o-', markersize=8, linewidth=2)
    plt.xticks(
        range(len(group_names) + 1),
        ["Baseline"] + [f"Permute {g}" for g in group_names],
        rotation=45,
        ha='right'
    )
    plt.ylabel("CSI Score", fontsize=12)
    plt.title("Multi-pass Permutation Importance", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure
    plot_path = '/projects/prjs1634/nowcasting/results/figures/permutation_importance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to: {plot_path}")