import torch
from torch.amp import autocast

import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Callable, Union, Any, Optional
import collections

def center_crop(tensor: Union[np.ndarray, torch.Tensor], size: Tuple[int, int]) -> Union[np.ndarray, torch.Tensor]:
    """Center-crop tensor to specified size."""
    h, w = tensor.shape[-2:]
    ch, cw = h // 2, w // 2
    h_start, w_start = ch - size[0] // 2, cw - size[1] // 2
    return tensor[..., h_start:h_start + size[0], w_start:w_start + size[1]]

def csi_metric(
    threshold: Union[float, List[float]] = 5,
    time_idx: Union[int, List[int], None] = None,
    reduce: str = 'mean',
    crop_center: Tuple[int, int] = None
) -> Callable[[torch.Tensor, torch.Tensor], float]:
    """
    Returns a CSI metric function with preset parameters.
    """
    def _csi_metric(pred: torch.Tensor, target: torch.Tensor) -> Union[float, torch.Tensor]:
        thresholds = torch.tensor(threshold, device=pred.device) if not isinstance(threshold, torch.Tensor) else threshold
        if crop_center:
            pred = center_crop(pred, crop_center)
            target = center_crop(target, crop_center)

        if thresholds.dim() == 0:
            thresholds = thresholds.unsqueeze(0)
        
        # Handle time indices
        if time_idx is None:
            time_indices = torch.arange(pred.shape[2], device=pred.device)
        elif isinstance(time_idx, int):
            time_indices = torch.tensor([time_idx], device=pred.device)
        else:
            time_indices = torch.tensor(time_idx, device=pred.device)
        
        # Select required timesteps
        pred_sel = pred.index_select(2, time_indices)  # (B, C, T_sel, H, W)
        target_sel = target.index_select(2, time_indices)
        
        # Reshape for broadcasting
        pred_sel = pred_sel.unsqueeze(-1)  # (B, C, T_sel, H, W, 1)
        target_sel = target_sel.unsqueeze(-1)
        thresholds_reshaped = thresholds.view(1, 1, 1, 1, 1, -1)
        
        # Binarize with broadcasting
        pred_bin = (pred_sel > thresholds_reshaped).float()
        target_bin = (target_sel > thresholds_reshaped).float()
        
        # Compute confusion matrix
        tp = (pred_bin * target_bin).sum(dim=(0, 1, 3, 4))  
        fp = (pred_bin * (1 - target_bin)).sum(dim=(0, 1, 3, 4))
        fn = ((1 - pred_bin) * target_bin).sum(dim=(0, 1, 3, 4))
        
        eps = 1e-8
        csi = tp / (tp + fp + fn + eps)  # (T_sel, K)
        
        if reduce == 'mean':
            return csi.mean().item()
        else:
            return csi.permute(1, 0)  # (K, T_sel)
    
    return _csi_metric

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
        verbose: bool = True,
        precompute_indices: bool = True,
        model_type: str = "earthformer",  # 'earthformer' or 'ldcast'
    ):
        self.model = model.to(device).eval()
        self.metric = metric
        self.device = device
        self.verbose = verbose
        self.precompute_indices = precompute_indices
        self.perm_cache = {}
        self.model_type = model_type.lower()
        self.use_amp = use_amp
        
        if self.use_amp and self.device == "cpu":
            print("Warning: Mixed precision (AMP) is only available on CUDA devices.")
            self.use_amp = False
            
        # Validate model type
        if self.model_type not in ["earthformer", "ldcast"]:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be 'earthformer' or 'ldcast'")
        
        print(self.model_type)

    def permute_tensor(
        self, 
        tensor: torch.Tensor, 
        channel_indices: List[int]
    ) -> torch.Tensor:
        """Permute channels in a tensor (B, C, T, H, W)"""
        if not channel_indices:
            return tensor
            
        B, C, T, H, W = tensor.shape
        C_perm = len(channel_indices)
        key = (B, tuple(channel_indices), T, H, W)
        
        # Generate or reuse permutation indices
        if self.precompute_indices and key in self.perm_cache:
            perm_idx = self.perm_cache[key]
        else:
            perm_idx = torch.rand(B, C_perm, T, H * W, device=tensor.device).argsort(dim=-1)
            if self.precompute_indices:
                self.perm_cache[key] = perm_idx
        
        x = tensor[:, channel_indices, ...]
        x_flat = x.reshape(B, C_perm, T, H * W)
        x_perm_flat = torch.gather(x_flat, dim=-1, index=perm_idx)
        tensor[:, channel_indices, ...] = x_perm_flat.reshape(B, C_perm, T, H, W)
        return tensor

    def _preprocess_earthformer(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[Any, Any]:
        """Earthformer: Convert to (B, T, H, W, C) format"""
        inputs, targets = batch
        inputs = inputs.permute(0, 2, 3, 4, 1).to(self.device)
        targets = targets.permute(0, 2, 3, 4, 1).to(self.device)
        return inputs, targets

    def _postprocess_earthformer(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Earthformer: Convert back to (B, C, T, H, W) and to mm/h"""
        pred = pred.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
        pred = to_mmh(pred)
        target = to_mmh(target)
        return pred, target

    def _preprocess_ldcast(self, batch: Tuple[list, torch.Tensor]) -> Tuple[Any, Any]:
        """Ldcast: Move all components to device"""
        input_list, target = batch

        # Process each modality in the input list
        processed_list = []
        for modality in input_list:
            # Each modality is a tuple (tensor, timestamps)
            tensor, timestamps = modality
            tensor = tensor.to(self.device)
            timestamps = timestamps.to(self.device)
            processed_list.append((tensor, timestamps))
            
        target = target.to(self.device)
        return processed_list, target

    def _postprocess_ldcast(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ldcast: Convert to mm/h"""
        pred = to_mmh(pred)
        target = to_mmh(target)
        return pred, target

    def _permute_batch(self, batch: Any, perm_groups: List[Any]) -> Any:
        """Apply permutations based on model type"""
        if self.model_type == "earthformer":
            inputs, targets = batch
            flat_channels = [idx for group in perm_groups for idx in group]
            inputs = self.permute_tensor(inputs, flat_channels)
            return inputs, targets
            
        elif self.model_type == "ldcast":
            input_list, target = batch
            # Create a mutable copy of the input list
            permuted_list = []
            
            # Apply permutations to each modality
            for i, modality in enumerate(input_list):
                tensor, timestamps = modality
                
                # Check if this modality has any groups to permute
                mod_groups = []
                for group in perm_groups:
                    mod_idx, channels = group
                    if mod_idx == i:
                        mod_groups.append(channels)
                
                # Flatten all channel indices for this modality
                flat_channels = [idx for channels in mod_groups for idx in channels]
                
                # Apply permutation if needed
                if flat_channels:
                    tensor = self.permute_tensor(tensor, flat_channels)
                
                permuted_list.append((tensor, timestamps))
            
            return (permuted_list, target)

    def multi_pass(
        self,
        dataloader: torch.utils.data.DataLoader,
        groups: List[Any],
        max_batches: int = None,
        group_to_name: Callable[[Any], str] = None
    ) -> Tuple[List[str], List[float]]:
        group_to_name = group_to_name or (lambda x: f"Group_{x}")
        baseline_metric = self._evaluate(dataloader, [], max_batches)
        metrics = [baseline_metric]
        ordered_groups = []
        unpermuted = groups.copy()
        
        for _ in tqdm(range(len(groups)), disable=not self.verbose):
            candidate_metrics = []

            print(metrics)
            print(ordered_groups)
            print(unpermuted)
            
            for candidate in unpermuted:
                candidate_perm = ordered_groups + [candidate]
                metric_val = self._evaluate(dataloader, candidate_perm, max_batches)
                candidate_metrics.append(metric_val)
            
            worst_idx = int(np.argmin(candidate_metrics))
            worst_group = unpermuted[worst_idx]
            ordered_groups.append(worst_group)
            unpermuted.remove(worst_group)
            metrics.append(candidate_metrics[worst_idx])
        
        ordered_group_names = [group_to_name(g) for g in ordered_groups]
        return ordered_group_names, metrics

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        perm_groups: List[Any],
        max_batches: int = None
    ) -> float:
        total_metric, total_samples = 0, 0
        batch_iter = dataloader if not self.verbose else tqdm(dataloader)
        amp_dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16
        
        for i, batch in enumerate(batch_iter):
            if max_batches and i >= max_batches:
                break
                
            # Apply permutations
            batch_perm = self._permute_batch(batch, perm_groups)
            
            # Preprocess
            if self.model_type == "earthformer":
                model_input, target = self._preprocess_earthformer(batch_perm)
            elif self.model_type == "ldcast":
                model_input, target = self._preprocess_ldcast(batch_perm)
            
            # Forward pass
            with torch.inference_mode(), autocast(
                device_type=self.device,
                dtype=amp_dtype,
                enabled=self.use_amp
            ):
                # For ldcast, model_input is a list of (tensor, timestamps) tuples
                pred = self.model(model_input)
                
                # Postprocess
                if self.model_type == "earthformer":
                    pred_metric, target_metric = self._postprocess_earthformer(pred, target)
                elif self.model_type == "ldcast":
                    pred_metric, target_metric = self._postprocess_ldcast(pred, target)
                
                metric_val = self.metric(pred_metric, target_metric)
            
            total_metric += metric_val
            total_samples += 1  # Assuaming metric is averaged per batch
        
        return total_metric / total_samples if total_samples > 0 else 0.0


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