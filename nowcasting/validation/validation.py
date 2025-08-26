import numpy as np
import torch
from typing import List, Dict, Any, Optional, Union, Tuple
import h5py
import json
import os
from pathlib import Path
from datetime import datetime
import warnings


def to_numpy(tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    return tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor


def center_crop(tensor: Union[np.ndarray, torch.Tensor], size: Tuple[int, int]) -> Union[np.ndarray, torch.Tensor]:
    """Center-crop tensor to specified size."""
    h, w = tensor.shape[-2:]
    ch, cw = h // 2, w // 2
    h_start, w_start = ch - size[0] // 2, cw - size[1] // 2
    return tensor[..., h_start:h_start + size[0], w_start:w_start + size[1]]


class MetricsComputer:
    """Efficient vectorized computation of precipitation nowcasting metrics."""
    
    def __init__(self, thresholds: List[float], scales: List[int]):
        self.thresholds = np.array(thresholds)
        self.scales = scales
    
    def compute_categorical_metrics(self, obs: Union[np.ndarray, torch.Tensor], 
                                  pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Compute categorical metrics for all thresholds vectorized."""
        # Expand for broadcasting: [..., H, W, 1] and [n_thresholds]
        obs_exp = obs.unsqueeze(-1) if isinstance(obs, torch.Tensor) else np.expand_dims(obs, -1)
        pred_exp = pred.unsqueeze(-1) if isinstance(pred, torch.Tensor) else np.expand_dims(pred, -1)
        
        if isinstance(obs, torch.Tensor):
            thresholds_t = torch.tensor(self.thresholds, device=obs.device, dtype=obs.dtype)
            obs_binary = obs_exp >= thresholds_t
            pred_binary = pred_exp >= thresholds_t
            
            hits = torch.sum((obs_binary & pred_binary).float(), dim=(-3, -2))
            misses = torch.sum((obs_binary & ~pred_binary).float(), dim=(-3, -2))
            false_alarms = torch.sum((~obs_binary & pred_binary).float(), dim=(-3, -2))
        else:
            obs_binary = obs_exp >= self.thresholds
            pred_binary = pred_exp >= self.thresholds
            
            hits = np.sum((obs_binary & pred_binary), axis=(-3, -2))
            misses = np.sum((obs_binary & ~pred_binary), axis=(-3, -2))
            false_alarms = np.sum((~obs_binary & pred_binary), axis=(-3, -2))
        
        eps = 1e-10
        pod = hits / (hits + misses + eps)
        far = false_alarms / (hits + false_alarms + eps)
        csi = hits / (hits + misses + false_alarms + eps)
        bias = (hits + false_alarms) / (hits + misses + eps)
        
        return {'POD': pod, 'FAR': far, 'CSI': csi, 'BIAS': bias,
                'hits': hits, 'misses': misses, 'false_alarms': false_alarms}
    
    def compute_continuous_metrics(self, obs: Union[np.ndarray, torch.Tensor], 
                                 pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """Compute continuous metrics."""
        diff = pred - obs
        spatial_axes = (-2, -1)
        
        if isinstance(obs, torch.Tensor):
            mse = torch.mean(diff**2, dim=spatial_axes)
            mae = torch.mean(torch.abs(diff), dim=spatial_axes)
            bias = torch.mean(diff, dim=spatial_axes)
            rmse = torch.sqrt(mse)
        else:
            mse = np.mean(diff**2, axis=spatial_axes)
            mae = np.mean(np.abs(diff), axis=spatial_axes)
            bias = np.mean(diff, axis=spatial_axes)
            rmse = np.sqrt(mse)
        
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'Bias': bias}
    
    def compute_fss(self, obs: Union[np.ndarray, torch.Tensor], pred: Union[np.ndarray, torch.Tensor],
                   threshold: float, scale: int) -> Union[np.ndarray, torch.Tensor]:
        """Compute Fractions Skill Score."""
        if isinstance(obs, torch.Tensor):
            obs_binary = (obs >= threshold).float()
            pred_binary = (pred >= threshold).float()
            
            if scale > 1:
                kernel = torch.ones((1, 1, scale, scale), device=obs.device) / (scale * scale)
                # Handle different input dimensions
                need_unsqueeze = obs_binary.ndim == 2
                if need_unsqueeze:
                    obs_binary = obs_binary.unsqueeze(0).unsqueeze(0)
                    pred_binary = pred_binary.unsqueeze(0).unsqueeze(0)
                elif obs_binary.ndim == 3:
                    obs_binary = obs_binary.unsqueeze(1)
                    pred_binary = pred_binary.unsqueeze(1)
                
                obs_frac = torch.nn.functional.conv2d(obs_binary, kernel, padding=scale//2)
                pred_frac = torch.nn.functional.conv2d(pred_binary, kernel, padding=scale//2)
                
                if need_unsqueeze:
                    obs_frac = obs_frac.squeeze(0).squeeze(0)
                    pred_frac = pred_frac.squeeze(0).squeeze(0)
                elif obs_binary.ndim == 4:
                    obs_frac = obs_frac.squeeze(1)
                    pred_frac = pred_frac.squeeze(1)
            else:
                obs_frac = obs_binary
                pred_frac = pred_binary
                
            mse_f = torch.mean((pred_frac - obs_frac)**2, dim=(-2, -1))
            mse_f_ref = torch.mean(pred_frac**2 + obs_frac**2, dim=(-2, -1))
            return 1 - mse_f / (mse_f_ref + 1e-10)
        else:
            from scipy import ndimage
            obs_binary = obs >= threshold
            pred_binary = pred >= threshold
            
            if scale > 1:
                if obs_binary.ndim > 2:
                    obs_frac = np.zeros_like(obs_binary, dtype=float)
                    pred_frac = np.zeros_like(pred_binary, dtype=float)
                    for idx in np.ndindex(obs_binary.shape[:-2]):
                        obs_frac[idx] = ndimage.uniform_filter(obs_binary[idx].astype(float), size=scale)
                        pred_frac[idx] = ndimage.uniform_filter(pred_binary[idx].astype(float), size=scale)
                else:
                    obs_frac = ndimage.uniform_filter(obs_binary.astype(float), size=scale)
                    pred_frac = ndimage.uniform_filter(pred_binary.astype(float), size=scale)
            else:
                obs_frac = obs_binary.astype(float)
                pred_frac = pred_binary.astype(float)
            
            mse_f = np.mean((pred_frac - obs_frac)**2, axis=(-2, -1))
            mse_f_ref = np.mean(pred_frac**2 + obs_frac**2, axis=(-2, -1))
            return 1 - mse_f / (mse_f_ref + 1e-10)
    
    def compute_crps(self, obs: Union[np.ndarray, torch.Tensor], 
                    pred_ensemble: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compute CRPS for ensemble predictions."""
        if pred_ensemble.shape[-1] <= 1:
            # Deterministic prediction - CRPS = MAE
            pred_det = pred_ensemble.squeeze(-1) if pred_ensemble.shape[-1] == 1 else pred_ensemble
            diff = torch.abs(pred_det - obs) if isinstance(obs, torch.Tensor) else np.abs(pred_det - obs)
            return torch.mean(diff, dim=(-2, -1)) if isinstance(obs, torch.Tensor) else np.mean(diff, axis=(-2, -1))
        
        # Ensemble CRPS computation
        if isinstance(obs, torch.Tensor):
            pred_sorted = torch.sort(pred_ensemble, dim=-1)[0]
            n_members = pred_ensemble.shape[-1]
            crps = torch.zeros_like(obs)
            
            for i in range(n_members):
                weight = (2 * i + 1 - n_members) / n_members
                crps += torch.abs(pred_sorted[..., i] - obs) * weight
            
            for i in range(n_members):
                for j in range(n_members):
                    crps -= torch.abs(pred_sorted[..., i] - pred_sorted[..., j]) / (2 * n_members**2)
            
            return torch.mean(crps, dim=(-2, -1))
        else:
            pred_sorted = np.sort(pred_ensemble, axis=-1)
            n_members = pred_ensemble.shape[-1]
            crps = np.zeros_like(obs)
            
            for i in range(n_members):
                weight = (2 * i + 1 - n_members) / n_members
                crps += np.abs(pred_sorted[..., i] - obs) * weight
            
            for i in range(n_members):
                for j in range(n_members):
                    crps -= np.abs(pred_sorted[..., i] - pred_sorted[..., j]) / (2 * n_members**2)
            
            return np.mean(crps, axis=(-2, -1))


class PrecipitationEvaluator:
    """Streamlined precipitation nowcasting evaluator."""
    
    def __init__(self,
                 nowcast_method: str,
                 save_dir: str = "./results",
                 eval_subdir: str = "evaluations",
                 thresholds: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
                 leadtimes: List[int] = [5, 10, 15, 30, 60, 90],
                 scales: List[int] = [1, 2, 4, 8, 16, 32],
                 crop_size: Optional[Tuple[int, int]] = (64, 64),
                 save_frequency: int = 100,
                 compute_crps: bool = True,
                 compute_fss: bool = True):
        
        self.nowcast_method = nowcast_method
        self.thresholds = thresholds
        self.leadtimes = leadtimes
        self.scales = scales
        self.crop_size = crop_size
        self.save_frequency = save_frequency
        self.compute_crps = compute_crps
        self.compute_fss = compute_fss
        
        # Setup paths
        self.save_path = Path(save_dir) / eval_subdir / nowcast_method
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics computer
        self.metrics_computer = MetricsComputer(thresholds, scales)
        
        # Initialize accumulators
        self.reset_accumulators()
        
        # Metadata
        self.metadata = {
            'nowcast_method': nowcast_method,
            'thresholds': list(thresholds),
            'leadtimes': list(leadtimes),
            'scales': list(scales),
            'crop_size': list(crop_size),
            'created_at': datetime.now().isoformat(),
            'n_samples': 0,
            'n_batches': 0
        }
    
    def reset_accumulators(self):
        """Reset all metric accumulators."""
        n_leadtimes, n_thresholds, n_scales = len(self.leadtimes), len(self.thresholds), len(self.scales)
        
        # Use structured approach for better memory efficiency
        self.accumulators = {
            'categorical': {
                'hits': np.zeros((n_leadtimes, n_thresholds)),
                'misses': np.zeros((n_leadtimes, n_thresholds)),
                'false_alarms': np.zeros((n_leadtimes, n_thresholds))
            },
            'continuous': {
                'mse_sum': np.zeros(n_leadtimes),
                'mae_sum': np.zeros(n_leadtimes),
                'bias_sum': np.zeros(n_leadtimes),
                'count': np.zeros(n_leadtimes)
            }
        }
        
        if self.compute_fss:
            self.accumulators['fss'] = {
                'fss_sum': np.zeros((n_leadtimes, n_thresholds, n_scales)),
                'count': np.zeros((n_leadtimes, n_thresholds, n_scales))
            }
        
        if self.compute_crps:
            self.accumulators['crps'] = {
                'crps_sum': np.zeros(n_leadtimes),
                'count': np.zeros(n_leadtimes)
            }
    
    def process_batch(self, observations: Union[np.ndarray, torch.Tensor], 
                     predictions: Union[np.ndarray, torch.Tensor]):
        """Process batch efficiently without unnecessary conversions."""
        # Handle ensemble dimension
        is_ensemble = len(predictions.shape) == 6
        pred_mean = predictions.mean(dim=-1) if (isinstance(predictions, torch.Tensor) and is_ensemble) else \
                   (np.mean(predictions, axis=-1) if is_ensemble else predictions)
        
        # Apply cropping once
        if self.crop_size:
            observations = center_crop(observations, self.crop_size)
            pred_mean = center_crop(pred_mean, self.crop_size)
            if is_ensemble:
                predictions = center_crop(predictions, self.crop_size)
        
        batch_size = observations.shape[0]
        n_channels = observations.shape[1]
        
        # Process all leadtimes in one go
        for t_idx, leadtime in enumerate(self.leadtimes):
            time_idx = leadtime // 5 - 1
            if time_idx >= observations.shape[2]:
                continue
            
            # Extract time slice and squeeze channel if single
            obs_t = observations[:, :, time_idx]
            pred_t = pred_mean[:, :, time_idx]
            if n_channels == 1:
                obs_t, pred_t = obs_t.squeeze(1), pred_t.squeeze(1)
            
            # Compute all metrics at once
            cat_metrics = self.metrics_computer.compute_categorical_metrics(obs_t, pred_t)
            cont_metrics = self.metrics_computer.compute_continuous_metrics(obs_t, pred_t)
            
            # Update accumulators (convert to numpy only when accumulating)
            for key in ['hits', 'misses', 'false_alarms']:
                values = to_numpy(cat_metrics[key])
                self.accumulators['categorical'][key][t_idx] += np.sum(values, axis=0)
            
            # Continuous metrics
            cont_np = {k: to_numpy(v) for k, v in cont_metrics.items()}
            self.accumulators['continuous']['mse_sum'][t_idx] += np.sum(cont_np['MSE'])
            self.accumulators['continuous']['mae_sum'][t_idx] += np.sum(cont_np['MAE'])
            self.accumulators['continuous']['bias_sum'][t_idx] += np.sum(cont_np['Bias'])
            self.accumulators['continuous']['count'][t_idx] += batch_size
            
            # FSS metrics
            if self.compute_fss:
                for thr_idx, threshold in enumerate(self.thresholds):
                    for scale_idx, scale in enumerate(self.scales):
                        fss = self.metrics_computer.compute_fss(obs_t, pred_t, threshold, scale)
                        fss_val = to_numpy(fss)
                        self.accumulators['fss']['fss_sum'][t_idx, thr_idx, scale_idx] += np.sum(fss_val)
                        self.accumulators['fss']['count'][t_idx, thr_idx, scale_idx] += batch_size
            
            # CRPS
            if self.compute_crps and is_ensemble:
                pred_ens_t = predictions[:, :, time_idx]
                if n_channels == 1:
                    pred_ens_t = pred_ens_t.squeeze(1)
                crps = self.metrics_computer.compute_crps(obs_t, pred_ens_t)
                crps_val = to_numpy(crps)
                self.accumulators['crps']['crps_sum'][t_idx] += np.sum(crps_val)
                self.accumulators['crps']['count'][t_idx] += batch_size
        
        # Update metadata
        self.metadata['n_samples'] += batch_size
        self.metadata['n_batches'] += 1
        
        # Periodic saving
        if self.metadata['n_batches'] % self.save_frequency == 0:
            self.save_results(f"checkpoint_batch_{self.metadata['n_batches']}")
    
    def get_final_scores(self) -> Dict[str, Any]:
        """Compute final scores from accumulated metrics."""
        results = {}
        eps = 1e-10
        
        # Categorical scores
        cat_scores = []
        cat_acc = self.accumulators['categorical']
        for t_idx, leadtime in enumerate(self.leadtimes):
            for thr_idx, threshold in enumerate(self.thresholds):
                h, m, fa = cat_acc['hits'][t_idx, thr_idx], cat_acc['misses'][t_idx, thr_idx], cat_acc['false_alarms'][t_idx, thr_idx]
                cat_scores.append({
                    'leadtime': leadtime, 'threshold': threshold,
                    'POD': float(h / (h + m + eps)),
                    'FAR': float(fa / (h + fa + eps)),
                    'CSI': float(h / (h + m + fa + eps)),
                    'BIAS': float((h + fa) / (h + m + eps))
                })
        results['categorical'] = cat_scores
        
        # Continuous scores
        cont_scores = []
        cont_acc = self.accumulators['continuous']
        for t_idx, leadtime in enumerate(self.leadtimes):
            count = cont_acc['count'][t_idx]
            if count > 0:
                mse = cont_acc['mse_sum'][t_idx] / count
                cont_scores.append({
                    'leadtime': leadtime,
                    'MSE': float(mse),
                    'MAE': float(cont_acc['mae_sum'][t_idx] / count),
                    'RMSE': float(np.sqrt(mse)),
                    'Bias': float(cont_acc['bias_sum'][t_idx] / count)
                })
        results['continuous'] = cont_scores
        
        # FSS scores
        if self.compute_fss:
            fss_scores = []
            fss_acc = self.accumulators['fss']
            for t_idx, leadtime in enumerate(self.leadtimes):
                for thr_idx, threshold in enumerate(self.thresholds):
                    for scale_idx, scale in enumerate(self.scales):
                        count = fss_acc['count'][t_idx, thr_idx, scale_idx]
                        if count > 0:
                            fss_scores.append({
                                'leadtime': leadtime, 'threshold': threshold, 'scale': scale,
                                'FSS': float(fss_acc['fss_sum'][t_idx, thr_idx, scale_idx] / count)
                            })
            results['fss'] = fss_scores
        
        # CRPS scores
        if self.compute_crps:
            crps_scores = []
            crps_acc = self.accumulators['crps']
            for t_idx, leadtime in enumerate(self.leadtimes):
                count = crps_acc['count'][t_idx]
                if count > 0:
                    crps_scores.append({
                        'leadtime': leadtime,
                        'CRPS': float(crps_acc['crps_sum'][t_idx] / count)
                    })
            results['crps'] = crps_scores
        
        return results
    
    def save_results(self, filename: str = "final_results"):
        """Save results efficiently."""
        scores = self.get_final_scores()
        
        # Save as JSON (compact and readable)
        json_path = self.save_path / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump({'metadata': self.metadata, 'scores': scores}, f, indent=2)
        
        # Save as HDF5 (efficient binary format)
        h5_path = self.save_path / f"{filename}.h5"
        with h5py.File(h5_path, 'w') as f:
            # Metadata
            for k, v in self.metadata.items():
                f.attrs[k] = str(v) if isinstance(v, (list, tuple)) else v
            
            # Raw accumulators for potential reprocessing
            for metric_type, data_dict in self.accumulators.items():
                group = f.create_group(metric_type)
                for key, array in data_dict.items():
                    group.create_dataset(key, data=array, compression='gzip')
        
        print(f"Results saved: {json_path}, {h5_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume evaluation."""
        with h5py.File(checkpoint_path, 'r') as f:
            # Load metadata
            for key in f.attrs:
                self.metadata[key] = f.attrs[key]
            
            # Load accumulators
            for metric_type in self.accumulators:
                if metric_type in f:
                    for key in self.accumulators[metric_type]:
                        if key in f[metric_type]:
                            self.accumulators[metric_type][key] = f[metric_type][key][:]
        
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Resuming from batch {self.metadata['n_batches']}")