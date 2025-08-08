import torch
import torch.nn as nn
from torch.nn import functional as F

class BalancedLoss(nn.Module):
    def __init__(self,
                 normalization="zscore",
                 min_val=0,
                 max_val=55,
                 max_weight_r=10.0,
                 weight_intensity=1.0,
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 extended=False):
        """
        Initializes the BalancedLoss.
        Args:
            normalization (str): One of ['minmax', 'zscore']
            min_val (float): Minimum dBZ value (corresponding to normalized 0). Default: 0.
            max_val (float): Maximum dBZ value (corresponding to normalized 1). Default: 55.
            mean_log_rain (float): Mean of log rain rate. Default: 0.03019706713265408.
            mean_log_rain (float): Standard Diviation of log rain rate. Default: 0.5392297631902654.
            min_weight (float): Minimum rain rate (mm/h) for clamping the raw weights. Default: 1.0.
            max_weight (float): Maximum rain rate (mm/h) for clamping the raw weights. Default: 10.0.
            weight_intensity (float): A scalar to control the influence of the calculated weights.
                                      0.0 means pure MSE, 1.0 means full weighting. Default: 1.0.
        """
        super(BalancedLoss, self).__init__()
        assert normalization in ['minmax', 'zscore'], "normalization must be 'minmax' or 'zscore'"
        self.normalization = normalization

        self.register_buffer("min_val_dbz", torch.tensor(float(min_val), dtype=torch.float32))
        self.register_buffer("max_val_dbz", torch.tensor(float(max_val), dtype=torch.float32))
        self.register_buffer("mean_log_rain", torch.tensor(float(mean_log_rain), dtype=torch.float32))
        self.register_buffer("std_log_rain", torch.tensor(float(std_log_rain), dtype=torch.float32))
        
        self.register_buffer("max_weight_r", torch.tensor(float(max_weight_r), dtype=torch.float32))
        self.register_buffer("weight_intensity", torch.tensor(float(weight_intensity), dtype=torch.float32))
        self.extended = extended

    def forward(self, pred, target):
        """
        Computes the weighted Mean Squared Error loss.
        Weights are derived from the ground truth target's rain rate, and their influence
        is scaled by `weight_intensity`. 
        Pred and target should both be in normalized form (either minmax or zscore).

        Args:
            pred (torch.Tensor): Model's prediction, normalized.
            target (torch.Tensor): Ground truth target, normalized.

        Returns:
            torch.Tensor: Scalar loss value.
        """

        """
        pred and target should both be in normalized form (either minmax or zscore).
        """
        if self.normalization == 'minmax':
            # Denormalize from [0, 1] to dBZ
            range_dbz = self.max_val_dbz - self.min_val_dbz
            target_dBZ = target * range_dbz + self.min_val_dbz

            # Convert to rain rate
            power_target = target_dBZ / 10.0
            base_for_r_calc = (torch.pow(10.0, power_target) - 1.0) / 200.0
            r_from_target = torch.pow(torch.relu(base_for_r_calc), 5.0 / 8.0)

        elif self.normalization == 'zscore':
            # Denormalize from z-score to log10(rain)
            target_log10_r = target * self.std_log_rain + self.mean_log_rain

            # Convert to rain rate
            r_from_target = torch.pow(10.0, target_log10_r)

        # Clamp and calculate effective weights
        raw_weights = torch.clamp(r_from_target, max=self.max_weight_r) + 1.0
        effective_weights = 1.0 + (raw_weights - 1.0) * self.weight_intensity

        # Compute error (in normalized space, regardless of type)
        mse = (pred - target) ** 2
        weighted_mse = effective_weights * mse

        if self.extended:
            mae = torch.abs(pred - target)
            weighted_mae = effective_weights * mae
            return torch.mean(weighted_mse) + torch.mean(weighted_mae)

        return weighted_mse.mean()
    

class MultiSourceLoss(nn.Module):
    def __init__(self, 
                 balanced_channels, 
                 mse_channels, 
                 channel_wegihts,
                 min_val=0, 
                 max_val=55, 
                 mean_log_rain=0.03019706713265408,
                 std_log_rain=0.5392297631902654,
                 max_weight_r=10.0, 
                 weight_intensity=1.0, 
                 extended=False):
        """
        Args:
            balanced_channels (list of int): Indices of channels to apply BalancedLoss.
            mse_channels (list of int): Indices of channels to apply plain MSE loss.
            weights (list of float): Weight for each channel (same length as balanced + mse).
            Other args are forwarded to BalancedLoss.
        """
        super(MultiSourceLoss, self).__init__()

        assert len(channel_wegihts) == len(balanced_channels) + len(mse_channels), \
            "weights length must match total number of channels"

        self.balanced_channels = balanced_channels
        self.mse_channels = mse_channels
        self.channel_wegihts = channel_wegihts

        self.balanced_loss = BalancedLoss(
            min_val=min_val,
            max_val=max_val,
            mean_log_rain=mean_log_rain,
            std_log_rain=std_log_rain,
            max_weight_r=max_weight_r,
            weight_intensity=weight_intensity,
            extended=extended
        )

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): (B, T, H, W, C) prediction tensor
            target (Tensor): (B, T, H, W, C) target tensor
        Returns:
            Scalar tensor representing the weighted total loss.
        """
        total_loss = 0.0
        weight_idx = 0

        for c in self.balanced_channels:
            loss = self.balanced_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * loss
            weight_idx += 1

        for c in self.mse_channels:
            mse_loss = F.mse_loss(pred[..., c], target[..., c])
            total_loss += self.channel_wegihts[weight_idx] * mse_loss
            weight_idx += 1

        return total_loss
