import lightning as L
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchvision.utils

import torchmetrics

from .cuboid_transformer import CuboidTransformerModel
from .utils.utils import get_parameter_names
from .utils.optim import SequentialLR, warmup_lambda
from ...utils import data_prep

from ..losses import BalancedLoss, MultiSourceLoss

torch.set_float32_matmul_precision('medium')

class CuboidSEVIRPLModule(L.LightningModule):
    """
    PyTorch Lightning module for training and evaluating the `CuboidTransformerModel`
    on tasks like precipitation nowcasting (e.g., using SEVIR dataset characteristics).

    This module encapsulates the training loop, loss calculation, metric logging,
    optimizer configuration (including advanced learning rate scheduling), and
    sample image visualization.

    It is designed as an extension to the EarthFormer codebase, adapting the
    Cuboid Transformer for specific meteorological forecasting applications.
    """
    def __init__(self, cfg):
        super(CuboidSEVIRPLModule, self).__init__()

        self.config = cfg
    
        self.torch_nn_module = CuboidTransformerModel(
            input_shape=self.config.model.pop('input_shape'),
            target_shape=self.config.model.pop('target_shape'),
            **self.config.model,
        )

        loss_name = self.config.loss.pop('loss_name')
        if loss_name == 'balanced':
            self.criterion = BalancedLoss(**self.config.loss)
        elif loss_name == 'multi-source':
            self.criterion = MultiSourceLoss(**self.config.loss)
        else:
            self.criterion = F.mse_loss

        self.sample_images = None

        self.valid_r_mse = torchmetrics.MeanSquaredError()
        self.valid_r_mae = torchmetrics.MeanAbsoluteError()
        self.valid_r_csi_20mmh = torchmetrics.CriticalSuccessIndex(20.0)
        

    def configure_optimizers(self):
        cfg = self.config.optimizer

        # Configure the optimizer. Disable the weight decay for layer norm weights and all bias terms.
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n in decay_parameters],
            'weight_decay': cfg.weight_decay
        }, {
            'params': [p for n, p in self.torch_nn_module.named_parameters()
                       if n not in decay_parameters],
            'weight_decay': 0.0
        }]

        if cfg.method == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=cfg.lr,
                                          weight_decay=cfg.weight_decay)
        else:
            raise NotImplementedError
        
        warmup_iter = int(np.round(cfg.warmup_percentage * self.trainer.estimated_stepping_batches))

        if cfg.lr_scheduler_mode == 'cosine':
            warmup_scheduler = LambdaLR(optimizer,
                                        lr_lambda=warmup_lambda(warmup_steps=warmup_iter,
                                                                min_lr_ratio=cfg.warmup_min_lr_ratio))
            cosine_scheduler = CosineAnnealingLR(optimizer,
                                                 T_max=(self.trainer.estimated_stepping_batches - warmup_iter),
                                                 eta_min=cfg.min_lr_ratio * cfg.lr)
            lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                        milestones=[warmup_iter])
            lr_scheduler_config = {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }

            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
        else:
            return {'optimizer': optimizer}

    def forward(self, in_seq, out_seq):
        output = self.torch_nn_module(in_seq)
        loss = self.criterion(output, out_seq)

        return output, loss

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Shape (B, C, T, W, H) -> Earthformer Shape (B, T, W, H, C)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        y = y.permute(0, 2, 3, 4, 1).contiguous()

        y_hat, loss = self(x, y)

        loss_mse = F.mse_loss(y_hat[:,:,:,:,0], y[:,:,:,:,0])

        self.log('train/loss_mse', loss_mse,
                prog_bar=True,
                on_step=True,
                on_epoch=True,  
                logger=True,
                sync_dist=True)

        self.log('train/loss', loss,
                prog_bar=True,
                on_step=True,
                on_epoch=True,  
                logger=True,
                sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Shape (B, C, T, W, H) -> Earthformer Shape (B, T, W, H, C)
        x = x.permute(0, 2, 3, 4, 1)
        y = y.permute(0, 2, 3, 4, 1)

        y_hat, loss = self(x, y)

        if y_hat.shape[-1] != 1:
            y_hat = y_hat[..., 0:1]
            y = y[..., 0:1]

        loss_mse = F.mse_loss(y_hat, y)
        
        y_mmh = self.to_mmh(y.clone())
        y_hat_mmh = self.to_mmh(y_hat.clone())
        # data_prep(y_mmh, convert_to_dbz=True, undo=True)
        # data_prep(y_hat_mmh, convert_to_dbz=True, undo=True)

        step_mse = self.valid_r_mse(y_hat_mmh, y_mmh)
        step_mae = self.valid_r_mae(y_hat_mmh, y_mmh)
        step_csi = self.valid_r_csi_20mmh(y_hat_mmh, y_mmh)

        self.log('val/loss', loss_mse,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                logger=True,
                sync_dist=True)
        
        self.log('val/loss_balanced', loss,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                logger=True,
                sync_dist=True)

        if batch_idx == 636 and self.global_rank == 0:
            # Detach and move to CPU to avoid memory issues
            input_img = x[0].permute(0, 3, 1, 2).detach().cpu()
            target_img = y[0].permute(0, 3, 1, 2).detach().cpu()
            pred_img = y_hat[0].permute(0, 3, 1, 2).detach().cpu()
            self.sample_images = (input_img, target_img, pred_img)

        return loss_mse

    def to_mmh(self, img, mean=0.03019706713265408, std=0.5392297631902654):
        # Undo z-score normalization and convert from log10 to mm/h
        img = (img * std + mean)
        img_mmh = 10 ** img

        # Clip to physical range depending on type
        if isinstance(img_mmh, torch.Tensor):
            img_mmh = torch.clamp(img_mmh, min=0.0, max=160.0)
        elif isinstance(img_mmh, np.ndarray):
            img_mmh = np.clip(img_mmh, 0.0, 160.0)
        else:
            raise TypeError("Input must be a numpy array or a torch tensor")

        return img_mmh

    def get_colormap(self, name):
        if name == 'mmh':
            reds = "#7D7D7D", "#640064", "#AF00AF", "#DC00DC", "#3232C8", "#0064FF", \
                "#009696", "#00C832", "#64FF00", "#96FF00", "#C8FF00", "#FFFF00", \
                "#FFC800", "#FFA000", "#FF7D00", "#E11900"
            clevs = [0.08, 0.16, 0.25, 0.40, 0.63, 1, 1.6, 2.5, 4, 6.3, 10, 16, 25, 40, 63, 100, 160]
            cmap = matplotlib.colors.ListedColormap(reds)
            norm = matplotlib.colors.BoundaryNorm(clevs, len(reds))
            return cmap, norm
        cmap = matplotlib.colormaps.get(name)
        if cmap is None:
            raise ValueError(f"Unknown colormap: {name}")
        return cmap, None

    def _apply_colormap(self, img, cmap="mmh", ):
        """
        Apply a matplotlib colormap (including 'mmh') to a Z-scored log10 rain rate image.

        img: (H, W) torch.Tensor in z-score space.
        Returns: (3, H, W) torch.Tensor in uint8 RGB.
        """
        img_np = img.numpy()

        img_np = self.to_mmh(img_np)

        # Get colormap and norm
        cmap_obj, norm_obj = self.get_colormap(cmap)

        if norm_obj is not None:
            img_colored = cmap_obj(norm_obj(img_np))[:, :, :3]
        else:
            # Fallback to normal continuous colormap
            img_min, img_max = img_np.min(), img_np.max()
            img_norm = (img_np - img_min) / (img_max - img_min) if img_max > img_min else np.zeros_like(img_np)
            img_colored = cmap_obj(img_norm)[:, :, :3]

        img_colored = (img_colored * 255).astype(np.uint8)
        img_colored = torch.from_numpy(img_colored).permute(2, 0, 1)  # (C, H, W)

        return img_colored

    def on_validation_epoch_end(self):

        valid_mse = self.valid_r_mse.compute()
        valid_mae = self.valid_r_mae.compute()
        valid_csi = self.valid_r_csi_20mmh.compute()
        self.log_dict({'mse_r': valid_mse, 'mae_r': valid_mae, 'csi_20mmh': valid_csi},
                        prog_bar=True,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        sync_dist=True)
        self.valid_r_mse.reset()
        self.valid_r_mae.reset()
        self.valid_r_csi_20mmh.reset()

        if self.sample_images is not None and self.global_rank == 0:
            input_img, target_img, pred_img = self.sample_images
            
            # input_img = self._undo_process_radar(input_img)
            # target_img = self._undo_process_radar(target_img)
            # pred_img = self._undo_process_radar(pred_img)

            # Get dimensions
            T_in, C, H, W = input_img.shape
            if C > 1:
                input_img = input_img[:, :1, :, :]

            T_out = target_img.shape[0]

            last_input = input_img[-1].squeeze()  # Ensure shape (H, W)

            interleaved = []
            for t in range(T_out):
                interleaved.append(target_img[t].squeeze())  # (H, W)
                interleaved.append(pred_img[t].squeeze())    # (H, W)

            all_frames = [last_input] + interleaved
            all_frames_colored = [self._apply_colormap(frame) for frame in all_frames]
            all_frames_colored = torch.stack(all_frames_colored)  # Shape (N, 3, H, W)

            grid = torchvision.utils.make_grid(all_frames_colored, nrow=len(all_frames_colored))

            self.logger.experiment.add_image(
                "forecast_samples_colored",
                grid,
                global_step=self.current_epoch,
            )

            self.sample_images = None