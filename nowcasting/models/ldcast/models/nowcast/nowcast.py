import collections

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import torchmetrics
import torchvision.utils

from ..blocks.afno import AFNOBlock3d
from ..blocks.attention import positional_encoding, TemporalTransformer
from ..autoenc.autoenc import AutoencoderKL
from ....losses import BalancedLoss

torch.set_float32_matmul_precision('medium')

class Nowcaster(L.LightningModule):
    def __init__(self, nowcast_net):
        super().__init__()
        self.nowcast_net = nowcast_net

    def forward(self, x):
        return self.nowcast_net(x)

    def _loss(self, batch):
        (x,y) = batch
        y_pred = self.forward(x)
        return (y-y_pred).square().mean()

    def training_step(self, batch, batch_idx):        
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        loss = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True}
        self.log(f"{split}/loss", loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1e-3,
            betas=(0.5, 0.9), weight_decay=1e-3
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25, verbose=True
        )

        optimizer_spec = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss",
                "frequency": 1,
            },
        }
        return optimizer_spec

class FusionBlock3d(nn.Module):
    def __init__(self, dim, size_ratios, dim_out=None, afno_fusion=False):
        super().__init__()

        N_sources = len(size_ratios)
        if not isinstance(dim, collections.abc.Sequence):
            dim = (dim,) * N_sources
        if dim_out is None:
            dim_out = dim[0]
        
        self.scale = nn.ModuleList()
        for (i,size_ratio) in enumerate(size_ratios):
            if size_ratio == 1:
                scale = nn.Identity()
            else:
                scale = []
                while size_ratio > 1:
                    scale.append(nn.ConvTranspose3d(
                        dim[i], dim_out if size_ratio==2 else dim[i],
                        kernel_size=(1,3,3), stride=(1,2,2),
                        padding=(0,1,1), output_padding=(0,1,1)
                    ))
                    size_ratio //= 2
                scale = nn.Sequential(*scale)
            self.scale.append(scale)

        self.afno_fusion = afno_fusion
        
        if self.afno_fusion:
            if N_sources > 1:
                self.fusion = nn.Sequential(
                    nn.Linear(sum(dim), sum(dim)),
                    AFNOBlock3d(dim*N_sources, mlp_ratio=2),
                    nn.Linear(sum(dim), dim_out)
                )
            else:
                self.fusion = nn.Identity()
        
    def resize_proj(self, x, i):
        x = x.permute(0,4,1,2,3)
        x = self.scale[i](x)
        x = x.permute(0,2,3,4,1)
        return x

    def forward(self, x):
        x = [self.resize_proj(xx, i) for (i, xx) in enumerate(x)]
        if self.afno_fusion:        
            x = torch.concat(x, axis=-1)
            x = self.fusion(x)
        else:
            x = sum(x)
        return x


class AFNONowcastNetBase(nn.Module):
    def __init__(
        self,
        autoencoder,
        embed_dim=128,
        embed_dim_out=None,
        analysis_depth=4,
        forecast_depth=4,
        per_input_forecast_depth=0,
        input_patches=(1,),
        input_size_ratios=(1,),
        output_patches=2,
        train_autoenc=False,
        afno_fusion=False,
        use_forecast=True,
    ):
        super().__init__()
        
        self.train_autoenc = train_autoenc
        if not isinstance(autoencoder, collections.abc.Sequence):
            autoencoder = [autoencoder]
        if not isinstance(input_patches, collections.abc.Sequence):
            input_patches = [input_patches]        
        num_inputs = len(autoencoder)
        if not isinstance(embed_dim, collections.abc.Sequence):
            embed_dim = [embed_dim] * num_inputs
        if embed_dim_out is None:
            embed_dim_out = embed_dim[0]
        if not isinstance(analysis_depth, collections.abc.Sequence):
            analysis_depth = [analysis_depth] * num_inputs
        if not isinstance(per_input_forecast_depth, collections.abc.Sequence):
            per_input_forecast_depth = [per_input_forecast_depth] * num_inputs
        self.embed_dim = embed_dim
        self.embed_dim_out = embed_dim_out
        self.output_patches = output_patches

        # encoding + analysis for each input
        self.autoencoder = nn.ModuleList()
        self.proj = nn.ModuleList()
        self.analysis = nn.ModuleList()
        self.per_input_forecast = nn.ModuleList()
        for i in range(num_inputs):
            ae = autoencoder[i].requires_grad_(train_autoenc)
            self.autoencoder.append(ae)

            proj = nn.Conv3d(ae.hidden_width, embed_dim[i], kernel_size=1)
            self.proj.append(proj)

            analysis = nn.Sequential(
                *(AFNOBlock3d(embed_dim[i]) for _ in range(analysis_depth[i]))
            )
            self.analysis.append(analysis)

            depth = per_input_forecast_depth[i]
            if depth > 0:
                forecast_block = nn.Sequential(
                    *(AFNOBlock3d(embed_dim[i]) for _ in range(depth)))
                self.per_input_forecast.append(forecast_block)
            else:
                self.per_input_forecast.append(nn.Identity())

        # temporal transformer
        self.use_temporal_transformer = \
            any((ipp != output_patches) for ipp in input_patches)
        if self.use_temporal_transformer:
            self.temporal_transformer = nn.ModuleList(
                TemporalTransformer(embed_dim[i]) for i in range(num_inputs)
            )

        # data fusion
        self.fusion = FusionBlock3d(embed_dim, input_size_ratios,
            afno_fusion=afno_fusion, dim_out=embed_dim_out)

        # forecast (optional)
        if use_forecast and forecast_depth > 0:
            self.forecast = nn.Sequential(
                *(AFNOBlock3d(embed_dim_out) for _ in range(forecast_depth))
            )
        else:
            self.forecast = nn.Identity()

    def add_pos_enc(self, x, t):
        if t.shape[1] != x.shape[1]:
            # this can happen if x has been compressed 
            # by the autoencoder in the time dimension
            ds_factor = t.shape[1] // x.shape[1]
            t = F.avg_pool1d(t.unsqueeze(1), ds_factor)[:,0,:]

        pos_enc = positional_encoding(t, x.shape[-1], add_dims=(2,3))
        return x + pos_enc

    def forward(self, x):
        (x, t_relative) = list(zip(*x))

        # encoding + analysis for each input
        def process_input(i):
            z = self.autoencoder[i].encode(x[i])[0]
            z = self.proj[i](z)
            z = z.permute(0,2,3,4,1)
            z = self.analysis[i](z)

            if self.use_temporal_transformer:
                # add positional encoding
                z = self.add_pos_enc(z, t_relative[i])
                
                # transform to output shape and coordinates
                expand_shape = z.shape[:1] + (-1,) + z.shape[2:]
                pos_enc_output = positional_encoding(
                    torch.arange(1,self.output_patches+1, device=z.device), 
                    self.embed_dim[i], add_dims=(0,2,3)
                )
                pe_out = pos_enc_output.expand(*expand_shape)
                z = self.temporal_transformer[i](pe_out, z)

            z = self.per_input_forecast[i](z)
            return z

        x = [process_input(i) for i in range(len(x))]
        
        # merge inputs
        x = self.fusion(x)

        # produce prediction
        x = self.forecast(x)
        return x.permute(0,4,1,2,3) # to channels-first order


class AFNONowcastNet(AFNONowcastNetBase):
    def __init__(self, autoencoder, output_autoencoder=None, **kwargs):
        super().__init__(autoencoder, **kwargs)
        if output_autoencoder is None or len(output_autoencoder)==0:
            output_autoencoder = autoencoder[0]
        self.output_autoencoder = output_autoencoder.requires_grad_(
            self.train_autoenc)
        self.out_proj = nn.Conv3d(
            self.embed_dim_out, output_autoencoder.hidden_width, kernel_size=1
        )

    def forward(self, x):
        x = super().forward(x)
        x = self.out_proj(x)
        return self.output_autoencoder.decode(x)
    
class AFNONowcastModule(L.LightningModule):
    """
    PyTorch Lightning module for training, validating, and testing an AFNO-based
    nowcasting model (`AFNONowcastNet`).

    This module handles:
    - Model initialization with optional input/output autoencoders.
    - Loading and optionally freezing pre-trained weights for various model components.
    - Selection of different loss functions (balanced, MSE, MAE).
    - Accumulation and logging of key meteorological verification metrics (MSE, MAE, CSI).
    - Visualization of forecast samples during validation using TensorBoard.

    This script is an addition to the original LDCast codebase by MeteoSwiss.
    """
    def __init__(
        self, config
    ):
        super().__init__()

        self.config = config
        input_autoencoder_configs = config.model.pop('input_autoencoders', [None])
        input_autoencoder_ckpts = config.model.pop('input_autoencoder_ckpts', [None])
        output_autoencoder_config = config.model.pop('output_autoencoder', None)
        output_autoencoder_ckpt = config.model.pop('output_autoencoder_ckpt', None)
        pretrained_paths = config.model.pop('pretrained_paths', None)
        freeze_pretrained = config.model.pop('freeze_pretrained', None)

        input_autoencoders = [
            self._load_autoencoder(cfg, ckpt)
            for cfg, ckpt in zip(input_autoencoder_configs, input_autoencoder_ckpts)
        ]
        output_autoencoder = self._load_autoencoder(
            output_autoencoder_config, output_autoencoder_ckpt
        )

        self.model = AFNONowcastNet(
            autoencoder=input_autoencoders,
            output_autoencoder=output_autoencoder,
            **config.model
        )

        loss_name = self.config.loss.pop('loss_name')
        if loss_name == 'balanced':
            self.criterion = BalancedLoss(**self.config.loss)
        elif loss_name == 'mse':
            self.criterion = F.mse_loss
        elif loss_name == 'mae':
            self.criterion = F.l1_loss
        else:
            print(f'Loss named {loss_name} is not implemented, use: balanced, mse or mae')
            raise

        # Load and freeze pretrained weights
        if pretrained_paths:
            num_inputs = len(input_autoencoders)
            if not isinstance(pretrained_paths, collections.abc.Sequence):
                pretrained_paths = [pretrained_paths] * num_inputs
            if freeze_pretrained is None:
                freeze_pretrained = [True] * num_inputs
            
            for i, path in enumerate(pretrained_paths):
                if path:
                    self._load_pretrained(i, path, freeze_pretrained[i])


        self.sample_images = None

        self.valid_r_mse = torchmetrics.MeanSquaredError()
        self.valid_r_mae = torchmetrics.MeanAbsoluteError()
        self.valid_r_csi_20mmh = torchmetrics.CriticalSuccessIndex(20.0)

    def _load_pretrained(self, input_idx, path, freeze=True):
        """Load pretrained weights for a specific input pipeline"""

        def extract_submodule_state(pretrained_state_dict, prefix):
            """Strip prefix and return a filtered state_dict for a submodule, or None if no match."""
            subdict = {
                k[len(prefix):]: v for k, v in pretrained_state_dict.items()
                if k.startswith(prefix)
            }
            return subdict if subdict else None
    
        # Load checkpoint
        map_location = self.device if hasattr(self, 'device') else 'cpu'
        pretrained = torch.load(path, weights_only=False, map_location=map_location)
        pretrained_state_dict = pretrained['state_dict']

        # Load autoencoder
        ae_prefix = f'model.autoencoder.0.'
        autoencoder_sd = extract_submodule_state(pretrained_state_dict, ae_prefix)
        self.model.autoencoder[input_idx].load_state_dict(autoencoder_sd)

        # Load proj
        proj_prefix = f'model.proj.0.'
        proj_sd = extract_submodule_state(pretrained_state_dict, proj_prefix)
        self.model.proj[input_idx].load_state_dict(proj_sd)

        # Load analysis
        analysis_prefix = f'model.analysis.0.'
        analysis_sd = extract_submodule_state(pretrained_state_dict, analysis_prefix)
        self.model.analysis[input_idx].load_state_dict(analysis_sd)

        # Load temporal transformer if keys are present
        tt_prefix = f'model.temporal_transformer.0.'
        tt_sd = extract_submodule_state(pretrained_state_dict, tt_prefix)
        if tt_sd and hasattr(self.model, 'temporal_transformer'):
            model_tt_keys = set(self.model.temporal_transformer[input_idx].state_dict().keys())
            pretrained_tt_keys = set(tt_sd.keys())

            if model_tt_keys == pretrained_tt_keys:
                self.model.temporal_transformer[input_idx].load_state_dict(tt_sd)
            else:
                print(f"[Warning] Temporal Transformer key mismatch for input {input_idx}.")
                print(f"Missing in pretrained: {model_tt_keys - pretrained_tt_keys}")
                print(f"Unexpected in pretrained: {pretrained_tt_keys - model_tt_keys}")

        # Load per_input_forecast if keys are present
        pf_prefix = f'model.per_input_forecast.0.'
        pf_sd = extract_submodule_state(pretrained_state_dict, pf_prefix)
        if pf_sd and hasattr(self.model, 'per_input_forecast'):
            self.model.per_input_forecast[input_idx].load_state_dict(pf_sd)

        # Optionally freeze the pipeline
        if freeze:
            self._freeze_input_pipeline(input_idx)

    def _freeze_input_pipeline(self, input_idx):
        """Freeze specific input pipeline"""
        modules = [
            self.model.autoencoder[input_idx],
            self.model.proj[input_idx],
            self.model.analysis[input_idx],
            self.model.per_input_forecast[input_idx]
        ]
        if hasattr(self.model, 'temporal_transformer'):
            modules.append(self.model.temporal_transformer[input_idx])
        
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def _load_autoencoder(self, config, checkpoint_path):
        """Instantiate autoencoder from config and load weights"""
        if config is None or checkpoint_path is None:
            return None
        
        # Create autoencoder
        autoencoder = AutoencoderKL(config)

        print(config)
        print(checkpoint_path)

        map_location = self.device if hasattr(self, 'device') else 'cpu'
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        autoencoder.load_state_dict(checkpoint['state_dict'], strict=False)
        
        return autoencoder

    def forward(self, x):
        return self.model(x)

    def _loss(self, batch):
        (x,y) = batch

        y_pred = self.forward(x)

        loss = self.criterion(y_pred, y)

        return y_pred, loss

    def training_step(self, batch, batch_idx):
        y_pred, loss = self._loss(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        y = batch[1]
        x = batch[0][0][0]
        y_pred, loss = self._loss(batch)

        if y_pred.shape[1] != 1:
            y_pred = y_pred[:, 0:1]
            y = y[:, 0:1]

        y_mmh = self.to_mmh(y.clone())
        y_hat_mmh = self.to_mmh(y_pred.clone())

        step_mse = self.valid_r_mse(y_hat_mmh, y_mmh)
        step_mae = self.valid_r_mae(y_hat_mmh, y_mmh)
        step_csi = self.valid_r_csi_20mmh(y_hat_mmh, y_mmh)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}/loss", loss, **log_params)

        if batch_idx == 18 and self.global_rank == 0:
            # Detach and move to CPU to avoid memory issues
            input_img = x[0].permute(1, 0, 2, 3).detach().cpu()
            target_img = y[0].permute(1, 0, 2, 3).detach().cpu()
            pred_img = y_pred[0].permute(1, 0, 2, 3).detach().cpu()
            self.sample_images = (input_img, target_img, pred_img)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")
        
    def to_mmh(self, img, norm_method='zscore', mean=0.03019706713265408, std=0.5392297631902654):
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
        self.log_dict({'val/mse_mmh': valid_mse, 'val/mae_mmh': valid_mae, 'val/csi_20mmh': valid_csi},
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

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        cfg = self.config.optimizer

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=cfg.lr,
            betas=cfg.betas, weight_decay=cfg.weight_decay
        )
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=cfg.patience, factor=cfg.factor
        )

        optimizer_spec = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": cfg.monitor,
                "frequency": 1,
            },
        }
        return optimizer_spec