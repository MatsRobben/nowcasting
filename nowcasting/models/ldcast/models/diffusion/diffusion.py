"""
From https://github.com/CompVis/latent-diffusion/main/ldm/models/diffusion/ddpm.py
Pared down to simplify code.

The original file acknowledges:
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision.utils
import matplotlib.pyplot as plt
import matplotlib
import lightning as L
from contextlib import contextmanager
from functools import partial
from omegaconf import DictConfig


from .utils import make_beta_schedule, extract_into_tensor, noise_like, timestep_embedding
from .ema import LitEma
from .plms import PLMSSampler
from ..blocks.afno import PatchEmbed3d, PatchExpand3d, AFNOBlock3d

from ..autoenc.autoenc import AutoencoderKL
from ..genforecast import analysis, unet

class LatentDiffusion(L.LightningModule):
    def __init__(self,
        config: DictConfig
    ):
        super().__init__()
        self.config = config

        # --- 1. Instantiate and Load Components from Config ---
        # Instantiate components in order of dependency:
        # Autoencoder -> Context Encoder -> UNet Model

        # Load the main autoencoder for the latent space
        autoencoders = [
            self._load_autoencoder(cfg, ckpt)
            for cfg, ckpt in zip(config.model.autoencoders, config.model.autoencoder_ckpts)
        ]

        # Assume the first autoencoder is the main
        self.autoencoder = autoencoders[0].requires_grad_(False)

        # Load the context encoder if specified (for conditional diffusion)
        self.context_encoder = None
        if "context_encoder" in config.model:
            self.context_encoder = self._load_context_encoder(autoencoders, config.model.context_encoder)
            self.conditional = True
        else:
            self.conditional = False

        # Instantiate the core diffusion model (e.g., UNet)
        # Its parameters can depend on the autoencoder and context encoder
        self.model = unet.UNetModel(
            in_channels=self.autoencoder.hidden_width,
            out_channels=self.autoencoder.hidden_width,
            context_ch=self.context_encoder.cascade_dims if self.conditional else None,
            **config.model.model  # Pass the rest of the UNet-specific params
        )

        # --- 2. Set Up Diffusion & Training Parameters ---
        self.lr = config.optimizer.lr
        self.lr_warmup = config.optimizer.get("lr_warmup", 0)
        self.val_num_diffusion_iters = config.model.get("num_diffusion_iters", 50)

        self.parameterization = config.model.get("parameterization", "eps")
        assert self.parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'

        self.loss_type = config.loss.get("loss_name", "l2")

        # --- 3. Initialize EMA and Noise Schedule ---
        self.use_ema = config.model.get("use_ema", True)
        if self.use_ema:
            self.model_ema = LitEma(self.model)

        self.register_schedule(
            beta_schedule=config.model.get("beta_schedule", "linear"),
            timesteps=config.model.get("timesteps", 1000),
            linear_start=config.model.get("linear_start", 1e-4),
            linear_end=config.model.get("linear_end", 2e-2),
            cosine_s=config.model.get("cosine_s", 8e-3)
        )

        self.sample_images = None
        self.sampler = PLMSSampler(self)

    def _load_autoencoder(self, config, checkpoint_path):
        """Instantiate autoencoder from config and load weights"""
        if config is None or checkpoint_path is None:
            return None
        
        # Create autoencoder
        autoencoder = AutoencoderKL(config)

        map_location = self.device if hasattr(self, 'device') else 'cpu'
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
        autoencoder.load_state_dict(checkpoint['state_dict'], strict=False)
        
        print("Instantiated autoencoder and loaded weights.")
        return autoencoder

    def _load_context_encoder(self, autoencoders, context_config: DictConfig) -> analysis.AFNONowcastNetCascade:
        """Instantiates a context encoder (AFNO cascade) from config."""
        # Instantiate the main context encoder model
        context_encoder = analysis.AFNONowcastNetCascade(
            autoencoders,
            **context_config # Pass other params like input_patches, embed_dim, etc.
        )
        print("Instantiated context encoder.")
        return context_encoder

    def register_schedule(self, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):

        betas = make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end,
            cosine_s=cosine_s
        )
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def apply_model(self, x_noisy, t, cond=None, return_ids=False):
        if self.conditional:
            cond = self.context_encoder(cond)
        with self.ema_scope():
            return self.model(x_noisy, t, context=cond)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t, context=context)

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

        return self.get_loss(model_out, target, mean=False).mean()

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def shared_step(self, batch):
        (x,y) = batch
        y = self.autoencoder.encode(y)[0]
        context = self.context_encoder(x) if self.conditional else None
        return self(y, context=context)

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", lr, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        with self.ema_scope():
            loss_ema = self.shared_step(batch)

        if batch_idx == 73 and self.global_rank == 0:
            (x, y) = batch
            x_s = x[0][0][0:1]
            x_t = x[0][1][0:1]
            y = y[0:1]
            x = [[x_s, x_t]]

            batch, channel, time, width, height = y.shape
            gen_shape = (self.autoencoder.hidden_width, time//4, width//4, height//4)

            with self.ema_scope():
                # The sampler needs the model (self), steps, batch_size, shape, condition
                # Condition should be the encoded context for the *first* batch item
                latent_sample, _ = self.sampler.sample(
                    S=self.val_num_diffusion_iters,
                    batch_size=1, # Generate only one sample
                    shape=gen_shape, # Shape without batch dim for sampler
                    conditioning=x, # Pass encoded context for the sample
                    verbose=False, # Disable inner progress bar
                )

            pred_img = self.autoencoder.decode(latent_sample)

            input_img = x_s[0].permute(1,0,2,3).cpu() # Input frames for the sample
            target_img = y[0].permute(1,0,2,3).cpu() # Target frames for the sample
            pred_img = pred_img[0].permute(1,0,2,3).cpu() # Predicted frames (remove batch dim)

            self.sample_images = (input_img, target_img, pred_img)
            print(f"Validation Sampling: Generated sample. Input: {input_img.shape}, Target: {target_img.shape}, Prediction: {pred_img.shape}")


        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log("val/loss", loss, **log_params)
        self.log("val/loss_ema", loss_ema, **log_params)

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

    def _apply_colormap(self, img, cmap="mmh", mean=0.03019706713265408, std=0.5392297631902654):
        """
        Apply a matplotlib colormap (including 'mmh') to a Z-scored log10 rain rate image.

        img: (H, W) torch.Tensor in z-score space.
        Returns: (3, H, W) torch.Tensor in uint8 RGB.
        """
        img_np = img.numpy()

        # Undo z-score normalization
        img_np = img_np * std + mean

        # Convert back from log10 to mm/h
        img_np = 10 ** img_np

        # Clip to reasonable physical range
        img_np = np.clip(img_np, 0, 160)  # max matches your clevs

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
            betas=(0.5, 0.9), weight_decay=1e-3)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.25
        )

        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": "val/loss_ema",
                "frequency": 1,
            },# reduce_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.trainer.estimated_stepping_batches,
        #     eta_min=1e-9  # Minimum learning rate
        # )
        }

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def optimizer_step(
        self, 
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
        **kwargs    
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = (self.trainer.global_step+1) / self.lr_warmup
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr

        super().optimizer_step(
            epoch, batch_idx, optimizer,
            optimizer_closure=optimizer_closure,
            **kwargs
        )
    
