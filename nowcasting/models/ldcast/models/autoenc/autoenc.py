import torch
from torch import nn
import lightning as L

from ..distributions import kl_from_standard_normal, ensemble_nll_normal
from ..distributions import sample_from_standard_normal

from .encoder import SimpleConvDecoder, SimpleConvEncoder

class AutoencoderKL(L.LightningModule):
    def __init__(
        self, config
    ):
        super(AutoencoderKL, self).__init__()
        enc_params = config.model.get('enc_params', {})
        dec_params = config.model.get('dec_params', {})
        kl_weight = config.model.get('kl_weight', 0.01)
        encoded_channels = config.model.get('encoded_channels', 64)
        hidden_width = config.model.get('hidden_width', 32)
        self.config = config

        self.encoder = SimpleConvEncoder(**enc_params)
        self.decoder = SimpleConvDecoder(**dec_params)
        self.hidden_width = config.model.hidden_width
        self.to_moments = nn.Conv3d(encoded_channels, 2*hidden_width,
            kernel_size=1)
        self.to_decoder = nn.Conv3d(hidden_width, encoded_channels,
            kernel_size=1)
        self.kl_weight = kl_weight

    def encode(self, x):
        h = self.encoder(x)
        (mean, log_var) = torch.chunk(self.to_moments(h), 2, dim=1)
        return (mean, log_var)

    def decode(self, z):
        z = self.to_decoder(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        (mean, log_var) = self.encode(input)
        if sample_posterior:
            z = sample_from_standard_normal(mean, log_var)
        else:
            z = mean
        dec = self.decode(z)
        return (dec, mean, log_var, z)

    def _loss(self, batch):
        if isinstance(batch, tuple):
            (x, y) = batch
        else:
            x = batch
            y = x.clone()
        
        # Get all forward pass outputs
        (y_pred, mean, log_var, z) = self.forward(x)

        # Standard loss calculations
        rec_loss = (y - y_pred).abs().mean()
        kl_loss = kl_from_standard_normal(mean, log_var)
        
        total_loss = rec_loss + self.kl_weight * kl_loss
        
        # --- New crop loss logic ---
        # Only apply during training and if weight > 0
        crop_loss = torch.tensor(0.0, device=self.device) # Default to zero

        self.crop_loss_weight = self.config.loss.get('crop_loss_weight', 0)
        self.latent_crop_size = self.config.loss.get('latent_crop_size', 4)
        self.pixel_per_latent = self.config.loss.get('pixel_per_latent', 4)

        if self.crop_loss_weight > 0:
            # 1. Define crop size in pixel space
            input_crop_size = self.latent_crop_size * self.pixel_per_latent
            
            # 2. Get pixel space dimensions and generate a valid random crop start in PIXEL space
            _, _, _, H_in, W_in = y.shape
            y_start_in = torch.randint(0, H_in - input_crop_size + 1, (1,)).item()
            x_start_in = torch.randint(0, W_in - input_crop_size + 1, (1,)).item()
            
            # 3. Get the crop from the original image `y`. This is now guaranteed to be valid.
            y_crop = y[:, :, :, y_start_in : y_start_in + input_crop_size,
                                x_start_in : x_start_in + input_crop_size]

            # 4. Scale the pixel-space coordinates DOWN to latent space. Use integer division.
            y_start_z = y_start_in // self.pixel_per_latent
            x_start_z = x_start_in // self.pixel_per_latent
            
            # 5. Get the corresponding crop from the latent tensor `z`
            z_crop = z[:, :, :, y_start_z : y_start_z + self.latent_crop_size,
                                x_start_z : x_start_z + self.latent_crop_size]

            # 6. Decode the small latent crop
            decoded_crop = self.decode(z_crop)
            
            # Ensure shapes match before calculating loss (optional, for debugging)
            if decoded_crop.shape == y_crop.shape:
                # Calculate the loss on the crop
                crop_loss = (decoded_crop - y_crop).abs().mean()
                
                # Add the weighted crop loss to the total loss
                total_loss = total_loss + self.crop_loss_weight * crop_loss
            else:
                # This can happen if pixel_per_latent is misconfigured. Warn the user.
                print(f"Warning: Decoded crop shape {decoded_crop.shape} != "
                    f"Input crop shape {y_crop.shape}. Skipping crop loss.")
                
        return (total_loss, rec_loss, kl_loss, crop_loss)
    
    def training_step(self, batch, batch_idx):
        (total_loss, rec_loss, kl_loss, crop_loss) = self._loss(batch)
        log_params = {"on_step": True, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log("train/loss", total_loss, **log_params)
        self.log("train/rec_loss", rec_loss, **log_params)
        self.log("train/kl_loss", kl_loss, **log_params)
        if self.crop_loss_weight > 0:
            self.log("train/crop_loss", crop_loss, **log_params)
        
        return total_loss

    @torch.no_grad()
    def val_test_step(self, batch, batch_idx, split="val"):
        (total_loss, rec_loss, kl_loss, crop_loss) = self._loss(batch)
        log_params = {"on_step": False, "on_epoch": True, "prog_bar": True, "sync_dist": True}
        self.log(f"{split}/loss", total_loss, **log_params)
        self.log(f"{split}/rec_loss", rec_loss.mean(), **log_params)
        self.log(f"{split}/kl_loss", kl_loss, **log_params)
        if self.crop_loss_weight > 0:
            self.log("train/crop_loss", crop_loss, **log_params)

    def validation_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="val")

    def test_step(self, batch, batch_idx):
        self.val_test_step(batch, batch_idx, split="test")

    def configure_optimizers(self):
        config = self.config.optimizer

        optimizer = torch.optim.AdamW(self.parameters(), lr=config.lr,
            betas=config.betas, weight_decay=config.weight_decay)
        reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=config.patience, factor=config.factor
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr,
                "monitor": config.monitor,
                "frequency": 1,
            },
        }