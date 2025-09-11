from typing import List
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import lightning as L
import torchvision.utils

from .conditioning import ConditioningStack, LatentCondStack
from .discriminators import TemporalDiscriminator, SpatialDiscriminator
from .generators import Generator, Sampler
from .losses import loss_hinge_disc, loss_hinge_gen, loss_grid_regularizer

from utils import data_prep


class DGMR(L.LightningModule):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 4,
        input_channels: int = 1,
        output_shape: int = 256,
        latent_channels: int = 768,
        context_channels: int = 384,
        # variable_channels: List[int] = None,
        generation_steps: int = 1,
        scale_channels=False,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        grid_lambda: float = 21.0,
        beta1: float = 0.01,
        beta2: float = 0.99,
        
        use_discriminators = True
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.generation_steps = generation_steps
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.grid_lambda = grid_lambda
        self.beta1 = beta1
        self.beta2 = beta2
        self.use_discriminators = use_discriminators

        self.conditioning_stack = ConditioningStack(
            input_channels=input_channels,
            output_channels=context_channels,
            scale_channels=scale_channels,
        )
        self.latent_stack = LatentCondStack(
            shape=(8, output_shape // 32, output_shape // 32),
            output_channels=latent_channels,
        )

        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=latent_channels,
            context_channels=(
                context_channels * input_channels
                if scale_channels
                else context_channels
            ),
        )
        self.generator = Generator(
            self.conditioning_stack, self.latent_stack, self.sampler
        )

        if self.use_discriminators:
            self.spatial_discriminator = SpatialDiscriminator(
                input_channels=input_channels,
                num_timesteps=8,
            )
            self.temporal_discriminator = TemporalDiscriminator(
                input_channels=input_channels
            )

        self.sample_images = None  # To store sample images for logging

        self.save_hyperparameters()

        self.global_iteration = 0

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        context, future = batch

        # Shape (B, C, T, W, H) -> DGMR Shape (B, T, C, W, H)
        context = context.permute(0, 2, 1, 3, 4).contiguous()
        future = future.permute(0, 2, 1, 3, 4).contiguous()

        self.global_iteration += 1

        if self.use_discriminators:
            opt_g, opt_ds, opt_dt = self.optimizers()
        else:
            opt_g = self.optimizers()


        ######################
        # Optimize Generator #
        ######################
        self.toggle_optimizer(opt_g)
        predictions = self.forward(context)

        # grid_cell_reg = loss_grid_regularizer(predictions, future, clip_value=48)
        grid_cell_reg = F.mse_loss(predictions, future)

        if self.use_discriminators:
            _, spatial_score_generated = self.discriminator_step(
                predictions,
                future,
                self.spatial_discriminator,
            )
            
            generated_sequence = torch.cat([context, predictions], dim=1)
            real_sequence = torch.cat([context, future], dim=1)
            _, temporal_score_generated = self.discriminator_step(
                generated_sequence,
                real_sequence,
                self.temporal_discriminator,
            )

            generated_scores = (spatial_score_generated + temporal_score_generated)

            generator_disc_loss = loss_hinge_gen(generated_scores)
        
            generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg

        else:
            generator_loss = grid_cell_reg

        self.manual_backward(generator_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        if self.use_discriminators:
            ###########################
            # Optimize Discriminators #
            ###########################
            predictions = predictions.detach()

            # Optimize spatial Discriminator
            self.toggle_optimizer(opt_ds)
            spatial_loss = self.discriminator_step(
                predictions,
                future,
                self.spatial_discriminator,
                loss_f=loss_hinge_disc,
                opt=opt_ds,
            )
            self.untoggle_optimizer(opt_ds)

            # Optimize Temporal Discriminator
            self.toggle_optimizer(opt_dt)

            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([context, predictions], dim=1)
            real_sequence = torch.cat([context, future], dim=1)

            temporal_loss = self.discriminator_step(
                generated_sequence,
                real_sequence,
                self.temporal_discriminator,
                loss_f=loss_hinge_disc,
                opt=opt_dt,
            )
            self.untoggle_optimizer(opt_dt)

        if self.use_discriminators:
            self.log_dict(
                {
                    "train/spatial_loss": spatial_loss,
                    "train/temporal_loss": temporal_loss,
                    "train/dis_loss": spatial_loss+temporal_loss,
                    "train/gen_dis_loss": generator_disc_loss,
                    "train/grid_loss": grid_cell_reg,
                    "train/gen_loss": generator_loss
                },
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        else:
            lr = self.trainer.optimizers[0].param_groups[0]['lr']

            self.log_dict(
                {
                    "train/loss": generator_loss,
                    "lr/epoch_end": lr
                },
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

    def validation_step(self, batch, batch_idx):
        context, future = batch

        # Shape (B, C, T, W, H) -> DGMR Shape (B, T, C, W, H)
        context = context.permute(0, 2, 1, 3, 4)
        future = future.permute(0, 2, 1, 3, 4)

        #############
        # Generator #
        #############
        predictions = self.forward(context)

        # grid_cell_reg = loss_grid_regularizer(predictions, future, clip_value=48)
        grid_cell_reg = F.mse_loss(predictions, future)

        if self.use_discriminators:
            _, spatial_score_generated = self.discriminator_step(
                predictions,
                future,
                self.spatial_discriminator,
            )
            
            generated_sequence = torch.cat([context, predictions], dim=1)
            real_sequence = torch.cat([context, future], dim=1)
            _, temporal_score_generated = self.discriminator_step(
                generated_sequence,
                real_sequence,
                self.temporal_discriminator,
            )

            generated_scores = (spatial_score_generated + temporal_score_generated)

            generator_disc_loss = loss_hinge_gen(generated_scores)
            
            generator_loss = self.grid_lambda * grid_cell_reg - generator_disc_loss

        else:
            generator_loss = grid_cell_reg

        if self.use_discriminators:
            ##################
            # Discriminators #
            ##################
            # Spatial Discriminator
            spatial_loss = self.discriminator_step(
                predictions,
                future,
                self.spatial_discriminator,
                loss_f=loss_hinge_disc,
            )

            # Temporal Discriminator

            # Cat along time dimension [B, T, C, H, W]
            generated_sequence = torch.cat([context, predictions], dim=1)
            real_sequence = torch.cat([context, future], dim=1)

            temporal_loss = self.discriminator_step(
                generated_sequence,
                real_sequence,
                self.temporal_discriminator,
                loss_f=loss_hinge_disc,
            )

        # Capture first batch's first sample for logging
        if batch_idx == 0 and self.global_rank == 0:
            # Detach and move to CPU to avoid memory issues
            input_img = context[0].detach().cpu()
            target_img = future[0].detach().cpu()
            pred_img = predictions[0].detach().cpu()
            self.sample_images = (input_img, target_img, pred_img)
        
        if self.use_discriminators:
            self.log_dict(
                {
                    "val/spatial_loss": spatial_loss,
                    "val/temporal_loss": temporal_loss,
                    "val/dis_loss": spatial_loss+temporal_loss,
                    "val/gen_dis_loss": generator_disc_loss,
                    "val/grid_loss": grid_cell_reg,
                    "val/gen_loss": generator_loss
                },
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
        else:
            self.log_dict(
                {
                    "val/loss": generator_loss
                },
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

    def on_validation_epoch_end(self):
        if self.sample_images is not None and self.global_rank == 0:
            input_img, target_img, pred_img = self.sample_images

            # Get dimensions
            T_in, C, H, W = input_img.shape
            if C > 1:
                input_img = input_img[:, :1, :, :]

            T_out = target_img.shape[0]

            # Create visualization grid
            last_input = input_img[-1].unsqueeze(0)  # Last input frame
            
            # Interleave target and prediction frames
            interleaved = []
            for t in range(T_out):
                interleaved.append(target_img[t])
                interleaved.append(pred_img[t])
            interleaved = torch.stack(interleaved)

            # Combine all frames
            all_frames = torch.cat([last_input, interleaved])

            # Create grid (nrow=1 for vertical stacking)
            grid = torchvision.utils.make_grid(all_frames, nrow=1 + 2*T_out)
            
            # Log to TensorBoard
            self.logger.experiment.add_image(
                "forecast_samples",
                grid,
                global_step=self.current_epoch,
            )
            
            # Clear sample cache
            self.sample_images = None

    def discriminator_step(
        self, generated_sequence, real_sequence, discriminator, loss_f=None, opt=None
        ):
        # Cat long batch for the real+generated
        concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

        concatenated_outputs = discriminator(concatenated_inputs)
        score_real, score_generated = torch.split(
            concatenated_outputs,
            [real_sequence.shape[0], generated_sequence.shape[0]],
            dim=0,
        )

        if loss_f is None:
            return score_real, score_generated

        loss = loss_f(score_generated, score_real)

        if opt is not None:  
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()

        return loss

    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2)
        )

        cosine_scheduler = CosineAnnealingLR(opt_g,
                                                T_max=self.trainer.estimated_stepping_batches,
                                                eta_min=0)

        if self.use_discriminators:
            opt_ds = torch.optim.Adam(
                self.spatial_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
            )
            opt_dt = torch.optim.Adam(
                self.temporal_discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2)
            )

            return [opt_g, opt_ds, opt_dt], []  # First optimizers, second schedulers
        else:
            return {'optimizer': opt_g, 'lr_scheduler': cosine_scheduler}