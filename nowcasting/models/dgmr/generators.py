import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.pixelshuffle import PixelShuffle
from typing import List
import einops

from .layers import ConvGRU, SNConv
from .layers.blocks import GBlock


class Sampler(nn.Module):
    def __init__(
        self,
        forecast_steps: int = 18,
        latent_channels: int = 768,
        context_channels: int = 384,
        output_channels: int = 1,
    ):
        """
        Sampler from Skillful Nowcasting (https://arxiv.org/pdf/2104.00954.pdf).
        The sampler takes the output from the Latent and Context conditioning stacks
        and creates one stack of ConvGRU layers per future timestep.

        Args:
            forecast_steps (int): Number of forecast steps.
            latent_channels (int): Number of input channels to the lowest ConvGRU layer.
            context_channels (int): Number of context channels.
            output_channels (int): Number of output channels.
        """
        super().__init__()
        self.forecast_steps = forecast_steps

        # Define 4 layers, progressively reducing channels
        self.layers = nn.ModuleList()
        for i in range(4):
            factor = 2**i  # 1, 2, 4, 8
            self.layers.append(
                nn.ModuleDict(
                    {
                        "convGRU": ConvGRU(
                            in_channels=(latent_channels // factor)
                            + (context_channels // factor),
                            out_channels=context_channels // factor,
                            kernel_size=3,
                        ),
                        "gru_conv_1x1": SNConv(
                            in_channels=context_channels // factor,
                            out_channels=latent_channels // factor,
                            kernel_size=1,
                            padding=0,
                        ),
                        "gblock": GBlock(
                            in_channels=latent_channels // factor,
                            out_channels=latent_channels // factor,
                        ),
                        "up_gblock": GBlock(
                            in_channels=latent_channels // factor,
                            out_channels=latent_channels // (factor * 2),
                            upsample=True,
                        ),
                    }
                )
            )

        # Final output processing
        self.bn = nn.BatchNorm2d(latent_channels // 16)
        self.conv_1x1 = SNConv(
            in_channels=latent_channels // 16,
            out_channels=4 * output_channels,
            kernel_size=1,
            padding=0,
        )
        self.depth2space = PixelShuffle(upscale_factor=2)

    def forward(
        self, conditioning_states: List[torch.Tensor], latent_dim: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs sampling from the Skillful Nowcasting model.

        Args:
            conditioning_states (List[torch.Tensor]): Outputs from ContextConditioningStack, ordered largest to smallest spatially.
            latent_dim (torch.Tensor): Output from LatentConditioningStack for input into ConvGRUs.

        Returns:
            torch.Tensor: Forecasted images for future timesteps.
        """
        batch_size = conditioning_states[0].shape[0]
        latent_dim = einops.repeat(
            latent_dim, "b c h w -> (repeat b) c h w", repeat=batch_size
        )
        hidden_states = [latent_dim] * self.forecast_steps

        for i, layer in enumerate(self.layers):
            hidden_states = layer["convGRU"](hidden_states, conditioning_states[3 - i])
            hidden_states = [layer["gru_conv_1x1"](h) for h in hidden_states]
            hidden_states = [layer["gblock"](h) for h in hidden_states]
            hidden_states = [layer["up_gblock"](h) for h in hidden_states]

        # Output processing
        hidden_states = [F.relu(self.bn(h)) for h in hidden_states]
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        hidden_states = [self.depth2space(h) for h in hidden_states]

        return torch.stack(hidden_states, dim=1)


class Generator(torch.nn.Module):
    def __init__(
        self,
        conditioning_stack: torch.nn.Module,
        latent_stack: torch.nn.Module,
        sampler: torch.nn.Module,
    ):
        """
        Wraps the three parts of the generator for simpler calling
        Args:
            conditioning_stack:
            latent_stack:
            sampler:
        """
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.latent_stack = latent_stack
        self.sampler = sampler

    def forward(self, x):
        conditioning_states = self.conditioning_stack(x)
        latent_dim = self.latent_stack(x)
        x = self.sampler(conditioning_states, latent_dim)
        return x


if __name__ == "__main__":
    sampler = Sampler(forecast_steps=18, latent_channels=768, context_channels=384)

    conditioning_states = [
        torch.rand((2, 48, 64, 64)),
        torch.rand((2, 96, 32, 32)),
        torch.rand((2, 192, 16, 16)),
        torch.rand((2, 384, 8, 8)),
    ]

    latent_dim = torch.rand((1, 768, 8, 8))

    print("Output shape (Sampler):", sampler(conditioning_states, latent_dim).shape)
