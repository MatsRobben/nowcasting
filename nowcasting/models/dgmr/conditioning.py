from typing import List
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from torch.nn.modules.pixelshuffle import PixelUnshuffle

from .layers.blocks import LBlock, DBlock
from .layers import SNConv, Attention


class LatentCondStack(nn.Module):
    def __init__(
        self,
        shape: tuple[int, int, int] = (8, 8, 8),
        output_channels: int = 768,
    ):
        super().__init__()

        self.shape = shape
        self.distribution = normal.Normal(
            loc=torch.Tensor([0.0]), scale=torch.Tensor([1.0])
        )

        self.conv1 = SNConv(shape[0], shape[0], kernel_size=3, padding=1)
        self.lblock1 = LBlock(shape[0], output_channels // 32)
        self.lblock2 = LBlock(output_channels // 32, output_channels // 16)
        self.lblock3 = LBlock(output_channels // 16, output_channels // 4)
        self.mini_attn_block = Attention(output_channels // 4)
        self.lblock4 = LBlock(output_channels // 4, output_channels)

    def forward(self, x):
        # Independent draws from Normal distribution
        z = self.distribution.sample(self.shape)
        # Batch is at end for some reason, reshape
        z = torch.permute(z, (3, 0, 1, 2)).type_as(x)

        # 3x3 Convolution
        z = self.conv1(z)

        # 3 L Blocks to increase number of channels
        z = self.lblock1(z)
        z = self.lblock2(z)
        z = self.lblock3(z)

        # Spatial attention module
        z = self.mini_attn_block(z)

        # L block to increase number of channel to output size
        z = self.lblock4(z)

        return z


class ConditioningStack(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 384,
        num_context_steps: int = 4,
        scale_channels=False,
    ):
        super().__init__()

        self.space2depth = PixelUnshuffle(downscale_factor=2)

        # Set input_channels dynamically based on scale_channels
        first_input = 4 * input_channels
        input_channels = 1 if not scale_channels else input_channels

        # Define the DBlocks
        self.d_blocks = nn.ModuleList(
            [
                DBlock(
                    in_channels=first_input,
                    out_channels=((output_channels // 4) * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=((output_channels // 4) * input_channels)
                    // num_context_steps,
                    out_channels=((output_channels // 2) * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=((output_channels // 2) * input_channels)
                    // num_context_steps,
                    out_channels=(output_channels * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=(output_channels * input_channels) // num_context_steps,
                    out_channels=(output_channels * 2 * input_channels)
                    // num_context_steps,
                ),
            ]
        )

        # Define the convolution layers
        self.convs = nn.ModuleList(
            [
                SNConv(
                    in_channels=(output_channels // 4) * input_channels,
                    out_channels=(output_channels // 8) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=(output_channels // 2) * input_channels,
                    out_channels=(output_channels // 4) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=output_channels * input_channels,
                    out_channels=(output_channels // 2) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=output_channels * 2 * input_channels,
                    out_channels=output_channels * input_channels,
                    kernel_size=3,
                ),
            ]
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.space2depth(x)
        steps = x.size(1)

        # Initialize lists to store scale outputs
        scales = {i: [] for i in range(1, 5)}

        for i in range(steps):
            temp = x[:, i, :, :, :]
            for j, d_block in enumerate(self.d_blocks):
                temp = d_block(temp)
                scales[j + 1].append(temp)

        # Stack the results along the time dimension (T)
        for key in scales:
            scales[key] = torch.stack(scales[key], dim=1)

        # Apply the mixing layer to each scale
        scale_1 = self._mixing_layer(scales[1], self.convs[0])
        scale_2 = self._mixing_layer(scales[2], self.convs[1])
        scale_3 = self._mixing_layer(scales[3], self.convs[2])
        scale_4 = self._mixing_layer(scales[4], self.convs[3])

        return scale_1, scale_2, scale_3, scale_4

    def _mixing_layer(self, inputs, conv_block):
        # Apply rearrange and convolution layer
        stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return F.relu(conv_block(stacked_inputs))


class ConditioningStackVarialbes(nn.Module):
    def __init__(
        self,
        input_channels: int | List[int] = 1,
        output_channels: int = 384,
        num_context_steps: int = 4,
        scale_channels=False,
    ):
        super().__init__()

        self.space2depth = PixelUnshuffle(downscale_factor=2)

        # Set the first import to the DBlock
        if isinstance(input_channels, List):
            first_input = 4 * input_channels[0] + input_channels[1]
        else:
            first_input = 4 * input_channels

        # Set input_channels dynamically based on scale_channels
        input_channels = 1 if not scale_channels else input_channels

        # Define the DBlocks
        self.db_blocks = nn.ModuleList(
            [
                DBlock(
                    in_channels=first_input,
                    out_channels=((output_channels // 4) * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=((output_channels // 4) * input_channels)
                    // num_context_steps,
                    out_channels=((output_channels // 2) * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=((output_channels // 2) * input_channels)
                    // num_context_steps,
                    out_channels=(output_channels * input_channels)
                    // num_context_steps,
                ),
                DBlock(
                    in_channels=(output_channels * input_channels) // num_context_steps,
                    out_channels=(output_channels * 2 * input_channels)
                    // num_context_steps,
                ),
            ]
        )

        # Define the convolution layers
        self.convs = nn.ModuleList(
            [
                SNConv(
                    in_channels=(output_channels // 4) * input_channels,
                    out_channels=(output_channels // 8) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=(output_channels // 2) * input_channels,
                    out_channels=(output_channels // 4) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=output_channels * input_channels,
                    out_channels=(output_channels // 2) * input_channels,
                    kernel_size=3,
                ),
                SNConv(
                    in_channels=output_channels * 2 * input_channels,
                    out_channels=output_channels * input_channels,
                    kernel_size=3,
                ),
            ]
        )

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x1.shape[0] == x2.shape[0], "Batch sizes are not equal"

        x1 = self.space2depth(x1)

        x = torch.concat([x1, x2], dim=2)

        steps = x.size(1)

        # Initialize lists to store scale outputs
        scales = {i: [] for i in range(1, 5)}

        for i in range(steps):
            temp = x[:, i, :, :, :]
            for j, db_block in enumerate(self.db_blocks):
                temp = db_block(temp)
                scales[j + 1].append(temp)

        # Stack the results along the time dimension (T)
        for key in scales:
            scales[key] = torch.stack(scales[key], dim=1)

        # Apply the mixing layer to each scale
        scale_1 = self._mixing_layer(scales[1], self.convs[0])
        scale_2 = self._mixing_layer(scales[2], self.convs[1])
        scale_3 = self._mixing_layer(scales[3], self.convs[2])
        scale_4 = self._mixing_layer(scales[4], self.convs[3])

        return scale_1, scale_2, scale_3, scale_4

    def _mixing_layer(self, inputs, conv_block):
        # Apply rearrange and convolution layer
        stacked_inputs = einops.rearrange(inputs, "b t c h w -> b (c t) h w")
        return F.relu(conv_block(stacked_inputs))
