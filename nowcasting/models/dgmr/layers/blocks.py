import torch
import torch.nn as nn
import torch.nn.functional as F

from .SNConv import SNConv


# Generalized Residual Generator Block with optional upsampling
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sn_eps=1e-4, upsample=False):
        super(GBlock, self).__init__()
        self.upsample = upsample
        self.conv1x1 = None
        if upsample or in_channels != out_channels:
            self.conv1x1 = SNConv(
                in_channels, out_channels, kernel_size=1, padding=0, sn_eps=sn_eps
            )

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = SNConv(
            in_channels, out_channels, kernel_size=3, padding=1, sn_eps=sn_eps
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = SNConv(
            out_channels, out_channels, kernel_size=3, padding=1, sn_eps=sn_eps
        )

    def forward(self, x):
        # x -> B C T W H

        # Main branch
        h = F.relu(self.bn1(x))
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = F.relu(self.bn2(h))
        h = self.conv2(h)

        # Residual connection
        sc = x
        if self.upsample:
            sc = F.interpolate(sc, scale_factor=2, mode="nearest")
        if self.conv1x1:
            sc = self.conv1x1(sc)

        return h + sc


# Convolutional residual block (D Block) using code from DeepMind
class DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        downsample=True,
        pre_activation=True,
        conv_type="standard",
    ):
        super(DBlock, self).__init__()
        self.downsample = downsample
        self.pre_activation = pre_activation

        # Determine if we are using 2D or 3D convolutions
        if conv_type == "3d":
            pooling = nn.AvgPool3d
        else:
            pooling = nn.AvgPool2d

        self.pooling = pooling(2) if downsample else nn.Identity()

        self.conv1 = SNConv(in_channels, out_channels, kernel_size, conv_type=conv_type)
        self.conv2 = SNConv(
            out_channels, out_channels, kernel_size, conv_type=conv_type
        )

        self.shortcut = None
        if in_channels != out_channels or downsample:
            self.shortcut = nn.Sequential(
                SNConv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=0,
                    conv_type=conv_type,
                ),
                self.pooling,
            )

    def forward(self, x):
        # Main branch
        h = F.relu(x) if self.pre_activation else x
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.pooling(h)

        # Residual connection
        sc = self.shortcut(x) if self.shortcut else x

        return h + sc


# Residual Block for Latent Stack using Spectral Normalization
class LBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(LBlock, self).__init__()

        self.conv1 = SNConv(in_channels, out_channels, kernel_size, padding=1)
        self.conv2 = SNConv(out_channels, out_channels, kernel_size, padding=1)

        # Shortcut connection
        self.shortcut = None
        if in_channels < out_channels:
            self.shortcut = SNConv(
                in_channels, out_channels - in_channels, kernel_size=1, padding=0
            )

    def forward(self, x):
        # Main branch
        h = F.relu(x)
        h = self.conv1(h)
        h = F.relu(h)
        h = self.conv2(h)

        # Residual connection
        if self.shortcut:
            sc = self.shortcut(x)
            sc = torch.cat([x, sc], dim=1)  # Concatenation along channel axis
        else:
            sc = x

        return h + sc


if __name__ == "__main__":
    g_block = GBlock(10, 20)
    g_block_up = GBlock(10, 20, upsample=True)

    d_block = DBlock(10, 20, kernel_size=3, downsample=False)
    d_block_no_pre_act = DBlock(
        10, 20, kernel_size=3, downsample=False, pre_activation=False
    )
    d_block_up = DBlock(10, 20, kernel_size=3, downsample=True)
    d_block_3d = DBlock(10, 20, kernel_size=3, downsample=False, conv_type="3d")
    d_block_3d_up = DBlock(10, 20, kernel_size=3, downsample=True, conv_type="3d")

    l_block = LBlock(10, 20, kernel_size=3)

    x = torch.rand(
        (1, 10, 128, 128)
    )  # Assuming 2D input with (batch, channels, height, width)
    x_3d = torch.rand(
        (1, 10, 3, 128, 128)
    )  # Assuming 3D input with (batch, channels, depth, height, width)

    print("Testing GBlock:")
    print("Output shape (GBlock):", g_block(x).shape)
    print("Output shape (GBlock upsample):", g_block_up(x).shape)

    print("\nTesting DBlock:")
    print("Output shape (DBlock):", d_block(x).shape)
    print("Output shape (DBlock no pre-activation):", d_block_no_pre_act(x).shape)
    print("Output shape (DBlock downsample):", d_block_up(x).shape)

    print("\nTesting DBlock 3D:")
    print("Output shape (DBlock 3D):", d_block_3d(x_3d).shape)
    print("Output shape (DBlock 3D downsample):", d_block_3d_up(x_3d).shape)

    print("\nTesting LBlock:")
    print("Output shape (LBlock):", l_block(x).shape)
