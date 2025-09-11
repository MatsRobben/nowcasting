import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class SNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        conv_type="standard",
        stride=1,
        padding=1,
        sn_eps=1e-4,
        use_bias=True,
        use_mobilenet=False,
    ):
        super(SNConv, self).__init__()

        # Determine convolution type
        if conv_type == "standard":
            conv_layer = nn.Conv2d
        elif conv_type == "3d":
            conv_layer = nn.Conv3d
        else:
            raise ValueError(f"{conv_type} is not a recognized Conv method")

        self.use_mobilenet = use_mobilenet

        if use_mobilenet:
            # Depthwise Separable Convolution (MobileNetV1) - Supports 2D & 3D
            self.depthwise = spectral_norm(
                conv_layer(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride,
                    padding,
                    groups=in_channels,
                    bias=use_bias,
                ),
                eps=sn_eps,
            )
            self.pointwise = spectral_norm(
                conv_layer(in_channels, out_channels, kernel_size=1, bias=use_bias),
                eps=sn_eps,
            )

            # Possibly add features from MobileNetV2 and V3
        else:
            self.conv = spectral_norm(
                conv_layer(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=use_bias,
                ),
                eps=sn_eps,
            )

    def forward(self, x):
        if self.use_mobilenet:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x


if __name__ == "__main__":
    conv = SNConv(10, 20, kernel_size=3)

    x = torch.rand((1, 10, 128, 128))

    print(conv(x).shape)
