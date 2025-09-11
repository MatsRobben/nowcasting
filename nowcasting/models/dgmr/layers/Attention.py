import torch
import torch.nn as nn
import torch.nn.functional as F

import einops


def attention_einsum(q, k, v):
    """Apply the attention operator to tensors of shape [h, w, c]."""

    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    k = einops.rearrange(k, "h w c -> (h w) c")  # [h, w, c] -> [L, c]
    v = einops.rearrange(v, "h w c -> (h w) c")  # [h, w, c] -> [L, c]

    # Einstein summation corresponding to the query * key operation.
    beta = F.softmax(torch.einsum("hwc, Lc->hwL", q, k), dim=-1)

    # Einstein summation corresponding to the attention * value operation.
    out = torch.einsum("hwL, Lc->hwc", beta, v)
    return out


class Attention(torch.nn.Module):
    """Attention Module"""

    def __init__(self, num_channels: int, ratio_kq=8, ratio_v=8):
        super(Attention, self).__init__()

        # Compute query, key and value using 1x1 convolutions.
        self.query = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )
        self.key = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // ratio_kq,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )
        self.value = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels // ratio_v,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        self.last_conv = nn.Conv2d(
            in_channels=num_channels // ratio_kq,
            out_channels=num_channels,
            kernel_size=(1, 1),
            padding="valid",
            bias=False,
        )

        # Learnable gain parameter
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute query, key and value using 1x1 convolutions.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        # Apply the attention operation.
        out = []
        for b in range(x.shape[0]):
            # Apply to each in batch
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = torch.stack(out, dim=0)
        out = self.gamma * self.last_conv(out)
        # Residual connection.
        return out + x


if __name__ == "__main__":

    attn_block = Attention(192)

    x = torch.rand((1, 192, 8, 8))

    print(attn_block(x).shape)
