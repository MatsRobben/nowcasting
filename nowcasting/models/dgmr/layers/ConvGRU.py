import torch
import torch.nn.functional as F

from .SNConv import SNConv


class ConvGRUCell(torch.nn.Module):
    """A ConvGRU implementation."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size=3, sn_eps=0.0001
    ):
        """Constructor.

        Args:
          kernel_size: kernel size of the convolutions. Default: 3.
          sn_eps: constant for spectral normalization. Default: 1e-4.
        """
        super().__init__()

        self.read_gate_conv = SNConv(
            in_channels, out_channels, kernel_size, sn_eps=sn_eps
        )

        self.update_gate_conv = SNConv(
            in_channels, out_channels, kernel_size, sn_eps=sn_eps
        )

        self.output_conv = SNConv(in_channels, out_channels, kernel_size, sn_eps=sn_eps)

    def forward(self, x, prev_state):
        """
        ConvGRU forward, returning the current+new state

        Args:
            x: Input tensor
            prev_state: Previous state

        Returns:
            New tensor plus the new state
        """
        # Concatenate the inputs and previous state along the channel axis.
        xh = torch.cat([x, prev_state], dim=1)

        # Read gate of the GRU.
        read_gate = F.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = F.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = torch.cat([x, read_gate * prev_state], dim=1)

        # Gate the cell and state / outputs.
        c = F.relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out

        return out, new_state


class ConvGRU(torch.nn.Module):
    """ConvGRU Cell wrapper to replace tf.static_rnn in TF implementation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        sn_eps=0.0001,
    ):
        super().__init__()
        self.cell = ConvGRUCell(in_channels, out_channels, kernel_size, sn_eps)

    def forward(self, x: torch.Tensor, hidden_state=None) -> torch.Tensor:
        outputs = []
        for step in range(len(x)):
            # Compute current timestep
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)
        # Stack outputs to return as tensor
        outputs = torch.stack(outputs, dim=0)
        return outputs


if __name__ == "__main__":

    forecast_steps = 18
    latent_dim = torch.rand((1, 768, 8, 8))

    hidden_states = [latent_dim] * forecast_steps
    conditioning_state = torch.rand((1, 384, 8, 8))

    convGRU = ConvGRU(768 + 384, 384, kernel_size=3)

    print(convGRU(hidden_states, conditioning_state).shape)
