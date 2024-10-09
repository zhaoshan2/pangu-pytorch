import sys
import os
from typing import List, Tuple, Optional
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.layers import PatchRecovery_power_surface
from models.pangu_model import PanguModel


class PanguPower(PanguModel):
    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[torch.device] = None,
    ) -> None:
        super(PanguPower, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        # Replace the output layer with PatchRecovery_transfer
        self._output_layer = PatchRecovery_power_surface(dims[-2])  # dims[-2] = 384

    def forward(
        self,
        input: Tensor,
        input_surface: Tensor,
        statistics: Tensor,
        maps: Tensor,
        const_h: Tensor,
    ) -> Tensor:
        """Backbone architecture"""
        # Embed the input fields into patches
        # input:(B, N, Z, H, W) ([1, 5, 13, 721, 1440])input_surface(B,N,H,W)([1, 4, 721, 1440])
        # x = checkpoint.checkpoint(self._input_layer, input, input_surface)
        x = self._input_layer(
            input, input_surface, statistics, maps, const_h
        )  # ([1, 521280, 192]) [B, spatial, C]

        # Encoder, composed of two layers
        # Layer 1, shape (8, 360, 181, C), C = 192 as in the original paper

        x = self.layers[0](x, 8, 181, 360)

        # Store the tensor for skip-connection
        skip = x

        # Downsample from (8, 360, 181) to (8, 180, 91)
        x = self.downsample(x, 8, 181, 360)

        x = self.layers[1](x, 8, 91, 180)
        # Decoder, composed of two layers
        # Layer 3, shape (8, 180, 91, 2C), C = 192 as in the original paper
        x = self.layers[2](x, 8, 91, 180)

        # Upsample from (8, 180, 91) to (8, 360, 181)
        x = self.upsample(x)

        # Layer 4, shape (8, 360, 181, 2C), C = 192 as in the original paper
        x = self.layers[3](x, 8, 181, 360)  # ([1, 521280, 192])

        # Skip connect, in last dimension(C from 192 to 384)
        x = torch.cat((skip, x), dim=-1)

        # Recover the output fields from patches
        output = self._output_layer(x, 8, 181, 360)

        return output
