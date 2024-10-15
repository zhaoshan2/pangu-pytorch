import sys
import os
from typing import List, Tuple, Optional
from torch import Tensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from models.layers import (
    PatchRecoveryPowerAll,
    PatchRecovery_pretrain,
    PowerConv,
    PowerConvWithSigmoid,
)
from era5_data.config import cfg
from models.pangu_model import PanguModel


class PanguPowerPatchRecovery(PanguModel):
    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[torch.device] = None,
    ) -> None:
        super(PanguPowerPatchRecovery, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        # Delete the pangu output layer
        del self._output_layer

        # Replace the output layer with PatchRecovery_transfer
        self._output_weather_layer = PatchRecovery_pretrain(dims[-2])
        self._output_power_layer = PatchRecoveryPowerAll(dims[-2])  # dims[-2] = 384

    def forward(
        self,
        input: Tensor,
        input_surface: Tensor,
        statistics: Tensor,
        maps: Tensor,
        const_h: Tensor,
    ) -> Tuple[Tensor, Tensor]:
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

        # Calculate weather output (just vor visualization)
        output_upper, output_surface = self._output_weather_layer(x, 8, 181, 360)

        # Recover the output fields from patches
        output = self._output_power_layer(x, 8, 181, 360)

        return output, output_surface

    def load_pangu_state_dict(self, device: torch.device) -> None:
        """Get the prepared state dict of the pretrained pangu weights"""
        checkpoint = torch.load(
            cfg.PG.BENCHMARK.PRETRAIN_24_torch, map_location=device, weights_only=False
        )
        pretrained_dict = checkpoint["model"]
        model_dict = self.state_dict()

        # Filter out keys in pretrained_dict & rename to _output_weather_layer
        pretrained_dict = {
            (
                k.replace("_output_layer", "_output_weather_layer")
                if "_output_layer" in k
                else k
            ): v
            for k, v in pretrained_dict.items()
        }

        # Update the model's state_dict except the _output_layer
        model_dict.update(pretrained_dict)

        # Remove keys from model_dict that contain _output_layer
        keys_to_remove = [k for k in model_dict.keys() if "_output_layer" in k]
        for k in keys_to_remove:
            del model_dict[k]

        self.load_state_dict(model_dict)


class PanguPowerConv(PanguModel):
    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[torch.device] = None,
    ) -> None:
        super(PanguPowerConv, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        self._conv_power_layers = PowerConv()

        # Re-Init weights
        super(PanguPowerConv, self).apply(self._init_weights)

    def forward(self, input, input_surface, statistics, maps, const_h):
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
        # output, output_surface = checkpoint.checkpoint(self._output_layer, x, 8, 181, 360)
        output_upper, output_surface = self._output_layer(x, 8, 181, 360)

        output_power = self._conv_power_layers(output_upper, output_surface)

        # Return output_surface for visualization purposes only
        return output_power, output_surface


class PanguPowerConvSigmoid(PanguPowerConv):
    def __init__(
        self,
        depths: List[int] = [2, 6, 6, 2],
        num_heads: List[int] = [6, 12, 12, 6],
        dims: List[int] = [192, 384, 384, 192],
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        device: Optional[torch.device] = None,
    ) -> None:
        super(PanguPowerConvSigmoid, self).__init__(
            depths=depths,
            num_heads=num_heads,
            dims=dims,
            patch_size=patch_size,
            device=device,
        )

        self._conv_power_layers = PowerConvWithSigmoid()


def main():
    pppr = PanguPowerPatchRecovery()
    pppr.load_pangu_state_dict(torch.device("cpu"))


if __name__ == "__main__":
    model = PanguPowerPatchRecovery()
    model.load_pangu_state_dict(torch.device("cpu"))
