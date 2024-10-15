import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch import nn
import torch
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from collections import OrderedDict


class PatchEmbedding_pretrain(nn.Module):
    def __init__(self, patch_size, dim):
        super(PatchEmbedding_pretrain, self).__init__()
        """Patch embedding operation"""
        # Here we use convolution to partition data into cubes
        self.conv = nn.Conv1d(
            in_channels=192, out_channels=dim, kernel_size=1, stride=1
        )
        self.conv_surface = nn.Conv1d(
            in_channels=112, out_channels=dim, kernel_size=1, stride=1
        )
        self.window_size = (2, 6, 12)  # Z,H,W
        # self.Pad2D = nn.ConstantPad2d((0, 0, 0, 3), 0)
        # self.Pad3D = nn.ConstantPad3d((0, 0, 0, 3, 0, 1),0)

    def check_image_size_2d(self, x):
        _, _, h, w = x.size()
        # mod_pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[
        #     1
        # ]  # 6
        mod_pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[
            2
        ]
        x = F.pad(x, (0, mod_pad_w, 0, 3), "constant")
        # x = self.Pad2D(x)
        return x

    def check_image_size_3d(self, x):
        _, _, d, h, w = x.size()
        # mod_pad_d = (self.window_size[0] - h % self.window_size[0]) % self.window_size[
        #     0
        # ]
        # mod_pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[
        #     1
        # ]
        # mod_pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[
        #     2
        # ]
        x = F.pad(x, (0, 0, 0, 3, 0, 1), "constant")
        # x = self.Pad3D(x)
        return x

    def forward(self, input, input_surface, statistics, maps, const_h):
        # input:(B, N, Z, H, W) input_surface(B,N,H,W)
        # Zero-pad the input
        self.surface_mean, self.surface_std, self.upper_mean, self.upper_std = (
            statistics[0],
            statistics[1],
            statistics[2],
            statistics[3],
        )
        self.constant_masks = maps

        input_surface = input_surface.reshape(
            input_surface.shape[0],
            input_surface.shape[1],
            1,
            input_surface.shape[-2],
            input_surface.shape[-1],
        )
        input_surface = torch.permute(
            input_surface, (0, 2, 3, 4, 1)
        )  # [1,1,721,1440,4]
        input_surface = (input_surface - self.surface_mean) / self.surface_std

        input_surface = torch.permute(
            input_surface, (0, 4, 1, 2, 3)
        )  # [1,4 1,721,1440]
        input_surface = input_surface.view(
            input_surface.shape[0],
            input_surface.shape[1],
            input_surface.shape[-2],
            input_surface.shape[-1],
        )  # [1,4,721,1440]

        input_surface = self.check_image_size_2d(input_surface)

        # input_surface has shape [batch_size, 4, 724, 1440]
        batch_size = input_surface.shape[0]

        # Repeat self.constant_masks along the batch dimension
        # self.constant_masks has shape [1, 4, 724, 1440]
        constant_masks_repeated = self.constant_masks.repeat(batch_size, 1, 1, 1)

        # Concatenate input_surface and constant_masks_repeated along dimension 1
        input_surface = torch.cat(
            (input_surface, constant_masks_repeated), dim=1, out=None
        )

        input_surface = input_surface.view(
            input_surface.shape[0],
            input_surface.shape[1],
            input_surface.shape[-2] // 4,
            4,
            input_surface.shape[-1] // 4,
            4,
        )  # (1,7,181,4,360,4)
        input_surface = torch.permute(
            input_surface, (0, 1, 3, 5, 2, 4)
        )  # (1,7,4,4,181,360)
        input_surface = input_surface.reshape(
            input_surface.shape[0],
            input_surface.shape[1] * input_surface.shape[2] * input_surface.shape[3],
            -1,
        )
        input_surface = self.conv_surface(input_surface)  # (1,192,65160)
        input_surface = input_surface.view(
            input_surface.shape[0], input_surface.shape[1], 1, 181, 360
        )

        input = input.reshape(
            input.shape[0],
            input.shape[1],
            1,
            input.shape[2],
            input.shape[-2],
            input.shape[-1],
        )
        input = torch.permute(input, (0, 2, 3, 4, 5, 1))  # [1,1,13,721,1440,5]
        input = torch.flip(input, [2])  # [1,1,13,721,1440,5]
        input = (input - self.upper_mean) / self.upper_std  # [1,1,13,721,1440,5]
        input = torch.permute(input, (0, 5, 1, 2, 3, 4))  # [1,5,1,13,721,1440]
        input = torch.flip(input, [3])
        input = torch.cat((input, const_h), dim=1)  # [1,6,1,13,721,1440]
        input = input.reshape(
            input.shape[0],
            input.shape[1],
            input.shape[3],
            input.shape[-2],
            input.shape[-1],
        )  # [1,6,13,721,1440]
        input = self.check_image_size_3d(input)
        # related to patch size
        input = input.reshape(
            input.shape[0],
            input.shape[1],
            input.shape[2] // 2,
            2,
            input.shape[-2] // 4,
            4,
            input.shape[-1] // 4,
            4,
        )
        input = input.permute(0, 1, 3, 5, 7, 2, 4, 6)  # (1,6,2,4,4, 7, 181,360)
        input = input.reshape(
            input.shape[0],
            input.shape[1] * input.shape[2] * input.shape[3] * input.shape[4],
            -1,
        )
        input = self.conv(input)  # (1,192,456120)
        input = input.view(input.shape[0], input.shape[1], 7, 181, 360)

        x = torch.cat((input_surface, input), dim=2)
        x = x.view(x.shape[0], x.shape[1], -1)  # (1, 192521280)

        x = torch.permute(x, (0, 2, 1))  # ->([1, 521280, 192]) [B, spatial, C]
        return x


class EarthSpecificLayer(nn.Module):
    def __init__(self, depth, dim, drop_path_ratio_list, heads, use_checkpoint, device):
        super(EarthSpecificLayer, self).__init__()
        self.device = device
        """Basic layer of our network, contains 2 or 6 blocks"""
        self.depth = depth

        block_list = OrderedDict()
        for i_layer in range(depth):
            block_list["EarthSpecificBlock{}".format(i_layer)] = EarthSpecificBlock(
                dim, drop_path_ratio_list[i_layer], heads, device=self.device
            )
        self.blocks = nn.Sequential(block_list)
        self.use_checkpoint = use_checkpoint
        self.device = device

    def forward(self, x, Z, H, W):
        # # Roll the input every two blocks
        i = -1
        for blk in self.blocks:
            i += 1
            if self.use_checkpoint:
                if i % 2 == 0:
                    x = checkpoint.checkpoint(
                        blk, x, Z, H, W, False, use_reentrant=True
                    )
                else:
                    x = checkpoint.checkpoint(blk, x, Z, H, W, True, use_reentrant=True)
            else:
                if i % 2 == 0:
                    x = blk(x, Z, H, W, roll=False)
                else:
                    x = blk(x, Z, H, W, roll=True)
        return x


class EarthSpecificBlock(nn.Module):
    def __init__(self, dim, drop_path_ratio, heads, device):
        super(EarthSpecificBlock, self).__init__()
        """
    3D transformer block with Earth-Specific bias and window attention,
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    The major difference is that we expand the dimensions to 3 and replace the relative position bias with Earth-Specific bias.
    """
        self.device = device
        # Define the window size of the neural network
        self.window_size = (2, 6, 12)

        # Initialize serveral operations
        self.drop_path = (
            DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = Mlp(dim, 0)
        self.attention = EarthAttention3D(
            dim, heads, 0, self.window_size, device=self.device
        )
        self.padding_front, self.padding_back = 0, 5
        # self.pad3D = nn.ConstantPad3d((0, 0, 0, 0, self.padding_front,  self.padding_back), 0)
        if dim == 192:
            input_shape = [8, 186]
        elif dim == 384:
            input_shape = [8, 96]
        self.type_of_windows = (input_shape[0] // self.window_size[0]) * (
            input_shape[1] // self.window_size[1]
        )  # (8//2*186//6=124) (8//2*96//6=124=64)

    def gen_mask(self, x):
        img_mask = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3], 1)).to(
            self.device
        )  # ? CHECK change to x.shape?  # 1 Z H W 1
        mB, mZ, mH, mW, mC = img_mask.shape
        # 1x8x96x180x1
        cnt = 0
        z_slices = (
            slice(0, -self.window_size[0]),
            slice(-self.window_size[0], -self.window_size[0] // 2),
            slice(-self.window_size[0] // 2, None),
        )
        h_slices = (
            slice(0, -self.window_size[1]),
            slice(self.window_size[1], -self.window_size[1] // 2),
            slice(-self.window_size[1] // 2, None),
        )
        for z in z_slices:
            for h in h_slices:
                img_mask[:, z, h, :, :] = cnt
                cnt += 1
        img_mask = img_mask.reshape(
            1,
            mZ // self.window_size[0],
            self.window_size[0],
            mH // self.window_size[1],
            self.window_size[1],
            mW // self.window_size[2],
            self.window_size[2],
            1,
        )
        img_mask = torch.permute(img_mask, (0, 5, 1, 3, 2, 4, 6, 7))
        mask_windows = img_mask.reshape(
            -1,
            self.type_of_windows,
            self.window_size[0],
            self.window_size[1],
            self.window_size[2],
            1,
        )

        mask_windows = mask_windows.view(
            -1,
            self.type_of_windows,
            self.window_size[0] * self.window_size[1] * self.window_size[2],
        )  # （15，64，144）

        attn_mask = mask_windows.unsqueeze(2) - mask_windows.unsqueeze(3)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, Z, H, W, roll):
        # Save the shortcut for skip-connection
        shortcut = x  # torch.Size([1, 521280, 192]) -- ([1, 131040, 384])

        # Reshape input to three dimensions to calculate window attention
        x = x.view(
            x.shape[0], Z, H, W, x.shape[2]
        )  # torch.Size([1, 8, 181, 360, 192]) - ([1, 8, 91, 180, 384])

        # Zero-pad input if needed
        # x = self.pad3D(x) #torch.Size([1, 8, 186, 360, 192]) - [1, 8, 96, 180, 384]
        x = F.pad(x, (0, 0, 0, 0, self.padding_front, self.padding_back), "constant")

        ori_shape = x.shape

        # Store the shape of the input for restoration

        if roll:
            # Roll x for half of the window for 3 dimensions
            x = torch.roll(
                x,
                shifts=[
                    -self.window_size[0] // 2,
                    -self.window_size[1] // 2,
                    -self.window_size[2] // 2,
                ],
                dims=(1, 2, 3),
            )
            # Generate mask of attention masks
            # If two pixels are not adjacent, then mask the attention between them
            # Your can set the matrix element to -1000 when it is not adjacent, then add it to the attention
            """
      To do: generate mask
      """
            mask = self.gen_mask(x)
            # mask = None

        else:
            # e.g., zero matrix when you add mask to attention
            mask = None

        # Reorganize data to calculate window attention
        x_window = x.view(
            x.shape[0],
            x.shape[1] // self.window_size[0],
            self.window_size[0],
            x.shape[2] // self.window_size[1],
            self.window_size[1],
            x.shape[3] // self.window_size[2],
            self.window_size[2],
            x.shape[-1],
        )

        x_window = torch.permute(
            x_window, (0, 5, 1, 3, 2, 4, 6, 7)
        )  # 1,30,4,31,2,6,12,192
        x_window = x_window.reshape(
            x_window.shape[1],
            x_window.shape[2] * x_window.shape[3],
            x_window.shape[4],
            x_window.shape[5],
            x_window.shape[6],
            x_window.shape[7],
        )  # nW*B, window_size*window_size,
        # x_window (30,124,2,6,12,192)
        x_window = x_window.contiguous().view(
            x_window.shape[0],
            x_window.shape[1],
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            x_window.shape[-1],
        )
        # Apply 3D window attention with Earth-Specific bias
        attn_windows = self.attention(
            x_window, mask
        )  # ？x_window:([30, 124, 144, 192]) -[15, 64, 144, 384])

        # Reorganize data to original shapes
        # x_shifted = attn_windows.view(-1, Z // self.window_size[0], H // self.window_size[1] + 1, W //self.window_size[2], self.window_size[0], self.window_size[1], self.window_size[2], x_window.shape[-1])
        x_shifted = attn_windows.view(
            1,
            attn_windows.shape[0],
            Z // self.window_size[0],
            H // self.window_size[1] + 1,
            self.window_size[0],
            self.window_size[1],
            self.window_size[2],
            -1,
        )  # 1,30,4,31,2,6,12,192
        # x_window torch.Size([30, 124, 144, 192]) -[15, 64, 144, 384])
        # x_shifted = torch.permute(x_shifted, (0, 1, 4, 2, 5, 3, 6, 7))
        x_shifted = torch.permute(x_shifted, (0, 2, 4, 3, 5, 1, 6, 7))
        # torch.Size([1, 4, 2, 31, 6, 30, 12, 192]) - ([1, 4, 2, 16, 6, 15, 12, 384])
        x_shifted = x_shifted.contiguous().view(
            ori_shape
        )  # ([1, 8, 186, 360, 192])-([1, 8, 96, 180, 384])

        if roll:
            # Roll x back for half of the window
            x = torch.roll(
                x_shifted,
                shifts=[
                    self.window_size[0] // 2,
                    self.window_size[1] // 2,
                    self.window_size[2] // 2,
                ],
                dims=(1, 2, 3),
            )
        else:
            x = x_shifted

        # Crop the zero-padding
        # Crop the tensor using the specified slices
        depth_slice = slice(self.padding_front, x.shape[2] - self.padding_back)
        x = x[:, :, depth_slice, :, :]

        # The resulting 'cropped_tensor' will have the cropped contents of the original tensor
        # Reshape the tensor back to the input shape
        x = x.contiguous().view(
            x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4]
        )

        # Main calculation stages
        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.linear(x)))

        return x


class Mlp(nn.Module):
    def __init__(self, dim, dropout_rate):
        super().__init__()
        """MLP layers, same as most vision transformer architectures."""
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        return x


class EarthAttention3D(nn.Module):
    def __init__(self, dim, heads, dropout_rate, window_size, device):
        super(EarthAttention3D, self).__init__()
        """
    3D window attention with the Earth-Specific bias,
    see https://github.com/microsoft/Swin-Transformer for the official implementation of 2D window attention.
    """
        # Initialize several operations
        self.device = device
        self.linear1 = nn.Linear(
            dim, dim * 3, bias=True
        )  # self.qkv = nn.Linear(dim, dim * 3)?
        self.linear2 = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

        # Store several attributes
        self.head_number = heads
        self.dim = dim
        self.scale = (dim // heads) ** -0.5  # 0.176776
        self.window_size = window_size

        # input_shape is current shape of the self.forward function
        # You can run your code to record it, modify the code and rerun it
        # Record the number of different window types
        # input_shape = [8,186]
        # input_shape = [8,96]

        if self.dim == 192:
            input_shape = [8, 186]
        elif self.dim == 384:
            input_shape = [8, 96]
        self.type_of_windows = (input_shape[0] // window_size[0]) * (
            input_shape[1] // window_size[1]
        )  # (8//2*186//6=124) (8//2*96//6=124=64)

        # For each type of window, we will construct a set of parameters according to the paper
        # self.earth_specific_bias = torch.zeros((2 * window_size[2] - 1) * window_size[1] * window_size[1] * window_size[0] * window_size[0], self.type_of_windows, heads).to(device)
        self.earth_specific_bias = torch.zeros(
            1,
            self.type_of_windows,
            heads,
            window_size[0] * window_size[1] * window_size[2],
            window_size[0] * window_size[1] * window_size[2],
        ).to(self.device)
        #
        # Making these tensors to be learnable parameters
        self.earth_specific_bias = nn.Parameter(self.earth_specific_bias)

        # Initialize the tensors using Truncated normal distribution
        trunc_normal_(self.earth_specific_bias, std=0.02)

        # Construct position index to reuse self.earth_specific_bias
        self._construct_index()

    def _construct_index(self):
        """This function construct the position index to reuse symmetrical parameters of the position bias"""
        # Index in the pressure level of query matrix
        coords_zi = torch.arange(self.window_size[0])
        coords_zi = coords_zi.to(self.device)
        # Index in the pressure level of key matrix
        coords_zj = -torch.arange(self.window_size[0]) * self.window_size[0]
        coords_zj = coords_zj.to(self.device)

        # Index in the latitude of query matrix
        coords_hi = torch.arange(self.window_size[1]).to(self.device)
        coords_hi = coords_hi.to(self.device)
        # Index in the latitude of key matrix
        coords_hj = -torch.arange(self.window_size[1]) * self.window_size[1]
        coords_hj = coords_hj.to(self.device)

        # Index in the longitude of the key-value pair
        coords_w = torch.arange(self.window_size[2])
        coords_w = coords_w.to(self.device)

        # Change the order of the index to calculate the index in total
        coords_1 = torch.stack(
            torch.meshgrid(coords_zi, coords_hi, coords_w, indexing="ij")
        )
        coords_2 = torch.stack(
            torch.meshgrid(coords_zj, coords_hj, coords_w, indexing="ij")
        )
        coords_flatten_1 = torch.flatten(coords_1, start_dim=1)
        coords_flatten_2 = torch.flatten(coords_2, start_dim=1)
        coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]
        coords = torch.permute(coords, (1, 2, 0))

        # Shift the index for each dimension to start from 0
        coords[:, :, 2] += self.window_size[2] - 1
        coords[:, :, 1] *= 2 * self.window_size[2] - 1
        coords[:, :, 0] *= (
            (2 * self.window_size[2] - 1) * self.window_size[1] * self.window_size[1]
        )

        # Sum up the indexes in three dimensions
        self.position_index = torch.sum(coords, dim=-1)

        # Flatten the position index to facilitate further indexing
        self.position_index = torch.flatten(self.position_index)  # ok

    def forward(self, x, mask):
        # Record the original shape of the input
        original_shape = (
            x.shape
        )  # ([30, 124, 144, 576]) swinir B_, N, C = x.shape#([30, 124, 144, 192])
        # Linear layer to create query, key and value

        x = self.linear1(x)  # ([30, 124, 144, 576])

        # reshape the data to calculate multi-head attention
        qkv = torch.reshape(
            x,
            shape=(
                x.shape[0],
                x.shape[1],
                x.shape[2],
                3,
                self.head_number,
                self.dim // self.head_number,
            ),
        )
        # 30，124，144，3，6，32
        qkv = torch.permute(
            qkv, (3, 0, 1, 4, 2, 5)
        )  # qkv torch.Size([3, 30, 124, 6, 144, 32])
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Scale the attention
        query = query * self.scale

        # Calculated the attention, a learnable bias is added to fix the nonuniformity of the grid.
        # attention = query @ key.T # @ denotes matrix multiplication
        attention = query @ key.transpose(
            -2, -1
        )  # attention torch.Size([30, 124, 6, 144, 144]) --  [15, 64, 12, 144, 144])
        # print("self.position_index", self.position_index.shape) #torch.Size([20736])
        # print("self.earth_specific_bias", self.earth_specific_bias.shape) #([3312, 124, 6])

        # self.earth_specific_bias is a set of neural network parameters to optimize.
        # commnet from here
        # EarthSpecificBias = self.earth_specific_bias[self.position_index]
        # # # print("EarthSpecificBias", EarthSpecificBias.shape) #([20736, 124, 6])
        #
        # # # Reshape the learnable bias to the same shape as the attention matrix
        # EarthSpecificBias = EarthSpecificBias.view(self.window_size[0]*self.window_size[1]*self.window_size[2], self.window_size[0]*self.window_size[1]*self.window_size[2], self.type_of_windows, self.head_number)
        # # #torch.Size([144, 144, 124, 6])
        # EarthSpecificBias = torch.permute(EarthSpecificBias, (2, 3, 0, 1))#torch.Size([124,6,144, 144])
        # EarthSpecificBias = EarthSpecificBias.unsqueeze(0)# ->[1,124,6,144, 144]
        EarthSpecificBias = self.earth_specific_bias

        # Add the Earth-Specific bias to the attention matrix
        attention = attention + EarthSpecificBias  # ([30, 124, 6, 144, 144])
        # attention = attention + EarthSpecificBias#([30, 124, 6, 144, 144])

        # Mask the attention between non-adjacent pixels, e.g., simply add -100 to the masked element.
        if mask is not None:
            nW = mask.shape[0]  # mask: 15x64x144x144
            attention = attention.view(
                1,
                nW,
                self.type_of_windows,
                self.head_number,
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
            ) + mask.unsqueeze(2).unsqueeze(0)  # 1x15x64x1x144x144
            attention = attention.reshape(
                nW,
                self.type_of_windows,
                self.head_number,
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2],
            )
            attention = self.softmax(attention)
        else:
            attention = self.softmax(attention)
        attention = self.dropout(attention)

        # Calculated the tensor after spatial mixing.
        x = attention @ value  # ([30, 124, 6, 144, 32])

        # Reshape tensor to the original shape
        x = torch.permute(x, (0, 1, 3, 2, 4))  # ([30, 124, 144, 6, 32])

        x = torch.reshape(x, shape=original_shape)

        # Linear layer to post-process operated tensor
        x = self.linear2(x)
        x = self.dropout(x)  # torch.Size([30, 124, 144, 192])

        return x


class DownSample(nn.Module):  # can check siwnir's up and downsample
    def __init__(self, dim):
        super().__init__()
        """Down-sampling operation"""
        # A linear function and a layer normalization
        self.linear = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        # self.Pad3D = nn.ConstantPad3d((0, 0, 0, 0, 0, 1),0)

    def forward(self, x, Z, H, W):
        # print(x.shape)

        # Reshape x to three dimensions for downsampling
        x = x.view(x.shape[0], Z, H, W, x.shape[-1])  # ([1, 8, 181, 360, 192])

        # Padding the input to facilitate downsampling

        # x = self.Pad3D(x)#[1, 8, 182, 360, 192])
        x = F.pad(x, (0, 0, 0, 0, 0, 1), "constant")

        # Reorganize x to reduce the resolution: simply change the order and downsample from (8, 360, 182) to (8, 180, 91)
        # Reshape x to facilitate downsampling
        Z, H, W = x.shape[1], x.shape[2], x.shape[3]
        x = x.view(x.shape[0], Z, H // 2, 2, W // 2, 2, x.shape[-1])
        # x = x.view(x.shape[0], Z, (H+1)//2 , 2, W//2, 2, x.shape[-1])
        # Change the order of x
        x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6))
        # Reshape to get a tensor of resolution (8, 180, 91)
        x = x.reshape(x.shape[0], Z * (H // 2) * (W // 2), 4 * x.shape[-1])

        # Call the layer normalization
        x = self.norm(x)

        # Decrease the channels of the data to reduce computation cost
        x = self.linear(x)  # torch.Size([1, 131040, 384])

        return x


class UpSample(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        """Up-sampling operation"""
        # Linear layers without bias to increase channels of the data
        self.linear1 = nn.Linear(input_dim, output_dim * 4, bias=False)

        # Linear layers without bias to mix the data up
        self.linear2 = nn.Linear(output_dim, output_dim, bias=False)

        # Normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # Call the linear functions to increase channels of the data
        x = self.linear1(x)

        # Reorganize x to increase the resolution: simply change the order and upsample from (8, 180, 91) to (8, 360, 182)
        # Reshape x to facilitate upsampling.
        x = x.view(x.shape[0], 8, 91, 180, 2, 2, x.shape[-1] // 4)
        # Change the order of x
        x = torch.permute(x, (0, 1, 2, 4, 3, 5, 6))  # ([1, 8, 91, 2, 180, 2, 192])
        # Reshape to get Tensor with a resolution of (8, 360, 182)
        x = x.contiguous().view(x.shape[0], 8, 182, 360, x.shape[-1])  #

        # Crop the output to the input shape of the network
        # x = Crop3D(x)
        depth_slice = slice(0, x.shape[2] - 1)
        x = x[:, :, depth_slice, :, :]  # ([1, 8, 181, 360, 192])

        # Reshape x back
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[-1])

        # Call the layer normalization
        x = self.norm(x)

        # Mixup normalized tensors
        x = self.linear2(x)
        return x


class PatchRecovery_pretrain(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """Patch recovery operation"""
        # Hear we use two transposed convolutions to recover data
        self.patch_size = (2, 4, 4)
        self.dim = dim  # 384
        # Bekomme 384 Enigabebilder, Projiziere runter auf 160 Aufgabebilder
        self.conv = nn.Conv1d(
            in_channels=dim, out_channels=160, kernel_size=1, stride=1
        )
        self.conv_surface = nn.Conv1d(
            in_channels=dim, out_channels=64, kernel_size=1, stride=1
        )

    def forward(self, x, Z, H, W):  # x: [1, 521280, 384], Z: 8, H: 181, W: 360
        # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
        # Reshape x back to three dimensions
        x = torch.permute(x, (0, 2, 1))  # [1, 384, 521280]
        x = x.view(x.shape[0], x.shape[1], Z, H, W)  # [1, 384, 8, 181, 360]

        # Slice out atmospheric data
        output = x[:, :, 1:, :, :]  # [1, 384, 7, 181, 360]

        # Flatten
        output = output.view(output.shape[0], output.shape[1], -1)  # [1, 384, 456120]

        # Apply upper convolution
        output = self.conv(output)  # [1, 160, 456120]

        # Recover [724, 1440] shape
        output = output.reshape(
            output.shape[0],
            5,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            Z - 1,
            H,
            W,
        )  # [1, 5, 2, 4, 4, 7, 181, 360]
        output = torch.permute(
            output, (0, 1, 5, 2, 6, 3, 7, 4)
        )  # [1, 5, 7, 2, 181, 4, 360, 4]
        output = output.reshape(
            output.shape[0], 5, 14, 724, 1440
        )  # [1, 5, 14, 724, 1440]

        # Remove padding
        depth_slice = slice(0, output.shape[-3] - 1)
        height_slice = slice(0, output.shape[-2] - 3)
        output = output[:, :, depth_slice, height_slice, :]  # [1, 5, 13, 721, 1440]
        output = output.view(
            output.shape[0], 5, 1, 13, 721, 1440
        )  # [1, 5, 1, 13, 721, 1440]
        output = output.view(output.shape[0], 5, 13, 721, 1440)  # [1, 5, 13, 721, 1440]

        # Slice out surface data
        output_surface = x[:, :, 0, :, :]  # [1, 384, 181, 360]

        # Flatten
        output_surface = output_surface.view(
            output_surface.shape[0], self.dim, -1
        )  # [1, 384, 65160]

        # Apply surface convolution
        output_surface = self.conv_surface(output_surface)  # [1, 64, 65160]

        # Recover [724, 1440] shape
        output_surface = output_surface.view(
            output_surface.shape[0], 4, self.patch_size[1], self.patch_size[2], H, W
        )  # [1, 4, 4, 4, 181, 360]
        output_surface = torch.permute(
            output_surface, (0, 1, 4, 2, 5, 3)
        )  # [1, 4, 181, 4, 360, 4]
        output_surface = output_surface.reshape(
            output_surface.shape[0], 4, 724, 1440
        )  # [1, 4, 724, 1440]

        # Remove apdding
        output_surface = output_surface[:, :, height_slice, :]  # [1, 4, 721, 1440]
        output_surface = output_surface.view(
            output_surface.shape[0], 4, 1, 721, 1440
        )  # [1, 4, 1, 721, 1440]
        # output_surface = output_surface * self.surface_std + self.surface_mean
        output_surface = output_surface.view(
            output_surface.shape[0], 4, 721, 1440
        )  # [1, 4, 721, 1440]

        return output, output_surface


class PatchRecoveryPowerSurface(nn.Module):
    """Patch recovery operation for wind power generation (leading to output dimensions of [1, 721, 1440]).
    Processing both surface and atmospheric in one step (with one convolution) is not done, since for atmospheric variables
    the data has 5 vars on 13 pressure levels, for surface, there are 4 vars on 1 level (ground).
    """

    def __init__(self, dim):
        super().__init__()
        """Patch recovery operation"""
        self.patch_size = (2, 4, 4)
        self.dim = dim  # 384
        # Receive 384 Input images, project down to 16 output images
        self.conv = nn.Conv1d(in_channels=dim, out_channels=16, kernel_size=1, stride=1)

    def forward(self, x, Z, H, W):  # x: [1, 521280, 384], Z: 8, H: 181, W: 360
        # Reshape x back to three dimensions
        x = torch.permute(x, (0, 2, 1))  # [1, 384, 521280]
        x = x.view(x.shape[0], x.shape[1], Z, H, W)  # [1, 384, 8, 181, 360]

        # Slice out surface data
        output = x[:, :, 0, :, :]  # [1, 384, 181, 360]

        # Flatten
        output = output.view(output.shape[0], self.dim, -1)  # [1, 384, 65160]

        # Apply convolution
        output = self.conv(output)  # [1, 16, 65160]

        # Recover [724, 1440] shape
        output = output.view(
            output.shape[0], 1, self.patch_size[1], self.patch_size[2], H, W
        )  # [1, 1, 4, 4, 181, 360]
        output = torch.permute(output, (0, 1, 4, 2, 5, 3))  # [1, 1, 181, 4, 360, 4]
        output = output.reshape(output.shape[0], 1, 724, 1440)  # [1, 1, 724, 1440]

        # Remove padding
        height_slice = slice(0, 724 - 3)
        output = output[:, :, height_slice, :]  # [1, 1, 721, 1440]
        output = output.view(output.shape[0], 1, 1, 721, 1440)  # [1, 1, 1, 721, 1440]
        # output_surface = output_surface * self.surface_std + self.surface_mean
        output = output.view(output.shape[0], 1, 721, 1440)  # [1, 1, 721, 1440]
        return output


class PatchRecoveryPowerAll(nn.Module):
    def __init__(self, dim):
        super().__init__()
        """Patch recovery operation"""
        # Hear we use two transposed convolutions to recover data
        self.patch_size = (2, 4, 4)
        self.dim = dim  # 384

        self.conv = nn.Conv1d(in_channels=dim, out_channels=32, kernel_size=1, stride=1)

    def forward(self, x, Z, H, W):  # x: [1, 521280, 384], Z: 8, H: 181, W: 360
        # The inverse operation of the patch embedding operation, patch_size = (2, 4, 4) as in the original paper
        # Reshape x back to three dimensions
        x = torch.permute(x, (0, 2, 1))  # [1, 384, 521280]
        x = x.view(x.shape[0], x.shape[1], Z, H, W)  # [1, 384, 8, 181, 360]

        # Flatten
        output = x.view(x.shape[0], x.shape[1], -1)  # [1, 384, 521280]

        # Apply upper convolution
        output = self.conv(output)  # [1, 32, 521280]

        # Recover [724, 1440] shape
        output = output.reshape(
            output.shape[0],
            1,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
            Z,
            H,
            W,
        )  # [1, 1, 2, 4, 4, 8, 181, 360]
        output = torch.permute(
            output, (0, 1, 5, 2, 6, 3, 7, 4)
        )  # [1, 1, 8, 2, 181, 4, 360, 4]
        output = output.reshape(
            output.shape[0], 1, 16, 724, 1440
        )  # [1, 1, 16, 724, 1440]

        # Remove padding
        depth_slice = slice(0, output.shape[-3] - 1)
        height_slice = slice(0, output.shape[-2] - 3)
        output = output[:, :, depth_slice, height_slice, :]  # [1, 1, 15, 721, 1440]
        output = output.view(
            output.shape[0], 1, 1, 15, 721, 1440
        )  # [1, 1, 1, 15, 721, 1440]
        output = output.view(output.shape[0], 1, 15, 721, 1440)  # [1, 1, 15, 721, 1440]

        # Sum the output along the pressure levels
        output = torch.sum(output, dim=2)  # [1, 1, 721, 1440]

        return output


class PowerConv(nn.Module):
    """
    A (series of) convolutional layer(s) to finetune on the power prediction task.
    Attributes:
        conv_layers (nn.Sequential): A sequential container of convolutional layers,
                                     batch normalization layers, and ReLU activation functions.
    """

    def __init__(
        self,
        in_channels=28,
        out_channels_list=[64, 128, 64, 1],
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        """
        Initializes the PowerPanguConv class with the given parameters.
        Args:
            in_channels (int): Number of input channels for the first convolutional layer. Default is 28. (u and v for 13 pressure levels, u10m, v10m)
            out_channels_list (list): List of output channels for each convolutional layer. Default is [1]. Could also be e.g., [64, 32, 16, 1].
            kernel_size (int or tuple): Size of the convolving kernel. Default is 1.
            stride (int or tuple): Stride of the convolution. Default is 1.
            padding (int or tuple): Zero-padding added to both sides of the input. Default is 1.
        """
        super().__init__()

        print("PowerConv initialized with the following parameters:")
        print(f"in_channels: {in_channels}")
        print(f"out_channels_list: {out_channels_list}")
        print(f"kernel_size: {kernel_size}")
        print(f"stride: {stride}")
        print(f"padding: {padding}")

        layers = []
        current_in_channels = in_channels

        # Build multiple convolutional layers
        for out_channels in out_channels_list:
            layers.append(
                nn.Conv2d(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    padding_mode="circular",
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))  # Add batch normalization
            layers.append(nn.ReLU())  # Add ReLU activation function
            current_in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)  # Combine layers sequentially

    def forward(self, output_upper, output_surface):
        # Slice out wind variables
        output_upper = output_upper[:, -2:, :, :, :]
        output_surface = output_surface[:, 1:3, :, :]

        # Reshape output_upper from [1, 2, 13, 721, 1440] to [1, 26, 721, 1440]
        batch_size = output_upper.size(0)  # Extract the batch size
        output_upper = output_upper.reshape(
            batch_size,
            output_upper.size(1) * output_upper.size(2),
            *output_upper.shape[3:],
        )

        # Concatenate with output_surface [1, 2, 721, 1440] along the channel dimension
        concatenated_output = torch.cat([output_upper, output_surface], dim=1)

        # Apply the sequential layers
        output = self.conv_layers(concatenated_output)
        return output


class PowerConvWithSigmoid(PowerConv):
    """Replaces the last layer of PowerConv with a Sigmoid layer to better reflect the output range of wind power generation which is between [0, 1]"""

    def __init__(self):
        super().__init__()

        # Replace the last ReLU layer with a Sigmoid layer
        if isinstance(self.conv_layers[-1], nn.ReLU):
            self.conv_layers[-1] = nn.Sigmoid()
        else:
            raise ValueError(
                "The last layer is not a ReLU layer and cannot be replaced with Sigmoid."
            )


def main():
    # Initialize random tensors
    x = torch.randn(1, 521280, 384)
    Z = 8
    H = 181
    W = 360

    # Instantiate the PatchRecoveryAll class
    model = PatchRecoveryPowerAll(dim=384)

    # Call the forward method
    output = model.forward(x, Z, H, W)

    # Print the output
    print(output)


if __name__ == "__main__":
    main()
