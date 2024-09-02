"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
from pickle import FALSE
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from functools import reduce, lru_cache
from operator import mul
import numpy as np
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_



def get_window_size(x_size, window_size, shift_size=None):
    """
    Adjust the window size and shift size based on the input dimensions.

    Parameters:
    x_size (tuple): 
        The dimensions of the input feature map, specified as (D, H, W).
    window_size (tuple): 
        The desired size of the window for the self-attention mechanism, specified as (D_w, H_w, W_w).
    shift_size (tuple, optional): 
        The desired shift size for the self-attention mechanism, specified as (D_s, H_s, W_s). Default is None.

    Returns:
        If shift_size is not provided, returns a tuple of the adjusted window size (D_w', H_w', W_w').
        
        If shift_size is provided, returns a tuple of the adjusted window size (D_w', H_w', W_w')
        and the adjusted shift size (D_s', H_s', W_s').
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)
    
    
# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """
    Compute the attention mask for a 3D windowed self-attention mechanism.

    Args:
        D (int): 
            Depth of the input feature map.
        H (int): 
            Height of the input feature map.
        W (int): 
            Width of the input feature map.
        window_size (tuple): 
            Size of the window for the self-attention mechanism, specified as (D_w, H_w, W_w).
        shift_size (tuple): 
            Shift size for the self-attention mechanism, specified as (D_s, H_s, W_s).
        device (torch.device): 
            The device on which the tensor should be allocated.

    Returns:
        torch.Tensor: 
        The computed attention mask of shape 
        (nW, window_size[0] * window_size[1] * window_size[2], window_size[0] * window_size[1] * window_size[2]), 
        where nW is the number of windows.
    """
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask



def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def batchconv(in_channels, out_channels, sz):
    conv_layer = nn.Conv3d
    batch_norm = nn.BatchNorm3d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        nn.ReLU(inplace=True),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )


def batchconv0(in_channels, out_channels, sz):
    conv_layer = nn.Conv3d
    batch_norm = nn.BatchNorm3d
    return nn.Sequential(
        batch_norm(in_channels, eps=1e-5, momentum=0.05),
        conv_layer(in_channels, out_channels, sz, padding=sz // 2),
    )



class PatchEmbed3D(nn.Module):
    """ 3D image input (In this case multiplex images) to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(1, 4, 4), embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(10, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function.
        The image to embedd always has shape [B, 1, D, H, W]"""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))
        
        
        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample

    def forward(self, x, block_num):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')

        # calculate the potentially necessary padding
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        
        v1, k1, q1, v2, k2, q2 = None, None, None, None, None, None
        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x, v1, k1, q1 = blk(x, attn_mask, None, None, None)
            else:
                x, v2, k2, q2 = blk(x, attn_mask, None, None, None)

        x = x.reshape(B, D, H, W, -1)

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x, v1, k1, q1, v2, k2, q2


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward_part1(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder):
        """The first part of the forward pass for the transformer model, handling cyclic shifts, window partitioning, and multi-head self-attention.

        Parameters:
        x (torch.Tensor): 
            The input tensor with shape (B, D, H, W, C), where B is the batch size, D is the depth, H is the height, W is the width, and C is the number of channels.
        mask_matrix (torch.Tensor): 
            The mask matrix for attention, used to mask out certain positions.
        prev_v (torch.Tensor): 
            The previous value tensor for cross-attention, used in decoder layers.
        prev_k (torch.Tensor): 
            The previous key tensor for cross-attention, used in decoder layers.
        prev_q (torch.Tensor): 
            The previous query tensor for cross-attention, used in decoder layers.
        is_decoder (bool): 
            A flag indicating whether the current layer is part of the decoder.

        Returns:
        tuple: A tuple containing the following elements:
            - torch.Tensor: The output tensor after self-attention and reversing cyclic shift, with the same shape as the input tensor.
            - torch.Tensor or None: The cross-attention output tensor if `is_decoder` is True, otherwise None.
            - torch.Tensor: The value tensor after self-attention.
            - torch.Tensor: The key tensor after self-attention.
            - torch.Tensor: The query tensor after self-attention.
    """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        
        x = self.norm1(x)
        
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        
        attn_windows, cross_attn_windows, v, k, q = self.attn(x_windows, mask=attn_mask, prev_v=prev_v, prev_k=prev_k,
                                                              prev_q=prev_q, is_decoder=is_decoder)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        # W-MSA/SW-MSA
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        x2 = None
        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        if cross_attn_windows is not None:
            # merge windows
            cross_attn_windows = cross_attn_windows.view(-1, *(window_size + (C,)))
            cross_shifted_x = window_reverse(cross_attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
            # reverse cyclic shift
            if any(i > 0 for i in shift_size):
                x2 = torch.roll(cross_shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                x2 = cross_shifted_x

            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x2 = x2[:, :D, :H, :W, :].contiguous()
        return x, x2, v, k, q

    def forward_part2(self, x):
        """The second part of the forward pass for the transformer model, applying normalization, the MLP and dropout.

            Parameters:
            x (torch.Tensor): 
                The input tensor with shape (B, D, H, W, C), where B is the batch size, D is the depth, H is the height, W is the width, and C is the number of channels.

            Returns:
            torch.Tensor: 
                The output tensor after applying layer normalization, MLP, and dropout.
        """
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward_part3(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix, prev_v, prev_k, prev_q, is_decoder=False):
        """Forward function for the transformer model, combining cyclic shift, multi-head self-attention, residual connections, and MLP layers.

        Parameters:
            x (torch.Tensor): 
                Input feature tensor with shape (B, D, H, W, C), where B is the batch size, D is the depth, H is the height, W is the width, and C is the number of channels.
            mask_matrix (torch.Tensor): 
                Attention mask for cyclic shift.
            prev_v (torch.Tensor): 
                Previous value tensor for cross-attention, used in decoder layers.
            prev_k (torch.Tensor): 
                Previous key tensor for cross-attention, used in decoder layers.
            prev_q (torch.Tensor): 
                Previous query tensor for cross-attention, used in decoder layers.
            is_decoder (bool, optional): 
                A flag indicating whether the current layer is part of the decoder. Default is False.

        Returns:
        tuple: A tuple containing the following elements:
            - torch.Tensor: The output tensor after applying all transformations and residual connections.
            - torch.Tensor or None: The value tensor after self-attention, if applicable.
            - torch.Tensor or None: The key tensor after self-attention, if applicable.
            - torch.Tensor or None: The query tensor after self-attention, if applicable.
        """

        alpha = 0.3
        shortcut = x
        x2, v, k, q = None, None, None, None
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x, x2, v, k, q = self.forward_part1(x = x, 
                                                mask_matrix = mask_matrix, 
                                                prev_v = prev_v, 
                                                prev_k = prev_k, 
                                                prev_q = prev_q, 
                                                is_decoder = is_decoder)

        # residual connection
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
            
        if x2 is not None:
            x2 = shortcut + self.drop_path(x2)
            if self.use_checkpoint:
                x2 = x2 + checkpoint.checkpoint(self.forward_part2, x2)
            else:
                x2 = x2 + self.forward_part2(x2)

            FPE = PositionalEncoding3D(x.shape[4])

            x = torch.add((1-alpha)*x, alpha*x2) + self.forward_part3(FPE(x))
        
        return x, v, k, q


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb_x = torch.zeros_like(emb_x)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:, :, :, :self.channels] = emb_x
        emb[:, :, :, self.channels:2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels:] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)



class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  #, indexing="ij" 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, prev_v=None, prev_k=None, prev_q=None, is_decoder=False):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        
        q = q * self.scale
        
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)  # B_, nH, N, N
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        x2 = None

        if is_decoder:
            q = q * self.scale
            attn2 = q @ prev_k.transpose(-2, -1)
            attn2 = attn2 + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn2 = attn2.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn2 = attn2.view(-1, self.num_heads, N, N)
                attn2 = self.softmax(attn2)
            else:
                attn2 = self.softmax(attn2)
            attn2 = self.attn_drop(attn2)

            x2 = (attn2 @ prev_v).transpose(1, 2).reshape(B_, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)

        return x, x2, v, k, q


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """Forward function for the transformer model, handling input padding, feature downsampling, normalization, and reduction.

        Parameters:
        x (torch.Tensor): 
            Input feature tensor with shape (B, D, H, W, C), where B is the batch size, D is the depth, H is the height, W is the width, and C is the number of channels.

        Returns:
        torch.Tensor: 
            The output tensor after downsampling, normalization, and reduction, with shape (B, D, H/2, W/2, 4*C).
        """
        B, D, H, W, C = x.shape

        
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        
        return x

class make_style(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.avg_pool = F.avg_pool3d

    def forward(self, x0):
        style = self.avg_pool(x0, kernel_size=x0.shape[2:])
        style = self.flatten(style)
        style = style / torch.sum(style**2, axis=1, keepdim=True)**.5
        return style


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)
        self.dim_scale = dim_scale

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.expand(x)
        B, L, C = x.shape
        # # assert L == D * H * W, "input feature has wrong size"
        # x = x.view(B, D, H*2, W*2, C//4)
        # # x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // 4)

        # x = self.norm(x)
        # x = x.permute(0, 4, 1, 2, 3)
        
        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale**2)

        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x
    
    
class FinalPatchExpand_X4(nn.Module):
    """ A module to expand the spatial dimensions of the input tensor by a factor of `dim_scale` using a linear layer.

    Args:
        dim (int): The number of input channels.
        dim_scale (tuple of int, optional): The factor by which to expand the spatial dimensions. Default is (4, 4, 4).
        norm_layer (callable, optional): The normalization layer to use. Default is nn.LayerNorm.
    """
    def __init__(self, dim, dim_scale=4, output_dim = 3, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = dim_scale
        self.expand = nn.ConvTranspose3d(48, output_dim, kernel_size=(4, 4, 4), stride=(4, 4, 4), bias=False)
        self.norm = norm_layer(output_dim)

    def forward(self, x):
        """ Forward pass for the FinalPatchExpand_X4 module.

        Parameters:
        x (torch.Tensor): 
            Input tensor with shape (B, D, H, W, C), where B is the batch size, D is the depth,
            H is the height, W is the width, and C is the number of channels.

        Returns:
        torch.Tensor: 
            The output tensor with shape (B, C, H, W, D),  expanded spatial dimensions and normalized channels.
        """
        x = self.expand(x).permute(0, 2, 3, 4, 1)
        x = self.norm(x) # B, D, H, W, C
        return x.permute(0, 4, 1, 2, 3) # B, C, H, W, D

    

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size tuple(int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size=(7, 7, 7),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.upsample = upsample

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            for i in range(depth)])

        # patch merging layer
        # if upsample is not None:
        #     self.upsample = PatchExpand(dim=dim, dim_scale=2, norm_layer=norm_layer)
        # else:
        #     self.upsample = None

    def forward(self, x, prev_x, prev_v1, prev_k1, prev_q1, prev_v2, prev_k2, prev_q2, style):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for idx, blk in enumerate(self.blocks):
            if idx % 2 == 0:
                x, _, _, _ = blk(x, attn_mask, prev_v1, prev_k1, prev_q1, True)
            else:
                x, _, _, _ = blk(x, attn_mask, prev_v2, prev_k2, prev_q2, True)
        if self.upsample is not None:
            x = x.permute(0, 4, 1, 2, 3)
            
            x = self.upsample(x)
            # x = self.upsample(x, prev_x, style)

        return x

class resup(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz):
        super().__init__()
        self.concatenation = False
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0",
                             batchconv(in_channels, out_channels, sz))
        self.conv.add_module(
            "conv_1",
            batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module(
            "conv_2",
            batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.conv.add_module(
            "conv_3",
            batchconvstyle(out_channels, out_channels, style_channels, sz))
        self.proj = batchconv0(in_channels, out_channels, 1)

    def forward(self, x, y, style, mkldnn=False):
        x = self.proj(x) + self.conv[1](style, self.conv[0](x), y=y, mkldnn=mkldnn)
        x = x + self.conv[3](style, self.conv[2](style, x, mkldnn=mkldnn),
                             mkldnn=mkldnn)
        return x


class batchconvstyle(nn.Module):

    def __init__(self, in_channels, out_channels, style_channels, sz):
        super().__init__()
        self.concatenation = False
        self.conv = batchconv(in_channels, out_channels, sz)
        self.full = nn.Linear(style_channels, out_channels)

    def forward(self, style, x, mkldnn=False, y=None):
        if y is not None:
            x = x + y
        feat = self.full(style)
        for k in range(len(x.shape[2:])):
            feat = feat.unsqueeze(-1)
        if mkldnn:
            x = x.to_dense()
            y = (x + feat).to_mkldnn()
        else:
            y = x + feat
        y = self.conv(y)
        return y

class AutoPool(nn.Module):
    def __init__(self, pool_dim: int = 1) -> None:
        super(AutoPool, self).__init__()
        self.pool_dim: int = pool_dim
        self.softmax: nn.modules.Module = nn.Softmax(dim=pool_dim)
        self.register_parameter("alpha", nn.Parameter(torch.ones(1), requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.softmax(torch.mul(x, self.alpha))
        out = torch.sum(torch.mul(x, weight), dim=self.pool_dim)
        return out


class Transformer(nn.Module):
    """
    CPnet is the Transformer based Cellpose neural network model used for cell segmentation and image restoration.

    Args:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image.
        mkldnn (bool, optional): Whether to use MKL-DNN acceleration. Defaults to False.
        conv_3D (bool, optional): Whether to use 3D convolution. Defaults to False.
        max_pool (bool, optional): Whether to use max pooling. Defaults to True.
        diam_mean (float, optional): Mean diameter of the cells. Defaults to 30.0.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        
    Attributes:
        nbase (list): List of integers representing the number of channels in each layer of the downsample path.
        nout (int): Number of output channels.
        sz (int): Size of the input image (Always has to be 1).
        residual_on (bool): Whether to use residual connections.
        style_on (bool): Whether to use style transfer.
        concatenation (bool): Whether to use concatenation.
        conv_3D (bool): Whether to use 3D convolution.
        mkldnn (bool): Whether to use MKL-DNN acceleration.
        downsample (nn.Module): Downsample blocks of the network.
        upsample (nn.Module): Upsample blocks of the network.
        make_style (nn.Module): Style module, avgpool's over all spatial positions.
        output (nn.Module): Output module - batchconv layer.
        diam_mean (nn.Parameter): Parameter representing the mean diameter to which the cells are rescaled to during training.
        diam_labels (nn.Parameter): Parameter representing the mean diameter of the cells in the training set (before rescaling).
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        
    """     
            
    def __init__(self, 
                 nbase = [96, 192, 384, 768], 
                 nout = 3, 
                 sz = 1,
                 mkldnn=False, 
                 max_pool=False,
                 diam_mean=30.,
                 embed_dim=48, 
                 
                #  depths=[2, 2, 2, 1],
                #  num_heads=[3, 6, 12, 24],
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                #  num_heads=[3, 6, 12],
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.1,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 
                 use_checkpoint = False,
                 mode = "normal",
                 encoder="mit_b5", 
                 encoder_weights=None, 
                 decoder="MAnet",
                 patch_norm=True):
        super().__init__()
        self.nbase = [96, 192, 384, 768]
        self.mkldnn = mkldnn
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.nout = nout
        self.mode = mode
        patch_size=(4, 4, 4)
        self.patch_size=patch_size
        self.embed_dim=embed_dim
        
        self.word_embedding = nn.Embedding(57, 10)
        
        norm_layer=nn.LayerNorm
        
        # Patch Eembedding
        self.patch_embed = PatchEmbed3D(patch_size=patch_size, 
                                        embed_dim=embed_dim, 
                        norm_layer=norm_layer if patch_norm else None
                        )
    
        # Downsampling
        if max_pool:
            self.maxpool = nn.MaxPool3d(2, stride=2)
        else:
            self.maxpool = nn.AvgPool3d(2, stride=2)
        self.down = nn.Sequential()
        channel_dims = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dim = self.embed_dim
        for n, depth, heads in zip(range(len(depths)), depths, num_heads):
            if n < (len(depths)-2):
                self.down.add_module("Basic_layer_%d" % n,
                                    BasicLayer(
                                        dim=dim,
                                        depth=depth,
                                        num_heads=heads,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop_rate,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:n]):sum(depths[:n + 1])],
                                        drop_path_rate=drop_path_rate,
                                        norm_layer=norm_layer,
                                        downsample=PatchMerging(dim=dim, 
                                                                norm_layer=norm_layer),
                                        use_checkpoint=use_checkpoint))
                dim=int(embed_dim * 2 ** (n+1))
            else:
                self.down.add_module("Basic_layer_%d" % n,
                                    BasicLayer(
                                        dim=dim,
                                        depth=depth,
                                        num_heads=heads,
                                        window_size=window_size,
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias,
                                        qk_scale=qk_scale,
                                        drop=drop_rate,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(depths[:n]):sum(depths[:n + 1])],
                                        drop_path_rate=drop_path_rate,
                                        norm_layer=norm_layer,
                                        downsample = None,
                                        use_checkpoint=use_checkpoint))
            channel_dims.append(dim)
        self.norm = norm_layer(dim)
    
        # Style Vector
        self.make_style = make_style()
        
        
        # Upsampling
        # reverse depth and num heads for the decoder, add 0 for the init step
        depths = list(reversed(depths[:-1]))
        num_heads = list(reversed(num_heads[:-1]))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.up = nn.Sequential()
        self.concat_back_dim = nn.Sequential()
        self.concat_back_style = nn.Sequential()
        channel_dims = channel_dims[:-2][::-1]
        channel_dims.append(embed_dim)
        # self.batchconv = nn.ModuleList()
        # self.batchconv_style = nn.ModuleList()
        for n, depth, heads in zip(range(len(depths)), depths, num_heads):
            
            if n == 0:
                self.up.add_module("Patch_expand_up_%d" % (n),
                                #    resup(in_channels = int(embed_dim * 2 ** len_depths - n), 
                                #    out_channels = int(embed_dim * 2 ** len_depths - n)//2, 
                                #    style_channels = int(embed_dim * 2 ** len_depths - n), 
                                #    sz = 1)
                                   PatchExpand(dim=channel_dims[n], 
                                               dim_scale=2, 
                                               norm_layer=norm_layer)
                )
                
                self.concat_back_style.add_module("Concat_style_%d" % (n),
                                                nn.Linear(channel_dims[n],
                                        channel_dims[n]//2,
                                        bias=False)
                                                )
            else:
                self.concat_back_dim.add_module("Concat_residual_%d" % (n),
                                                nn.Linear(2 * channel_dims[n],
                                        channel_dims[n],
                                        bias=False)
                                                )
                if not n >= len(depths)-1:
                    self.concat_back_style.add_module("Concat_style_%d" % (n),
                                                    nn.Linear(channel_dims[n],
                                            channel_dims[n]//2,
                                            bias=False)
                                                    )
                
                self.up.add_module("Basic_layer_up_%d" % (n),
                                    BasicLayer_up(
                    dim=channel_dims[n],
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:n]):sum(
                        depths[:n+1])],
                    norm_layer=norm_layer,
                    # upsample=resup(in_channels = channel_dims[n], 
                    #                out_channels = channel_dims[n]//2, 
                    #                style_channels = channel_dims[n]//2, 
                    #                sz = 1) if (n < len(depths) - 1) else None,
                    upsample=PatchExpand(dim=channel_dims[n], # int(embed_dim * 2 ** len_depths - n), 
                                                dim_scale=2, 
                                                norm_layer=norm_layer) if (n < len(depths)-1) else None,
                    use_checkpoint=use_checkpoint)
                )
        self.norm_up = norm_layer(embed_dim)
        
        # Output
        
        self.up4x = FinalPatchExpand_X4(dim=embed_dim, output_dim = 3, dim_scale=patch_size)
        # self.outConv = nn.Linear(in_features=embed_dim, out_features=3, bias=False)
        # # self.DeepSet2 = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, 5, 5), stride = 1, padding = (0, 2, 2), bias=False, )
        self.pooling = AutoPool(2)
        # self.DeepSet2 = nn.Sequential(
        #     nn.BatchNorm3d(num_features = 3, eps=1e-5, momentum=0.05),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(1, 5, 5), stride = 1, padding = (0, 2, 2), bias=False)
        # )
        
        self.norm_upx4 = norm_layer(embed_dim)
        
        # Model parameters
        self.diam_mean = nn.Parameter(data=torch.ones(1) * diam_mean,
                                      requires_grad=False)
        self.diam_labels = nn.Parameter(data=torch.ones(1) * diam_mean,
                                        requires_grad=False)
        self.model_string = "Transformer"
        
        
    @property
    def device(self):
        """
        Get the device of the model.

        Returns:
            torch.device: The device of the model.
        """
        return next(self.parameters()).device

    def downsample(self, x):
        """Apply dropout to the input, then pass it through a series of downsampling 
        layers while storing intermediate results and cross-attention parameters.

        Parameters:
        x (torch.Tensor): 
            Input tensor with shape (B, C, D, H, W), where B is the batch size, 
            C is the number of channels, D is the depth, H is the height, and W is the width.

        Returns:
        tuple: A tuple containing the following elements:
            - torch.Tensor: The output tensor after downsampling and normalization, with shape (B, C, D, H, W).
            - list of torch.Tensor: List of tensors representing the residual connections after each downsampling layer.
            - list of torch.Tensor: List of value tensors from the first set of cross-attention layers.
            - list of torch.Tensor: List of key tensors from the first set of cross-attention layers.
            - list of torch.Tensor: List of query tensors from the first set of cross-attention layers.
            - list of torch.Tensor: List of value tensors from the second set of cross-attention layers.
            - list of torch.Tensor: List of key tensors from the second set of cross-attention layers.
            - list of torch.Tensor: List of query tensors from the second set of cross-attention layers.
    """
        # apply Dropout
        x = self.pos_drop(x)
        
        # initialise the lists storing residual connections and cross attention
        # parameters
        x_downsample = []
        v_values_1 = []
        k_values_1 = []
        q_values_1 = []
        v_values_2 = []
        k_values_2 = []
        q_values_2 = []

        for i, layer in enumerate(self.down):
            if not i >= len(self.down)-1:
                x_downsample.append(x)
            x, v1, k1, q1, v2, k2, q2 = layer(x, i)
            if not i >= len(self.down)-1:
                v_values_1.append(v1)
                k_values_1.append(k1)
                q_values_1.append(q1)
                v_values_2.append(v2)
                k_values_2.append(k2)
                q_values_2.append(q2)

        x = rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2


    def upsample(self, style, x, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2,
                            q_values_2):
        
        for inx, layer_up in enumerate(self.up):
                if inx == 0:
                    B, C, D, H, W = x.shape
                    # add style vector
                    # concat style 
                    x += style.view(B, C, 1, 1, 1)
                    x = layer_up(x)
                    style = self.concat_back_style[inx](style)
                else:
                            
                    B, C, D, H, W = x.shape
                    # add style vector
                    # concat style 
                    x += style.view(B, C, 1, 1, 1)
                    x = torch.cat([x, x_downsample[::-1][inx]], 1)
                    x = x.flatten(2).transpose(1, 2)
                    # here i have to -1 because there is no concat back dim 
                    # in the first step
                    x = self.concat_back_dim[inx-1](x)
                    x = x.view(B, C, D, H, W)
                    
                    if inx < len(self.up)-1:
                        style = self.concat_back_style[inx](style)
                    x = layer_up(x, 
                                    prev_x = x_downsample[::-1][inx],
                                    prev_v1 = v_values_1[::-1][inx], 
                                    prev_k1 = k_values_1[::-1][inx], 
                                    prev_q1 = q_values_1[::-1][inx], 
                                    prev_v2 = v_values_2[::-1][inx],
                                    prev_k2 = k_values_2[::-1][inx], 
                                    prev_q2 = q_values_2[::-1][inx],
                                    style = style)
        x = self.norm_up(x)
        return x.permute(0, 4, 1, 2, 3) # (B, C, D, H, W)
        
    
    def find_largest_abs_positions(self, input_tensor):
        # Find the indices of the maximum absolute values along the Depth dimension
        _, max_abs_indices = torch.max(torch.abs(input_tensor), dim=2, keepdim=True)

        # Gather the corresponding values from the original tensor
        largest_abs_values = torch.gather(input_tensor, 2, max_abs_indices)

        # return torch.squeeze(largest_abs_values)
        return largest_abs_values
        
    def output(self, X):
        X = X.permute(0, 2, 3, 4, 1) # (B, D, H, W, C)
        X = self.norm_upx4(X)
        X = X.permute(0, 4, 1, 2, 3) # B, D, C, H, W 
        X = self.up4x(X)
        B, C, D, H, W = X.shape
        # X = X.permute(0, 2, 3, 4, 1)
        # X = self.outConv(X).permute(0, 4, 1, 2, 3)
        X= nn.LeakyReLU()(X)
        if self.mode == "normal":
            X = self.pooling(X).unsqueeze(2)
            X = self.DeepSet2(X).squeeze(2)
        else:
            X = self.pooling(X.permute(0, 2, 1, 3, 4))
        # X = self.find_largest_abs_positions(X)
        # X = torch.mean(X, 2, keepdim=True)
        return X

    def forward(self, X, channels):
        """
        Forward pass of the CPnet model.

        Args:
            data (torch.Tensor): Input data.

        Returns:
            tuple: A tuple containing the output tensor, style tensor, and downsampled tensors.
        """
        if self.mkldnn:
            X = X.to_mkldnn()
            channels = channels.to_mkldnn()
        X = torch.unsqueeze(X, 1)
        min_val = []
        channels = self.word_embedding(channels).permute(2, 0, 1)
        max_val = []
        for n in range(X.shape[0]):
            for img in range(X.shape[2]):
                min_val.append(torch.min(X[n, :, img]).detach().cpu().numpy())
                max_val.append(torch.max(X[n, :, img]).detach().cpu().numpy())
        # Embedd the channels
        X = X.permute(3, 4, 1, 0, 2)
        X = channels + X
        X = X.permute(3, 2, 4, 0, 1)
        
        # Patch Embed
        X = self.patch_embed(X)
        out = self.downsample(X)
        (X, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2, q_values_2) = out
        # Calculate style as global average pool of each feature map
        if self.mkldnn:
            style = self.make_style(x_downsample[-1].to_dense())
        else:
            style = self.make_style(x_downsample[-1])
        # Upsample
        
        X = self.upsample(style, X, x_downsample, v_values_1, k_values_1, q_values_1, v_values_2, k_values_2,
                            q_values_2)
    
        X = self.output(X)
        
        if self.mkldnn:
            x_downsample = [t0.to_dense() for t0 in x_downsample]
            X = X.to_dense()
        return X, style, x_downsample        
        

    def save_model(self, filename):
        """
        Save the model to a file.

        Args:
            filename (str): The path to the file where the model will be saved.
        """
        torch.save(self.state_dict(), filename)

    def load_model(self, filename, device=None):
        """
        Load the model from a file.

        Args:
            filename (str): The path to the file where the model is saved.
            device (torch.device, optional): The device to load the model on. Defaults to None.
        """
        if (device is not None) and (device.type != "cpu"):
            state_dict = torch.load(filename, map_location=device)
        else:
            self.__init__(mkldnn = self.mkldnn, max_pool = True, diam_mean = self.diam_mean)#.to(self.device)
            state_dict = torch.load(filename, map_location=torch.device("cpu"))
        
        self.load_state_dict(
            dict([(name, param) for name, param in state_dict.items()]),
            strict=False)
