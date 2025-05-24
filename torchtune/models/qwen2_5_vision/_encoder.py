# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Callable

import torch
from torch import nn

from torchtune.modules import Fp32LayerNorm
from torchtune.modules.transformer import _get_clones
from torchtune.modules.fusion import register_fusion_module


class Qwen2_5_VisionMLP(nn.Module):
    """
    MLP for Qwen 2.5 Vision.
    """

    def __init__(
        self,
        *,
        gate_proj: nn.Module,
        down_proj: nn.Module,
        up_proj: Optional[nn.Module] = None,
        activation: nn.Module = nn.SiLU(),
    ):
        super().__init__()
        self.gate_proj = gate_proj
        self.down_proj = down_proj
        self.up_proj = up_proj
        self.act_fn = activation

    def forward(self, x: torch.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down
    
class Qwen2_5_VisionPatchEmbed(nn.Module):
    """
    Patch embedding for Qwen 2.5 Vision. ZL: check if this is correct.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Todo ZL: check if this is correct
        return self.conv(x)

       
class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    """
    Rotary embedding for Qwen 2.5 Vision. ZL: check if this is correct.
    """

    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Qwen2_5_VLVisionBlock(nn.Module):
    """
    Vision block for Qwen 2.5 Vision. ZL: check if this is correct.
    """

    def __init__(
        self,
        config: nn.Module,
        attn_implementation: Callable,
    ) -> None:
        super().__init__()
        self.attn = attn_implementation(config)
        self.mlp = Qwen2_5_VisionMLP(
            gate_proj=nn.Linear(config.hidden_size, config.intermediate_size),
            down_proj=nn.Linear(config.intermediate_size, config.hidden_size),
            up_proj=nn.Linear(config.hidden_size, config.intermediate_size),
            activation=nn.SiLU(),
        )
        self.norm1 = Fp32LayerNorm(config.hidden_size)
        self.norm2 = Fp32LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x

class Qwen2_5_VLPatchMerger(nn.Module):
    """
    Patch merger for Qwen 2.5 Vision. ZL: check if this is correct.
    """

    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.spatial_merge_size = spatial_merge_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class Qwen2_5_VisionTransformer(nn.Module):
    """
    Qwen2_5_VisionTransformer ZL: check if this is correct.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        spatial_merge_size: int,
        num_layers: int,
        num_heads: int,
        embed_dim: int, # hidden_size
        window_size: int,
        out_hidden_size: int,
        fullatt_block_indexes: List[int],
        in_channels: int = 3,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_dim = embed_dim # hidden_size
        self.window_size = window_size
        self.out_hidden_size = out_hidden_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.in_channels = in_channels
    
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        _attn_implementation = 'sdpa' # TODO: implement this ZL
        
        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        head_dim = self.embed_dim // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLVisionBlock(self.embed_dim, self.num_heads, _attn_implementation) for _ in range(self.num_layers)]
        )

        self.merger = Qwen2_5_VLPatchMerger(
            dim = self.out_hidden_size,
            context_dim = self.hidden_size,
            spatial_merge_size = self.spatial_merge_size,
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        images: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Processes images and returns the tokens and hidden states.

        Multiple images per sample: we add a dimension n_imgs to the input. This is useful when a single
        sample constains multiple images, for example:

        - sample 1: "<image> what animal is this?"
        - sample 2: "I like <image> more than <image>"

        In this case, sample 1 has one image, and sample 2 has two images. max_n_imgs = max(2,1) = 2.
        So your input should have shape (bsz=2, n_imgs=2, num_tiles, n_channels, tile_size, tile_size).

        Notice that to batch it, you will have to pad n_imgs to max_n_imgs and max_num_tiles.

        Args:
            images (torch.Tensor): torch.Tensor with shape (bsz, n_imgs, n_tiles, n_channels, tile_size, tile_size).
            aspect_ratio (Optional[torch.Tensor]): torch.Tensor with shape (bsz, n_imgs, 2). If all
                images have a single tile, i.e. they were not tile-cropped, it should be None.
                Used to calculate the positional embeddings for the tiles.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: A tuple: (x, hidden_states),
                where x is a torch.tensor of shape (bsz, n_imgs, n_tiles, n_tokens, embed_dim) and
                hidden_states has shape is a list of len(out_indices) torch.tensor with shape
                (bsz, n_imgs, n_tiles, n_tokens, embed_dim).

        Raises:
            ValueError: If aspect_ratio is None, but n_tiles > 1 in the batch.

        Examples:

            >>> from torchtune.modules.transforms.vision_utils.tile_crop import tile_crop
            >>> from torchtune.modules import VisionTransformer
            >>>
            >>> num_channels = 3
            >>> image_size = (800,400)
            >>> tile_size = 400
            >>> patch_size=40
            >>> patch_grid_size = tile_size // patch_size
            >>>
            >>> # for details about preprocessing, please check
            >>> # torchtune.models.clip._transforms.CLIPImageTransform
            >>>
            >>> # create a random image
            >>> image = torch.rand(num_channels, image_size[0], image_size[1])
            >>>
            >>> # (num_tiles, nch, h, w) -> (2, 3, 400, 400)
            >>> tile_cropped_image = tile_crop(image, tile_size)
            >>> aspect_ratio = torch.tensor([2,1])
            >>>
            >>> # make it a batch of 1 image
            >>> batch_image = tile_cropped_image.unsqueeze(0)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(0)
            >>>
            >>> # make it have only 1 image per sample
            >>> batch_image = tile_cropped_image.unsqueeze(1)
            >>> batch_aspect_ratio = aspect_ratio.unsqueeze(1)
            >>>
            >>> # For a detailed example, please check
            >>> # torchtune.models.clip._position_embeddings.clip_vision_encoder
            >>> # model = VisionTransformer(
            ... #           out_indices = [1,2,3,4,5],
            ... #           patch_size=40,
            ... #           patch_grid_size = patch_grid_size,
            ... #           embed_dim = 32,
            ... #           num_layers = 6,
            ... #           in_channels = num_channels,
            ... #           ...)
            >>>
            >>> x, hidden_states = model(images = batch_image, aspect_ratio = batch_aspect_ratio)
            >>>
            >>> # (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(x.shape)
            torch.Size([1, 1, 2, 101, 32])
            >>>
            >>> # list with tensors of shape (bsz, n_imgs, num_tiles, num_patches_per_tile + CLS token, embed_dim)
            >>> print(len(hidden_states))
            5
        """
        hidden_states = []

        # parse inputs
        bsz, n_imgs, n_tiles, nch, w, h = images.shape
        bsz_and_n_imgs = bsz * n_imgs

        # if aspect_ratio is not provided, it defaults to one tile [1,1]
        if aspect_ratio is None:
            aspect_ratio = torch.ones(
                (bsz_and_n_imgs, 2), dtype=torch.int, device=images.device
            )
            if n_tiles > 1:
                raise ValueError(
                    f"aspect_ratio was not provided, but found n_tiles>1 for {images.shape=}. Please provide aspect_ratio."
                )

        images = images.reshape(bsz_and_n_imgs * n_tiles, nch, w, h)
        aspect_ratio = aspect_ratio.reshape(bsz_and_n_imgs, 2)

        # patch embeddings (tokens)
        # A tile becomes a grid of patch_grid_size X patch_grid_size patches
        # these patches are flatenned, and called tokens from here on.

        # out: (bsz * n_imgs * n_tiles, embed_dim, patch_grid_size, patch_grid_size)
        x = self.conv(images)

        # out: (bsz * n_imgs, n_tiles, n_tokens, embed_dim)
        x = x.reshape(bsz_and_n_imgs, n_tiles, -1, self.patches_per_tile).permute(
            0, 1, 3, 2
        )
        bsz_and_n_imgs, n_tiles, n_tokens, embed_dim = x.shape

        # pre_tile_pos_embed
        if self.pre_tile_pos_embed:
            x = self.pre_tile_pos_embed(x, aspect_ratio)

        # insert cls token
        x = self.cls_token_embedding(x)
        n_tokens += 1

        # token_pos_embedding
        x = self.token_pos_embedding(x, aspect_ratio)

        # norm
        x = self.ln_pre(x)

        # transformer with optional hidden layer outputs
        x = x.reshape(bsz_and_n_imgs, n_tiles * n_tokens, embed_dim)
        for layer_idx, transformer_layer in enumerate(self.layers):
            if layer_idx in self.out_indices:
                h = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)
                hidden_states.append(h)
            x = transformer_layer(x)

        # norm
        x = self.ln_post(x)

        # post_tile_pos_embed
        if self.post_tile_pos_embed:
            x = x.reshape(bsz_and_n_imgs, n_tiles, n_tokens, embed_dim)
            x = self.post_tile_pos_embed(x, aspect_ratio)

        # reshape output
        x = x.reshape(bsz, n_imgs, n_tiles, n_tokens, embed_dim)


        return x, hidden_states




