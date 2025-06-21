# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional

import torch
from torch import nn


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors.
    
    This is the MRoPE (Multimodal Rotary Position Embedding) from Qwen2.5-VL which extends
    standard RoPE to handle 3D position embeddings (temporal, height, width).
    
    Args:
        q (torch.Tensor): The query tensor.
        k (torch.Tensor): The key tensor.
        cos (torch.Tensor): The cosine part of the rotary embedding.
        sin (torch.Tensor): The sine part of the rotary embedding.
        mrope_section (List[int]): Multimodal rope section [temporal_dim, height_dim, width_dim].
        unsqueeze_dim (int): The dimension to unsqueeze for broadcasting.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors.
    """
    # Double the mrope_section for cos/sin pairs
    mrope_section = [x * 2 for x in mrope_section]
    
    # Split cos/sin into temporal, height, width sections and recombine
    cos_parts = cos.split(mrope_section, dim=-1)
    sin_parts = sin.split(mrope_section, dim=-1)
    
    cos = torch.cat([cos_parts[i % 3] for i in range(len(cos_parts))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([sin_parts[i % 3] for i in range(len(sin_parts))], dim=-1).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen25VLRotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Multimodal Rotary Positional Embeddings (MRoPE) for Qwen2.5-VL
    based on the implementation in https://arxiv.org/abs/2409.12191.

    MRoPE extends standard RoPE to handle 3D position embeddings:
    - Temporal dimension (for videos)
    - Height dimension (spatial)
    - Width dimension (spatial)

    For text-only tokens, all three dimensions use the same position IDs, making it
    equivalent to standard 1D RoPE. The key innovation is that different parts of
    the embedding dimension handle different spatial dimensions.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        mrope_section (List[int]): The dimensions allocated to temporal, height, and width.
            Should sum to head_dim. Default: [16, 24, 24] (sum=64 for typical head_dim)
        max_seq_len (int): Maximum expected sequence length for the model, if exceeded
            the cached freqs will be recomputed. Default: 32768
        base (float): The base for the geometric progression used to compute
            the rotation angles. Default: 1000000.0
    """

    def __init__(
        self,
        dim: int,
        mrope_section: List[int] = [16, 24, 24],
        max_seq_len: int = 32768,
        base: float = 1000000.0,
    ) -> None:
        super().__init__()
        if sum(mrope_section) != dim:
            raise ValueError(f"mrope_section {mrope_section} must sum to dim {dim}")
        
        self.dim = dim
        self.mrope_section = mrope_section
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        # Compute theta for each section separately
        # Temporal section
        temporal_dim = self.mrope_section[0]
        temporal_theta = 1.0 / (
            self.base ** (torch.arange(0, temporal_dim, 2).float() / temporal_dim)
        )
        
        # Height section  
        height_dim = self.mrope_section[1]
        height_theta = 1.0 / (
            self.base ** (torch.arange(0, height_dim, 2).float() / height_dim)
        )
        
        # Width section
        width_dim = self.mrope_section[2]
        width_theta = 1.0 / (
            self.base ** (torch.arange(0, width_dim, 2).float() / width_dim)
        )
        
        self.register_buffer("temporal_theta", temporal_theta, persistent=False)
        self.register_buffer("height_theta", height_theta, persistent=False)
        self.register_buffer("width_theta", width_theta, persistent=False)
        
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 32768) -> None:
        # Create position indexes for each dimension
        seq_idx = torch.arange(max_seq_len, dtype=self.temporal_theta.dtype, device=self.temporal_theta.device)

        # Compute frequency matrices for each dimension
        temporal_freqs = torch.outer(seq_idx, self.temporal_theta).float()
        height_freqs = torch.outer(seq_idx, self.height_theta).float()
        width_freqs = torch.outer(seq_idx, self.width_theta).float()

        # Cache includes both cos and sin components for each dimension
        # Shape: [max_seq_len, dim_section//2, 2]
        temporal_cache = torch.stack([torch.cos(temporal_freqs), torch.sin(temporal_freqs)], dim=-1)
        height_cache = torch.stack([torch.cos(height_freqs), torch.sin(height_freqs)], dim=-1) 
        width_cache = torch.stack([torch.cos(width_freqs), torch.sin(width_freqs)], dim=-1)
        
        self.register_buffer("temporal_cache", temporal_cache, persistent=False)
        self.register_buffer("height_cache", height_cache, persistent=False)
        self.register_buffer("width_cache", width_cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids.
                Can be either:
                - 2D tensor with shape [b, s] for standard RoPE (will be expanded to 3D)
                - 3D tensor with shape [3, b, s] for MRoPE where 3 represents [temporal, height, width]
                If None, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        if input_pos is None:
            # Create default sequential positions for all dimensions
            device = x.device
            pos_1d = torch.arange(seq_len, device=device)
            input_pos = pos_1d.unsqueeze(0).expand(3, 1, -1)  # [3, 1, s]
            input_pos = input_pos.expand(3, x.size(0), -1)    # [3, b, s]
        elif input_pos.dim() == 2:  # [b, s]
            # Convert 2D to 3D by replicating across all 3 dimensions
            input_pos = input_pos.unsqueeze(0).expand(3, -1, -1)  # [3, b, s]

        # Extract position indices for each dimension
        temporal_pos = input_pos[0]  # [b, s]
        height_pos = input_pos[1]    # [b, s]  
        width_pos = input_pos[2]     # [b, s]

        # Extract cached values for each dimension
        temporal_rope = self.temporal_cache[temporal_pos]  # [b, s, temporal_dim//2, 2]
        height_rope = self.height_cache[height_pos]        # [b, s, height_dim//2, 2]
        width_rope = self.width_cache[width_pos]           # [b, s, width_dim//2, 2]

        # Apply rotations for each section of the embedding
        return self._apply_mrope_rotation(x, temporal_rope, height_rope, width_rope)

    def _apply_mrope_rotation(
        self, 
        x: torch.Tensor, 
        temporal_rope: torch.Tensor, 
        height_rope: torch.Tensor, 
        width_rope: torch.Tensor
    ) -> torch.Tensor:
        """Apply MRoPE rotation to different sections of the embedding dimension."""
        b, s, n_h, h_d = x.shape
        
        # Split input into sections corresponding to temporal, height, width
        temporal_dim, height_dim, width_dim = self.mrope_section
        x_temporal = x[..., :temporal_dim]                    # [b, s, n_h, temporal_dim]
        x_height = x[..., temporal_dim:temporal_dim+height_dim]  # [b, s, n_h, height_dim]
        x_width = x[..., temporal_dim+height_dim:]            # [b, s, n_h, width_dim]

        # Apply rotation to each section
        x_temporal_rotated = self._apply_rotation_to_section(x_temporal, temporal_rope)
        x_height_rotated = self._apply_rotation_to_section(x_height, height_rope)  
        x_width_rotated = self._apply_rotation_to_section(x_width, width_rope)

        # Concatenate rotated sections back together
        x_out = torch.cat([x_temporal_rotated, x_height_rotated, x_width_rotated], dim=-1)
        return x_out

    def _apply_rotation_to_section(self, x_section: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        """Apply rotation to a specific section of the embedding."""
        # x_section: [b, s, n_h, section_dim]
        # rope_cache: [b, s, section_dim//2, 2]
        
        # Reshape input for rotation: [b, s, n_h, section_dim//2, 2]
        x_shaped = x_section.float().reshape(*x_section.shape[:-1], -1, 2)
        
        # Reshape cache for broadcasting: [b, s, 1, section_dim//2, 2]
        rope_cache = rope_cache.unsqueeze(2)
        
        # Apply rotation
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0] - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0] + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            dim=-1,
        )
        
        # Flatten back to original shape
        x_out = x_out.flatten(-2)
        return x_out.type_as(x_section)
