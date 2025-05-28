# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torch

from torchtune.models.convert_weights import get_mapped_key

# state dict key mappings from HF's format to torchtune's format key from checkpoint, value from model
_FROM_HF = {
    "visual.patch_embed.proj.weight":"encoders.vision.patch_embed.proj.weight",
    "visual.blocks.{}.attn.proj.bias": "encoders.vision.layers.{}.attn.proj.bias", 
    "visual.blocks.{}.attn.proj.weight": "encoders.vision.layers.{}.attn.proj.weight", 
    "visual.blocks.{}.attn.qkv.bias": "encoders.vision.layers.{}.attn.qkv.bias",
    "visual.blocks.{}.attn.qkv.weight": "encoders.vision.layers.{}.attn.qkv.weight",
    "visual.blocks.{}.mlp.down_proj.bias": "encoders.vision.layers.{}.mlp.down_proj.bias", 
    "visual.blocks.{}.mlp.down_proj.weight": "encoders.vision.layers.{}.mlp.down_proj.weight", 
    "visual.blocks.{}.mlp.gate_proj.bias": "encoders.vision.layers.{}.mlp.gate_proj.bias",
    "visual.blocks.{}.mlp.gate_proj.weight": "encoders.vision.layers.{}.mlp.gate_proj.weight",
    "visual.blocks.{}.mlp.up_proj.bias": "encoders.vision.layers.{}.mlp.up_proj.bias",
    "visual.blocks.{}.mlp.up_proj.weight": "encoders.vision.layers.{}.mlp.up_proj.weight",
    "visual.blocks.{}.norm1.weight": "encoders.vision.layers.{}.norm1.weight",
    "visual.blocks.{}.norm2.weight": "encoders.vision.layers.{}.norm2.weight",
    "visual.merger.ln_q.weight": "encoders.vision.merger.ln_q.weight",
    "visual.merger.mlp.{}.weight": "encoders.vision.merger.mlp.{}.weight",
    "visual.merger.mlp.{}.bias": "encoders.vision.merger.mlp.{}.bias",
    "model.embed_tokens.weight": "decoder.tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "decoder.layers.{}.attn.q_proj.weight",
    "model.layers.{}.self_attn.q_proj.bias": "decoder.layers.{}.attn.q_proj.bias",
    "model.layers.{}.self_attn.k_proj.weight": "decoder.layers.{}.attn.k_proj.weight",
    "model.layers.{}.self_attn.k_proj.bias": "decoder.layers.{}.attn.k_proj.bias",
    "model.layers.{}.self_attn.v_proj.weight": "decoder.layers.{}.attn.v_proj.weight",
    "model.layers.{}.self_attn.v_proj.bias": "decoder.layers.{}.attn.v_proj.bias",
    "model.layers.{}.self_attn.o_proj.weight": "decoder.layers.{}.attn.output_proj.weight",
    "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
    "model.layers.{}.mlp.gate_proj.weight": "decoder.layers.{}.mlp.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "decoder.layers.{}.mlp.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "decoder.layers.{}.mlp.w2.weight",
    "model.layers.{}.input_layernorm.weight": "decoder.layers.{}.sa_norm.scale",
    "model.layers.{}.post_attention_layernorm.weight": "decoder.layers.{}.mlp_norm.scale",
    "model.norm.weight": "decoder.norm.scale",
    "lm_head.weight": "decoder.output.weight",
    # TODO: Add vision weights
}


QWEN2_TIED_KEY = "lm_head.weight"

def qwen2_5_vision_hf_to_tune(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Convert a state dict from HF's format to TorchTune's format, which contains the weights
    of a Qwen2 model.
    State dicts from multiple checkpoint files should be consolidated into a single state dict
    before calling this function.
    The logic is identical to :func:`~torchtune.models.convert_weights.hf_to_tune`, but may not load
    output projection weights.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in HF's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        Dict[str, torch.Tensor]: State dict in torchtune's format.
    """
    converted_state_dict = {}
    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        if (
            tie_word_embeddings and QWEN2_TIED_KEY in key
        ):  # Skip loading the output projection weights
            continue
        if "rotary_emb.inv_freq" in key:  # Skip loading the position embeddings
            continue

        new_key = get_mapped_key(key, _FROM_HF)
        #new_key = key
        converted_state_dict[new_key] = value
    return converted_state_dict



def qwen2_5_vision_tune_to_hf(
    state_dict: Dict[str, torch.Tensor],
    num_heads: int = 32,
    num_kv_heads: int = 32,
    dim: int = 4096,
    head_dim: int = None,
    tie_word_embeddings: bool = False,
):
    """
    Convert a state dict from torchtune's format to HF's format. This function
    doesn't handle any sharding or splitting of state dicts. It follows the
    state_dict IN -> state_dict OUT pattern.

    Args:
        state_dict (Dict[str, torch.Tensor]): State dict in torchtune's format.
        num_heads (int): Number of heads in the model.
        num_kv_heads (int): Number of heads in the key/value projection layers.
        dim (int): Dimension of the model.
        head_dim (int): Dimension of the head. If not provided, it will be calculated
            as dim // num_heads.
        tie_word_embeddings (bool): Whether the model's input and output word embeddings should be tied.

    Returns:
        Dict[str, torch.Tensor]: State dict in HF's format.
    """
    converted_state_dict = {}
    inverted_mapping_dict = {v: k for k, v in _FROM_HF.items()}

    if head_dim is None:
        head_dim = dim // num_heads

    for key, value in state_dict.items():
        new_key = get_mapped_key(key, inverted_mapping_dict)
        converted_state_dict[new_key] = value

    return converted_state_dict
