from ._config import GptOssConfig
import torch.nn as nn

from torchtune.modules import (
    FeedForward,
    FrozenNF4Linear,
    MultiHeadAttention,
    rms_norm,
    RMSNorm,
    RotaryPositionalEmbeddings,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)
from torchtune.modules.moe import (
    GroupedExperts,
    MoE,
    TokenChoiceTopKRouter,
)

def gpt_oss_moe(
    dim: int,
    hidden_dim: int,
    num_experts: int = 8,
    experts_per_token: int = 1,
    use_shared_expert: bool = True,
) -> MoE:
    """
    Build the MoE layer associated with the Llama model.

    Args:
        dim (int): Input dimension of experts.
        hidden_dim (int): Hidden dimension of experts.
        num_experts (int): Number of experts in each MoE layer. Default: 8
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
        use_shared_expert (bool): Whether to use a shared expert or not. Default: True

    Returns:
        MoE: Instantiation of MoE layer.
    """
    router = TokenChoiceTopKRouter(
        gate=nn.Linear(dim, num_experts, bias=False),
        dim=dim,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
    )
    experts = GroupedExperts(dim=dim, hidden_dim=hidden_dim, num_experts=num_experts)
    return MoE(
        experts=experts,
        router=router,
    )

def gpt_oss(config: GptOssConfig):
    rope = RotaryPositionalEmbeddings(
        dim=config.hidden_size, max_seq_len=config.max_seq_len, base=config.rope_theta
    )
    layers = []
    for i in range(config.num_hidden_layers):
        
        self_attn = MultiHeadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            q_proj=nn.Linear(config.hidden_size, config.num_attention_heads * config.hidden_size // config.num_attention_heads, bias=False),
            k_proj=nn.Linear(config.hidden_size, config.num_key_value_heads * config.hidden_size // config.num_key_value_heads, bias=False),
            v_proj=nn.Linear(config.hidden_size, config.num_key_value_heads * config.hidden_size // config.num_key_value_heads, bias=False),
            output_proj=nn.Linear(config.hidden_size, config.hidden_size, bias=False),
            pos_embeddings=rope,
            q_norm=rms_norm(eps=config.rms_norm_eps),
            k_norm=rms_norm(eps=config.rms_norm_eps),
            max_seq_len=config.max_seq_len,
            attn_dropout=config.attn_dropout,
        )
        moe = gpt_oss_moe(config.hidden_size, config.hidden_size, config.num_experts, config.experts_per_token, config.use_shared_expert)
    return 

