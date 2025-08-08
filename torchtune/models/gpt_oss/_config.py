from dataclasses import dataclass
from typing import List


@dataclass
class GptOssConfig:
    """
    Simple configuration class for GPT-OSS models.
    
    Contains only the essential parameters needed to build 20B and 120B variants.
    """
    # Model dimensions
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    num_hidden_layers: int = 36
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    
    # MoE parameters
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    
    # Attention parameters
    max_seq_len: int = 131072
    rope_theta: float = 150000
    sliding_window: int = 128
    layer_types: List[str] = None
    
    # Normalization
    rms_norm_eps: float = 1e-05

    # Other
    attn_dropout: float = 0.0
    
    def __post_init__(self):
        """Set default layer types if not provided."""
        if self.layer_types is None:
            # Alternating sliding and full attention
            self.layer_types = ["sliding_attention", "full_attention"] * (self.num_hidden_layers // 2)
            # Ensure we have exactly the right number of layers
            self.layer_types = self.layer_types[:self.num_hidden_layers]
    
    @classmethod
    def gpt_oss_20b(cls) -> "GptOssConfig":
        """Configuration for GPT-OSS 20B model (21B total params, 3.6B active)."""
        return cls(
            hidden_size=2880,
            intermediate_size=2880,
            num_hidden_layers=36,
            num_attention_heads=64,
            num_key_value_heads=8,
            num_local_experts=128,
        )
    
    @classmethod
    def gpt_oss_120b(cls) -> "GptOssConfig":
        """Configuration for GPT-OSS 120B model (117B total params, 5.1B active)."""
        return cls(
            hidden_size=2880,
            intermediate_size=2880, 
            num_hidden_layers=36,
            num_attention_heads=64,
            num_key_value_heads=8,
            num_local_experts=128,
        )

