#!/usr/bin/env python3
"""
HF Transformers + vLLM Medusa Speculative Decoding
with Greedy Sampling & KV Cache Eviction
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from vllm.model_executor.models.medusa import Medusa
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_model_parallel,
    destroy_distributed_environment,
    get_world_group
)

# --- CONFIG ---
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MEDUSA_CHECKPOINT = "./vllm_medusa_model"
MAX_STEPS = 50
PROMPT = "The quick brown fox jumps over the lazy dog."

# --- DEFAULT ENV ---
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

# --- INIT DISTRIBUTED ---
init_distributed_environment()
initialize_model_parallel(1, 1)
world_group = get_world_group()
print(f"🔄 Rank {world_group.rank} | Local Rank {world_group.local_rank} | World Size {world_group.world_size}")

# --- DEVICE ---
device = torch.device(f"cuda:{world_group.local_rank}")
torch.cuda.set_device(device)
dtype = torch.float16

# --- LOAD MODEL + TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype).to(device)
model.eval()

# --- LOAD MEDUSA ---
if os.path.isdir(MEDUSA_CHECKPOINT):
    pt_files = [f for f in os.listdir(MEDUSA_CHECKPOINT) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {MEDUSA_CHECKPOINT}")
    checkpoint_path = os.path.join(MEDUSA_CHECKPOINT, pt_files[0])
else:
    checkpoint_path = MEDUSA_CHECKPOINT

checkpoint = torch.load(checkpoint_path, map_location=device)

from dataclasses import dataclass
@dataclass
class HFConfig:
    hidden_size: int = 4096
    vocab_size: int = 128256
    truncated_vocab_size: int = 128256
    medusa_fc_bias: bool = False
    logit_scale: float = 1.0
    original_lm_head: bool = False
    num_heads: int = 5
    num_hidden_layers: int = 3
@dataclass
class DraftModelConfig: hf_config: HFConfig
@dataclass
class SpeculativeConfig: draft_model_config: DraftModelConfig
@dataclass
class VllmConfig: speculative_config: SpeculativeConfig

config = VllmConfig(
    speculative_config=SpeculativeConfig(
        draft_model_config=DraftModelConfig(hf_config=HFConfig())
    )
)
medusa_model = Medusa(vllm_config=config, prefix="").to(device=device, dtype=dtype)
weights = [(k, v) for k, v in checkpoint.items() if k != "config"]
medusa_model.load_weights(weights)
medusa_model.eval()

# --- TOKENIZE ---
input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
valid_len = input_ids.shape[1]

# --- SAMPLING ---
sampling_params = SamplingParams(temperature=1.0, top_k=50, top_p=0.9)
seq_group = SequenceGroupToSample(
    seq_ids=[0], sampling_params=sampling_params,
    seq_data={0: None}, seq_len=input_ids.shape[1],
    query_len=1, generator=None, is_prompt=True,
    prompt_logprob_indices=[], sample_indices=[0]
)
sampling_metadata = SamplingMetadata(
    seq_groups=[seq_group],
    selected_token_indices=torch.tensor([0], device=device),
    categorized_sample_indices={SamplingType.GREEDY: torch.tensor([0], device=device)},
    num_prompts=1
)

# --- INIT ---
past_key_values = DynamicCache()
preds = input_ids
accept_lengths = []
accepted_candidates = []
with torch.inference_mode():
    # First forward

    outputs = model(preds, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]
    logits = outputs.logits
    past_key_values = outputs.past_key_values

    # Greedy base token
    pred_base = torch.argmax(logits[:, -1:, :], dim=-1)  # (bs,1)

    # Get Medusa predictions using top-1 sampling
    medusa_outputs = medusa_model.generate_proposals(hidden_states, sampling_metadata)
    pred_medusa = medusa_outputs[0].sampled_token_ids[:, -1].unsqueeze(0)  # (num_heads,)
    
    preds = torch.cat((pred_base, pred_medusa), dim = -1)
    valid_len += 1
    accept_lengths.append(1)
    print(f"[Init] Base={tokenizer.decode(pred_base[0].tolist())} | "f"Medusa={[tokenizer.decode([t.item()]) for t in pred_medusa[0]]}")
    cache_length = past_key_values.get_seq_length()
    print("cache_length:", cache_length)
    # Iter loop
    for step in range(MAX_STEPS):
        outputs = model(preds, past_key_values=past_key_values, use_cache=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # Base greedy token
        pred_base = torch.argmax(logits[:, :], dim=-1)
        print("pred_base.shape", pred_base.shape)
        # breakpoint()
        decoded_base = tokenizer.decode(pred_base[0, :].tolist(), skip_special_tokens=True)
        print('Base:', decoded_base)
        # Get Medusa predictions using top-1 sampling
        medusa_outputs = medusa_model.generate_proposals(hidden_states, sampling_metadata)
        pred_medusa = medusa_outputs[0].sampled_token_ids.unsqueeze(0)  # (num_heads,)
        # Verify Medusa against base
        # breakpoint()
        posterior_mask = (preds[:,1:] == pred_base[:, :-1]).int()
        accept_length = torch.cumprod(posterior_mask, dim=-1).sum().item()
        valid_len += accept_length + 1

        # Update KV cache - keep the accepted tokens plus the new base prediction
        cache_length = past_key_values.get_seq_length()
        if accept_length < cache_length:
            for layer_idx in range(len(past_key_values.layers)):
                layer_cache = past_key_values.layers[layer_idx]
                if layer_cache.keys is not None and layer_cache.values is not None:
                    # Keep accepted tokens plus one new token
                    layer_cache.keys = layer_cache.keys[:, :, :valid_len, :]
                    layer_cache.values = layer_cache.values[:, :, :valid_len, :]
        new_cache_length = past_key_values.get_seq_length()
        print("new_cache_length:", new_cache_length)
        assert(new_cache_length==valid_len)
        preds = torch.cat([pred_base[:, accept_length].unsqueeze(0), pred_medusa[:, :, accept_length]], dim=-1)  # (B, 1+H)
        # breakpoint()
        accept_lengths.append(accept_length + 1)
        accepted_tokens = pred_base[:, :accept_length+1]  # shape (1, accept_length)
        accepted_candidates.extend(accepted_tokens[0])
        if accept_length > 0:
            decoded_base = tokenizer.decode(accepted_tokens[0].tolist(), skip_special_tokens=True)
        else:
            decoded_base = ""  # no accepted tokens
        
        print(f"[Step {step+1}] Accepted={decoded_base} | Accept={accept_length}")


        if tokenizer.eos_token_id in pred_base:
            break
print('Average accept length:', sum(accept_lengths)/len(accept_lengths))
print(f"\n[Rank {world_group.rank}] Final text:\n{tokenizer.decode(accepted_candidates, skip_special_tokens=False)}")

# --- CLEANUP ---
destroy_model_parallel()
destroy_distributed_environment()