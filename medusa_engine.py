#!/usr/bin/env python3
"""
Custom vLLM Engine with Medusa Support - Matching Original Implementation
"""
import os
import torch
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.model_executor.models.medusa import Medusa
from vllm.model_executor.sampling_metadata import SamplingMetadata, SequenceGroupToSample
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.outputs import RequestOutput
from vllm.sequence import SequenceGroup, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# --- Medusa Config ---
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

class MedusaState:
    def __init__(self):
        self.past_key_values = DynamicCache()
        self.valid_len = 0
        self.accept_lengths = []
        self.accepted_candidates = []
        self.preds = None

class MedusaLLMEngine(LLMEngine):
    def __init__(self, engine_args: AsyncEngineArgs):
        super().__init__(engine_args)
        
        # Initialize Medusa specific components
        self.device = torch.device(f"cuda:{self.worker.local_rank}")
        self.dtype = torch.float16
        
        # Load Medusa checkpoint
        medusa_checkpoint = "./vllm_medusa_model"
        if os.path.isdir(medusa_checkpoint):
            pt_files = [f for f in os.listdir(medusa_checkpoint) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in {medusa_checkpoint}")
            checkpoint_path = os.path.join(medusa_checkpoint, pt_files[0])
        else:
            checkpoint_path = medusa_checkpoint
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize Medusa model
        config = VllmConfig(
            speculative_config=SpeculativeConfig(
                draft_model_config=DraftModelConfig(hf_config=HFConfig())
            )
        )
        self.medusa_model = Medusa(vllm_config=config, prefix="").to(device=self.device, dtype=self.dtype)
        weights = [(k, v) for k, v in checkpoint.items() if k != "config"]
        self.medusa_model.load_weights(weights)
        self.medusa_model.eval()
        
        # State tracking per sequence
        self.seq_states: Dict[int, MedusaState] = {}
        
    def _prepare_medusa_metadata(self, seq_group: SequenceGroup) -> SamplingMetadata:
        """Prepare Medusa sampling metadata for a sequence group."""
        sampling_params = seq_group.sampling_params
        seq_ids = [seq.seq_id for seq in seq_group.seqs]
        
        seq_data = {seq.seq_id: None for seq in seq_group.seqs}
        sample_indices = list(range(len(seq_ids)))
        
        group = SequenceGroupToSample(
            seq_ids=seq_ids,
            sampling_params=sampling_params,
            seq_data=seq_data,
            seq_len=seq_group.seqs[0].get_len(),
            query_len=1,
            generator=None,
            is_prompt=True,
            prompt_logprob_indices=[],
            sample_indices=sample_indices
        )
        
        return SamplingMetadata(
            seq_groups=[group],
            selected_token_indices=torch.tensor([0], device=self.device),
            categorized_sample_indices={SamplingType.GREEDY: torch.tensor([0], device=self.device)},
            num_prompts=1
        )

    def _process_sequence_group(self, seq_group: SequenceGroup, prompt_tokens: List[int]) -> None:
        """Process a sequence group with Medusa speculative decoding."""
        seq_id = seq_group.seqs[0].seq_id
        
        # Initialize state if needed
        if seq_id not in self.seq_states:
            self.seq_states[seq_id] = MedusaState()
        state = self.seq_states[seq_id]
        
        # Convert tokens to tensor
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        
        if state.preds is None:
            state.preds = input_ids
            state.valid_len = input_ids.shape[1]
            
            # First forward pass
            outputs = self.model(
                state.preds,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            state.past_key_values = outputs.past_key_values
            
            # Get base and Medusa predictions
            pred_base = torch.argmax(logits[:, -1:, :], dim=-1)
            sampling_metadata = self._prepare_medusa_metadata(seq_group)
            medusa_outputs = self.medusa_model.generate_proposals(hidden_states, sampling_metadata)
            pred_medusa = medusa_outputs[0].sampled_token_ids[:, -1].unsqueeze(0)
            
            state.preds = torch.cat((pred_base, pred_medusa), dim=-1)
            state.valid_len += 1
            state.accept_lengths.append(1)
            
            # Debug prints
            print(f"[Init] Base={self.tokenizer.decode(pred_base[0].tolist())} | "
                  f"Medusa={[self.tokenizer.decode([t.item()]) for t in pred_medusa[0]]}")
            cache_length = state.past_key_values.get_seq_length()
            print("cache_length:", cache_length)
            
        else:
            # Subsequent iterations
            outputs = self.model(
                state.preds,
                past_key_values=state.past_key_values,
                use_cache=True,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            logits = outputs.logits
            state.past_key_values = outputs.past_key_values
            
            # Base prediction
            pred_base = torch.argmax(logits[:, :], dim=-1)
            print("pred_base.shape", pred_base.shape)
            decoded_base = self.tokenizer.decode(pred_base[0, :].tolist(), skip_special_tokens=True)
            print('Base:', decoded_base)
            
            # Medusa predictions
            sampling_metadata = self._prepare_medusa_metadata(seq_group)
            medusa_outputs = self.medusa_model.generate_proposals(hidden_states, sampling_metadata)
            pred_medusa = medusa_outputs[0].sampled_token_ids.unsqueeze(0)
            
            # Verify predictions
            posterior_mask = (state.preds[:, 1:] == pred_base[:, :-1]).int()
            accept_length = torch.cumprod(posterior_mask, dim=-1).sum().item()
            state.valid_len += accept_length + 1
            
            # Update KV cache - keep the accepted tokens plus the new base prediction
            cache_length = state.past_key_values.get_seq_length()
            if accept_length < cache_length:
                for layer_idx in range(len(state.past_key_values.layers)):
                    layer_cache = state.past_key_values.layers[layer_idx]
                    if layer_cache.keys is not None and layer_cache.values is not None:
                        # Keep accepted tokens plus one new token
                        layer_cache.keys = layer_cache.keys[:, :, :state.valid_len, :]
                        layer_cache.values = layer_cache.values[:, :, :state.valid_len, :]
            new_cache_length = state.past_key_values.get_seq_length()
            assert(new_cache_length == state.valid_len), f"Cache length {new_cache_length} != Valid length {state.valid_len}"
            
            # Update predictions and track acceptance
            state.preds = torch.cat([
                pred_base[:, accept_length].unsqueeze(0),
                pred_medusa[:, :, accept_length]
            ], dim=-1)
            state.accept_lengths.append(accept_length + 1)
            accepted_tokens = pred_base[:, :accept_length+1]  # shape (1, accept_length)
            state.accepted_candidates.extend(accepted_tokens[0])
            
            if accept_length > 0:
                decoded_base = self.tokenizer.decode(accepted_tokens[0].tolist(), skip_special_tokens=True)
            else:
                decoded_base = ""  # no accepted tokens
            
            print(f"[Step {len(state.accept_lengths)}] Accepted={decoded_base} | Accept={accept_length}")

    def _generate_token(self, seq_group: SequenceGroup, prompt_tokens: List[int]) -> RequestOutput:
        """Override token generation to include Medusa speculative decoding."""
        self._process_sequence_group(seq_group, prompt_tokens)
        
        # Get the next token from the accepted candidates
        seq_id = seq_group.seqs[0].seq_id
        state = self.seq_states[seq_id]
        
        if state.accepted_candidates:
            next_token = state.accepted_candidates[-1].item()
        else:
            # Fallback to base model if no accepted candidates
            output = super()._generate_token(seq_group, prompt_tokens)
            next_token = output.outputs[0].token_ids[-1]
            
        return RequestOutput(
            request_id=seq_group.request_id,
            prompt_token_ids=prompt_tokens,
            outputs=[Sequence(seq_id=seq_id, token_ids=[next_token])]
        )

class AsyncMedusaLLMEngine(AsyncLLMEngine):
    def __init__(self, engine_args: AsyncEngineArgs):
        engine_args.trust_remote_code = True
        engine_args.enforce_eager = True
        super().__init__(engine_args)
        self._llm_engine = MedusaLLMEngine(engine_args)