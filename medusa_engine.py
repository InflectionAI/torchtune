#!/usr/bin/env python3
"""
Custom vLLM Engine with Medusa Support - Matching Original Implementation
"""
import os
import gc
import torch
from typing import List, Optional, Dict, Any
import torch.nn as nn
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
from vllm.config import VllmConfig

def print_tensor_info(name: str, tensor: torch.Tensor):
    """Helper to print tensor details."""
    print(f"📊 {name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Device: {tensor.device}")
    print(f"  Dtype: {tensor.dtype}")
    if tensor.numel() < 10:
        print(f"  Values: {tensor.tolist()}")
    print()

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
    num_hidden_layers: int = 3  # Each block has 3 layers, matching the checkpoint

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

    @property
    def avg_accept_length(self) -> Optional[float]:
        """Calculate average acceptance length if available."""
        if self.accept_lengths:
            return sum(self.accept_lengths) / len(self.accept_lengths)
        return None

    def print_state(self):
        """Print current state details."""
        print("\n📋 Current Medusa State:")
        print(f"  Valid Length: {self.valid_len}")
        print(f"  Accept Lengths: {self.accept_lengths}")
        print(f"  Avg Accept Length: {self.avg_accept_length}")
        print(f"  Num Accepted Candidates: {len(self.accepted_candidates)}")
        if self.preds is not None:
            print_tensor_info("Predictions", self.preds)
        print()

class MedusaRequestOutput(RequestOutput):
    """Extended RequestOutput with Medusa metrics."""
    def __init__(self, request_id: str, prompt_token_ids: List[int], outputs: List[Sequence], medusa_metrics: Optional[Dict[str, float]] = None):
        super().__init__(request_id=request_id, prompt_token_ids=prompt_token_ids, outputs=outputs)
        self.medusa_metrics = medusa_metrics or {}

class MedusaLLMEngine(LLMEngine):
    def __init__(self, vllm_config: VllmConfig, executor_class=None, log_stats: bool = False):
        print("\n🚀 Initializing MedusaLLMEngine...")

        # 1️⃣ Initialize base model first (super() allocates base LLM)
        super().__init__(vllm_config, executor_class=executor_class, log_stats=log_stats)

        self.dtype = torch.float16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Clear CUDA cache before loading Medusa
        torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleared CUDA cache before loading Medusa")

        # 2️⃣ Load Medusa checkpoint on CPU
        # Use environment variable or default path
        medusa_checkpoint = os.environ.get("MEDUSA_CHECKPOINT", "./vllm_medusa_model/vllm_medusa_heads_final.pt")
        if os.path.isdir(medusa_checkpoint):
            pt_files = [f for f in os.listdir(medusa_checkpoint) if f.endswith(".pt")]
            if not pt_files:
                raise FileNotFoundError(f"No .pt files found in {medusa_checkpoint}")
            checkpoint_path = os.path.join(medusa_checkpoint, pt_files[0])
        else:
            checkpoint_path = medusa_checkpoint

        print(f"🔄 Loading Medusa checkpoint from {checkpoint_path} on CPU")
        print(f"📁 Checkpoint exists: {os.path.exists(checkpoint_path)}")
        if os.path.exists(checkpoint_path):
            print(f"📊 Checkpoint size: {os.path.getsize(checkpoint_path) / (1024**3):.2f} GB")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Count layers per block from checkpoint
        layers_per_block = sum(1 for k in checkpoint.keys() if k.startswith("blocks.0.layers."))
        print(f"📊 Found {layers_per_block} layers per block in checkpoint")

        # 3️⃣ Define custom ResidualBlock and Medusa class
        class CustomResidualBlock(nn.Module):
            def __init__(self, hidden_size: int, num_layers: int, fc_bias: bool):
                super().__init__()
                # Initialize layers on CPU to avoid GPU memory allocation
                self.layers = nn.ModuleList([
                    nn.Linear(hidden_size, hidden_size, bias=fc_bias, device="cpu")
                    for _ in range(num_layers)
                ])
                self.act = nn.SiLU()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Keep computation on CPU to save GPU memory
                original_device = x.device
                x = x.cpu()
                for layer in self.layers:
                    x = x + self.act(layer(x))
                # Only move result back to original device when returning
                return x.to(device=original_device, dtype=x.dtype)

        class CustomMedusa(Medusa):
            def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
                super().__init__(vllm_config=vllm_config, prefix=prefix)
                config = vllm_config.speculative_config.draft_model_config.hf_config
                # Initialize blocks on CPU
                self.blocks = nn.ModuleList([
                    CustomResidualBlock(
                        hidden_size=self.config.hidden_size,
                        num_layers=layers_per_block,  # Use layers_per_block from checkpoint
                        fc_bias=config.medusa_fc_bias
                    )
                    for _ in range(self.config.num_heads)
                ])
                # Keep blocks on CPU
                self.blocks = self.blocks.cpu()

        # 4️⃣ Initialize Medusa model on CPU first
        with torch.no_grad():
            hf_config = HFConfig()
            config = VllmConfig(
                speculative_config=SpeculativeConfig(
                    draft_model_config=DraftModelConfig(hf_config=hf_config)
                )
            )
            self.medusa_model = CustomMedusa(vllm_config=config, prefix="")
            # Set to eval mode and disable gradients before loading weights
            self.medusa_model.eval()
            for param in self.medusa_model.parameters():
                param.requires_grad = False

        # 5️⃣ Load weights on CPU and keep them there for memory efficiency
        with torch.no_grad():
            for k, v in checkpoint.items():
                if k != "config":
                    # Find the corresponding parameter in the model
                    param_path = k.split('.')
                    current_module = self.medusa_model
                    
                    # Navigate to the parameter
                    for part in param_path[:-1]:
                        if hasattr(current_module, part):
                            current_module = getattr(current_module, part)
                        else:
                            break
                    else:
                        # If we successfully navigated to the parent module
                        param_name = param_path[-1]
                        if hasattr(current_module, param_name):
                            param = getattr(current_module, param_name)
                            if isinstance(param, torch.nn.Parameter):
                                # Convert to float16 and keep on CPU
                                v = v.to(dtype=self.dtype, device='cpu')
                                param.data.copy_(v)
                                del v
        
        # Clear checkpoint from memory
        del checkpoint
        gc.collect()  # Run garbage collection
        torch.cuda.empty_cache()  # Clear CUDA cache

        print("✅ Medusa model initialized and loaded on GPU in eval mode with gradients disabled")

        # 7️⃣ Initialize per-sequence state tracking
        self.seq_states: Dict[int, MedusaState] = {}
        
    def _prepare_medusa_metadata(self, seq_group: SequenceGroup) -> SamplingMetadata:
        """Prepare Medusa sampling metadata for a sequence group."""
        print("\n🔧 Preparing Medusa metadata...")
        sampling_params = seq_group.sampling_params
        seq_ids = [seq.seq_id for seq in seq_group.seqs]
        
        seq_data = {seq.seq_id: None for seq in seq_group.seqs}
        sample_indices = list(range(len(seq_ids)))
        
        print(f"  Sequence IDs: {seq_ids}")
        print(f"  Sample Indices: {sample_indices}")
        
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
        
        metadata = SamplingMetadata(
            seq_groups=[group],
            selected_token_indices=torch.tensor([0], device=self.device),
            categorized_sample_indices={SamplingType.GREEDY: torch.tensor([0], device=self.device)},
            num_prompts=1
        )
        print("✅ Metadata prepared")
        
        # Clean up group data to avoid memory leaks
        group.seq_data.clear()
        group.prompt_logprob_indices.clear()
        group.sample_indices.clear()
        
        return metadata

    def _process_sequence_group(self, seq_group: SequenceGroup, prompt_tokens: List[int]) -> None:
        """Process a sequence group with Medusa speculative decoding."""
        seq_id = seq_group.seqs[0].seq_id
        
        # Initialize state if needed
        if seq_id not in self.seq_states:
            print(f"\n🔄 Starting new sequence {seq_id}")
            self.seq_states[seq_id] = MedusaState()
        state = self.seq_states[seq_id]
        
        # Convert tokens to tensor
        input_ids = torch.tensor([prompt_tokens], device=self.device)
        print("\n📥 Input tokens:")
        print_tensor_info("input_ids", input_ids)
        
        if state.preds is None:
            print("\n🎯 First token generation:")
            state.preds = input_ids
            state.valid_len = input_ids.shape[1]
            
            # First forward pass
            print("  Running base model forward pass...")
            with torch.no_grad():
                outputs = self.model(
                    state.preds,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                # Get required tensors and immediately delete the outputs
                hidden_states = outputs.hidden_states[-1].detach()  # Detach to break computation graph
                logits = outputs.logits.detach()  # Detach to break computation graph
                state.past_key_values = outputs.past_key_values
                del outputs  # Free memory from outputs object
            
            print("\n📊 Base model outputs:")
            print_tensor_info("hidden_states", hidden_states)
            print_tensor_info("logits", logits)
            
            # Get base and Medusa predictions
            pred_base = torch.argmax(logits[:, -1:, :], dim=-1)
            print("\n🎯 Base model prediction:")
            print_tensor_info("pred_base", pred_base)
            
            print("\n🔮 Getting Medusa predictions...")
            sampling_metadata = self._prepare_medusa_metadata(seq_group)
            with torch.no_grad():
                medusa_outputs = self.medusa_model.generate_proposals(hidden_states, sampling_metadata)
                # Extract and detach predictions
                pred_medusa = medusa_outputs[0].sampled_token_ids[:, -1].unsqueeze(0).detach()
                # Clean up outputs
                del medusa_outputs[0].sampled_token_ids
                del medusa_outputs[0]
                del medusa_outputs
                torch.cuda.empty_cache()
            print("\n🔮 Medusa predictions:")
            print_tensor_info("pred_medusa", pred_medusa)
            
            state.preds = torch.cat((pred_base, pred_medusa), dim=-1)
            state.valid_len += 1
            state.accept_lengths.append(1)
            
            # Debug prints
            print(f"\n🎯 [Init] Base={self.tokenizer.decode(pred_base[0].tolist())} | "
                  f"Medusa={[self.tokenizer.decode([t.item()]) for t in pred_medusa[0]]}")
            cache_length = state.past_key_values.get_seq_length()
            print("📦 Cache length:", cache_length)
            
        else:
            print("\n🔄 Subsequent token generation:")
            # Subsequent iterations
            print("  Running base model forward pass...")
            with torch.no_grad():
                outputs = self.model(
                    state.preds,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                # Get required tensors and immediately delete the outputs
                hidden_states = outputs.hidden_states[-1].detach()  # Detach to break computation graph
                logits = outputs.logits.detach()  # Detach to break computation graph
                state.past_key_values = outputs.past_key_values
                del outputs  # Free memory from outputs object
            
            print("\n📊 Base model outputs:")
            print_tensor_info("hidden_states", hidden_states)
            print_tensor_info("logits", logits)
            
            # Base prediction
            pred_base = torch.argmax(logits[:, :], dim=-1)
            print("\n🎯 Base model prediction:")
            print_tensor_info("pred_base", pred_base)
            decoded_base = self.tokenizer.decode(pred_base[0, :].tolist(), skip_special_tokens=True)
            print('📝 Base:', decoded_base)
            
            # Medusa predictions
            print("\n🔮 Getting Medusa predictions...")
            sampling_metadata = self._prepare_medusa_metadata(seq_group)
            with torch.no_grad():
                medusa_outputs = self.medusa_model.generate_proposals(hidden_states, sampling_metadata)
                # Extract and detach predictions
                pred_medusa = medusa_outputs[0].sampled_token_ids.unsqueeze(0).detach()
                # Clean up outputs
                del medusa_outputs[0].sampled_token_ids
                del medusa_outputs[0]
                del medusa_outputs
                torch.cuda.empty_cache()
            print("\n🔮 Medusa predictions:")
            print_tensor_info("pred_medusa", pred_medusa)
            
            # Verify predictions
            print("\n🔍 Verifying predictions...")
            posterior_mask = (state.preds[:, 1:] == pred_base[:, :-1]).int()
            print("  Posterior mask:")
            print_tensor_info("posterior_mask", posterior_mask)
            
            accept_length = torch.cumprod(posterior_mask, dim=-1).sum().item()
            print(f"  Accept length: {accept_length}")
            state.valid_len += accept_length + 1
            
            # Update KV cache - keep the accepted tokens plus the new base prediction
            print("\n📦 Updating KV cache...")
            cache_length = state.past_key_values.get_seq_length()
            if accept_length < cache_length:
                for layer_idx in range(len(state.past_key_values.layers)):
                    layer_cache = state.past_key_values.layers[layer_idx]
                    if layer_cache.keys is not None and layer_cache.values is not None:
                        # Keep accepted tokens plus one new token
                        old_keys = layer_cache.keys
                        old_values = layer_cache.values
                        layer_cache.keys = layer_cache.keys[:, :, :state.valid_len, :]
                        layer_cache.values = layer_cache.values[:, :, :state.valid_len, :]
                        # Clean up old tensors
                        del old_keys
                        del old_values
                        torch.cuda.empty_cache()
            new_cache_length = state.past_key_values.get_seq_length()
            print(f"  Old cache length: {cache_length}")
            print(f"  New cache length: {new_cache_length}")
            assert(new_cache_length == state.valid_len), f"Cache length {new_cache_length} != Valid length {state.valid_len}"
            
            # Update predictions and track acceptance
            print("\n🔄 Updating predictions...")
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
            
            print(f"\n✨ [Step {len(state.accept_lengths)}] Accepted={decoded_base} | Accept={accept_length}")
            
        # Print final state
        print("\n📋 Final state for this step:")
        state.print_state()

    def _generate_token(self, seq_group: SequenceGroup, prompt_tokens: List[int]) -> MedusaRequestOutput:
        """Override token generation to include Medusa speculative decoding."""
        print("\n🔄 Generating token with Medusa...")
        
        # Clean up old sequence states
        current_seq_ids = {seq.seq_id for seq in seq_group.seqs}
        for seq_id in list(self.seq_states.keys()):
            if seq_id not in current_seq_ids:
                state = self.seq_states.pop(seq_id)
                # Clean up state resources
                if state.past_key_values is not None:
                    del state.past_key_values
                if state.preds is not None:
                    del state.preds
                if state.accepted_candidates:
                    state.accepted_candidates.clear()
                del state
                gc.collect()
                torch.cuda.empty_cache()
                print(f"🧹 Cleaned up state for sequence {seq_id}")
        
        self._process_sequence_group(seq_group, prompt_tokens)
        
        # Get the next token from the accepted candidates
        seq_id = seq_group.seqs[0].seq_id
        state = self.seq_states[seq_id]
        
        if state.accepted_candidates:
            next_token = state.accepted_candidates[-1].item()
            print("✅ Using accepted candidate token:", next_token)
            print("  Decoded:", self.tokenizer.decode([next_token]))
        else:
            # Fallback to base model if no accepted candidates
            print("⚠️ No accepted candidates, falling back to base model")
            output = super()._generate_token(seq_group, prompt_tokens)
            next_token = output.outputs[0].token_ids[-1]
            print("  Base model token:", next_token)
            print("  Decoded:", self.tokenizer.decode([next_token]))
            
        # Include Medusa metrics in the output
        medusa_metrics = None
        if state.avg_accept_length is not None:
            medusa_metrics = {
                "avg_accept_length": state.avg_accept_length
            }
            print(f"📊 Medusa metrics: {medusa_metrics}")
            
        return MedusaRequestOutput(
            request_id=seq_group.request_id,
            prompt_token_ids=prompt_tokens,
            outputs=[Sequence(seq_id=seq_id, token_ids=[next_token])],
            medusa_metrics=medusa_metrics
        )

class AsyncMedusaLLMEngine(AsyncLLMEngine):
    def __init__(self, vllm_config: VllmConfig, executor_class=None, log_stats: bool = False):
        print("\n🚀 Initializing AsyncMedusaLLMEngine...")
        self._engine_class = MedusaLLMEngine
        super().__init__(vllm_config=vllm_config, executor_class=executor_class, log_stats=log_stats)
        print("✅ AsyncMedusaLLMEngine initialized")

    @classmethod
    def from_vllm_config(cls, vllm_config: VllmConfig, executor_class=None, log_stats: bool = False) -> "AsyncMedusaLLMEngine":
        return cls(vllm_config, executor_class=executor_class, log_stats=log_stats)