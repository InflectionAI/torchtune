#!/usr/bin/env python3
"""
Simple vLLM Medusa Speculative Decoding Demo
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.model_executor.models.medusa import Medusa
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
    destroy_distributed_environment,
    destroy_model_parallel
)

def initialize_vllm():
    """Initialize vLLM's distributed environment."""
    print("🔧 Initializing vLLM...")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    
    # Initialize distributed environment
    init_distributed_environment()
    initialize_model_parallel(1, 1)  # tensor_parallel_size, pipeline_parallel_size
    print("✅ vLLM initialized!")

def cleanup_vllm():
    """Clean up vLLM's distributed environment."""
    try:
        destroy_model_parallel()
        destroy_distributed_environment()
        print("🧹 vLLM cleaned up!")
    except Exception as e:
        print(f"⚠️ Warning during cleanup: {e}")

def load_models(base_model_path: str, medusa_checkpoint: str):
    """Load base model and Medusa model."""
    print("🔄 Loading models...")
    
    # Set device and dtype
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    # Load base model directly to GPU
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map={"": 0},  # Map all modules to GPU 0
        low_cpu_mem_usage=True  # Load directly to GPU
    ).to(device)
    
    # Load Medusa checkpoint
    if os.path.isdir(medusa_checkpoint):
        pt_files = [f for f in os.listdir(medusa_checkpoint) if f.endswith('.pt')]
        if pt_files:
            checkpoint_path = os.path.join(medusa_checkpoint, pt_files[0])
        else:
            raise FileNotFoundError(f"No .pt files found in {medusa_checkpoint}")
    else:
        checkpoint_path = medusa_checkpoint
    
    # Load checkpoint directly to GPU
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create Medusa config
    config = type('obj', (object,), {
        'hidden_size': 4096,
        'vocab_size': 128256,
        'truncated_vocab_size': 128256,
        'medusa_fc_bias': False,
        'logit_scale': 1.0,
        'original_lm_head': False,
        'num_heads': 5,
        'num_hidden_layers': 3,
        'torch_dtype': dtype,
        'speculative_config': type('obj', (object,), {
            'draft_model_config': type('obj', (object,), {
                'hf_config': type('obj', (object,), {
                    'hidden_size': 4096,
                    'vocab_size': 128256,
                    'truncated_vocab_size': 128256,
                    'medusa_fc_bias': False,
                    'logit_scale': 1.0,
                    'original_lm_head': False,
                    'num_heads': 5,
                    'num_hidden_layers': 3
                })()
            })()
        })()
    })()
    
    # Create Medusa model directly on GPU
    with torch.device(device):
        medusa_model = Medusa(vllm_config=config, prefix="").to(device=device, dtype=dtype)
        
        # Load weights (already on GPU from checkpoint loading)
        weights = [(k, v) for k, v in checkpoint.items() if k != 'config']
        medusa_model.load_weights(weights)
    
    print("✅ Models loaded successfully!")
    print(f"   Base model device: {next(base_model.parameters()).device}")
    print(f"   Base model dtype: {next(base_model.parameters()).dtype}")
    print(f"   Medusa model device: {next(medusa_model.parameters()).device}")
    print(f"   Medusa model dtype: {next(medusa_model.parameters()).dtype}")
    return base_tokenizer, base_model, medusa_model

def run_inference(prompt: str, base_tokenizer, base_model, medusa_model, max_tokens: int = 20):
    """Run speculative decoding with Medusa."""
    print(f"\n📝 Processing prompt:")
    print(f"   {prompt}")
    
    # Get device and dtype from models
    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    
    # Tokenize input and move to device
    inputs = base_tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
    inputs = {k: v.to(device=device) for k, v in inputs.items()}
    
    # Get initial hidden states
    with torch.no_grad():
        outputs = base_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        # Convert to same dtype as Medusa model
        hidden_states = hidden_states.to(dtype=dtype)
    
    # Setup for generation
    current_hidden_states = hidden_states
    tokens_generated = 0
    head_stats = {i: {'correct': 0, 'total': 0} for i in range(medusa_model.config.num_heads)}
    
    while tokens_generated < max_tokens:
        # Get base model prediction
        with torch.no_grad():
            base_outputs = base_model(**inputs)
            next_token_logits = base_outputs.logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            next_token_prob = torch.softmax(next_token_logits, dim=-1)[next_token_id].item()
            next_token_text = base_tokenizer.decode([next_token_id])
        
        print(f"\n📊 Forward Pass {tokens_generated + 1}")
        print("=" * 60)
        print(f"\n🔵 Base Model Output:")
        print(f"   Next Token: {next_token_text}")
        print(f"   Token ID: {next_token_id}")
        print(f"   Probability: {next_token_prob:.4f}")
        
        # Get Medusa predictions
        sampling_metadata = SamplingMetadata(
            seq_groups=[type('SeqGroup', (), {
                'sample_indices': [0],
                'seq_ids': [0],
                'num_seqs': 1,
                'prompt_logprob_indices': [],
                'sampling_params': type('SamplingParams', (), {
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 50,
                    'use_beam_search': False,
                    'logprobs': None,
                    'logits_processors': []
                })()
            })()],
            selected_token_indices=torch.tensor([0], device=current_hidden_states.device, dtype=torch.long),
            categorized_sample_indices={'temperature': [0], 'top_p': [0], 'top_k': [0]},
            num_prompts=1
        )
        
        with torch.no_grad():
            proposals = medusa_model.generate_proposals(
                previous_hidden_states=current_hidden_states,
                sampling_metadata=sampling_metadata
            )
        
        print("\n🟢 Medusa Head Outputs:")
        for i, proposal in enumerate(proposals):
            if hasattr(proposal, 'sampled_token_ids') and proposal.sampled_token_ids is not None:
                medusa_tokens = proposal.sampled_token_ids  # Shape: [num_heads, seq_len]
                medusa_probs = proposal.sampled_token_probs if hasattr(proposal, 'sampled_token_probs') else None
                
                for head_idx in range(medusa_tokens.shape[0]):
                    token_id = medusa_tokens[head_idx, -1].item()
                    token_text = base_tokenizer.decode([token_id])
                    prob = medusa_probs[head_idx, -1, token_id].item() if medusa_probs is not None else None
                    
                    print(f"   Head {head_idx + 1}:")
                    print(f"      Token: {token_text}")
                    print(f"      Token ID: {token_id}")
                    if prob is not None:
                        print(f"      Probability: {prob:.4f}")
        
        print("\n🔍 Verification:")
        for i, proposal in enumerate(proposals):
            if hasattr(proposal, 'sampled_token_ids') and proposal.sampled_token_ids is not None:
                medusa_tokens = proposal.sampled_token_ids
                medusa_probs = proposal.sampled_token_probs if hasattr(proposal, 'sampled_token_probs') else None
                
                for head_idx in range(medusa_tokens.shape[0]):
                    token_id = medusa_tokens[head_idx, -1].item()
                    prob = medusa_probs[head_idx, -1, token_id].item() if medusa_probs is not None else None
                    
                    head_stats[head_idx]['total'] += 1
                    if token_id == next_token_id:
                        head_stats[head_idx]['correct'] += 1
                        print(f"   ✅ Head {head_idx + 1}:")
                        print(f"      Correctly predicted: {base_tokenizer.decode([token_id])}")
                    else:
                        print(f"   ❌ Head {head_idx + 1}:")
                        print(f"      Predicted: {base_tokenizer.decode([token_id])}")
                        print(f"      Expected: {base_tokenizer.decode([next_token_id])}")
                        print(f"      Token IDs: {token_id} vs {next_token_id}")
                    if prob is not None:
                        print(f"      Confidence: {prob:.4f}")
        
        print("\n   Head Performance:")
        for head_idx, stats in head_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   Head {head_idx + 1}: {stats['correct']}/{stats['total']} correct ({accuracy:.2%})")
        
        # Update inputs for next iteration
        inputs['input_ids'] = torch.cat([
            inputs['input_ids'],
            torch.tensor([[next_token_id]], device=device)
        ], dim=1)
        inputs['attention_mask'] = torch.cat([
            inputs['attention_mask'],
            torch.ones((1, 1), device=device)
        ], dim=1)
        
        # Get hidden states for next iteration
        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)
            current_hidden_states = outputs.hidden_states[-1][:, -1:, :]
            if current_hidden_states.dtype != dtype:
                current_hidden_states = current_hidden_states.to(dtype=dtype)
        
        tokens_generated += 1
        print("\n" + "=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Simple Medusa Speculative Decoding Demo")
    parser.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. The weather today is",
                      help="Prompt to test")
    parser.add_argument("--medusa_checkpoint", default="./vllm_medusa_model",
                      help="Path to Medusa checkpoint")
    parser.add_argument("--base_model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                      help="Base model path")
    parser.add_argument("--max_tokens", type=int, default=5,
                      help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    try:
        # Initialize vLLM
        initialize_vllm()
        
        # Load models
        base_tokenizer, base_model, medusa_model = load_models(args.base_model, args.medusa_checkpoint)
        
        # Run inference
        run_inference(args.prompt, base_tokenizer, base_model, medusa_model, args.max_tokens)
        
    finally:
        # Clean up vLLM
        cleanup_vllm()

if __name__ == "__main__":
    main()
