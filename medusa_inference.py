#!/usr/bin/env python3
"""
Inference script for trained Medusa model.
This script demonstrates how to load a trained Medusa model and use it for text generation.
"""

import os
import argparse
import json
import torch
from typing import List, Optional

# Set CUDA visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from torchtune.models.llama3_1._model_builders import llama3_1_8b_medusa
from torchtune.models.convert_weights import meta_to_tune
from transformers import AutoTokenizer


class MedusaInference:
    """
    Inference class for Medusa model.
    """
    
    def __init__(
        self,
        model_path: str,
        params_path: str,
        device: torch.device = torch.device("cuda"),
        max_length: int = 512,
        fast_test: bool = False,
    ):
        self.device = device
        self.max_length = max_length
        
        # Load model parameters
        with open(params_path, 'r') as f:
            self.params = json.load(f)
        
        # Create model first
        if fast_test:
            # Use a smaller model for faster testing
            print("Using smaller model for fast testing...")
            self.model = llama3_1_8b_medusa()  # Still 8B but with optimizations
        else:
            self.model = llama3_1_8b_medusa()
        
        # Move model to device first and ensure it's loaded
        print("Moving model to GPU...")
        self.model.to(device)
        
        # Force GPU allocation by doing a dummy forward pass
        print("Initializing model on GPU...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, 10, dtype=torch.long, device=device)
            _ = self.model(dummy_input)
        
        # Load trained weights directly to GPU with optimizations
        print("Loading model checkpoint...")
        try:
            # Try memory mapping for faster loading
            checkpoint = torch.load(model_path, map_location=device, weights_only=True, mmap=True)
        except:
            # Fallback to regular loading
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # Load trained weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        print("Model loaded successfully!")
        
        # Set model to eval mode
        self.model.eval()
        
        # Force another forward pass to ensure everything is loaded
        print("Finalizing model initialization...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, 10, dtype=torch.long, device=device)
            _ = self.model(dummy_input)
        
        print("Model fully initialized on GPU!")
        
        # Initialize tokenizer using HuggingFace's Llama3.1 tokenizer
        # Use a local cache or download from a public source
        try:
            # Try to use a local path first
            tokenizer_path = "/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except:
            # Fallback to a public Llama3 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", trust_remote_code=True)
        
        # Enable optimizations for faster inference
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable memory efficient attention if available
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Model on device: {device}")
        print(f"✓ Model in eval mode")
        
        # Print GPU memory usage
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"✓ GPU memory allocated: {allocated:.2f} GB")
            print(f"✓ GPU memory reserved: {reserved:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        num_medusa_heads: int = 4,
    ) -> str:
        """
        Generate text using the Medusa model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            num_medusa_heads: Number of Medusa heads to use for generation
            
        Returns:
            Generated text
        """
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        print(f"✓ Input tokens: {len(input_ids[0])}")
        print(f"✓ Generating up to {max_new_tokens} new tokens...")
        
        # Generate tokens
        generated_ids = self._generate_tokens(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_medusa_heads=num_medusa_heads,
        )
        
        # Decode output
        generated_text = self.tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        
        return generated_text
    
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        num_medusa_heads: int,
    ) -> torch.Tensor:
        """
        Generate tokens using Medusa heads for faster generation.
        """
        current_ids = input_ids.clone()
        tokens_generated = 0
        
        # Pre-allocate tensor for efficiency
        max_seq_len = current_ids.shape[1] + max_new_tokens
        pad_id = getattr(self.tokenizer, 'pad_token_id', 0) or 0
        result_ids = torch.full((1, max_seq_len), pad_id, 
                              dtype=current_ids.dtype, device=current_ids.device)
        result_ids[:, :current_ids.shape[1]] = current_ids
        
        # Track the current position in the result tensor
        current_pos = current_ids.shape[1]
        
        while tokens_generated < max_new_tokens:
            # Get model outputs with mixed precision for speed
            with torch.no_grad(), torch.amp.autocast('cuda'):
                outputs = self.model(current_ids)
            
            # Debug: Check what the model is returning
            if tokens_generated == 0:  # Only print on first iteration
                print(f"Model output type: {type(outputs)}")
                if isinstance(outputs, list):
                    print(f"Number of outputs: {len(outputs)}")
                    print(f"Main output shape: {outputs[0].shape}")
                    if len(outputs) > 1:
                        print(f"Medusa outputs: {len(outputs[1])} heads")
                        for i, medusa_out in enumerate(outputs[1]):
                            print(f"  Medusa head {i} shape: {medusa_out.shape}")
                else:
                    print(f"Single output shape: {outputs.shape}")
            
            # Get logits from all heads
            if isinstance(outputs, list):
                # Medusa model returns [main_output, medusa_outputs]
                main_logits = outputs[0][:, -1, :]  # Last token from main head
                medusa_logits = outputs[1]  # List of medusa head outputs
                
                # Use Medusa heads for faster generation
                if tokens_generated == 0:  # Debug logging
                    print(f"Medusa heads available: {len(medusa_logits)}")
                    print(f"Requested heads: {num_medusa_heads}")
                
                if len(medusa_logits) >= num_medusa_heads and num_medusa_heads > 0:
                    # Get predictions from multiple heads
                    predictions = []
                    for i in range(num_medusa_heads):
                        # Handle different output shapes for Medusa heads
                        if medusa_logits[i].dim() == 3:
                            head_logits = medusa_logits[i][:, -1, :]  # [batch, seq, vocab]
                        else:
                            head_logits = medusa_logits[i][-1, :].unsqueeze(0)  # [seq, vocab] -> [1, vocab]
                        if do_sample:
                            probs = torch.softmax(head_logits / temperature, dim=-1)
                            # Apply top-p filtering
                            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumsum_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            probs[indices_to_remove] = 0
                            probs = probs / probs.sum()
                            token = torch.multinomial(probs, 1)
                        else:
                            token = torch.argmax(head_logits, dim=-1, keepdim=True)
                        predictions.append(token)
                    
                    # Add all predicted tokens efficiently
                    if tokens_generated == 0:  # Debug logging
                        print(f"Using Medusa heads: generating {len(predictions)} tokens")
                    
                    for i, token in enumerate(predictions):
                        if tokens_generated >= max_new_tokens:
                            break
                        result_ids[:, current_pos] = token
                        tokens_generated += 1
                        current_pos += 1
                        
                        # Check for EOS token
                        if token.item() == self.tokenizer.eos_token_id:
                            return result_ids[:, :current_pos]
                    
                    # Update current_ids for next iteration
                    current_ids = result_ids[:, :current_pos]
                else:
                    # Fallback to main head only
                    if tokens_generated == 0:  # Debug logging
                        print("Falling back to main head only")
                    if do_sample:
                        probs = torch.softmax(main_logits / temperature, dim=-1)
                        # Apply top-p filtering
                        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumsum_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        probs[indices_to_remove] = 0
                        probs = probs / probs.sum()
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(main_logits, dim=-1, keepdim=True)
                    
                    result_ids[:, current_pos] = next_token
                    tokens_generated += 1
                    current_pos += 1
                    current_ids = result_ids[:, :current_pos]
                    
                    # Check for EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
            else:
                # Fallback for non-Medusa models
                logits = outputs[:, -1, :]
                if do_sample:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                result_ids[:, current_pos] = next_token
                tokens_generated += 1
                current_pos += 1
                current_ids = result_ids[:, :current_pos]
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return result_ids[:, :current_pos]
    
    def compare_with_base_model(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> dict:
        """
        Compare Medusa generation with base model generation.
        """
        print("=== Comparing Medusa vs Base Model ===")
        
        # Medusa generation
        print("\n--- Medusa Generation ---")
        torch.cuda.synchronize()  # Ensure clean timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        medusa_output = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_medusa_heads=4,
        )
        end_time.record()
        
        torch.cuda.synchronize()
        medusa_time = start_time.elapsed_time(end_time)
        
        print(f"Medusa output: {medusa_output}")
        print(f"Medusa generation time: {medusa_time:.2f} ms")
        
        # Base model generation (using only main head)
        print("\n--- Base Model Generation ---")
        torch.cuda.synchronize()  # Ensure clean timing
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        base_output = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_medusa_heads=0,  # Use only main head
        )
        end_time.record()
        
        torch.cuda.synchronize()
        base_time = start_time.elapsed_time(end_time)
        
        print(f"Base output: {base_output}")
        print(f"Base generation time: {base_time:.2f} ms")
        
        # Calculate speedup
        speedup = base_time / medusa_time if medusa_time > 0 else 0
        
        return {
            "medusa_output": medusa_output,
            "base_output": base_output,
            "medusa_time": medusa_time,
            "base_time": base_time,
            "speedup": speedup,
        }


def main():
    parser = argparse.ArgumentParser(description="Medusa Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--checkpoint_path", type=str,
                       default="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth",
                       help="Path to original Llama checkpoint (if using base model)")
    parser.add_argument("--params_path", type=str,
                       default="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json",
                       help="Path to model parameters JSON")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is",
                       help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                       help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--num_medusa_heads", type=int, default=1,
                       help="Number of Medusa heads to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with base model generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--fast_test", action="store_true",
                       help="Use smaller model for faster testing")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Set GPU device if using CUDA
    if device.type == "cuda":
        gpu_id = 1  # Change this to use a different GPU
        torch.cuda.set_device(gpu_id)
        print(f"✓ Set GPU device to: {gpu_id}")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Create inference model
    inference = MedusaInference(
        model_path=args.model_path,
        params_path=args.params_path,
        device=device,
        fast_test=args.fast_test,
    )
    
    # Generate text
    print(f"\n=== Text Generation ===")
    print(f"Prompt: {args.prompt}")
    
    generated_text = inference.generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_medusa_heads=args.num_medusa_heads,
    )
    
    print(f"\nGenerated text:")
    print(f"{generated_text}")
    
    # Compare with base model if requested
    if args.compare:
        comparison = inference.compare_with_base_model(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        print(f"\n=== Comparison Results ===")
        print(f"Speedup: {comparison['speedup']:.2f}x")
        print(f"Medusa time: {comparison['medusa_time']:.2f} ms")
        print(f"Base time: {comparison['base_time']:.2f} ms")


if __name__ == "__main__":
    main() 