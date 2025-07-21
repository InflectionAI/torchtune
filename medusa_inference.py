#!/usr/bin/env python3
"""
Inference script for trained Medusa model.
This script demonstrates how to load a trained Medusa model and use it for text generation.
"""

import argparse
import json
import torch
from typing import List, Optional

from torchtune.models.llama3_1._model_builders import llama3_1_8b_medusa
from torchtune.models.convert_weights import meta_to_tune
from torchtune.modules.transforms.tokenizers import Llama3Tokenizer


class MedusaInference:
    """
    Inference class for Medusa model.
    """
    
    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
        params_path: str,
        device: torch.device = torch.device("cuda"),
        max_length: int = 512,
    ):
        self.device = device
        self.max_length = max_length
        
        # Load model parameters
        with open(params_path, 'r') as f:
            self.params = json.load(f)
        
        # Create model
        self.model = llama3_1_8b_medusa()
        
        # Load trained weights
        if model_path.endswith('.pt'):
            # Load from training checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            # Load from original checkpoint and convert
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            converted_checkpoint = meta_to_tune(checkpoint)
            self.model.load_state_dict(converted_checkpoint, strict=False)
        
        # Move to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize tokenizer
        self.tokenizer = Llama3Tokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Model on device: {device}")
        print(f"✓ Model in eval mode")
    
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
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        
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
        
        for _ in range(max_new_tokens):
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(current_ids)
            
            # Get logits from all heads
            if isinstance(outputs, list):
                # Medusa model returns [main_output, medusa_outputs]
                main_logits = outputs[0][:, -1, :]  # Last token from main head
                medusa_logits = outputs[1]  # List of medusa head outputs
                
                # Use Medusa heads for faster generation
                if len(medusa_logits) >= num_medusa_heads:
                    # Get predictions from multiple heads
                    predictions = []
                    for i in range(num_medusa_heads):
                        head_logits = medusa_logits[i][:, -1, :]  # Last token from each head
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
                    
                    # Add all predicted tokens
                    for token in predictions:
                        current_ids = torch.cat([current_ids, token], dim=1)
                    
                    # Skip some iterations since we generated multiple tokens
                    if len(predictions) > 1:
                        continue
                else:
                    # Fallback to main head only
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
                    
                    current_ids = torch.cat([current_ids, next_token], dim=1)
            else:
                # Fallback for non-Medusa models
                logits = outputs[:, -1, :]
                if do_sample:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
            
            # Check for EOS token
            if current_ids[0, -1].item() == self.tokenizer.eos_id:
                break
        
        return current_ids
    
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
    parser.add_argument("--num_medusa_heads", type=int, default=4,
                       help="Number of Medusa heads to use")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with base model generation")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
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
        checkpoint_path=args.checkpoint_path,
        params_path=args.params_path,
        device=device,
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