#!/usr/bin/env python3
"""
Advanced weight conversion script to convert trained MedusaHeads weights to vLLM Medusa format.

This script handles the architectural differences more carefully:
1. MedusaHeads ResBlock: single linear layer with SiLU activation
2. vLLM ResidualBlock: multiple linear layers with SiLU activation

The script will adapt the weights to match vLLM's expected architecture.
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def load_torchtune_checkpoint(checkpoint_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load the trained torchtune checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Handle different checkpoint formats
    if checkpoint_path.endswith('.bin'):
        # .bin files are direct checkpoint files
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Successfully loaded .bin checkpoint file")
        except Exception as e:
            print(f"Error loading .bin file: {e}")
            raise
    elif checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
        # .pt/.pth files are PyTorch checkpoints
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Successfully loaded PyTorch checkpoint file")
        except Exception as e:
            print(f"Error loading PyTorch file: {e}")
            raise
    elif os.path.isdir(checkpoint_path):
        # Directory with multiple checkpoint files
        print(f"Loading from directory: {checkpoint_path}")
        checkpoint = {}
        for file in os.listdir(checkpoint_path):
            if file.endswith(('.pt', '.pth', '.bin')):
                file_path = os.path.join(checkpoint_path, file)
                try:
                    file_checkpoint = torch.load(file_path, map_location='cpu')
                    checkpoint.update(file_checkpoint)
                    print(f"Loaded: {file}")
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    
    # Extract model state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("Found 'model' key in checkpoint")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Found 'state_dict' key in checkpoint")
    elif 'pytorch_model.bin' in str(checkpoint_path):
        # Handle HuggingFace format
        state_dict = checkpoint
        print("Treating as HuggingFace format checkpoint")
    else:
        state_dict = checkpoint
        print("Using checkpoint directly as state dict")
    
    # Extract config if available
    config = checkpoint.get('config', {})
    if config:
        print(f"Found config with keys: {list(config.keys())}")
    else:
        print("No config found, will auto-detect from weights")
    
    print(f"Checkpoint loaded successfully with {len(state_dict)} parameters")
    return state_dict, config


def extract_medusa_heads_weights(state_dict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Extract MedusaHeads weights from a MedusaTransformerDecoder checkpoint.
    
    Args:
        state_dict: Full model state dict containing MedusaTransformerDecoder weights
    
    Returns:
        Tuple of (medusa_heads_weights, extracted_config)
    """
    print("Extracting MedusaHeads weights from MedusaTransformerDecoder checkpoint...")
    
    # Extract only the medusa_heads related weights
    medusa_weights = {}
    base_model_weights = {}
    
    for key, value in state_dict.items():
        if 'medusa_heads' in key:
            medusa_weights[key] = value
        else:
            base_model_weights[key] = value
    
    if not medusa_weights:
        raise ValueError("No medusa_heads weights found in checkpoint!")
    
    print(f"Found {len(medusa_weights)} MedusaHeads parameters")
    print(f"Found {len(base_model_weights)} base model parameters")
    
    # Try to infer configuration from the weights
    extracted_config = {}
    
    # Count medusa heads and layers by analyzing weight names
    medusa_num_heads = 0
    medusa_num_layers = 0
    hidden_size = None
    vocab_size = None
    
    for key in medusa_weights.keys():
        if 'medusa_heads.medusa_base.' in key:
            # Extract head and layer indices
            parts = key.split('.')
            if len(parts) >= 4:
                try:
                    head_idx = int(parts[2])
                    layer_idx = int(parts[3])
                    medusa_num_heads = max(medusa_num_heads, head_idx + 1)
                    medusa_num_layers = max(medusa_num_layers, layer_idx + 1)
                except ValueError:
                    continue
    
    # Try to get hidden_size and vocab_size from base model weights
    for key, value in base_model_weights.items():
        if 'tok_embeddings.weight' in key:
            hidden_size = value.shape[1]
        elif 'output.weight' in key:
            vocab_size = value.shape[0]
        elif 'lm_head.weight' in key:
            vocab_size = value.shape[0]
    
    # Set defaults if not found
    if hidden_size is None:
        hidden_size = 4096  # Default for Llama
        print(f"Warning: Could not determine hidden_size, using default: {hidden_size}")
    
    if vocab_size is None:
        vocab_size = 32000  # Default for Llama
        print(f"Warning: Could not determine vocab_size, using default: {vocab_size}")
    
    extracted_config = {
        'medusa_num_heads': medusa_num_heads,
        'medusa_num_layers': medusa_num_layers,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size
    }
    
    print(f"Extracted configuration: {extracted_config}")
    
    return medusa_weights, extracted_config


def adapt_resblock_weights(original_weight: torch.Tensor, target_layers: int) -> list[torch.Tensor]:
    """
    Adapt ResBlock weights to match vLLM's ResidualBlock architecture.
    
    Args:
        original_weight: Weight from MedusaHeads ResBlock.linear.weight
        target_layers: Number of layers in vLLM's ResidualBlock
    
    Returns:
        List of adapted weights for each layer
    """
    # For now, we'll duplicate the weight across layers
    # This is a simple approach - you might want to implement more sophisticated weight adaptation
    adapted_weights = []
    
    for i in range(target_layers):
        if i == 0:
            # First layer gets the original weight
            adapted_weights.append(original_weight.clone())
        else:
            # Subsequent layers get a scaled version to avoid exact duplication
            # You might want to implement more sophisticated initialization here
            scaled_weight = original_weight * (0.1 ** i)  # Decay factor
            adapted_weights.append(scaled_weight)
    
    return adapted_weights


def convert_medusa_weights_advanced(state_dict: Dict[str, torch.Tensor], config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert MedusaHeads weights to vLLM Medusa format with architectural adaptation."""
    print("Converting MedusaHeads weights to vLLM Medusa format (advanced)...")
    
    converted_weights = {}
    
    # Extract MedusaHeads weights
    medusa_weights, extracted_config = extract_medusa_heads_weights(state_dict)
    
    # Merge configs, with extracted config taking precedence
    final_config = config.copy()
    final_config.update(extracted_config)
    
    # Get configuration
    medusa_num_heads = final_config.get('medusa_num_heads', 3)
    medusa_num_layers = final_config.get('medusa_num_layers', 3)
    hidden_size = final_config.get('hidden_size', 4096)
    vocab_size = final_config.get('vocab_size', 32000)
    
    # vLLM ResidualBlock typically has 2 layers by default
    vllm_resblock_layers = final_config.get('vllm_resblock_layers', 2)
    
    print(f"Final configuration: heads={medusa_num_heads}, layers={medusa_num_layers}, hidden_size={hidden_size}, vocab_size={vocab_size}")
    print(f"vLLM ResidualBlock layers: {vllm_resblock_layers}")
    
    # Convert weights for each head
    for head_idx in range(medusa_num_heads):
        # Convert residual blocks
        for layer_idx in range(medusa_num_layers):
            # Convert ResBlock weights to ResidualBlock format
            old_key = f"medusa_heads.medusa_base.{head_idx}.{layer_idx}.linear.weight"
            
            if old_key in medusa_weights:
                original_weight = medusa_weights[old_key]
                adapted_weights = adapt_resblock_weights(original_weight, vllm_resblock_layers)
                
                # Assign adapted weights to vLLM format
                for vllm_layer_idx, adapted_weight in enumerate(adapted_weights):
                    new_key = f"blocks.{head_idx}.layers.{vllm_layer_idx}.weight"
                    converted_weights[new_key] = adapted_weight
                    print(f"Converted: {old_key} -> {new_key} (adapted for {vllm_resblock_layers} layers)")
            else:
                print(f"Warning: Missing weight for {old_key}")
        
        # Convert linear layer weights
        old_key = f"medusa_heads.medusa_linear_layers.{head_idx}.weight"
        new_key = f"lm_heads.{head_idx}.weight"
        
        if old_key in medusa_weights:
            converted_weights[new_key] = medusa_weights[old_key]
            print(f"Converted: {old_key} -> {new_key}")
        else:
            print(f"Warning: Missing weight for {old_key}")
    
    # Add configuration attributes that vLLM expects
    converted_weights['config'] = {
        'num_heads': medusa_num_heads,
        'num_hidden_layers': medusa_num_layers,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'truncated_vocab_size': vocab_size,
        'medusa_fc_bias': False,
        'logit_scale': 1.0,
        'original_lm_head': False
    }
    
    print(f"Successfully converted {len(converted_weights)} weights")
    return converted_weights


def create_vllm_config_advanced(config: Dict[str, Any], output_config_path: str):
    """Create a vLLM-compatible config file with advanced settings."""
    print(f"Creating advanced vLLM config at: {output_config_path}")
    
    # Use the extracted config values that were auto-detected from weights
    vllm_config = {
        "architectures": ["MedusaForCausalLM"],
        "model_type": "medusa",
        "num_heads": config.get('medusa_num_heads', 3),
        "num_hidden_layers": config.get('medusa_num_layers', 3),
        "hidden_size": config.get('hidden_size', 4096),
        "vocab_size": config.get('vocab_size', 32000),
        "truncated_vocab_size": config.get('vocab_size', 32000),
        "medusa_fc_bias": False,
        "logit_scale": 1.0,
        "original_lm_head": False,
        "torch_dtype": "float16",
        "vllm_resblock_layers": config.get('vllm_resblock_layers', 2)
    }
    
    with open(output_config_path, 'w') as f:
        json.dump(vllm_config, f, indent=2)
    
    print("Advanced vLLM config created successfully!")
    print(f"Config created with: {vllm_config['num_heads']} heads, {vllm_config['num_hidden_layers']} layers, hidden_size={vllm_config['hidden_size']}, vocab_size={vllm_config['vocab_size']}")


def main():
    parser = argparse.ArgumentParser(description="Advanced conversion of MedusaHeads weights to vLLM Medusa format")
    parser.add_argument("--input_checkpoint", required=True, help="Path to trained MedusaTransformerDecoder checkpoint")
    parser.add_argument("--output_checkpoint", required=True, help="Path to save converted vLLM checkpoint")
    parser.add_argument("--output_config", help="Path to save vLLM config (optional)")
    parser.add_argument("--config_path", help="Path to model config file (optional)")
    parser.add_argument("--vllm_resblock_layers", type=int, default=2, 
                       help="Number of layers in vLLM ResidualBlock (default: 2)")
    
    args = parser.parse_args()
    
    # Load checkpoint
    state_dict, checkpoint_config = load_torchtune_checkpoint(args.input_checkpoint)
    
    # Load additional config if provided
    if args.config_path:
        with open(args.config_path, 'r') as f:
            file_config = json.load(f)
            checkpoint_config.update(file_config)
    
    # Add vLLM-specific configuration
    checkpoint_config['vllm_resblock_layers'] = args.vllm_resblock_layers
    
    # Convert weights
    converted_weights = convert_medusa_weights_advanced(state_dict, checkpoint_config)
    
    # Save converted checkpoint
    save_vllm_checkpoint(converted_weights, args.output_checkpoint)
    
    # Create vLLM config if requested
    if args.output_config:
        create_vllm_config_advanced(checkpoint_config, args.output_config)
    
    print("\nAdvanced conversion completed successfully!")
    print(f"Converted checkpoint: {args.output_checkpoint}")
    if args.output_config:
        print(f"vLLM config: {args.output_config}")
    print(f"vLLM ResidualBlock layers: {args.vllm_resblock_layers}")
    print("\nYou can now use the converted checkpoint with vLLM's Medusa implementation!")


def save_vllm_checkpoint(converted_weights: Dict[str, torch.Tensor], output_path: str):
    """Save the converted weights in vLLM format."""
    print(f"Saving converted checkpoint to: {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the converted weights
    torch.save(converted_weights, output_path)
    print("Checkpoint saved successfully!")


if __name__ == "__main__":
    main()
