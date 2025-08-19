#!/usr/bin/env python3
"""
Direct Medusa Usage with Proper vLLM Initialization

This script properly initializes vLLM's distributed environment before
using the Medusa class directly.
"""

import argparse
import json
import os
import sys
import time
import torch
import atexit

try:
    from vllm.model_executor.models.medusa import Medusa
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel, destroy_distributed_environment, destroy_model_parallel
    from vllm.config import VllmConfig
    from vllm.model_executor.models.medusa import ResidualBlock
except ImportError as e:
    print(f"❌ vLLM import error: {e}")
    print("Please make sure you're in the correct vLLM environment")
    sys.exit(1)


def cleanup_vllm_environment():
    """Clean up vLLM's distributed environment."""
    try:
        print("🧹 Cleaning up vLLM distributed environment...")
        destroy_model_parallel()
        destroy_distributed_environment()
        print("✅ Cleanup completed successfully!")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def initialize_vllm_environment():
    """Initialize vLLM's distributed environment."""
    print("🔄 Initializing vLLM distributed environment...")
    
    try:
        # Set up basic distributed environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        
        # Initialize distributed environment
        init_distributed_environment()
        print("✅ vLLM distributed environment initialized successfully!")
        
        # Initialize model parallel groups
        print("🔄 Initializing model parallel groups...")
        initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1
        )
        print("✅ Model parallel groups initialized successfully!")
        
        # Register cleanup function
        atexit.register(cleanup_vllm_environment)
        
    except Exception as e:
        print(f"❌ Failed to initialize vLLM environment: {e}")
        raise


def create_vllm_config(config_data: dict):
    """Create a proper VllmConfig object for Medusa."""
    print("🔄 Creating VllmConfig for Medusa...")
    
    # Create a mock VllmConfig with the required structure
    class MockVllmConfig:
        def __init__(self, config_dict):
            self.speculative_config = type('obj', (object,), {
                'draft_model_config': type('obj', (object,), {
                    'hf_config': type('obj', (object,), {
                        'hidden_size': config_dict.get('hidden_size', 4096),
                        'vocab_size': config_dict.get('vocab_size', 128256),
                        'truncated_vocab_size': config_dict.get('vocab_size', 128256),
                        'medusa_fc_bias': config_dict.get('medusa_fc_bias', False),
                        'logit_scale': config_dict.get('logit_scale', 1.0),
                        'original_lm_head': config_dict.get('original_lm_head', False),
                        'num_heads': config_dict.get('num_heads', 5),
                        'num_hidden_layers': config_dict.get('num_hidden_layers', 3)
                    })()
                })()
            })()
    
    config = MockVllmConfig(config_data)
    print("✅ VllmConfig created successfully!")
    return config


def load_medusa_model_directly(checkpoint_path: str, config_path: str):
    """Load Medusa model directly using the Medusa class."""
    print("🔄 Loading Medusa model directly with Medusa class...")
    
    # Load config
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    print(f"📋 Config loaded: {config_data['num_heads']} heads, {config_data['num_hidden_layers']} layers")
    
    # Create VllmConfig
    vllm_config = create_vllm_config(config_data)
    
    # Initialize vLLM environment
    initialize_vllm_environment()
    
    try:
        # Create Medusa model instance
        print("🎯 Creating Medusa model instance using vllm.model_executor.models.medusa.Medusa")
        medusa_model = Medusa(vllm_config=vllm_config, prefix="")
        print("✅ Medusa model instance created successfully!")
        
        # Load weights
        print("🔄 Loading converted weights into Medusa model...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Filter out config from weights
        weights = []
        for key, value in checkpoint.items():
            if key != 'config':
                weights.append((key, value))
        
        print(f"🔄 Loading {len(weights)} weight parameters...")
        loaded_params = medusa_model.load_weights(weights)
        print(f"✅ Loaded {len(loaded_params)} parameters successfully!")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            medusa_model = medusa_model.cuda()
            print("✅ Medusa model moved to GPU")
        else:
            print("⚠️  GPU not available, using CPU")
        
        return medusa_model
        
    except Exception as e:
        print(f"❌ Failed to load Medusa model: {e}")
        raise


def run_medusa_inference_directly(medusa_model, prompt: str, max_tokens: int = 50):
    """Run inference directly using the Medusa model."""
    print("🎯 Running direct Medusa inference...")
    
    if medusa_model is None:
        raise RuntimeError("Medusa model not loaded")
    
    try:
        # Create dummy hidden states (you'd normally get these from a base model)
        batch_size = 1
        seq_len = len(prompt.split())  # Simple tokenization
        hidden_size = 4096  # From your config
        
        print(f"📝 Creating dummy hidden states: batch_size={batch_size}, seq_len={seq_len}, hidden_size={hidden_size}")
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        if torch.cuda.is_available():
            hidden_states = hidden_states.cuda()
        
        # Run forward pass
        print("🔄 Running Medusa forward pass...")
        with torch.no_grad():
            medusa_outputs = medusa_model.forward(hidden_states)
        
        print(f"✅ Medusa forward pass completed!")
        print(f"   Number of heads: {len(medusa_outputs)}")
        for i, output in enumerate(medusa_outputs):
            print(f"   Head {i}: {output.shape}")
        
        # This is a demonstration - in practice you'd need to:
        # 1. Tokenize the input properly
        # 2. Get hidden states from a base model
        # 3. Process the Medusa outputs to generate tokens
        
        return medusa_outputs
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        raise


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Direct Medusa usage with proper vLLM initialization")
    parser.add_argument("--checkpoint", default="./vllm_medusa_heads.pt", 
                       help="Path to converted Medusa checkpoint")
    parser.add_argument("--config", default="./vllm_medusa_config.json",
                       help="Path to vLLM config file")
    parser.add_argument("--prompt", default="Hello, how are you today?", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    try:
        print("=" * 60)
        print("Direct Medusa Usage with vLLM Initialization")
        print("=" * 60)
        print(f"Using: from vllm.model_executor.models.medusa import Medusa")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Config: {args.config}")
        print("=" * 60)
        
        # Load model directly
        medusa_model = load_medusa_model_directly(args.checkpoint, args.config)
        
        # Run inference
        outputs = run_medusa_inference_directly(medusa_model, args.prompt, args.max_tokens)
        
        print("✅ Direct Medusa usage completed successfully!")
        print(f"📊 Output shapes: {[out.shape for out in outputs]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
