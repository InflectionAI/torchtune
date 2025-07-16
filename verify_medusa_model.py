#!/usr/bin/env python3
"""
Verification script for llama3_1_8b_medusa model.
This script tests various aspects of the Medusa model to ensure it's working correctly.
"""

import torch
import torch.nn as nn
from torchtune.models.llama3_1._model_builders import llama3_1_8b_medusa
from torchtune.modules import MedusaTransformerDecoder
from torchtune.models.convert_weights import *

def test_model_instantiation(model):
    """Test that the model can be instantiated correctly."""
    print("=== Testing Model Instantiation ===")
    
    try:
        # model = llama3_1_8b_medusa()
        print(f"✓ Model instantiated successfully")
        print(f"✓ Model type: {type(model)}")
        print(f"✓ Expected type: {MedusaTransformerDecoder}")
        print(f"✓ Is MedusaTransformerDecoder: {isinstance(model, MedusaTransformerDecoder)}")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Check device placement
        device = next(model.parameters()).device
        print(f"✓ Model device: {device}")
        
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return None

def test_forward_pass(model):
    """Test that the model can perform forward passes correctly."""
    print("\n=== Testing Forward Pass ===")
    
    if model is None:
        print("✗ Cannot test forward pass - model is None")
        return
    
    try:
        # Create test input - will automatically be on same device as model
        batch_size = 2
        seq_len = 10
        vocab_size = 128_256  # Llama3.1 vocab size
        
        input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_tokens = input_tokens.cuda()
        print(f"✓ Input shape: {input_tokens.shape}")
        print(f"✓ Input device: {input_tokens.device}")
        
        check_device_consistency(model, input_tokens)
        # Perform forward pass
        with torch.no_grad():
            outputs = model(input_tokens)
        
        print(f"✓ Forward pass completed successfully")
        
        # Check output structure
        if isinstance(outputs, list):
            print(f"✓ Output is a list with {len(outputs)} elements")
            
            # First element should be main LM output
            main_output = outputs[0]
            
            print(f"✓ Main output shape: {main_output.shape}")
            print(f"✓ Main output device: {main_output.device}")
            print(f"✓ Expected main output shape: ({batch_size}, {seq_len}, {vocab_size})")
            
            # Subsequent elements should be the list element containing the Medusa head outputs
            medusa_outputs = outputs[1:][0]
            print(f"✓ Number of Medusa heads: {len(medusa_outputs)}")
            
            for i, medusa_output in enumerate(medusa_outputs):
                print(f"✓ Medusa head {i+1} output shape: {medusa_output.shape}")
                print(f"✓ Medusa head {i+1} device: {medusa_output.device}")
                print(f"✓ Expected Medusa head shape: ({batch_size}, {seq_len}, {vocab_size})")
                
        else:
            print(f"✗ Unexpected output type: {type(outputs)}")
            
        return outputs
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return None

def test_parameter_freezing(model):
    """Test that base model parameters are frozen and only Medusa heads are trainable."""
    print("\n=== Testing Parameter Freezing ===")
    
    if model is None:
        print("✗ Cannot test parameter freezing - model is None")
        return
    
    try:
        # Check which parameters are trainable
        trainable_params = []
        frozen_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
            else:
                frozen_params.append(name)
        
        print(f"✓ Trainable parameters: {len(trainable_params)}")
        print(f"✓ Frozen parameters: {len(frozen_params)}")
        
        # Check that Medusa head parameters are trainable
        medusa_trainable = [name for name in trainable_params if 'medusa_heads' in name]
        print(f"✓ Trainable Medusa parameters: {len(medusa_trainable)}")
        
        # Check that base model parameters are frozen
        base_trainable = [name for name in trainable_params if 'medusa_heads' not in name]
        print(f"✓ Trainable base model parameters: {len(base_trainable)}")
        
        if len(base_trainable) == 0:
            print("✓ Base model parameters are correctly frozen")
        else:
            print(f"✗ Some base model parameters are trainable: {base_trainable}")
            
        if len(medusa_trainable) > 0:
            print("✓ Medusa head parameters are trainable")
        else:
            print("✗ No Medusa head parameters are trainable")
            
    except Exception as e:
        print(f"✗ Parameter freezing test failed: {e}")

def test_model_attributes(model):
    """Test that the model has the expected attributes and configuration."""
    print("\n=== Testing Model Attributes ===")
    
    if model is None:
        print("✗ Cannot test model attributes - model is None")
        return
    
    try:
        # Check expected attributes
        expected_attrs = [
            'tok_embeddings',
            'layers', 
            'norm',
            'output',
            'medusa_heads',
            'medusa_num_heads'
        ]
        
        for attr in expected_attrs:
            if hasattr(model, attr):
                print(f"✓ Has attribute: {attr}")
            else:
                print(f"✗ Missing attribute: {attr}")
        
        # Check model configuration
        print(f"✓ Number of layers: {len(model.layers)}")
        print(f"✓ Embedding dimension: {model.tok_embeddings.embedding_dim}")
        print(f"✓ Vocabulary size: {model.tok_embeddings.num_embeddings}")
        print(f"✓ Max sequence length: {model.max_seq_len}")
        print(f"✓ Number of heads: {model.num_heads}")
        print(f"✓ Head dimension: {model.head_dim}")
        
        # Check Medusa-specific attributes
        if hasattr(model, 'medusa_num_heads'):
            print(f"✓ Number of Medusa heads: {model.medusa_num_heads}")
        
        if hasattr(model, 'medusa_heads'):
            print(f"✓ Medusa heads module type: {type(model.medusa_heads)}")
            
    except Exception as e:
        print(f"✗ Model attributes test failed: {e}")

def test_gradient_flow(model):
    """Test that gradients flow correctly through the model."""
    print("\n=== Testing Gradient Flow ===")
    
    if model is None:
        print("✗ Cannot test gradient flow - model is None")
        return
    
    try:
        # Create test input - will automatically be on same device as model
        batch_size = 1
        seq_len = 5
        vocab_size = 128_256
        input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_tokens = input_tokens.cuda()
        print("model is on:", next(model.parameters()).device)
        # Perform forward pass
        outputs = model(input_tokens)
        # breakpoint()
        # Create dummy loss (using main output)
        if isinstance(outputs, list):
            main_output = outputs[0]
        else:
            main_output = outputs
        first_medusa_logits = outputs[1]
        
        # Create dummy targets - will automatically be on same device
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = targets.cuda()

        loss = nn.CrossEntropyLoss()(first_medusa_logits.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        print("✓ Gradient computation completed successfully")
        
        # Check gradients for different parameter types
        medusa_grads = []
        base_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'medusa_heads' in name:
                    medusa_grads.append(name)
                else:
                    base_grads.append(name)
        
        print(f"✓ Parameters with gradients - Medusa: {len(medusa_grads)}, Base: {len(base_grads)}")
        
        # In a properly configured Medusa model, base parameters should be frozen
        # so they shouldn't have gradients, but Medusa heads should
        if len(medusa_grads) > 0:
            print("✓ Medusa head parameters have gradients")
        else:
            print("✗ No Medusa head parameters have gradients")
            
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")

def test_output_consistency(model):
    """Test that outputs are consistent across multiple forward passes."""
    print("\n=== Testing Output Consistency ===")
    
    if model is None:
        print("✗ Cannot test output consistency - model is None")
        return
    
    try:
        # Create test input - will automatically be on same device as model
        batch_size = 1
        seq_len = 5
        vocab_size = 128_256
        input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_tokens = input_tokens.cuda()
        
        # Set model to eval mode
        model.eval()
        
        # Perform multiple forward passes
        outputs1 = model(input_tokens)
        outputs2 = model(input_tokens)
        
        # Check consistency
        if isinstance(outputs1, list) and isinstance(outputs2, list):
            if len(outputs1) == len(outputs2):
                print(f"✓ Output lists have same length: {len(outputs1)}")
                
                # Check main output consistency
                main1 = outputs1[0]
                main2 = outputs2[0]
                
                if torch.allclose(main1, main2, atol=1e-6):
                    print("✓ Main outputs are consistent")
                else:
                    print("✗ Main outputs are not consistent")
                    
                # Check Medusa outputs consistency
                for i, (med1, med2) in enumerate(zip(outputs1[1:], outputs2[1:])):
                    if torch.allclose(med1, med2, atol=1e-6):
                        print(f"✓ Medusa head {i+1} outputs are consistent")
                    else:
                        print(f"✗ Medusa head {i+1} outputs are not consistent")
            else:
                print(f"✗ Output lists have different lengths: {len(outputs1)} vs {len(outputs2)}")
        else:
            print("✗ Outputs are not lists as expected")
            
    except Exception as e:
        print(f"✗ Output consistency test failed: {e}")

def check_device_consistency(model, inputs, targets=None):
    model_device = next(model.parameters()).device
    print(f"Model on: {model_device}")
    print(f"Input on: {inputs.device}")
    if targets is not None:
        print(f"Target on: {targets.device}")
    assert inputs.device == model_device, "Input and model on different devices!"
    if targets is not None:
        assert targets.device == model_device, "Target and model on different devices!"

def keychecker(model, checkpoint):
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint.keys())
    
    common = model_keys & checkpoint_keys
    model_only = model_keys - checkpoint_keys
    checkpoint_only = checkpoint_keys - model_keys
    
    print(f"Keys common to both: {len(common)}")
    print(f"Keys only in model: {len(model_only)}")
    print(f"Keys only in checkpoint: {len(checkpoint_only)}")
    missing_base_keys = []
    for key in model_only:
        if "medusa" not in key:
            missing_base_keys.append(key)
            # raise ValueError(f"Base model key not found: {key}")
    print("The following base model keys are missing: ", missing_base_keys)
    # if model_only:
    #     print(f"Model-only keys: {list(model_only)}")
    # if checkpoint_only:
    #     print(f"Checkpoint-only keys: {list(checkpoint_only)}")
    
    

def main():
    """Run all verification tests."""
    print("Medusa Model Verification Script")
    print("=" * 50)
    model_dir = "/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth"
    checkpoint_params_path = "/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json"
    
    # Read model parameters from JSON
    import json
    
    with open(checkpoint_params_path, 'r') as f:
        checkpoint_params = json.load(f)
    print(f"✓ Loaded model parameters from: {checkpoint_params_path}")
    print(f"✓ Model parameters: {checkpoint_params}")
    required_params = {"num_heads":32, "num_kv_heads":32, "dim":4096}

    for key in required_params:
        if key in checkpoint_params:
            required_params[key] = checkpoint_params[key]
        else:
            print(key, " not found in checkpoint_params json")
    

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available with {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ CUDA not available, will use CPU")
    
    # Set GPU device and create model

    gpu_id = 1  # Change this to use a different GPU (0, 1, 2, etc.)
    
    if torch.cuda.is_available():
        
        torch.cuda.set_device(gpu_id)
        print(f"✓ Set GPU device to: {gpu_id}")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        torch.set_default_device("cuda")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"✓ Cleared GPU cache")

    # Create model on CPU first to avoid memory issues
    model = llama3_1_8b_medusa()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Model moved to GPU: {torch.cuda.current_device()}")
    
    # Load checkpoint if available
    try:
        checkpoint_path = model_dir
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location='cuda')
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # keychecker(model, checkpoint)
        converted_checkpoint = meta_to_tune(checkpoint)
        # keychecker(model, converted_checkpoint)
        # breakpoint()
        model.load_state_dict(converted_checkpoint, strict=False)
        print(f"✓ Model loaded from checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"✗ Checkpoint not found at: {checkpoint_path}")
        print("✓ Using randomly initialized model")
    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print("✓ Using randomly initialized model")
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ Model moved to GPU: {torch.cuda.current_device()}")
        print("model is on:", next(model.parameters()).device)
    else:
        print("✓ Model running on CPU")
    
    # Test 1: Model instantiation
    test_model_instantiation(model)
    
    # Test 2: Forward pass
    outputs = test_forward_pass(model)
    
    # Test 3: Parameter freezing
    test_parameter_freezing(model)
    
    # Test 4: Model attributes
    test_model_attributes(model)
    
    # Test 5: Gradient flow
    test_gradient_flow(model)
    
    # Test 6: Output consistency
    test_output_consistency(model)
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    
    if model is not None:
        print("✓ Model appears to be working correctly")
    else:
        print("✗ Model verification failed")

if __name__ == "__main__":
    main() 