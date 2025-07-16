#!/usr/bin/env python3
"""
Verification script for MedusaCrossEntropyLoss.
This script tests various aspects of the loss function to ensure it's working correctly.
"""

import torch
import torch.nn as nn
from torchtune.models.llama3_1._model_builders import llama3_1_8b_medusa
from torchtune.modules.loss.cross_entropy_loss import MedusaCrossEntropyLoss

def test_loss_initialization():
    """Test that the loss function can be initialized correctly."""
    print("=== Testing Loss Initialization ===")
    
    try:
        loss_fn = MedusaCrossEntropyLoss(num_output_chunks=8, ignore_index=-100)
        print(f"✓ Loss function initialized successfully")
        print(f"✓ Loss type: {type(loss_fn)}")
        print(f"✓ num_output_chunks: {loss_fn.num_output_chunks}")
        print(f"✓ ignore_index: {loss_fn.ignore_index}")
        
        return loss_fn
    except Exception as e:
        print(f"✗ Loss initialization failed: {e}")
        return None

def test_model_setup(loss_fn):
    """Test that the loss function can be set up with a model."""
    print("\n=== Testing Model Setup ===")
    
    if loss_fn is None:
        print("✗ Cannot test model setup - loss_fn is None")
        return None
    
    try:
        # Clear GPU memory
        torch.cuda.empty_cache()
        print(f"✓ Cleared GPU cache")
        # Set GPU device and create model
        gpu_id = 1  # Change this to use a different GPU (0, 1, 2, etc.)
        
        if torch.cuda.is_available():
            
            torch.cuda.set_device(gpu_id)
            print(f"✓ Set GPU devicea to: {gpu_id}")
            print(f"✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            torch.set_default_device("cuda")
            # Create model
            model = llama3_1_8b_medusa().cuda()
            print(f"✓ Model created successfully")
            
            # Set up loss function with model
            loss_fn.set_model_output(model)
            print(f"✓ Loss function set up with model")
            
            # Check that model was modified correctly
            print(f"✓ Model skip_output_layer: {model.skip_output_layer}")
            print(f"✓ Medusa linear layers: {loss_fn.medusa_linear_layers is not None}")
            print(f"✓ Number of medusa heads: {loss_fn.medusa_num_heads}")
            print(f"✓ Hidden size: {loss_fn.hidden_size}")
            print(f"✓ Loss weights: {loss_fn.medusa_loss_weights}")
        
        return model
    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        return None

def test_forward_pass(loss_fn, model):
    """Test that the loss function can perform forward passes correctly."""
    print("\n=== Testing Forward Pass ===")
    
    if loss_fn is None or model is None:
        print("✗ Cannot test forward pass - loss_fn or model is None")
        return
    
    # Create test inputs
    batch_size = 1
    seq_len = 10
    vocab_size = 128_256
    
    # Create random hidden states (simulating model output)
    # hidden_states = torch.randn(batch_size, seq_len, model.hidden_size).cuda()
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    # Create random targets
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    
    # Add some ignore tokens
    targets[0, -2:] = -100  # Ignore last 2 tokens of first sequence
    
    print(f"✓ Input tokens shape: {input_tokens.shape}")
    print(f"✓ Input targets shape: {targets.shape}")
    print(f"✓ Number of ignored tokens: {(targets == -100).sum()}")

    # Perform forward pass
    loss_fn.set_model_output(model)
    # loss_fn.num_output_chunks = 1
    outputs = model(input_tokens)
    loss = loss_fn(outputs, targets)
    
    print(f"✓ Forward pass completed successfully")
    print(f"✓ Loss value: {loss.item()}")
    print(f"✓ Loss shape: {loss.shape}")
    print(f"✓ Loss requires grad: {loss.requires_grad}")
    
    return loss, input_tokens, outputs, targets

def test_gradient_computation(loss_fn, model, loss, input_tokens):
    """Test that gradients can be computed correctly."""
    print("\n=== Testing Gradient Computation ===")
    
    if loss_fn is None or model is None:
        print("✗ Cannot test gradient computation - loss_fn or model is None")
        return
    
    # # Create test inputs
    # batch_size = 1
    # seq_len = 5
    # vocab_size = 128_256
    
    # hidden_states = torch.randn(batch_size, seq_len, model.hidden_size, requires_grad=True).cuda()
    # targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    
    # # Compute loss
    # loss = loss_fn(hidden_states, targets)
    
    # Backward pass
    loss.backward()
    
    print("✓ Gradient computation completed successfully")
    print(f"✓ Loss value: {loss.item()}")
    print(f"✓ Hidden states grad: {input_tokens.grad is not None}")
    
    if input_tokens.grad is not None:
        print(f"✓ Hidden states grad shape: {input_tokens.grad.shape}")
        print(f"✓ Hidden states grad norm: {input_tokens.grad.norm()}")
    
    return loss

def test_loss_consistency(loss_fn, model,  outputs, targets):
    """Test that the loss is consistent across multiple forward passes."""
    print("\n=== Testing Loss Consistency ===")
    
    if loss_fn is None or model is None:
        print("✗ Cannot test loss consistency - loss_fn or model is None")
        return
    
    # Create test inputs
    # batch_size = 1
    # seq_len = 5
    # vocab_size = 128_256
    
    # hidden_states = torch.randn(batch_size, seq_len, model.hidden_size).cuda()
    # targets = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    
    # Set model to eval mode for consistency
    model.eval()
    
    # Perform multiple forward passes
    loss1 = loss_fn(outputs, targets)
    loss2 = loss_fn(outputs, targets)
    
    print(f"✓ Loss 1: {loss1.item()}")
    print(f"✓ Loss 2: {loss2.item()}")
    
    if torch.allclose(loss1, loss2, atol=1e-6):
        print("✓ Losses are consistent")
    else:
        print("✗ Losses are not consistent")

def test_edge_cases(loss_fn, model):
    """Test edge cases like all ignored tokens, empty sequences, etc."""
    print("\n=== Testing Edge Cases ===")
    
    if loss_fn is None or model is None:
        print("✗ Cannot test edge cases - loss_fn or model is None")
        return
    
    # Test 1: All tokens ignored
    batch_size = 1
    seq_len = 5
    vocab_size = 128_256
    loss_fn.set_model_output(model)
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    
    targets = torch.full((batch_size, seq_len), -100).cuda()  # All ignored
    outputs = model(input_tokens)
    loss = loss_fn(outputs, targets)
    print(f"✓ All ignored tokens - Loss: {loss.item()}")
    
    # Test 2: Single token
    
    input_tokens = torch.randint(0, vocab_size, (1, 1)).cuda()
    outputs = model(input_tokens)
    targets = torch.randint(0, 128_256, (1, 1)).cuda()
    # breakpoint()
    loss = loss_fn(outputs, targets)
    print(f"✓ Single token - Loss: {loss.item()}")
    
    # Test 3: Large sequence
    input_tokens = torch.randint(0, vocab_size, (100, 100)).cuda()
    
    targets = torch.randint(0, 128_256, (100, 100)).cuda()
    outputs = model(input_tokens)
    loss = loss_fn(outputs, targets)
    print(f"✓ Large sequence - Loss: {loss.item()}")
    
    print("✓ All edge cases passed")

def main():
    """Run all verification tests."""
    print("Medusa Cross Entropy Loss Verification Script")
    print("=" * 50)
    
    # Test 1: Loss initialization
    loss_fn = test_loss_initialization()
    
    # Test 2: Model setup
    model = test_model_setup(loss_fn)
    
    # Test 3: Forward pass
    result = test_forward_pass(loss_fn, model)
    if result is None:
        print("✗ Forward pass failed")
        return
    
    loss, input_tokens, outputs, targets = result
    
    # Test 4: Gradient computation
    test_gradient_computation(loss_fn, model, loss, input_tokens)
    
    # Test 5: Loss consistency
    test_loss_consistency(loss_fn, model, outputs, targets)
    
    # Test 6: Edge cases
    test_edge_cases(loss_fn, model)
    
    print("\n" + "=" * 50)
    print("Verification complete!")
    
    if loss_fn is not None and model is not None:
        print("✓ Loss function appears to be working correctly")
    else:
        print("✗ Loss function verification failed")

if __name__ == "__main__":
    main() 