#!/usr/bin/env python3
"""
Test script for Medusa finetuning setup.
This script verifies that all components work correctly before running full training.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from medusa_finetune import (
    setup_model_and_loss,
    setup_optimizer_and_scheduler,
    MedusaTrainer,
    SimpleTextDataset,
    collate_fn
)


def test_model_setup():
    """Test model and loss function setup."""
    print("=== Testing Model Setup ===")
    
    try:
        # Setup model and loss
        model, loss_fn = setup_model_and_loss(
            checkpoint_path="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth",
            params_path="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json",
            num_medusa_heads=4,
            device=torch.device("cuda"),
            freeze_base_model=True
        )
        
        print("✓ Model and loss setup successful")
        return model, loss_fn
        
    except Exception as e:
        print(f"✗ Model setup failed: {e}")
        return None, None


def test_data_loading():
    """Test data loading and batching."""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Create small dataset
        dataset = SimpleTextDataset(num_samples=10, max_length=128)
        print(f"✓ Created dataset with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, max_length=128),
            num_workers=0
        )
        
        print(f"✓ Created dataloader with {len(dataloader)} batches")
        
        # Test one batch
        batch = next(iter(dataloader))
        print(f"✓ Batch keys: {list(batch.keys())}")
        print(f"✓ Input shape: {batch['input_ids'].shape}")
        print(f"✓ Labels shape: {batch['labels'].shape}")
        print(f"✓ Attention mask shape: {batch['attention_mask'].shape}")
        
        return dataloader
        
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None


def test_forward_backward(model, loss_fn, dataloader):
    """Test forward and backward pass."""
    print("\n=== Testing Forward/Backward Pass ===")
    
    try:
        # Move to GPU
        device = torch.device("cuda")
        model.to(device)
        loss_fn.to(device)
        
        # Get one batch
        batch = next(iter(dataloader))
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        model.train()
        outputs = model(batch["input_ids"])
        loss = loss_fn(outputs, batch["labels"])
        
        print(f"✓ Forward pass successful")
        print(f"✓ Loss value: {loss.item():.4f}")
        print(f"✓ Loss requires grad: {loss.requires_grad}")
        
        # Backward pass
        loss.backward()
        print("✓ Backward pass successful")
        
        # Check gradients
        medusa_grads = 0
        base_grads = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'medusa_heads' in name:
                    medusa_grads += 1
                else:
                    base_grads += 1
        
        print(f"✓ Parameters with gradients - Medusa: {medusa_grads}, Base: {base_grads}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward/backward pass failed: {e}")
        return False


def test_optimizer_setup(model):
    """Test optimizer and scheduler setup."""
    print("\n=== Testing Optimizer Setup ===")
    
    try:
        optimizer, scheduler = setup_optimizer_and_scheduler(
            model=model,
            learning_rate=1e-4,
            weight_decay=0.01,
            total_steps=100
        )
        
        print("✓ Optimizer and scheduler setup successful")
        print(f"✓ Optimizer type: {type(optimizer)}")
        print(f"✓ Scheduler type: {type(scheduler)}")
        
        return optimizer, scheduler
        
    except Exception as e:
        print(f"✗ Optimizer setup failed: {e}")
        return None, None


def test_trainer_setup(model, loss_fn, optimizer, scheduler):
    """Test trainer setup."""
    print("\n=== Testing Trainer Setup ===")
    
    try:
        trainer = MedusaTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=torch.device("cuda"),
            max_grad_norm=1.0,
            save_dir="./test_checkpoints"
        )
        
        print("✓ Trainer setup successful")
        return trainer
        
    except Exception as e:
        print(f"✗ Trainer setup failed: {e}")
        return None


def test_single_training_step(trainer, dataloader):
    """Test a single training step."""
    print("\n=== Testing Single Training Step ===")
    
    try:
        # Get one batch
        batch = next(iter(dataloader))
        
        # Move to device
        batch = {k: v.to(trainer.device) for k, v in batch.items()}
        
        # Training step
        trainer.model.train()
        trainer.optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=True):
            outputs = trainer.model(batch["input_ids"])
            loss = trainer.loss_fn(outputs, batch["labels"])
        
        loss.backward()
        
        if trainer.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
        
        trainer.optimizer.step()
        if trainer.scheduler is not None:
            trainer.scheduler.step()
        
        print("✓ Single training step successful")
        print(f"✓ Loss: {loss.item():.4f}")
        print(f"✓ Learning rate: {trainer.optimizer.param_groups[0]['lr']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Single training step failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Medusa Finetuning Test Script")
    print("=" * 50)
    
    # Test 1: Model setup
    model, loss_fn = test_model_setup()
    if model is None or loss_fn is None:
        print("✗ Model setup failed, aborting tests")
        return
    
    # Test 2: Data loading
    dataloader = test_data_loading()
    if dataloader is None:
        print("✗ Data loading failed, aborting tests")
        return
    
    # Test 3: Forward/backward pass
    if not test_forward_backward(model, loss_fn, dataloader):
        print("✗ Forward/backward pass failed, aborting tests")
        return
    
    # Test 4: Optimizer setup
    optimizer, scheduler = test_optimizer_setup(model)
    if optimizer is None or scheduler is None:
        print("✗ Optimizer setup failed, aborting tests")
        return
    
    # Test 5: Trainer setup
    trainer = test_trainer_setup(model, loss_fn, optimizer, scheduler)
    if trainer is None:
        print("✗ Trainer setup failed, aborting tests")
        return
    
    # Test 6: Single training step
    if not test_single_training_step(trainer, dataloader):
        print("✗ Single training step failed")
        return
    
    print("\n" + "=" * 50)
    print("✓ All tests passed! The finetuning setup is working correctly.")
    print("✓ You can now run the full training with:")
    print("  python medusa_finetune.py --freeze_base_model")


if __name__ == "__main__":
    main() 