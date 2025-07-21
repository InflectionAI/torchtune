#!/usr/bin/env python3
"""
Medusa Finetuning Script for Llama3.1-8B with Medusa heads.

This script demonstrates how to:
1. Load a pre-trained Llama3.1-8B model into MedusaTransformerDecoder
2. Set up MedusaCrossEntropyLoss for multi-token prediction training
3. Create a simple training loop with optimizer and scheduler
4. Handle data preparation and batching
5. Monitor training progress and save checkpoints

Usage:
    python medusa_finetune.py --config config.yaml
    or
    python medusa_finetune.py --checkpoint_path /path/to/llama/checkpoint --data_path /path/to/data
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchtune.models.llama3_1._model_builders import llama3_1_8b_medusa
from torchtune.modules.loss.cross_entropy_loss import MedusaCrossEntropyLoss
from torchtune.models.convert_weights import meta_to_tune
# from torchtune.modules.transforms.tokenizers import Llama3Tokenizer


class SimpleTextDataset(Dataset):
    """
    Simple dataset for text completion training.
    Creates synthetic data for demonstration purposes.
    """
    
    def __init__(self, num_samples: int = 1000, max_length: int = 512, vocab_size: int = 128256):
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size
        
        # Generate synthetic data
        self.data = []
        for _ in range(num_samples):
            # Create random sequences
            seq_len = torch.randint(50, max_length, (1,)).item()
            tokens = torch.randint(0, vocab_size, (seq_len,)).tolist()
            self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        
        # Create input and target sequences
        # For Medusa training, we need to create targets for multiple future tokens
        input_tokens = tokens[:-1]  # All but last token
        target_tokens = tokens[1:]  # All but first token (shifted by 1)
        
        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "labels": torch.tensor(target_tokens, dtype=torch.long),
            "attention_mask": torch.ones(len(input_tokens), dtype=torch.long)
        }


def collate_fn(batch, pad_token_id: int = 0, max_length: int = 512):
    """
    Custom collate function to handle variable length sequences.
    """
    # Find max length in this batch
    max_len = max(len(item["input_ids"]) for item in batch)
    max_len = min(max_len, max_length)
    
    batch_size = len(batch)
    
    # Initialize tensors
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)  # -100 is ignore_index
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = min(len(item["input_ids"]), max_len)
        
        input_ids[i, :seq_len] = item["input_ids"][:seq_len]
        labels[i, :seq_len] = item["labels"][:seq_len]
        attention_mask[i, :seq_len] = 1
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }


class MedusaTrainer:
    """
    Trainer class for Medusa model finetuning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: MedusaCrossEntropyLoss,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: torch.device = torch.device("cuda"),
        max_grad_norm: float = 1.0,
        log_every: int = 10,
        save_every: int = 100,
        save_dir: str = "./checkpoints",
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Setup tensorboard logging
        self.writer = SummaryWriter(log_dir="./logs/medusa_training")
        
        # Move model to device
        self.model.to(device)
        self.loss_fn.to(device)
        
        print(f"✓ Model moved to device: {device}")
        print(f"✓ Loss function moved to device: {device}")
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs (hidden states)
            with torch.amp.autocast("cuda", enabled=True):
                outputs = self.model(batch["input_ids"])
                loss = self.loss_fn(outputs, batch["labels"])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update training state
            self.global_step += 1
            total_loss += loss.item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Logging
            if self.global_step % self.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/avg_loss', avg_loss, self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]["lr"], self.global_step)
                
                # Log memory usage
                if torch.cuda.is_available():
                    self.writer.add_scalar('train/gpu_memory_allocated', 
                                         torch.cuda.memory_allocated() / 1024**3, 
                                         self.global_step)
            
            # Save checkpoint
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        
        self.epoch += 1
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(batch["input_ids"])
                    loss = self.loss_fn(outputs, batch["labels"])
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        
        # Save best model
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint("best_model.pt")
        
        return avg_loss
    
    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'loss_fn_state_dict': self.loss_fn.state_dict(),
        }
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        if 'loss_fn_state_dict' in checkpoint:
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  - Epoch: {self.epoch}")
        print(f"  - Global step: {self.global_step}")
        print(f"  - Best loss: {self.best_loss}")


def setup_model_and_loss(
    checkpoint_path: str,
    params_path: str,
    device: torch.device = torch.device("cuda"), 
    num_medusa_heads: int = 4,
    freeze_base_model: bool = True,
) -> Tuple[nn.Module, MedusaCrossEntropyLoss]:
    """
    Set up the Medusa model and loss function.
    """
    print("=== Setting up Model and Loss ===")
    
    # Load model parameters
    with open(params_path, 'r') as f:
        checkpoint_params = json.load(f)
    print(f"✓ Loaded model parameters from: {params_path}")
    
    # Create model
    with device:
        model = llama3_1_8b_medusa()
    print(f"✓ Created Medusa model with {num_medusa_heads} heads")
    print(f"✓ The model is located on device: {next(model.parameters()).device}")

    
    # Load checkpoint if available
    if os.path.exists(checkpoint_path):
        print(f"✓ Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        converted_checkpoint = meta_to_tune(checkpoint)
        model.load_state_dict(converted_checkpoint, strict=False)
        print("✓ Checkpoint loaded successfully")
    else:
        print("⚠ No checkpoint found, using randomly initialized model")
    
    # Freeze base model parameters if requested
    print("Base Model is Frozen: ", freeze_base_model)
    if freeze_base_model:
        for name, param in model.named_parameters():
            if 'medusa_heads' not in name:
                param.requires_grad = False
        print("✓ Base model parameters frozen")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    # Create loss function
    loss_fn = MedusaCrossEntropyLoss(num_output_chunks=8, ignore_index=-100)
    print(f"✓ Created MedusaCrossEntropyLoss")
    
    # Set up loss function with model
    loss_fn.set_model_output(model)
    print("✓ Loss function configured with model")
    
    return model, loss_fn


def setup_optimizer_and_scheduler(
    model: nn.Module,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    total_steps: int = 1000,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """
    Set up optimizer and learning rate scheduler.
    """
    print("=== Setting up Optimizer and Scheduler ===")
    
    # Get only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Create optimizer
    optimizer = optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    print(f"✓ Created AdamW optimizer with lr={learning_rate}, weight_decay={weight_decay}")
    
    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=learning_rate * 0.1
    )
    print(f"✓ Created CosineAnnealingLR scheduler")
    
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(description="Medusa Model Finetuning")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth",
                       help="Path to Llama checkpoint")
    parser.add_argument("--params_path", type=str,
                       default="/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json",
                       help="Path to model parameters JSON")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to training data (optional, uses synthetic data if not provided)")
    parser.add_argument("--num_medusa_heads", type=int, default=4,
                       help="Number of Medusa heads")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--freeze_base_model", action="store_true",
                       help="Freeze base model parameters")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    # Set device
    gpu_id = 1
    torch.cuda.set_device(gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    print(f"✓ Set GPU device to: {gpu_id}")
    print(f"✓ Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    
    # Set default device only for model creation
    # torch.set_default_device(device)
    
    # Setup model and loss
    model, loss_fn = setup_model_and_loss(
        checkpoint_path=args.checkpoint_path,
        params_path=args.params_path,
        device = device,
        num_medusa_heads=args.num_medusa_heads,
        freeze_base_model=args.freeze_base_model
    )
    
    # Reset default device to CPU for data operations
    # torch.set_default_device('cpu')
    
    # Create dataset
    if args.data_path:
        # TODO: Implement loading from actual data file
        print(f"⚠ Loading from {args.data_path} not implemented yet, using synthetic data")
        dataset = SimpleTextDataset(num_samples=1000, max_length=args.max_length)
    else:
        print("✓ Using synthetic dataset for demonstration")
        dataset = SimpleTextDataset(num_samples=1000, max_length=args.max_length)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"✓ Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, max_length=args.max_length),
        num_workers=0,  # Set to 0 for simplicity
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, max_length=args.max_length),
        num_workers=0,
        drop_last=True
    )
    
    # Setup optimizer and scheduler
    total_steps = len(train_dataloader) * args.num_epochs
    optimizer, scheduler = setup_optimizer_and_scheduler(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        total_steps=total_steps
    )
    
    # Create trainer
    trainer = MedusaTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=args.max_grad_norm,
        save_dir=args.save_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Training loop
    print("\n=== Starting Training ===")
    start_time = time.time()
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        
        # Train
        train_loss = trainer.train_epoch(train_dataloader)
        print(f"✓ Training loss: {train_loss:.4f}")
        
        # Validate
        val_loss = trainer.validate(val_dataloader)
        print(f"✓ Validation loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        trainer.save_checkpoint(f"epoch_{epoch + 1}.pt")
    
    # Final save
    trainer.save_checkpoint("final_model.pt")
    
    total_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"✓ Total training time: {total_time/3600:.2f} hours")
    print(f"✓ Best validation loss: {trainer.best_loss:.4f}")
    print(f"✓ Final model saved to: {args.save_dir}/final_model.pt")
    print(f"✓ Best model saved to: {args.save_dir}/best_model.pt")
    
    # Close tensorboard writer
    trainer.writer.close()


if __name__ == "__main__":
    main() 