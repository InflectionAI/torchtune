# Medusa Model Finetuning

This directory contains scripts for finetuning Llama3.1-8B models with Medusa heads for multi-token prediction. The Medusa approach adds multiple prediction heads to a language model, enabling it to predict multiple future tokens simultaneously, which can significantly speed up text generation.

## Overview

The finetuning setup includes:

1. **`medusa_finetune.py`** - Main finetuning script
2. **`test_medusa_finetune.py`** - Test script to verify setup
3. **`medusa_config.yaml`** - Configuration file
4. **`README_medusa_finetune.md`** - This documentation

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Llama3.1-8B checkpoint files

## Installation

1. Install the required dependencies:
```bash
pip install torch torchvision torchaudio
pip install tensorboard tqdm
```

2. Ensure you have the Llama3.1-8B checkpoint files:
   - `consolidated.00.pth` - Model weights
   - `params.json` - Model parameters

## Quick Start

### 1. Test the Setup

Before running full training, test that everything works:

```bash
python test_medusa_finetune.py
```

This will verify:
- Model loading and setup
- Data loading and batching
- Forward/backward pass
- Optimizer and scheduler setup
- Single training step

### 2. Run Finetuning

#### Basic Training (with synthetic data):
```bash
python medusa_finetune.py --freeze_base_model
```

#### Training with custom parameters:
```bash
python medusa_finetune.py \
    --checkpoint_path /path/to/llama/checkpoint \
    --params_path /path/to/params.json \
    --batch_size 2 \
    --learning_rate 5e-5 \
    --num_epochs 5 \
    --freeze_base_model
```

#### Resume from checkpoint:
```bash
python medusa_finetune.py \
    --resume_from ./checkpoints/checkpoint_step_100.pt \
    --freeze_base_model
```

## Configuration

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_path` | `/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth` | Path to Llama checkpoint |
| `--params_path` | `/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json` | Path to model parameters JSON |
| `--num_medusa_heads` | 4 | Number of Medusa heads |
| `--batch_size` | 4 | Training batch size |
| `--learning_rate` | 1e-4 | Learning rate |
| `--weight_decay` | 0.01 | Weight decay |
| `--max_grad_norm` | 1.0 | Maximum gradient norm for clipping |
| `--num_epochs` | 3 | Number of training epochs |
| `--max_length` | 512 | Maximum sequence length |
| `--freeze_base_model` | False | Freeze base model parameters |
| `--resume_from` | None | Resume training from checkpoint |
| `--save_dir` | `./checkpoints` | Directory to save checkpoints |

### Configuration File

You can also use the YAML configuration file:

```bash
# Edit medusa_config.yaml with your settings
python medusa_finetune.py --config medusa_config.yaml
```

## Model Architecture

The finetuning setup uses:

1. **MedusaTransformerDecoder**: Extends the base Llama3.1-8B model with multiple prediction heads
2. **MedusaCrossEntropyLoss**: Computes loss for multiple future token predictions
3. **Parameter Freezing**: Option to freeze base model parameters and only train Medusa heads

### Medusa Heads

The model includes multiple prediction heads that predict:
- Head 1: Next token (t+1)
- Head 2: Token after next (t+2)
- Head 3: Third token (t+3)
- Head 4: Fourth token (t+4)

Each head is trained with exponentially decreasing weights: [1.0, 0.8, 0.64, 0.512]

## Training Process

### 1. Model Setup
- Load pre-trained Llama3.1-8B weights
- Convert to MedusaTransformerDecoder format
- Optionally freeze base model parameters
- Set up MedusaCrossEntropyLoss

### 2. Data Preparation
- Currently uses synthetic data for demonstration
- Supports custom data loading (to be implemented)
- Handles variable length sequences with padding

### 3. Training Loop
- Forward pass through model to get hidden states
- Compute loss using MedusaCrossEntropyLoss
- Backward pass with gradient clipping
- Optimizer and scheduler updates
- Regular checkpointing and logging

### 4. Monitoring
- TensorBoard logging for loss, learning rate, and memory usage
- Progress bars with real-time metrics
- Automatic best model saving

## Output Files

The training generates:

- `./checkpoints/` - Model checkpoints
  - `checkpoint_step_X.pt` - Regular checkpoints
  - `epoch_X.pt` - End-of-epoch checkpoints
  - `best_model.pt` - Best validation loss model
  - `final_model.pt` - Final trained model

- `./logs/medusa_training/` - TensorBoard logs
  - Training and validation loss curves
  - Learning rate schedules
  - GPU memory usage

## Memory Optimization

The script includes several memory optimization features:

1. **Mixed Precision Training**: Uses `torch.cuda.amp.autocast`
2. **Gradient Clipping**: Prevents gradient explosion
3. **Chunked Loss Computation**: MedusaCrossEntropyLoss uses chunking
4. **Parameter Freezing**: Only train Medusa heads to reduce memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch_size 2`
   - Reduce sequence length: `--max_length 256`
   - Use gradient accumulation (not implemented yet)

2. **Checkpoint Loading Errors**:
   - Verify checkpoint paths are correct
   - Ensure checkpoint format matches expected format

3. **Import Errors**:
   - Ensure torchtune is properly installed
   - Check Python path includes torchtune modules

### Debug Mode

Run the test script to identify issues:
```bash
python test_medusa_finetune.py
```

## Customization

### Adding Real Data

To use real data instead of synthetic data:

1. Create a custom dataset class that inherits from `torch.utils.data.Dataset`
2. Implement `__len__` and `__getitem__` methods
3. Return dictionaries with `input_ids`, `labels`, and `attention_mask`
4. Update the data loading section in `medusa_finetune.py`

### Modifying Loss Function

The MedusaCrossEntropyLoss can be customized:

- Change loss weights for different heads
- Modify the number of output chunks
- Adjust the ignore index

### Adding New Features

Common additions:
- Gradient accumulation
- Multi-GPU training
- Custom learning rate schedules
- Additional metrics and logging
- Model evaluation on test sets

## Performance Tips

1. **Use GPU**: Ensure CUDA is available and properly configured
2. **Batch Size**: Start with small batch sizes and increase if memory allows
3. **Sequence Length**: Shorter sequences use less memory
4. **Freeze Base Model**: Only training Medusa heads is much faster
5. **Mixed Precision**: Already enabled by default

## Example Training Run

```bash
# Test setup first
python test_medusa_finetune.py

# Run training with frozen base model
python medusa_finetune.py \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 3 \
    --max_length 512 \
    --freeze_base_model \
    --save_dir ./my_medusa_checkpoints

# Monitor training
tensorboard --logdir ./logs/medusa_training
```

## Next Steps

After training, you can:

1. **Load the trained model** for inference
2. **Evaluate performance** on test data
3. **Compare generation speed** with base model
4. **Fine-tune hyperparameters** for better performance

## References

- [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
- [TorchTune Documentation](https://github.com/facebookresearch/torchtune)
- [Llama3.1 Model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) 