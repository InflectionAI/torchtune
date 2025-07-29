#!/usr/bin/env python3
"""
Convert Meta format checkpoint to torchtune format using torchtune's convert_weights.

Usage:
    python convert_meta_to_torchtune.py --checkpoint_dir /path/to/meta/checkpoint --output_path /path/to/output/torchtune_checkpoint.pt
"""

import argparse
import os
import torch
from torchtune.models.convert_weights import meta_to_tune


def convert_meta_to_torchtune(checkpoint_dir: str, output_path: str = None):
    """
    Convert Meta format checkpoint to torchtune format.
    
    Args:
        checkpoint_dir (str): Path to directory containing Meta format checkpoint files
        output_path (str): Path to save the converted torchtune checkpoint
    """
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # Look for the consolidated weights file
    checkpoint_file = None
    for file in os.listdir(checkpoint_dir):
        if file.startswith('consolidated') and file.endswith('.pth'):
            checkpoint_file = os.path.join(checkpoint_dir, file)
            break
    
    if checkpoint_file is None:
        raise FileNotFoundError(f"No consolidated checkpoint file found in {checkpoint_dir}")
    
    print(f"Loading Meta checkpoint from: {checkpoint_file}")
    
    # Load the Meta format checkpoint
    meta_checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print(f"Loaded checkpoint with keys: {list(meta_checkpoint.keys())}")
    
    # Convert from Meta format to torchtune format
    print("Converting Meta format to torchtune format...")
    torchtune_state_dict = meta_to_tune(meta_checkpoint)
    
    # Prepare the final checkpoint in torchtune format
    torchtune_checkpoint = {
        'model': torchtune_state_dict,
    }
    
    # Set default output path if not provided
    if output_path is None:
        output_path = os.path.join(checkpoint_dir, 'consolidated_torchtune.00.pth')
    
    # Save the converted checkpoint
    print(f"Saving torchtune checkpoint to: {output_path}")
    torch.save(torchtune_checkpoint, output_path)
    
    print("Conversion completed successfully!")
    print(f"Torchtune checkpoint saved at: {output_path}")
    print(f"Converted checkpoint has {len(torchtune_state_dict)} parameters")


def main():
    parser = argparse.ArgumentParser(description='Convert Meta format checkpoint to torchtune format')
    parser.add_argument('--checkpoint_dir', type=str, required=False, default='/mnt/vast/share/inf2-training/models/open_source/Meta-Llama-3.1-8B-Instruct/original',
                        help='Path to directory containing Meta format checkpoint files')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the converted torchtune checkpoint (default: checkpoint_dir/torchtune_checkpoint.pt)')
    
    args = parser.parse_args()
    
    convert_meta_to_torchtune(args.checkpoint_dir, args.output_path)


if __name__ == '__main__':
    main() 