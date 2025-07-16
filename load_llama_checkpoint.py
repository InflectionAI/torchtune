import torch
import os
from torchtune.models.llama3_1 import llama3_1_8b

def load_checkpoint(checkpoint_dir):
    """
    Load a checkpoint from a directory containing checkpoint files.
    
    Args:
        checkpoint_dir (str): Path to directory containing checkpoint files
        
    Returns:
        model: Loaded model with checkpoint weights
    """
    # Check if checkpoint directory exists
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    # List files in checkpoint directory
    files = os.listdir(checkpoint_dir)
    print(f"Files in checkpoint directory: {files}")
    
    # Look for the consolidated weights file
    checkpoint_file = None
    for file in files:
        if file.startswith('consolidated') and file.endswith('.pth'):
            checkpoint_file = os.path.join(checkpoint_dir, file)
            break
    
    if checkpoint_file is None:
        raise FileNotFoundError(f"No consolidated checkpoint file found in {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_file}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    # Create the model (regular Llama 3.1, not Medusa)
    model = llama3_1_8b()
    
    # Load the state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the weights
    model.load_state_dict(state_dict, strict=False)
    
    print("Checkpoint loaded successfully!")
    return model

def main():
    # Replace this with your actual checkpoint directory path
    checkpoint_dir = "/path/to/your/checkpoint/directory"
    
    try:
        model = load_checkpoint(checkpoint_dir)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"Model loaded on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 1
        seq_len = 10
        vocab_size = 128256
        
        # Create dummy input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        with torch.no_grad():
            output = model(input_ids)
            print(f"Output shape: {output.shape}")
            print("Model is working correctly!")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    main() 