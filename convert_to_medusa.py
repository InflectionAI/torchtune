import torch
import os
from torchtune.models.llama3_1 import llama3_1_8b, llama3_1_8b_medusa

def convert_to_medusa(checkpoint_dir, output_path=None):
    """
    Load a regular Llama 3.1 checkpoint and convert it to a Medusa model.
    
    Args:
        checkpoint_dir (str): Path to directory containing checkpoint files
        output_path (str): Path to save the converted model (optional)
        
    Returns:
        model: Medusa model with loaded weights
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
    
    # Create the regular model first
    regular_model = llama3_1_8b()
    
    # Load the state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load the weights into regular model
    regular_model.load_state_dict(state_dict, strict=False)
    print("Regular model loaded successfully!")
    
    # Create Medusa model
    medusa_model = llama3_1_8b_medusa()
    print("Created Medusa model")
    
    # Copy weights from regular model to Medusa model
    # We need to map the parameter names
    medusa_state_dict = {}
    
    # Copy all the regular model parameters
    for name, param in regular_model.state_dict().items():
        medusa_state_dict[name] = param
    
    # Load the copied weights into Medusa model
    medusa_model.load_state_dict(medusa_state_dict, strict=False)
    print("Converted to Medusa model successfully!")
    
    # Save the converted model if output path is provided
    if output_path:
        torch.save(medusa_model.state_dict(), output_path)
        print(f"Saved converted model to: {output_path}")
    
    return medusa_model

def main():
    # Replace this with your actual checkpoint directory path
    checkpoint_dir = "/path/to/your/checkpoint/directory"
    output_path = "medusa_converted_model.pth"  # Optional: save the converted model
    
    try:
        model = convert_to_medusa(checkpoint_dir, output_path)
        
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
            print("Medusa model is working correctly!")
            
    except Exception as e:
        print(f"Error converting model: {e}")

if __name__ == "__main__":
    main() 