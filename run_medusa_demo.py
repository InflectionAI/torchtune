#!/usr/bin/env python3
"""
Demo script for Medusa finetuning workflow.
This script demonstrates the complete workflow from testing to training to inference.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✓ Command completed successfully")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        if e.stdout:
            print("Stdout:")
            print(e.stdout)
        if e.stderr:
            print("Stderr:")
            print(e.stderr)
        return False


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("=== Checking Prerequisites ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ required")
        return False
    print(f"✓ Python version: {sys.version}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA not available, will use CPU")
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    
    # Check required packages
    required_packages = ['tqdm', 'tensorboard']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} not installed")
            return False
    
    # Check torchtune
    try:
        import torchtune
        print("✓ TorchTune available")
    except ImportError:
        print("✗ TorchTune not available")
        return False
    
    # Check checkpoint files
    checkpoint_path = "/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated.00.pth"
    params_path = "/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/params.json"
    
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint found: {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
    
    if os.path.exists(params_path):
        print(f"✓ Params found: {params_path}")
    else:
        print(f"⚠ Params not found: {params_path}")
    
    return True


def run_tests():
    """Run the test script."""
    print("\n=== Running Tests ===")
    return run_command("python test_medusa_finetune.py", "Testing Medusa finetuning setup")


def run_training():
    """Run a short training session."""
    print("\n=== Running Training Demo ===")
    
    # Create training command with minimal settings for demo
    training_command = (
        "python medusa_finetune.py "
        "--batch_size 2 "
        "--learning_rate 1e-4 "
        "--num_epochs 1 "
        "--max_length 256 "
        "--freeze_base_model "
        "--save_dir ./demo_checkpoints"
    )
    
    return run_command(training_command, "Running short training demo")


def run_inference():
    """Run inference with the trained model."""
    print("\n=== Running Inference Demo ===")
    
    # Check if we have a trained model
    model_path = "./demo_checkpoints/final_model.pt"
    if not os.path.exists(model_path):
        print(f"⚠ No trained model found at {model_path}")
        print("Skipping inference demo")
        return True
    
    inference_command = (
        f"python medusa_inference.py "
        f"--model_path {model_path} "
        "--prompt 'The future of artificial intelligence is' "
        "--max_new_tokens 30 "
        "--temperature 0.7 "
        "--num_medusa_heads 4"
    )
    
    return run_command(inference_command, "Running inference demo")


def show_usage_examples():
    """Show usage examples."""
    print("\n=== Usage Examples ===")
    
    examples = [
        {
            "title": "Test the setup",
            "command": "python test_medusa_finetune.py",
            "description": "Verify that all components work correctly"
        },
        {
            "title": "Run training with frozen base model",
            "command": "python medusa_finetune.py --freeze_base_model --batch_size 2 --num_epochs 3",
            "description": "Train only Medusa heads (faster and uses less memory)"
        },
        {
            "title": "Run training with custom parameters",
            "command": "python medusa_finetune.py --batch_size 1 --learning_rate 5e-5 --max_length 512",
            "description": "Customize training parameters"
        },
        {
            "title": "Resume training from checkpoint",
            "command": "python medusa_finetune.py --resume_from ./checkpoints/checkpoint_step_100.pt",
            "description": "Continue training from a saved checkpoint"
        },
        {
            "title": "Run inference",
            "command": "python medusa_inference.py --model_path ./checkpoints/final_model.pt --prompt 'Hello world'",
            "description": "Generate text with trained model"
        },
        {
            "title": "Compare Medusa vs base model",
            "command": "python medusa_inference.py --model_path ./checkpoints/final_model.pt --compare",
            "description": "Compare generation speed between Medusa and base model"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['description']}")
        print(f"   $ {example['command']}")


def main():
    """Main demo function."""
    print("Medusa Finetuning Demo")
    print("=" * 60)
    print("This script demonstrates the complete Medusa finetuning workflow.")
    print("It will test the setup, run a short training session, and show inference.")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n✗ Prerequisites not met. Please install required packages.")
        return
    
    # Ask user what to run
    print("\nWhat would you like to do?")
    print("1. Run tests only")
    print("2. Run tests + short training demo")
    print("3. Run tests + training + inference demo")
    print("4. Show usage examples only")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
        return
    
    if choice == "1":
        # Run tests only
        if run_tests():
            print("\n✓ Tests completed successfully!")
        else:
            print("\n✗ Tests failed. Please check the errors above.")
    
    elif choice == "2":
        # Run tests + training
        if run_tests():
            if run_training():
                print("\n✓ Training demo completed successfully!")
            else:
                print("\n✗ Training demo failed.")
        else:
            print("\n✗ Tests failed. Skipping training.")
    
    elif choice == "3":
        # Run full demo
        if run_tests():
            if run_training():
                if run_inference():
                    print("\n✓ Full demo completed successfully!")
                else:
                    print("\n✗ Inference demo failed.")
            else:
                print("\n✗ Training demo failed. Skipping inference.")
        else:
            print("\n✗ Tests failed. Skipping training and inference.")
    
    elif choice == "4":
        # Show usage examples
        show_usage_examples()
    
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("Check the generated files:")
    print("- ./demo_checkpoints/ - Training checkpoints")
    print("- ./logs/medusa_training/ - TensorBoard logs")
    print("\nFor more information, see README_medusa_finetune.md")


if __name__ == "__main__":
    main() 