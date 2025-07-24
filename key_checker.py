import torch
import os

# Set CUDA devices to use GPUs 1 and 2
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# Load both models
print("Loading original model...")
original = torch.load('/home/ubuntu/.llama/checkpoints/Llama3.1-8B-Instruct/consolidated_torchtune.00.pth', map_location='cuda')


print("Loading new model...")
new = torch.load('/home/ubuntu/vanshaj/inf2-training/3rdparty/torchtune/medusa_val_checkpoints/epoch_1000002/model-00001-of-00001.bin', map_location='cuda')

# Compare keys
print(f"\nOriginal model keys: {len(original)}")
print(f"New model keys: {len(new)}")

# Find missing and extra keys
missing_in_new = set(original.keys()) - set(new.keys())
extra_in_new = set(new.keys()) - set(original.keys())

print(f"\nKeys in original but missing in new: {len(missing_in_new)}")
print(f"Extra keys in new (should be medusa heads): {len(extra_in_new)}")

# Show some examples
if missing_in_new:
    print("\Missing keys:")
    for k in list(missing_in_new):
        print(f"  {k}")
else:
    print("\n✅ All original keys are present in new model!")

if extra_in_new:
    print("Missing keys:")
    for k in list(extra_in_new):
        print(f"  {k}")
    
    # Check if extra keys are medusa heads
    medusa_keys = [k for k in extra_in_new if 'medusa' in k.lower()]
    print(f"\nExtra keys that are medusa heads: {len(medusa_keys)}/{len(extra_in_new)}")

# Verify a few key parameters have the same shape
print("\nVerifying key parameter shapes:")
key_params = ['tok_embeddings.weight', 'layers.0.attn.q_proj.weight', 'layers.0.mlp.w1.weight']
for param in key_params:
    if param in original and param in new:
        orig_shape = original[param].shape
        new_shape = new[param].shape
        print(f"  {param}: {orig_shape} -> {new_shape} {'✅' if orig_shape == new_shape else '❌'}")
    else:
        print(f"  {param}: {'❌ Missing in original' if param not in original else '❌ Missing in new'}")

print("\nVerification complete!")