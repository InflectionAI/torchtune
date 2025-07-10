import torch
import torch.nn as nn

# Create a linear layer
# Parameters: input_features, output_features, bias=True
linear_layer = nn.Linear(in_features=512, out_features=256, bias=True)

# Create some input data
batch_size = 4
seq_length = 10
input_features = 512
x = torch.randn(batch_size, seq_length, input_features)

# Forward pass through the linear layer
output = linear_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Linear layer parameters:")
print(f"  Weight shape: {linear_layer.weight.shape}")
print(f"  Bias shape: {linear_layer.bias.shape}")

# You can also create a linear layer without bias
linear_no_bias = nn.Linear(in_features=256, out_features=128, bias=False)
print(f"\nLinear layer without bias:")
print(f"  Weight shape: {linear_no_bias.weight.shape}")
print(f"  Bias: {linear_no_bias.bias}")  # None 