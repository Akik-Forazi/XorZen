import torch
import os

# Define the model
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Create data
dim = 768
B, T = 2, 128
torch.manual_seed(42)
x = torch.randn(B, T, dim, dtype=torch.float32)
model = RMSNorm(dim)
model.weight.data = torch.randn(dim, dtype=torch.float32)
y = model(x)

# Save as raw binary
os.makedirs("tests/parity", exist_ok=True)
x.detach().numpy().tofile("tests/parity/rmsnorm_x.bin")
model.weight.data.detach().numpy().tofile("tests/parity/rmsnorm_weight.bin")
y.detach().numpy().tofile("tests/parity/rmsnorm_y.bin")
print("Golden vectors saved as raw binary to tests/parity/rmsnorm_*.bin")
