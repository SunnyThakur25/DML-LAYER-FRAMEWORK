import torch
import torch.nn as nn

class MemoryDML(nn.Module):
    def __init__(self, d_model=128, mem_size=50):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, d_model))
        self.read = nn.Linear(d_model, mem_size)
        self.write = nn.Linear(d_model, mem_size)
        self.write_proj = nn.Linear(mem_size, d_model)

    def forward(self, x):
        read_weights = torch.softmax(self.read(x.mean(dim=1)), -1)
        retrieved = read_weights @ self.memory
        write_weights = torch.sigmoid(self.write(x.mean(dim=1)))
        projected_weights = self.write_proj(write_weights).unsqueeze(1)
        self.memory.data += 0.1 * (projected_weights * x.mean(dim=1).unsqueeze(1))
        return x + retrieved