import torch
import torch.nn as nn
import torch.nn.functional as F

class DMLLayer(nn.Module):
    def __init__(self, d_model=768, sparsity_ratio=0.3):
        super().__init__()
        self.hypernet = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.GELU(),
            nn.Linear(16, d_model * d_model)
        )
        self.sparsity_ratio = sparsity_ratio
        self.scale = 0.1

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        delta_W = self.hypernet(x.mean(dim=1)).view(batch_size, d_model, d_model)
        mask = (torch.rand_like(delta_W) < self.sparsity_ratio).float()
        delta_W = delta_W * mask * self.scale
        identity = torch.eye(d_model, device=x.device)
        x = torch.bmm(x, identity.unsqueeze(0) + delta_W.tanh())
        return x