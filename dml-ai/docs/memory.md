
`


# Memory-Augmented DML

Adds learnable memory matrix with soft read/write heads for long-term knowledge retention.

## Features

- Soft attention-based reading
- Delta update writing
- Task suggestion system

## Implementation

class MemoryDML(nn.Module):
    def __init__(self, d_model=128, mem_size=50):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(mem_size, d_model))
        self.read = nn.Linear(d_model, mem_size)
        self.write = nn.Linear(d_model, mem_size)
        self.write_proj = nn.Linear(mem_size, d_model)