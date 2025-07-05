import torch
import torch.nn as nn
import ast
import astunparse

class SafeSelfModifyingDML(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.code_gen = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 1024)
        )
        self.memory = nn.Parameter(torch.randn(50, d_model))
        self.safety_whitelist = {'Module', 'Expr', 'Call', 'Name', 'Load', 'Attribute', 'Constant', 'nn', 'torch'}

    def generate_safe_code(self, x):
        logits = self.code_gen(x.mean(dim=1))
        patterns = [
            "nn.Sequential(nn.Linear(128,256), nn.GELU())",
            "nn.LayerNorm(128)",
            "nn.Dropout(0.1)"
        ]
        pattern_id = torch.argmax(logits) % len(patterns)
        return patterns[pattern_id]

    def apply_code(self, code_str):
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if type(node).__name__ not in self.safety_whitelist:
                    raise ValueError(f"Unsafe node: {type(node).__name__}")
            new_layer = eval(code_str, {'nn': nn, 'torch': torch})
            self.add_module(f"dynamic_{len(self._modules)}", new_layer)
            print(f"✅ Added: {code_str}")
        except Exception as e:
            print(f"❌ Failed to apply: {e}")