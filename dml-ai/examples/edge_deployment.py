import torch
from dml.selfmodify import SafeSelfModifyingDML

trained_dml = SafeSelfModifyingDML()
quantized_dml = torch.quantization.quantize_dynamic(
    trained_dml,
    {torch.nn.Linear, torch.nn.LayerNorm},
    dtype=torch.qint8
)

torch.jit.save(torch.jit.script(quantized_dml), "phase3_quantized.pt")
print("Model saved for edge deployment!")