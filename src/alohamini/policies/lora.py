"""Shared low-rank linear adapters with stable base checkpoint names."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    """Frozen base weights, FP32 residual parameters, optional input dropout."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        nn.Module.__init__(self)
        if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA requires a positive rank and alpha")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        residual = F.linear(
            F.linear(self.dropout(x).to(self.lora_A.dtype), self.lora_A), self.lora_B
        )
        return base + residual.to(base.dtype) * self.scaling
