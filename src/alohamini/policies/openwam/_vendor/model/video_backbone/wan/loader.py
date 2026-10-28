"""Wan tensor metadata; local construction is in the native adapter."""
from typing import Optional
import torch.nn as nn

def infer_text_dim(dit) -> Optional[int]:
    """Wan DiT text_embedding input width (the raw context dim it expects)."""
    text_embedding = getattr(dit, "text_embedding", None)
    if isinstance(text_embedding, nn.Linear):
        return int(text_embedding.in_features)
    if isinstance(text_embedding, nn.Module):
        for module in text_embedding.modules():
            if isinstance(module, nn.Linear):
                return int(module.in_features)
    return None

