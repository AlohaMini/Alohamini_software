"""Trainable weight storage, independent of autocast compute precision."""

from contextlib import contextmanager

import torch

TRAINABLE_DTYPES = ("preserve", "float32")


def reduction_device(device):
    """Keep FP64 accounting on CPU for Metal, without changing loss weights."""
    device = torch.device(device)
    return torch.device("cpu") if device.type == "mps" else device


def synchronize_device(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def resolve_tf32(settings):
    """Keep legacy defaults while making the matmul/convolution policy explicit."""
    value = settings.get("tf32", not settings.get("cudnn_deterministic", False))
    if type(value) is not bool:
        raise ValueError("tf32 must be a boolean")
    return value


@contextmanager
def tf32_context(enabled):
    """Apply the same FP32 math policy to forward, backward and recomputation."""
    if type(enabled) is not bool:
        raise ValueError("tf32 must be a boolean")
    if not torch.backends.cuda.is_built():
        yield
        return
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def validate_trainable_dtype(value):
    if value not in TRAINABLE_DTYPES:
        raise ValueError("trainable_dtype must be preserve or float32")
    return value


def apply_trainable_dtype(model, value="preserve"):
    """Set parameter storage before loading weights and constructing the optimizer.

    FP32 trainable parameters are the optimizer's authoritative weights; autocast
    supplies lower-precision compute without a separate master-weight optimizer.
    Preserve parameter identity, tied weights, frozen weights and all buffers.
    """
    validate_trainable_dtype(value)
    if value == "float32":
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.is_floating_point():
                    parameter.data = parameter.data.to(dtype=torch.float32)
    model._trainable_dtype = value
    return model
