"""Trainable weight storage, independent of autocast compute precision."""

import torch

TRAINABLE_DTYPES = ("preserve", "float32")


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
