# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache

import torch

# Pillow's 8-bit resampling keeps fixed-point weights with this many fractional bits.
_PIL_PRECISION_BITS = 22


@lru_cache
def _pil_bilinear_weights(in_size: int, out_size: int) -> tuple[list[list[int]], list[list[int]]]:
    """Indices and fixed-point Pillow BILINEAR weights along one axis."""
    scale = in_size / out_size
    support = max(scale, 1.0)
    indices, weights = [], []
    for i in range(out_size):
        center = (i + 0.5) * scale
        start = max(int(center - support + 0.5), 0)
        stop = min(int(center + support + 0.5), in_size)
        taps = [max(1 - abs((j - center + 0.5) / support), 0.0) for j in range(start, stop)]
        total = sum(taps)
        indices.append(list(range(start, stop)))
        weights.append([int(0.5 + tap / total * (1 << _PIL_PRECISION_BITS)) for tap in taps])
    return indices, weights


def _pil_resample_axis(image: torch.Tensor, out_size: int, dim: int) -> torch.Tensor:
    """One Pillow resampling pass over `dim` of an integer image, rounded back to 8 bits."""
    indices, weights = _pil_bilinear_weights(image.shape[dim], out_size)
    taps = max(len(row) for row in indices)
    index = torch.tensor(
        [row + [row[-1]] * (taps - len(row)) for row in indices], device=image.device
    )
    weight = torch.tensor([row + [0] * (taps - len(row)) for row in weights], device=image.device)
    image = image.movedim(dim, -1)
    gathered = image[..., index.flatten()].unflatten(-1, index.shape)
    total = (gathered * weight).sum(-1) + (1 << (_PIL_PRECISION_BITS - 1))
    return (total >> _PIL_PRECISION_BITS).clamp(0, 255).movedim(-1, dim)


def _pil_bilinear_resize(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bit-exact Pillow BILINEAR resize of uint8 images on their own device."""
    image = image.long()
    if image.shape[-1] != width:
        image = _pil_resample_axis(image, width, -1)
    if image.shape[-2] != height:
        image = _pil_resample_axis(image, height, -2)
    return image.to(torch.uint8)
