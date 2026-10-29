"""Geometry operations for explicitly declared point-cloud observations."""

import torch


def normalize_points(points):
    xyz = points[..., :3].float()
    xyz = xyz - xyz.mean(-2, keepdim=True)
    radius = xyz.norm(dim=-1).amax(-1, keepdim=True).unsqueeze(-1)
    return xyz / (radius + 1e-8)


def farthest_indices(points, count):
    """Deterministic FPS starting at point zero; no custom CUDA extension."""
    batch, length, _ = points.shape
    selected = torch.empty(batch, count, dtype=torch.long, device=points.device)
    distance = torch.full((batch, length), float("inf"), device=points.device)
    current = torch.zeros(batch, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for step in range(count):
        selected[:, step] = current
        square = (points - points[rows, current, None]).square().sum(-1)
        distance = torch.minimum(distance, square)
        current = distance.argmax(-1)
    return selected


def resample_points(points, count):
    """Keep recorded points unchanged when possible; deterministic FPS otherwise."""
    if points.shape[-2] == count:
        return points
    if points.shape[-2] < count:
        indices = torch.arange(count, device=points.device) % points.shape[-2]
        return points[..., indices, :]
    flat = points.reshape(-1, *points.shape[-2:])
    indices = farthest_indices(flat[..., :3].float(), count)
    rows = torch.arange(flat.shape[0], device=points.device)[:, None]
    return flat[rows, indices].reshape(*points.shape[:-2], count, points.shape[-1])
