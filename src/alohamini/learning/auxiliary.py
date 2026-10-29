"""Declared numeric sensor/context inputs shared by native policy loading and live use."""

import torch


def auxiliary_features(info, windows, known):
    result = {}
    for key in windows:
        if key in known or not key.startswith("observation."):
            continue
        feature = info["features"].get(key)
        if feature is None or feature.get("dtype") in ("image", "video", "string"):
            raise ValueError(
                f"Checkpoint auxiliary observation is not a recorded numeric field: {key}"
            )
        shape = tuple(feature["shape"])
        if not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"Invalid auxiliary observation shape: {key}")
        result[key] = ("ENV", shape)
    return result


def snapshot_auxiliary(payload, features, info):
    result = {}
    for key, feature in features.items():
        if feature.type != "ENV":
            continue
        if key not in payload:
            raise ValueError(f"Missing required sensor/context observation: {key}")
        dtype = getattr(torch, info["features"][key]["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported numeric observation dtype: {key}")
        value = torch.as_tensor(payload[key], dtype=dtype)
        if tuple(value.shape) != feature.shape or not torch.isfinite(value).all():
            raise ValueError(f"Sensor/context observation has invalid shape or values: {key}")
        result[key] = value
    return result
