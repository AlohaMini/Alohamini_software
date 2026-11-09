"""OpenWAM-alpha's named 80D boundary, not a learned joint-to-pose projection.

Official rotation6d concatenates two matrix columns. Slots 68:72 are our
explicit mobile-platform extension; upstream does not standardize their meaning.
FK/IK and frozen calibration live in the shared FieldTransforms layer.
"""

from copy import deepcopy

import numpy as np
import torch

from alohamini.datasets.rotations import RotationEncoding
from alohamini.learning.processor import Processor
from alohamini.policies.chunk import TaskProcessor

ROTATION = "rotation_6d_columns"
WIDTH = 80
AUXILIARY = {
    "x.vel": (68, "m/s"),
    "y.vel": (69, "m/s"),
    "theta.vel": (70, "deg/s"),
    "lift_axis.height_mm": (71, "mm"),
}


def semantic_fields(grippers):
    fields = dict(AUXILIARY)
    for side, start in (("left", 0), ("right", 34)):
        for offset, name in enumerate(("x", "y", "z", *RotationEncoding(ROTATION).names)):
            fields[f"{side}.{name}"] = (start + offset, "m" if offset < 3 else "dimensionless")
        if f"{side}.gripper" in grippers:
            fields[f"{side}.gripper"] = (start + 9, grippers[f"{side}.gripper"]["unit"])
    return fields


def make_layout(transform, features):
    if (
        transform.state != "tcp"
        or transform.action != "tcp_absolute"
        or transform.geometry.rotation.representation != ROTATION
    ):
        raise ValueError("OpenWAM alpha80 requires TCP state, absolute TCP actions and column 6D")
    descriptor = transform.geometry.descriptor
    frames, grippers = {}, {}
    for arm in descriptor["arms"]:
        side = arm["name"]
        if side not in ("left", "right") or side in frames or not arm.get("gripper"):
            raise ValueError("alpha80 requires explicitly named left/right arms with grippers")
        frames[side] = arm["frame"]
        grippers[f"{side}.gripper"] = {
            key: arm["gripper"][key] for key in ("closed", "open", "unit")
        }
    slots = semantic_fields(grippers)
    result = dict(version=1, rotation=ROTATION, frames=frames, grippers=grippers, features={})
    for key in ("observation.state", "action"):
        feature = features[key]
        if any(name not in slots for name in feature["names"]):
            raise ValueError(f"alpha80 has no declared semantic slot for {feature['names']}")
        result["features"][key] = dict(
            names=list(feature["names"]),
            units=list(feature["units"]),
            slots=[slots[name][0] for name in feature["names"]],
        )
    validate_layout(result)
    return result


def validate_layout(layout, *, state_width=None, action_width=None):
    if layout.get("version") != 1 or layout.get("rotation") != ROTATION:
        raise ValueError("alpha80 requires a saved version 1 semantic layout with column 6D")
    frames, grips = layout["frames"], layout["grippers"]
    if not frames or set(frames) - {"left", "right"} or not all(frames.values()):
        raise ValueError("alpha80 requires explicit left/right reference frames")
    if set(grips) != {f"{side}.gripper" for side in frames}:
        raise ValueError("alpha80 grippers must match the declared arms")
    for grip in grips.values():
        if (
            not np.isfinite([grip["closed"], grip["open"]]).all()
            or grip["open"] <= grip["closed"]
            or not grip["unit"]
        ):
            raise ValueError("alpha80 grippers require finite increasing closed/open endpoints")
    declared = semantic_fields(grips)
    if set(layout["features"]) != {"observation.state", "action"}:
        raise ValueError("alpha80 requires state and action semantic layouts")
    for key, width in (("observation.state", state_width), ("action", action_width)):
        feature = layout["features"][key]
        names, units, slots = (feature[k] for k in ("names", "units", "slots"))
        if (
            not names
            or len(set(names)) != len(names)
            or len(units) != len(names)
            or len(slots) != len(names)
            or (width is not None and len(names) != width)
            or any(name not in declared for name in names)
            or any(type(slot) is not int for slot in slots)
            or list(zip(slots, units, strict=True)) != [declared[name] for name in names]
        ):
            raise ValueError(f"Invalid alpha80 names, units or slot mapping: {key}")
        # Every declared arm needs the complete pose and gripper in both streams.
        required = {name for name in declared if name.split(".")[0] in frames}
        if not required <= set(names) or any(
            name.startswith(("left.", "right.")) and name.split(".")[0] not in frames
            for name in names
        ):
            raise ValueError(f"alpha80 requires complete named arm fields: {key}")


def normalization_bounds(layout, key):
    """Only rotations and physical gripper endpoints override fitted train limits."""
    feature = layout["features"][key]
    result = {}
    for i, (name, slot) in enumerate(zip(feature["names"], feature["slots"], strict=True)):
        if slot in (*range(3, 9), *range(37, 43)):
            result[i] = (-1.0, 1.0)
        elif name in layout["grippers"]:
            grip = layout["grippers"][name]
            result[i] = (grip["closed"], grip["open"])
    return result


def semantic_statistics(stats, layout):
    result = deepcopy(stats)
    for key in layout["features"]:
        for index, (low, high) in normalization_bounds(layout, key).items():
            result[key]["min"][index], result[key]["max"][index] = low, high
            result[key]["mean"][index], result[key]["std"][index] = (
                (low + high) / 2,
                (high - low) / 2,
            )
    return result


def validate_limits(stats, layout):
    for key, feature in layout["features"].items():
        low, high = (np.asarray(stats[key][name]) for name in ("min", "max"))
        if (
            low.shape != (len(feature["names"]),)
            or high.shape != low.shape
            or not np.isfinite(low).all()
            or not np.isfinite(high).all()
            or (high < low).any()
        ):
            raise ValueError(f"Invalid alpha80 normalization limits: {key}")
        for i, bounds in normalization_bounds(layout, key).items():
            if (low[i], high[i]) != bounds:
                raise ValueError(f"alpha80 rotation/gripper normalization mismatch: {key}[{i}]")


class Alpha80NumericProcessor(Processor):
    def __init__(self, config, stats, device):
        super().__init__(
            stats,
            device,
            modes={key: "IDENTITY" for key in (*config.input_features, *config.output_features)},
        )
        validate_limits(stats, config.alpha80_layout)
        self.limits = {
            key: tuple(
                torch.tensor(stats[key][name], dtype=torch.float32, device=device)
                for name in ("min", "max")
            )
            for key in config.alpha80_layout["features"]
        }

    def __call__(self, batch):
        result = super().__call__(batch)
        for key, (low, high) in self.limits.items():
            if key in result:
                span = high - low
                value = 2 * (result[key] - low) / torch.where(span == 0, 1, span) - 1
                result[key] = torch.where(span == 0, 0, value).clamp(-1, 1)
        return result

    def unnormalize(self, key, tensor):
        low, high = self.limits[key]
        # Exact constant training labels remain constant even with diffusion residuals.
        # This is also the platform contract for a stationary base/lift.
        return (tensor + 1) * ((high - low) / 2) + low


class Alpha80Processor(TaskProcessor):
    def __init__(self, config, stats, device):
        self.numeric = Alpha80NumericProcessor(config, stats, device)


def scatter(values, slots):
    result = values.new_zeros((*values.shape[:-1], WIDTH))
    result[..., slots] = values
    return result


def action_padding(padding, values, slots):
    if padding.ndim == values.ndim - 1:
        padding = padding[..., None].expand_as(values)
    if padding.shape != values.shape:
        raise ValueError("alpha80 action padding must match time or time/coordinate dimensions")
    result = torch.ones((*values.shape[:-1], WIDTH), dtype=torch.bool, device=values.device)
    result[..., slots] = padding
    return result
