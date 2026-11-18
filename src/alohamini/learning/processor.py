"""Shared tensor transforms for training and inference; no robot execution logic."""

import numpy as np
import torch
import torch.nn.functional as F

from alohamini.learning.data_index import ordered_indices
from alohamini.policies.configuration import NormalizationMode

DEFAULT_IMAGE_SIZE = (480, 640)  # Height, width; original AlohaMini ACT resolution.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def source_image_size(settings):
    """Preserve recorded pixels until an algorithm's own image processor runs.

    The current dataset interface has one common H/W. Heterogeneous source
    camera sizes require an explicit shared resize rather than a silent stretch.
    """
    from alohamini.datasets.tools import dataset_info

    info = dataset_info(settings["dataset"])
    selected = settings.get("cameras")
    shapes = set()
    for key, feature in info["features"].items():
        if not key.startswith("observation.images."):
            continue
        if selected is not None and key.removeprefix("observation.images.") not in selected:
            continue
        shape = feature["shape"]
        shapes.add(tuple(shape[-2:] if shape[0] == 3 else shape[:2]))
    if len(shapes) != 1:
        raise ValueError(
            "Selected cameras have different source sizes; set --image_size explicitly"
        )
    return list(shapes.pop())


def validate_statistics(stats, expected, names, *, manifest=None):
    if set(stats) != {*expected, "action"}:
        raise ValueError("Checkpoint is missing normalization statistics")
    for key, values in stats.items():
        shape = (
            (3, 1, 1)
            if key.startswith("observation.images.")
            else ((len(names),) if key == "action" else expected[key][1])
        )
        windows = (manifest or {}).get("config", {}).get("normalization_windows", {})
        if key in windows:
            if values.get("offsets") != windows[key]:
                raise ValueError(f"Checkpoint temporal normalization offsets mismatch: {key}")
            shape = (len(windows[key]), *shape)
        elif "offsets" in values:
            raise ValueError(f"Unexpected temporal normalization statistics: {key}")
        if any(np.asarray(values[k]).shape != shape for k in ("mean", "std")):
            raise ValueError(f"Checkpoint normalization shape mismatch: {key}")


def act_statistics(samples):
    """Fit numeric training fields and use fixed RGB statistics for ACT/AM-ACT.

    Checkpoint loading must use saved statistics instead of calling this recipe.
    Dataset.statistics() remains an empirical view for other policies/analysis.
    """
    image_keys = {
        key for key, feature in samples.input_features.items() if feature.type == "VISUAL"
    }
    stats = samples.statistics(keys=[key for key in samples.sample_keys if key not in image_keys])
    for key in image_keys:
        stats[key] = {
            "mean": [[[value]] for value in IMAGENET_MEAN],
            "std": [[[value]] for value in IMAGENET_STD],
        }
    return stats


def image_tensor(
    rgb, size=DEFAULT_IMAGE_SIZE, *, return_uint8=False, resize_mode="stretch", pad_value=0.0
):
    """Convert decoded HWC uint8 RGB to CHW float32 in [0, 1]."""
    # Keep snapshot ownership while avoiding Tensor -> NumPy -> Tensor for video.
    tensor = (
        rgb.detach().clone()
        if isinstance(rgb, torch.Tensor)
        else torch.from_numpy(np.array(rgb, copy=True))
    )
    tensor = tensor.permute(2, 0, 1)
    if resize_mode not in ("stretch", "pad") or not 0 <= pad_value <= 1:
        raise ValueError("image_resize_mode must be stretch/pad; image_pad_value must be in [0,1]")
    if return_uint8 and tuple(tensor.shape[1:]) == tuple(size):
        return tensor
    # Resizing stays on the existing float path: uint8 interpolation/requantizing
    # would change model inputs and the empirical image statistics.
    tensor = tensor.float() / 255
    if tuple(tensor.shape[1:]) != tuple(size):
        target = size
        if resize_mode == "pad":
            ratio = min(size[0] / tensor.shape[1], size[1] / tensor.shape[2])
            target = (max(1, int(tensor.shape[1] * ratio)), max(1, int(tensor.shape[2] * ratio)))
        tensor = F.interpolate(
            tensor[None], size=target, mode="bilinear", align_corners=False, antialias=True
        )[0]
        if resize_mode == "pad":
            height, width = size[0] - target[0], size[1] - target[1]
            tensor = F.pad(
                tensor,
                (width // 2, width - width // 2, height // 2, height - height // 2),
                value=pad_value,
            )
    return tensor


class Processor:
    """Normalize declared fields, preserving masks, labels and autograd.

    Modes are resolved per field from the policy configuration. Unlisted fields
    pass through unchanged apart from device placement. With no explicit modes,
    only fields present in stats use MEAN_STD. Statistics remain fixed; neither
    validation nor inference fits new statistics.
    """

    def __init__(
        self, stats, device="cpu", *, modes=None, clip=None, quantile_options=None,
        normalization_windows=None,
    ):
        self.device = torch.device(device)
        self._image_divisor = (
            torch.tensor(255.0, dtype=torch.float32, device=self.device)
            if self.device.type == "cuda"
            else None
        )
        self.modes = {
            key: NormalizationMode(mode)
            for key, mode in (
                modes if modes is not None else dict.fromkeys(stats, "MEAN_STD")
            ).items()
        }
        if quantile_options is not None and not isinstance(quantile_options, dict):
            raise ValueError("quantile_options must map field names to options")
        self.quantile_options = dict(quantile_options or {})
        for key, options in self.quantile_options.items():
            if (
                self.modes.get(key) != NormalizationMode.QUANTILES
                or not isinstance(options, dict)
                or options.keys() - {"epsilon", "clip_to_bounds", "zero_if_zero_bounds"}
            ):
                raise ValueError(f"{key}: invalid quantile options")
            epsilon = options.get("epsilon", 0.0)
            if (
                isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not np.isfinite(epsilon)
                or epsilon < 0
                or any(
                    type(options[name]) is not bool
                    for name in ("clip_to_bounds", "zero_if_zero_bounds")
                    if name in options
                )
            ):
                raise ValueError(f"{key}: invalid quantile epsilon or bounds policy")
            self.quantile_options[key] = dict(options)
        # Clipping is a policy decision, not implied by a normalization mode.
        self.clip = dict(clip or {})
        for key, bounds in self.clip.items():
            if (
                key not in self.modes
                or not isinstance(bounds, (list, tuple))
                or len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{key}: clip requires finite increasing [low, high] bounds")
        self.stats = {}
        self.normalization_windows = dict(normalization_windows or {})
        for key, offsets in self.normalization_windows.items():
            if (
                key not in self.modes or self.modes[key] == NormalizationMode.IDENTITY
                or not isinstance(offsets, (list, tuple)) or not offsets
                or any(type(n) is not int for n in offsets) or len(set(offsets)) != len(offsets)
                or stats.get(key, {}).get("offsets") != list(offsets)
            ):
                raise ValueError(f"{key}: temporal normalization needs matching declared offsets")
        self.decode_before_queue = "action" in self.normalization_windows
        for key, mode in self.modes.items():
            if mode == NormalizationMode.IDENTITY:
                continue
            fields = {
                NormalizationMode.MIN_MAX: ("min", "max"),
                NormalizationMode.QUANTILES: ("q01", "q99"),
                NormalizationMode.MEAN_STD: ("mean", "std"),
            }[mode]
            if key not in stats or not set(fields) <= stats[key].keys():
                raise ValueError(f"{key}: {mode.value} requires {fields} statistics")
            values = {
                name: torch.tensor(stats[key][name], dtype=torch.float32, device=self.device)
                for name in fields
            }
            low, high = (values[name] for name in fields)
            if key in self.normalization_windows and (
                low.ndim != 2 or low.shape[0] != len(self.normalization_windows[key])
            ):
                raise ValueError(f"{key}: temporal statistics require [time, coordinate] arrays")
            if (
                low.shape != high.shape
                or not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or (high < (0 if mode == NormalizationMode.MEAN_STD else low)).any()
            ):
                raise ValueError(f"{key}: Invalid normalization statistics")
            self.stats[key] = values
        self.quantile_spans = {}
        for key, mode in self.modes.items():
            if mode == NormalizationMode.QUANTILES:
                span = self.stats[key]["q99"] - self.stats[key]["q01"]
                span = span + self.quantile_options.get(key, {}).get("epsilon", 0.0)
                self.quantile_spans[key] = torch.where(span == 0, torch.full_like(span, 1e-8), span)

    @classmethod
    def from_config(cls, config, stats, device="cpu"):
        features = {**config.input_features, **config.output_features}
        modes = {
            key: config.normalization_mapping.get(feature.type, "IDENTITY")
            for key, feature in features.items()
        }
        return cls(
            stats,
            device,
            modes=modes,
            clip=getattr(config, "normalization_clip", None),
            quantile_options=getattr(config, "quantile_options", None),
            normalization_windows=getattr(config, "normalization_windows", None),
        )

    def __call__(self, batch):
        result = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                if key in self.modes or key.startswith("observation.images."):
                    raise TypeError(f"{key}: policy numeric/image fields require tensors")
                result[key] = value
                continue
            if key.startswith("observation.images.") and value.dtype == torch.uint8:
                if value.device.type == "cpu" and self.device.type == "cuda":
                    # Transfer one byte per pixel. A device divisor avoids CUDA's
                    # CPU-scalar reciprocal shortcut and retains CPU /255 rounding.
                    value = value.to(self.device).float() / self._image_divisor
                else:
                    value = value.float() / 255
            value = value.to(self.device)
            if key in self.normalization_windows and (
                value.ndim != 3 or value.shape[1] != len(self.normalization_windows[key])
                or value.shape[2] != next(iter(self.stats[key].values())).shape[1]
            ):
                raise ValueError(f"{key}: temporal normalization needs a complete [B,T,D] window")
            if not key.endswith("_is_pad") and key in self.stats:
                stats = self.stats[key]
                if self.modes[key] == NormalizationMode.MIN_MAX:
                    span = stats["max"] - stats["min"]
                    center = torch.where(
                        span < 1e-4, stats["min"], (stats["max"] + stats["min"]) / 2
                    )
                    # The author's limits normalizer maps constant dimensions to zero.
                    span = torch.where(span < 1e-4, torch.full_like(span, 2.0), span)
                    value = (value - center) * (2 / span)
                elif self.modes[key] == NormalizationMode.QUANTILES:
                    options = self.quantile_options.get(key, {})
                    if options.get("clip_to_bounds", False):
                        value = value.clamp(min=stats["q01"], max=stats["q99"])
                    value = 2 * (value - stats["q01"]) / self.quantile_spans[key] - 1
                    if options.get("zero_if_zero_bounds", False):
                        zero = (stats["q01"] == 0) & (stats["q99"] == 0)
                        value = value.masked_fill(zero, 0)
                else:
                    value = (value - stats["mean"]) / (stats["std"] + 1e-8)
            if not key.endswith("_is_pad") and key in self.clip:
                value = value.clamp(*self.clip[key])
            result[key] = value
        return result

    def unnormalize(self, key, tensor):
        """Restore recorded units without detaching, clipping or execution scaling."""
        if key not in self.modes:
            raise KeyError(f"No normalization mode configured for {key}")
        if self.modes[key] == NormalizationMode.IDENTITY:
            return tensor
        stats = self.stats[key]
        if key in self.normalization_windows:
            if tensor.ndim != 3 or not 1 <= tensor.shape[1] <= len(self.normalization_windows[key]):
                raise ValueError("Decode temporal normalization on a batched chunk before queuing")
            stats = {name: value[:tensor.shape[1]] for name, value in stats.items()}
        if self.modes[key] == NormalizationMode.QUANTILES:
            span = self.quantile_spans[key]
            if key in self.normalization_windows:
                span = span[:tensor.shape[1]]
            result = (tensor + 1) * (span / 2) + stats["q01"]
            if self.quantile_options.get(key, {}).get("zero_if_zero_bounds", False):
                zero = (stats["q01"] == 0) & (stats["q99"] == 0)
                result = result.masked_fill(zero, 0)
            return result
        if self.modes[key] == NormalizationMode.MIN_MAX:
            span = stats["max"] - stats["min"]
            center = torch.where(span < 1e-4, stats["min"], (stats["max"] + stats["min"]) / 2)
            span = torch.where(span < 1e-4, torch.full_like(span, 2.0), span)
            return tensor * (span / 2) + center
        return tensor * stats["std"] + stats["mean"]

    def action(self, tensor, *, context=None):
        return self.unnormalize("action", tensor)


def scale_action(action, dims=(), scale=1.0):
    """Scale selected physical outputs after unnormalization for execution."""
    if not dims or scale == 1.0:
        return action
    action = action.clone()
    action[..., list(dims)] *= scale
    return action


def limits_statistics(samples):
    keys = [k for k in samples.sample_keys if not k.startswith("observation.images.")]
    stats = samples.statistics(keys=keys)
    for key in keys:
        # Numeric field views already fit limits in their transformed units.
        # Reading source._value here would replace them with raw joint limits.
        if "min" in stats[key] and "max" in stats[key]:
            continue
        minimum = maximum = None
        for i in ordered_indices(samples._used_rows[key]):
            value = samples._value(i, key).double()
            minimum = value if minimum is None else torch.minimum(minimum, value)
            maximum = value if maximum is None else torch.maximum(maximum, value)
        stats[key].update(min=minimum.tolist(), max=maximum.tolist())
    for key in samples.input_features:
        if key.startswith("observation.images."):
            stats[key] = {
                "mean": [[[0.5]]] * 3,
                "std": [[[0.5]]] * 3,
                "min": [[[0.0]]] * 3,
                "max": [[[1.0]]] * 3,
            }
    return stats


def quantile_statistics(samples):
    """Fit training-only numeric quantiles without decoding images or padded duplicates.

    Field views may already supply statistics in their transformed units; retain
    those values rather than reading raw joints through a delegated `_value`.
    """
    from alohamini.datasets.statistics import ExactQuantileStats

    stats = limits_statistics(samples)
    for key in samples.sample_keys:
        if key.startswith("observation.images.") or {"q01", "q99"} <= stats[key].keys():
            continue
        tracker = ExactQuantileStats()
        batch = []
        for index in ordered_indices(samples._used_rows[key]):
            batch.append(samples._value(index, key).numpy())
            if len(batch) == 128:
                tracker.update(np.asarray(batch))
                batch.clear()
        if batch:
            tracker.update(np.asarray(batch))
        stats[key].update(
            {name: value.tolist() for name, value in tracker.get_statistics().items()}
        )
    return stats
