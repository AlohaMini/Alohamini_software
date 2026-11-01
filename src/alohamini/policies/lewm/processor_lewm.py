"""Shared normalization, physical action bounds and explicit goal images."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from alohamini.learning.processor import Processor, image_tensor


class LeWMProcessor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        values = stats["action"]
        mean = torch.tensor(values["mean"], device=device)
        std = torch.tensor(values["std"], device=device) + 1e-8
        self.low = (torch.tensor(values["min"], device=device) - mean) / std
        self.high = (torch.tensor(values["max"], device=device) - mean) / std
        if (
            self.low.shape != mean.shape
            or self.high.shape != mean.shape
            or not torch.isfinite(self.low).all()
            or not torch.isfinite(self.high).all()
            or (self.low > self.high).any()
        ):
            raise ValueError("LeWM requires finite ordered training action bounds")

    def __call__(self, batch):
        result = self.numeric({k: v for k, v in batch.items() if k != "task"})
        result.update(_action_low=self.low, _action_high=self.high)
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)

    def prepare_goal(self, images, *, resize_mode="stretch", pad_value=0.0):
        if not isinstance(images, dict):
            raise ValueError("Goal images must map camera names to uint8 HWC RGB or local paths")
        renamed = {
            k if k.startswith("observation.images.") else "observation.images." + k: v
            for k, v in images.items()
        }
        if len(renamed) != len(images) or set(renamed) != set(self.config.image_features):
            raise ValueError("Goal images must cover exactly the checkpoint cameras")
        batch = {}
        for key, feature in self.config.image_features.items():
            value = renamed[key]
            if isinstance(value, (str, Path)):
                with Image.open(Path(value).expanduser()) as image:
                    value = np.array(image.convert("RGB"))
            value = torch.as_tensor(value)
            if value.dtype != torch.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                raise ValueError("Goal images must be uint8 HWC RGB")
            batch[key] = image_tensor(
                value, feature.shape[1:], resize_mode=resize_mode, pad_value=pad_value
            )[None]
        return self.numeric(batch)
