# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera canvas arithmetic adapted from the pinned Cosmos3 processor."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .backbone_configuration import RESOLUTION_TIER_LIMITS

VIDEO_CANVAS_SIZES: dict[str, dict[str, tuple[int, int]]] = {
    "256": {
        "1,1": (256, 256),
        "4,3": (320, 256),
        "3,4": (256, 320),
        "16,9": (320, 192),
        "9,16": (192, 320),
    },
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
        "2,3": (512, 768),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}


@dataclass
class VideoCanvas:
    canvas_hw: tuple | None = None

    def prepare_view(self, view: Tensor, view_size: tuple[int, int]) -> Tensor:
        """Convert current frames or clips to resized [B,T,3,h,w] views in [0,1]."""
        if view.dtype != torch.float32:
            raise TypeError(
                f"Camera views reach the canvas as float32 in [0, 1], got {view.dtype}."
            )
        if view.ndim == 4:
            view = view[:, None]
        height, width = view_size
        if tuple(view.shape[-2:]) != (height, width):
            frames = view.flatten(0, 1) * 255.0
            frames = F.interpolate(
                frames, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            view = (frames.round().clamp(0, 255) / 255.0).unflatten(0, view.shape[:2])
        return view

    def compose(self, views: list[Tensor]) -> Tensor:
        """The multi view layout of the native datasets on `[B,T,3,h,w]` views in [0, 1]."""
        if len(views) == 1:
            return views[0]
        if len(views) == 2:
            return torch.cat(views, dim=-1)
        top, *bottom = views
        height, width = top.shape[-2:]
        halves = []
        for view, half_width in zip(bottom, (width // 2, width - width // 2), strict=True):
            half = F.interpolate(
                view.flatten(0, 1),
                size=(height // 2, half_width),
                mode="bilinear",
                align_corners=False,
            )
            halves.append(half.unflatten(0, view.shape[:2]))
        return torch.cat([top, torch.cat(halves, dim=-1)], dim=-2)

    def fit_canvas(self, frames: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Fit uint8 clips into a resolution-tier canvas; return canvas and content size."""
        height, width = frames.shape[-2:]
        canvas_h, canvas_w = self.canvas_size(height, width)
        scale = min(canvas_w / width, canvas_h / height, 1.0)
        content_h, content_w = (int(scale * height + 0.5), int(scale * width + 0.5))
        batch_size, num_frames = frames.shape[:2]
        canvas = frames.flatten(0, 1)
        if (content_h, content_w) != (height, width):
            content_size = (content_h, content_w)
            canvas = F.interpolate(
                canvas.float(), content_size, mode="bicubic", align_corners=False, antialias=True
            )
            canvas = canvas.clamp(0, 255).round().to(torch.uint8)
        pad_right, pad_bottom = (canvas_w - content_w, canvas_h - content_h)
        if pad_right or pad_bottom:
            mode = "replicate" if pad_right >= content_w or pad_bottom >= content_h else "reflect"
            canvas = F.pad(canvas.float(), (0, pad_right, 0, pad_bottom), mode=mode).to(torch.uint8)
        return (canvas.unflatten(0, (batch_size, num_frames)), (content_h, content_w))

    def canvas_size(self, height: int, width: int) -> tuple[int, int]:
        """Height and width of the canvas of the resolution tier of a `height` x `width` frame."""
        if self.canvas_hw is not None:
            return tuple(self.canvas_hw)
        shorter_side = min(height, width)
        tiers = [tier for tier, limit in RESOLUTION_TIER_LIMITS if shorter_side <= limit]
        if not tiers:
            raise ValueError(
                f"Canvas {height}x{width} exceeds side limit {RESOLUTION_TIER_LIMITS[-1][1]}"
            )
        sizes = VIDEO_CANVAS_SIZES[tiers[0]].values()
        return min(sizes, key=lambda size: abs(height / width - size[1] / size[0]))[::-1]
