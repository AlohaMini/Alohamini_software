"""Causal history windows shared by local and remote native inference."""

import math
from bisect import bisect_right
from collections import deque

import torch


class ObservationHistory:
    def __init__(self, offsets, fps):
        self.offsets = {key: tuple(value) for key, value in offsets.items()}
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("History requires a positive recording FPS")
        for value in self.offsets.values():
            if not value or value[-1] != 0 or any(type(i) is not int or i > 0 for i in value):
                raise ValueError("Live history offsets must be causal and end at zero")
            if list(value) != sorted(set(value)):
                raise ValueError("History offsets must be ordered and unique")
        self.fps = fps
        self.span = max((-min(value) for value in self.offsets.values()), default=0) / fps
        self.frames = deque()

    def append(self, batch, timestamp):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("History requires a finite source-client observation timestamp")
        if self.frames and timestamp < self.frames[-1][0]:
            raise ValueError("Observation clock moved backwards; reset the policy first")
        if not self.frames or timestamp > self.frames[-1][0]:
            self.frames.append(
                (timestamp, {key: batch[key].detach().clone() for key in self.offsets})
            )
        # Keep one sample before the earliest requested time for causal sample-and-hold.
        while len(self.frames) > 1 and self.frames[1][0] <= timestamp - self.span:
            self.frames.popleft()
        stamps = [frame[0] for frame in self.frames]
        result = dict(batch)
        for key, offsets in self.offsets.items():
            wanted = [timestamp + offset / self.fps for offset in offsets]
            indices = [max(0, bisect_right(stamps, value + 1e-9) - 1) for value in wanted]
            result[key] = torch.stack([self.frames[i][1][key] for i in indices], dim=1)
            result[key + "_is_pad"] = torch.tensor(
                [value < stamps[0] - 1e-9 for value in wanted],
                dtype=torch.bool,
                device=batch[key].device,
            )[None].expand(batch[key].shape[0], -1)
        return result
