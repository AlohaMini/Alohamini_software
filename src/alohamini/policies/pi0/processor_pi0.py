"""OpenPI Pi0 mean/std normalization and prompt-only SentencePiece tokens."""

import numpy as np
import torch

from alohamini.policies.pi05.adapter import PI05BatchProcessor
from alohamini.policies.pi05.processor_pi05 import PI05Processor, validate_camera_names


class PI0Processor(PI05Processor):
    def __post_init__(self):
        if self.cameras is not None:
            self.cameras = validate_camera_names(self.cameras)
        if set(self.stats) != {"state", "actions"}:
            raise ValueError("Pi0 requires state and actions statistics")
        self.stats = {
            key: {name: np.asarray(values[name], np.float64) for name in ("mean", "std")}
            for key, values in self.stats.items()
        }
        for values in self.stats.values():
            if any(v.shape != (18,) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Pi0 statistics require 18 finite named coordinates")
            if (values["std"] < 0).any():
                raise ValueError("Pi0 standard deviation cannot be negative")
        self.delta_dims = tuple(self.delta_dims)
        if len(set(self.delta_dims)) != len(self.delta_dims) or any(
            type(i) is not int or not 0 <= i < 18 for i in self.delta_dims
        ):
            raise ValueError("Invalid Pi0 relative joint indices")

    def normalize(self, key, value):
        stats = self.stats[key]
        values = np.asarray(value, np.float64)
        if values.shape[-1] != 18 or not np.isfinite(values).all():
            raise ValueError("Pi0 inputs require finite named 18D coordinates")
        return ((values - stats["mean"]) / (stats["std"] + 1e-6)).astype(np.float32)

    def restore_actions(self, actions, state):
        values = actions[..., :18].double()
        stats = self.stats["actions"]
        mean = torch.as_tensor(stats["mean"], device=values.device)
        std = torch.as_tensor(stats["std"], device=values.device)
        state = torch.as_tensor(state, device=values.device, dtype=torch.float64)
        offset = torch.zeros_like(state)
        offset[..., list(self.delta_dims)] = state[..., list(self.delta_dims)]
        return (values * (std + 1e-6) + mean + offset.unsqueeze(-2)).float()

    def encode_actions(self, actions, state):
        stats = self.stats["actions"]
        mean = torch.as_tensor(stats["mean"], device=actions.device)
        std = torch.as_tensor(stats["std"], device=actions.device)
        state = torch.as_tensor(state, device=actions.device, dtype=torch.float64)
        offset = torch.zeros_like(state)
        offset[..., list(self.delta_dims)] = state[..., list(self.delta_dims)]
        return ((actions.double() - offset.unsqueeze(-2) - mean) / (std + 1e-6)).float()


class ContinuousTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, prompt, state=None):
        return self.tokenizer.tokenize(prompt)


class PI0BatchProcessor(PI05BatchProcessor):
    transform_class = PI0Processor

    def __init__(self, config, stats, device):
        super().__init__(config, stats, device)
        self.transform.tokenizer = ContinuousTokenizer(self.transform.tokenizer)
