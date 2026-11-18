"""Normalize recorded commands before 30D placement, including initial physical zeros."""

import torch

from alohamini.learning.processor import Processor


class LingBotVAProcessor:
    def __init__(self, config, stats, device):
        self.config = config
        self.numeric = Processor.from_config(config, stats, device)
        zeros = torch.zeros(
            1, config.action_per_frame, config.action_feature.shape[0], device=self.numeric.device
        )
        self.bootstrap = self.numeric({"action": zeros})["action"]

    def __call__(self, batch):
        tasks = batch.get("task")
        if not isinstance(tasks, (list, tuple)) or any(
            not isinstance(t, str) or not t.strip() for t in tasks
        ):
            raise ValueError("LingBot-VA requires a nonempty task per sample")
        result = self.numeric(batch)
        result["action.bootstrap"] = self.bootstrap.expand(len(tasks), -1, -1)
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)
