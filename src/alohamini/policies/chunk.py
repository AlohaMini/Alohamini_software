"""Shared native adapters for pretrained policies with queued action chunks."""

from collections import deque
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from alohamini.learning.processor import Processor, limits_statistics, validate_statistics


class ChunkPolicy(nn.Module):
    def reset(self):
        self._actions = deque()

    def get_optim_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    @torch.no_grad()
    def select_action(self, batch):
        if not self._actions:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._actions.extend(actions.transpose(0, 1))
        return self._actions.popleft()


class TaskProcessor:
    """Use platform normalization and retain a validated task for each sample."""

    def __init__(self, config, stats, device):
        self.numeric = Processor.from_config(config, stats, device)

    def __call__(self, batch):
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (list, tuple))
            or len(tasks) != len(batch["observation.state"])
            or any(not isinstance(task, str) or not task.strip() for task in tasks)
        ):
            raise ValueError("A nonempty task is required for every sample")
        result = self.numeric({key: value for key, value in batch.items() if key != "task"})
        result["task"] = tasks
        return result

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor)


class ChunkAlgorithm:
    include_task = True
    supports_remote_chunks = True
    loss_terms = ()
    validate_statistics = staticmethod(validate_statistics)

    @staticmethod
    def action_horizon(config):
        return config.chunk_size

    def metric_specs(self, config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        return [
            MetricSpec(key, key, "samples", loss_count("_loss_weight"), console=key)
            for key in self.loss_terms
        ]

    def statistics(self, samples, options=None):
        return limits_statistics(samples)

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": len(batch["action"])}

    def processor(self, model, stats, device):
        return TaskProcessor(model.config, stats, device)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to this policy")
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        result["assets_path"] = str(path / "assets")
        return result

    def save(self, path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        model.save_assets(path / "assets")
        manifest["config"]["assets_path"] = "assets"
        save_weights(path / "model.safetensors", model, model_state)

    def load(self, path, model):
        from safetensors.torch import load_model

        load_model(model, path / "model.safetensors", strict=True)


def local_assets(value):
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Pretrained assets must be a local directory: {path}")
    return path


def padded_mean(loss, is_pad):
    """Mean per sample over valid time and action coordinates, then over samples."""
    valid = (~is_pad).to(loss.dtype)
    while valid.ndim < loss.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.expand_as(loss)
    axes = tuple(range(1, loss.ndim))
    return (loss.mul(valid).sum(axes) / valid.sum(axes).clamp_min(1)).mean()
