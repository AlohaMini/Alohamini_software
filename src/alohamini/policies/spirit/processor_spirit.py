"""Compose shared dataset field transforms with Spirit's physical batch API."""

import numpy as np
import torch

from alohamini.datasets.action_transforms import RelativeActions
from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.datasets.statistics import ExactQuantileStats

from .image_transforms import ColorJitter


class SpiritFields:
    def __init__(self, config):
        self.geometry = (
            KinematicFields(config.kinematic_fields)
            if config.action_representation == "eef_delta"
            else None
        )
        self.relative = None
        if self.geometry:
            linear, rotations = [], []
            for a in range(len(self.geometry.arms)):
                linear.extend(range(7 * a, 7 * a + 3))
                rotations.append(list(range(7 * a + 3, 7 * a + 6)))
            self.relative = RelativeActions(
                self.geometry.width, linear_dims=linear, rotation_groups=rotations
            )

    def encode(self, state, actions=None):
        state = np.asarray(state, dtype=np.float64)
        if not self.geometry:
            return state, actions
        observation = self.geometry.absolute(state)
        targets = (
            None
            if actions is None
            else self.relative.encode(observation, self.geometry.absolute(actions))
        )
        return observation, targets

    def decode(self, state, actions):
        if not self.geometry:
            return actions
        anchor = self.geometry.absolute(state)
        return self.geometry.restore_absolute(self.relative.decode(anchor, actions), state)


def fit_statistics(samples, config):
    fields = SpiritFields(config)
    if fields.geometry and (
        samples.selection.feature["names"] != fields.geometry.names
        or samples.info["features"]["action"]["names"] != fields.geometry.names
    ):
        raise ValueError("Kinematic source field names/order differ from selected state/actions")
    trackers = {key: ExactQuantileStats() for key in ("observation.state", "action")}
    anchors = samples.sample_indices
    action_indices = sorted(samples._used_rows["action"])
    states = np.asarray(
        [samples.selection.frame(samples.rows[i]) for i in anchors], dtype=np.float64
    )
    actions = np.asarray([samples.rows[i]["action"] for i in action_indices], dtype=np.float64)
    if fields.geometry:
        states = fields.geometry.absolute(states)
        actions = fields.geometry.absolute(actions)
    action_rows = dict(zip(action_indices, actions, strict=True))
    for state, anchor in zip(states, anchors, strict=True):
        indices, _ = samples._get_query_indices(anchor)
        targets = np.asarray([action_rows[i] for i in indices["action"]])
        if fields.relative:
            targets = fields.relative.encode(state, targets)
        trackers["observation.state"].update(state[None])
        trackers["action"].update(targets)
    result = {}
    for key, tracker in trackers.items():
        stats = tracker.get_statistics()
        result[key] = {"min": stats["q01"].tolist(), "max": stats["q99"].tolist()}
    return result


class SpiritProcessor:
    def __init__(self, config, stats, device):
        self.config, self.stats, self.device = config, stats, device
        self.fields = SpiritFields(config)
        self.is_training = lambda: False
        self.jitter = ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.1, p=0.5)
        self.image_divisor = torch.tensor(255.0, dtype=torch.float32, device=device)

    def __call__(self, batch):
        state = batch["observation.state"]
        targets = batch.get("action")
        encoded, actions = self.fields.encode(
            state.detach().cpu().numpy(),
            None if targets is None else targets.detach().cpu().numpy(),
        )
        result = {k: v for k, v in batch.items() if k.startswith("_")}
        for slot, source in self.config.camera_mapping.items():
            image = batch[f"observation.images.{source}"].to(self.device)
            if image.dtype == torch.uint8:
                image = image.float() / self.image_divisor
            elif not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f"{source}: expected CHW RGB in [0,1]")
            result[f"observation.images.{slot}"] = image
        if self.config.image_augmentation and self.is_training():
            for key, image in list(result.items()):
                if key.startswith("observation.images."):
                    result[key] = torch.stack([self.jitter(frame) for frame in image])
        result["observation.state"] = torch.as_tensor(
            np.asarray(encoded, dtype=np.float32), device=self.device
        )
        result["_action_context"] = state.to(self.device)
        tasks = batch.get("task")
        if (
            not isinstance(tasks, (tuple, list))
            or len(tasks) != len(state)
            or any(not isinstance(t, str) or not t.strip() for t in tasks)
        ):
            raise ValueError("Spirit requires one nonempty task string per observation")
        result["task"] = list(tasks)
        result["robot_type"] = [self.config.robot_type] * len(state)
        if actions is not None:
            result["action"] = torch.as_tensor(
                np.asarray(actions, dtype=np.float32), device=self.device
            )
            padding = batch["action_is_pad"].to(self.device)
            result["action_is_pad"] = padding
            result["action_mask"] = (~padding)[..., None].expand_as(result["action"])
        return result

    def action(self, tensor, *, context=None):
        if not self.fields.geometry:
            return tensor
        if context is None:
            raise ValueError("Cartesian delta actions require their original observation anchor")
        single = tensor.ndim == 2
        values = tensor[:, None] if single else tensor
        restored = self.fields.decode(context.detach().cpu().numpy(), values.detach().cpu().numpy())
        result = torch.as_tensor(restored, device=tensor.device, dtype=tensor.dtype)
        return result[:, 0] if single else result
