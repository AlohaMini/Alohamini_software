"""Common training data views and reversible numeric processor composition."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.kinematic_fields import descriptor_from_dataset
from alohamini.datasets.statistics import ExactQuantileStats
from alohamini.policies.configuration import PolicyFeature


def active(settings):
    return any(
        settings.get(name, "recorded") != "recorded"
        for name in ("state_representation", "action_representation")
    )


def prepare_fields(samples, settings, components, *, options=None):
    """Resolve coordinate metadata once, without an intermediate user-managed file."""
    configuration = deepcopy(settings.get("data", {}))
    if not active(configuration):
        return samples
    if not getattr(components, "supports_field_transforms", False):
        raise ValueError("This algorithm has not registered the common numeric-field adapter")
    if (
        configuration.get("action_representation", "recorded") != "recorded"
        and options
        and options.get("temporal_ensemble_coeff") is not None
    ):
        raise ValueError("TCP actions require chunk execution without ACT temporal blending")
    configuration.setdefault("version", 1)
    kinematics = configuration.pop("kinematics", {})
    if kinematics and configuration.get("kinematic_fields"):
        raise ValueError("Choose either a frozen kinematic descriptor or a new kinematics source")
    if not configuration.get("kinematic_fields"):
        choices = [name for name in ("reference_pose", "path", "spec") if kinematics.get(name)]
        if len(choices) != 1:
            raise ValueError("Choose one data.kinematics.reference_pose, path or spec")
        if choices[0] == "reference_pose":
            descriptor = descriptor_from_dataset(
                samples.root, reference_pose=kinematics["reference_pose"]
            )
        else:
            path = Path(kinematics[choices[0]]).expanduser().resolve()
            specification = json.loads(path.read_text())
            descriptor = descriptor_from_dataset(
                samples.root, spec=specification, directory=path.parent
            )
        configuration["kinematic_fields"] = descriptor
    selection = dict(
        source_state_names=samples.info["features"].get("observation.state", {}).get("names", []),
        selected_state_feature=samples.selection.feature if samples.selection else None,
        selected_state_units=samples.selection.units if samples.selection else [],
    )
    for key, value in selection.items():
        if key in configuration and configuration[key] != value:
            raise ValueError(f"Data field-transform source selection mismatch: {key}")
        configuration[key] = deepcopy(value)
    transform = FieldTransforms(configuration)
    transform.validate_source(
        samples.info, samples.selection.feature if samples.selection else None
    )
    settings["data"] = deepcopy(configuration)
    return FieldDataset(samples, transform)


class FieldDataset(torch.utils.data.Dataset):
    """A numeric view preserving raw records, image reads and episode/window indices."""

    def __init__(self, source, transform):
        self.source, self.field_transform = source, transform
        transform.validate_source(
            source.info, source.selection.feature if source.selection else None
        )
        self.model_features = transform.features(
            source.selection.feature if source.selection else None,
            source.selection.units if source.selection else [],
            source.info["features"]["action"],
        )
        self.input_features = deepcopy(source.input_features)
        if "observation.state" in self.model_features:
            self.input_features["observation.state"] = PolicyFeature(
                "STATE", tuple(self.model_features["observation.state"]["shape"])
            )
        self.output_features = {
            "action": PolicyFeature("ACTION", tuple(self.model_features["action"]["shape"]))
        }

    def __getattr__(self, name):
        source = self.__dict__.get("source")
        if source is None:
            raise AttributeError(name)
        return getattr(source, name)

    def __len__(self):
        return len(self.source)

    def _convert(self, sample, index):
        anchor = self.field_transform.context(self.rows[self.sample_indices[index]])
        state, actions = self.field_transform.encode(
            sample["observation.state"].numpy() if "observation.state" in sample else None,
            sample["action"].numpy() if "action" in sample else None,
            anchor=anchor,
        )
        result = {
            **sample,
            "_action_context": torch.as_tensor(anchor, dtype=torch.float32),
            "_fields_encoded": torch.tensor(True),
        }
        if state is not None:
            result["observation.state"] = torch.as_tensor(state, dtype=torch.float32)
            result["_recorded_state"] = sample["observation.state"]
        if actions is not None:
            result["action"] = torch.as_tensor(actions, dtype=torch.float32)
            result["_recorded_action"] = sample["action"]
        return result

    def __getitem__(self, index):
        return self._convert(self.source[index], index)

    def observation(self, index):
        return self._convert(self.source.observation(index), index)

    def action_metadata(self, index):
        raw = self.source.action_metadata(index)
        row = self.sample_indices[index]
        state = self.selection.frame(self.rows[row]) if self.selection else None
        _, actions = self.field_transform.encode(
            state, raw["action"].numpy(), anchor=self.field_transform.context(self.rows[row])
        )
        return {**raw, "action": torch.as_tensor(actions, dtype=torch.float32)}

    def numeric_statistics(self, *, action_windows=True):
        """Fit transformed values using the algorithm's action sampling population."""
        fields = self.field_transform
        anchors = (
            fields.geometry.absolute(
                np.asarray([fields.context(self.rows[i]) for i in self.sample_indices])
            )
            if fields.action == "tcp_delta"
            else None
        )
        rows = sorted(self.source._used_rows["action"])
        actions = np.asarray([self.rows[i]["action"] for i in rows])
        if fields.action != "recorded":
            actions = fields.geometry.absolute(actions)
        lookup = dict(zip(rows, actions, strict=True))
        trackers = {"action": ExactQuantileStats()}
        if self.selection:
            state_rows = sorted(self.source._used_rows["observation.state"])
            states = np.asarray([self.selection.frame(self.rows[i]) for i in state_rows])
            context = np.asarray([fields.context(self.rows[i]) for i in state_rows])
            state_values, _ = fields.encode(states, anchor=context)
            trackers["observation.state"] = ExactQuantileStats()
            trackers["observation.state"].update(state_values)
        if action_windows or anchors is not None:
            for number, row in enumerate(self.sample_indices):
                indices, _ = self._get_query_indices(row)
                targets = np.asarray([lookup[i] for i in indices["action"]])
                if anchors is not None:
                    targets = fields.relative.encode(anchors[number], targets)
                trackers["action"].update(targets)
        else:
            trackers["action"].update(actions)
        return {
            key: {name: value.tolist() for name, value in tracker.get_statistics().items()}
            for key, tracker in trackers.items()
        }

    def recorded_state_statistics(self):
        """Retain raw selected-state means for existing fixed-input deployment controls."""
        if self.selection is None:
            return None
        if "_recorded_state_statistics" not in self.__dict__:
            rows = sorted(self.source._used_rows["observation.state"])
            values = np.asarray(
                [self.selection.frame(self.rows[i]) for i in rows], dtype=np.float64
            )
            self._recorded_state_statistics = {
                "mean": values.mean(0).tolist(),
                "std": values.std(0).tolist(),
            }
        return self._recorded_state_statistics

    def statistics(self, *, keys=None):
        keys = list(self.source._used_rows) if keys is None else list(keys)
        numeric = self.numeric_statistics(action_windows=False)
        others = [key for key in keys if key not in numeric]
        result = self.source.statistics(keys=others) if others else {}
        result.update({key: numeric[key] for key in keys if key in numeric})
        return result


class FieldProcessor:
    """Apply the same field view to live observations and invert after model decoding."""

    def __init__(self, processor, transform, device):
        self.processor, self.field_transform, self.device = processor, transform, device
        self.decode_before_queue = transform.action != "recorded"

    def __getattr__(self, name):
        return getattr(self.processor, name)

    def __call__(self, batch):
        encoded = batch.get("_fields_encoded")
        already_encoded = encoded is not None and bool(torch.as_tensor(encoded).all())
        if already_encoded:
            anchor = batch["_action_context"]
            prepared = batch
        else:
            anchor = batch.get("_action_context")
            observed = batch.get("observation.state")
            if (
                anchor is None
                and observed is not None
                and observed.ndim == 2
                and self.field_transform.selected is not None
                and self.field_transform.selected["names"] == self.field_transform.geometry.names
            ):
                anchor = observed
            if anchor is None:
                raise ValueError("TCP transforms require the original joint observation context")
            state, actions = self.field_transform.encode(
                batch["observation.state"].detach().cpu().numpy()
                if "observation.state" in batch
                else None,
                batch["action"].detach().cpu().numpy() if "action" in batch else None,
                anchor=anchor.detach().cpu().numpy(),
            )
            prepared = dict(batch)
            if state is not None:
                prepared["observation.state"] = torch.as_tensor(state, dtype=torch.float32)
            if actions is not None:
                prepared["action"] = torch.as_tensor(actions, dtype=torch.float32)
        result = self.processor(prepared)
        result["_action_context"] = anchor.to(self.device)
        return result

    def action(self, tensor, *, context=None):
        values = self.processor.action(tensor, context=context)
        if self.field_transform.action == "recorded":
            return values
        if context is None:
            raise ValueError("TCP decoding requires the original numeric observation anchor")
        single = values.ndim == 2
        actions = values[:, None] if single else values
        restored = self.field_transform.decode(
            context.detach().float().cpu().numpy(), actions.detach().float().cpu().numpy()
        )
        result = torch.as_tensor(restored, dtype=torch.float32, device=values.device)
        return result[:, 0] if single else result
