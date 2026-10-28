"""Named numeric representations shared by algorithm data adapters."""

from copy import deepcopy

import numpy as np

from .kinematic_fields import KinematicFields


class FieldTransforms:
    """Select representations independently of model architecture and normalization."""

    def __init__(self, configuration):
        self.configuration = deepcopy(configuration)
        if configuration.get("version", 1) != 1:
            raise ValueError("Unsupported data field-transform version")
        self.state = configuration.get("state_representation", "recorded")
        self.action = configuration.get("action_representation", "recorded")
        if self.state not in ("recorded", "tcp") or self.action not in (
            "recorded",
            "tcp_absolute",
            "tcp_delta",
        ):
            raise ValueError("Unknown state/action data representation")
        self.geometry = (
            KinematicFields(configuration["kinematic_fields"])
            if self.state == "tcp" or self.action != "recorded"
            else None
        )
        self.relative = self.geometry.relative_actions() if self.geometry else None
        self.selected = configuration.get("selected_state_feature")
        self.selected_units = configuration.get("selected_state_units", [])
        self.state_projection = None
        if self.state == "tcp" and self.selected is not None:
            self._state_layout(self.selected, self.selected_units)

    def _state_layout(self, feature, units):
        absolute = self.geometry.feature()
        selected = set(feature["names"])
        coordinate_arm = {}
        source_to_tcp = {}
        for start, (_arm, indices, grip, _) in zip(
            self.geometry.arm_offsets, self.geometry.arms, strict=True
        ):
            present = [self.geometry.names[i] in selected for i in indices]
            if any(present) and not all(present):
                raise ValueError("TCP state requires the complete joint-position group for an arm")
            for i in indices:
                coordinate_arm[self.geometry.names[i]] = start
            if grip is not None:
                source_to_tcp[self.geometry.names[grip]] = start + 6
        if not selected.intersection(coordinate_arm):
            raise ValueError("TCP state requires at least one complete joint-position group")
        source_to_tcp.update(
            {
                self.geometry.names[i]: self.geometry.pose_width + j
                for j, i in enumerate(self.geometry.passthrough)
            }
        )
        projection, names, quantities, emitted = [], [], [], set()
        for index, name in enumerate(feature["names"]):
            if name in coordinate_arm:
                start = coordinate_arm[name]
                if start in emitted:
                    continue
                emitted.add(start)
                for i in range(start, start + 6):
                    projection.append(("tcp", i))
                    names.append(absolute["names"][i])
                    quantities.append(absolute["units"][i])
            elif name in source_to_tcp:
                i = source_to_tcp[name]
                projection.append(("tcp", i))
                names.append(absolute["names"][i])
                quantities.append(absolute["units"][i])
            else:
                projection.append(("selected", index))
                names.append(name)
                quantities.append(units[index])
        self.state_projection = projection
        return {**absolute, "shape": [len(names)], "names": names, "units": quantities}

    def context(self, row):
        source = self.configuration.get("source_state_names", self.geometry.names)
        indices = [source.index(name) for name in self.geometry.names]
        return np.asarray(row["observation.state"], dtype=np.float64)[indices]

    def encode(self, state, actions=None, *, anchor=None):
        """Keep the observed anchor separate from any observation-history dimension."""
        if self.state == "tcp":
            full = np.array(state if anchor is None else anchor, copy=True)
            if self.selected is not None:
                shape = (*np.shape(state)[:-1], len(self.geometry.names))
                while full.ndim < len(shape):
                    full = np.expand_dims(full, -2)
                full = np.broadcast_to(full, shape).copy()
                for i, name in enumerate(self.selected["names"]):
                    if name in self.geometry.names:
                        full[..., self.geometry.names.index(name)] = np.asarray(state)[..., i]
            absolute = self.geometry.absolute(full)
            observed = (
                np.stack(
                    [
                        absolute[..., i] if kind == "tcp" else np.asarray(state)[..., i]
                        for kind, i in self.state_projection
                    ],
                    axis=-1,
                )
                if self.state_projection is not None
                else absolute
            )
        else:
            observed = state
        if actions is None or self.action == "recorded":
            return observed, actions
        targets = self.geometry.absolute(actions)
        if self.action == "tcp_delta":
            reference = self.geometry.absolute(state if anchor is None else anchor)
            targets = self.relative.encode(reference, targets)
        return observed, targets

    def decode(self, anchor, actions):
        if self.action == "recorded":
            return actions
        if self.action == "tcp_delta":
            actions = self.relative.decode(self.geometry.absolute(anchor), actions)
        return self.geometry.restore_absolute(actions, anchor)

    def features(self, state_feature, state_units, action_feature):
        raw_state = (
            {**deepcopy(state_feature), "units": list(state_units)} if state_feature else None
        )
        units = dict(zip(raw_state["names"], state_units, strict=True)) if raw_state else {}
        raw_action = {
            **deepcopy(action_feature),
            "units": [
                units.get(name, "recorded (unit unverified)") for name in action_feature["names"]
            ],
        }
        absolute = self.geometry.feature() if self.geometry else None
        result = {
            "action": (
                self.relative.feature(absolute)
                if self.action == "tcp_delta"
                else absolute
                if self.action == "tcp_absolute"
                else raw_action
            ),
        }
        if raw_state is not None:
            result["observation.state"] = (
                self._state_layout(state_feature, state_units) if self.state == "tcp" else raw_state
            )
        return result

    def validate_source(self, info, state_feature):
        if self.geometry:
            names = info["features"].get("observation.state", {}).get("names", [])
            if (
                "source_state_names" in self.configuration
                and names != self.configuration["source_state_names"]
            ):
                raise ValueError(
                    "Recorded observation context names differ from field-transform source"
                )
            if "selected_state_feature" in self.configuration and state_feature != self.selected:
                raise ValueError("Selected state fields differ from field-transform source")
            if any(name not in names for name in self.geometry.names):
                raise ValueError("TCP transforms require recorded joint observations as anchors")
            if self.state == "tcp" and state_feature is None:
                raise ValueError("TCP state representation requires selected numeric inputs")
            self.geometry.validate_source(info)

    def manifest(self, state_feature, state_units, action_feature):
        features = self.features(state_feature, state_units, action_feature)
        operations = []
        if self.geometry:
            operations.append(
                dict(
                    operation="forward_kinematics",
                    fields=(["observation.state"] if self.state == "tcp" else [])
                    + (["action"] if self.action != "recorded" else []),
                    descriptor=self.geometry.descriptor,
                )
            )
        if self.action == "tcp_delta":
            operations.append(
                dict(
                    operation="relative_actions",
                    field="action",
                    anchor="observation.state",
                    configuration=self.relative.configuration(),
                )
            )
        return dict(
            data_fields=deepcopy(self.configuration),
            statistics_features=features,
            field_transforms=operations,
        )
