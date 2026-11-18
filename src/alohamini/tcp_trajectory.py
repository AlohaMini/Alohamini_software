"""Timed absolute TCP references; querying a reference never solves IK."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.trajectory import _layout, _matrix, _number


@dataclass(frozen=True)
class TCPActionPlan:
    kind = "tcp"
    names: tuple[str, ...]
    actions: tuple[tuple[float, ...], ...]
    period_s: float
    descriptor: dict
    start_s: float = 0.0
    fixed_targets: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "names", _layout(self.names))
        descriptor = json.loads(json.dumps(self.descriptor, allow_nan=False))
        geometry = KinematicFields(descriptor)
        if tuple(geometry.names) != self.names:
            raise ValueError("TCP trajectory source names differ from physical action names")
        actions = _matrix(self.actions, geometry.width)
        period = _number(self.period_s, "period_s")
        if period <= 0 or len(actions) > 16384:
            raise ValueError("Invalid TCP trajectory period or length")
        fixed = {name: _number(value, "fixed target") for name, value in self.fixed_targets.items()}
        if not fixed.keys() <= set(self.names):
            raise ValueError("Unknown TCP fixed target")
        joint_fields = {c["field"] for arm, _, _, _ in geometry.arms for c in arm["coordinates"]}
        if joint_fields & fixed.keys():
            raise ValueError(
                "TCP tracking cannot override individual arm joints with fixed targets"
            )
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "period_s", period)
        object.__setattr__(self, "start_s", _number(self.start_s, "start_s"))
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "fixed_targets", fixed)
        object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(self, "geometry_signature", json.dumps(descriptor, sort_keys=True))

    @property
    def end_s(self):
        return self.start_s + len(self.actions) * self.period_s

    def reference_at(self, elapsed_s):
        """Linear translation and shortest-path SO(3) interpolation; hold grippers."""
        from scipy.spatial.transform import Rotation

        t = _number(elapsed_s, "elapsed_s")
        if not self.start_s <= t < self.end_s:
            raise ValueError("Requested reference lies outside the TCP trajectory")
        sample = max(0.0, (t - self.start_s) / self.period_s)
        index = min(math.floor(sample + 1e-9), len(self.actions) - 1)
        following = min(index + 1, len(self.actions) - 1)
        alpha = min(1.0, max(0.0, sample - index))
        first, second = np.asarray(self.actions[index]), np.asarray(self.actions[following])
        reference = (1 - alpha) * first + alpha * second
        geometry = self._geometry
        for offset, (_, _, grip, _) in zip(geometry.arm_offsets, geometry.arms, strict=True):
            rotation = slice(offset + 3, offset + geometry.arm_pose_width)
            r0, r1 = (
                geometry.rotation.decode(first[rotation]),
                geometry.rotation.decode(second[rotation]),
            )
            delta = Rotation.from_matrix(r0.T @ r1).as_rotvec()
            reference[rotation] = geometry.rotation.encode(
                r0 @ Rotation.from_rotvec(alpha * delta).as_matrix()
            )
            if grip is not None:
                reference[offset + geometry.arm_pose_width] = first[
                    offset + geometry.arm_pose_width
                ]
        return reference

    def motion_rates_at(self, elapsed_s):
        """Maximum segment translation/angular rates across the two arms."""
        from scipy.spatial.transform import Rotation

        sample = max(0.0, (elapsed_s - self.start_s) / self.period_s)
        index = min(math.floor(sample + 1e-9), len(self.actions) - 1)
        following = min(index + 1, len(self.actions) - 1)
        first, second = np.asarray(self.actions[index]), np.asarray(self.actions[following])
        linear, angular = 0.0, 0.0
        geometry = self._geometry
        for offset in geometry.arm_offsets:
            linear = max(
                linear,
                float(np.linalg.norm(second[offset : offset + 3] - first[offset : offset + 3]))
                / self.period_s,
            )
            rotation = slice(offset + 3, offset + geometry.arm_pose_width)
            r0, r1 = (
                geometry.rotation.decode(first[rotation]),
                geometry.rotation.decode(second[rotation]),
            )
            angular = max(
                angular,
                float(Rotation.from_matrix(r1 @ r0.T).magnitude()) / self.period_s,
            )
        return linear, angular

    def action_at(self, elapsed_s):
        raise ValueError("TCP references require feedback tracking before becoming joint commands")

    def to_dict(self):
        return dict(
            version=1,
            kind="tcp",
            names=list(self.names),
            actions=[list(row) for row in self.actions],
            period_s=self.period_s,
            start_s=self.start_s,
            descriptor=json.loads(json.dumps(self.descriptor)),
            fixed_targets=dict(self.fixed_targets),
        )
