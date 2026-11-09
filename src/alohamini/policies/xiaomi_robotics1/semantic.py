"""Separate joint-state and local TCP-action slots for Xiaomi's 60D interface."""

import numpy as np

from alohamini.datasets.kinematic_fields import KinematicFields


class XiaomiFields:
    def __init__(self, descriptor):
        self.geometry = KinematicFields(descriptor, rotation_representation="rotvec")
        if {arm[0]["name"] for arm in self.geometry.arms} != {"left", "right"}:
            raise ValueError("Xiaomi requires explicitly named left/right calibrated arms")
        self.action_slots, self.action_indices, self.grippers = [], [], []
        self.states = []
        for start, (arm, indices, gi, _) in zip(
            self.geometry.arm_offsets, self.geometry.arms, strict=True
        ):
            if len(indices) > 7 or gi is None:
                raise ValueError("Xiaomi arms require <=7 joints and a calibrated gripper")
            side = 0 if arm["name"] == "left" else 8
            self.action_indices.extend(range(start, start + 7))
            self.action_slots.extend(range(side, side + 7))
            self.grippers.append((start + 6, side + 6, gi, arm["gripper"]))
            self.states.append((side, indices, gi, arm))
        for name, slot, unit in (
            ("x.vel", 17, "m/s"),
            ("y.vel", 18, "m/s"),
            ("theta.vel", 19, "deg/s"),
        ):
            if name in self.geometry.feature_names:
                index = self.geometry.feature_names.index(name)
                if self.geometry.units[index] != unit:
                    raise ValueError(f"Xiaomi platform base mapping expects {name} in {unit}")
                self.action_indices.append(index)
                self.action_slots.append(slot)
        self.held = [i for i in range(self.geometry.width) if i not in self.action_indices]
        if any(
            self.geometry.feature_names[i] not in ("lift_axis.height_mm", "lift.height")
            for i in self.held
        ):
            raise ValueError("Xiaomi has undeclared auxiliary coordinates")

    def state(self, raw):
        raw = np.asarray(raw)
        result = np.zeros((*raw.shape[:-1], 60), dtype=np.float32)
        for slot, indices, gi, arm in self.states:
            for j, (index, coordinate) in enumerate(zip(indices, arm["coordinates"], strict=True)):
                values = raw[..., index]
                result[..., slot + j] = np.array(
                    [self.geometry.joint(x, coordinate) for x in values.reshape(-1)]
                ).reshape(values.shape)
            result[..., slot + 7] = self.geometry.gripper(raw[..., gi], arm["gripper"])
        return result

    def action(self, raw, tcp_delta):
        raw, values = np.asarray(raw), np.array(tcp_delta, copy=True)
        for index, _, gi, grip in self.grippers:
            values[..., index] -= self.geometry.gripper(raw[..., None, gi], grip)
        result = np.zeros((*values.shape[:-1], 60), dtype=np.float32)
        result[..., self.action_slots] = values[..., self.action_indices]
        # Platform extension declares SI yaw velocity; source Host records degrees/s.
        result[..., 19] = np.deg2rad(result[..., 19])
        return result

    def decode(self, raw, packed):
        raw, packed = np.asarray(raw), np.array(packed, copy=True)
        packed[..., 19] = np.rad2deg(packed[..., 19])
        result = np.zeros((*packed.shape[:-1], self.geometry.width), dtype=np.float32)
        result[..., self.action_indices] = packed[..., self.action_slots]
        for index, _, gi, grip in self.grippers:
            result[..., index] += self.geometry.gripper(raw[..., None, gi], grip)
        if self.held:
            anchor = self.geometry.absolute(raw)
            result[..., self.held] = anchor[..., None, self.held]
        return result
