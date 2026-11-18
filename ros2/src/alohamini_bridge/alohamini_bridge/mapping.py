# SPDX-License-Identifier: Apache-2.0
# Adapted from alohamini_lerobot_bridge/protocol.py.
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from typing import Any

from alohamini._validation import finite_number as validate_number
from alohamini.calibration.encoder import (
    EncoderCalibration,
    HostPositionUnits,
    JointPositionDecoder,
)
from alohamini.calibration.lift import LiftCalibration
from alohamini.calibration.reference import reference_coordinates
from alohamini.model.joint_spec import ModelJointSpecification

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
    "gripper",
)


def finite_number(value: Any, field: str) -> float:
    validate_number(value, field)
    return float(value)


class JointMapper:
    """Convert between Host motor normalization and authoritative URDF space."""

    def __init__(self, calibration=None, lift_calibration=None) -> None:
        self._host_binding = None
        self._automatic = calibration is None and lift_calibration is None
        self.decoder = None
        if self._automatic:
            return
        self._configure(calibration, lift_calibration)

    def _configure(self, calibration, lift_calibration):
        if set(calibration) != {"left", "right"}:
            raise ValueError("Provide independent left and right arm mappings")
        self.period = 4096
        self.joints_by_side = {side: calibration[side]["joints"] for side in ("left", "right")}
        encoders = {}
        for side in ("left", "right"):
            if calibration[side]["ticks_per_revolution"] != self.period:
                raise ValueError("STS mappings require 4096 ticks per revolution")
            if set(self.joints_by_side[side]) != set(ARM_JOINTS):
                raise ValueError(f"Incomplete arm mapping: {side}")
            for joint in ARM_JOINTS:
                entry = self._entry(side, joint)
                name = f"{side}_{'wrist_yaw_joint' if joint == 'wrist_yaw' else joint}"
                encoders[name] = EncoderCalibration(
                    self.period,
                    entry["reference_tick"],
                    entry["reference_q_rad"],
                    entry["sign"],
                    entry.get("joint_per_encoder_ratio", 1.0),
                    entry.get("safe_q_min_rad"),
                    entry.get("safe_q_max_rad"),
                )
        self.decoder = JointPositionDecoder(encoders)
        self.encoders = encoders
        mechanism, urdf = lift_calibration["mechanism"], lift_calibration["urdf"]
        self.lift = LiftCalibration(
            mechanism["physical_min_mm"] / 1000,
            mechanism["physical_max_mm"] / 1000,
            urdf["q_at_physical_min_m"],
            urdf["q_at_physical_max_m"],
        )

    def reset(self) -> None:
        if self.decoder is not None:
            self.decoder.reset()

    def _entry(self, side: str, joint: str) -> dict[str, Any]:
        try:
            return self.joints_by_side[side][joint]
        except KeyError as error:
            raise ValueError(f"missing calibration for {side}_{joint}") from error

    def lift_height_to_urdf(self, height_mm: float) -> float:
        return self.lift.height_to_position(finite_number(height_mm, "lift height") / 1000)

    def lift_urdf_to_height(self, position_m: float) -> float:
        return self.lift.position_to_height(position_m) * 1000

    def host_to_tick(self, value: float, metadata: dict[str, Any]) -> int:
        return HostPositionUnits(
            **{
                key: metadata[key]
                for key in ("normalization", "drive_mode", "range_min", "range_max")
            }
        ).to_tick(value)

    def tick_to_host(self, tick: int, metadata: dict[str, Any]) -> float:
        return HostPositionUnits(
            **{
                key: metadata[key]
                for key in ("normalization", "drive_mode", "range_min", "range_max")
            }
        ).from_tick(tick)

    def validate_host(self, metadata):
        """Use this Host's verified Home binding, pinned for the process lifetime."""
        if self._automatic:
            binding = {
                key: metadata.get(key)
                for key in (
                    "schema_version",
                    "robot_model",
                    "motors",
                    "lift_axis",
                    "reference_calibration",
                )
            }
            if self._host_binding is None:
                reference_coordinates(metadata)
                spec = ModelJointSpecification.from_model(metadata["robot_model"])
                lift = spec.joints["vertical_move"]
                # Host height is measured from lift Home in millimetres. It
                # shares the model's SI scale; soft limits never rescale it.
                travel_mm = (lift.upper - lift.lower) * 1000
                limits = metadata["lift_axis"]
                if not 0 <= limits["soft_min_mm"] < limits["soft_max_mm"] <= travel_mm + 1e-6:
                    raise ValueError("Host lift limits exceed the model's lift travel")
                self._configure(
                    metadata["reference_calibration"]["arms"],
                    dict(
                        mechanism=dict(physical_min_mm=0.0, physical_max_mm=travel_mm),
                        urdf=dict(q_at_physical_min_m=lift.lower, q_at_physical_max_m=lift.upper),
                    ),
                )
                self._host_binding = deepcopy(binding)
            elif binding != self._host_binding:
                raise ValueError("Host calibration changed; restart the bridge")
        for side in ("left", "right"):
            for joint, entry in self.joints_by_side[side].items():
                captured = entry.get("motor_calibration")
                if captured is None:
                    continue
                name = f"arm_{side}_{joint}"
                motor = metadata["motors"][name]
                if any(entry[key] != motor[key] for key in ("id", "model")) or any(
                    captured[key] != motor[key]
                    for key in ("homing_offset", "drive_mode", "range_min", "range_max")
                ):
                    raise ValueError(f"Arm mapping differs from installed Host calibration: {name}")

    def urdf_to_host(self, name: str, position: float, metadata: dict) -> tuple[str, float]:
        self.validate_host(metadata)
        side, _, suffix = name.partition("_")
        joint = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
        entry = self._entry(side, joint)
        encoder = self.encoders[name]
        if encoder.position_min_rad is None:
            # Preserve the original wrist-roll and gripper command bounds.
            limits = (
                (-math.pi, math.pi)
                if joint == "wrist_roll"
                else (entry["urdf_open_rad"], entry["urdf_closed_rad"])
            )
            encoder = replace(encoder, position_min_rad=min(limits), position_max_rad=max(limits))
        tick = encoder.position_to_tick(position)
        motor = f"arm_{side}_{joint}"
        units = metadata["motors"][motor]
        if not units["range_min"] <= tick <= units["range_max"]:
            raise ValueError(f"{name} target exceeds installed Host encoder range")
        return f"{motor}.pos", self.tick_to_host(tick, units)

    def observation_to_joint_positions(
        self, observation: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, float]:
        self.validate_host(metadata)
        feedback = observation.get("_motor_feedback")
        if feedback is not None and (
            not isinstance(feedback, dict)
            or type(feedback.get("version")) is not int
            or feedback["version"] != 1
        ):
            raise ValueError("Invalid raw motor feedback version")
        ticks = {}
        for side in ("left", "right"):
            for joint in ARM_JOINTS:
                motor_name = f"arm_{side}_{joint}"
                suffix = "wrist_yaw_joint" if joint == "wrist_yaw" else joint
                if feedback is None:
                    tick = self.host_to_tick(
                        observation[f"{motor_name}.pos"], metadata["motors"][motor_name]
                    )
                else:
                    measured = feedback["motors"][motor_name]
                    if not isinstance(measured, dict) or not isinstance(
                        measured.get("field_errors", {}), dict
                    ):
                        raise ValueError(f"Invalid raw encoder feedback: {motor_name}")
                    tick = measured.get("position_raw")
                    if (
                        type(tick) is not int
                        or not 0 <= tick < self.period
                        or measured.get("packet_error")
                        or measured.get("field_errors", {}).get("position_raw")
                    ):
                        raise ValueError(f"Invalid raw encoder feedback: {motor_name}")
                ticks[f"{side}_{suffix}"] = tick
        height = finite_number(observation["lift_axis.height_mm"], "lift_axis.height_mm")
        lift = self.lift_height_to_urdf(height)
        positions = self.decoder.decode(ticks)
        positions["vertical_move"] = lift
        return positions
