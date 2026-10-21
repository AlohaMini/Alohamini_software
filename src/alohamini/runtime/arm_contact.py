# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from AlohaMini joint/gripper current limiters; SI interface and explicit calibration.
"""Advisory joint stall detection and gripper holds; no hardware access."""

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

from alohamini._validation import finite_number, identifier
from alohamini.calibration import EncoderCalibration
from alohamini.model import ActuatorSpec
from alohamini.runtime.current_protection import collision_current_limit_a

# Initial early-contact settings require tuning against gripper feedback traces.
# Keep the existing absolute-current fallback for slow contact and first samples.
_GRIPPER_CURRENT_RISE_A_S = 5.0
_GRIPPER_RISING_CURRENT_MIN_A = 0.3
_GRIPPER_HOLD_CURRENT_A = 0.5
_GRIPPER_MAX_SAMPLE_GAP_S = 0.1


@dataclass(frozen=True)
class JointContactCalibration:
    """Legacy joint contact calibration retained for installed configurations.

    Joint stalls are advisory; release_margin_rad no longer gates joint targets.
    """

    release_margin_rad: float

    def __post_init__(self) -> None:
        finite_number(self.release_margin_rad, "release_margin_rad")
        if self.release_margin_rad <= 0:
            raise ValueError("Release margin must be positive")


@dataclass(frozen=True)
class GripperContactCalibration:
    """Physical endpoints corresponding to legacy 0% closed and 100% open."""

    closed_position_rad: float
    open_position_rad: float

    def __post_init__(self) -> None:
        finite_number(self.closed_position_rad, "closed_position_rad")
        finite_number(self.open_position_rad, "open_position_rad")
        finite_number(self.open_position_rad - self.closed_position_rad, "gripper stroke")
        if self.closed_position_rad == self.open_position_rad:
            raise ValueError("Gripper endpoints must differ")


@dataclass(frozen=True)
class GripperHoldSettings:
    """Initial grip tuning; output fractions are not current or torque setpoints."""

    preload_fraction: float = 0.03
    target_current_a: float = 0.6
    current_deadband_a: float = 0.05
    initial_output_fraction: float = 0.25
    minimum_output_fraction: float = 0.05
    maximum_output_fraction: float = 0.35
    output_gain_per_a_s: float = 0.5
    output_slew_per_s: float = 0.5

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            finite_number(value, name)
        if not 0 <= self.preload_fraction <= 0.03:
            raise ValueError("Gripper preload must be within 3% of the calibrated stroke")
        if not 0 < self.current_deadband_a < self.target_current_a:
            raise ValueError("Gripper current target must exceed its positive deadband")
        if not (
            0
            <= self.minimum_output_fraction
            <= self.initial_output_fraction
            <= self.maximum_output_fraction
            <= 1
        ):
            raise ValueError("Gripper output fractions must be ordered within [0, 1]")
        if self.output_gain_per_a_s <= 0 or self.output_slew_per_s <= 0:
            raise ValueError("Gripper output adjustment rates must be positive")


DEFAULT_GRIPPER_HOLD = GripperHoldSettings()


@dataclass(frozen=True)
class ArmJointSpec:
    actuator: ActuatorSpec
    calibration: EncoderCalibration
    contact: JointContactCalibration | GripperContactCalibration

    def __post_init__(self) -> None:
        if not isinstance(self.actuator, ActuatorSpec):
            raise TypeError("Expected ActuatorSpec")
        identifier(self.actuator.name, "joint name")
        collision_current_limit_a(self.actuator.motor_model)
        if not isinstance(self.calibration, EncoderCalibration):
            raise TypeError("Expected EncoderCalibration")
        if self.calibration.position_min_rad is None:
            raise ValueError("Arm commands require explicit device joint limits")
        if isinstance(self.contact, GripperContactCalibration):
            for position in (self.contact.closed_position_rad, self.contact.open_position_rad):
                self.calibration.validate_position(position)
        elif not isinstance(self.contact, JointContactCalibration):
            raise TypeError("Expected a joint or gripper contact calibration")


@dataclass(frozen=True)
class _StallCandidate:
    started_s: float
    position_rad: float
    direction: float


@dataclass(frozen=True)
class _GripperCurrentSample:
    received_s: float
    current_a: float
    closing: bool


class ArmContactGuard:
    """Report joint stalls without changing targets; hold grippers on contact.

    Gripper holds survive current falling and owner changes until a retreat.
    Missing feedback raises. Sustained overcurrent is independently enforced
    by HostSupervisor before this limiter runs.
    """

    def __init__(
        self,
        joints: Mapping[str, ArmJointSpec],
        *,
        gripper_hold: GripperHoldSettings = DEFAULT_GRIPPER_HOLD,
    ) -> None:
        if not joints:
            raise ValueError("Provide at least one arm joint")
        for name, spec in joints.items():
            if not isinstance(spec, ArmJointSpec) or spec.actuator.name != name:
                raise ValueError("Joint keys must match their actuator specification")
        self._joints = dict(joints)
        if not isinstance(gripper_hold, GripperHoldSettings):
            raise TypeError("Expected GripperHoldSettings")
        self.gripper_hold = gripper_hold
        self._candidates: dict[str, _StallCandidate] = {}
        self._gripper_samples: dict[str, _GripperCurrentSample] = {}
        self._gripper_outputs: dict[str, float] = {}
        self._gripper_output_requests: dict[str, float] = {}
        self._restore_outputs: set[str] = set()
        self._opening_retreats: dict[str, float] = {}
        self._active_grips: set[str] = set()
        self._holds: dict[str, float] = {}
        self._release_directions: dict[str, float] = {}
        self._last_time: float | None = None
        self._hold_events = 0
        self._joint_stall_currents_a: dict[str, float] = {}
        self._last_warning_s: dict[str, float] = {}

    @property
    def holds(self) -> dict[str, float]:
        return dict(self._holds)

    @property
    def hold_events(self) -> int:
        return self._hold_events

    @property
    def joint_hold_events(self) -> int:
        """Compatibility counter: advisory stalls are not joint hold events."""
        return 0

    @property
    def joint_stall_currents_a(self) -> dict[str, float]:
        """Present advisory stalls and their measured current magnitudes, in A."""
        return dict(self._joint_stall_currents_a)

    @property
    def output_limits(self) -> dict[str, float]:
        """Requested output ceilings for grippers supervised in this cycle."""
        return {name: self._gripper_output_requests[name] for name in self._gripper_samples}

    @property
    def gripper_status(self) -> dict[str, dict]:
        return {
            name: {
                "phase": "holding"
                if name in self._active_grips
                else ("stopped" if name in self._holds else "moving"),
                "current_a": sample.current_a,
                "target_current_a": self.gripper_hold.target_current_a
                if name in self._active_grips
                else None,
                "output_limit_fraction": self._gripper_outputs[name],
            }
            for name, sample in self._gripper_samples.items()
        }

    def confirm_output_limits(self, applied: Mapping[str, float]) -> None:
        """Track verified device ceilings, including lower installed limits."""
        requested = self.output_limits
        if not applied.keys() <= requested.keys():
            raise ValueError("Unexpected gripper output confirmation")
        for name, value in applied.items():
            finite_number(value, "confirmed output fraction")
            if not 0 <= value <= requested[name] + 1e-12:
                raise ValueError("Device output ceiling exceeds the requested limit")
        for name, value in applied.items():
            self._gripper_outputs[name] = value
            # Preserve sub-register increments. Only a lower installed ceiling
            # should clamp the integrator, not normal per-mille quantization.
            quantized = int(requested[name] * 1000) / 1000
            if value < quantized - 1e-12:
                self._gripper_output_requests[name] = value

    def stop_holds(self, positions_rad: Mapping[str, float]) -> None:
        """Match watchdog's measured-position hold; never resume old preload."""
        for name in self._holds.keys() & positions_rad.keys():
            self._joints[name].calibration.validate_position(positions_rad[name])
            self._holds[name] = positions_rad[name]
        self.cancel_candidates()

    def cancel_candidates(self) -> None:
        """Cancel pending timers on stop without silently releasing contact holds."""
        self._candidates.clear()
        self._gripper_samples.clear()
        self._active_grips.clear()
        self._restore_outputs.clear()
        self._opening_retreats.clear()
        self._gripper_output_requests.update(self._gripper_outputs)
        self._joint_stall_currents_a.clear()

    def limit(
        self,
        goals_rad: Mapping[str, float],
        positions_rad: Mapping[str, float],
        currents_a: Mapping[str, float],
        *,
        now: float,
        feedback_times_s: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        try:
            finite_number(now, "contact sample time")
            if now < 0 or (self._last_time is not None and now <= self._last_time):
                raise ValueError("Contact sample times must increase")
            if not goals_rad.keys() <= self._joints.keys():
                raise ValueError("Unknown arm target")
            for name, goal in goals_rad.items():
                self._joints[name].calibration.validate_position(goal)
                finite_number(positions_rad[name], "joint position")
                finite_number(currents_a[name], "joint current")
                if isinstance(self._joints[name].contact, GripperContactCalibration):
                    received = now if feedback_times_s is None else feedback_times_s[name]
                    finite_number(received, "gripper feedback time")
                    previous = self._gripper_samples.get(name)
                    if not 0 <= received <= now or (
                        previous is not None and received <= previous.received_s
                    ):
                        raise ValueError("Gripper feedback times must increase and not be future")
        except (KeyError, ValueError, TypeError):
            self.cancel_candidates()
            raise
        self._last_time = now
        self._candidates = {k: v for k, v in self._candidates.items() if k in goals_rad}
        self._gripper_samples = {k: v for k, v in self._gripper_samples.items() if k in goals_rad}
        self._joint_stall_currents_a.clear()
        limited = dict(goals_rad)
        for name, goal in goals_rad.items():
            spec = self._joints[name]
            present, current = positions_rad[name], abs(currents_a[name])
            contact = spec.contact
            if isinstance(contact, GripperContactCalibration):
                span = contact.open_position_rad - contact.closed_position_rad
                open_direction = 1.0 if span > 0 else -1.0
                release_margin = 0.01 * abs(span)
                delta = goal - present
                closing = delta * open_direction < 0
                received = now if feedback_times_s is None else feedback_times_s[name]
                previous = self._gripper_samples.get(name)
                self._gripper_samples[name] = _GripperCurrentSample(received, current, closing)
                self._gripper_outputs.setdefault(name, self.gripper_hold.initial_output_fraction)
                self._gripper_output_requests.setdefault(
                    name, self.gripper_hold.initial_output_fraction
                )
                if name in self._restore_outputs:
                    self._gripper_output_requests[name] = self.gripper_hold.initial_output_fraction
                    self._restore_outputs.remove(name)
                if name in self._holds and (
                    (goal - self._holds[name]) * self._release_directions[name] >= release_margin
                ):
                    if self._release_directions[name] == open_direction:
                        self._opening_retreats[name] = goal
                    del self._holds[name]
                    del self._release_directions[name]
                    self._active_grips.discard(name)
                    # Send the opening target under the previous verified cap.
                    # Restore the moving cap only after that target was sent.
                    self._gripper_output_requests[name] = self._gripper_outputs[name]
                    self._restore_outputs.add(name)
                    continue
                if name in self._opening_retreats:
                    retreat_goal = self._opening_retreats[name]
                    if (goal - retreat_goal) * open_direction > -release_margin:
                        if (goal - retreat_goal) * open_direction > 0:
                            self._opening_retreats[name] = goal
                        # Residual contact current must not latch a new hold
                        # while the operator is explicitly opening the gripper.
                        continue
                    del self._opening_retreats[name]
                if name not in self._holds:
                    rising_contact = False
                    # Require consecutive closing requests, without a fixed arming delay.
                    # Host acquisition times approximate sample intervals; servo sampling
                    # timestamps are unavailable. Never bridge a feedback interruption.
                    if closing and previous is not None and previous.closing:
                        elapsed = received - previous.received_s
                        rising_contact = (
                            elapsed <= _GRIPPER_MAX_SAMPLE_GAP_S
                            and current >= _GRIPPER_RISING_CURRENT_MIN_A
                            and (current - previous.current_a) / elapsed
                            >= _GRIPPER_CURRENT_RISE_A_S
                        )
                    if current < _GRIPPER_HOLD_CURRENT_A and not rising_contact:
                        continue
                    release = -math.copysign(1.0, delta) if delta else open_direction
                    spec.calibration.validate_position(present)
                    hold = present
                    if closing:
                        # One bounded inward preload, never accumulated each tick.
                        remaining = max(
                            0.0, (present - contact.closed_position_rad) * open_direction
                        )
                        preload = min(
                            self.gripper_hold.preload_fraction * abs(span), abs(delta), remaining
                        )
                        hold -= open_direction * preload
                        self._active_grips.add(name)
                    self._holds[name] = hold
                    self._release_directions[name] = release
                    self._hold_events += 1
                elif name in self._active_grips and previous is not None:
                    elapsed = received - previous.received_s
                    if elapsed <= _GRIPPER_MAX_SAMPLE_GAP_S:
                        settings = self.gripper_hold
                        error = current - settings.target_current_a
                        if abs(error) > settings.current_deadband_a:
                            error -= math.copysign(settings.current_deadband_a, error)
                            adjustment = -settings.output_gain_per_a_s * error * elapsed
                            maximum_step = settings.output_slew_per_s * elapsed
                            adjustment = min(maximum_step, max(-maximum_step, adjustment))
                            self._gripper_output_requests[name] = min(
                                settings.maximum_output_fraction,
                                max(
                                    settings.minimum_output_fraction,
                                    self._gripper_output_requests[name] + adjustment,
                                ),
                            )
            else:
                # Preserve the source's encoder-space trigger, but never latch a joint target.
                calibration = spec.calibration
                ratio = calibration.joint_per_encoder_ratio * (
                    (calibration.ticks_per_revolution - 1) / calibration.ticks_per_revolution
                )
                error = (goal - present) / ratio
                if current < collision_current_limit_a(spec.actuator.motor_model) or abs(
                    error
                ) < math.radians(2):
                    self._candidates.pop(name, None)
                    continue
                direction = math.copysign(1.0, error)
                candidate = self._candidates.get(name)
                if candidate is None or candidate.direction != direction:
                    self._candidates[name] = _StallCandidate(now, present, direction)
                    continue
                progress = (present - candidate.position_rad) * direction / ratio
                if progress >= math.radians(0.2):
                    self._candidates[name] = _StallCandidate(now, present, direction)
                    continue
                if now - candidate.started_s >= 0.150:
                    self._joint_stall_currents_a[name] = current
                    if now - self._last_warning_s.get(name, -math.inf) >= 5.0:
                        logging.getLogger(__name__).warning(
                            "Joint stall suspected: %s, current=%.3f A; "
                            "target unchanged, overload protection remains active",
                            name,
                            current,
                        )
                        self._last_warning_s[name] = now
                continue
            limited[name] = self._holds[name]
        return limited
