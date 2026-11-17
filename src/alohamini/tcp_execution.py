"""Feedback tracking and bounded recovery for local and remote TCP references."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass

import numpy as np

from alohamini.datasets.kinematic_fields import KinematicFields
from alohamini.fixed import VELOCITIES
from alohamini.kinematics.arm import _rotation_vector

logger = logging.getLogger(__name__)


def warmup_backend():
    """Load CPU bindings before entering the control loop, never on its first tick."""
    try:
        import pink  # noqa: F401
        import pinocchio  # noqa: F401
        import qpsolvers
    except ImportError as exc:
        raise ImportError(
            "TCP execution requires the kinematics extra on the execution machine"
        ) from exc
    if "daqp" not in qpsolvers.available_solvers:
        raise ImportError("TCP execution requires the DAQP QP backend")


@dataclass(frozen=True)
class TCPTrackingConfig:
    max_joint_velocity: float = 1.5
    max_joint_acceleration: float = 8.0
    max_joint_step: float = 0.08
    joint_limit_margin: float = 0.03
    max_tcp_linear_velocity: float = 0.2
    max_tcp_linear_acceleration: float = 0.5
    max_tcp_angular_velocity: float = 1.0
    max_tcp_angular_acceleration: float = 2.0
    max_target_lead_m: float = 0.025
    max_target_lead_rad: float = 0.15
    max_position_error_m: float = 0.05
    max_orientation_error_rad: float = 0.5
    max_joint_tracking_error_rad: float = 0.2
    error_timeout_s: float = 0.5
    recovery_timeout_s: float = 1.0
    max_failed_plans: int = 3

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "max_failed_plans":
                if type(value) is not int or value < 1:
                    raise ValueError("TCP max_failed_plans must be a positive integer")
            elif (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and name != "joint_limit_margin")
            ):
                raise ValueError(f"TCP {name} must be finite and positive")


class TCPTracker:
    """One controller per execution session, independent of model inference threads."""

    def __init__(self, *, fps, settings=None):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("TCP control FPS must be positive")
        self.fps = fps
        warmup_backend()
        self.settings = TCPTrackingConfig(**(settings or {}))
        self.geometry = self.signature = None
        self.last_time = self.bad_since = self.recovery_started = None
        self.failed_plan = None
        self.failed_plans = self.recoveries = 0
        self.needs_replan = False
        self.status = "waiting"
        self.last_command = None
        self.command_joints = None
        self.reference_governor = None
        self.stopped_reason = None

    def completed(self, plan):
        governor = self.reference_governor
        return governor is not None and governor.plan is plan and governor.elapsed_s >= plan.end_s

    @property
    def stationary(self):
        if self.geometry is None:
            return True
        velocities = [
            model.pink_solver().previous_velocity for _, _, _, model in self.geometry.arms
        ]
        if self.reference_governor is not None:
            velocities += [
                self.reference_governor.linear_velocity,
                self.reference_governor.angular_velocity,
            ]
        return all(np.max(np.abs(velocity)) <= 1e-8 for velocity in velocities)

    def _bind(self, plan, snapshot):
        signature = plan.geometry_signature
        if self.signature is not None and signature != self.signature:
            raise ValueError("TCP geometry/calibration changed during execution")
        if self.geometry is None:
            self.geometry = KinematicFields(plan.descriptor)
            self.geometry.validate_source(
                {
                    "features": {"action": {"names": list(plan.names)}},
                    "robot_metadata": snapshot.payload.get("_robot_metadata"),
                }
            )
            # Check dependencies before any physical command is emitted.
            for _, _, _, model in self.geometry.arms:
                model.pink_solver()
            self.signature = signature
        return np.asarray([snapshot.payload[name] for name in plan.names], dtype=float)

    def _hold(self, source, previous, dt):
        """Brake arm commands; keep grippers/position axes and zero velocity axes."""
        output = source.copy()
        if self.last_command is not None:
            for _, _, grip, _ in self.geometry.arms:
                if grip is not None:
                    output[grip] = self.last_command[grip]
        config = self.settings
        command_joints = []
        for (_, indices, _, model), q, velocity in zip(
            self.geometry.arms, self.command_joints, previous, strict=True
        ):
            velocity = np.sign(velocity) * np.maximum(
                0, np.abs(velocity) - config.max_joint_acceleration * dt
            )
            candidate = q + velocity * dt
            if model.joint_limits is not None:
                limits = np.asarray(model.joint_limits)
                if ((candidate < limits[:, 0]) | (candidate > limits[:, 1])).any():
                    raise RuntimeError("TCP recovery cannot brake within declared joint limits")
            for i, value, coordinate in zip(
                indices, candidate, self._coordinates(model), strict=True
            ):
                output[i] = self.geometry.source_joint(value, coordinate)
            model.pink_solver().previous_velocity = velocity.copy()
            command_joints.append(candidate)
        for i, unit in zip(
            self.geometry.passthrough, self.geometry.units[self.geometry.pose_width :], strict=True
        ):
            if self.geometry.names[i] in VELOCITIES or unit.endswith("/s"):
                output[i] = 0.0
        self.last_command = output.copy()
        self.command_joints = command_joints
        if self.reference_governor is not None:
            self.reference_governor.brake(dt)
        return output

    def _coordinates(self, model):
        return next(arm["coordinates"] for arm, _, _, m in self.geometry.arms if m is model)

    def _recover(self, plan, source, previous, dt, now, reason):
        if self.recovery_started is None:
            self.recovery_started = now
            self.recoveries += 1
            logger.warning("TCP execution paused; requesting a fresh plan: %s", reason)
        if plan is not self.failed_plan:
            self.failed_plans += 1
        self.failed_plan = plan
        self.needs_replan, self.status = True, "holding"
        if (
            now - self.recovery_started >= self.settings.recovery_timeout_s
            or self.failed_plans >= self.settings.max_failed_plans
        ):
            self.stopped_reason = reason
            raise RuntimeError(f"TCP recovery exhausted; control stopped: {reason}")
        return self._hold(source, previous, dt)

    def step(self, plan, snapshot, elapsed_s, now, *, expired=False, holding=False):
        if self.stopped_reason is not None:
            raise RuntimeError(f"TCP controller stopped: {self.stopped_reason}")
        source = self._bind(plan, snapshot)
        if not np.isfinite(source).all() or not math.isfinite(now):
            raise ValueError("TCP tracking requires finite measured feedback and time")
        dt = 1 / self.fps if self.last_time is None else now - self.last_time
        if dt <= 0:
            raise ValueError("TCP control time must increase")
        self.last_time = now
        measured = [
            np.array(
                [
                    self.geometry.joint(source[i], c)
                    for i, c in zip(indices, arm["coordinates"], strict=True)
                ]
            )
            for arm, indices, _, _ in self.geometry.arms
        ]
        if self.command_joints is None:
            self.command_joints = [q.copy() for q in measured]
        if self.reference_governor is None:
            from alohamini.tcp_reference import TCPReferenceGovernor

            self.reference_governor = TCPReferenceGovernor(
                self.geometry, self.settings, self.geometry.absolute(source)
            )
        previous = [
            model.pink_solver().previous_velocity.copy() for _, _, _, model in self.geometry.arms
        ]
        if dt > 0.25:
            self.stopped_reason = "control tick gap exceeded 0.25 s"
            raise RuntimeError("TCP control tick gap exceeded 0.25 s; control stopped")
        if holding and plan is not self.failed_plan:
            # Between synchronous predictions there is no old trajectory to
            # advance. Brake at control cadence without spending the IK-failure
            # budget; the executor separately bounds prediction wait time.
            if any(
                np.max(np.abs(q - actual)) > self.settings.max_joint_tracking_error_rad
                for q, actual in zip(self.command_joints, measured, strict=True)
            ):
                self.stopped_reason = "Measured joints diverged during TCP hold"
                raise RuntimeError(self.stopped_reason)
            self.status, self.needs_replan = "waiting", False
            return tuple(map(float, self._hold(source, previous, dt)))
        if plan is self.failed_plan:
            return tuple(
                map(
                    float,
                    self._recover(plan, source, previous, dt, now, "waiting for a new prediction"),
                )
            )
        new_plan = plan is not self.reference_governor.plan
        if expired or (new_plan and not plan.start_s <= elapsed_s < plan.end_s):
            return tuple(
                map(float, self._recover(plan, source, previous, dt, now, "reference expired"))
            )
        output, worst_position, worst_orientation = source.copy(), 0.0, 0.0
        phase = "reference"
        command_joints = []
        try:
            actual_poses = [
                model.forward(observed)
                for (_, _, _, model), observed in zip(self.geometry.arms, measured, strict=True)
            ]
            proposal = self.reference_governor.propose(plan, elapsed_s, dt, actual_poses)
            reference = proposal.reference
            for offset, (arm, indices, grip_index, model), q, observed, actual in zip(
                self.geometry.arm_offsets,
                self.geometry.arms,
                self.command_joints,
                measured,
                actual_poses,
                strict=True,
            ):
                phase = arm["name"]
                if np.max(np.abs(q - observed)) > self.settings.max_joint_tracking_error_rad:
                    raise ValueError("Measured joints are not following the command reference")
                target = np.eye(4)
                target[:3, 3] = reference[offset : offset + 3]
                target[:3, :3] = self.geometry.rotation.decode(
                    reference[offset + 3 : offset + self.geometry.arm_pose_width]
                )
                reach = sum(math.hypot(row[0], row[2]) for row in model.rows)
                reach += float(np.linalg.norm(model.tool_transform[:3, 3]))
                raw_position = proposal.goal[offset : offset + 3]
                if np.linalg.norm(raw_position - model.base_transform[:3, 3]) > reach + 1e-6:
                    raise ValueError("TCP target lies outside the arm's reach bound")
                worst_position = max(
                    worst_position, float(np.linalg.norm(target[:3, 3] - actual[:3, 3]))
                )
                worst_orientation = max(
                    worst_orientation,
                    float(np.linalg.norm(_rotation_vector(target[:3, :3] @ actual[:3, :3].T))),
                )
                candidate, _ = model.pink_solver().step(
                    q,
                    target,
                    dt,
                    **{
                        name: getattr(self.settings, name)
                        for name in (
                            "max_joint_velocity",
                            "max_joint_acceleration",
                            "max_joint_step",
                            "joint_limit_margin",
                        )
                    },
                )
                for i, value, coordinate in zip(
                    indices, candidate, arm["coordinates"], strict=True
                ):
                    output[i] = self.geometry.source_joint(value, coordinate)
                command_joints.append(candidate)
                if grip_index is not None:
                    value = self.geometry.gripper(
                        reference[offset + self.geometry.arm_pose_width],
                        arm["gripper"],
                        inverse=True,
                    )
                    bounds = sorted(arm["gripper"][key] for key in ("source_closed", "source_open"))
                    tolerance = 1e-6 * max(1, bounds[1] - bounds[0])
                    if not bounds[0] - tolerance <= value <= bounds[1] + tolerance:
                        raise ValueError("TCP gripper target exceeds declared range")
                    output[grip_index] = np.clip(value, *bounds)
            for n, i in enumerate(self.geometry.passthrough):
                output[i] = reference[self.geometry.pose_width + n]
            for name, value in plan.fixed_targets.items():
                output[plan.names.index(name)] = value
            phase = "tracking"
            bad = (
                worst_position > self.settings.max_position_error_m
                or worst_orientation > self.settings.max_orientation_error_rad
            )
            self.needs_replan = bad or proposal.needs_replan
            if bad:
                self.bad_since = now if self.bad_since is None else self.bad_since
                if now - self.bad_since >= self.settings.error_timeout_s:
                    raise ValueError(
                        f"persistent TCP tracking error: {worst_position:.4f} m, "
                        f"{worst_orientation:.4f} rad"
                    )
            else:
                self.bad_since = None
            if self.recovery_started is not None and not bad:
                logger.info("TCP execution resumed with a fresh prediction")
                self.recovery_started, self.failed_plan = None, None
                self.failed_plans = 0
            self.status = "replanning" if self.needs_replan else "tracking"
            self.reference_governor.commit(proposal)
            self.last_command = output.copy()
            self.command_joints = command_joints
            return tuple(map(float, output))
        except (ValueError, RuntimeError) as exc:
            # A successful first arm is not committed if the second arm fails.
            for (_, _, _, model), velocity in zip(self.geometry.arms, previous, strict=True):
                model.pink_solver().previous_velocity = velocity.copy()
            return tuple(
                map(float, self._recover(plan, source, previous, dt, now, f"{phase}: {exc}"))
            )
