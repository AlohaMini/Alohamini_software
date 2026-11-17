"""Bounded Cartesian references and feedback-paced traversal of sampled TCP plans.

The sampled path keeps linear translation and shortest-path SO(3) interpolation,
as in Diffusion Policy's PoseTrajectoryInterpolator. A persistent reference
governor bounds linear/angular speed and acceleration across plan replacements.
These are reference limits, not a dynamics or collision feasibility guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def _velocity(error, previous, dt, speed, acceleration):
    distance = float(np.linalg.norm(error))
    # Leave one control period of braking distance before a stationary target.
    braking = math.sqrt((acceleration * dt) ** 2 + 2 * acceleration * distance)
    requested = min(speed, max(0.0, braking - acceleration * dt))
    desired = error * (requested / distance) if distance > 1e-12 else np.zeros(3)
    change = desired - previous
    length = float(np.linalg.norm(change))
    if length > acceleration * dt:
        change *= acceleration * dt / length
    # A convex combination of two speed-bounded velocities stays speed-bounded.
    return previous + change


@dataclass(frozen=True)
class ReferenceStep:
    plan: object
    elapsed_s: float
    reference: np.ndarray
    goal: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    rate: float
    needs_replan: bool


class TCPReferenceGovernor:
    def __init__(self, geometry, settings, initial_reference):
        self.geometry, self.settings = geometry, settings
        self.reference = np.array(initial_reference, dtype=float, copy=True)
        self.linear_velocity = np.zeros((len(geometry.arms), 3))
        # Angular velocities are expressed in each arm's fixed base frame.
        self.angular_velocity = np.zeros_like(self.linear_velocity)
        self.plan, self.elapsed_s, self.rate = None, 0.0, 0.0

    def _pose(self, reference, offset):
        position = reference[offset : offset + 3]
        rotation = self.geometry.rotation.decode(
            reference[offset + 3 : offset + self.geometry.arm_pose_width]
        )
        return position, rotation

    def _integrate(self, reference, linear, angular, dt):
        result = reference.copy()
        for n, offset in enumerate(self.geometry.arm_offsets):
            position, rotation = self._pose(self.reference, offset)
            result[offset : offset + 3] = position + linear[n] * dt
            result[offset + 3 : offset + self.geometry.arm_pose_width] = (
                self.geometry.rotation.encode(
                    Rotation.from_rotvec(angular[n] * dt).as_matrix() @ rotation
                )
            )
        return result

    def propose(self, plan, elapsed_s, dt, feedback):
        """Stage a step; the executor commits it only after both arms succeed."""
        config = self.settings
        changed = plan is not self.plan
        phase = max(plan.start_s, elapsed_s) if changed else self.elapsed_s
        if phase >= plan.end_s:
            raise ValueError("TCP reference exhausted")
        goal = plan.reference_at(phase)
        lag = 0.0
        for offset, actual in zip(self.geometry.arm_offsets, feedback, strict=True):
            position, rotation = self._pose(self.reference, offset)
            target_position, target_rotation = self._pose(goal, offset)
            # Pace against both the bounded reference and measured execution.
            for other_position, other_rotation in (
                (target_position, target_rotation),
                (actual[:3, 3], actual[:3, :3]),
            ):
                lag = max(
                    lag,
                    float(np.linalg.norm(other_position - position)) / config.max_target_lead_m,
                    float(Rotation.from_matrix(other_rotation @ rotation.T).magnitude())
                    / config.max_target_lead_rad,
                )
        linear_rate, angular_rate = plan.motion_rates_at(phase)
        rate = min(
            1.0,
            config.max_tcp_linear_velocity / max(linear_rate, 1e-12),
            config.max_tcp_angular_velocity / max(angular_rate, 1e-12),
        ) * max(0.0, 1.0 - lag)
        if not changed:
            # Visit sample boundaries in order, including gripper events.
            sample = math.floor((phase - plan.start_s) / plan.period_s + 1e-9)
            boundary = plan.start_s + (sample + 1) * plan.period_s
            phase = min(plan.end_s, boundary, phase + rate * dt)
            goal = plan.reference_at(min(phase, math.nextafter(plan.end_s, -math.inf)))
        linear, angular = np.empty_like(self.linear_velocity), np.empty_like(self.angular_velocity)
        for n, offset in enumerate(self.geometry.arm_offsets):
            position, rotation = self._pose(self.reference, offset)
            target_position, target_rotation = self._pose(goal, offset)
            linear[n] = _velocity(
                target_position - position,
                self.linear_velocity[n],
                dt,
                config.max_tcp_linear_velocity,
                config.max_tcp_linear_acceleration,
            )
            angular[n] = _velocity(
                Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec(),
                self.angular_velocity[n],
                dt,
                config.max_tcp_angular_velocity,
                config.max_tcp_angular_acceleration,
            )
        return ReferenceStep(
            plan,
            phase,
            self._integrate(goal, linear, angular, dt),
            goal,
            linear,
            angular,
            rate,
            lag >= 0.5,  # Ask for fresh predictions before the pacing budget is exhausted.
        )

    def commit(self, step):
        self.plan, self.elapsed_s, self.rate = step.plan, step.elapsed_s, step.rate
        self.reference = step.reference
        self.linear_velocity, self.angular_velocity = step.linear_velocity, step.angular_velocity

    def brake(self, dt):
        """Preserve reference continuity while the executor brakes its joints."""
        config = self.settings
        linear, angular = self.linear_velocity.copy(), self.angular_velocity.copy()
        for velocities, acceleration in (
            (linear, config.max_tcp_linear_acceleration),
            (angular, config.max_tcp_angular_acceleration),
        ):
            for velocity in velocities:
                length = float(np.linalg.norm(velocity))
                velocity *= max(0.0, 1 - acceleration * dt / max(length, 1e-12))
        self.reference = self._integrate(self.reference, linear, angular, dt)
        self.linear_velocity, self.angular_velocity, self.rate = linear, angular, 0.0
