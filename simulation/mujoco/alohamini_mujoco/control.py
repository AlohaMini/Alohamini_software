"""Time-defined joint trajectories and torque-limited dynamic following.

Only reset/IK scratch data write qpos. Following advances MuJoCo dynamics,
with bounded motor effort including any gravity/Coriolis feed-forward torque.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .audit import contacts
from .frames import rigid_transform


class JointTrajectory:
    def __init__(self, robot, joint_names, times_s, positions):
        self.names = tuple(joint_names)
        self.times = np.array(times_s, dtype=np.float64, copy=True)
        self.positions = np.array(positions, dtype=np.float64, copy=True)
        if (
            not self.names
            or len(set(self.names)) != len(self.names)
            or self.times.ndim != 1
            or len(self.times) < 2
            or self.times[0] != 0
            or not np.isfinite(self.times).all()
            or (np.diff(self.times) <= 0).any()
            or self.positions.shape != (len(self.times), len(self.names))
            or not np.isfinite(self.positions).all()
        ):
            raise ValueError(
                "Trajectory needs unique names, increasing times starting at zero, "
                "and finite waypoints"
            )
        joints = []
        for name in self.names:
            if name not in robot.description.joints or name in robot.profile.locked_positions:
                raise ValueError(f"Unknown or locked trajectory joint: {name}")
            joint = robot.description.joints[name]
            if joint.kind == "fixed" or joint.velocity is None:
                raise ValueError(f"Joint has no dynamic drive contract: {name}")
            for position in self.positions[:, len(joints)]:
                joint.check_position(position)
            joints.append(joint)
        self.velocity_limits = np.array([j.velocity for j in joints])
        # Cubic smoothstep has peak speed 1.5 * displacement / duration.
        self.peak_velocities = (
            1.5 * np.abs(np.diff(self.positions, axis=0)) / np.diff(self.times)[:, None]
        )
        if (self.peak_velocities > self.velocity_limits * (1 + 1e-12)).any():
            raise ValueError("Trajectory exceeds a URDF joint velocity limit")
        self.peak_accelerations = (
            6 * np.abs(np.diff(self.positions, axis=0)) / np.diff(self.times)[:, None] ** 2
        )

    def reference(self, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("Trajectory time must be finite and nonnegative")
        if elapsed_s >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(len(self.names))
        index = np.searchsorted(self.times, elapsed_s, side="right") - 1
        duration = self.times[index + 1] - self.times[index]
        u = (elapsed_s - self.times[index]) / duration
        delta = self.positions[index + 1] - self.positions[index]
        return (
            self.positions[index] + (3 * u**2 - 2 * u**3) * delta,
            6 * u * (1 - u) / duration * delta,
        )


@dataclass(frozen=True)
class FollowerSettings:
    bandwidth_hz: float = 2.0
    damping_ratio: float = 1.0
    settle_time_s: float = 1.0
    angular_tolerance_rad: float = 0.005
    linear_tolerance_m: float = 0.0005
    start_angular_tolerance_rad: float = 1e-5
    start_linear_tolerance_m: float = 1e-7
    penetration_tolerance_m: float = 0.0005


class DynamicFollower:
    def __init__(self, robot, data=None, settings=None):
        self.robot = robot
        self.model = robot.model
        self.data = robot.data() if data is None else data
        self.settings = settings or FollowerSettings()
        settings = self.settings
        if any(not math.isfinite(v) or v <= 0 for v in vars(settings).values()):
            raise ValueError("Follower settings must be finite and positive")
        self.names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        self.joint_ids = np.array([self.model.joint(name).id for name in self.names])
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.dof_ids = self.model.jnt_dofadr[self.joint_ids]
        self.effort_limits = self.model.actuator_ctrlrange[:, 1].copy()
        self.kinds = [robot.description.joints[n].kind for n in self.names]
        self.target = self.data.qpos[self.qpos_ids].copy()
        self.target_velocity = np.zeros(len(self.names))
        mujoco.mj_forward(self.model, self.data)
        matrix = np.empty((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, matrix, self.data.qM)
        # A declared bandwidth and damping ratio determine gains from each
        # joint's effective inertia at initialization. These are simulation
        # controller settings, not fitted hardware servo parameters.
        inertia = np.diag(matrix)[self.dof_ids]
        omega = 2 * math.pi * settings.bandwidth_hz
        self.kp = inertia * omega**2
        self.kd = 2 * settings.damping_ratio * omega * inertia

    def step(self):
        data = self.data
        mujoco.mj_forward(self.model, data)
        effort = (
            self.kp * (self.target - data.qpos[self.qpos_ids])
            + self.kd * (self.target_velocity - data.qvel[self.dof_ids])
            + data.qfrc_bias[self.dof_ids]
        )
        saturated = np.abs(effort) > self.effort_limits
        data.ctrl[:] = np.clip(effort, -self.effort_limits, self.effort_limits)
        mujoco.mj_step(self.model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Nonfinite dynamics state")
        if any(w.number for w in data.warning):
            raise RuntimeError("MuJoCo issued a dynamics warning")
        return saturated

    def trajectory_indices(self, trajectory):
        """Resolve driven joints and require a continuous start at the measured pose."""
        indices = np.array([self.names.index(n) for n in trajectory.names])
        position_ids = self.qpos_ids[indices]
        start = self.data.qpos[position_ids].copy()
        start_tolerances = np.array(
            [
                self.settings.start_linear_tolerance_m
                if self.kinds[i] == "prismatic"
                else self.settings.start_angular_tolerance_rad
                for i in indices
            ]
        )
        if (np.abs(start - trajectory.positions[0]) > start_tolerances).any():
            raise ValueError("Trajectory must start at the measured joint positions")
        return indices

    def follow(self, trajectory, *, observer=None, observation_period_s=0.02):
        if (
            not math.isfinite(observation_period_s)
            or observation_period_s < self.model.opt.timestep
        ):
            raise ValueError("Observation period must be finite and at least one physics step")
        indices = self.trajectory_indices(trajectory)
        position_ids = self.qpos_ids[indices]
        begin = float(self.data.time)
        duration = trajectory.times[-1] + self.settings.settle_time_s
        steps = math.ceil(duration / self.model.opt.timestep)
        saturated_steps = np.zeros(len(self.names), dtype=int)
        max_error = np.zeros(len(indices))
        peak_torque = np.zeros(len(self.names))
        peak_speed = np.zeros(len(self.names))
        next_observation = 0.0
        minimum_contact_distance = 0.0
        limit_violations = np.zeros(len(self.names))
        for _ in range(steps):
            elapsed = float(self.data.time) - begin
            q, velocity = trajectory.reference(elapsed)
            self.target[indices] = q
            self.target_velocity[indices] = velocity
            # Record s_t and u_t before advancing, never next-state feedback as
            # an action label. The optional observer cannot reset the follower.
            if observer is not None and elapsed + 1e-12 >= next_observation:
                observer(
                    {
                        "time_s": elapsed,
                        "joint_names": self.names,
                        "position": self.data.qpos[self.qpos_ids].copy(),
                        "reference": self.target.copy(),
                        "reference_velocity": self.target_velocity.copy(),
                    }
                )
                next_observation += observation_period_s
            saturated_steps += self.step()
            expected, _ = trajectory.reference(float(self.data.time) - begin)
            max_error = np.maximum(max_error, np.abs(self.data.qpos[position_ids] - expected))
            peak_torque = np.maximum(peak_torque, np.abs(self.data.ctrl))
            peak_speed = np.maximum(peak_speed, np.abs(self.data.qvel[self.dof_ids]))
            for index, name in enumerate(self.names):
                joint = self.robot.description.joints[name]
                q = self.data.qpos[self.qpos_ids[index]]
                limit_violations[index] = max(
                    limit_violations[index], joint.lower - q, q - joint.upper
                )
            if self.data.ncon:
                minimum_contact_distance = min(
                    minimum_contact_distance, min(c.dist for c in self.data.contact)
                )
        mujoco.mj_forward(self.model, self.data)
        error = np.abs(self.data.qpos[position_ids] - trajectory.positions[-1])
        tolerances = np.array(
            [
                self.settings.linear_tolerance_m
                if self.kinds[i] == "prismatic"
                else self.settings.angular_tolerance_rad
                for i in indices
            ]
        )
        drive_tolerances = np.array(
            [
                self.settings.linear_tolerance_m
                if kind == "prismatic"
                else self.settings.angular_tolerance_rad
                for kind in self.kinds
            ]
        )
        speed_limits = np.array([self.robot.description.joints[n].velocity for n in self.names])
        all_final_errors = np.abs(self.data.qpos[self.qpos_ids] - self.target)
        passed = (
            (error <= tolerances).all()
            and (all_final_errors <= drive_tolerances).all()
            and (limit_violations <= drive_tolerances).all()
            and (peak_speed <= speed_limits * (1 + 1e-6)).all()
            and minimum_contact_distance >= -self.settings.penetration_tolerance_m
        )
        return {
            "passed": bool(passed),
            "duration_s": float(self.data.time) - begin,
            "joint_names": list(trajectory.names),
            "joint_units": ["m" if self.kinds[i] == "prismatic" else "rad" for i in indices],
            "final_error": error.tolist(),
            "maximum_tracking_error": max_error.tolist(),
            "trajectory_peak_velocity": trajectory.peak_velocities.max(axis=0).tolist(),
            "trajectory_peak_acceleration": trajectory.peak_accelerations.max(axis=0).tolist(),
            "drive_joint_names": list(self.names),
            "all_drives_final_error": all_final_errors.tolist(),
            "drive_saturated_steps": saturated_steps.tolist(),
            "maximum_motor_effort": peak_torque.tolist(),
            "maximum_realized_speed": peak_speed.tolist(),
            "maximum_joint_limit_violation": limit_violations.tolist(),
            "minimum_contact_distance_m": float(minimum_contact_distance),
            "final_contacts": contacts(self.robot, self.data),
            "controller": {
                **vars(self.settings),
                "kp": self.kp.tolist(),
                "kd": self.kd.tolist(),
                "derivation": (
                    "diagonal effective inertia at initialization; "
                    "kp=I*(2*pi*f)^2, kd=2*zeta*I*2*pi*f"
                ),
            },
        }


def solve_ik(
    robot,
    side,
    T_world_tcp,
    *,
    seed_positions=None,
    max_iterations=200,
    position_tolerance_m=1e-5,
    rotation_tolerance_rad=1e-3,
):
    """Full-pose IK in independent scratch data; never mutate a live rollout."""
    if side not in ("left", "right"):
        raise ValueError("Expected left or right arm")
    target = rigid_transform(T_world_tcp)
    if (
        type(max_iterations) is not int
        or max_iterations <= 0
        or any(
            not math.isfinite(v) or v <= 0 for v in (position_tolerance_m, rotation_tolerance_rad)
        )
    ):
        raise ValueError("Invalid IK convergence settings")
    names = [
        f"{side}_{suffix}"
        for suffix in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw_joint",
            "wrist_roll",
        )
    ]
    if set(names) & robot.profile.locked_positions.keys():
        raise ValueError("IK arm contains a locked joint")
    data = robot.data(seed_positions)
    model = robot.model
    joints = [robot.description.joints[n] for n in names]
    qpos_ids = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    dof_ids = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    lower, upper = np.array([j.lower for j in joints]), np.array([j.upper for j in joints])
    body_id = model.body(f"{side}_tcp").id
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    position_error = rotation_error = float("inf")
    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)
        delta_p = target[:3, 3] - data.xpos[body_id]
        delta_R = target[:3, :3] @ data.xmat[body_id].reshape(3, 3).T
        quaternion, delta_angle = np.empty(4), np.empty(3)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(delta_R).ravel())
        mujoco.mju_quat2Vel(delta_angle, quaternion, 1.0)
        position_error, rotation_error = np.linalg.norm(delta_p), np.linalg.norm(delta_angle)
        if position_error <= position_tolerance_m and rotation_error <= rotation_tolerance_rad:
            return {
                "joint_positions": dict(zip(names, data.qpos[qpos_ids].tolist(), strict=True)),
                "position_error_m": float(position_error),
                "rotation_error_rad": float(rotation_error),
                "iterations": iteration,
                "contacts": contacts(robot, data),
            }
        mujoco.mj_jacBody(model, data, linear, angular, body_id)
        # Scale position by a declared 0.1 m characteristic tool length so
        # translation/rotation conditioning does not depend on a unit accident.
        jacobian = np.vstack((linear[:, dof_ids] / 0.1, angular[:, dof_ids]))
        residual = np.r_[delta_p / 0.1, delta_angle]
        step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), residual)
        step /= max(1, np.max(np.abs(step)) / 0.1)
        data.qpos[qpos_ids] = np.clip(data.qpos[qpos_ids] + step, lower, upper)
    raise RuntimeError(
        f"Full-pose IK did not converge: position={position_error:.6g} m, "
        f"rotation={rotation_error:.6g} rad"
    )
