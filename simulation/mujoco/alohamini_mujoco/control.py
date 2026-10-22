"""Time-defined joint trajectories and torque-limited dynamic following.

Only reset/IK scratch data write qpos. Following advances MuJoCo dynamics,
with bounded motor effort including any gravity/Coriolis feed-forward torque.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import least_squares

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
        if "cad_component_fixture" in robot.manifest:
            raise ValueError(
                "Isolated CAD component contact fixtures cannot run dynamics or collection"
            )
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
        wall_begin = time.perf_counter()
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
        wall_time = time.perf_counter() - wall_begin
        return {
            "passed": bool(passed),
            "duration_s": float(self.data.time) - begin,
            "wall_time_s": wall_time,
            "real_time_factor": (float(self.data.time) - begin) / wall_time,
            "wall_time_includes_observer": observer is not None,
            "maximum_arena_use_bytes": int(self.data.maxuse_arena),
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


def arm_joint_names(side):
    if side not in ("left", "right"):
        raise ValueError("Expected left or right arm")
    return tuple(
        f"{side}_{suffix}"
        for suffix in (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_yaw_joint",
            "wrist_roll",
        )
    )


def solve_ik(
    robot,
    side,
    T_world_tcp,
    *,
    seed_positions=None,
    reference_data=None,
    max_iterations=200,
    position_tolerance_m=1e-5,
    rotation_tolerance_rad=1e-3,
):
    """Full-pose single-arm IK, with the lift and other arm held at their seed."""
    names = arm_joint_names(side)
    return solve_ik_targets(
        robot,
        {side: T_world_tcp},
        movable_joints=names,
        seed_positions=seed_positions,
        reference_data=reference_data,
        max_iterations=max_iterations,
        position_tolerance_m=position_tolerance_m,
        rotation_tolerance_rad=rotation_tolerance_rad,
    )


def solve_ik_targets(
    robot,
    targets,
    *,
    movable_joints=None,
    seed_positions=None,
    reference_data=None,
    max_iterations=200,
    position_tolerance_m=1e-5,
    rotation_tolerance_rad=1e-3,
):
    """Solve simultaneous world TCP poses in independent scratch data.

    Explicitly including vertical_move allows coordinated lift/arm IK. A copied
    reference_data preserves measured scene-object poses for contact diagnostics.
    Iterations update kinematics only; the final full forward pass computes
    contacts. Convergence does not establish a collision-free path.
    """
    if not targets or any(side not in ("left", "right") for side in targets):
        raise ValueError("Provide world TCP poses keyed by left and/or right")
    targets = {side: rigid_transform(pose) for side, pose in targets.items()}
    if (
        type(max_iterations) is not int
        or max_iterations <= 0
        or any(
            not math.isfinite(v) or v <= 0 for v in (position_tolerance_m, rotation_tolerance_rad)
        )
    ):
        raise ValueError("Invalid IK convergence settings")
    names = (
        tuple(movable_joints)
        if movable_joints is not None
        else tuple(name for side in targets for name in arm_joint_names(side))
    )
    if not names or len(set(names)) != len(names):
        raise ValueError("IK requires unique movable joint names")
    joints = []
    for name in names:
        joint = robot.description.joints.get(name)
        if (
            joint is None
            or name in robot.profile.locked_positions
            or joint.kind not in ("revolute", "prismatic")
            or joint.lower is None
            or joint.upper is None
        ):
            raise ValueError(f"IK joint must be an unlocked bounded robot joint: {name}")
        joints.append(joint)
    model = robot.model
    if reference_data is None:
        data = robot.data(seed_positions)
    else:
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, reference_data)
        for name, position in (seed_positions or {}).items():
            if name not in robot.description.joints:
                raise ValueError(f"Unknown IK seed joint: {name}")
            joint = robot.description.joints[name]
            joint.check_position(position)
            if name in robot.profile.locked_positions:
                if position != robot.profile.locked_positions[name]:
                    raise ValueError(f"IK seed conflicts with locked joint: {name}")
                continue
            if joint.kind == "fixed":
                if position != 0:
                    raise ValueError(f"Fixed IK seed joint has a nonzero position: {name}")
                continue
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = position
    qpos_ids = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
    dof_ids = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
    lower, upper = np.array([j.lower for j in joints]), np.array([j.upper for j in joints])
    body_ids = {side: model.body(f"{side}_tcp").id for side in targets}
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    errors, cached = {}, {}

    def evaluate(q):
        if "q" in cached and np.array_equal(q, cached["q"]):
            return cached["residual"], cached["jacobian"]
        data.qpos[qpos_ids] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        residuals, jacobians = [], []
        for side, target in targets.items():
            body_id = body_ids[side]
            delta_p = data.xpos[body_id] - target[:3, 3]
            delta_R = data.xmat[body_id].reshape(3, 3) @ target[:3, :3].T
            quaternion, delta_angle = np.empty(4), np.empty(3)
            mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(delta_R).ravel())
            mujoco.mju_quat2Vel(delta_angle, quaternion, 1.0)
            errors[side] = {
                "position_error_m": float(np.linalg.norm(delta_p)),
                "rotation_error_rad": float(np.linalg.norm(delta_angle)),
            }
            mujoco.mj_jacBody(model, data, linear, angular, body_id)
            # Explicit 0.1 m characteristic length conditions mixed SI residuals.
            residuals.append(np.r_[delta_p / 0.1, delta_angle])
            jacobians.append(
                np.vstack(
                    (
                        linear[:, dof_ids] / 0.1,
                        _rotation_log_inverse(delta_angle) @ angular[:, dof_ids],
                    )
                )
            )
        cached.update(q=q.copy(), residual=np.concatenate(residuals), jacobian=np.vstack(jacobians))
        return cached["residual"], cached["jacobian"]

    solution = least_squares(
        lambda q: evaluate(q)[0],
        np.clip(data.qpos[qpos_ids], lower, upper),
        jac=lambda q: evaluate(q)[1],
        bounds=(lower, upper),
        x_scale=upper - lower,
        method="trf",
        max_nfev=max_iterations,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    evaluate(solution.x)
    if all(
        e["position_error_m"] <= position_tolerance_m
        and e["rotation_error_rad"] <= rotation_tolerance_rad
        for e in errors.values()
    ):
        mujoco.mj_forward(model, data)
        return {
            "joint_positions": dict(zip(names, data.qpos[qpos_ids].tolist(), strict=True)),
            "position_error_m": max(e["position_error_m"] for e in errors.values()),
            "rotation_error_rad": max(e["rotation_error_rad"] for e in errors.values()),
            "target_errors": errors,
            "iterations": int(solution.nfev),
            "solver": (
                "bounded trust-region reflective least squares, analytic SE(3) residual Jacobian"
            ),
            "contacts": contacts(robot, data),
        }
    raise RuntimeError(f"Full-pose IK did not converge: {errors}")


def _rotation_log_inverse(rotation_vector):
    """Inverse SO(3) left Jacobian for a world-frame log-rotation residual."""
    x, y, z = rotation_vector
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = float(np.linalg.norm(rotation_vector))
    coefficient = (
        1 / 12 + angle**2 / 720
        if angle < 1e-4
        else (1 - angle / (2 * math.tan(angle / 2))) / angle**2
    )
    return np.eye(3) - 0.5 * cross + coefficient * cross @ cross
