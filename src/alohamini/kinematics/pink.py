"""Constrained pose IK and feedback tracking over portable DH snapshots.

Pinocchio owns SE(3) derivatives; Pink assembles the QP.  Building the chain
from the frozen geometry keeps checkpoint coordinates independent of installed
URDF revisions.  This arm-only model has no collision geometry.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .arm import _rotation_vector


class IKError(ValueError):
    """A failed solve with the target, seed and achieved pose error attached."""

    def __init__(self, reason, *, target, seed, metrics):
        self.reason = reason
        self.target = np.asarray(target).tolist()
        self.seed = np.asarray(seed).tolist()
        self.metrics = dict(metrics)
        super().__init__(
            f"IK did not converge: {reason}; position={metrics['position_error_m']:.6g} m, "
            f"rotation={metrics['orientation_error_rad']:.6g} rad, "
            f"iterations={metrics['iterations']}, elapsed={metrics['elapsed_s']:.4f} s"
        )


class _JointBounds:
    """Position, velocity and displacement bounds in Pink's displacement QP."""

    def __init__(self, lower, upper, velocity, max_step, margin):
        self.lower, self.upper = lower, upper
        self.velocity, self.max_step, self.margin = velocity, max_step, margin

    def compute_qp_inequalities(self, configuration, dt):
        q = configuration.q
        # Inside the margin, permit recovery inward without demanding a jump.
        lower = np.minimum(self.lower + self.margin, q)
        upper = np.maximum(self.upper - self.margin, q)
        step = np.minimum(self.velocity * dt, self.max_step)
        identity = np.eye(len(q))
        high, low = np.minimum(upper - q, step), np.minimum(q - lower, step)
        return np.vstack((identity, -identity)), np.r_[high, low]


class PinkArmSolver:
    """One cached Pinocchio model per arm; all external angles are model radians."""

    def __init__(self, geometry, *, qp_solver="daqp"):
        try:
            import pink
            import pinocchio as pin
            import qpsolvers
        except ImportError as exc:
            raise ImportError(
                "Pink IK requires the kinematics extra in this Python environment: "
                "python -m pip install -e '.[kinematics]'"
            ) from exc
        if qp_solver not in qpsolvers.available_solvers:
            raise ValueError(f"QP backend {qp_solver!r} is not installed")
        self.pin, self.pink = pin, pink
        self.geometry, self.qp_solver = geometry, qp_solver
        model = pin.Model()
        placement = np.array(geometry.base_transform, copy=True)
        parent = 0
        names = getattr(geometry, "joint_names", [f"q{i}" for i in range(len(geometry.rows))])
        for name, (a, alpha, d, offset) in zip(names, geometry.rows, strict=True):
            ct, st = math.cos(offset), math.sin(offset)
            fixed = np.array([[ct, -st, 0, 0], [st, ct, 0, 0], [0, 0, 1, d], [0, 0, 0, 1]])
            parent = model.addJoint(parent, pin.JointModelRZ(), pin.SE3(placement @ fixed), name)
            ca, sa = math.cos(alpha), math.sin(alpha)
            placement = np.array([[1, 0, 0, a], [0, ca, -sa, 0], [0, sa, ca, 0], [0, 0, 0, 1]])
        self.frame = "tcp"
        model.addFrame(
            pin.Frame(
                self.frame, parent, 0, pin.SE3(placement @ geometry.tool_transform), pin.OP_FRAME
            )
        )
        size = len(geometry.rows)
        if geometry.joint_limits is None:
            self.lower, self.upper = np.full(size, -np.inf), np.full(size, np.inf)
        else:
            self.lower, self.upper = np.asarray(geometry.joint_limits, dtype=float).T.copy()
        model.lowerPositionLimit = self.lower
        model.upperPositionLimit = self.upper
        model.velocityLimit = np.full(size, 1.5)
        self.model = model
        self.configuration = pink.Configuration(model, model.createData(), np.zeros(size))
        self.previous_velocity = np.zeros(size)
        self.last_tracking_time = None

    def reset(self):
        self.previous_velocity.fill(0)
        self.last_tracking_time = None

    def _prepare(self, joints, target):
        joints = np.asarray(joints, dtype=float)
        self.geometry.forward(joints)
        target = np.asarray(target, dtype=float)
        if (
            target.shape != (4, 4)
            or not np.isfinite(target).all()
            or not np.allclose(target[3], [0, 0, 0, 1])
            or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(target[:3, :3]), 1)
        ):
            raise ValueError("IK target must be a finite rigid transform")
        if ((joints < self.lower - 1e-9) | (joints > self.upper + 1e-9)).any():
            raise ValueError("IK seed lies outside the declared joint limits")
        joints = np.clip(joints, self.lower, self.upper)
        self.configuration.update(joints)
        return joints.copy(), target

    def _metrics(self, target, *, iterations=1, started=None):
        actual = self.configuration.get_transform_frame_to_world(self.frame).homogeneous
        return {
            "position_error_m": float(np.linalg.norm(target[:3, 3] - actual[:3, 3])),
            "orientation_error_rad": float(
                np.linalg.norm(_rotation_vector(target[:3, :3] @ actual[:3, :3].T))
            ),
            "iterations": iterations,
            "elapsed_s": 0.0 if started is None else time.perf_counter() - started,
            "backend": "pink",
            "qp_solver": self.qp_solver,
        }

    def inverse(
        self,
        target,
        seed,
        *,
        max_iterations=100,
        position_tolerance=1e-5,
        rotation_tolerance=1e-4,
        timeout_s=0.15,
        damping=1e-6,
    ):
        """Iterate local QPs; internal steps are numerical, not execution time."""
        seed, target = self._prepare(seed, target)
        self.reset()
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if any(
            not math.isfinite(x) or x <= 0
            for x in (position_tolerance, rotation_tolerance, timeout_s, damping)
        ):
            raise ValueError("IK tolerances, damping and timeout must be finite and positive")
        task = self.pink.tasks.FrameTask(
            self.frame, position_cost=10.0, orientation_cost=1.0, lm_damping=1e-3
        )
        task.set_target(self.pin.SE3(target))
        bounds = _JointBounds(self.lower, self.upper, 4.0, 0.2, 0.0)
        started, reason = time.perf_counter(), "iteration limit"
        for iteration in range(max_iterations + 1):
            metrics = self._metrics(target, iterations=iteration, started=started)
            if (
                metrics["position_error_m"] <= position_tolerance
                and metrics["orientation_error_rad"] <= rotation_tolerance
            ):
                self.last_metrics = metrics
                return self.configuration.q.copy()
            if metrics["elapsed_s"] >= timeout_s or iteration == max_iterations:
                reason = "time budget" if metrics["elapsed_s"] >= timeout_s else reason
                break
            try:
                velocity = self.pink.solve_ik(
                    self.configuration,
                    [task],
                    0.05,
                    self.qp_solver,
                    damping=damping,
                    limits=[bounds],
                )
            except self.pink.exceptions.NoSolutionFound:
                reason = "infeasible local QP"
                break
            self.configuration.integrate_inplace(velocity, 0.05)
        self.last_metrics = metrics
        raise IKError(reason, target=target, seed=seed, metrics=metrics)

    def step(
        self,
        joints,
        target,
        dt,
        *,
        position_gain=8.0,
        orientation_gain=5.0,
        orientation_weight=0.35,
        damping=0.04,
        max_joint_velocity=1.5,
        max_joint_step=0.08,
        joint_limit_margin=0.03,
        max_joint_acceleration=8.0,
    ):
        """Solve one constrained feedback step; residual tracking error is reported."""
        joints, target = self._prepare(joints, target)
        positive = (dt, damping, max_joint_velocity, max_joint_step, max_joint_acceleration)
        nonnegative = (position_gain, orientation_gain, orientation_weight, joint_limit_margin)
        if any(not math.isfinite(x) or x <= 0 for x in positive) or any(
            not math.isfinite(x) or x < 0 for x in nonnegative
        ):
            raise ValueError("Tracking requires finite positive limits/dt and nonnegative gains")
        if np.any(self.upper - self.lower <= 2 * joint_limit_margin):
            raise ValueError("Joint margin leaves no tracking interval")
        started = time.perf_counter()
        if self.last_tracking_time is None or started - self.last_tracking_time > 0.25:
            self.previous_velocity.fill(0)
        tasks = []
        for position_cost, orientation_cost, gain in (
            (10.0, 0.0, position_gain),
            (0.0, orientation_weight, orientation_gain),
        ):
            task = self.pink.tasks.FrameTask(
                self.frame,
                position_cost=position_cost,
                orientation_cost=orientation_cost,
                gain=min(1.0, gain * dt),
                lm_damping=damping * damping,
            )
            task.set_target(self.pin.SE3(target))
            tasks.append(task)
        bounds = _JointBounds(
            self.lower, self.upper, max_joint_velocity, max_joint_step, joint_limit_margin,
        )
        from pink.limits import AccelerationLimit

        acceleration = AccelerationLimit(self.model, np.full(len(joints), max_joint_acceleration))
        acceleration.set_last_integration(self.previous_velocity, dt)
        try:
            velocity = self.pink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.qp_solver,
                damping=damping * damping,
                limits=[bounds, acceleration],
            )
        except self.pink.exceptions.NoSolutionFound as exc:
            self.reset()
            raise IKError(
                "infeasible tracking QP",
                target=target,
                seed=joints,
                metrics=self._metrics(target, started=started),
            ) from exc
        self.configuration.integrate_inplace(velocity, dt)
        candidate = self.configuration.q.copy()
        self.previous_velocity = velocity.copy()
        self.last_tracking_time = started
        metrics = self._metrics(target, started=started)
        metrics.update(
            max_joint_step_rad=float(np.max(np.abs(candidate - joints))),
            max_joint_velocity_rad_s=float(np.max(np.abs(velocity))),
            damping=damping,
            minimum_singular_value=float(
                np.linalg.svd(self.configuration.get_frame_jacobian(self.frame), compute_uv=False)[
                    -1
                ]
            ),
            joint_limit_hits=np.flatnonzero(
                (candidate <= self.lower + joint_limit_margin + 1e-8)
                | (candidate >= self.upper - joint_limit_margin - 1e-8)
            ).tolist(),
        )
        return candidate, metrics
