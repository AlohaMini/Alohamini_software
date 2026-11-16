"""SI joint targets with explicit velocity and total motor-effort limits."""

import math

import numpy as np


class PositionControl:
    def __init__(self, description, names, positions, *, bandwidth_hz=2, timestep=0.001):
        if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0 or not 0 < timestep <= 0.01:
            raise ValueError("Provide positive bandwidth and a physics step in (0, .01] s")
        self.description, self.names, self.timestep = description, tuple(names), timestep
        self.target = np.array(positions, dtype=float, copy=True)
        self.goal = self.target.copy()
        self.velocity = np.zeros_like(self.target)
        joints = [description.joints[n] for n in self.names]
        if any(j.velocity is None or j.effort is None for j in joints):
            raise ValueError("Every driven joint needs explicit effort and speed limits")
        self.speed_limits = np.array([j.velocity for j in joints])
        self.effort_limits = np.array([j.effort for j in joints])
        self.set_goal(self.goal)
        self.target[:] = self.goal
        values = dict(zip(self.names, self.target, strict=True))
        poses = description.forward(values)
        parent = {j.child: j for j in description.joints.values()}
        effective = np.zeros(len(joints))
        indices = {n: i for i, n in enumerate(self.names)}
        for link, (mass, com, inertia) in description.inertials.items():
            world_com = poses[link] @ com
            tensor = world_com[:3, :3] @ inertia @ world_com[:3, :3].T
            while link in parent:
                joint = parent[link]
                if joint.name in indices:
                    i = indices[joint.name]
                    frame = poses[joint.parent] @ joint.T_parent_joint
                    axis = frame[:3, :3] @ joint.axis
                    if joint.kind == "prismatic":
                        effective[i] += mass
                    else:
                        linear = np.cross(axis, world_com[:3, 3] - frame[:3, 3])
                        effective[i] += mass * (linear @ linear) + axis @ tensor @ axis
                link = joint.parent
        if (effective <= 0).any():
            raise ValueError("A driven joint has no positive generalized inertia")
        omega = 2 * np.pi * bandwidth_hz
        self.kp, self.kd = effective * omega**2, 2 * effective * omega

    def set_goal(self, positions):
        positions = np.asarray(positions, dtype=float)
        if positions.shape != self.goal.shape:
            raise ValueError("Target shape differs from the named joint order")
        positions = positions.copy()
        for i, (name, value) in enumerate(zip(self.names, positions, strict=True)):
            joint = self.description.joints[name]
            if joint.lower is not None:
                # PhysX/ManiSkill expose float32 states and actions. Recover an
                # exact declared endpoint only for its float32 representation;
                # no tolerance or wider physical interval is introduced.
                for endpoint in (joint.lower, joint.upper):
                    if value == float(np.float32(endpoint)):
                        positions[i] = endpoint
            joint.check_position(float(positions[i]))
        self.goal[:] = positions

    def effort(self, positions, velocities, passive):
        delta = np.clip(
            self.goal - self.target,
            -self.speed_limits * self.timestep,
            self.speed_limits * self.timestep,
        )
        self.target += delta
        self.velocity[:] = delta / self.timestep
        effort = (
            self.kp * (self.target - positions) + self.kd * (self.velocity - velocities) + passive
        )
        if not np.isfinite(effort).all():
            raise ValueError("Nonfinite actuator effort")
        return np.clip(effort, -self.effort_limits, self.effort_limits)

    def state(self):
        return {
            name: getattr(self, name).copy() for name in ("target", "goal", "velocity", "kp", "kd")
        }

    def restore(self, state):
        values = {
            k: np.asarray(state[k], dtype=float).copy()
            for k in ("target", "goal", "velocity", "kp", "kd")
        }
        if any(v.shape != self.target.shape or not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Invalid controller snapshot dimensions or values")
        for key in ("target", "goal"):
            for n, q in zip(self.names, values[key], strict=True):
                self.description.joints[n].check_position(float(q))
        if (
            (values["kp"] <= 0).any()
            or (values["kd"] < 0).any()
            or (np.abs(values["velocity"]) > self.speed_limits * (1 + 1e-12)).any()
        ):
            raise ValueError("Invalid controller snapshot gains or velocity")
        for name, value in values.items():
            getattr(self, name)[:] = value
