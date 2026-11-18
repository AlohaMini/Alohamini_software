"""Planar base motion from named wheel geometry in the shared robot frame."""

import math

import numpy as np

from alohamini.schema import BodyVelocity

from .description import origin
from .frames import rigid_transform


class BaseKinematics:
    """Ideal rolling kinematics in wheel1, wheel2, wheel3 joint order.

    Wheel centers, axle directions and radii come from URDF collision cylinders.
    This describes wheel speeds, not uncalibrated omni-roller contact dynamics.
    Host nominal geometry remains in alohamini.kinematics.base.
    """

    names = ("wheel1_joint", "wheel2_joint", "wheel3_joint")

    def __init__(self, description):
        poses = description.forward()
        rows, self.geometry = [], []
        for name in self.names:
            joint = description.joints[name]
            if joint.kind != "continuous":
                raise ValueError(f"Wheel must retain its continuous joint: {name}")
            collision = description.links[joint.child].find("collision")
            cylinder = collision.find("geometry/cylinder")
            if cylinder is None:
                raise ValueError(f"Wheel needs an explicit collision cylinder: {name}")
            pose = poses[joint.child]
            center = (pose @ origin(collision))[:3, 3]
            axle = pose[:3, :3] @ joint.axis
            tangent = np.cross(axle, [0, 0, 1])[:2]
            if np.linalg.norm(tangent) < 0.99:
                raise ValueError("Planar wheel axle must be horizontal")
            tangent /= np.linalg.norm(tangent)
            radius = float(cylinder.get("radius"))
            lever = tangent @ [-center[1], center[0]]
            rows.append([*tangent / radius, lever / radius])
            self.geometry.append(
                {
                    "joint": name,
                    "center_base_m": center.tolist(),
                    "rolling_direction_base": tangent.tolist(),
                    "radius_m": radius,
                    "yaw_lever_arm_m": float(lever),
                }
            )
        self.matrix = np.asarray(rows)
        if not np.isfinite(self.matrix).all() or np.linalg.cond(self.matrix) > 100:
            raise ValueError("Invalid or singular wheel geometry")

    def body_to_wheels(self, velocity):
        if not isinstance(velocity, BodyVelocity):
            raise TypeError("Provide a body-frame BodyVelocity in SI units")
        return self.matrix @ [velocity.x_m_s, velocity.y_m_s, velocity.yaw_rad_s]

    def wheels_to_body(self, wheels_rad_s):
        wheels = np.asarray(wheels_rad_s, dtype=float)
        if wheels.shape != (3,) or not np.isfinite(wheels).all():
            raise ValueError("Expected finite wheel1/wheel2/wheel3 speeds in rad/s")
        return BodyVelocity(*map(float, np.linalg.solve(self.matrix, wheels)))


def integrate_base(T_world_base, velocity, dt):
    """Exact constant-body-twist SE(2) integration; Z stays at its initial value."""
    pose = rigid_transform(T_world_base)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive")
    if not np.allclose(pose[:3, 2], [0, 0, 1], rtol=0, atol=1e-10):
        raise ValueError("Planar base pose must keep Z up")
    angle = velocity.yaw_rad_s * dt
    if abs(angle) < 1e-8:
        sine = dt * np.sinc(angle / np.pi)
        cosine = dt * angle / 2 * np.sinc(angle / (2 * np.pi)) ** 2
    else:
        sine, cosine = (
            math.sin(angle) / velocity.yaw_rad_s,
            (1 - math.cos(angle)) / velocity.yaw_rad_s,
        )
    local = np.array(
        [
            sine * velocity.x_m_s - cosine * velocity.y_m_s,
            cosine * velocity.x_m_s + sine * velocity.y_m_s,
            0,
        ]
    )
    result = pose.copy()
    result[:3, 3] += pose[:3, :3] @ local
    yaw = math.atan2(pose[1, 0], pose[0, 0]) + angle
    c, s = math.cos(yaw), math.sin(yaw)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    return result


class BaseMotion:
    """A prescribed planar base for scene research; forces do not drive its pose."""

    def __init__(self, description, T_world_base, *, mode="fixed", wheel_positions=None):
        if mode not in ("fixed", "kinematic"):
            raise ValueError("Choose fixed or kinematic base mode")
        self.kinematics = BaseKinematics(description)
        self.pose = rigid_transform(T_world_base)
        self.mode, self.velocity = mode, BodyVelocity()
        self.wheel_positions = (
            np.zeros(3) if wheel_positions is None else np.array(wheel_positions, dtype=float)
        )
        self.speed_limits = np.array(
            [description.joints[n].velocity for n in self.kinematics.names]
        )

    def set_velocity(self, velocity):
        wheels = self.kinematics.body_to_wheels(velocity)
        if self.mode == "fixed" and np.any(wheels):
            raise ValueError(
                "Scene uses a fixed base; choose base_mode=kinematic to command motion"
            )
        if (np.abs(wheels) > self.speed_limits).any():
            raise ValueError("Base velocity exceeds a wheel speed limit")
        self.velocity = velocity

    def step(self, dt):
        if self.mode == "kinematic":
            self.pose = integrate_base(self.pose, self.velocity, dt)
            self.wheel_positions += self.kinematics.body_to_wheels(self.velocity) * dt
        return self.pose
