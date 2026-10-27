"""SI coordinates. T_A_B maps coordinates in frame B into frame A.

URDF RPY means Rz(yaw) @ Ry(pitch) @ Rx(roll). MuJoCo quaternions are wxyz.
All matrices here are active transforms of column vectors in right-handed frames.
"""

from __future__ import annotations

import math

import numpy as np


def vector(value, size, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must contain {size} finite numbers")
    return result


def rotation_rpy(rpy):
    roll, pitch, yaw = vector(rpy, 3, "RPY")
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def rotation_axis(axis, angle):
    axis = vector(axis, 3, "axis")
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or not math.isfinite(angle):
        raise ValueError("Joint rotation requires a nonzero axis and finite angle")
    x, y, z = axis / norm
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    result = np.eye(4)
    result[:3, :3] = rotation_rpy(rpy)
    result[:3, 3] = vector(xyz, 3, "translation")
    return result


def rigid_transform(value):
    result = np.asarray(value, dtype=np.float64)
    if (
        result.shape != (4, 4)
        or not np.isfinite(result).all()
        or not np.allclose(result[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
        or not np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), rtol=0, atol=1e-10)
        or not np.isclose(np.linalg.det(result[:3, :3]), 1, rtol=0, atol=1e-10)
    ):
        raise ValueError("Expected a finite right-handed rigid transform")
    return result.copy()


def points_in_frame(T_A_B, points_B):
    pose = rigid_transform(T_A_B)
    points = np.asarray(points_B, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must have finite shape [N, 3]")
    return points @ pose[:3, :3].T + pose[:3, 3]
