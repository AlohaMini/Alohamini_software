"""Explicit TCP orientation encodings, independent of reference-frame composition."""

import numpy as np
from scipy.spatial.transform import Rotation


class RotationEncoding:
    """6D stores the first two matrix rows (r00, r01, r02, r10, r11, r12).

    Decode with Gram-Schmidt, as in HyVLA/UMI. Degenerate predictions cannot
    define a rotation and must fail before reaching inverse kinematics.
    """

    def __init__(self, representation="rotvec"):
        if representation not in ("rotvec", "rotation_6d"):
            raise ValueError("Rotation representation must be rotvec or rotation_6d")
        self.representation = representation
        self.names = (
            ("rx", "ry", "rz")
            if representation == "rotvec"
            else ("r00", "r01", "r02", "r10", "r11", "r12")
        )
        self.width = len(self.names)
        self.units = ["rad" if representation == "rotvec" else "dimensionless"] * self.width

    def encode(self, matrices):
        matrices = np.asarray(matrices, dtype=np.float64)
        shape = matrices.shape[:-2]
        if self.representation == "rotation_6d":
            return matrices[..., :2, :].reshape((*shape, 6)).copy()
        if not matrices.size:
            return np.empty((*shape, 3), dtype=np.float64)
        return Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape((*shape, 3))

    def decode(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1:] != (self.width,) or not np.isfinite(values).all():
            raise ValueError("Expected finite coordinates for the declared rotation representation")
        shape = values.shape[:-1]
        if not values.size:
            return np.empty((*shape, 3, 3), dtype=np.float64)
        if self.representation == "rotvec":
            return (
                Rotation.from_rotvec(values.reshape(-1, 3).copy())
                .as_matrix()
                .reshape((*shape, 3, 3))
            )
        # Scale first to avoid overflow and make collinearity checks independent
        # of the magnitude of unnormalized network outputs.
        rows = values.reshape((*shape, 2, 3))
        scales = np.max(np.abs(rows), axis=-1, keepdims=True)
        if np.any(scales == 0):
            raise ValueError("Degenerate rotation_6d: rows must be nonzero and independent")
        rows = rows / scales
        first = rows[..., 0, :]
        first = first / np.linalg.norm(first, axis=-1, keepdims=True)
        second = rows[..., 1, :]
        second = second - np.sum(first * second, axis=-1, keepdims=True) * first
        norm = np.linalg.norm(second, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Degenerate rotation_6d: rows must be nonzero and independent")
        second = second / norm
        return np.stack((first, second, np.cross(first, second)), axis=-2)
