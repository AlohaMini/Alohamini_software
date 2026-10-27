"""ROS teleoperation tracking built on shared robot geometry."""

import numpy as np

from alohamini.kinematics.arm import (
    ARM_SUFFIXES,
    _rotation_vector,
    _scale_to_max_abs,
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from alohamini.kinematics.arm import (
    AlohaMiniArmKinematics as ArmGeometry,
)

__all__ = [
    "ARM_SUFFIXES",
    "AlohaMiniArmKinematics",
    "_rotation_vector",
    "_scale_to_max_abs",
    "matrix_to_quaternion",
    "quaternion_to_matrix",
]


class AlohaMiniArmKinematics(ArmGeometry):
    """Periodic damped tracking with teleoperation gains and velocity limits."""

    def step(
        self,
        joints: list[float] | np.ndarray,
        target: np.ndarray,
        dt: float,
        *,
        position_gain: float = 8.0,
        orientation_gain: float = 5.0,
        orientation_weight: float = 0.35,
        damping: float = 0.04,
        max_joint_velocity: float = 1.5,
        max_joint_step: float = 0.08,
        joint_limit_margin: float = 0.03,
    ) -> tuple[np.ndarray, dict[str, float | list[int]]]:
        joints = np.asarray(joints, dtype=float)
        target = np.asarray(target, dtype=float)
        current = self.forward(joints)
        position_error = target[:3, 3] - current[:3, 3]
        orientation_error = _rotation_vector(target[:3, :3] @ current[:3, :3].T)
        jacobian = self.jacobian(joints)
        weighted_jacobian = jacobian.copy()
        weighted_jacobian[3:] *= orientation_weight
        twist = np.concatenate(
            (position_gain * position_error, orientation_gain * orientation_error)
        )
        twist[3:] *= orientation_weight
        singular_values = np.linalg.svd(weighted_jacobian, compute_uv=False)
        minimum_singular = float(singular_values[-1])
        adaptive_damping = damping * (1.0 + max(0.0, 0.08 - minimum_singular) / 0.08 * 4.0)
        regularizer = (adaptive_damping * adaptive_damping) * np.eye(6)
        velocity = weighted_jacobian.T @ np.linalg.solve(
            weighted_jacobian @ weighted_jacobian.T + regularizer,
            twist,
        )
        # Scale the complete DLS vector instead of clipping joints
        # independently. Component-wise saturation changes the joint-space
        # direction whenever a different joint reaches the bound, which shows
        # up as a small Cartesian kink during continuous teleoperation.
        velocity = _scale_to_max_abs(velocity, max_joint_velocity)
        delta = _scale_to_max_abs(velocity * max(0.0, float(dt)), max_joint_step)
        candidate = joints + delta
        joint_limit_hits: list[int] = []
        if self.joint_limits is not None:
            lower = np.asarray([limit[0] for limit in self.joint_limits])
            upper = np.asarray([limit[1] for limit in self.joint_limits])
            margin = max(0.0, float(joint_limit_margin))
            safe_lower = lower + margin
            safe_upper = upper - margin
            for index, (joint, proposed_delta) in enumerate(zip(joints, delta, strict=True)):
                if proposed_delta < 0.0:
                    if joint <= safe_lower[index]:
                        candidate[index] = joint
                        joint_limit_hits.append(index)
                    elif candidate[index] < safe_lower[index]:
                        candidate[index] = safe_lower[index]
                        joint_limit_hits.append(index)
                elif proposed_delta > 0.0:
                    if joint >= safe_upper[index]:
                        candidate[index] = joint
                        joint_limit_hits.append(index)
                    elif candidate[index] > safe_upper[index]:
                        candidate[index] = safe_upper[index]
                        joint_limit_hits.append(index)
        return candidate, {
            "position_error_m": float(np.linalg.norm(position_error)),
            "orientation_error_rad": float(np.linalg.norm(orientation_error)),
            "minimum_singular_value": minimum_singular,
            "damping": adaptive_damping,
            "max_joint_step_rad": float(np.max(np.abs(candidate - joints))),
            "joint_limit_hits": joint_limit_hits,
        }
