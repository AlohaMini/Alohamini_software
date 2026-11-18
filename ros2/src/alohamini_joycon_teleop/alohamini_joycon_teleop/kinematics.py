"""ROS tracking adapter over the platform Pink/Pinocchio implementation."""

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
    """Feedback tracking shares the platform Pink/Pinocchio backend."""

    def step(self, joints, target, dt, *, qp_solver="daqp", **settings):
        client = getattr(self, "ik_client", None)
        if client is not None:
            return client.step(self, joints, target, dt, qp_solver=qp_solver, **settings)
        return self.pink_solver(qp_solver=qp_solver).step(joints, target, dt, **settings)

    def reset_tracking(self):
        client = getattr(self, "ik_client", None)
        if client is not None:
            client.reset(self)
        solver = getattr(self, "_pink_solver", None)
        if solver is not None:
            solver.reset()
