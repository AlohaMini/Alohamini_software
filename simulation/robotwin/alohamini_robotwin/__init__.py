"""RoboTwin embodiment loading and SI control."""

from alohamini.simulation.runtime import check_backend_runtime

check_backend_runtime("alohamini-robotwin")

from .robot import create_robot  # noqa: E402

__all__ = ["create_robot"]
