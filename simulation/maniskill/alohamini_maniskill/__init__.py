"""ManiSkill robot and task registration."""

from alohamini.simulation.runtime import check_backend_runtime

check_backend_runtime("alohamini-maniskill")

# Validate the installed SDK before importing SAPIEN and registering the agent.
from .environment import AlohaMini2Pro, AlohaMiniResearchEnv  # noqa: E402

__all__ = ["AlohaMiniResearchEnv", "AlohaMini2Pro"]
