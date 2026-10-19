"""Read the authoritative robot tree without inferring frames from vector widths."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .frames import rigid_transform, rotation_axis, transform, vector


def numbers(text, size, label):
    return vector([float(x) for x in text.split()], size, label)


def origin(element):
    node = element.find("origin")
    if node is None:
        return np.eye(4)
    return transform(
        numbers(node.get("xyz", "0 0 0"), 3, "origin xyz"),
        numbers(node.get("rpy", "0 0 0"), 3, "origin rpy"),
    )


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    T_parent_joint: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    damping: float
    friction: float

    def motion(self, position):
        if not math.isfinite(position):
            raise ValueError(f"Nonfinite position for {self.name}")
        value = np.eye(4)
        if self.kind in ("revolute", "continuous"):
            value[:3, :3] = rotation_axis(self.axis, position)
        elif self.kind == "prismatic":
            value[:3, 3] = self.axis * position
        elif position != 0:
            raise ValueError(f"Fixed joint {self.name} has a nonzero position")
        return value

    def check_position(self, value):
        if not math.isfinite(value) or (
            self.lower is not None and not self.lower <= value <= self.upper
        ):
            raise ValueError(f"Position outside URDF limits: {self.name}={value}")


class Description:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.path = robot_model.description_path("collision")
        self.tree = ET.parse(self.path).getroot()
        self.links = {link.get("name"): link for link in self.tree.findall("link")}
        if len(self.links) != len(self.tree.findall("link")) or None in self.links:
            raise ValueError("URDF must contain unique named links")
        self.joints = {}
        children = set()
        self.by_parent = {}
        for node in self.tree.findall("joint"):
            name, kind = node.get("name"), node.get("type")
            if (
                not name
                or name in self.joints
                or kind not in ("fixed", "revolute", "continuous", "prismatic")
            ):
                raise ValueError("URDF contains an unsupported or duplicate joint")
            if node.find("mimic") is not None:
                raise ValueError(f"Mimic joints need an explicit coupling: {name}")
            parent, child = node.find("parent").get("link"), node.find("child").get("link")
            if parent not in self.links or child not in self.links or child in children:
                raise ValueError(f"Invalid joint tree at {name}")
            children.add(child)
            axis = np.array([1.0, 0, 0])
            lower = upper = effort = velocity = None
            if kind != "fixed":
                axis_node = node.find("axis")
                axis = numbers(
                    axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0",
                    3,
                    f"{name} axis",
                )
                norm = np.linalg.norm(axis)
                if norm < 1e-12:
                    raise ValueError(f"Zero axis at {name}")
                axis /= norm
                limit = node.find("limit")
                if kind != "continuous":
                    if limit is None:
                        raise ValueError(f"Missing limits: {name}")
                    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                    effort, velocity = float(limit.get("effort")), float(limit.get("velocity"))
                    if not np.isfinite([lower, upper, effort, velocity]).all() or not (
                        lower < upper and effort > 0 and velocity > 0
                    ):
                        raise ValueError(f"Invalid limits: {name}")
            dynamics = node.find("dynamics")
            damping = float(dynamics.get("damping", 0)) if dynamics is not None else 0.0
            friction = float(dynamics.get("friction", 0)) if dynamics is not None else 0.0
            if not np.isfinite([damping, friction]).all() or min(damping, friction) < 0:
                raise ValueError(f"Invalid passive dynamics: {name}")
            joint = Joint(
                name,
                kind,
                parent,
                child,
                origin(node),
                axis,
                lower,
                upper,
                effort,
                velocity,
                damping,
                friction,
            )
            self.joints[name] = joint
            self.by_parent.setdefault(parent, []).append(joint)
        roots = self.links.keys() - children
        if len(roots) != 1:
            raise ValueError("URDF must contain exactly one root")
        self.root = next(iter(roots))
        if len(self.forward()) != len(self.links):
            raise ValueError("URDF contains a disconnected tree")
        self.inertials = {}
        for name, link in self.links.items():
            node = link.find("inertial")
            if node is None:
                continue
            mass = float(node.find("mass").get("value"))
            i = node.find("inertia")
            xx, yy, zz, xy, xz, yz = [
                float(i.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ]
            inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            if not np.isfinite(inertia).all() or not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"Invalid inertial data: {name}")
            principal = np.linalg.eigvalsh(inertia)
            if principal[0] <= 0 or principal[2] > principal[0] + principal[1] + 1e-12:
                raise ValueError(f"Nonphysical inertia tensor: {name}")
            self.inertials[name] = (mass, origin(node), inertia)
        semantic = ET.parse(robot_model.description_path("semantic")).getroot()
        self.exclusions = []
        for node in semantic.findall("disable_collisions"):
            a, b = node.get("link1"), node.get("link2")
            if a not in self.links or b not in self.links or a == b:
                raise ValueError("Invalid SRDF collision exclusion")
            self.exclusions.append((a, b, node.get("reason", "unspecified")))
        self.home = {}
        for state in semantic.findall("group_state"):
            if state.get("name") not in ("home", "open"):
                continue
            for joint in state.findall("joint"):
                name, value = joint.get("name"), float(joint.get("value"))
                if name not in self.joints:
                    raise ValueError(f"Unknown SRDF joint: {name}")
                self.joints[name].check_position(value)
                if name in self.home and self.home[name] != value:
                    raise ValueError(f"Conflicting SRDF initial values: {name}")
                self.home[name] = value

    def mesh_path(self, mesh):
        relative = mesh.get("filename", "")
        if not relative or ":" in relative or Path(relative).is_absolute():
            raise ValueError("Model mesh references must be relative paths")
        path = (self.path.parent / relative).resolve()
        if not path.is_relative_to(self.robot_model.directory) or not path.is_file():
            raise ValueError(f"Missing or external mesh: {relative}")
        return path

    def forward(self, positions=None, T_world_root=None):
        positions = {} if positions is None else positions
        unknown = positions.keys() - self.joints.keys()
        if unknown:
            raise ValueError(f"Unknown joint positions: {sorted(unknown)}")
        poses = {self.root: rigid_transform(np.eye(4) if T_world_root is None else T_world_root)}
        visiting = set()

        def visit(parent):
            if parent in visiting:
                raise ValueError("Cycle in URDF")
            visiting.add(parent)
            for joint in self.by_parent.get(parent, []):
                poses[joint.child] = (
                    poses[parent]
                    @ joint.T_parent_joint
                    @ joint.motion(positions.get(joint.name, 0))
                )
                visit(joint.child)
            visiting.remove(parent)

        visit(self.root)
        return poses
