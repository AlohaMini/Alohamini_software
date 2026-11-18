# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_ros2/alohamini_validation; native model asset paths.
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from alohamini.model import RobotModel, get_robot_model
from alohamini.simulation.description import Description

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"expected YAML mapping: {path}")
    return value


def rpy_matrix(rpy: str) -> np.ndarray:
    roll, pitch, yaw = np.fromstring(rpy, sep=" ")
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


def axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, one_minus_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ]
    )


def origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    transform = np.eye(4)
    if origin is not None:
        transform[:3, :3] = rpy_matrix(origin.get("rpy", "0 0 0"))
        transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
    return transform


class UrdfKinematics:
    def __init__(self, path: Path) -> None:
        self.root = path if isinstance(path, ET.Element) else ET.parse(path).getroot()
        self.joint_by_child = {
            joint.find("child").get("link"): joint for joint in self.root.findall("joint")
        }

    def chain(self, base: str, tip: str) -> list[ET.Element]:
        result = []
        current = tip
        visited = set()
        while current != base:
            if current in visited:
                raise AssertionError(f"cycle in chain from {base} to {tip}: {current}")
            visited.add(current)
            if current not in self.joint_by_child:
                raise AssertionError(f"{tip} is not downstream of {base}")
            joint = self.joint_by_child[current]
            result.append(joint)
            current = joint.find("parent").get("link")
        return list(reversed(result))

    def fk(self, base: str, tip: str, positions: dict[str, float]) -> np.ndarray:
        transform = np.eye(4)
        for joint in self.chain(base, tip):
            transform = transform @ origin_transform(joint)
            value = float(positions.get(joint.get("name"), 0.0))
            motion = np.eye(4)
            if joint.get("type") in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:3, :3] = axis_angle(axis, value)
            elif joint.get("type") == "prismatic":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                motion[:3, 3] = axis * value
            transform = transform @ motion
        return transform


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [[c, -s * ca, s * sa, a * c], [s, c * ca, -c * sa, a * s], [0, sa, ca, d], [0, 0, 0, 1]]
    )


def dh_fk(config: dict, q_rad: list[float]) -> np.ndarray:
    standard = config["standard_dh"]
    transform = np.asarray(standard["base_transform"], dtype=float)
    for row, value in zip(standard["rows"], q_rad, strict=True):
        transform = transform @ dh_transform(
            float(row["a"]),
            float(row["alpha"]),
            float(row["d"]),
            float(value) + float(row["theta_offset"]),
        )
    return transform @ np.asarray(standard["tool_transform"], dtype=float)


def validate_tree(model: UrdfKinematics) -> None:
    """Check the generated planar-base topology, including its virtual root."""
    links = {link.get("name") for link in model.root.findall("link")}
    joints = model.root.findall("joint")
    children = [joint.find("child").get("link") for joint in joints]
    parents = [joint.find("parent").get("link") for joint in joints]
    assert len(links) == 32 and len(joints) == 31
    assert len(children) == len(set(children))
    assert links - set(children) == {"root"}
    assert set(children) <= links and set(parents) <= links
    assert {"wheel1", "wheel2", "wheel3"} <= links
    assert not {"Link2_dp", "Link3_dp", "Link4_dp"} & links
    joints_by_name = {joint.get("name"): joint for joint in joints}
    cad_joint = joints_by_name["base_cad_joint"]
    assert cad_joint.get("type") == "fixed"
    assert cad_joint.find("parent").get("link") == "base_link"
    assert cad_joint.find("child").get("link") == "base_cad_link"
    assert np.allclose(
        np.fromstring(cad_joint.find("origin").get("xyz"), sep=" "),
        [0.0, 0.0, 0.0],
    )
    assert np.allclose(
        np.fromstring(cad_joint.find("origin").get("rpy"), sep=" "),
        [0.0, 0.0, math.pi / 2.0],
    )
    cad_rotation = origin_transform(cad_joint)[:3, :3]
    assert np.allclose(cad_rotation @ [0.0, -1.0, 0.0], [1.0, 0.0, 0.0])
    assert np.allclose(cad_rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    for wheel in ("wheel1", "wheel2", "wheel3"):
        joint = joints_by_name[f"{wheel}_joint"]
        assert joint.get("type") == "continuous"
        assert joint.find("parent").get("link") == "base_cad_link"
        assert joint.find("child").get("link") == wheel


def validate_fk(model: UrdfKinematics, description: Path, validation: Path) -> None:
    golden = load_yaml(validation / "config/fk_golden.yaml")
    dh = load_yaml(description / "config/kinematics/right_arm_kinematics.yaml")
    tool = load_yaml(description / "config/kinematics/kinematics.yaml")["tool_frames"]
    fixed_to_tcp = np.eye(4)
    fixed_to_tcp[:3, :3] = np.asarray(tool["delta_matrix"], dtype=float)
    fixed_to_tcp[:3, 3] = fixed_to_tcp[:3, :3] @ np.asarray(tool["tcp_tool_m"], dtype=float)
    tolerance = float(golden["tolerance"])
    for sample_name, sample in golden["samples"].items():
        q_rad = [float(value) for value in sample["q_rad"]]
        expected = np.asarray(sample["transform"], dtype=float)
        results = []
        for side in ("left", "right"):
            positions = {
                f"{side}_{joint}": value for joint, value in zip(ARM_JOINTS, q_rad, strict=True)
            }
            actual = model.fk(f"{side}_Base", f"{side}_tcp", positions)
            assert np.allclose(actual, expected, atol=tolerance), f"{sample_name}/{side} FK drift"
            results.append(actual)
        assert np.allclose(results[0], results[1], atol=tolerance)
        assert np.allclose(dh_fk(dh, q_rad) @ fixed_to_tcp, expected, atol=tolerance), (
            f"{sample_name} DH drift"
        )


def resolve_package_uri(uri: str, description: Path) -> Path:
    path = (description / "urdf" / uri).resolve()
    assert path.is_relative_to(description.resolve()) and path.is_file(), uri
    return path


def validate_physical_assets(robot: RobotModel) -> None:
    """Validate canonical physical assets without applying the planning baseline."""
    # Use the same checksum, tree, inertia and semantic rules as native consumers.
    physical = Description(robot, name="collision")
    manifest = physical.prepared
    assert manifest is not None, "Canonical physical assets require a physical model manifest"
    assert physical.root == manifest.get("base_frame") == "base_link", (
        "Physical URDF must use base_link as its root"
    )
    assert not {"root", "root_x_link", "root_y_link"} & physical.links.keys(), (
        "Physical URDF must not include the planning virtual root"
    )
    for name in ("collision", "semantic"):
        relative = robot.description_path(name).relative_to(robot.directory).as_posix()
        assert relative in manifest["files"], f"Physical {name} is not covered by its manifest"
    for mesh in physical.tree.findall(".//mesh"):
        relative = physical.mesh_path(mesh).relative_to(robot.directory).as_posix()
        assert relative in manifest["files"], (
            f"Physical mesh is not covered by its manifest: {relative}"
        )

    semantic = ET.parse(physical.semantic_path).getroot()
    assert semantic.get("name") == physical.tree.get("name"), "Physical URDF/SRDF names differ"
    kinematics = UrdfKinematics(physical.path)
    for chain in semantic.findall(".//chain"):
        base, tip = chain.get("base_link"), chain.get("tip_link")
        assert base in physical.links and tip in physical.links, "Unknown physical SRDF chain link"
        kinematics.chain(base, tip)
    for joint in (*semantic.findall(".//joint"), *semantic.findall("passive_joint")):
        assert joint.get("name") in physical.joints, "Unknown physical SRDF joint"


def validate_derived_views(robot: RobotModel) -> None:
    """Every physical element and semantic rule must have exactly one source."""
    source = robot.description_tree("collision")
    planning = robot.description_tree("planning")
    for tag in ("link", "joint", "material"):
        actual = {node.get("name"): ET.tostring(node) for node in planning.findall(tag)}
        for node in source.findall(tag):
            assert actual.get(node.get("name")) == ET.tostring(node), (
                f"Derived {tag} differs from physical source: {node.get('name')}"
            )
    assert robot.description_path("semantic") == robot.description_path("semantic_planning")
    assert not any(
        robot.description_tree("kinematic").findall(".//" + tag)
        for tag in ("visual", "collision", "inertial")
    ), "Kinematic view must contain only frames and joints"


def validate_collision_baseline(robot: RobotModel, validation: Path) -> None:
    """Bind the reviewed self-collision samples to the unified physical assets."""
    model = robot.description_tree("collision")
    semantic = robot.description_tree("semantic")
    baseline = load_yaml(validation / "config/collision_baseline.yaml")
    assert baseline.get("schema_version") == 2, "Unknown collision baseline schema"
    for relative, expected in baseline["source_description_sha256"].items():
        assert hashlib.sha256(robot.asset_path(relative).read_bytes()).hexdigest() == expected, (
            f"Collision baseline source changed: {relative}"
        )
    actual = {
        "links": len(model.findall("link")), "joints": len(model.findall("joint")),
        "visuals": len(model.findall(".//visual")),
        "collisions": len(model.findall(".//collision")),
        "collision_boxes": len(model.findall(".//collision/geometry/box")),
        "collision_cylinders": len(model.findall(".//collision/geometry/cylinder")),
        "collision_meshes": len(model.findall(".//collision/geometry/mesh")),
        "srdf_disabled_pairs": len(semantic.findall("disable_collisions")),
    }
    assert actual == baseline["geometry_counts"], f"collision baseline drift: {actual}"
    for mesh in model.findall(".//mesh"):
        resolve_package_uri(mesh.get("filename"), robot.directory)


def main() -> None:
    robot = get_robot_model("alohamini2pro")
    description = robot.directory
    validation = Path(get_package_share_directory("alohamini_validation"))
    models = {
        name: UrdfKinematics(robot.description_tree(name))
        for name in ("collision", "planning", "kinematic")
    }
    checks = (
        ("physical manifest + tree", lambda: validate_physical_assets(robot)),
        ("planning URDF tree", lambda: validate_tree(models["planning"])),
        ("kinematic URDF tree", lambda: validate_tree(models["kinematic"])),
        *(
            (
                f"{name} FK golden + DH",
                lambda model=model: validate_fk(model, description, validation),
            )
            for name, model in models.items()
        ),
        ("derived view identity", lambda: validate_derived_views(robot)),
        ("shared collision baseline", lambda: validate_collision_baseline(robot, validation)),
    )
    for name, check in checks:
        check()
        print(f"[PASS] {name}")
    print(
        "[INFO] Run validate_moveit on the active graph to check the named self-collision samples."
    )


if __name__ == "__main__":
    main()
