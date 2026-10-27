"""Pinned LIBERO scene snapshots and explicit AlohaMini coordinate adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path

import mujoco
import numpy as np

from .compilation import compile_xml
from .frames import rigid_transform
from .model import pose_attributes, text
from .native_distance import InvalidNativeDistance, checked_geom_distance

LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_EXPORT_MUJOCO_VERSION = "3.8.1"
SPATIAL_TASKS = tuple(range(10))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_revision(source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    if git("rev-parse", "HEAD") != LIBERO_COMMIT or git("status", "--porcelain"):
        raise ValueError(f"LIBERO source must be a clean checkout of {LIBERO_COMMIT}")


def import_libero(source, python, output, *, tasks=SPATIAL_TASKS):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or any(type(t) is not int or not 0 <= t < 10 for t in tasks)
    ):
        raise ValueError("Spatial task indices must be unique integers in [0, 9]")
    _source_revision(source)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        subprocess.run(
            [
                str(Path(python).expanduser().resolve()),
                "-I",
                str(Path(__file__).with_name("_libero_worker.py")),
                "--source",
                str(source),
                "--output",
                str(stage),
                "--suite",
                "libero_spatial",
                "--tasks",
                *map(str, tasks),
            ],
            check=True,
        )
        packages = json.loads((stage / "upstream.json").read_text())["upstream_packages"]
        if packages["mujoco"] != LIBERO_EXPORT_MUJOCO_VERSION or packages["robosuite"] != "1.4.0":
            raise ValueError("Upstream export requires MuJoCo 3.8.1 and robosuite 1.4.0")
        _source_revision(source)
        manifest = {
            "schema_version": 2,
            "upstream": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "source_commit": LIBERO_COMMIT,
            "upstream_packages": packages,
            "suite": "libero_spatial",
            "tasks": list(tasks),
            "files": {
                p.relative_to(stage).as_posix(): sha256(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            },
        }
        (stage / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return output


@dataclass
class LiberoScene:
    directory: Path
    task_index: int
    configuration: dict
    reset_index: int = 0

    def __post_init__(self):
        self.directory = Path(self.directory).expanduser().resolve()
        manifest_path = self.directory / "bundle.json"
        self.bundle = json.loads(manifest_path.read_text())
        if (
            self.bundle.get("schema_version") != 2
            or self.bundle.get("source_commit") != LIBERO_COMMIT
        ):
            raise ValueError("Unsupported LIBERO scene bundle")
        for relative, expected in self.bundle["files"].items():
            path = (self.directory / relative).resolve()
            if not path.is_relative_to(self.directory) or sha256(path) != expected:
                raise ValueError(f"Scene bundle checksum mismatch: {relative}")
        if type(self.task_index) is not int or self.task_index not in self.bundle["tasks"]:
            raise ValueError("Task is not in this scene bundle")
        self.folder = self.directory / f"task_{self.task_index:03d}"
        self.task = json.loads((self.folder / "task.json").read_text())
        if type(self.reset_index) is not int or not 0 <= self.reset_index < len(
            self.task["initial_states"]
        ):
            raise ValueError("Initial state index is outside this task's bank")
        self.T_world_libero = rigid_transform(self.configuration["T_world_libero"])
        # Upright support surfaces and On predicates use the world's vertical.
        if not np.allclose(self.T_world_libero[:3, 2], [0, 0, 1], atol=1e-12):
            raise ValueError("LIBERO adaptation must preserve the vertical axis")
        self.fingerprint = sha256(manifest_path)
        self.configuration = copy.deepcopy(self.configuration)
        upstream_packages = None
        if "upstream.json" in self.bundle["files"]:
            upstream_packages = json.loads((self.directory / "upstream.json").read_text())[
                "upstream_packages"
            ]
            if (
                not isinstance(upstream_packages, dict)
                or upstream_packages.get("mujoco") != LIBERO_EXPORT_MUJOCO_VERSION
                or upstream_packages.get("robosuite") != "1.4.0"
                or self.bundle.get("upstream_packages", upstream_packages) != upstream_packages
            ):
                raise ValueError("LIBERO snapshot has an incompatible export environment")
        self.manifest = {
            "benchmark": "alohamini-libero-spatial-v1",
            "protocol": "robot adaptation; scores are not original Panda LIBERO scores",
            "source_commit": LIBERO_COMMIT,
            "upstream_packages": upstream_packages,
            "bundle_sha256": self.fingerprint,
            "suite": self.task["suite"],
            "task_index": self.task_index,
            "task_name": self.task["task"]["name"],
            "language": self.task["task"]["language"],
            "source_bddl_sha256": self.task["source_bddl_sha256"],
            "source_init_sha256": self.task["source_init_sha256"],
            "reset_index": self.reset_index,
            "configuration": self.configuration,
            "initial_state_policy": (
                "named object joints only; robot state independently configured; zero velocities"
            ),
            "physical_parameters": (
                "unchanged upstream simulation values; not real-object calibration"
            ),
            "upstream_physics": {
                "timestep_s": self.task["source_timestep_s"],
                "gravity_m_s2": self.task["source_gravity_m_s2"],
            },
            "goals": self.task["parsed_problem"]["goal_state"],
        }
        for goal in self.manifest["goals"]:
            if (
                len(goal) != 3
                or goal[0].lower() != "on"
                or any(n not in self.task["objects"] for n in goal[1:])
            ):
                raise ValueError(f"Unsupported goal; no predicate approximation allowed: {goal}")

    def augment(self, root, assets):
        scene = ET.parse(self.folder / "scene.xml").getroot()
        existing = {n.get("name") for n in root.iter() if n.get("name")}
        names = {n.get("name") for n in scene.iter() if n.get("name")}
        if existing & names:
            raise ValueError(f"Robot/scene name collision: {sorted(existing & names)}")
        default = scene.find("default")
        if default is not None:
            root_default = root.find("default")
            if root_default is None:
                root_default = ET.SubElement(root, "default")
            root_default.append(default)
        for node in scene.find("asset"):
            filename = node.get("file")
            if filename:
                if Path(filename).name != filename:
                    raise ValueError("Scene assets must use flat, relative filenames")
                contents = (self.directory / "assets" / filename).read_bytes()
                if filename in assets and assets[filename] != contents:
                    raise ValueError("Robot/scene asset filename collision")
                assets[filename] = contents
            root.find("asset").append(node)
        for node in scene.find("worldbody"):
            # Top-level free bodies must remain children of worldbody. Bake one
            # rigid transform into every world entity; never scale task geometry.
            p = np.fromstring(node.get("pos", "0 0 0"), sep=" ")
            q = np.fromstring(node.get("quat", "1 0 0 0"), sep=" ")
            q /= np.linalg.norm(q)
            R = np.empty(9)
            mujoco.mju_quat2Mat(R, q)
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = R.reshape(3, 3), p
            node.attrib.pop("quat", None)
            node.attrib.update(pose_attributes(self.T_world_libero @ transform))
            if node.tag == "light":
                # Lights use direction rather than orientation attributes.
                node.attrib.pop("quat", None)
                direction = np.fromstring(node.get("dir", "0 0 -1"), sep=" ")
                node.set("dir", text(self.T_world_libero[:3, :3] @ direction))
            root.find("worldbody").append(node)
        for camera in self.configuration.get("cameras", []):
            body = root.find(f".//body[@name='{camera['body']}']")
            if body is None:
                raise ValueError(f"Unknown camera parent: {camera['body']}")
            ET.SubElement(
                body,
                "camera",
                name=camera["name"],
                fovy=str(camera["fovy_deg"]),
                **pose_attributes(rigid_transform(camera["T_body_camera"])),
            )
        support = self.configuration.get("robot_support")
        if support is not None:
            size = np.asarray(support["size_m"], dtype=float)
            if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
                raise ValueError("Robot support needs three positive dimensions in metres")
            body = ET.SubElement(
                root.find("worldbody"),
                "body",
                name="alohamini_support",
                **pose_attributes(rigid_transform(support["T_world_body"])),
            )
            for kind, group in [("visual", "2"), ("collision", "3")]:
                ET.SubElement(
                    body,
                    "geom",
                    name=f"alohamini_support_{kind}",
                    type="box",
                    size=text(size / 2),
                    group=group,
                    mass="0",
                    contype="0" if kind == "visual" else "1",
                    conaffinity="0" if kind == "visual" else "1",
                    rgba="0.25 0.25 0.28 1" if kind == "visual" else "0.1 0.8 0.3 0.3",
                )

    def attach_to(self, robot):
        """Reuse immutable robot mesh bytes when compiling a suite of task scenes."""
        if robot.scene is not None:
            raise ValueError("Attach a scene to a robot-only model")
        root, assets = ET.fromstring(robot.xml), dict(robot.assets)
        self.augment(root, assets)
        xml = ET.tostring(root, encoding="unicode")
        model = compile_xml(xml, assets=assets)
        return replace(
            robot,
            model=model,
            xml=xml,
            assets=assets,
            manifest={**robot.manifest, "scene": self.manifest},
            scene=self,
        )

    def initialize(self, robot, data):
        for name, values in self.task["initial_states"][self.reset_index].items():
            joint = robot.model.joint(name)
            address = robot.model.jnt_qposadr[joint.id]
            if robot.model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_FREE:
                q = np.asarray(values, dtype=float)
                T = np.eye(4)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, q[3:])
                T[:3, :3], T[:3, 3] = rotation.reshape(3, 3), q[:3]
                attributes = pose_attributes(self.T_world_libero @ T)
                data.qpos[address : address + 7] = np.fromstring(
                    attributes["pos"] + " " + attributes["quat"], sep=" "
                )
            else:
                if len(values) != 1:
                    raise ValueError(f"Unsupported object joint: {name}")
                data.qpos[address] = values[0]

    def goal_status(self, robot, data):
        """Exact upstream object On: contact, target below source, XY distance < 3 cm."""
        statuses = []
        for _, source, target in self.manifest["goals"]:
            source_object, target_object = (self.task["objects"][n] for n in (source, target))
            a, b = (data.body(obj["root_body"]).xpos for obj in (source_object, target_object))
            source_geoms = {robot.model.geom(n).id for n in source_object["contact_geoms"]}
            target_geoms = {robot.model.geom(n).id for n in target_object["contact_geoms"]}
            contact = any(
                (c.geom1 in source_geoms and c.geom2 in target_geoms)
                or (c.geom2 in source_geoms and c.geom1 in target_geoms)
                for c in data.contact
            )
            distance = float(np.linalg.norm(a[:2] - b[:2]))
            statuses.append(
                {
                    "source": source,
                    "target": target,
                    "contact": contact,
                    "xy_distance_m": distance,
                    "passed": bool(contact and a[2] >= b[2] and distance < 0.03),
                }
            )
        return {"passed": all(s["passed"] for s in statuses), "predicates": statuses}

    def validate(self, robot):
        errors = []
        maximum_position_error, maximum_size_error, maximum_rotation_error = 0.0, 0.0, 0.0
        for name, expected in self.task["body_inertials"].items():
            body = robot.model.body(name)
            for key, value in expected.items():
                if expected["mass"] == 0 and key in ("ipos", "iquat"):
                    # A massless fixed frame has no COM or inertial orientation.
                    # Compilers may fill these unused fields differently.
                    continue
                if not np.allclose(np.atleast_1d(getattr(body, key)), value, rtol=0, atol=1e-12):
                    errors.append(f"Body {name}: {key}")
        for name, expected in self.task["contact_parameters"].items():
            geom = robot.model.geom(name)
            for key, value in expected.items():
                if not np.allclose(np.atleast_1d(getattr(geom, key)), value, rtol=0, atol=1e-12):
                    errors.append(f"Geom {name}: {key}")
        for name, expected in self.task["geom_geometry"].items():
            geom = robot.model.geom(name)
            target, actual = np.eye(4), np.eye(4)
            for transform, position, quaternion in [
                (target, expected["pos"], expected["quat"]),
                (actual, geom.pos, geom.quat),
            ]:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float))
                transform[:3, :3], transform[:3, 3] = rotation.reshape(3, 3), position
            if expected["body"] == "world":
                target = self.T_world_libero @ target
            maximum_position_error = max(
                maximum_position_error, float(np.max(np.abs(target[:3, 3] - actual[:3, 3])))
            )
            maximum_rotation_error = max(
                maximum_rotation_error, float(np.max(np.abs(target[:3, :3] - actual[:3, :3])))
            )
            maximum_size_error = max(
                maximum_size_error, float(np.max(np.abs(geom.size - expected["size"])))
            )
            if (
                geom.type[0] != expected["type"][0]
                or robot.model.body(geom.bodyid[0]).name != expected["body"]
            ):
                errors.append(f"Geom {name}: type or parent body changed")
        if (
            maximum_position_error > 1e-6
            or maximum_size_error > 1e-6
            or maximum_rotation_error > 5e-6
        ):
            errors.append("Scene geometry changed beyond canonical XML precision tolerance")
        return {
            "passed": not errors,
            "scope": (
                "upstream object inertia and collision-parameter equivalence; not task feasibility"
            ),
            "errors": errors,
            "maximum_geom_position_error_m": maximum_position_error,
            "maximum_geom_size_error_m": maximum_size_error,
            "maximum_geom_rotation_matrix_error": maximum_rotation_error,
            "initial_goal": self.goal_status(robot, robot.data()),
            "object_initial_state_count": len(self.task["initial_states"]),
            "static_robot_scene": self.static_intersections(robot),
        }

    def static_intersections(self, robot, *, tolerance_m=0.0005):
        """MuJoCo skips same-weld fixed bodies; inspect their real shapes explicitly."""
        model, data = robot.model, robot.data()
        robot_ids = [
            model.geom(n).id
            for n, g in robot.geometry.items()
            if g["kind"] == "collision" and model.body_weldid[model.geom(n).bodyid[0]] == 0
        ]
        scene_ids = [
            model.geom(n).id
            for n in self.task["contact_parameters"]
            if (model.geom(n).contype[0] or model.geom(n).conaffinity[0])
            and model.body_weldid[model.geom(n).bodyid[0]] == 0
        ]
        if self.configuration.get("robot_support") is not None:
            scene_ids.append(model.geom("alohamini_support_collision").id)
        intersections, invalid_queries = [], []
        for a in robot_ids:
            for b in scene_ids:
                try:
                    distance, _ = checked_geom_distance(model, data, a, b, tolerance_m)
                except InvalidNativeDistance as exc:
                    invalid_queries.append(
                        {
                            "robot_geom": model.geom(a).name,
                            "scene_geom": model.geom(b).name,
                            "error": str(exc),
                        }
                    )
                    continue
                if distance < -tolerance_m:
                    intersections.append(
                        {
                            "robot_geom": model.geom(a).name,
                            "scene_geom": model.geom(b).name,
                            "distance_m": float(distance),
                        }
                    )
        return {
            "passed": not intersections and not invalid_queries,
            "tolerance_m": tolerance_m,
            "intersections": intersections,
            "invalid_distance_queries": invalid_queries,
        }
