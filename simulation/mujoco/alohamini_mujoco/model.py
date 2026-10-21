"""Derive MJCF from named URDF frames, explicit inertias and collision pieces."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import mujoco
import numpy as np

from alohamini.model import get_robot_model

from .collision import load_bundle
from .description import Description, numbers, origin
from .frames import rigid_transform
from .meshes import binary_stl

PARKED_JOINTS = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "wheel1_joint",
    "wheel2_joint",
    "wheel3_joint",
)


def text(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


def pose_attributes(pose):
    pose = rigid_transform(pose)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(pose[:3, :3]).ravel())
    return {"pos": text(pose[:3, 3]), "quat": text(quaternion)}


@dataclass(frozen=True)
class ModelProfile:
    # A parked-base manipulation fixture. The complete robot geometry remains
    # present; this is not a mobile-base dynamics model.
    locked_positions: dict = field(default_factory=lambda: dict.fromkeys(PARKED_JOINTS, 0.0))
    T_world_root: np.ndarray = field(default_factory=lambda: np.eye(4))
    initial_positions: dict = field(default_factory=dict)
    timestep: float = 0.001
    gravity: tuple = (0.0, 0.0, -9.80665)
    friction: tuple = (1.0, 0.005, 0.0001)
    collision_profile: str = "authoritative"
    collision_bundle: Path | None = None

    def __post_init__(self):
        for name in ("locked_positions", "initial_positions"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} must be a mapping of named joint coordinates")
            if any(
                not isinstance(k, str)
                or isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for k, v in value.items()
            ):
                raise ValueError(f"{name} must contain finite numeric joint coordinates")
            object.__setattr__(self, name, MappingProxyType(dict(value)))
        for name, value in self.initial_positions.items():
            if name in self.locked_positions and value != self.locked_positions[name]:
                raise ValueError(f"Initial position conflicts with locked joint: {name}")
        pose = rigid_transform(self.T_world_root)
        pose.flags.writeable = False
        object.__setattr__(self, "T_world_root", pose)


@dataclass
class RobotModel:
    description: Description
    model: mujoco.MjModel
    profile: ModelProfile
    xml: str
    assets: dict
    meshes: dict
    geometry: dict
    manifest: dict
    initial_positions: dict
    scene: object | None = None

    def data(self, positions=None):
        data = mujoco.MjData(self.model)
        values = {**self.initial_positions, **(positions or {})}
        unknown = values.keys() - self.description.joints.keys()
        if unknown:
            raise ValueError(f"Unknown joints: {sorted(unknown)}")
        for name, position in values.items():
            joint = self.description.joints[name]
            joint.check_position(position)
            if name in self.profile.locked_positions:
                if position != self.profile.locked_positions[name]:
                    raise ValueError(f"Joint is locked in this profile: {name}")
                continue
            index = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if index >= 0:
                data.qpos[self.model.jnt_qposadr[index]] = position
        if self.scene is not None:
            self.scene.initialize(self, data)
        mujoco.mj_forward(self.model, data)
        return data


def build_model(robot_model=None, profile=None, *, scene=None):
    description = Description(robot_model or get_robot_model("alohamini2pro"))
    profile = profile or ModelProfile()
    T_world_root = rigid_transform(profile.T_world_root)
    if not math.isfinite(profile.timestep) or not 0 < profile.timestep <= 0.01:
        raise ValueError("Physics timestep must be in (0, 0.01] seconds")
    gravity = numbers(text(profile.gravity), 3, "gravity")
    friction = numbers(text(profile.friction), 3, "friction")
    if profile.collision_profile not in ("authoritative", "cad-pieces"):
        raise ValueError("Unknown collision profile")
    replacements, collision_metadata = {}, None
    if profile.collision_bundle is not None:
        if profile.collision_profile != "authoritative":
            raise ValueError("Use a collision bundle or cad-pieces, not both")
        replacements, collision_metadata = load_bundle(description, profile.collision_bundle)
    if (friction < 0).any():
        raise ValueError("Friction coefficients must be nonnegative")
    for name, value in {**profile.initial_positions, **profile.locked_positions}.items():
        if name not in description.joints:
            raise ValueError(f"Unknown profile joint: {name}")
        description.joints[name].check_position(value)
    # Missing inertial properties are never repaired with tiny fabricated masses.
    for name, joint in description.joints.items():
        if joint.kind != "fixed" and name not in profile.locked_positions:
            if joint.child not in description.inertials:
                raise ValueError(f"Active body has no measured/declared inertia: {joint.child}")
            if joint.effort is None or joint.velocity is None:
                raise ValueError(f"Active joint has no explicit drive limits: {name}")
    root = ET.Element("mujoco", model=description.robot_model.model_id)
    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        fusestatic="false",
        inertiafromgeom="false",
        balanceinertia="false",
        autolimits="true",
    )
    option = ET.SubElement(
        root,
        "option",
        timestep=text([profile.timestep]),
        gravity=text(gravity),
        integrator="implicitfast",
        cone="elliptic",
    )
    # Use explicit SRDF exclusions, rather than relying on parent filtering.
    ET.SubElement(option, "flag", filterparent="disable")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        name="inspection_background",
        type="skybox",
        builtin="gradient",
        rgb1="0.75 0.80 0.85",
        rgb2="0.9 0.93 0.96",
        width="512",
        height="3072",
    )
    world = ET.SubElement(root, "worldbody")
    contact = ET.SubElement(root, "contact")
    actuators = ET.SubElement(root, "actuator")
    assets, meshes, geometry, files, mesh_names, mesh_files = {}, {}, {}, {}, {}, {}
    mesh_by_digest, compiled_mesh_names = {}, set()
    for path, data in description.source_bytes.items():
        files[path.relative_to(description.robot_model.directory).as_posix()] = hashlib.sha256(
            data
        ).hexdigest()
    for relative in (
        "model.json",
        "config/kinematics/physical_parameters.yaml",
        "config/kinematics/kinematics.yaml",
    ):
        path = description.robot_model.directory / relative
        if path.is_file():
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    materials = {}
    for node in description.tree.findall("material"):
        if node.find("color") is not None:
            materials[node.get("name")] = node.find("color").get("rgba")

    def add_geometry(body, link_name, kind, index, node, generated=None):
        if kind == "collision" and profile.collision_profile == "cad-pieces":
            # An explicit comparison candidate restores the original generated
            # distal convex pieces. Do not fit pads to a chosen object width.
            replacements = {
                f"{side}_{suffix}": filename
                for side in ("left", "right")
                for suffix, filename in (
                    ("fixed_finger_pad", "fixed_jaw_vhacd_06.stl"),
                    ("moving_finger_pad", "moving_jaw_vhacd_00.stl"),
                )
            }
            if node.get("name") in replacements:
                node = ET.fromstring(ET.tostring(node))
                for child in list(node):
                    node.remove(child)
                ET.SubElement(
                    ET.SubElement(node, "geometry"),
                    "mesh",
                    filename="../meshes/collision/" + replacements[node.get("name")],
                )
        name = f"{link_name}__{kind}_{index}"
        shape = node.find("geometry")
        if shape is None or len(shape) != 1:
            raise ValueError(f"Expected one geometry: {name}")
        shape = shape[0]
        attributes = {"name": name, **pose_attributes(origin(node)), "mass": "0"}
        if kind == "visual":
            attributes.update(contype="0", conaffinity="0", group="2")
            mat = node.find("material")
            rgba = "0.65 0.65 0.65 1"
            if mat is not None:
                color = mat.find("color")
                rgba = (
                    color.get("rgba") if color is not None else materials.get(mat.get("name"), rgba)
                )
            attributes["rgba"] = text(numbers(rgba, 4, "material rgba"))
        else:
            attributes.update(
                contype="1",
                conaffinity="1",
                group="3",
                friction=text(friction),
                rgba="0.1 0.8 0.3 0.3",
            )
        entry = {
            "link": link_name,
            "kind": kind,
            "T_link_geometry": origin(node),
            "shape": shape.tag,
            "source_name": node.get("name"),
        }
        if shape.tag == "mesh":
            if generated is None:
                path = description.mesh_path(shape)
                relative = path.relative_to(description.robot_model.directory).as_posix()
            else:
                relative = f"generated/{link_name}/{index}"
            if relative not in mesh_names:
                data = path.read_bytes() if generated is None else generated.msh_bytes()
                digest = hashlib.sha256(data).hexdigest()
                if generated is None:
                    files[relative] = digest
                if digest not in mesh_by_digest:
                    mesh = binary_stl(data) if generated is None else generated
                    mesh_by_digest[digest] = (mesh, mesh.msh_bytes())
                mesh, transport = mesh_by_digest[digest]
                meshes[relative] = mesh
                virtual = f"{digest}.msh"
                assets[virtual] = transport
                mesh_files[relative] = virtual
                # Identical source bytes and scale share immutable mesh data;
                # link-local transforms and every separate geom remain intact.
                mesh_names[relative] = f"mesh_{digest}"
            scale = numbers(shape.get("scale", "1 1 1"), 3, "mesh scale")
            if (scale <= 0).any():
                raise ValueError(
                    "Mesh scales must be positive; reflected geometry needs explicit winding"
                )
            # Different scales need different asset identities even for one file.
            mesh_name = (
                mesh_names[relative] + "_" + hashlib.sha256(text(scale).encode()).hexdigest()[:8]
            )
            if mesh_name not in compiled_mesh_names:
                ET.SubElement(
                    asset, "mesh", name=mesh_name, file=mesh_files[relative], scale=text(scale)
                )
                compiled_mesh_names.add(mesh_name)
            attributes.update(type="mesh", mesh=mesh_name)
            entry.update(path=relative, scale=scale)
        elif shape.tag == "box":
            size = numbers(shape.get("size", ""), 3, "box size")
            if (size <= 0).any():
                raise ValueError("Box dimensions must be positive")
            attributes.update(type="box", size=text(size / 2))
            entry["size"] = size
        elif shape.tag in ("cylinder", "sphere"):
            radius = float(shape.get("radius"))
            dimensions = [radius]
            if shape.tag == "cylinder":
                dimensions.append(float(shape.get("length")) / 2)
            if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
                raise ValueError("Primitive dimensions must be positive and finite")
            attributes.update(type=shape.tag, size=text(dimensions))
            entry["size"] = np.asarray(dimensions)
        else:
            raise ValueError(f"Unsupported geometry: {shape.tag}")
        ET.SubElement(body, "geom", **attributes)
        geometry[name] = entry

    def add_link(parent, link_name, pose):
        body = ET.SubElement(parent, "body", name=link_name, **pose_attributes(pose))
        if link_name in description.inertials:
            mass, T_link_inertial, inertia = description.inertials[link_name]
            rotation = T_link_inertial[:3, :3]
            inertia = rotation @ inertia @ rotation.T
            # Explicit float64 diagonalization avoids the compiler's approximate
            # fullinertia eigensolver changing nearly degenerate CAD tensors.
            principal, axes = np.linalg.eigh(inertia)
            if np.linalg.det(axes) < 0:
                axes[:, -1] *= -1
            principal_pose = np.eye(4)
            principal_pose[:3, :3] = axes
            principal_pose[:3, 3] = T_link_inertial[:3, 3]
            ET.SubElement(
                body,
                "inertial",
                **pose_attributes(principal_pose),
                mass=text([mass]),
                diaginertia=text(principal),
            )
        for kind in ("visual", "collision"):
            if kind == "collision" and link_name in replacements:
                for index, mesh in enumerate(replacements[link_name]):
                    node = ET.Element("collision", name=f"cad_candidate_{index}")
                    ET.SubElement(ET.SubElement(node, "geometry"), "mesh")
                    add_geometry(body, link_name, kind, index, node, generated=mesh)
                continue
            for index, node in enumerate(description.links[link_name].findall(kind)):
                add_geometry(body, link_name, kind, index, node)
        for joint in description.by_parent.get(link_name, []):
            child_pose = joint.T_parent_joint
            locked = joint.name in profile.locked_positions
            if locked:
                child_pose = child_pose @ joint.motion(profile.locked_positions[joint.name])
            child = add_link(body, joint.child, child_pose)
            if joint.kind != "fixed" and not locked:
                attributes = dict(
                    name=joint.name,
                    type="slide" if joint.kind == "prismatic" else "hinge",
                    axis=text(joint.axis),
                    damping=text([joint.damping]),
                    frictionloss=text([joint.friction]),
                )
                if joint.lower is not None:
                    attributes.update(limited="true", range=text([joint.lower, joint.upper]))
                ET.SubElement(child, "joint", **attributes)
                ET.SubElement(
                    actuators,
                    "motor",
                    name=joint.name,
                    joint=joint.name,
                    gear="1",
                    ctrllimited="true",
                    forcelimited="true",
                    ctrlrange=text([-joint.effort, joint.effort]),
                    forcerange=text([-joint.effort, joint.effort]),
                )
        return body

    add_link(world, description.root, T_world_root)
    for index, (a, b, _) in enumerate(description.exclusions):
        ET.SubElement(contact, "exclude", name=f"srdf_{index}", body1=a, body2=b)
    for side in ("left", "right"):
        body = root.find(f".//body[@name='{side}_tcp']")
        if body is None:
            raise ValueError(f"Missing authoritative {side} TCP")
        ET.SubElement(body, "site", name=f"{side}_tcp", size="0.003", rgba="1 0 0 1")
    if scene is not None:
        scene.augment(root, assets)
    xml = ET.tostring(root, encoding="unicode")
    compiled = mujoco.MjModel.from_xml_string(xml, assets=assets)
    for relative, expected in files.items():
        if (
            hashlib.sha256((description.robot_model.directory / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"Robot source changed during model construction: {relative}")
    initial = {**description.home, **profile.initial_positions, **profile.locked_positions}
    manifest = {
        "schema_version": 1,
        "robot_model": description.robot_model.model_id,
        "asset_revision": description.robot_model.asset_revision,
        "mujoco_version": mujoco.__version__,
        "frames": {
            "transform": "T_A_B maps B coordinates into A",
            "quaternion": "wxyz",
            "root": description.root,
            "translation": "m",
            "angle": "rad",
        },
        "source_sha256": files,
        "locked_positions": dict(profile.locked_positions),
        "initial_positions": initial,
        "T_world_root": T_world_root.tolist(),
        "physics": {
            "timestep_s": profile.timestep,
            "gravity_m_s2": gravity.tolist(),
            "integrator": "implicitfast",
            "friction": friction.tolist(),
            "parameter_status": "declared simulation parameters; not identified on hardware",
        },
        "collision_exclusions": [
            dict(link1=a, link2=b, reason=reason) for a, b, reason in description.exclusions
        ],
        "collision_profile": "cad-bundle" if collision_metadata else profile.collision_profile,
        "collision_bundle": collision_metadata,
        "collision_profile_status": (
            collision_metadata["definition"].get(
                "status", "explicit CAD candidate; not contact-certified"
            )
            if collision_metadata
            else "URDF collision geometry retained; distal contact pads remain provisional"
            if profile.collision_profile == "authoritative"
            else "comparison candidate; original generated distal hulls, not contact-certified"
        ),
        "mesh_transport": "binary STL to indexed MSH; all triangles and winding retained",
        "inertia": (
            "URDF COM and tensor; float64 principal-axis decomposition; "
            "no geometry inference or balancing"
        ),
    }
    if scene is not None:
        manifest["scene"] = scene.manifest
    return RobotModel(
        description, compiled, profile, xml, assets, meshes, geometry, manifest, initial, scene
    )
