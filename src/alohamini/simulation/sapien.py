"""SAPIEN import of shared rigid groups with exact fixed-frame retention."""

import copy

import numpy as np

from alohamini.model import get_robot_model

from .base import BaseKinematics, BaseMotion
from .control import PositionControl
from .description import Description


def rigid_groups(description):
    poses = description.forward()
    owners = {description.root: description.root}

    def visit(parent):
        for joint in description.by_parent.get(parent, []):
            owners[joint.child] = owners[parent] if joint.kind == "fixed" else joint.child
            visit(joint.child)

    visit(description.root)
    frames = {n: (o, np.linalg.inv(poses[o]) @ poses[n]) for n, o in owners.items()}
    groups = {}
    for name, (mass, com, tensor) in description.inertials.items():
        owner, pose = frames[name]
        com = pose @ com
        groups.setdefault(owner, []).append(
            (mass, com[:3, 3], com[:3, :3] @ tensor @ com[:3, :3].T)
        )
    inertias = {}
    for owner, parts in groups.items():
        mass = sum(p[0] for p in parts)
        com = sum(m * c for m, c, _ in parts) / mass
        tensor = sum(
            i + m * (np.dot(c - com, c - com) * np.eye(3) - np.outer(c - com, c - com))
            for m, c, i in parts
        )
        inertias[owner] = mass, com, tensor
    if set(owners.values()) != set(inertias):
        raise ValueError("Every independent rigid body needs positive declared mass")
    return frames, inertias


def fuse_fixed_links(builder, description):
    """Fuse fixed bodies before PhysX can fabricate mass for empty frames.

    Joint-frame composition and per-shape SRDF masks are preserved. Camera/TCP
    frames remain available through the returned owner-relative transforms.
    """
    import sapien

    frames, inertias = rigid_groups(description)
    original = list(builder.link_builders)
    by_name = {b.name: b for b in original}
    collisions, visuals, masks = {}, {}, {}
    for source in original:
        owner, transform = frames[source.name]
        pose = sapien.Pose(transform)
        for source_records, records in (
            (source.collision_records, collisions),
            (source.visual_records, visuals),
        ):
            for record in source_records:
                record = copy.copy(record)
                record.pose = pose * record.pose
                records.setdefault(owner, []).append(record)
                if records is collisions:
                    masks.setdefault(owner, []).append(list(source.collision_groups))
    kept = []
    for body in original:
        if frames[body.name][0] != body.name:
            continue
        if body.parent is not None:
            parent_name = body.parent.name
            body.parent = by_name[frames[parent_name][0]]
            body.joint_record.pose_in_parent = (
                sapien.Pose(frames[parent_name][1]) * body.joint_record.pose_in_parent
            )
        mass, com, tensor = inertias[body.name]
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = axes, com
        body.set_mass_and_inertia(mass, sapien.Pose(pose), moments)
        body.collision_records = collisions.get(body.name, [])
        body.visual_records = visuals.get(body.name, [])
        body.index = len(kept)
        kept.append(body)
    builder.link_builders = kept
    return frames, masks


def apply_shape_filters(articulation, masks, *, name_map=None):
    for link in articulation.get_links():
        # ManiSkill prefixes native names; its public link mapping is applied by
        # the caller before passing each native articulation.
        name = (name_map or {}).get(link.name, link.name)
        shapes = link.get_collision_shapes()
        expected = masks.get(name, [])
        if len(shapes) != len(expected):
            raise RuntimeError(
                f"Collision cooking changed shape coverage: {link.name}: "
                f"{len(shapes)} != {len(expected)}"
            )
        for shape, groups in zip(shapes, expected, strict=True):
            shape.set_collision_groups(groups)


class SapienRobot:
    def __init__(
        self,
        scene,
        *,
        render=True,
        initial_positions=None,
        T_world_base=None,
        timestep=0.001,
        fixed_base=True,
        base_mode="fixed",
    ):
        import sapien

        self.scene = scene
        self.description = Description(get_robot_model("alohamini2pro"))
        loader = scene.create_urdf_loader()
        loader.fix_root_link = fixed_base
        parsed = loader.parse(str(self.description.path), str(self.description.semantic_path))
        builder = parsed[0][0]
        self.frames, masks = fuse_fixed_links(builder, self.description)
        if not render:
            for link in builder.link_builders:
                link.visual_records.clear()
        builder.initial_pose = sapien.Pose(np.eye(4) if T_world_base is None else T_world_base)
        self.robot = builder.build(fix_root_link=fixed_base)
        apply_shape_filters(self.robot, masks)
        self.names = tuple(j.name for j in self.robot.get_active_joints())
        self.links = {link.name: link for link in self.robot.get_links()}
        values = {**self.description.home, **(initial_positions or {})}
        if values.keys() - self.description.joints.keys():
            raise ValueError("Unknown initial joint names")
        positions = np.array([values.get(n, 0) for n in self.names])
        for n, q in zip(self.names, positions, strict=True):
            self.description.joints[n].check_position(float(q))
        self.robot.set_qpos(positions)
        self.control = PositionControl(self.description, self.names, positions, timestep=timestep)
        self.base_motion = BaseMotion(
            self.description,
            np.eye(4) if T_world_base is None else T_world_base,
            mode=base_mode,
            wheel_positions=[positions[self.names.index(n)] for n in BaseKinematics.names],
        )
        for joint in self.robot.get_active_joints():
            joint.set_drive_properties(0, 0)

    def frame(self, name):
        owner, local = self.frames[name]
        return self.links[owner].entity_pose.to_transformation_matrix() @ local

    def set_targets(self, targets):
        goal = self.control.goal.copy()
        for name, value in targets.items():
            if name not in self.names:
                raise ValueError(f"Unknown or fixed joint target: {name}")
            goal[self.names.index(name)] = value
        self.control.set_goal(goal)

    def before_step(self):
        if self.base_motion.mode == "kinematic":
            import sapien

            self.robot.set_root_pose(sapien.Pose(self.base_motion.step(self.control.timestep)))
            for name, q in zip(
                self.base_motion.kinematics.names, self.base_motion.wheel_positions, strict=True
            ):
                self.control.goal[self.names.index(name)] = q
        self.robot.set_qf(
            self.control.effort(
                self.robot.get_qpos(), self.robot.get_qvel(), self.robot.compute_passive_force()
            )
        )
