"""Object-relative reference motions, qualified by ordinary dynamic collection.

An admitted reference supplies commanded TCP poses, measured grasp geometry and
unchanged jaw commands. Object poses define rigid frame changes; interpolating
these changes joins the source-relative and target-relative segments. This is
the object-relative demonstration reuse used by MimicGen (CoRL 2023):
https://mimicgen.github.io/ . No upstream implementation code is incorporated.

Only planning scratch data change object coordinates. Collected motion always
uses the torque-limited follower, contacts and original task predicates.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .control import (
    DynamicFollower,
    FollowerSettings,
    JointTrajectory,
    arm_joint_names,
    solve_ik_targets,
)
from .export import model_xml
from .libero import settle_scene, sha256


@dataclass(frozen=True)
class MotionRecipe:
    side: str
    pregrasp_end_frame: int
    approach_end_frame: int
    grasp_start_frame: int
    source_end_frame: int
    target_start_frame: int
    release_start_frame: int
    allow_lift_motion: bool = True

    def __post_init__(self):
        frames = (
            self.pregrasp_end_frame,
            self.approach_end_frame,
            self.grasp_start_frame,
            self.source_end_frame,
            self.target_start_frame,
            self.release_start_frame,
        )
        if self.side not in ("left", "right") or any(type(v) is not int for v in frames):
            raise ValueError("Motion recipe requires an arm side and integer frame boundaries")
        if not 0 < frames[0] < frames[1] < frames[2] < frames[3] < frames[4] < frames[5]:
            raise ValueError(
                "Motion phases must increase: pregrasp, approach, grasp, source, target, release"
            )
        if type(self.allow_lift_motion) is not bool:
            raise ValueError("allow_lift_motion must be true or false")

    @classmethod
    def read(cls, path):
        specification = json.loads(Path(path).expanduser().read_text())
        if specification.pop("schema_version", None) != 1:
            raise ValueError("Motion recipe must declare schema_version 1")
        required = set(cls.__dataclass_fields__) - {"allow_lift_motion"}
        if not required <= set(specification) <= set(cls.__dataclass_fields__):
            raise ValueError("Motion recipe requires side and six named phase boundaries")
        return cls(**specification)


def _pose(data, name):
    result = np.eye(4)
    body = data.body(name)
    result[:3, :3], result[:3, 3] = body.xmat.reshape(3, 3), body.xpos
    return result


def _set_pose(model, data, joint_name, pose):
    address = model.jnt_qposadr[model.joint(joint_name).id]
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(pose[:3, :3]).ravel())
    data.qpos[address : address + 3] = pose[:3, 3]
    data.qpos[address + 3 : address + 7] = quaternion


def _collisions(model, data, relevant, tolerance, allowed=()):
    return [
        (model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist))
        for c in data.contact
        if (c.geom1 in relevant or c.geom2 in relevant)
        and c.dist < -tolerance
        and frozenset((int(c.geom1), int(c.geom2))) not in allowed
    ]


class ReferenceMotion:
    """Reuse one training reference on named resets of identical scene physics."""

    def __init__(self, robot, directory, recipe):
        import h5py

        from .collection import STATE_SPEC, restore_state

        if robot.scene is None:
            raise ValueError("Reference retargeting requires a task scene")
        directory = Path(directory).expanduser().resolve()
        manifest = json.loads((directory / "dataset.json").read_text())
        if (
            manifest.get("format") != "alohamini-sim-v1"
            or not manifest.get("admitted")
            or manifest.get("split") != "train"
            or not manifest.get("quality", {}).get("replay", {}).get("passed")
        ):
            raise ValueError("Reference must be an admitted, replay-verified training episode")
        if manifest["mujoco_version"] != mujoco.__version__ or manifest["state_spec"] != int(
            STATE_SPEC
        ):
            raise ValueError("Reference requires the same MuJoCo version and integration state")
        self.model_hash = hashlib.sha256(model_xml(robot).encode()).hexdigest()
        for relative, expected in manifest["model_files"].items():
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or sha256(path) != expected:
                raise ValueError(f"Reference model checksum mismatch: {relative}")
        if manifest["model_files"]["model/model.xml"] != self.model_hash:
            raise ValueError(
                "Reference robot, scene geometry or physics differ from the target model"
            )
        episode = (directory / manifest["episode"]).resolve()
        if not episode.is_relative_to(directory) or sha256(episode) != manifest["episode_sha256"]:
            raise ValueError("Reference episode checksum mismatch")
        goals = robot.scene.manifest["goals"]
        if len(goals) != 1 or goals[0][0].lower() != "on":
            raise ValueError("Reference retargeting supports a single On task")
        if manifest["environment"]["scene"]["goals"] != goals:
            raise ValueError("Reference and target must have identical named task goals")
        _, source_name, target_name = goals[0]
        self.source = robot.scene.task["objects"][source_name]
        self.target = robot.scene.task["objects"][target_name]
        if (
            len(self.source["joints"]) != 1
            or robot.model.joint(self.source["joints"][0]).type[0] != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise ValueError("The source object must have one free joint")
        trajectory = json.loads((directory / "trajectory.json").read_text())
        self.fps = manifest["collection_settings"]["fps"]
        if self.fps != 20:
            raise ValueError("Reference commands must use 20 Hz")
        count = math.ceil(float(trajectory["times_s"][-1]) * self.fps) + 1
        with h5py.File(episode, "r") as file:
            if json.loads(file.attrs["env_args"]) != manifest["environment"]:
                raise ValueError("Reference environment differs from its dataset manifest")
            group = file["data/demo_0"]
            self.names = tuple(json.loads(group.attrs["joint_names"]))
            if self.names != tuple(robot.model.actuator(i).name for i in range(robot.model.nu)):
                raise ValueError("Reference actuator names/order differ from the target model")
            self.actions = group["actions"][:count]
            states = group["states"][:count]
            controller = json.loads(group.attrs["controller"])
        if (
            len(self.actions) != count
            or len(states) != count
            or recipe.release_start_frame >= count - 1
        ):
            raise ValueError("Motion recipe extends beyond the recorded reference commands")
        self.recipe = recipe
        self.settings = FollowerSettings(**controller["settings"])
        self.times = np.arange(count) / self.fps
        self.qpos_ids = [robot.model.jnt_qposadr[robot.model.joint(n).id] for n in self.names]
        self.gripper = f"{recipe.side}_gripper"
        self.tcp = f"{recipe.side}_tcp"
        gripper_qpos = robot.model.jnt_qposadr[robot.model.joint(self.gripper).id]
        data = mujoco.MjData(robot.model)
        restore_state(robot.model, data, states[0])
        self.source_initial = _pose(data, self.source["root_body"])
        self.target_initial = _pose(data, self.target["root_body"])
        self.command_tcp, self.measured_tcp, self.object_poses, self.jaw_openings = [], [], [], []
        self.tracking_offsets = []
        object_geoms = {robot.model.geom(n).id for n in self.source["contact_geoms"]}
        jaws = {robot.model.body(f"{recipe.side}_{jaw}").id for jaw in ("Fixed_Jaw", "Moving_Jaw")}
        finger_geoms, self.contact_frames = set(), []
        for state, command in zip(states, self.actions, strict=True):
            restore_state(robot.model, data, state)
            touching = set()
            for contact in data.contact:
                other = (
                    contact.geom2
                    if contact.geom1 in object_geoms
                    else contact.geom1
                    if contact.geom2 in object_geoms
                    else None
                )
                if other is not None and robot.model.geom_bodyid[other] in jaws:
                    touching.add(int(other))
            self.contact_frames.append(bool(touching))
            finger_geoms.update(touching)
            self.object_poses.append(_pose(data, self.source["root_body"]))
            self.measured_tcp.append(_pose(data, self.tcp))
            self.jaw_openings.append(float(data.qpos[gripper_qpos]))
            self.tracking_offsets.append(data.qpos[self.qpos_ids].copy() - command)
            data.qpos[self.qpos_ids] = command
            mujoco.mj_kinematics(robot.model, data)
            self.command_tcp.append(_pose(data, self.tcp))
        # A contact can move across convex pieces of the same certified printed
        # jaw. Preserve the separate camera/motor shapes on the fixed-jaw body.
        parts = (
            (robot.manifest.get("collision_bundle") or {}).get("definition", {}).get("links", {})
        )
        for jaw in ("Fixed_Jaw", "Moving_Jaw"):
            link = f"{recipe.side}_{jaw}"
            part = parts.get(link, {})
            interval = part.get("printed_contact_reference", {}).get("generated_piece_range")
            if (
                interval is None
                and jaw == "Moving_Jaw"
                and part.get("cad_geometry", {}).get("passed")
            ):
                interval = [0, len(part["pieces"])]
            if interval is not None:
                finger_geoms.update(
                    robot.model.geom(f"{link}__collision_{i}").id for i in range(*interval)
                )
        self.allowed_grasp_contacts = {
            frozenset((obj, finger)) for obj in object_geoms for finger in finger_geoms
        }
        self.first_contact = next((i for i, v in enumerate(self.contact_frames) if v), count)
        self.provenance = {
            "method": "object_relative_reference_v4",
            "reference_manifest_sha256": sha256(directory / "dataset.json"),
            "reference_episode_sha256": manifest["episode_sha256"],
            "model_xml_sha256": self.model_hash,
            "recipe": asdict(recipe),
            "generator_sha256": sha256(__file__),
            "source_object": source_name,
            "target_object": target_name,
            "fps": self.fps,
        }

    def plan(self, robot, *, progress=None):
        if not self.recipe.allow_lift_motion:
            return self._plan(robot, coordinated=False, progress=progress)
        try:
            return self._plan(robot, coordinated=False, progress=progress)
        except _PlanningFailure as primary:
            if progress:
                progress({"stage": "coordinated_planning", "primary_failure": str(primary)})
            try:
                trajectory, settings, generation = self._plan(
                    robot, coordinated=True, progress=progress
                )
            except _PlanningFailure as secondary:
                raise RuntimeError(
                    f"Fixed-lift: {primary}; coordinated: {secondary}"
                ) from secondary
            generation["primary_failure"] = str(primary)
            return trajectory, settings, generation

    def _plan(self, robot, *, coordinated, progress):
        if hashlib.sha256(model_xml(robot).encode()).hexdigest() != self.model_hash:
            raise ValueError("Target model changed since loading the reference")
        recipe, names, model = self.recipe, self.names, robot.model
        arm = arm_joint_names(recipe.side)
        active = ("vertical_move", *arm)
        follower = DynamicFollower(robot, settings=self.settings)
        settle_scene(follower)
        source_delta = _pose(follower.data, self.source["root_body"]) @ np.linalg.inv(
            self.source_initial
        )
        target_delta = _pose(follower.data, self.target["root_body"]) @ np.linalg.inv(
            self.target_initial
        )
        rotation = Slerp([0, 1], Rotation.from_matrix([source_delta[:3, :3], target_delta[:3, :3]]))
        scratch = mujoco.MjData(model)
        mujoco.mj_copyData(scratch, model, follower.data)
        commands = self.actions.copy()
        first = follower.data.qpos[follower.qpos_ids].copy()
        if not recipe.allow_lift_motion:
            commands[:, names.index("vertical_move")] = first[names.index("vertical_move")]
        for index, name in enumerate(names):
            if name not in {*active, self.gripper}:
                commands[:, index] = first[index]
        robot_geoms = {model.geom(name).id for name in robot.geometry}
        source_geoms = {model.geom(name).id for name in self.source["contact_geoms"]}
        tolerance = follower.settings.penetration_tolerance_m
        gripper_qpos = model.jnt_qposadr[model.joint(self.gripper).id]
        pre = recipe.pregrasp_end_frame

        def solve(target, movable, seed, frame):
            try:
                return solve_ik_targets(
                    robot,
                    {recipe.side: target},
                    movable_joints=movable,
                    seed_positions=seed,
                    reference_data=scratch,
                    max_iterations=2000 if coordinated else 400,
                )
            except RuntimeError as exc:
                raise _PlanningFailure(f"IK at frame {frame}: {exc}") from exc

        free_end = max(pre, min(recipe.approach_end_frame, self.first_contact - 1))
        candidates = (
            range(pre, free_end + 1) if coordinated or not recipe.allow_lift_motion else [pre]
        )
        failure = None
        for pre_reference in candidates:
            try:
                scratch.qpos[self.qpos_ids] = first
                seed = dict(zip(names, self.actions[pre_reference], strict=True))
                if not recipe.allow_lift_motion:
                    seed["vertical_move"] = first[names.index("vertical_move")]
                solution = solve(
                    source_delta @ self.command_tcp[pre_reference],
                    active if coordinated else arm,
                    seed,
                    pre_reference,
                )
                end = commands[pre].copy()
                for name, value in solution["joint_positions"].items():
                    end[names.index(name)] = value
                u = np.arange(pre + 1) / pre
                values = first + (3 * u * u - 2 * u * u * u)[:, None] * (end - first)
                for frame, value in enumerate(values):
                    scratch.qpos[self.qpos_ids] = value
                    mujoco.mj_forward(model, scratch)
                    if collisions := _collisions(model, scratch, robot_geoms, tolerance):
                        raise _PlanningFailure(
                            f"Approach collision at frame {frame}: {collisions[:2]}"
                        )
                commands[: pre + 1] = values
                break
            except _PlanningFailure as exc:
                failure = exc
        else:
            raise _PlanningFailure(f"No reachable free approach waypoint: {failure}")
        previous = commands[pre].copy()
        driven_indices = [names.index(n) for n in (*active, self.gripper)]
        for frame in range(pre + 1, len(commands)):
            reference_index = float(frame)
            target_pose = self.command_tcp[frame]
            if pre_reference > pre and frame <= recipe.approach_end_frame:
                fraction = (frame - pre) / (recipe.approach_end_frame - pre)
                fraction = 3 * fraction**2 - 2 * fraction**3
                reference_index = (
                    pre_reference + (recipe.approach_end_frame - pre_reference) * fraction
                )
                low = int(np.floor(reference_index))
                high = min(low + 1, len(commands) - 1)
                weight = reference_index - low
                target_pose = np.eye(4)
                target_pose[:3, 3] = (
                    self.command_tcp[low][:3, 3] * (1 - weight)
                    + self.command_tcp[high][:3, 3] * weight
                )
                target_pose[:3, :3] = Slerp(
                    [0, 1],
                    Rotation.from_matrix(
                        [self.command_tcp[low][:3, :3], self.command_tcp[high][:3, :3]]
                    ),
                )(weight).as_matrix()
            u = np.clip(
                (frame - recipe.source_end_frame)
                / (recipe.target_start_frame - recipe.source_end_frame),
                0,
                1,
            )
            u = 3 * u * u - 2 * u * u * u
            delta = np.eye(4)
            delta[:3, :3] = rotation(u).as_matrix()
            delta[:3, 3] = source_delta[:3, 3] * (1 - u) + target_delta[:3, 3] * u
            # Predict actual grasp geometry at the commanded TCP, rather than
            # forcing the unloaded jaw command through a rigid object wall.
            predicted = (
                delta
                @ self.command_tcp[frame]
                @ np.linalg.inv(self.measured_tcp[frame])
                @ self.object_poses[frame]
            )
            scratch.qpos[self.qpos_ids] = previous
            _set_pose(model, scratch, self.source["joints"][0], predicted)
            scratch.qpos[gripper_qpos] = self.jaw_openings[frame]
            movable = (
                active
                if recipe.allow_lift_motion and (coordinated or frame > recipe.target_start_frame)
                else arm
            )
            guess = (
                self.actions[frame] + previous - self.actions[frame - 1]
                if coordinated
                else previous
            )
            seed = (
                {
                    n: float(
                        np.clip(
                            guess[names.index(n)],
                            robot.description.joints[n].lower,
                            robot.description.joints[n].upper,
                        )
                    )
                    for n in movable
                }
                if coordinated
                else {n: previous[names.index(n)] for n in movable}
            )
            solution = solve(delta @ target_pose, movable, seed, frame)
            command = commands[frame].copy()
            for name, value in solution["joint_positions"].items():
                command[names.index(name)] = value
            # Preview the reference servo deflection in the same joint units.
            # Nominal commands alone falsely move a supported object through
            # the table when the reference arm lags its position target.
            expected = command.copy()
            expected[driven_indices] += self.tracking_offsets[frame][driven_indices]
            scratch.qpos[self.qpos_ids] = expected
            mujoco.mj_kinematics(model, scratch)
            if frame < recipe.grasp_start_frame:
                predicted = source_delta @ self.object_poses[frame]
            elif frame >= recipe.release_start_frame:
                predicted = target_delta @ self.object_poses[frame]
            else:
                predicted = (
                    _pose(scratch, self.tcp)
                    @ np.linalg.inv(self.measured_tcp[frame])
                    @ self.object_poses[frame]
                )
            _set_pose(model, scratch, self.source["joints"][0], predicted)
            mujoco.mj_forward(model, scratch)
            contact_index = min(round(reference_index), len(commands) - 1)
            allowed = self.allowed_grasp_contacts if self.contact_frames[contact_index] else ()
            if collisions := _collisions(
                model, scratch, robot_geoms | source_geoms, tolerance, allowed
            ):
                raise _PlanningFailure(
                    f"Predicted transfer collision at frame {frame}: {collisions[:2]}"
                )
            if not solution["gravity_load"]["within_limits"]:
                raise _PlanningFailure(f"Static motor effort exceeds rated limits at frame {frame}")
            commands[frame], previous = command, command
            if progress is not None and frame % 200 == 0:
                progress({"stage": "planning", "frame": frame, "frames": len(commands)})
        try:
            trajectory = JointTrajectory(robot, names, self.times, commands)
        except ValueError as exc:
            raise _PlanningFailure(str(exc)) from exc
        generation = {
            **self.provenance,
            "reset_index": robot.scene.reset_index,
            "source_transform_world": source_delta.tolist(),
            "target_transform_world": target_delta.tolist(),
            "planner": "coordinated_reference" if coordinated else "fixed_lift_reference",
            "pregrasp_reference_frame": pre_reference,
        }
        return trajectory, self.settings, generation


class _PlanningFailure(RuntimeError):
    pass
