"""Bind RoboTwin's robot API to the common physical model and TCP frames."""

import importlib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien

from alohamini.model import get_robot_model
from alohamini.simulation.sapien import SapienRobot


class Frame:
    def __init__(self, shared, frame, name=None):
        self.shared, self.frame_name, self.name = shared, frame, name or frame

    @property
    def pose(self):
        return sapien.Pose(self.shared.frame(self.frame_name))

    @property
    def global_pose(self):
        return self.pose

    @property
    def entity(self):
        return self.shared.links[self.shared.frames[self.frame_name][0]].entity

    @property
    def local_pose(self):
        return sapien.Pose(self.shared.frames[self.frame_name][1])

    def get_pose(self):
        return self.pose

    def get_name(self):
        return self.name


class ArticulationFrames:
    def __init__(self, shared):
        self.shared = shared

    def __getattr__(self, name):
        return getattr(self.shared.robot, name)

    def find_link_by_name(self, name):
        link = self.shared.robot.find_link_by_name(name)
        return (
            link
            if link is not None
            else Frame(self.shared, name)
            if name in self.shared.frames
            else None
        )

    def find_joint_by_name(self, name):
        joint = self.shared.robot.find_joint_by_name(name)
        if joint is not None:
            return joint
        definition = self.shared.description.joints.get(name)
        return (
            Frame(self.shared, definition.child, name)
            if definition is not None and definition.kind == "fixed"
            else None
        )


def create_robot(
    scene,
    source,
    *,
    render=True,
    initial_positions=None,
    T_world_base=None,
    timestep=0.001,
    base_mode="fixed",
):
    source = Path(source).expanduser().resolve()
    if not (source / "envs/robot/robot.py").is_file():
        raise ValueError("--source must point to a RoboTwin checkout")
    sys.path.insert(0, str(source))
    NativeRobot = importlib.import_module("envs.robot.robot").Robot
    shared = SapienRobot(
        scene,
        render=render,
        initial_positions=initial_positions,
        T_world_base=T_world_base,
        timestep=timestep,
        base_mode=base_mode,
    )
    model = get_robot_model("alohamini2pro")

    class Loader:
        fix_root_link = True

        def __init__(self, adapter):
            self.adapter = adapter

        def load(self, path):
            if Path(path).resolve() != self.adapter.description.path:
                raise ValueError("RoboTwin requested a different robot asset")
            return ArticulationFrames(self.adapter)

    class Scene:
        def __init__(self, actual_scene, adapter):
            self.actual_scene, self.adapter = actual_scene, adapter

        def create_urdf_loader(self):
            return Loader(self.adapter)

        def __getattr__(self, name):
            return getattr(self.actual_scene, name)

    class SharedRobot(NativeRobot):
        def apply_passive_force(self):
            self.shared.before_step()

        def reset(self, scene, need_topp=False, **kwargs):
            for link in self.shared.robot.get_links():
                if link.entity.scene is self.shared.scene:
                    self.shared.scene.remove_entity(link.entity)
            self.shared = SapienRobot(
                scene,
                render=render,
                initial_positions=initial_positions,
                T_world_base=T_world_base,
                timestep=timestep,
                base_mode=base_mode,
            )
            options = {**creation_options, **kwargs}
            self._init_robot_(Scene(scene, self.shared), need_topp, **options)
            self.embodiment_config = options["left_embodiment_config"]
            self.shared.base_motion.pose = (
                self.shared.robot.get_root_pose().to_transformation_matrix()
            )
            self.init_joints()
            self._initialize_command_state()

        def _initialize_command_state(self):
            closed, opened = self.embodiment_config["gripper_scale"]
            for side in ("left", "right"):
                q = self.shared.control.goal[self.shared.names.index(side + "_gripper")]
                setattr(self, side + "_gripper_val", float((q - closed) / (opened - closed)))
            for joint in self.shared.robot.get_active_joints():
                joint.set_drive_properties(0, 0)

        def _arm_state(self, side, *, measured=False):
            names = self.embodiment_config["arm_joints_name"][0 if side == "left" else 1]
            positions = self.shared.robot.get_qpos() if measured else self.shared.control.goal
            values = [float(positions[self.shared.names.index(n)]) for n in names]
            closed, opened = self.embodiment_config["gripper_scale"]
            q = positions[self.shared.names.index(side + "_gripper")]
            return [*values, float((q - closed) / (opened - closed))]

        def get_left_arm_jointState(self):
            return self._arm_state("left")

        def get_right_arm_jointState(self):
            return self._arm_state("right")

        def get_left_arm_real_jointState(self):
            return self._arm_state("left", measured=True)

        def get_right_arm_real_jointState(self):
            return self._arm_state("right", measured=True)

        def set_arm_joints(self, target_position, target_velocity, arm_tag):
            if arm_tag not in ("left", "right"):
                raise ValueError("Choose left or right arm")
            names = self.embodiment_config["arm_joints_name"][0 if arm_tag == "left" else 1]
            if len(target_position) != len(names) or len(target_velocity) != len(names):
                raise ValueError("Arm targets require six named SI joint coordinates")
            if not np.isfinite(target_velocity).all():
                raise ValueError("Nonfinite arm velocity")
            if any(
                abs(v) > self.shared.description.joints[n].velocity
                for n, v in zip(names, target_velocity, strict=True)
            ):
                raise ValueError("Arm velocity exceeds a URDF speed limit")
            self.shared.set_targets(dict(zip(names, target_position, strict=True)))

        def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):
            if (
                arm_tag not in ("left", "right")
                or not np.isfinite(gripper_val)
                or not 0 <= gripper_val <= 1
            ):
                raise ValueError("Use left/right and normalized gripper command in [0,1]")
            closed, opened = self.embodiment_config["gripper_scale"]
            self.shared.set_targets(
                {arm_tag + "_gripper": closed + gripper_val * (opened - closed)}
            )
            setattr(self, arm_tag + "_gripper_val", float(gripper_val))

        def _trans_endpose(self, arm_tag=None, is_endpose=False):
            if arm_tag not in ("left", "right"):
                raise ValueError("Choose left or right TCP")
            pose = sapien.Pose(self.shared.frame(arm_tag + "_tcp"))
            return [*map(float, pose.p), *map(float, pose.q)]

        def _trans_from_gripper_to_endlink(self, target_pose, arm_tag=None):
            # The registered move group is already the shared TCP. RoboTwin's
            # generic 12 cm end-link offset is therefore not applied again.
            values = np.asarray(target_pose, dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("TCP pose requires XYZ and WXYZ quaternion")
            return sapien.Pose(values[:3], values[3:])

    suffixes = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_yaw_joint",
        "wrist_roll",
    )
    arms = [[side + "_" + n for n in suffixes] for side in ("left", "right")]
    home = shared.description.home
    pose = sapien.Pose(np.eye(4) if T_world_base is None else T_world_base)
    config = {
        "urdf_path": str(shared.description.path),
        "srdf_path": str(shared.description.semantic_path),
        "move_group": ["left_tcp", "right_tcp"],
        "ee_joints": ["left_tcp_joint", "right_tcp_joint"],
        "arm_joints_name": arms,
        "gripper_name": [{"base": side + "_gripper", "mimic": []} for side in ("left", "right")],
        "gripper_bias": 0,
        "homestate": [[home[n] for n in names] for names in arms],
        "joint_stiffness": 0,
        "joint_damping": 0,
        "gripper_stiffness": 0,
        "gripper_damping": 0,
        "robot_pose": [[*map(float, pose.p), *map(float, pose.q)]],
        "planner": "mplib_RRT",
        "continuous_gravity_compensation": True,
    }
    semantic = ET.parse(shared.description.semantic_path).getroot()
    config["gripper_scale"] = [
        float(
            semantic.find(f"group_state[@group='left_gripper'][@name='{state}']/joint").get("value")
        )
        for state in ("closed", "open")
    ]
    creation_options = dict(
        dual_arm_embodied=True,
        left_robot_file=str(model.directory),
        right_robot_file=str(model.directory),
        left_embodiment_config=config,
        right_embodiment_config=config,
    )
    robot = SharedRobot(Scene(scene, shared), **creation_options)
    robot.shared, robot.embodiment_config = shared, config
    robot.init_joints()
    robot._initialize_command_state()
    return robot
