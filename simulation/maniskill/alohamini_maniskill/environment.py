"""A native ManiSkill agent and extensible, state-observation research scene."""

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent
from mani_skill.agents.controllers import PassiveControllerConfig, PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig

from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity
from alohamini.simulation.base import BaseKinematics, BaseMotion
from alohamini.simulation.control import PositionControl
from alohamini.simulation.description import Description
from alohamini.simulation.sapien import apply_shape_filters, fuse_fixed_links
from alohamini.simulation.scene import load_scene


@register_agent()
class AlohaMini2Pro(BaseAgent):
    uid = "alohamini2pro"
    urdf_path = str(get_robot_model(uid).description_path("collision"))
    fix_root_link = True

    def _load_articulation(self, initial_pose=None):
        if self.scene.num_envs != 1 or self.scene.device.type != "cpu":
            raise ValueError("This research profile uses one PhysX CPU environment")
        self.description = Description(get_robot_model(self.uid))
        loader = self.scene.create_urdf_loader()
        loader.name, loader.fix_root_link = self.uid, True
        parsed = loader.parse(str(self.description.path), str(self.description.semantic_path))
        builder = parsed["articulation_builders"][0]
        self.frames, masks = fuse_fixed_links(builder, self.description)
        builder.set_name(self.uid)
        builder.initial_pose = initial_pose or sapien.Pose()
        self.robot = builder.build()
        self.raw_robot = self.robot._objs[0]
        names = {link._objs[0].name: link.name for link in self.robot.links}
        apply_shape_filters(self.raw_robot, masks, name_map=names)
        self.robot_link_names = [link.name for link in self.robot.links]

    @property
    def _controller_configs(self):
        names = [
            j.name
            for j in self.robot.active_joints
            if self.description.joints[j.name].lower is not None
        ]
        wheels = [
            j.name
            for j in self.robot.active_joints
            if self.description.joints[j.name].kind == "continuous"
        ]
        return {
            "joint_position": {
                "joints": PDJointPosControllerConfig(
                    names,
                    lower=None,
                    upper=None,
                    stiffness=0,
                    damping=0,
                    force_limit=[self.description.joints[n].effort for n in names],
                    normalize_action=False,
                ),
                "wheels": PassiveControllerConfig(wheels, damping=0),
                "balance_passive_force": False,
            }
        }

    def reset(self, init_qpos=None):
        super().reset(init_qpos)
        names = tuple(j.name for j in self.robot.active_joints)
        self.position_control = PositionControl(
            self.description,
            names,
            self.raw_robot.get_qpos(),
            timestep=self.scene.px.timestep,
        )
        self._position_indices = [
            names.index(n) for n in self.controller.controllers["joints"].config.joint_names
        ]

    def set_action(self, action):
        super().set_action(action)
        controller = self.controller.controllers["joints"]
        goal = self.position_control.goal.copy()
        goal[self._position_indices] = controller._target_qpos[0].cpu().numpy()
        self.position_control.set_goal(goal)

    def before_simulation_step(self):
        super().before_simulation_step()
        if self.base_motion.mode == "kinematic":
            self.robot.set_pose(
                Pose.create(sapien.Pose(self.base_motion.step(self.scene.px.timestep)))
            )
            for name, q in zip(
                self.base_motion.kinematics.names, self.base_motion.wheel_positions, strict=True
            ):
                self.position_control.goal[self.position_control.names.index(name)] = q
        self.raw_robot.set_qf(
            self.position_control.effort(
                self.raw_robot.get_qpos(),
                self.raw_robot.get_qvel(),
                self.raw_robot.compute_passive_force(),
            )
        )

    def frame(self, name):
        owner, local = self.frames[name]
        pose = self.robot.links_map[owner].pose.to_transformation_matrix()[0].cpu().numpy()
        return pose @ local

    def hold_action(self):
        return self.robot.get_qpos()[
            :, self.controller.controllers["joints"].active_joint_indices
        ].clone()

    @property
    def action_joint_names(self):
        return tuple(self.controller.controllers["joints"].config.joint_names)

    def set_base_velocity(self, velocity):
        self.base_motion.set_velocity(velocity)


@register_env("AlohaMiniResearch-v1", max_episode_steps=2000)
class AlohaMiniResearchEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["alohamini2pro"]

    def __init__(self, *args, scene_config=None, viewer_refresh_hz=None, **kwargs):
        self.configuration = load_scene(scene_config)
        self.objects = {}
        self._view_clock = None
        super().__init__(*args, robot_uids="alohamini2pro", **kwargs)
        if viewer_refresh_hz is not None:
            if self.render_mode != "human":
                raise ValueError("Viewer refresh requires render_mode='human'")
            from .viewer import ViewerClock

            self._view_clock = ViewerClock(
                self.render_human,
                timestep=1 / self.sim_freq,
                refresh_hz=viewer_refresh_hz,
                control_period=1 / self.control_freq,
            )

    def _after_simulation_step(self):
        super()._after_simulation_step()
        if self._view_clock is not None:
            self._view_clock.step()

    @property
    def _default_sim_config(self):
        return SimConfig(
            sim_freq=round(1 / self.configuration["timestep_s"]),
            control_freq=self.configuration["control_frequency_hz"],
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.5, 1.3, 1.4], target=[0.3, 0, 0.6])
        return CameraConfig("render_camera", pose, 640, 480, 1.0, 0.01, 100)

    def _load_agent(self, options):
        super()._load_agent(
            options, initial_agent_poses=sapien.Pose(self.configuration["T_world_base"])
        )

    def _load_scene(self, options):
        from alohamini.simulation.sapien_scene import add_objects

        self.objects = add_objects(self.scene, self.configuration, render=self.scene.can_render())

    def _initialize_episode(self, env_idx, options):
        values = {**self.agent.description.home, **self.configuration["initial_positions"]}
        for n, value in values.items():
            self.agent.description.joints[n].check_position(float(value))
        names = [j.name for j in self.agent.robot.active_joints]
        qpos = torch.tensor([[values.get(n, 0) for n in names]], device=self.device)
        self.agent.robot.set_pose(Pose.create(sapien.Pose(self.configuration["T_world_base"])))
        self.agent.reset(qpos)
        self.agent.base_motion = BaseMotion(
            self.agent.description,
            self.configuration["T_world_base"],
            mode=self.configuration["base_mode"],
            wheel_positions=[values.get(n, 0) for n in BaseKinematics.names],
        )
        for obj in self.configuration["objects"]:
            actor = self.objects[obj["name"]]
            actor.set_pose(Pose.create(sapien.Pose(obj.get("T_world_object", np.eye(4)))))
            if obj.get("mass_kg", 0) > 0:
                actor.set_linear_velocity(torch.zeros((1, 3), device=self.device))
                actor.set_angular_velocity(torch.zeros((1, 3), device=self.device))

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info):
        return {"base_pose": self.agent.robot.pose.raw_pose}

    def get_state_dict(self):
        state = super().get_state_dict()
        control = self.agent.position_control.state()
        base = self.agent.base_motion
        control.update(
            T_world_base=base.pose.ravel(),
            wheel_positions=base.wheel_positions,
            body_velocity=np.array(list(vars(base.velocity).values())),
        )
        state["simulation_control"] = {
            k: torch.tensor(v[None], dtype=torch.float64, device=self.device)
            for k, v in control.items()
        }
        state["episode_step"] = self._elapsed_steps[:, None].clone()
        return state

    def set_state_dict(self, state, env_idx=None):
        super().set_state_dict(state, env_idx)
        if "simulation_control" in state:
            values = {k: v[0].cpu().numpy().copy() for k, v in state["simulation_control"].items()}
            self.agent.position_control.restore(values)
            base = self.agent.base_motion
            base.pose = values["T_world_base"].reshape(4, 4)
            base.wheel_positions[:] = values["wheel_positions"]
            base.set_velocity(BodyVelocity(*map(float, values["body_velocity"])))
            controller = self.agent.controller.controllers["joints"]
            goal = [
                self.agent.position_control.goal[self.agent.position_control.names.index(n)]
                for n in controller.config.joint_names
            ]
            controller.set_action(torch.tensor([goal], dtype=torch.float32, device=self.device))
        if "episode_step" in state:
            self._elapsed_steps[:] = state["episode_step"].reshape(self._elapsed_steps.shape)

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)
