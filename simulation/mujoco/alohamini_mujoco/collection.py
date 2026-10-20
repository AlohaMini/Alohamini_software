"""Twenty-Hz joint-command episodes with lossless images and dynamics replay admission."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from .control import DynamicFollower, FollowerSettings
from .libero import sha256
from .viewer import set_geometry

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def physics_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def restore_state(model, data, state):
    mujoco.mj_setState(model, data, state, STATE_SPEC)
    mujoco.mj_forward(model, data)
    # Rebuild derived fields without changing the recorded warm-start state.
    mujoco.mj_setState(model, data, state, STATE_SPEC)


@dataclass(frozen=True)
class CollectionSettings:
    fps: int = 20
    image_size: int = 128
    reset_settle_s: float = 0.5
    final_stability_s: float = 0.5
    minimum_lift_m: float = 0.02
    grasp_contact_s: float = 0.1
    maximum_penetration_m: float = 0.0005
    maximum_final_linear_speed_m_s: float = 0.02
    maximum_final_angular_speed_rad_s: float = 0.2
    replay_tolerance: float = 1e-8

    def __post_init__(self):
        if type(self.fps) is not int or self.fps != 20:
            raise ValueError("This collection protocol requires 20 Hz commands")
        if type(self.image_size) is not int or self.image_size < 32:
            raise ValueError("Image size must be an integer >= 32")
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Collection settings must be finite and positive")


def _substeps(robot, settings):
    steps = 1 / (settings.fps * robot.model.opt.timestep)
    if not np.isclose(steps, round(steps), rtol=0, atol=1e-10):
        raise ValueError("The command period must contain an integer number of physics steps")
    return int(round(steps))


class Admission:
    """Quality criteria supplement the unmodified upstream task predicate."""

    def __init__(self, robot, follower, settings):
        self.robot, self.follower, self.settings = robot, follower, settings
        self.sources = [g[1] for g in robot.scene.manifest["goals"]]
        self.initial_height = {
            name: float(follower.data.body(robot.scene.task["objects"][name]["root_body"]).xpos[2])
            for name in self.sources
        }
        self.object_geoms = {
            name: {
                robot.model.geom(g).id for g in robot.scene.task["objects"][name]["contact_geoms"]
            }
            for name in self.sources
        }
        self.jaws = {
            side: {
                jaw: {
                    i
                    for i in range(robot.model.ngeom)
                    if robot.model.body(robot.model.geom_bodyid[i]).name == f"{side}_{jaw}"
                }
                for jaw in ("Fixed_Jaw", "Moving_Jaw")
            }
            for side in ("left", "right")
        }
        self.grasp_run = {name: 0.0 for name in self.sources}
        self.grasped = {name: False for name in self.sources}
        self.lifted = {name: False for name in self.sources}
        self.minimum_distance = 0.0
        self.maximum_speed_ratio = 0.0
        self.maximum_limit_violation = 0.0
        self.saturated_steps = 0
        self.stable_frames = 0
        self.final_goal = None

    def physics_step(self, saturated):
        f, model, data = self.follower, self.robot.model, self.follower.data
        self.saturated_steps += int(np.any(saturated))
        limits = np.array([self.robot.description.joints[n].velocity for n in f.names])
        self.maximum_speed_ratio = max(
            self.maximum_speed_ratio, float(np.max(np.abs(data.qvel[f.dof_ids]) / limits))
        )
        for i, name in enumerate(f.names):
            joint = self.robot.description.joints[name]
            q = data.qpos[f.qpos_ids[i]]
            tolerance = 0.0005 if joint.kind == "prismatic" else 0.005
            self.maximum_limit_violation = max(
                self.maximum_limit_violation,
                (joint.lower - q) / tolerance,
                (q - joint.upper) / tolerance,
            )
        self.minimum_distance = min(
            self.minimum_distance, min((c.dist for c in data.contact), default=0.0)
        )
        for name, geoms in self.object_geoms.items():
            touching = set()
            for contact in data.contact:
                if contact.geom1 in geoms:
                    touching.add(contact.geom2)
                if contact.geom2 in geoms:
                    touching.add(contact.geom1)
            bilateral = any(
                all(touching & jaw for jaw in jaws.values()) for jaws in self.jaws.values()
            )
            self.grasp_run[name] = self.grasp_run[name] + model.opt.timestep if bilateral else 0.0
            self.grasped[name] |= self.grasp_run[name] + 1e-12 >= self.settings.grasp_contact_s
            height = float(data.body(self.robot.scene.task["objects"][name]["root_body"]).xpos[2])
            self.lifted[name] |= bool(
                bilateral and height - self.initial_height[name] >= self.settings.minimum_lift_m
            )

    def frame(self):
        robot, data, settings = self.robot, self.follower.data, self.settings
        self.final_goal = robot.scene.goal_status(robot, data)
        stable = self.final_goal["passed"]
        for name in self.sources:
            obj = robot.scene.task["objects"][name]
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(
                robot.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                robot.model.body(obj["root_body"]).id,
                velocity,
                0,
            )
            stable &= np.linalg.norm(velocity[:3]) <= settings.maximum_final_angular_speed_rad_s
            stable &= np.linalg.norm(velocity[3:]) <= settings.maximum_final_linear_speed_m_s
            # Release requires absence of all robot contacts, not just an open command.
            for contact in data.contact:
                other = (
                    contact.geom2
                    if contact.geom1 in self.object_geoms[name]
                    else contact.geom1
                    if contact.geom2 in self.object_geoms[name]
                    else None
                )
                if other is not None and robot.model.geom(other).name in robot.geometry:
                    stable = False
        self.stable_frames = self.stable_frames + 1 if stable else 0

    def report(self):
        checks = {
            "task_success": bool(self.final_goal and self.final_goal["passed"]),
            "stable_released_goal": self.stable_frames / self.settings.fps + 1e-12
            >= self.settings.final_stability_s,
            "bilateral_grasp": all(self.grasped.values()),
            "lifted_while_grasped": all(self.lifted.values()),
            "penetration": self.minimum_distance >= -self.settings.maximum_penetration_m,
            "joint_speed": self.maximum_speed_ratio <= 1 + 1e-6,
            "joint_limits": self.maximum_limit_violation <= 1,
        }
        checks = {name: bool(value) for name, value in checks.items()}
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "final_goal": self.final_goal,
            "minimum_contact_distance_m": self.minimum_distance,
            "maximum_joint_speed_ratio": self.maximum_speed_ratio,
            "maximum_joint_limit_violation_in_tolerances": float(self.maximum_limit_violation),
            "saturated_physics_steps": self.saturated_steps,
            "bilateral_grasp_by_object": self.grasped,
            "lift_by_object": self.lifted,
        }


def replay_episode(directory, *, episode=None):
    import h5py

    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "dataset.json").read_text())
    if manifest["mujoco_version"] != mujoco.__version__:
        raise ValueError("Replay requires the recorded MuJoCo version")
    for relative, expected in manifest["model_files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or sha256(path) != expected:
            raise ValueError(f"Model checksum mismatch: {relative}")
    model = mujoco.MjModel.from_xml_path(str(directory / "model/model.xml"))
    episode = Path(episode) if episode else directory / manifest["episode"]
    if manifest.get("episode_sha256") and sha256(episode) != manifest["episode_sha256"]:
        raise ValueError("Episode checksum mismatch")
    with h5py.File(episode, "r") as file:
        group = file["data/demo_0"]
        specification = json.loads(group.attrs["controller"])
        names = json.loads(group.attrs["joint_names"])
        if names != [model.actuator(i).name for i in range(model.nu)]:
            raise ValueError("Episode actuator order differs from its model")
        data = mujoco.MjData(model)
        restore_state(model, data, group["states"][0])
        # Replay only needs drive types, limits and the saved gains, not source assets.
        joints = {
            name: SimpleNamespace(kind=kind)
            for name, kind in zip(names, specification["kinds"], strict=True)
        }
        robot = SimpleNamespace(model=model, description=SimpleNamespace(joints=joints))
        follower = DynamicFollower(
            robot, data=data, settings=FollowerSettings(**specification["settings"])
        )
        follower.kp[:] = specification["kp"]
        follower.kd[:] = specification["kd"]
        restore_state(model, data, group["states"][0])
        maximum_error = 0.0
        for index, command in enumerate(group["actions"]):
            follower.target[:] = command
            follower.target_velocity[:] = 0
            for _ in range(manifest["physics_steps_per_command"]):
                follower.step()
            mujoco.mj_forward(model, data)
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(physics_state(model, data) - group["states"][index + 1]))),
            )
        return {
            "passed": maximum_error <= manifest["collection_settings"]["replay_tolerance"],
            "maximum_state_error": maximum_error,
            "frames": len(group["actions"]),
        }


def collect_episode(
    robot, trajectory, output, *, settings=None, cameras=("agentview", "left_wrist", "right_wrist")
):
    import h5py

    from .cli import export_robot

    settings = settings or CollectionSettings()
    if robot.scene is None:
        raise ValueError("Formal collection requires a task scene and success predicates")
    validation = robot.scene.validate(robot)
    if not validation["passed"]:
        raise ValueError(f"Scene equivalence failed: {validation['errors']}")
    if not validation["static_robot_scene"]["passed"]:
        raise ValueError(
            "Fixed robot/scene geometry intersects: "
            f"{validation['static_robot_scene']['intersections']}"
        )
    steps = _substeps(robot, settings)
    if len(set(cameras)) != len(cameras) or not cameras:
        raise ValueError("Use distinct named cameras")
    for camera in cameras:
        robot.model.camera(camera)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    follower = DynamicFollower(robot)
    for _ in range(math.ceil(settings.reset_settle_s / robot.model.opt.timestep)):
        follower.step()
    mujoco.mj_forward(robot.model, follower.data)
    if robot.scene.goal_status(robot, follower.data)["passed"]:
        raise ValueError("Initial goal is already satisfied; no demonstration was collected")
    indices = follower.trajectory_indices(trajectory)
    admission = Admission(robot, follower, settings)
    duration = float(trajectory.times[-1]) + follower.settings.settle_time_s
    frame_count = math.ceil(duration * settings.fps)
    begin = float(follower.data.time)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "episode"
        stage.mkdir()
        export_robot(robot, {"scene": validation}, stage / "model")
        trajectory_specification = {
            "schema_version": 1,
            "joint_names": list(trajectory.names),
            "times_s": trajectory.times.tolist(),
            "positions": trajectory.positions.tolist(),
        }
        (stage / "trajectory.json").write_text(
            json.dumps(trajectory_specification, indent=2) + "\n"
        )
        pending = stage / "attempt.hdf5"
        with h5py.File(pending, "w") as file:
            file.attrs["format"] = "alohamini-sim-v1"
            file.attrs["env_args"] = json.dumps(robot.manifest, allow_nan=False)
            group = file.create_group("data/demo_0")
            group.attrs["joint_names"] = json.dumps(follower.names)
            group.attrs["joint_units"] = json.dumps(
                ["m" if k == "prismatic" else "rad" for k in follower.kinds]
            )
            group.attrs["language"] = robot.scene.manifest["language"]
            group.attrs["controller"] = json.dumps(
                {
                    "kp": follower.kp.tolist(),
                    "kd": follower.kd.tolist(),
                    "kinds": follower.kinds,
                    "settings": asdict(follower.settings),
                }
            )
            states = group.create_dataset(
                "states",
                (frame_count + 1, len(physics_state(robot.model, follower.data))),
                dtype="f8",
                compression="gzip",
            )
            actions = group.create_dataset(
                "actions", (frame_count, len(follower.names)), dtype="f8"
            )
            times = group.create_dataset("simulation_time_s", (frame_count + 1,), dtype="f8")
            positions = group.create_dataset("obs/joint_positions", actions.shape, dtype="f8")
            velocities = group.create_dataset("obs/joint_velocities", actions.shape, dtype="f8")
            images, extrinsics = {}, {}
            for camera in cameras:
                images[camera] = group.create_dataset(
                    f"obs/{camera}_rgb",
                    (frame_count, settings.image_size, settings.image_size, 3),
                    dtype="u1",
                    chunks=(1, settings.image_size, settings.image_size, 3),
                    compression="gzip",
                )
                extrinsics[camera] = group.create_dataset(
                    f"camera/{camera}/T_world_camera", (frame_count, 4, 4), dtype="f8"
                )
                focal = settings.image_size / (
                    2 * math.tan(math.radians(float(robot.model.camera(camera).fovy[0])) / 2)
                )
                group.create_dataset(
                    f"camera/{camera}/K",
                    data=[
                        [focal, 0, settings.image_size / 2],
                        [0, focal, settings.image_size / 2],
                        [0, 0, 1],
                    ],
                )
            states[0], times[0] = physics_state(robot.model, follower.data), follower.data.time
            options = mujoco.MjvOption()
            set_geometry(options, collisions=False)
            with mujoco.Renderer(
                robot.model,
                height=settings.image_size,
                width=settings.image_size,
                max_geom=max(10000, robot.model.ngeom + 1000),
            ) as renderer:
                for frame in range(frame_count):
                    q, _ = trajectory.reference(frame / settings.fps)
                    follower.target[indices] = q
                    follower.target_velocity[:] = 0
                    actions[frame] = follower.target
                    positions[frame] = follower.data.qpos[follower.qpos_ids]
                    velocities[frame] = follower.data.qvel[follower.dof_ids]
                    for camera in cameras:
                        renderer.update_scene(follower.data, camera=camera, scene_option=options)
                        images[camera][frame] = renderer.render()
                        transform = np.eye(4)
                        cam = follower.data.camera(camera)
                        transform[:3, :3], transform[:3, 3] = cam.xmat.reshape(3, 3), cam.xpos
                        extrinsics[camera][frame] = transform
                    for _ in range(steps):
                        admission.physics_step(follower.step())
                    mujoco.mj_forward(robot.model, follower.data)
                    admission.frame()
                    states[frame + 1], times[frame + 1] = (
                        physics_state(robot.model, follower.data),
                        follower.data.time,
                    )
            report = admission.report()
            group.attrs["num_samples"] = frame_count
            group.attrs["quality"] = json.dumps(report, allow_nan=False)
            group.attrs["initial_simulation_time_s"] = begin
        manifest = {
            "format": "alohamini-sim-v1",
            "mujoco_version": mujoco.__version__,
            "collection_settings": asdict(settings),
            "physics_steps_per_command": steps,
            "state_spec": int(STATE_SPEC),
            "cameras": list(cameras),
            "action_semantics": (
                "absolute URDF joint position command, held for 1/20 s; "
                "reference velocity zero; all uncommanded joints held"
            ),
            "sample_alignment": (
                "obs(s_t), action(u_t), state(s_t) before physics; states has N+1 entries"
            ),
            "images": (
                "lossless RGB, uint8, top-left origin; MuJoCo Renderer output without extra flip"
            ),
            "camera_frame": (
                "OpenGL: X right, Y up, -Z forward; K uses pixel centers at half-integers"
            ),
            "split": "train" if robot.scene.reset_index < 40 else "evaluation",
            "split_group": f"{robot.scene.task_index}:{robot.scene.reset_index}",
            "environment": robot.manifest,
            "model_files": {
                p.relative_to(stage).as_posix(): sha256(p)
                for p in (stage / "model").rglob("*")
                if p.is_file()
            },
            "episode": "attempt.hdf5",
            "admitted": False,
        }
        (stage / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        report["replay"] = replay_episode(stage)
        report["passed"] = report["passed"] and report["replay"]["passed"]
        with h5py.File(pending, "r+") as file:
            file["data/demo_0"].attrs["quality"] = json.dumps(report, allow_nan=False)
            file["data/demo_0"].attrs["admitted"] = report["passed"]
        destination = Path("episodes" if report["passed"] else "rejected") / "episode_000000.hdf5"
        (stage / destination.parent).mkdir()
        pending.rename(stage / destination)
        manifest.update(
            episode=destination.as_posix(),
            admitted=report["passed"],
            episode_sha256=sha256(stage / destination),
            quality=report,
        )
        (stage / "dataset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return {"passed": report["passed"], "output": str(output), "quality": report}
