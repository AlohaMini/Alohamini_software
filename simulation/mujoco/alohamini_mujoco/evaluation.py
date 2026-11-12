"""Fixed-horizon policy trials using named SI joints and recorded-control replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import numpy as np

from .collection import (
    STATE_SPEC,
    Admission,
    CollectionSettings,
    _substeps,
    physics_state,
    replay_episode,
)
from .control import DynamicFollower, FollowerSettings
from .export import export_robot, model_xml
from .libero import SceneResetError, settle_scene, sha256
from .viewer import run_viewer, set_geometry


class PolicyPlayback:
    """Policy actions are held at 20 Hz; every physical step is measured."""

    is_policy = True

    def __init__(
        self,
        robot,
        action,
        *,
        reset_policy,
        follower_settings,
        horizon_s=60,
        paused=False,
        collection_settings=None,
    ):
        if (
            not math.isfinite(horizon_s)
            or horizon_s <= 0
            or not np.isclose(horizon_s * 20, round(horizon_s * 20))
        ):
            raise ValueError("Policy horizon must be positive and align with 20 Hz")
        self.robot, self.action, self.reset_policy = robot, action, reset_policy
        self.follower_settings = follower_settings
        self.settings = collection_settings or CollectionSettings()
        self.horizon_frames = round(horizon_s * self.settings.fps)
        self.steps = _substeps(robot, self.settings)
        self.reset()
        self.paused = paused

    @property
    def data(self):
        return self.follower.data

    def reset(self):
        self.follower = DynamicFollower(self.robot, settings=self.follower_settings)
        settle_scene(self.follower, self.settings.reset_settle_s)
        if self.robot.scene.goal_status(self.robot, self.data)["passed"]:
            raise SceneResetError("Evaluation initial goal is already satisfied")
        self.reset_policy()
        self.admission = Admission(self.robot, self.follower, self.settings)
        self.states = [physics_state(self.robot.model, self.data)]
        self.times = [float(self.data.time)]
        self.actions, self.target_clips = [], np.zeros(len(self.follower.names), dtype=int)
        self.maximum_target_violation = np.zeros(len(self.follower.names))
        self.frame_index = self.substep = self.saturated_steps = 0
        self.paused, self.completed, self.success = True, False, False
        self.failure = None

    def step(self, *, single=False):
        if self.completed or (self.paused and not single):
            return False
        if self.substep == 0:
            command = np.asarray(self.action(self.data), dtype=float)
            if command.shape != (len(self.follower.names),) or not np.isfinite(command).all():
                self.failure, self.completed, self.paused = "invalid_policy_action", True, True
                return False
            joints = [self.robot.description.joints[n] for n in self.follower.names]
            applied = np.clip(command, [j.lower for j in joints], [j.upper for j in joints])
            violation = np.abs(applied - command)
            self.target_clips += violation > 0
            self.maximum_target_violation = np.maximum(self.maximum_target_violation, violation)
            self.follower.target[:] = applied
            self.follower.target_velocity[:] = 0
            self.actions.append(applied.copy())
        saturated = self.follower.step()
        self.saturated_steps += int(np.any(saturated))
        self.admission.physics_step(saturated)
        self.substep += 1
        if self.substep == self.steps:
            mujoco.mj_forward(self.robot.model, self.data)
            self.admission.frame()
            self.states.append(physics_state(self.robot.model, self.data))
            self.times.append(float(self.data.time))
            self.frame_index, self.substep = self.frame_index + 1, 0
            quality = self.admission.report()
            safe = all(quality["checks"][n] for n in ("penetration", "joint_speed", "joint_limits"))
            self.success = bool(safe and quality["checks"]["stable_released_goal"])
            if not safe:
                self.failure = "physical_limit_violation"
            if self.success or self.failure or self.frame_index >= self.horizon_frames:
                self.completed, self.paused = True, True
                if not self.success and self.failure is None:
                    self.failure = "horizon_exceeded"
        return True

    def report(self):
        return {
            "initialization": {"passed": True},
            "success": self.success,
            "completed": self.completed,
            "failure": self.failure,
            "frames": self.frame_index,
            "quality": self.admission.report(),
            "target_projection": {
                name: {
                    "frames": int(count),
                    "maximum_violation": float(value),
                    "unit": "m" if kind == "prismatic" else "rad",
                }
                for name, count, value, kind in zip(
                    self.follower.names,
                    self.target_clips,
                    self.maximum_target_violation,
                    self.follower.kinds,
                    strict=True,
                )
                if count
            },
        }

    def interrupt(self):
        """Finish the current held command so the interrupted trace can replay."""
        while self.substep:
            self.step(single=True)
        if not self.completed:
            self.failure = "interactive_interrupted"
            self.completed = self.paused = True


def _save_trial(playback, output, policy_fingerprint, weights_fingerprint):
    import h5py

    report = playback.report()
    output.mkdir(parents=True, exist_ok=False)
    if playback.actions:
        robot, follower = playback.robot, playback.follower
        export_robot(robot, {"purpose": "policy evaluation"}, output / "model")
        with h5py.File(output / "episode.hdf5", "w") as file:
            file.attrs["env_args"] = json.dumps(robot.manifest)
            group = file.create_group("data/demo_0")
            group.attrs["joint_names"] = json.dumps(follower.names)
            group.attrs["controller"] = json.dumps(
                {
                    "settings": follower.settings.to_dict(),
                    "kp": follower.kp.tolist(),
                    "kd": follower.kd.tolist(),
                    "kinds": follower.kinds,
                }
            )
            group.create_dataset("states", data=playback.states, compression="gzip")
            group.create_dataset("actions", data=playback.actions)
            group.create_dataset("simulation_time_s", data=playback.times)
        manifest = {
            "format": "alohamini-sim-v1",
            "admitted": False,
            "split": "evaluation",
            "mujoco_version": mujoco.__version__,
            "environment": robot.manifest,
            "collection_settings": asdict(playback.settings),
            "physics_steps_per_command": playback.steps,
            "state_spec": int(STATE_SPEC),
            "model_files": {
                p.relative_to(output).as_posix(): sha256(p)
                for p in (output / "model").rglob("*")
                if p.is_file()
            },
            "episode": "episode.hdf5",
            "episode_sha256": sha256(output / "episode.hdf5"),
            "policy_manifest_sha256": policy_fingerprint,
            "policy_weights_sha256": weights_fingerprint,
        }
        (output / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
        report["replay"] = replay_episode(output)
        if not report["replay"]["passed"]:
            raise RuntimeError("Policy trial did not reproduce its recorded integration states")
    else:
        report["replay"] = {"performed": False, "reason": "no actions executed"}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def evaluate_policy(
    robot, checkpoint, reset_indices, output, *, horizon_s=60, device="cpu", view=False
):
    import torch

    from alohamini.learning.policy import NativePolicy
    from alohamini.learning.processor import image_tensor

    policy = NativePolicy(checkpoint, device=device, offline=True)
    weights_fingerprint = sha256(policy.checkpoint_path / "model.safetensors")
    source = policy.source
    if source.get("training_only", {}).get("source") != "alohamini_mujoco":
        raise ValueError("MuJoCo policy evaluation requires explicit simulation joint coordinates")
    follower = DynamicFollower(robot)
    units = ["m" if k == "prismatic" else "rad" for k in follower.kinds]
    if (
        policy.names != list(follower.names)
        or source["features"]["action"].get("units") != units
        or policy.fps != 20
    ):
        raise ValueError("Checkpoint action names, units or rate differ from the simulator")
    specification = source["simulation"]
    if (
        hashlib.sha256(model_xml(robot).encode()).hexdigest()
        not in specification["model_xml_sha256"]
    ):
        raise ValueError("Evaluation model differs from the checkpoint's recorded scene physics")
    if robot.initial_positions not in specification["initial_positions"]:
        raise ValueError("Evaluation robot initial pose differs from the training protocol")
    settings = FollowerSettings(**specification["follower_settings"])
    collection_settings = CollectionSettings(**specification["collection_settings"])
    cameras = policy.manifest["cameras"]
    supported = {"action", "observation.state", *(f"observation.images.{c}" for c in cameras)}
    if policy.field_transform or any(
        key not in supported or (key != "action" and offsets != [0])
        for key, offsets in policy.manifest.get("sample_windows", {}).items()
    ):
        raise ValueError(
            "This evaluator requires current observations and absolute SI joint actions"
        )
    resets = list(reset_indices)
    if not resets or len(set(resets)) != len(resets):
        raise ValueError("Evaluation requires distinct initial-state indices")
    if view and len(resets) != 1:
        raise ValueError("Interactive evaluation requires one reset; use --reset-indices N")
    shapes = [source["features"][f"observation.images.{c}"]["shape"][:2] for c in cameras]
    if not shapes or any(shape != shapes[0] for shape in shapes):
        raise ValueError("Evaluation requires identically sized recorded cameras")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    options = mujoco.MjvOption()
    set_geometry(options, collisions=False)
    options.sitegroup[:] = 0
    report = {
        "protocol": (
            "AlohaMini LIBERO adaptation; original On predicate, stable released goal for 0.5 s"
        ),
        "horizon_s": horizon_s,
        "fps": 20,
        "task_index": robot.scene.task_index,
        "reset_indices": resets,
        "requested_trials": len(resets),
        "complete": False,
        "interactive": view,
        "action_guard": (
            "targets projected to URDF position bounds; "
            "physical effort/speed/contact checks retained"
        ),
        "policy_manifest_sha256": policy.checkpoint_manifest_sha256,
        "policy_weights_sha256": weights_fingerprint,
        "trials": [],
        "device": str(device),
        "torch_version": torch.__version__,
    }
    from .cameras import CameraCapture

    with CameraCapture(robot, cameras, height=shapes[0][0], width=shapes[0][1]) as capture:

        def action(data):
            sample = {}
            if policy.selection:
                sample["observation.state"] = torch.from_numpy(
                    policy.selection.frame({"observation.state": data.qpos[follower.qpos_ids]})
                )
            for camera in policy.manifest["cameras"]:
                sample[f"observation.images.{camera}"] = image_tensor(
                    capture.image(data, camera, options),
                    policy.manifest["image_size"],
                    resize_mode=policy.manifest.get("image_resize_mode", "stretch"),
                    pad_value=policy.manifest.get("image_pad_value", 0.0),
                )
            if policy.algorithm.include_task:
                sample["task"] = robot.scene.manifest["language"]
            batch = policy.processor(
                {k: [v] if k == "task" else v[None] for k, v in sample.items()}
            )
            with torch.inference_mode(), policy.inference_context():
                normalized = policy.model.select_action(batch)
                return (
                    policy.execution_action(normalized, context=batch.get("_action_context"))[0]
                    .float()
                    .cpu()
                    .numpy()
                )

        for reset in resets:
            scene = replace(robot.scene, reset_index=reset)
            current = replace(
                robot, scene=scene, manifest={**robot.manifest, "scene": scene.manifest}
            )
            directory = output / f"reset_{reset:03d}"
            try:
                playback = PolicyPlayback(
                    current,
                    action,
                    reset_policy=policy.reset,
                    follower_settings=settings,
                    horizon_s=horizon_s,
                    paused=view,
                    collection_settings=collection_settings,
                )
            except SceneResetError as exc:
                result = {
                    "initialization": {"passed": False, "error": str(exc)},
                    "success": False,
                    "completed": True,
                    "failure": "scene_reset_rejected",
                    "frames": 0,
                    "replay": {"performed": False, "reason": "no policy actions executed"},
                }
                directory.mkdir()
                (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            else:
                if view:
                    run_viewer(current, playback=playback, paused=True)
                    playback.interrupt()
                else:
                    while not playback.completed:
                        playback.step(single=True)
                result = _save_trial(
                    playback,
                    directory,
                    policy.checkpoint_manifest_sha256,
                    weights_fingerprint,
                )
            report["trials"].append({"reset_index": reset, **result})
            report["successes"] = sum(r["success"] for r in report["trials"])
            report["completed_trials"] = len(report["trials"])
            report["complete"] = report["completed_trials"] == len(resets)
            report["reset_failures"] = sum(
                r["failure"] == "scene_reset_rejected" for r in report["trials"]
            )
            report["success_rate"] = report["successes"] / len(resets)
            (output / "benchmark.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
            print(
                json.dumps(
                    {
                        "reset_index": reset,
                        "success": result["success"],
                        "failure": result["failure"],
                    }
                ),
                flush=True,
            )
    return report
