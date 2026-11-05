"""Read admitted MuJoCo episodes in their named physical joint coordinates."""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np


def is_simulation_dataset(root):
    root = Path(root).expanduser()
    for name, expected in (
        ("index.json", "alohamini-sim-batch-v1"),
        ("dataset.json", "alohamini-sim-v1"),
    ):
        path = root / name
        if path.is_file():
            specification = json.loads(path.read_text())
            if isinstance(specification, dict) and specification.get("format") == expected:
                return True
    return False


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Simulation file is outside its dataset: {relative}")
    return path


class SimulationSource:
    """HDF5 storage adapter; windows and normalization remain in the common loader.

    No Host timing, motor normalization or calibration is synthesized. Only
    admitted training episodes are exposed; original task/reset groups remain
    explicit. Images are read unchanged from obs(s_t), before action(u_t).
    """

    unannotated = True  # No Host feedback or Host-clock sidecar exists in simulation.

    def __init__(self, root, *, verify=False, decode_images=False):
        import h5py

        self.root = Path(root).expanduser().resolve()
        self.episodes = {}
        self.metadata_paths = []
        self._local = threading.local()
        path = self.root / "index.json"
        if path.is_file():
            index = json.loads(path.read_text())
            if index.get("format") != "alohamini-sim-batch-v1" or not index.get("complete"):
                raise ValueError("Simulation training requires a completed reference batch")
            rows = index["episodes"]
            self.metadata_paths.append(path)
            self.attempted = len(index["reset_indices"])
        else:
            path = self.root / "dataset.json"
            manifest = json.loads(path.read_text())
            rows = [{"dataset": ".", "admitted": manifest["admitted"]}]
            self.attempted = 1
        names = units = cameras = shapes = semantics = None
        groups, model_hashes, total, versions = [], set(), 0, set()
        control = None
        initial_poses = []
        for row in rows:
            if not row.get("admitted") or "dataset" not in row:
                continue
            directory = _contained(self.root, row["dataset"])
            manifest_path = directory / "dataset.json"
            if (
                verify
                and row.get("dataset_sha256")
                and _digest(manifest_path) != row["dataset_sha256"]
            ):
                raise ValueError("Simulation dataset manifest checksum mismatch")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("format") != "alohamini-sim-v1" or not manifest.get("admitted"):
                raise ValueError("Simulation samples must come from admitted episodes")
            if manifest["split"] != "train":
                continue
            quality = manifest["quality"]
            if not quality["passed"] or not quality["replay"]["passed"]:
                raise ValueError("Simulation episode did not pass task/dynamics replay admission")
            episode_path = _contained(directory, manifest["episode"])
            if verify and _digest(episode_path) != manifest["episode_sha256"]:
                raise ValueError("Simulation episode checksum mismatch")
            if manifest["collection_settings"]["fps"] != 20:
                raise ValueError("Simulation episodes must declare the 20 Hz command protocol")
            with h5py.File(episode_path, "r") as file:
                if json.loads(file.attrs["env_args"]) != manifest["environment"]:
                    raise ValueError("Simulation episode environment differs from its manifest")
                data = file["data/demo_0"]
                episode_names = json.loads(data.attrs["joint_names"])
                episode_units = json.loads(data.attrs["joint_units"])
                episode_control = {
                    "follower_settings": json.loads(data.attrs["controller"])["settings"],
                    "collection_settings": manifest["collection_settings"],
                    "physics_steps_per_command": manifest["physics_steps_per_command"],
                    "reset_settle_s": manifest["collection_settings"]["reset_settle_s"],
                    "observation_rendering": manifest["observation_rendering"],
                }
                if control is None:
                    control = episode_control
                elif episode_control != control:
                    raise ValueError(
                        "Simulation episodes use different controller or rendering protocols"
                    )
                count = len(data["actions"])
                episode_cameras = manifest["cameras"]
                episode_shapes = {c: list(data[f"obs/{c}_rgb"].shape[1:]) for c in episode_cameras}
                layout = (
                    episode_names,
                    episode_units,
                    episode_cameras,
                    episode_shapes,
                    manifest["action_semantics"],
                    manifest["sample_alignment"],
                )
                if semantics is None:
                    names, units, cameras, shapes = deepcopy(layout[:4])
                    semantics = layout[4:]
                elif layout != (names, units, cameras, shapes, *semantics):
                    raise ValueError("Simulation episodes have incompatible named fields or timing")
                if (
                    not names
                    or len(set(names)) != len(names)
                    or len(units) != len(names)
                    or any(unit not in ("m", "rad") for unit in units)
                    or data["actions"].shape != (count, len(names))
                    or data["obs/joint_positions"].shape != (count, len(names))
                    or data["obs/joint_velocities"].shape != (count, len(names))
                    or len(data["states"]) != count + 1
                    or len(data["simulation_time_s"]) != count + 1
                ):
                    raise ValueError(
                        "Simulation numeric fields have inconsistent names/units/shapes"
                    )
                for c in cameras:
                    image = data[f"obs/{c}_rgb"]
                    if (
                        image.shape != (count, *shapes[c])
                        or len(shapes[c]) != 3
                        or shapes[c][-1] != 3
                        or image.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"Simulation camera {c} must contain N lossless RGB frames"
                        )
                    if decode_images:
                        image[0]
                        image[-1]
                if verify:
                    for key in ("actions", "obs/joint_positions", "obs/joint_velocities", "states"):
                        if not np.isfinite(data[key][:]).all():
                            raise ValueError(f"Simulation {key} contains non-finite values")
                    if not np.allclose(
                        np.diff(data["simulation_time_s"][:]), 0.05, rtol=0, atol=1e-10
                    ):
                        raise ValueError("Simulation action and observation times do not use 20 Hz")
                language = str(data.attrs["language"])
            group = manifest["split_group"]
            groups.append(group)
            initial = manifest["environment"].get("initial_positions")
            if initial is not None and initial not in initial_poses:
                initial_poses.append(deepcopy(initial))
            model_hashes.add(manifest["model_files"]["model/model.xml"])
            versions.add(manifest["mujoco_version"])
            self.episodes[len(self.episodes)] = {
                "path": episode_path,
                "length": count,
                "language": language,
                "split_group": group,
                "manifest": manifest_path,
            }
            total += count
            self.metadata_paths.append(manifest_path)
        if not self.episodes:
            raise ValueError("Simulation dataset has no admitted training episodes")
        self.cameras = cameras
        position = {"dtype": "float32", "shape": [len(names)], "names": names, "units": units}
        features = {
            "observation.state": deepcopy(position),
            "action": deepcopy(position),
            "observation.joint_velocity": {
                **deepcopy(position),
                "units": ["m/s" if u == "m" else "rad/s" for u in units],
            },
            **{
                f"observation.images.{c}": {
                    "dtype": "image",
                    "shape": shapes[c],
                    "names": ["height", "width", "channels"],
                }
                for c in cameras
            },
        }
        self.info = {
            "format": "alohamini-simulation",
            "fps": 20,
            "features": features,
            "cameras": cameras,
            "total_episodes": len(self.episodes),
            "total_frames": total,
            "episode_groups": groups,
            "training_only": {
                "source": "alohamini_mujoco",
                "action_units": "explicit URDF absolute joint positions in rad/m",
                "calibration": "simulation joint coordinates; no Host motor calibration",
                "time_basis": "simulation time; obs(s_t) before action(u_t); no Host timestamps",
            },
            "simulation": {
                "mujoco_versions": sorted(versions),
                "model_xml_sha256": sorted(model_hashes),
                "attempted_resets": self.attempted,
                "admitted_training_episodes": len(self.episodes),
                "training_reset_groups": len(set(groups)),
                "action_semantics": semantics[0],
                "sample_alignment": semantics[1],
                "initial_positions": initial_poses,
                **control,
            },
        }
        self.original_state = deepcopy(features["observation.state"])

    def report(self):
        return {
            "dataset_root": str(self.root),
            "valid": True,
            "training_review": "not_assessed",
            "errors": 0,
            "warnings": 0,
            "issues": [],
            "summary": {
                "episodes": len(self.episodes),
                "frames": self.info["total_frames"],
                "pending_episodes": 0,
            },
        }

    def read_episodes(self, episodes, columns):
        import h5py

        sources = {
            "action": "actions",
            "observation.state": "obs/joint_positions",
            "observation.joint_velocity": "obs/joint_velocities",
        }
        for number in episodes:
            metadata = self.episodes[number]
            with h5py.File(metadata["path"], "r") as file:
                data = file["data/demo_0"]
                arrays = {
                    key: data[sources[key]][:].astype(np.float32)
                    for key in set(columns)
                    if key in sources
                }
                rows = []
                for frame in range(metadata["length"]):
                    row = {"episode_index": number, "frame_index": frame}
                    for key in set(columns):
                        if key in arrays:
                            row[key] = arrays[key][frame]
                        elif key == "task":
                            row[key] = metadata["language"]
                        elif key.startswith("observation.images."):
                            row[key] = (str(metadata["path"]), frame)
                        else:
                            raise ValueError(f"Simulation field is not recorded: {key}")
                    rows.append(row)
            yield rows, None, [*self.metadata_paths, metadata["path"]]

    def image(self, reference, camera, *, video_reader=None):
        import h5py

        files = getattr(self._local, "files", None)
        if files is None:
            files = self._local.files = OrderedDict()
        path, frame = reference
        handle = files.pop(path, None)
        if handle is None:
            handle = h5py.File(path, "r", rdcc_nbytes=64 * 1024)
        files[path] = handle
        while len(files) > 4:
            files.popitem(last=False)[1].close()
        return np.array(handle[f"data/demo_0/obs/{camera}_rgb"][frame], copy=True)

    def __getstate__(self):
        return {key: value for key, value in self.__dict__.items() if key != "_local"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()
