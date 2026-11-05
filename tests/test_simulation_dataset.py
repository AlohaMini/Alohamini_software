import json
import pickle
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from alohamini.datasets.simulation import SimulationSource, _digest
from alohamini.datasets.tools import check_dataset
from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.statistics import sample_arguments


def episode(root, *, reset=0, split="train"):
    """Small storage fixture; physics admission is covered in simulation tests."""
    root.mkdir(parents=True)
    names, units = ["elbow_joint", "vertical_move"], ["rad", "m"]
    actions = np.array([[0.1, 0.2], [0.2, 0.21], [0.3, 0.22]])
    positions = np.array([[0, 0.19], [0.09, 0.20], [0.18, 0.21]])
    rgb = np.empty((3, 32, 32, 3), dtype=np.uint8)
    rgb[:] = np.array([[[[240, 50, 10]]], [[[100, 90, 80]]], [[[20, 30, 220]]]])
    environment = {"scene": {"task_index": 8, "reset_index": reset}}
    with h5py.File(root / "episode.hdf5", "w") as file:
        file.attrs["env_args"] = json.dumps(environment)
        group = file.create_group("data/demo_0")
        group.attrs["joint_names"] = json.dumps(names)
        group.attrs["joint_units"] = json.dumps(units)
        group.attrs["language"] = "place the bowl on the plate"
        group.attrs["controller"] = json.dumps({"settings": {"bandwidth_hz": 2}})
        for key, data in {
            "actions": actions,
            "obs/joint_positions": positions,
            "obs/joint_velocities": np.zeros((3, 2)),
            "states": np.zeros((4, 7)),
            "simulation_time_s": [0.5, 0.55, 0.6, 0.65],
            "obs/agentview_rgb": rgb,
        }.items():
            group.create_dataset(key, data=data)
    manifest = {
        "format": "alohamini-sim-v1",
        "admitted": True,
        "split": split,
        "split_group": f"8:{reset}",
        "quality": {"passed": True, "replay": {"passed": True}},
        "environment": environment,
        "mujoco_version": "3.12.0",
        "collection_settings": {"fps": 20, "reset_settle_s": 0.5},
        "physics_steps_per_command": 50,
        "observation_rendering": {"geom_groups": [2], "site_groups": [], "frames": "none"},
        "episode": "episode.hdf5",
        "episode_sha256": _digest(root / "episode.hdf5"),
        "cameras": ["agentview"],
        "model_files": {"model/model.xml": "unit-physics"},
        "action_semantics": "absolute named URDF positions, zero-velocity 20 Hz holds",
        "sample_alignment": "obs(s_t), action(u_t) before physics",
    }
    (root / "dataset.json").write_text(json.dumps(manifest))
    return actions, positions, rgb


def batch(root, rows, *, complete=True):
    (root / "index.json").write_text(
        json.dumps(
            {
                "format": "alohamini-sim-batch-v1",
                "complete": complete,
                "reset_indices": list(range(len(rows))),
                "episodes": [{"dataset": name, "admitted": admitted} for name, admitted in rows],
            }
        )
    )


def test_simulation_loader_keeps_si_units_alignment_pixels_and_padding(tmp_path):
    actions, positions, rgb = episode(tmp_path / "sim")
    root = tmp_path / "sim"
    assert check_dataset(root, decode_images=True)["valid"]
    data = AlohaMiniDataset(
        root, episodes=[0], state="recorded", chunk_size=4, image_size=(32, 32), return_uint8=True
    )
    assert data.selection.units == ["rad", "m"]
    assert data.info["features"]["action"]["names"] == ["elbow_joint", "vertical_move"]
    first, last = data[0], data[2]
    np.testing.assert_array_equal(first["observation.state"], positions[0].astype(np.float32))
    np.testing.assert_array_equal(first["action"][0], actions[0].astype(np.float32))
    np.testing.assert_array_equal(first["observation.images.agentview"].permute(1, 2, 0), rgb[0])
    assert last["action_is_pad"].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(last["action"], np.tile(actions[-1].astype(np.float32), (4, 1)))
    assert data.records == [{}, {}, {}]
    assert data.info["training_only"]["source"] == "alohamini_mujoco"


def test_simulation_check_cli_uses_the_common_dataset_report(tmp_path, capsys):
    from alohamini.cli import main

    episode(tmp_path / "sim")
    assert main(["dataset", "check", str(tmp_path / "sim"), "--decode-images"]) == 0
    output = capsys.readouterr().out
    assert "Episodes=1 frames=3 pending=0" in output
    assert "Result: VALID" in output


def test_simulation_does_not_train_on_reserved_or_rejected_episodes(tmp_path):
    episode(tmp_path / "train", reset=0)
    episode(tmp_path / "evaluation", reset=40, split="evaluation")
    batch(tmp_path, [("train", True), ("evaluation", True), ("not-recorded", False)])
    source = SimulationSource(tmp_path, verify=True)
    assert source.info["total_episodes"] == 1
    assert source.info["episode_groups"] == ["8:0"]
    assert source.info["simulation"]["attempted_resets"] == 3
    batch(tmp_path, [("train", True)], complete=False)
    with pytest.raises(ValueError, match="completed reference batch"):
        SimulationSource(tmp_path)


def test_simulation_hdf_handles_are_not_pickled(tmp_path):
    _, _, rgb = episode(tmp_path / "sim")
    source = SimulationSource(tmp_path / "sim")
    reference = (str(tmp_path / "sim/episode.hdf5"), 1)
    np.testing.assert_array_equal(source.image(reference, "agentview"), rgb[1])
    restored = pickle.loads(pickle.dumps(source))
    np.testing.assert_array_equal(restored.image(reference, "agentview"), rgb[1])


def test_simulation_integrity_check_detects_corrupted_episode(tmp_path):
    episode(tmp_path / "sim")
    with (tmp_path / "sim/episode.hdf5").open("ab") as stream:
        stream.write(b"corrupt")
    report = check_dataset(tmp_path / "sim")
    assert not report["valid"]
    assert any("checksum" in issue["message"] for issue in report["issues"])


def test_simulation_selection_uses_recorded_state_and_keeps_reset_groups_together(tmp_path):
    episode(tmp_path / "a", reset=0)
    episode(tmp_path / "b", reset=0)
    batch(tmp_path, [("a", True), ("b", True)])
    components = SimpleNamespace(sample_spec=lambda options, **kwargs: {"chunk_size": 4})
    settings = {
        "dataset": str(tmp_path),
        "state": "auto",
        "train_episodes": [0],
        "val_episodes": [1],
    }
    with pytest.raises(ValueError, match="disjoint task/reset groups"):
        sample_arguments(settings, components, {})
    settings["val_episodes"] = []
    args, episodes, _ = sample_arguments(settings, components, {})
    assert args["state"] == "recorded" and episodes == [0]
