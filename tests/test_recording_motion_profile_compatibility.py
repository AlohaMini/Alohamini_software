"""Runtime motion profiles may change without changing recording coordinates."""

import contextlib
import json
from copy import deepcopy
from unittest.mock import patch

import pytest
from test_dataset import frame, jpeg
from test_dataset_tools import hashes
from test_recording_stats_resume import assert_exact_numeric_stats, no_history_content
from test_recording_storage import metadata

from alohamini.datasets.lerobotv3 import recover_recording
from alohamini.datasets.record import LocalDataset, _EpisodeWriter
from alohamini.datasets.tools import check_dataset


def profiled_metadata(*, accelerated=False):
    result = metadata()
    ordinary = [name for name in result["motors"] if not name.endswith("gripper")]
    result["arm_profile"] = {
        "arm_goal_velocity": 2400 if accelerated else 2000,
        "arm_acceleration": 120 if accelerated else 100,
    }
    if accelerated:
        result["arm_profile"]["motor_names"] = ordinary
    for index, (name, motor) in enumerate(result["motors"].items(), 1):
        motor.update(id=index, model="sts3215", homing_offset=0)
        if accelerated:
            motor["motion_profile"] = {
                "goal_velocity": 2000 if name.endswith("gripper") else 2400,
                "acceleration": 100 if name.endswith("gripper") else 120,
            }
    return result


def add_episode(dataset, count=2):
    dataset.begin_episode()
    for index in range(count):
        values = frame(dataset)
        values["action"][:] = index
        # Exercise both the application-supplied snapshot and writer-owned fallback.
        record = {"robot_metadata": deepcopy(dataset.robot_metadata)} if index % 2 else {}
        assert dataset.add_frame(values, {"forward": jpeg()}, record)


def create(root, *, native=False):
    writer = _EpisodeWriter if native else LocalDataset
    with contextlib.closing(
        writer(root, fps=30, task="pick", robot_metadata=profiled_metadata())
    ) as dataset:
        add_episode(dataset)
        dataset.save_episode()
    return root


def frame_metadata(path):
    return [
        row["robot_metadata"]
        for row in map(json.loads, path.read_text().splitlines())
        if row.get("frame_index") is not None
    ]


def test_v3_profile_resume_save_preserves_history_and_exact_statistics_cache(tmp_path):
    root = create(tmp_path / "record")
    original = json.loads((root / "meta/alohamini.json").read_text())["source_info"]
    committed = hashes(root)
    current = profiled_metadata(accelerated=True)
    profiles = [current, deepcopy(current)]
    profiles[1].pop("arm_profile")
    for motor in profiles[1]["motors"].values():
        motor.pop("motion_profile")

    for episode, live in enumerate(profiles, 1):
        with (
            no_history_content(root),
            contextlib.closing(
                LocalDataset(root, fps=30, task="pick", robot_metadata=live, resume=True)
            ) as dataset,
        ):
            assert dataset.robot_metadata == live
            assert dataset._recording_info == original
            assert json.loads((root / ".recording/meta/info.json").read_text()) == original
            add_episode(dataset)
            dataset.save_episode()
        assert frame_metadata(root / f"meta/safety/episode_{episode:06d}.jsonl") == [live, live]
        assert json.loads((root / "meta/alohamini.json").read_text())["source_info"] == original

    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (3, 6)
    assert_exact_numeric_stats(root)
    report = check_dataset(root, decode_videos=True)
    assert report["valid"], report
    current_files = hashes(root)
    for path, digest in committed.items():
        if path.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
            assert current_files[path] == digest


def test_native_profile_resume_keeps_manifest_and_records_current_metadata(tmp_path):
    root = create(tmp_path / "native", native=True)
    original = (root / "meta/info.json").read_bytes()
    live = profiled_metadata(accelerated=True)
    with contextlib.closing(
        _EpisodeWriter(root, fps=30, task="pick", robot_metadata=live, resume=True)
    ) as dataset:
        add_episode(dataset)
        dataset.save_episode()
    assert (root / "meta/info.json").read_bytes() == original
    assert frame_metadata(root / "episodes/episode_000001/safety.jsonl") == [live, live]
    report = check_dataset(root, decode_videos=True)
    assert report["valid"], report


def test_previous_failed_profile_resume_empty_staging_uses_historical_manifest(tmp_path):
    root = create(tmp_path / "record")
    source = json.loads((root / "meta/alohamini.json").read_text())["source_info"]
    live = profiled_metadata(accelerated=True)
    # The former strict LocalDataset comparison failed after making this directory.
    with contextlib.closing(
        _EpisodeWriter(root / ".recording", fps=30, task="pick", robot_metadata=live)
    ):
        pass
    staging_info = root / ".recording/meta/info.json"
    assert json.loads(staging_info.read_text())["robot_metadata"] == live
    with contextlib.closing(
        LocalDataset(root, fps=30, task="pick", robot_metadata=live, resume=True)
    ) as dataset:
        assert json.loads(staging_info.read_text()) == source
        add_episode(dataset)
        dataset.save_episode()
    assert json.loads((root / "meta/alohamini.json").read_text())["source_info"] == source
    assert frame_metadata(root / "meta/safety/episode_000001.jsonl") == [live, live]


@pytest.mark.parametrize("failure", ["encode", "publish"])
def test_profile_resume_interrupted_save_recovers_without_rewriting_source_info(tmp_path, failure):
    root = create(tmp_path / "record")
    source = json.loads((root / "meta/alohamini.json").read_text())["source_info"]
    committed = hashes(root)
    live = profiled_metadata(accelerated=True)
    dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=live, resume=True)
    try:
        add_episode(dataset)
        target = (
            "alohamini.datasets.video._encode_frames"
            if failure == "encode"
            else "alohamini.datasets.lerobotv3.publish_recorded_episode"
        )
        with patch(target, side_effect=OSError("interrupted save")):
            with pytest.raises(OSError, match="interrupted save"):
                dataset.save_episode()
    finally:
        dataset.close()
    assert json.loads((root / ".recording/meta/info.json").read_text()) == source
    recovered = tmp_path / "recovered"
    recover_recording(root, recovered)
    assert json.loads((recovered / "meta/alohamini.json").read_text())["source_info"] == source
    assert frame_metadata(recovered / "meta/safety/episode_000001.jsonl") == [live, live]
    report = check_dataset(recovered, decode_videos=True)
    assert report["valid"], report
    assert json.loads((recovered / "meta/info.json").read_text())["total_frames"] == 4
    current_files = hashes(root)
    for path, digest in committed.items():
        if path.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
            assert current_files[path] == digest


@pytest.mark.parametrize(
    "path,value",
    [
        (("motors", "arm_left_shoulder_pan", "id"), 99),
        (("motors", "arm_left_shoulder_pan", "model"), "different-model"),
        (("motors", "arm_left_shoulder_pan", "range_min"), 999),
        (("motors", "arm_left_shoulder_pan", "range_max"), 3001),
        (("motors", "arm_left_shoulder_pan", "drive_mode"), 1),
        (("motors", "arm_left_shoulder_pan", "homing_offset"), 1),
        (("motors", "arm_left_shoulder_pan", "normalization"), "degrees"),
        (("motors", "arm_left_shoulder_pan", "extra"), {"motion_profile": 1}),
        (("lift_axis", "soft_max_mm"), 550.0),
        (("lift_axis", "motion_profile"), {"acceleration": 120}),
        (("cameras",), []),
        (("robot_model",), "alohamini2"),
    ],
)
def test_profile_exception_does_not_allow_other_robot_contract_changes(tmp_path, path, value):
    root = create(tmp_path / "record")
    original = (root / "meta/alohamini.json").read_bytes()
    live = profiled_metadata(accelerated=True)
    target = live
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        LocalDataset(root, fps=30, task="pick", robot_metadata=live, resume=True)
    assert (root / "meta/alohamini.json").read_bytes() == original
    # A failed compatibility check must not strand a staging schema or either lock.
    with contextlib.closing(
        LocalDataset(
            root,
            fps=30,
            task="pick",
            robot_metadata=profiled_metadata(accelerated=True),
            resume=True,
        )
    ) as dataset:
        assert dataset.num_episodes == 1


@pytest.mark.parametrize(
    "options",
    [
        {"fps": 25},
        {"task": "different"},
        {"fixed_dimensions": {"version": 1, "targets": {"x.vel": 0.0}}},
    ],
)
def test_profile_exception_does_not_allow_fps_task_or_fixed_changes(tmp_path, options):
    root = create(tmp_path / "record")
    with pytest.raises(ValueError, match="schema, task, fps or robot calibration changed"):
        LocalDataset(
            root,
            **{
                "fps": 30,
                "task": "pick",
                "robot_metadata": profiled_metadata(accelerated=True),
                **options,
            },
            resume=True,
        )


@pytest.mark.parametrize("field,value", [("range_min", 999), ("id", 99)])
def test_native_safety_checker_still_rejects_calibration_or_identity_changes(
    tmp_path, field, value
):
    root = tmp_path / "native"
    original = profiled_metadata()
    with contextlib.closing(
        _EpisodeWriter(root, fps=30, task="pick", robot_metadata=original)
    ) as dataset:
        dataset.begin_episode()
        changed = profiled_metadata(accelerated=True)
        changed["motors"]["arm_left_shoulder_pan"][field] = value
        assert dataset.add_frame(frame(dataset), {"forward": jpeg()}, {"robot_metadata": changed})
        dataset.save_episode()
    report = check_dataset(root, decode_videos=True)
    assert not report["valid"]
    assert any(issue["code"] == "SAFETY_MODEL_MISMATCH" for issue in report["issues"])
