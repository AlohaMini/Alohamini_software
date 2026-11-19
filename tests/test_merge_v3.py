import copy
import hashlib
import json

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg
from test_replay import replay_snapshot

from alohamini.datasets.edit import edit_dataset, parse_args
from alohamini.datasets.record import LocalDataset
from alohamini.learning.data import AlohaMiniDataset


def capture(root, task, value, *, offset=0):
    metadata = copy.deepcopy(replay_snapshot().payload["_robot_metadata"])
    metadata["cameras"] = ["forward"]
    metadata["motors"]["arm_left_shoulder_pan"]["homing_offset"] += offset
    data = LocalDataset(root, fps=30, task=task, robot_metadata=metadata)
    try:
        data.begin_episode()
        for index in range(3):
            row = frame(data)
            row["action"][:] = value
            stamp = 100 + index / 30
            assert data.add_frame(
                row,
                {"forward": jpeg((value, 20, 240 - value))},
                {
                    "safety": {"feedback_valid": True},
                    "intervention": task == "correct",
                    "requested_action": dict.fromkeys(data.names, float(value)),
                    "host_timing": {
                        "state_sample_monotonic_s": stamp,
                        "camera_capture_monotonic_s": {"forward": stamp},
                    },
                },
            )
        data.save_episode()
    finally:
        data.close()
    return root


def merge(roots, output):
    return edit_dataset(
        parse_args(
            [
                "--operation.type=merge",
                "--operation.roots",
                json.dumps([str(root) for root in roots]),
                "--output",
                str(output),
            ]
        )
    )


def hashes(root):
    return sorted(hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*.mp4"))


def test_v3_merge_preserves_media_actions_timing_and_remaps_tasks(tmp_path):
    first = capture(tmp_path / "first", "demo", 10)
    second = capture(tmp_path / "second", "correct", 20)
    output = tmp_path / "merged"
    source_hashes = hashes(first) + hashes(second)
    report = merge([first, second], output)
    assert report["valid"]
    assert hashes(output) == sorted(source_hashes)
    assert hashes(first) + hashes(second) == source_hashes
    data = AlohaMiniDataset(
        output, episodes=[0, 1], state="none", chunk_size=3, include_task=True, image_size=(32, 32)
    )
    assert len(data) == 6
    assert data[0]["task"] == "demo" and data[3]["task"] == "correct"
    assert np.asarray(data[0]["action"])[0, 0] == 10
    assert np.asarray(data[3]["action"])[0, 0] == 20
    for index, root in enumerate((first, second)):
        old = [
            json.loads(line)
            for line in (root / "meta/safety/episode_000000.jsonl").read_text().splitlines()
        ]
        new = [
            json.loads(line)
            for line in (output / "meta/safety" / f"episode_{index:06d}.jsonl")
            .read_text()
            .splitlines()
        ]
        for row in old:
            row["episode_index"] = index
        assert new == old
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["mean"] == [15.0] * 18
    assert stats["index"]["mean"] == [2.5]
    assert stats["task_index"]["mean"] == [0.5]
    assert "q99" in stats["observation.images.forward"]


def test_v3_merge_rejects_calibration_changes_and_existing_output(tmp_path):
    first = capture(tmp_path / "first", "demo", 10)
    changed = capture(tmp_path / "changed", "correct", 20, offset=1)
    with pytest.raises(ValueError, match="robot_metadata"):
        merge([first, changed], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    with pytest.raises(ValueError, match="distinct"):
        merge([first, first], tmp_path / "duplicate")
    with pytest.raises(FileExistsError):
        merge([first], first)


def test_v3_merge_without_image_histograms_keeps_moments_without_fake_quantiles(tmp_path):
    first = capture(tmp_path / "first", "demo", 10)
    second = capture(tmp_path / "second", "correct", 20)
    for path in (second / "meta/episodes").rglob("*.parquet"):
        table = pq.read_table(path)
        pq.write_table(table.drop(["image_statistics"]), path)
    output = tmp_path / "merged"
    assert merge([first, second], output)["valid"]
    stats = json.loads((output / "meta/stats.json").read_text())
    image = stats["observation.images.forward"]
    assert "mean" in image and "q99" not in image
