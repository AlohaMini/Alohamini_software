"""Terminal SIGINT preserves partial episodes, including background video and stats."""

import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from test_dataset import frame, jpeg
from test_recording_stats_incremental import assert_all_statistics
from test_recording_storage import metadata

from alohamini.cli import main
from alohamini.datasets.record import LocalDataset
from alohamini.datasets.tools import check_dataset


def add_rows(dataset, length, offset):
    dataset.begin_episode()
    for index in range(length):
        values = frame(dataset)
        values["action"][:] = offset + index
        assert dataset.add_frame(
            values,
            {c: jpeg(((offset + index) % 256, 70, 140), (32, 32, 3)) for c in dataset.cameras},
            {"safety": {"valid": True}},
        )
        dataset._writer._queue.join()


def test_early_end_and_rerecord_exclude_discarded_values_from_all_statistics(tmp_path):
    roots = [tmp_path / "clean", tmp_path / "rerecorded"]
    summaries = []
    for root in roots:
        with contextlib.closing(
            LocalDataset(
                root, fps=30, task="test", robot_metadata=metadata(), background_video_encoding=True
            )
        ) as dataset:
            add_rows(dataset, 137, 0)
            dataset.save_episode()  # Finalize the actual length, without a scheduled end time.
            before = (root / "meta/stats.json").read_bytes()
            if root == roots[1]:
                # Different pixels and extreme numeric values expose leaked statistics.
                add_rows(dataset, 11, 1_000_000)
                dataset.discard_episode()
                assert (root / "meta/stats.json").read_bytes() == before
                assert dataset.total_frames == 137 and dataset.num_episodes == 1
            add_rows(dataset, 7, 200)
            dataset.save_episode()
            assert dataset.total_frames == 144 and dataset.num_episodes == 2
        assert_all_statistics(root)  # NumPy oracle includes q01/q99, not just row counts.
        summaries.append(json.loads((root / "meta/stats.json").read_text()))
    assert summaries[0] == summaries[1]


@pytest.mark.parametrize("kind", ["record", "evaluate"])
def test_real_terminal_sigint_saves_complete_dataset_and_statistics(tmp_path, kind):
    root = tmp_path / kind
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
    # Own the process group: the probe's real terminal-style SIGINT must never
    # reach pytest, the developer shell or another robot/application session.
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), kind, str(root)],
        env=env,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((root.parent / f"{kind}-interrupt.json").read_text())
    assert summary["cli_status"] == 130 and summary["client_closed"]
    if kind == "record":
        assert summary["encoder_isolated"] and summary["background_reused"]
        assert summary["encoder_returncode"] == 0
    else:
        assert summary["robot_stop_called"]
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (1, 3)
    assert not (root / ".recording").exists()
    assert check_dataset(root, decode_images=True, decode_videos=True)["valid"]
    assert_all_statistics(root)


def test_ctrlc_during_rerecord_reset_never_saves_the_discarded_attempt(tmp_path):
    from test_teleoperation import snapshot

    from alohamini.apps import recording

    root = tmp_path / "record"
    initial = snapshot()
    initial.payload["_robot_metadata"] = metadata()
    client = MagicMock(client_id="client")
    client.connect_control.return_value = initial
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    keyboard = recording.RecordingKeyboard()
    keys = MagicMock()
    keys.__enter__.return_value = keyboard
    calls = 0
    saved_stats = None

    def capture(*args, dataset=None, **kwargs):
        nonlocal calls, saved_stats
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt  # Ctrl+C in the reset stage after pressing R.
        if dataset is None:
            return
        if calls == 3:
            saved_stats = (root / "meta/stats.json").read_bytes()
            keyboard.events["rerecord_episode"] = True
        for index in range(3):
            values = frame(dataset)
            values["action"][:] = index if calls == 1 else 1_000_000 + index
            assert dataset.add_frame(values, {"forward": jpeg((calls, index, 140))}, {})

    with (
        patch.object(recording, "HostClient", return_value=client),
        patch.object(recording, "BimanualLeader", return_value=MagicMock()),
        patch.object(recording, "RecordingKeyboard", return_value=keys),
        patch.object(recording, "record_loop", side_effect=capture),
    ):
        assert (
            main(
                [
                    "record",
                    "--host",
                    "fake",
                    "--robot_model",
                    "alohamini2pro",
                    "--dataset",
                    "test",
                    "--root",
                    str(root),
                    "--task",
                    "test",
                    "--num_episodes",
                    "2",
                    "--reset_time",
                    "1",
                ]
            )
            == 130
        )
    info = json.loads((root / "meta/info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (1, 3)
    assert (root / "meta/stats.json").read_bytes() == saved_stats
    assert_all_statistics(root)
    assert check_dataset(root, decode_videos=True)["valid"]


def _record_probe(root):
    from test_teleoperation import snapshot

    from alohamini.apps import recording
    from alohamini.datasets import video

    initial = snapshot()
    initial.payload["_robot_metadata"] = metadata()
    client = MagicMock(client_id="client")
    client.connect_control.return_value = initial
    client.__enter__.return_value = client
    client.__exit__.return_value = False
    keys = MagicMock()
    keys.__enter__.return_value = recording.RecordingKeyboard()
    details = {}
    encoder = None
    finalize = video.finalize_recording_video

    def save(episode, *args, **kwargs):
        details["background_reused"] = kwargs.get("preencoded") is not None
        return finalize(episode, *args, **kwargs)

    def capture(*args, dataset, **kwargs):
        nonlocal encoder
        # record() has already begun the episode. Use its real writer and encoder.
        for index in range(3):
            values = frame(dataset)
            values["action"][:] = index
            assert dataset.add_frame(values, {"forward": jpeg((index, 70, 140))}, {})
        encoder = dataset._video_encoder
        details["encoder_isolated"] = os.getpgid(encoder.process.pid) != os.getpgrp()
        os.killpg(os.getpgrp(), signal.SIGINT)

    with (
        patch.object(recording, "HostClient", return_value=client),
        patch.object(recording, "BimanualLeader", return_value=MagicMock()),
        patch.object(recording, "RecordingKeyboard", return_value=keys),
        patch.object(recording, "record_loop", side_effect=capture),
        patch.object(video, "finalize_recording_video", side_effect=save),
    ):
        status = main(
            [
                "record",
                "--host",
                "fake",
                "--robot_model",
                "alohamini2pro",
                "--dataset",
                "test",
                "--root",
                str(root),
                "--task",
                "test",
                "--reset_time",
                "0",
            ]
        )
    details.update(
        cli_status=status,
        client_closed=client.__exit__.call_count == 1,
        encoder_returncode=encoder.process.returncode,
    )
    return details


def _evaluation_probe(root):
    from test_evaluation import EvaluationClient
    from test_replay import Clock

    from alohamini.apps import evaluation
    from alohamini.datasets.record import state_names

    clock = Clock()
    client = EvaluationClient(clock)
    client.state.payload["_robot_metadata"]["cameras"] = ["forward"]
    policy = Mock()
    policy.robot_metadata = copy.deepcopy(client.state.payload["_robot_metadata"])
    names = state_names("alohamini2pro")

    def images(current):
        current.state.images["forward"] = jpeg((len(current.sent), 70, 140))
        current.state.payload["_host_timing"] = {
            "state_sample_finished_monotonic_s": clock.now,
            "camera_capture_monotonic_s": {"forward": clock.now},
        }

    client.on_read = images

    def select(snapshot):
        if policy.select_action.call_count == 4:
            os.killpg(os.getpgrp(), signal.SIGINT)
        return {name: snapshot.payload[name] for name in names}

    policy.select_action.side_effect = select
    connection = MagicMock()
    connection.__enter__.return_value = client
    connection.__exit__.return_value = False
    with (
        patch.object(evaluation, "HostClient", return_value=connection),
        patch.object(evaluation, "WorkspacePaths") as paths,
        patch.object(evaluation, "stop_owned_robot") as stop,
        patch.object(evaluation.time, "monotonic", side_effect=lambda: clock.now),
        patch.object(evaluation.time, "sleep", side_effect=clock.sleep),
        patch.dict(sys.modules, interrupt_test_policy=SimpleNamespace(load=lambda: policy)),
    ):
        paths.return_value.dataset.return_value = root
        status = main(
            [
                "evaluate",
                "--host",
                "fake",
                "--robot_model",
                "alohamini2pro",
                "--policy",
                "interrupt_test_policy:load",
                "--dataset",
                "test",
                "--task",
                "test",
            ]
        )
    return dict(
        cli_status=status,
        client_closed=connection.__exit__.call_count == 1,
        robot_stop_called=stop.call_count == 1,
    )


if __name__ == "__main__":
    root = Path(sys.argv[2])
    probe = _record_probe if sys.argv[1] == "record" else _evaluation_probe
    result = probe(root)
    (root.parent / f"{sys.argv[1]}-interrupt.json").write_text(json.dumps(result))
