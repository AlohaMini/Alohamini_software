"""Disk-spooled encoders preserve accepted frames and survive worker failure."""

import contextlib
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from unittest.mock import patch

import av
import pyarrow.parquet as pq
import pytest
from test_dataset import frame, jpeg
from test_recording_storage import add_episode, metadata

from alohamini.cli import main
from alohamini.datasets.record import LocalDataset, _EpisodeWriter
from alohamini.datasets.recording_video import _references
from alohamini.datasets.video import inspect_video


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail("Timed out waiting for encoder")
        time.sleep(0.01)


def submit(writer, index, *, invalid=False, shape=(32, 32, 3)):
    values = frame(writer)
    values["action"][:] = index
    pictures = {c: jpeg((index % 256, 140, 70), shape) for c in writer.cameras}
    if invalid:
        pictures[writer.cameras[0]] = b"not a jpeg"
    assert writer.add_frame(values, pictures, {"safety": {"valid": True}})
    # Only pace this fixture's bounded disk writer, never the encoder.
    writer._queue.join()


def rgb_hash(path):
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            digest.update(frame.to_ndarray(format="rgb24").tobytes())
    return digest.hexdigest()


@pytest.mark.parametrize("workers", [1, 3])
def test_background_cpu_matches_original_with_rejected_rows_and_early_end(tmp_path, workers):
    cameras = ("forward", "left", "right")
    outputs = []
    for background in (False, True):
        root = tmp_path / str(background)
        with contextlib.closing(
            _EpisodeWriter(
                root,
                fps=30,
                task="test",
                robot_metadata=metadata(cameras),
                background_video_encoding=background,
                video_encoding_workers=workers,
            )
        ) as writer:
            writer.begin_episode()
            for index in range(138):
                submit(writer, index, invalid=index == 4)
                if index == 12:
                    writer.event({"type": "test_event"})
            process = writer._video_encoder.process if background else None
            if background:
                wait_until(lambda: list((writer._pending / "videos").glob("*.mp4")))
                assert process.poll() is None
                assert not (writer._pending / ".video/result.json").exists()
            writer.save_episode()
            if background:
                assert process.returncode == 0
                assert writer._video_encoder is None
            assert writer.rejected_images == 1
            assert writer.saved == 137 and writer.queue_overflows == 0
            episode = root / "episodes/episode_000000"
            assert not (episode / ".video").exists()
            assert not (episode / "images").exists()
            for camera in cameras:
                assert inspect_video(episode / "videos" / f"{camera}.mp4", decode=True) == {
                    "frames": 137,
                    "fps": 30.0,
                    "shape": [32, 32, 3],
                }
            outputs.append(
                (
                    pq.read_table(episode / "frames.parquet").to_pylist(),
                    json.loads((episode / "image_stats.json").read_text()),
                    {c: rgb_hash(episode / "videos" / f"{c}.mp4") for c in cameras},
                )
            )
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("failure", ["killed", "memory"])
def test_encoder_failure_falls_back_without_losing_samples(tmp_path, failure, caplog):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        writer.begin_episode()
        encoder = writer._video_encoder
        submit(writer, 1)
        original = {p: p.read_bytes() for p in (writer._pending / "images").rglob("*.jpg")}
        if failure == "killed":
            encoder.process.kill()
        else:
            # Exercise the real RSS supervisor without exhausting machine RAM.
            encoder._memory_bytes = 1
        wait_until(lambda: encoder.process.poll() is not None)
        assert all(p.read_bytes() == value for p, value in original.items())
        for index in range(2, 6):
            submit(writer, index)
        writer.save_episode()
        assert writer.saved == 5 and writer.num_episodes == 1
        episode = writer.root / "episodes/episode_000000"
        assert inspect_video(episode / "videos/forward.mp4", decode=True)["frames"] == 5
        assert "CPU H.264 encoding will run at save" in caplog.text


def test_discard_stops_worker_before_removing_files_and_next_episode_works(tmp_path):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        writer.begin_episode()
        submit(writer, 1)
        pending, process = writer._pending, writer._video_encoder.process
        writer.discard_episode()
        assert process.poll() is not None and not pending.exists()
        writer.begin_episode()
        submit(writer, 2)
        writer.close()
        assert writer.num_episodes == 1 and writer.saved == 1


def test_failed_save_keeps_background_inputs_and_resume_discards_only_pending(tmp_path):
    root = tmp_path / "record"
    dataset = LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(), background_video_encoding=True
    )
    add_episode(dataset, 3)
    process = dataset._video_encoder.process
    with patch(
        "alohamini.datasets.video.finalize_recording_video", side_effect=OSError("disk error")
    ):
        with pytest.raises(OSError, match="disk error"):
            dataset.save_episode()
    dataset.close()
    assert process.poll() is not None
    assert len(list(root.rglob("*.jpg"))) == 3
    with contextlib.closing(
        LocalDataset(
            root,
            fps=30,
            task="pick",
            robot_metadata=metadata(),
            resume=True,
            background_video_encoding=True,
        )
    ) as resumed:
        assert resumed.num_episodes == 0
        add_episode(resumed, 2)
        resumed.save_episode()
        assert resumed.num_episodes == 1 and resumed.total_frames == 2


def test_numeric_only_recording_never_starts_encoder(tmp_path):
    with patch("subprocess.Popen", side_effect=AssertionError("No images to encode")):
        with contextlib.closing(
            LocalDataset(
                tmp_path / "record",
                fps=30,
                task="test",
                robot_metadata=metadata(()),
                background_video_encoding=True,
                video_encoder="nvenc",
            )
        ) as dataset:
            add_episode(dataset)
            dataset.save_episode()
            assert dataset.total_frames == 3


def test_tail_accepts_only_complete_rows_and_requires_final_count(tmp_path):
    (tmp_path / ".video").mkdir()
    journal = tmp_path / "journal.jsonl"
    row = json.dumps({"frame": {"frame_index": 0, "observation.images.forward": "first.jpg"}})
    journal.write_text(row + "\n" + '{"record": {}}\n')
    (tmp_path / ".video/finished.json").write_text('{"length": 1}')
    assert list(_references(tmp_path, "forward")) == ["first.jpg"]
    journal.write_text(row + "\n" + '{"frame":')
    with pytest.raises(ValueError, match="Incomplete recording journal"):
        list(_references(tmp_path, "forward"))


def test_cli_defaults_and_explicit_gpu_postcapture_options():
    base = [
        "record",
        "--host",
        "test",
        "--robot_model",
        "alohamini2pro",
        "--dataset",
        "test",
        "--task",
        "test",
    ]
    for options in (
        [],
        [
            "--video-encoder",
            "nvenc",
            "--no-background-video-encoding",
            "--video-encoding-memory-mib",
            "1024",
        ],
    ):
        with patch("alohamini.apps.recording.record") as record:
            assert main(base + options) == 0
        args = record.call_args.kwargs
        assert args["background_video_encoding"] is (not options)
        assert args["video_encoder"] == ("nvenc" if options else "cpu")
        assert args["video_encoding_memory_mib"] == (1024 if options else 2048)


def test_finish_marker_race_does_not_skip_last_journal_rows(tmp_path, monkeypatch):
    directory = tmp_path / ".video"
    directory.mkdir()
    journal, finished = tmp_path / "journal.jsonl", directory / "finished.json"

    def row(index):
        return (
            json.dumps(
                {"frame": {"frame_index": index, "observation.images.forward": f"{index}.jpg"}}
            )
            + "\n"
        )

    journal.write_text(row(0))
    exists = Path.exists
    appended = False

    def publish_after_first_eof(path):
        nonlocal appended
        if path == finished and not appended:
            appended = True
            with journal.open("a") as stream:
                stream.write(row(1))
            finished.write_text('{"length": 2}')
        return exists(path)

    monkeypatch.setattr(Path, "exists", publish_after_first_eof)
    assert list(_references(tmp_path, "forward")) == ["0.jpg", "1.jpg"]


def test_input_digest_mismatch_reencodes_original_images(tmp_path, caplog):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        writer.begin_episode()
        submit(writer, 1)
        writer._video_references["forward"] = hashlib.sha256(b"incorrect input order")
        writer.save_episode()
        assert "input order/count differs" in caplog.text
        assert writer.saved == 1 and writer.num_episodes == 1


def test_process_start_failure_uses_original_cpu_save(tmp_path, caplog):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        with patch("subprocess.Popen", side_effect=OSError("process limit")):
            writer.begin_episode()
        submit(writer, 1)
        writer.save_episode()
        assert writer.saved == 1 and "could not start encoder" in caplog.text


@pytest.mark.parametrize("background", [False, True])
def test_unsupported_nvenc_image_size_falls_back_to_cpu(tmp_path, background, caplog):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=background,
            video_encoder="nvenc",
        )
    ) as writer:
        writer.begin_episode()
        for index in range(3):
            submit(writer, index, shape=(17, 23, 3))
        writer.save_episode()
        path = writer.root / "episodes/episode_000000/videos/forward.mp4"
        assert inspect_video(path, decode=True) == {"frames": 3, "fps": 30.0, "shape": [17, 23, 3]}
        assert "CPU" in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        {"video_encoder": "av1"},
        {"background_video_encoding": 1},
        {"video_encoding_memory_mib": 255},
        {"video_encoding_memory_mib": True},
    ],
)
def test_invalid_video_options_fail_before_creating_dataset(tmp_path, options):
    root = tmp_path / "record"
    with pytest.raises(ValueError):
        LocalDataset(root, fps=30, task="pick", robot_metadata=metadata(), **options)
    assert not root.exists()


def test_corrupt_background_video_reencodes_originals(tmp_path, caplog):
    from alohamini.datasets.recording_video import RecordingVideoEncoder

    finish = RecordingVideoEncoder.finish

    def corrupt_after_finish(encoder, *args):
        result = finish(encoder, *args)
        assert result is not None
        (encoder.episode / "videos/forward.mp4").write_bytes(b"broken mp4")
        return result

    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        writer.begin_episode()
        submit(writer, 1)
        with patch.object(RecordingVideoEncoder, "finish", corrupt_after_finish):
            writer.save_episode()
        path = writer.root / "episodes/episode_000000/videos/forward.mp4"
        assert inspect_video(path, decode=True)["frames"] == 1
        assert "Background video validation failed" in caplog.text


def test_failed_background_save_still_supports_explicit_recovery(tmp_path):
    from alohamini.datasets.lerobotv3 import recover_recording
    from alohamini.datasets.tools import check_dataset

    root, output = tmp_path / "record", tmp_path / "recovered"
    dataset = LocalDataset(
        root, fps=30, task="pick", robot_metadata=metadata(), background_video_encoding=True
    )
    add_episode(dataset, 3)
    with patch(
        "alohamini.datasets.video.finalize_recording_video", side_effect=OSError("disk error")
    ):
        with pytest.raises(OSError):
            dataset.save_episode()
    dataset.close()
    report = recover_recording(root, output)
    assert report["valid"], report
    assert check_dataset(output, decode_videos=True)["valid"]
    assert json.loads((output / "meta/info.json").read_text())["total_frames"] == 3


def test_disk_writer_continues_while_encoder_is_paused(tmp_path):
    with contextlib.closing(
        _EpisodeWriter(
            tmp_path / "record",
            fps=30,
            task="test",
            robot_metadata=metadata(),
            background_video_encoding=True,
        )
    ) as writer:
        writer.begin_episode()
        submit(writer, 1)
        process = writer._video_encoder.process
        os.kill(process.pid, signal.SIGSTOP)
        try:
            for index in range(2, 41):
                submit(writer, index)
            assert writer.saved == 40 and writer.queue_overflows == 0
            assert len(list((writer._pending / "images").rglob("*.jpg"))) == 40
        finally:
            os.kill(process.pid, signal.SIGCONT)
        writer.save_episode()
        path = writer.root / "episodes/episode_000000/videos/forward.mp4"
        assert inspect_video(path, decode=True)["frames"] == 40


def test_disk_writer_error_stops_encoder_and_retains_inputs(tmp_path):
    writer = _EpisodeWriter(
        tmp_path / "record",
        fps=30,
        task="test",
        robot_metadata=metadata(),
        background_video_encoding=True,
    )
    try:
        writer.begin_episode()
        submit(writer, 1)
        process = writer._video_encoder.process
        writer._error = OSError("disk error")
        with pytest.raises(OSError, match="Dataset writer failed"):
            writer.discard_episode()
        assert process.poll() is not None and writer.save_failed
        assert list((writer._pending / "images").rglob("*.jpg"))
    finally:
        writer.close()
