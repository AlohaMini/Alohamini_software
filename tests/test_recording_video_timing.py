"""Per-camera profiling preserves encoded output and the serial save-time accounting."""

import contextlib
import math
import shutil
from collections import Counter
from pathlib import Path
from threading import Lock, get_ident
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import av
import pyarrow.parquet as pq
import pytest
from test_recording_storage import add_episode, metadata
from test_recording_video_statistics import episode_fixture
from test_teleoperation import snapshot

from alohamini.apps import recording
from alohamini.datasets import images, video
from alohamini.datasets.record import LocalDataset

SECONDS = {
    "setup_s",
    "read_decode_s",
    "image_statistics_s",
    "frame_build_s",
    "convert_s",
    "encode_s",
    "mux_s",
    "flush_s",
    "close_s",
    "headers_s",
    "hash_s",
    "other_s",
    "wall_s",
}
CONFIGURATION = {
    "codec",
    "pix_fmt",
    "threads",
    "workers",
    "preset",
    "crf",
    "gop",
    "frames",
    "width",
    "height",
}
FLAT_VIDEO = {"video_encode_headers_hash", "image_statistics", "video_index"}


def assert_camera_timing(detail, *, frames, shape, workers):
    assert set(detail) == SECONDS | CONFIGURATION
    assert all(type(detail[key]) is float and math.isfinite(detail[key]) for key in SECONDS)
    assert all(detail[key] >= 0 for key in SECONDS)
    assert sum(detail[key] for key in SECONDS - {"wall_s"}) == pytest.approx(
        detail["wall_s"], abs=1e-6
    )
    assert detail["wall_s"] > 0
    assert {key: detail[key] for key in CONFIGURATION} == {
        "codec": "libx264",
        "pix_fmt": "yuv444p" if shape[0] % 2 or shape[1] % 2 else "yuv420p",
        "threads": 2,
        "workers": workers,
        "preset": "fast",
        "crf": 18,
        "gop": 2,
        "frames": frames,
        "width": shape[1],
        "height": shape[0],
    }


@pytest.mark.parametrize(
    "extension,shape,workers",
    [("jpg", (18, 24, 3), 1), ("jpg", (17, 23, 3), 2), ("png", (17, 23, 3), 1)],
)
def test_detailed_profiling_preserves_bytes_and_decodes_each_input_only_once(
    tmp_path, monkeypatch, extension, shape, workers
):
    cameras, length = ("forward", "wrist"), 17
    plain, profiled = tmp_path / "plain", tmp_path / "profiled"
    features = episode_fixture(plain, extension, shape, length=length, cameras=cameras)
    shutil.copytree(plain, profiled)
    video.finalize_recording_video(plain, 30, features, cameras, workers=workers)
    references = pq.read_table(profiled / "frames.parquet").to_pylist()
    expected = Counter(
        (camera, row[f"observation.images.{camera}"]) for row in references for camera in cameras
    )
    decoded, reads, lock = Counter(), Counter(), Lock()
    image_rgb, read_bytes, inspect_video, av_open = (
        images.image_rgb,
        Path.read_bytes,
        video.inspect_video,
        av.open,
    )
    header_checks = []

    def counted_rgb(root, camera, reference):
        with lock:
            decoded[camera, reference] += 1
        return image_rgb(root, camera, reference)

    def counted_read(path):
        if path.is_relative_to(profiled / "images"):
            with lock:
                reads[str(path.relative_to(profiled))] += 1
        return read_bytes(path)

    class HeaderOnly:
        def __init__(self, container):
            self.container = container

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return self.container.__exit__(*exc)

        def __getattr__(self, key):
            return getattr(self.container, key)

        def decode(self, *args, **kwargs):
            pytest.fail("Save profiling must not decode the completed MP4")

    def header_only_open(path, mode="r", **kwargs):
        container = av_open(path, mode, **kwargs)
        return container if mode == "w" else HeaderOnly(container)

    def checked_headers(path, **kwargs):
        header_checks.append(kwargs.get("decode", False))
        return inspect_video(path, **kwargs)

    monkeypatch.setattr(video, "image_rgb", counted_rgb)
    monkeypatch.setattr(images, "image_rgb", counted_rgb)
    monkeypatch.setattr(Path, "read_bytes", counted_read)
    monkeypatch.setattr(video, "inspect_video", checked_headers)
    monkeypatch.setattr(av, "open", header_only_open)
    flat, details = {}, {}
    video.finalize_recording_video(
        profiled, 30, features, cameras, workers=workers, timings=flat, camera_timings=details
    )
    assert decoded == expected
    assert reads == Counter({reference: 1 for _, reference in expected})
    assert header_checks == [False] * len(cameras)
    assert set(flat) == FLAT_VIDEO
    assert set(details) == set(cameras)
    for camera in cameras:
        assert_camera_timing(details[camera], frames=length, shape=shape, workers=workers)
        assert details[camera]["wall_s"] <= flat["video_encode_headers_hash"]
        relative = f"videos/{camera}.mp4"
        assert (profiled / relative).read_bytes() == (plain / relative).read_bytes()
        with av.open(str(profiled / relative)) as container:
            stream = container.streams.video[0]
            assert stream.pix_fmt == details[camera]["pix_fmt"]
            assert stream.frames == length
    assert (profiled / "frames.parquet").read_bytes() == (plain / "frames.parquet").read_bytes()
    assert (profiled / "image_stats.json").read_bytes() == (plain / "image_stats.json").read_bytes()


def test_no_cameras_produces_no_camera_detail_and_preserves_numeric_rows(tmp_path, monkeypatch):
    episode = tmp_path / "numeric"
    features = episode_fixture(episode, "jpg", (17, 23, 3), length=3, cameras=())
    original = pq.read_table(episode / "frames.parquet")

    def forbidden(*args, **kwargs):
        pytest.fail("A numeric-only save must not read or encode an image")

    monkeypatch.setattr(video, "image_rgb", forbidden)
    monkeypatch.setattr(video, "_encode_frames", forbidden)
    monkeypatch.setattr(av, "open", forbidden)
    flat, details = {}, {}
    video.finalize_recording_video(episode, 30, features, (), timings=flat, camera_timings=details)
    assert details == {}
    assert set(flat) == FLAT_VIDEO
    assert all(math.isfinite(value) and value >= 0 for value in flat.values())
    assert pq.read_table(episode / "frames.parquet").equals(original)


@pytest.mark.parametrize("profile", [False, True])
def test_writer_collects_camera_details_only_when_enabled_and_resets_each_save(
    tmp_path, monkeypatch, profile
):
    main_thread = get_ident()

    def checked_clock():
        if not profile and get_ident() != main_thread:
            pytest.fail("Disabled profiling must not read the clock in video workers")
        return perf_counter()

    monkeypatch.setattr(video, "time", SimpleNamespace(perf_counter=checked_clock))
    cameras = ("forward", "left", "right")
    with contextlib.closing(
        LocalDataset(
            tmp_path / "record",
            fps=30,
            task="pick",
            robot_metadata=metadata(cameras),
            **({"profile_timing": True} if profile else {}),
        )
    ) as dataset:
        previous = None
        for frames in (3, 2):
            add_episode(dataset, frames)
            dataset.save_episode()
            flat, details = dataset.last_save_timings, dataset.last_video_timings
            assert set(details) == (set(cameras) if profile else set())
            assert details is not previous
            previous = details
            for detail in details.values():
                assert_camera_timing(detail, frames=frames, shape=(32, 32, 3), workers=3)
            assert not (set(flat) & SECONDS)
            assert all(type(value) is float for value in flat.values())
            assert sum(value for key, value in flat.items() if key != "total") <= flat["total"]


@pytest.mark.parametrize("profile", [False, True])
def test_recording_prints_camera_details_only_as_separate_profile_lines(
    tmp_path, monkeypatch, capsys, profile
):
    initial = snapshot()
    initial.payload["_robot_metadata"] = metadata()
    client = Mock(client_id="client")
    client.connect_control.return_value = initial
    client.__enter__, client.__exit__ = Mock(return_value=client), Mock(return_value=False)
    keyboard = recording.RecordingKeyboard()
    detail = {key: 0.001 for key in SECONDS}
    detail.update(
        codec="libx264",
        pix_fmt="yuv420p",
        threads=2,
        workers=1,
        preset="fast",
        crf=18,
        gop=2,
        frames=3,
        width=32,
        height=32,
    )
    dataset = Mock(
        num_episodes=0,
        submitted=3,
        saved=3,
        cameras=("forward",),
        last_save_timings={"video_encode_headers_hash": 0.02, "total": 0.03},
        last_video_timings={"forward": detail},
    )
    dataset.save_episode.side_effect = lambda: setattr(dataset, "num_episodes", 1)
    leader, keys = MagicMock(), MagicMock()
    keys.return_value.__enter__.return_value = keyboard
    monkeypatch.setattr(recording, "BimanualLeader", leader)
    monkeypatch.setattr(recording, "RecordingKeyboard", keys)
    monkeypatch.setattr(recording, "HostClient", Mock(return_value=client))
    monkeypatch.setattr(recording, "record_loop", Mock())
    factory = Mock(return_value=dataset)
    monkeypatch.setattr("alohamini.datasets.record.LocalDataset", factory)
    recording.record(
        "in-memory",
        "alohamini2pro",
        dataset_name="test",
        task="pick",
        root=tmp_path / "capture",
        reset_time_s=0,
        profile_timing=profile,
    )
    lines = capsys.readouterr().out.splitlines()
    camera_lines = [line for line in lines if line.startswith("[VIDEO TIMING camera=")]
    flat_lines = [line for line in lines if line.startswith("[SAVE TIMING seconds]")]
    assert len(camera_lines) == len(flat_lines) == int(profile)
    assert factory.call_args.kwargs["profile_timing"] is profile
    if profile:
        assert camera_lines[0].startswith("[VIDEO TIMING camera=forward seconds]")
        assert "read_decode_s=" in camera_lines[0]
        assert "read_decode_s=" not in flat_lines[0]
    dataset.close.assert_called_once()
