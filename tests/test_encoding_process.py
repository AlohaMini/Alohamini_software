"""Process memory and orphan cleanup, including real subprocess lifetimes."""

import ctypes
import errno
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from alohamini.datasets import encoding_process


def test_darwin_reads_current_resident_bytes_through_libproc(monkeypatch):
    def read(pid, flavor, arg, buffer, size):
        assert pid == 123 and flavor == 4 and arg == 0 and size == 96
        info = ctypes.cast(buffer, ctypes.POINTER(encoding_process._TaskInfo)).contents
        info.virtual_bytes = 2**40
        info.resident_bytes = 12345678
        return size

    monkeypatch.setattr(encoding_process.sys, "platform", "darwin")
    monkeypatch.setattr(encoding_process, "_libproc", lambda: SimpleNamespace(proc_pidinfo=read))
    assert encoding_process.resident_bytes(123) == 12345678


@pytest.mark.parametrize("returned", [0, 8])
def test_darwin_rejects_missing_or_partial_process_information(monkeypatch, returned):
    ctypes.set_errno(errno.ESRCH)
    monkeypatch.setattr(encoding_process.sys, "platform", "darwin")
    monkeypatch.setattr(encoding_process, "_libproc", lambda: SimpleNamespace(
        proc_pidinfo=lambda *args: returned,
    ))
    with pytest.raises(OSError):
        encoding_process.resident_bytes(123)


def test_native_resident_memory_is_positive():
    assert encoding_process.resident_bytes(os.getpid()) > 0


def _running(pid):
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True, check=False
    )
    return (
        result.returncode == 0 and bool(result.stdout.strip())
        and not result.stdout.lstrip().startswith("Z")
    )


def test_parent_pipe_closure_exits_child_without_linux_prctl(tmp_path):
    child = (
        "import os,sys,time; "
        "from alohamini.datasets.encoding_process import watch_parent; "
        "watch_parent(int(sys.argv[1]),int(sys.argv[2])); "
        "print('ready',flush=True); time.sleep(300)"
    )
    parent = """
import json, os, subprocess, sys, time
read_fd, write_fd = os.pipe()
process = subprocess.Popen(
    [sys.executable, '-c', sys.argv[1], str(read_fd), str(os.getpid())],
    pass_fds=(read_fd,), stdout=subprocess.PIPE, text=True, start_new_session=True,
)
os.close(read_fd)
assert process.stdout.readline().strip() == 'ready'
print(json.dumps({'pid': process.pid}), flush=True)
time.sleep(300)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", parent, child], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    pid = None
    try:
        # Bound the startup read too; a broken child must not hang the test suite.
        import selectors

        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=15), "Parent guard did not start"
        line = process.stdout.readline()
        assert line, process.stderr.read() if process.poll() is not None else "Parent exited"
        pid = json.loads(line)["pid"]
        assert _running(pid)
        process.kill()  # No Python cleanup, signal handler, or explicit pipe close.
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while _running(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _running(pid), "Encoder remained active after recorder death"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if pid is not None and _running(pid):
            os.kill(pid, 9)
        process.stdout.close()
        process.stderr.close()


def test_encoder_monitor_failure_preserves_frames_and_falls_back(tmp_path, monkeypatch, caplog):
    import contextlib

    from test_recording_background_video import submit, wait_until
    from test_recording_storage import metadata

    from alohamini.datasets import recording_video
    from alohamini.datasets.record import _EpisodeWriter
    from alohamini.datasets.video import inspect_video

    def unavailable(pid):
        raise OSError("process accounting unavailable")

    monkeypatch.setattr(recording_video, "resident_bytes", unavailable)
    with contextlib.closing(_EpisodeWriter(
        tmp_path / "record", fps=30, task="test", robot_metadata=metadata(),
        background_video_encoding=True,
    )) as writer:
        writer.begin_episode()
        process = writer._video_encoder.process
        for index in range(4):
            submit(writer, index)
        wait_until(lambda: process.poll() is not None)
        assert len(list((writer._pending / "images").rglob("*.jpg"))) == 4
        writer.save_episode()
        path = writer.root / "episodes/episode_000000/videos/forward.mp4"
        assert inspect_video(path, decode=True)["frames"] == 4
        assert "memory monitoring unavailable" in caplog.text


@pytest.mark.parametrize("background", [False, True])
def test_videotoolbox_encoding_or_cpu_fallback_preserves_frames(tmp_path, background):
    import contextlib

    from test_recording_background_video import submit
    from test_recording_storage import metadata

    from alohamini.datasets.record import _EpisodeWriter
    from alohamini.datasets.video import inspect_video

    with contextlib.closing(_EpisodeWriter(
        tmp_path / "record", fps=30, task="test", robot_metadata=metadata(),
        video_encoder="videotoolbox", background_video_encoding=background, profile_timing=True,
    )) as writer:
        writer.begin_episode()
        for index in range(4):
            submit(writer, index, shape=(480, 640, 3))
        writer.save_episode()
        path = writer.root / "episodes/episode_000000/videos/forward.mp4"
        assert inspect_video(path, decode=True) == {
            "frames": 4, "fps": 30.0, "shape": [480, 640, 3],
        }


@pytest.mark.skipif(sys.platform != "darwin", reason="VideoToolbox hardware requires macOS")
def test_native_videotoolbox_hardware_encoding(tmp_path):
    import av
    import numpy as np

    from alohamini.datasets.recording_video import encoder_settings
    from alohamini.datasets.video import _encode_frames, inspect_video

    codec, options = encoder_settings("videotoolbox")
    if codec not in av.codecs_available:
        pytest.skip("VideoToolbox encoder is absent from this FFmpeg build")
    path = tmp_path / "hardware.mp4"
    try:
        _encode_frames(
            [av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="rgb24")
             for _ in range(4)], path, 30, [480, 640, 3], codec=codec, options=options,
        )
    except av.error.ExternalError as exc:
        pytest.skip(f"VideoToolbox hardware unavailable: {exc}")
    assert inspect_video(path, decode=True) == {
        "frames": 4, "fps": 30.0, "shape": [480, 640, 3],
    }


def test_cli_accepts_apple_video_encoder():
    from unittest.mock import patch

    from alohamini.cli import main

    with patch("alohamini.apps.recording.record") as record:
        assert main([
            "record", "--host", "test", "--robot_model", "alohamini2pro",
            "--dataset", "test", "--task", "test", "--video-encoder", "videotoolbox",
        ]) == 0
    assert record.call_args.kwargs["video_encoder"] == "videotoolbox"
