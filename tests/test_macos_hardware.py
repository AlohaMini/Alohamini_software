"""Mac platform paths are emulated; no cameras, serial ports or motor writes."""

from pathlib import Path
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from alohamini import environment
from alohamini.hardware import camera, find_cameras


def test_mac_locks_distinguish_native_apple_silicon_and_intel(monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment.platform, "machine", lambda: "arm64")
    root = Path("/repository")
    assert environment.lock_path(root, "pc") == root / "env/pc-osx-arm64.lock"
    monkeypatch.setattr(environment.platform, "machine", lambda: "x86_64")
    with pytest.raises(ValueError, match="Host runs on Ubuntu"):
        environment.lock_path(root, "host")
    with pytest.raises(ValueError, match="native arm64"):
        environment.lock_path(root, "pc")


def test_mac_serial_commands_check_only_hardware_versions(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "env").mkdir(parents=True)
    (root / "env/hardware.lock").write_text("pyserial==3.5\nfeetech-servo-sdk==1.0.0\n")
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(environment, "checkout", lambda: root)
    monkeypatch.setattr(environment, "installed_versions", lambda: {
        "pyserial": "3.5", "feetech-servo-sdk": "1.0.0",
    })
    with environment.runtime_environment(["calibrate", "leader"]):
        pass
    monkeypatch.setattr(environment, "installed_versions", lambda: {"pyserial": "3.5"})
    with pytest.raises(ValueError, match="hardware.lock"):
        with environment.runtime_environment(["find-port"]):
            pass


def test_mac_environment_sync_detects_other_python_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    prefix = tmp_path / "environment"
    run = Mock(return_value=SimpleNamespace(stdout=(
        f"101 {prefix}/bin/python3.12\n102 /usr/bin/python3\n103 {prefix}/bin/bash\n"
    )))
    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment.active_python_processes(prefix, tmp_path / "absent-proc") == [101]
    assert run.call_args.args[0] == ["ps", "-axo", "pid=,comm="]


def test_camera_indices_and_backend_follow_os(monkeypatch):
    import cv2

    assert camera.CameraConfig("0").device == 0
    assert camera.CameraConfig(1).device == 1
    for invalid in (-1, 60, True, "http://camera", "video.mp4"):
        with pytest.raises(ValueError):
            camera.CameraConfig(invalid)
    monkeypatch.setattr(camera.platform, "system", lambda: "Darwin")
    assert camera.camera_backend() == cv2.CAP_AVFOUNDATION
    monkeypatch.setattr(camera.platform, "system", lambda: "Linux")
    assert camera.camera_backend() == cv2.CAP_V4L2


def test_mac_camera_capture_keeps_rgb_protocol_and_timestamp(monkeypatch):
    import cv2

    monkeypatch.setattr(camera.platform, "system", lambda: "Darwin")
    stop, queue = Event(), Queue()
    queue.cancel_join_thread = lambda: None
    image = np.full((8, 16, 3), (10, 80, 220), dtype=np.uint8)
    capture = Mock()
    capture.isOpened.return_value = capture.set.return_value = True
    capture.get.return_value = 30.0

    def read():
        stop.set()
        return True, image

    capture.read.side_effect = read
    open_capture = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    camera._capture(camera.CameraConfig(0, width=16, height=8), queue, stop,
                    SimpleNamespace(value=False))
    stamp, jpeg, timing, stream = queue.get_nowait()
    assert stamp > 0 and timing["frames"] == 1 and stream is None
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    np.testing.assert_allclose(decoded[0, 0], [220, 80, 10], atol=3)
    open_capture.assert_called_once_with(0, cv2.CAP_AVFOUNDATION)
    capture.release.assert_called_once()


def test_mac_discovery_and_snapshot_preserve_numeric_devices(monkeypatch):
    monkeypatch.setattr(find_cameras.platform, "system", lambda: "Darwin")
    probe = Mock(side_effect=lambda index: {"id": index} if index in (0, 2) else None)
    monkeypatch.setattr(find_cameras, "_probe_one", probe)
    assert find_cameras.find_all_opencv_cameras() == [{"id": 0}, {"id": 2}]
    factory = Mock()
    monkeypatch.setattr(find_cameras, "OpenCVCamera", factory)
    find_cameras.create_camera_instance({"id": 0})
    assert factory.call_args.args[0].device == 0


def test_mac_camera_debug_passes_index_to_capture(monkeypatch):
    from examples.debug import test_cv

    monkeypatch.setattr(test_cv.platform, "system", lambda: "Darwin")
    capture = Mock()
    monkeypatch.setattr(test_cv, "save_images_from_cameras", capture)
    assert test_cv.main(["--camera", "2"]) == 0
    assert capture.call_args.args[0] == [{"type": "OpenCV", "id": 2}]


