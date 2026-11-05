import os
import sys

import pytest
from alohamini_mujoco import cli


@pytest.mark.parametrize("empty", ["", " \t "])
@pytest.mark.parametrize(
    "arguments,flag",
    [
        (["view", "--libero-bundle"], "--libero-bundle"),
        (["view", "--collision-bundle"], "--collision-bundle"),
        (["view", "--inertial-bundle"], "--inertial-bundle"),
        (["view", "--scene-config"], "--scene-config"),
        (["view", "--trajectory"], "--trajectory"),
        (["replay"], "dataset"),
        (["collect", "trajectory.json", "--output"], "--output"),
        (
            ["libero-import", "--source", "source", "--python", "python", "--output"],
            "--output",
        ),
    ],
)
def test_empty_paths_fail_before_loading_assets(monkeypatch, capsys, empty, arguments, flag):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid paths must not reach model loading")

    monkeypatch.setattr(cli, "build_model", unexpected_load)
    with pytest.raises(SystemExit) as error:
        cli.main([*arguments, empty])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert flag in message
    assert "path cannot be empty" in message


@pytest.mark.parametrize(
    "arguments",
    [
        ["view", "--window-egl-device", "0"],
        ["replay", "dataset", "--view", "--window-egl-device=0"],
        [
            "evaluate", "checkpoint", "--view", "--window-egl-device", "0",
            "--libero-bundle", "bundle", "--scene-config", "scene.json", "--output", "report",
        ],
    ],
)
def test_egl_window_relaunches_before_model_loading(monkeypatch, tmp_path, arguments):
    tool = tmp_path / "opt/VirtualGL/bin/vglrun"
    tool.parent.mkdir(parents=True)
    tool.touch()
    library = tmp_path / "usr/lib/libvglfaker.so"
    library.parent.mkdir(parents=True)
    library.touch()
    monkeypatch.setattr(cli.shutil, "which", lambda name: str(tool))
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.delenv("VGL_ISACTIVE", raising=False)
    for key in ("__NV_PRIME_RENDER_OFFLOAD", "__NV_PRIME_RENDER_OFFLOAD_PROVIDER",
                "__GLX_VENDOR_LIBRARY_NAME"):
        monkeypatch.setenv(key, "requested-gpu")
    monkeypatch.setenv("MUJOCO_GL", "egl")

    def unexpected_load(*args, **kwargs):
        pytest.fail("EGL window routing must happen before model loading")

    monkeypatch.setattr(cli, "build_model", unexpected_load)

    class Relaunched(BaseException):
        pass

    def execute(executable, command, environment):
        assert executable == str(tool)
        assert command == [
            str(tool), "-d", "egl0", "-c", "proxy", "-sp", "-g", "-ld", str(library.parent),
            sys.executable, "-m", "alohamini_mujoco.cli", *arguments,
        ]
        assert environment["MUJOCO_GL"] == "glfw"
        assert environment["PYGLFW_LIBRARY_VARIANT"] == "x11"
        assert not any(key.startswith("__NV_PRIME_RENDER_OFFLOAD") for key in environment)
        assert "__GLX_VENDOR_LIBRARY_NAME" not in environment
        raise Relaunched

    monkeypatch.setattr(cli.os, "execvpe", execute)
    with pytest.raises(Relaunched):
        cli.main(arguments)
    assert os.environ["MUJOCO_GL"] == "egl"  # The caller's environment stays intact.


def test_active_egl_window_does_not_relaunch_and_replays_normally(monkeypatch):
    from alohamini_mujoco import collection

    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":test")
    for key in ("__NV_PRIME_RENDER_OFFLOAD", "__NV_PRIME_RENDER_OFFLOAD_PROVIDER",
                "__GLX_VENDOR_LIBRARY_NAME"):
        monkeypatch.delenv(key, raising=False)
    for key, value in {"VGL_ISACTIVE": "1", "VGL_DISPLAY": "egl0", "VGL_COMPRESS": "proxy",
                       "VGL_SPOIL": "0", "VGL_GAMMA": "0", "MUJOCO_GL": "glfw",
                       "PYGLFW_LIBRARY_VARIANT": "x11"}.items():
        monkeypatch.setenv(key, value)

    def unexpected_exec(*args):
        pytest.fail("An already configured EGL viewer must not recursively relaunch")

    monkeypatch.setattr(cli.os, "execvpe", unexpected_exec)
    calls = []

    def replay(dataset, **kwargs):
        calls.append((str(dataset), kwargs))
        return {"passed": True}

    monkeypatch.setattr(collection, "replay_episode", replay)
    assert cli.main(["replay", "dataset", "--view", "--paused", "--window-egl-device", "0"]) == 0
    assert calls == [("dataset", {"view": True, "paused": True, "collisions": False})]


@pytest.mark.parametrize("arguments", [
    ["view", "--window-egl-device", "-1"],
    ["replay", "dataset", "--window-egl-device", "0"],
])
def test_invalid_egl_window_request_fails_before_loading(monkeypatch, arguments):
    monkeypatch.setattr(cli.sys, "platform", "linux")

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid window routing must not load a model")

    monkeypatch.setattr(cli, "build_model", unexpected_load)
    assert cli.main(arguments) == 1


def test_missing_virtualgl_fails_without_loading_or_changing_devices(monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.delenv("VGL_ISACTIVE", raising=False)
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli.Path, "is_file", lambda path: False)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Missing VirtualGL must not load a robot or silently choose another GPU")

    monkeypatch.setattr(cli, "build_model", unexpected_load)
    assert cli.main(["view", "--window-egl-device", "0"]) == 1
    assert "Install VirtualGL 3.1.5" in capsys.readouterr().err
