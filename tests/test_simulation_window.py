"""Display selection and owned X11 process cleanup without touching a desktop."""

import os
import subprocess

import pytest

from alohamini.simulation import x11_window as window


def test_auto_window_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(window.sys, "platform", "linux")
    desktop = {"DISPLAY": ":0", "XDG_SESSION_TYPE": "x11"}
    for card, vendor in ((0, "0x8086"), (1, "0x10de")):
        path = tmp_path / f"card{card}/device/vendor"
        path.parent.mkdir(parents=True)
        path.write_text(vendor)
    assert window.isolated_window("auto", environment=desktop, drm_directory=tmp_path)
    assert not window.isolated_window(
        "auto", environment={**desktop, "XDG_SESSION_TYPE": "wayland"}, drm_directory=tmp_path
    )
    assert not window.isolated_window("auto", environment={}, drm_directory=tmp_path)
    (tmp_path / "card1/device/vendor").unlink()
    assert not window.isolated_window("auto", environment=desktop, drm_directory=tmp_path)
    assert window.isolated_window("isolated", environment={})
    assert not window.isolated_window("native", environment=desktop)
    with pytest.raises(ValueError, match="Window mode"):
        window.isolated_window("unknown")


class Process:
    def __init__(self, polls):
        self.polls = iter(polls)
        self.returncode = None
        self.terminated = False

    def poll(self):
        self.returncode = next(self.polls, self.returncode)
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout):
        return self.returncode


@pytest.mark.parametrize("close_server", [False, True])
def test_private_display_and_cleanup(tmp_path, monkeypatch, close_server):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XAUTHORITY", str(tmp_path / "desktop-authority"))
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(window.shutil, "which", lambda name: f"/usr/bin/{name}")
    authorizations, launches = [], []
    server = Process([None, 0] if close_server else [None])
    child = Process([None] if close_server else [7])

    def launch(command, **kwargs):
        launches.append((command, kwargs))
        if len(launches) == 1:
            fd = kwargs["pass_fds"][0]
            os.write(fd, b"19\n")
            return server
        authority = kwargs["env"]["XAUTHORITY"]
        assert os.stat(authority).st_mode & 0o777 == 0o600
        return child

    monkeypatch.setattr(window.subprocess, "Popen", launch)
    monkeypatch.setattr(
        window.subprocess, "run", lambda command, **_: authorizations.append(command)
    )
    result = window.run_isolated_window(["viewer", "--window-mode", "native"], title="Robot")
    assert result == (0 if close_server else 7)
    assert server.terminated == (not close_server)
    assert child.terminated == close_server
    assert "-no-host-grab" in launches[0][0]
    assert "-glamor" not in launches[0][0]
    assert launches[1][1]["env"]["DISPLAY"] == ":19"
    assert os.environ["DISPLAY"] == ":0"
    assert os.environ["XAUTHORITY"] == str(tmp_path / "desktop-authority")
    assert [command[4] for command in authorizations] == [":0", ":19"]
    assert not os.path.exists(launches[1][1]["env"]["XAUTHORITY"])


def test_startup_timeout_reaps_server(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(window.shutil, "which", lambda name: name)
    monkeypatch.setattr(window.subprocess, "run", lambda *_, **__: None)
    server = Process([None])
    monkeypatch.setattr(window.subprocess, "Popen", lambda *_, **__: server)
    monkeypatch.setattr(window.select, "select", lambda *_, **__: ([], [], []))
    with pytest.raises(RuntimeError, match="startup timed out"):
        window.run_isolated_window(["viewer"], title="Robot")
    assert server.terminated


def test_missing_tools_explain_installation(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(window.shutil, "which", lambda _: None)
    with pytest.raises(FileNotFoundError, match="xserver-xephyr and xauth"):
        window.run_isolated_window(["viewer"], title="Robot")


def test_unresponsive_owned_process_is_killed():
    class Stuck(Process):
        def wait(self, timeout):
            if self.returncode != -9:
                raise subprocess.TimeoutExpired("viewer", timeout)
            return self.returncode

        def kill(self):
            self.returncode = -9

    process = Stuck([None])
    window._stop(process)
    assert process.returncode == -9
