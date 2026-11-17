"""USB insertion and rule installation are emulated; no /dev or /etc writes."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from alohamini.hardware import usb_setup
from alohamini.hardware.usb_setup import USBController
from alohamini.paths import WorkspacePaths


def port(serial):
    return SimpleNamespace(serial_number=serial, vid=0x1234, pid=0x5678)


def test_capture_assigns_sides_by_insertion_without_reading_motor_ports(monkeypatch):
    left, right = port("LEFT123"), port("RIGHT456")
    snapshots = iter(
        [{}, *[{"/dev/ttyACM3": left}] * 4, *[{"/dev/ttyACM3": left, "/dev/ttyACM7": right}] * 3]
    )
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: next(snapshots))
    monkeypatch.setattr(usb_setup.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompt = Mock(return_value="")
    monkeypatch.setattr("builtins.input", prompt)
    bindings = usb_setup.capture_bindings("leader")
    assert bindings == dict(
        left=USBController("LEFT123", 0x1234, 0x5678),
        right=USBController("RIGHT456", 0x1234, 0x5678),
    )
    prompt.assert_called_once()
    rules = usb_setup.render_rules("leader", bindings).decode()
    assert 'ATTRS{serial}=="LEFT123", SYMLINK+="am_arm_leader_left"' in rules
    assert 'ATTRS{idVendor}=="1234"' in rules


@pytest.mark.parametrize("serial", [None, "", 'a"b', "a*b", "a\nb", "../../etc/passwd"])
def test_unsafe_or_missing_serial_is_rejected(serial):
    with pytest.raises(ValueError, match="serial number"):
        USBController(serial, 1, 2)


def test_ambiguous_insertion_and_duplicate_serials_are_rejected(monkeypatch):
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: {"a": port("A"), "b": port("B")})
    with pytest.raises(ValueError, match="one arm at a time"):
        usb_setup._wait_added({})
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: {"a": port("A"), "b": port("A")})
    monkeypatch.setattr(usb_setup.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="duplicate serial"):
        usb_setup._wait_added({"a": port("A")})


def test_setup_persists_serials_installs_aliases_and_reuses_after_port_renumbering(
    tmp_path, monkeypatch
):
    workspace, devices = WorkspacePaths(tmp_path), tmp_path / "dev"
    devices.mkdir()
    bindings = dict(
        left=USBController("L", 0x1234, 0x5678), right=USBController("R", 0x1234, 0x5678)
    )
    capture = Mock(return_value=bindings)
    monkeypatch.setattr(usb_setup, "capture_bindings", capture)
    current = {}
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: current)
    installed = []

    def install(role, prepared, matched, aliases, backup_dir):
        assert role == "leader"
        assert prepared.read_bytes() == usb_setup.render_rules(role, bindings)
        installed.append(dict(matched))
        for side, alias in aliases.items():
            alias.unlink(missing_ok=True)
            alias.symlink_to(matched[side])

    monkeypatch.setattr(usb_setup, "_install_rules", install)
    for left, right in (("ttyACM0", "ttyACM1"), ("ttyACM7", "ttyACM2")):
        for alias in devices.glob("am_arm_*"):
            alias.unlink()
        current.clear()
        for name, serial in ((left, "L"), (right, "R")):
            physical = devices / name
            physical.touch()
            current[str(physical)] = port(serial)
        aliases = usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=devices)
        assert aliases == tuple(
            str(devices / f"am_arm_leader_{side}") for side in ("left", "right")
        )
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=devices)
    capture.assert_called_once()
    assert len(installed) == 2
    stored = json.loads((workspace.calibration / "devices/leader.json").read_text())
    assert stored["sides"]["left"]["serial_number"] == "L"
    assert stored["sides"]["right"]["serial_number"] == "R"
    current.clear()
    with pytest.raises(ValueError, match="Connect the saved left"):
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=devices)
    assert len(installed) == 2


def test_privilege_failure_leaves_prepared_files_and_does_not_open_motors(tmp_path, monkeypatch):
    workspace = WorkspacePaths(tmp_path)
    bindings = dict(left=USBController("L", 1, 2), right=USBController("R", 1, 2))
    monkeypatch.setattr(usb_setup, "capture_bindings", lambda _: bindings)
    monkeypatch.setattr(
        usb_setup,
        "usb_ports",
        lambda: {
            "/dev/a": SimpleNamespace(serial_number="L", vid=1, pid=2),
            "/dev/b": SimpleNamespace(serial_number="R", vid=1, pid=2),
        },
    )
    install = Mock(side_effect=PermissionError("sudo cancelled"))
    monkeypatch.setattr(usb_setup, "_install_rules", install)
    with pytest.raises(PermissionError, match="sudo cancelled"):
        usb_setup.ensure_arm_ports("follower", workspace=workspace, device_dir=tmp_path / "dev")
    assert (workspace.calibration / "devices/follower.json").is_file()
    assert (workspace.calibration / "devices/follower.rules").is_file()
