"""Resolve registered Mac leader controllers without opening hardware."""

import json
from unittest.mock import Mock

import pytest

from alohamini.hardware import usb_setup
from alohamini.paths import WorkspacePaths


def test_mac_leader_registration_reuses_serials_after_port_renumbering(tmp_path, monkeypatch):
    from test_usb_setup import pair, port

    workspace = WorkspacePaths(tmp_path)
    bindings = pair("MAC")
    monkeypatch.setattr(usb_setup.sys, "platform", "darwin")
    capture = Mock(return_value=bindings)
    monkeypatch.setattr(usb_setup, "capture_bindings", capture)
    available = {"/dev/cu.usbmodem1": port("MAC-left"),
                 "/dev/cu.usbmodem2": port("MAC-right")}
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: available)
    monkeypatch.setattr(usb_setup, "_install_rules", lambda *a: pytest.fail("udev on macOS"))
    assert usb_setup.ensure_arm_ports("leader", workspace=workspace) == tuple(available)
    available = {"/dev/cu.usbmodem11": port("MAC-left"),
                 "/dev/cu.usbmodem22": port("MAC-right")}
    assert usb_setup.ensure_arm_ports("leader", workspace=workspace) == tuple(available)
    assert usb_setup.connected_leader_ports(workspace) == dict(
        zip(("left", "right"), available, strict=True)
    )
    capture.assert_called_once()
    assert not list(workspace.calibration.rglob("*.rules"))
    stored = workspace.calibration / "devices/leader.json"
    assert json.loads(stored.read_text())["sides"]["left"]["serial_number"] == "MAC-left"


def test_mac_leader_teleoperation_resolves_registered_pair(tmp_path, monkeypatch):
    from alohamini.hardware import leader

    workspace = WorkspacePaths(tmp_path)
    ports = {"left": "/dev/cu.usbmodem11", "right": "/dev/cu.usbmodem22"}
    monkeypatch.setattr(leader.sys, "platform", "darwin")
    monkeypatch.setattr(leader, "WorkspacePaths", lambda: workspace)
    monkeypatch.setattr(leader, "connected_leader_ports", lambda *_: ports)
    directory = workspace.calibration / "teleoperators/usb/pair"
    directory.mkdir(parents=True)
    for side in ports:
        (directory / f"am_leader_bi_{side}.json").touch()
    monkeypatch.setattr(leader, "leader_calibration_directory", lambda *a: directory)
    loaded = []
    monkeypatch.setattr(leader.BimanualLeader, "_load_device", lambda self, side, port, motors:
                        loaded.append((side, port)))
    arms = leader.BimanualLeader("alohamini2pro")
    assert arms._ports == ports and loaded == list(ports.items())


def test_mac_single_leader_respects_fixed_arm_and_rejects_ambiguous_pairs(tmp_path, monkeypatch):
    from test_usb_setup import pair, port, save_pair

    workspace = WorkspacePaths(tmp_path)
    first = pair("A")
    save_pair(workspace, first)
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: {"/dev/cu.left": port("A-left")})
    assert usb_setup.connected_leader_ports(workspace, ("left",)) == {"left": "/dev/cu.left"}
    second = {**pair("B"), "left": first["left"]}
    save_pair(workspace, second)
    with pytest.raises(ValueError, match="Multiple registered"):
        usb_setup.connected_leader_ports(workspace, ("left",))


