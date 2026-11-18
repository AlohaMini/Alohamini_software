"""USB insertion and rule installation are emulated; no /dev or /etc writes."""

import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from alohamini.hardware import usb_setup
from alohamini.hardware.usb_setup import USBController
from alohamini.paths import WorkspacePaths


def port(serial):
    return SimpleNamespace(serial_number=serial, vid=0x1234, pid=0x5678)


def pair(name):
    return {side: USBController(f"{name}-{side}", 0x1234, 0x5678) for side in ("left", "right")}


def save_pair(workspace, bindings, *, active=False):
    directory = workspace.calibration / "devices"
    path = (
        directory / "leader.json"
        if active
        else directory / "leaders" / f"{usb_setup._pair_key(bindings)}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(schema_version=1, role="leader", sides={s: asdict(b) for s, b in bindings.items()})
        )
    )
    return path


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


def test_switching_leader_sets_preserves_bindings_and_reuses_rules(tmp_path, monkeypatch):
    workspace, devices = WorkspacePaths(tmp_path), tmp_path / "dev"
    devices.mkdir()
    first, second = pair("A"), pair("B")
    save_pair(workspace, first, active=True)
    current = {}
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: current)
    capture = Mock(return_value=second)
    monkeypatch.setattr(usb_setup, "capture_bindings", capture)
    installed = []

    def install(role, prepared, matched, aliases, backup_dir):
        installed.append(prepared.read_bytes())
        for side, alias in aliases.items():
            alias.unlink(missing_ok=True)
            alias.symlink_to(matched[side])

    monkeypatch.setattr(usb_setup, "_install_rules", install)
    aliases = {s: devices / f"am_arm_leader_{s}" for s in ("left", "right")}
    for bindings in (first, second, first, second):
        current.clear()
        for side, board in bindings.items():
            physical = devices / board.serial_number
            physical.touch()
            current[str(physical)] = port(board.serial_number)
            # Emulate udev using the previously installed rule on reconnection.
            if installed and board.serial_number.encode() in installed[-1]:
                aliases[side].unlink(missing_ok=True)
                aliases[side].symlink_to(physical)
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=devices)
        assert (
            usb_setup._load_bindings(workspace.calibration / "devices/leader.json", "leader")
            == bindings
        )
    capture.assert_called_once_with("leader")
    assert len(installed) == 2  # First registration and adding B; returning A needs no sudo.
    assert len(usb_setup._known_leaders(workspace.calibration / "devices")) == 2
    for board in (*first.values(), *second.values()):
        assert installed[-1].count(f'ATTRS{{serial}}=="{board.serial_number}"'.encode()) == 1


def test_legacy_calibration_stays_with_original_usb_pair_after_switch_or_cancel(
    tmp_path, monkeypatch
):
    workspace = WorkspacePaths(tmp_path)
    first, second = pair("A"), pair("B")
    save_pair(workspace, first, active=True)
    flat = workspace.calibration / "teleoperators"
    flat.mkdir()
    originals = {}
    for side in ("left", "right"):
        for suffix in ("json", "reference.json"):
            path = flat / f"am_leader_bi_{side}.{suffix}"
            path.write_text(f"original-{side}-{suffix}")
            originals[path.name] = path.read_bytes()
    usb_setup.preserve_legacy_leader_calibration(workspace, "am_leader_bi")
    first_cache = flat / "usb" / usb_setup._pair_key(first)
    assert {p.name: p.read_bytes() for p in first_cache.iterdir()} == originals
    save_pair(workspace, first)
    save_pair(workspace, second, active=True)
    monkeypatch.setattr(
        usb_setup,
        "usb_ports",
        lambda: {
            "/dev/left": port(second["left"].serial_number),
            "/dev/right": port(second["right"].serial_number),
        },
    )
    # Selecting or retrying a new set must not attach the old flat files to it.
    for _ in range(2):
        selected = usb_setup.leader_calibration_directory(
            workspace, "am_leader_bi", {"left": "/dev/left", "right": "/dev/right"}
        )
        assert selected == flat / "usb" / usb_setup._pair_key(second)
        assert not selected.exists()
    assert {p.name: p.read_bytes() for p in first_cache.iterdir()} == originals
    assert {p.name: p.read_bytes() for p in flat.glob("*.json")} == originals


def test_current_usb_pair_overrides_last_selected_pair_and_supports_one_side(tmp_path, monkeypatch):
    workspace = WorkspacePaths(tmp_path)
    first, second = pair("A"), pair("B")
    save_pair(workspace, first)
    save_pair(workspace, second, active=True)
    available = {f"/dev/{side}": port(board.serial_number) for side, board in first.items()}
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: available)
    ports = {side: f"/dev/{side}" for side in first}
    expected = workspace.calibration / "teleoperators/usb" / usb_setup._pair_key(first)
    assert usb_setup.leader_calibration_directory(workspace, "id", ports) == expected
    available.pop("/dev/right")
    assert usb_setup.leader_calibration_directory(workspace, "id", ports, ("left",)) == expected
    with pytest.raises(ValueError, match="both USB"):
        usb_setup.leader_calibration_directory(workspace, "id", ports)
    available["/dev/right"] = port(second["right"].serial_number)
    with pytest.raises(ValueError, match="not registered"):
        usb_setup.leader_calibration_directory(workspace, "id", ports)


def test_multiple_connected_sets_and_conflicting_sides_are_rejected(tmp_path, monkeypatch):
    workspace = WorkspacePaths(tmp_path)
    first, second = pair("A"), pair("B")
    save_pair(workspace, first, active=True)
    save_pair(workspace, second)
    monkeypatch.setattr(
        usb_setup,
        "usb_ports",
        lambda: {
            f"/dev/{b.serial_number}": port(b.serial_number)
            for b in (*first.values(), *second.values())
        },
    )
    capture = Mock(side_effect=AssertionError("must not guess sides"))
    monkeypatch.setattr(usb_setup, "capture_bindings", capture)
    with pytest.raises(ValueError, match="Multiple registered"):
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=tmp_path / "dev")
    with pytest.raises(ValueError, match="both left and right"):
        usb_setup._leader_rules({"A": first, "B": dict(left=second["left"], right=first["left"])})


def test_cancelling_new_usb_registration_keeps_existing_binding(tmp_path, monkeypatch):
    workspace = WorkspacePaths(tmp_path)
    path = save_pair(workspace, pair("A"), active=True)
    before = path.read_bytes()
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: {"/dev/new": port("NEW")})
    monkeypatch.setattr(usb_setup, "capture_bindings", Mock(side_effect=InterruptedError("cancel")))
    monkeypatch.setattr(usb_setup, "_install_rules", Mock(side_effect=AssertionError("install")))
    with pytest.raises(InterruptedError, match="cancel"):
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=tmp_path / "dev")
    assert path.read_bytes() == before
    assert list((workspace.calibration / "devices").iterdir()) == [path]


def test_new_registration_does_not_claim_unidentified_legacy_calibration(tmp_path, monkeypatch):
    workspace, devices = WorkspacePaths(tmp_path), tmp_path / "dev"
    devices.mkdir()
    bindings = pair("NEW")
    flat = workspace.calibration / "teleoperators"
    flat.mkdir(parents=True)
    for side in ("left", "right"):
        (flat / f"am_leader_bi_{side}.json").write_text("old file from an unknown leader")
    available = {}
    for side, board in bindings.items():
        physical = devices / side
        physical.touch()
        available[str(physical)] = port(board.serial_number)
    monkeypatch.setattr(usb_setup, "usb_ports", lambda: available)
    monkeypatch.setattr(usb_setup, "capture_bindings", lambda _: bindings)
    monkeypatch.setattr(usb_setup, "_install_rules", lambda *_: None)
    for _ in range(2):
        usb_setup.ensure_arm_ports("leader", workspace=workspace, device_dir=devices)
        usb_setup.preserve_legacy_leader_calibration(workspace, "am_leader_bi")
    assert not (flat / "usb").exists()
    assert not (workspace.calibration / "devices/leader-legacy.json").exists()
