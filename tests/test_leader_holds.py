import importlib.util

import pytest
from test_replay import replay_snapshot

from alohamini.apps.teleoperation import ready_units


@pytest.fixture
def rig():
    from test_teleoperation import LeaderTests

    if importlib.util.find_spec("scservo_sdk") is None:
        pytest.skip("optional Feetech SDK not installed")
    case = LeaderTests()
    case.setUp()
    for serial in case.serials.values():
        for motor in serial.registers:
            for address, width, value in ((14, 1, 74), (15, 1, 40), (62, 1, 50), (63, 1, 25)):
                serial.set(motor, address, width, value)
    try:
        with case.leader() as leader:
            yield leader, case.serials
    finally:
        case.doCleanups()


def test_hold_is_current_position_only_with_ram_ceiling_and_verified_release(rig):
    leader, serials = rig
    serial = serials["left"]
    serial.requests.clear()
    leader.hold_joint("arm_left_shoulder_pan.pos", output_limit=0.2)
    assert serial.get(1, 42, 2) == serial.get(1, 56, 2) == 1500
    assert serial.get(1, 48, 2) == 200
    assert serial.get(1, 40, 1) == 1
    writes = [p for p in serial.requests if p[4] == 3]
    assert [p[5] for p in writes] == [48, 42, 40]
    assert all(serial.get(i, 40, 1) == 0 for i in serial.registers if i != 1)
    leader.release_holds()
    assert serial.get(1, 40, 1) == 0 and serial.get(1, 48, 2) == 1000
    leader.read_handover(ready_units(replay_snapshot(), "alohamini2pro", "client"))


def test_old_goal_write_failure_never_enables_torque(rig):
    leader, serials = rig
    serials["left"].ignore_writes.add((1, 42))
    with pytest.raises(ConnectionError, match="write not accepted"):
        leader.hold_joint("arm_left_shoulder_pan.pos", output_limit=0.2)
    assert serials["left"].get(1, 40, 1) == 0
    assert serials["left"].get(1, 48, 2) == 1000


def test_lost_enable_ack_still_unloads_attempted_joint(rig, monkeypatch):
    leader, serials = rig
    device = leader.devices["left"]
    write = device._write_verified

    def lost(name, address, width, value):
        write(name, address, width, value)
        if address == 40 and value == 1:
            raise ConnectionError("lost enable ACK")

    monkeypatch.setattr(device, "_write_verified", lost)
    with pytest.raises(ConnectionError, match="lost enable ACK"):
        leader.hold_joint("arm_left_shoulder_pan.pos", output_limit=0.2)
    assert serials["left"].get(1, 40, 1) == 0
    assert serials["left"].get(1, 48, 2) == 1000


def test_handover_rejects_voltage_alarm_and_out_of_range_position(rig):
    leader, serials = rig
    units = ready_units(replay_snapshot(), "alohamini2pro", "client")
    serials["left"].errors[(1, 0x82, 56)] = 1
    with pytest.raises(ConnectionError, match="alarm|voltage"):
        leader.read_handover(units)
    serials["left"].errors.clear()
    serials["left"].set(1, 56, 2, 900)
    with pytest.raises(ConnectionError, match="outside"):
        leader.read_handover(units)


def test_configured_voltage_limit_blocks_hold_without_rewriting_protection(rig):
    leader, serials = rig
    serial = serials["left"]
    serial.set(1, 15, 1, 60)
    serial.requests.clear()
    with pytest.raises(RuntimeError, match="voltage"):
        leader.hold_joint("arm_left_shoulder_pan.pos", output_limit=0.2)
    assert not [p for p in serial.requests if p[4] == 3]
    assert serial.get(1, 15, 1) == 60


def test_hold_slip_is_detected_and_close_restores_cap(rig):
    leader, serials = rig
    leader.hold_joint("arm_left_shoulder_pan.pos", output_limit=0.2)
    serials["left"].set(1, 56, 2, 1550)
    with pytest.raises(ConnectionError, match="slipped"):
        leader.read_handover(ready_units(replay_snapshot(), "alohamini2pro", "client"))
    leader.close()
    assert serials["left"].get(1, 40, 1) == 0
    assert serials["left"].get(1, 48, 2) == 1000
