import math
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
import yaml
from alohamini_joycon_teleop.mapping import (
    LEVEL_TCP_QUATERNION,
    base_command,
    base_relative_quaternion,
    faucet_translation_velocity,
    integrate_base_preview,
    lift_stick_requested,
    next_lift_stick_latch,
    normalize_stick,
    quaternion_axis_signs,
    quaternion_multiply,
    relative_quaternion,
    step_orientation_toward,
    tcp_button_delta,
)
from ament_index_python.packages import get_package_share_directory

from alohamini.model import get_robot_model


@pytest.fixture
def teleop_factory():
    import rclpy
    from alohamini_joycon_teleop.teleop_node import JoyConTeleop

    @contextmanager
    def create(hardware_mode):
        rclpy.init(
            args=[
                "--ros-args",
                "-p",
                f"hardware_mode:={str(hardware_mode).lower()}",
                "-p",
                "arm_control_mode:=moveit",
            ]
        )
        node = None
        try:
            node = JoyConTeleop()
            yield node
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()

    return create


def test_hardware_ik_uses_shared_ranges_and_the_host_execution_margin(teleop_factory):
    with teleop_factory(True) as node:
        assert node.model_joint_margin > math.radians(2)
        for side in ("left", "right"):
            geometry = node.kinematics[side]
            assert geometry.joint_limits[0] == pytest.approx(
                (-math.radians(125), math.radians(125))
            )
            assert geometry.joint_limits[4] == pytest.approx((-math.radians(82), math.radians(82)))


@pytest.mark.parametrize("hardware_mode", [False, True], ids=["preview", "hardware"])
def test_fk_and_ik_requests_use_only_the_selected_model_joints(teleop_factory, hardware_mode):
    from alohamini_joycon_teleop.teleop_node import ROOT_JOINTS

    model = get_robot_model("alohamini2pro")
    description = "collision" if hardware_mode else "planning"
    expected = {
        joint.attrib["name"]
        for joint in model.description_tree(description).findall("joint")
        if joint.attrib["type"] != "fixed" and joint.find("mimic") is None
    }
    with teleop_factory(hardware_mode) as node:
        assert node.positions.keys() == expected
        assert node.positions["vertical_move"] == node.lift_target == 0.0
        node.positions["vertical_move"] = 0.17
        if hardware_mode:
            assert not set(ROOT_JOINTS) & node.positions.keys()
        else:
            node.update_base(None, {"buttons": _buttons(), "stick": [2000.0, 4095.0]}, False, 0.1)
            assert node.preview_base[0] > 0.0
            assert [node.positions[name] for name in ROOT_JOINTS] == node.preview_base
        node.fk_client = Mock()
        node.ik_client = Mock()
        node.request_fk("left")
        node.request_ik(
            "right",
            {"position": [0.0, -0.2, 0.1], "orientation": [0.0, 0.0, 0.0, 1.0]},
            1.0,
        )
        fk = node.fk_client.call_async.call_args.args[0]
        ik = node.ik_client.call_async.call_args.args[0]
        for state in (fk.robot_state, ik.ik_request.robot_state):
            assert len(state.joint_state.name) == len(expected)
            assert set(state.joint_state.name) == expected
            positions = dict(zip(state.joint_state.name, state.joint_state.position, strict=True))
            assert positions == node.positions
            assert positions["vertical_move"] == 0.17


@pytest.mark.parametrize("hardware_mode", [False, True], ids=["preview", "hardware"])
def test_initial_pose_uses_the_selected_models_semantic_pair(
    teleop_factory, monkeypatch, tmp_path, hardware_mode
):
    from alohamini_joycon_teleop import teleop_node

    model = get_robot_model("alohamini2pro")
    semantic_paths = {}
    expected_home = {"semantic": 0.4, "semantic_planning": 0.5}
    for name, value in expected_home.items():
        semantic = ET.parse(model.description_path(name))
        semantic.find(
            "group_state[@name='home'][@group='left_arm']/joint[@name='left_wrist_flex']"
        ).set("value", str(value))
        path = tmp_path / f"{name}.srdf"
        semantic.write(path)
        semantic_paths[name] = path

    def description_path(name):
        return semantic_paths[name] if name in semantic_paths else model.description_path(name)

    selected_model = Mock()
    selected_model.description_path.side_effect = description_path
    selected_model.description_tree.side_effect = model.description_tree
    monkeypatch.setattr(teleop_node, "get_robot_model", lambda _model_id: selected_model)
    with teleop_factory(hardware_mode) as node:
        semantic_name = "semantic" if hardware_mode else "semantic_planning"
        assert node.positions["left_wrist_flex"] == expected_home[semantic_name]


@pytest.mark.parametrize("hardware_mode", [False, True], ids=["preview", "hardware"])
@pytest.mark.parametrize("invalid", ["unknown", "missing", "nonfinite"])
def test_robot_state_rejects_invalid_model_state(teleop_factory, hardware_mode, invalid):
    from moveit_msgs.msg import RobotState

    with teleop_factory(hardware_mode) as node:
        if invalid == "unknown":
            node.positions["unknown_joint"] = 0.0
        elif invalid == "missing":
            del node.positions["vertical_move"]
        else:
            node.positions["vertical_move"] = math.nan
        with pytest.raises(ValueError, match="Robot state"):
            node.fill_robot_state(RobotState())


def test_stick_deadzone_and_range():
    assert normalize_stick(2000, 0.25) == 0.0
    assert normalize_stick(2200, 0.25) == 0.0
    assert normalize_stick(4000, 0.25) == pytest.approx(1.0)
    assert normalize_stick(0, 0.25) == pytest.approx(-1.0)


def _buttons(**overrides) -> dict[str, bool]:
    buttons = {
        "stick": False,
        "shoulder": False,
        "trigger": False,
        "sl": False,
        "sr": False,
        "menu": False,
        "relatch": False,
        "up": False,
        "down": False,
        "left": False,
        "right": False,
    }
    buttons.update(overrides)
    return buttons


def test_tcp_buttons_forward_back_left_right():
    speed, dt = 0.1, 0.5
    step = speed * dt
    assert tcp_button_delta(_buttons(up=True), speed, dt) == pytest.approx([0.0, -step, 0.0])
    assert tcp_button_delta(_buttons(down=True), speed, dt) == pytest.approx([0.0, step, 0.0])
    assert tcp_button_delta(_buttons(left=True), speed, dt) == pytest.approx([step, 0.0, 0.0])
    assert tcp_button_delta(_buttons(right=True), speed, dt) == pytest.approx([-step, 0.0, 0.0])


def test_tcp_shoulder_combo_is_z():
    speed, dt = 0.1, 0.5
    step = speed * dt
    assert tcp_button_delta(_buttons(shoulder=True, up=True), speed, dt) == pytest.approx(
        [0.0, 0.0, step]
    )
    assert tcp_button_delta(_buttons(shoulder=True, down=True), speed, dt) == pytest.approx(
        [0.0, 0.0, -step]
    )
    assert tcp_button_delta(_buttons(shoulder=True, left=True), speed, dt) == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert tcp_button_delta(_buttons(shoulder=True, right=True), speed, dt) == pytest.approx(
        [0.0, 0.0, 0.0]
    )


def test_lift_requires_shoulder_and_vertical_stick_outside_deadzone():
    assert lift_stick_requested(_buttons(shoulder=True), 4095.0, 0.25)
    assert not lift_stick_requested(_buttons(shoulder=True), 2048.0, 0.25)
    assert not lift_stick_requested(_buttons(), 4095.0, 0.25)
    assert not lift_stick_requested(_buttons(shoulder=True, sl=True), 4095.0, 0.25)


def test_lift_gesture_stays_latched_until_stick_centers():
    held = _buttons(shoulder=True)
    false_report = _buttons(shoulder=False)
    clutch = _buttons(shoulder=False, sl=True)

    assert next_lift_stick_latch(False, held, 4095.0, 0.25)
    assert next_lift_stick_latch(True, false_report, 4095.0, 0.25)
    assert not next_lift_stick_latch(True, false_report, 2048.0, 0.25)
    assert not next_lift_stick_latch(True, clutch, 4095.0, 0.25)


def test_base_command_translation():
    # Single stick: up = forward, right = right strafe; yaw is handled by the
    # caller (shoulder-modified turn stick).
    assert base_command([(0.0, 1.0)], 0.2, 0.5) == pytest.approx((0.2, 0.0, 0.0))
    assert base_command([(1.0, 0.0)], 0.2, 0.5) == pytest.approx((0.0, -0.2, 0.0))
    # Two sticks sum and clamp.
    assert base_command([(0.5, 0.0), (0.5, 0.0)], 0.2, 0.5) == pytest.approx((0.0, -0.2, 0.0))
    assert base_command([(0.0, 1.0), (0.7, 0.0)], 0.2, 0.5) == pytest.approx((0.2, -0.14, 0.0))


def test_base_preview_uses_rep103_axes():
    assert integrate_base_preview([1.0, 2.0, 0.25], 0.4, -0.2, 0.5, 0.5) == pytest.approx(
        [1.2, 1.9, 0.5]
    )


def test_faucet_translation_is_xyz_not_tcp_frame_translation():
    identity = [0.0, 0.0, 0.0, 1.0]
    assert faucet_translation_velocity(0.0, 1.0, identity, 0.1) == pytest.approx([0.0, -0.1, 0.0])
    # {side}_Base +X is robot-left, so stick-right is -X and stick-left is +X.
    assert faucet_translation_velocity(1.0, 0.0, identity, 0.1) == pytest.approx([-0.1, 0.0, 0.0])
    assert faucet_translation_velocity(-1.0, 0.0, identity, 0.1) == pytest.approx([0.1, 0.0, 0.0])
    # Relative +90 degree yaw turns the nozzle's forward direction toward +X.
    yaw_90 = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]
    assert faucet_translation_velocity(0.0, 1.0, yaw_90, 0.1) == pytest.approx(
        [0.1, 0.0, 0.0], abs=1.0e-9
    )


def test_relative_quaternion_removes_arbitrary_latch_attitude():
    anchor = [math.sin(0.3), 0.0, 0.0, math.cos(0.3)]
    assert relative_quaternion(anchor, anchor) == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_base_relative_quaternion_uses_fixed_axis_composition_order():
    roll = [math.sin(0.25), 0.0, 0.0, math.cos(0.25)]
    yaw = [0.0, 0.0, math.sin(0.35), math.cos(0.35)]
    current = quaternion_multiply(yaw, roll)
    assert base_relative_quaternion(current, roll) == pytest.approx(yaw)


def test_relative_quaternion_roll_sign_matches_gripper_command():
    roll = [math.sin(0.25), 0.0, 0.0, math.cos(0.25)]
    corrected = quaternion_axis_signs(roll, [-1.0, 1.0, 1.0])
    assert corrected == pytest.approx([-math.sin(0.25), 0.0, 0.0, math.cos(0.25)])
    assert quaternion_axis_signs(
        [0.0, math.sin(0.2), 0.0, math.cos(0.2)], [-1.0, 1.0, 1.0]
    ) == pytest.approx([0.0, math.sin(0.2), 0.0, math.cos(0.2)])


def test_relative_quaternion_yaw_sign_matches_gripper_command():
    yaw = [0.0, 0.0, math.sin(0.3), math.cos(0.3)]
    assert quaternion_axis_signs(yaw, [-1.0, 1.0, -1.0]) == pytest.approx(
        [0.0, 0.0, -math.sin(0.3), math.cos(0.3)]
    )


def test_orientation_step_is_rate_limited():
    identity = [0.0, 0.0, 0.0, 1.0]
    half_turn = [0.0, 0.0, 1.0, 0.0]  # 180 degrees about z
    # A big step caps the rotation at max_step_rad.
    small = step_orientation_toward(identity, half_turn, math.pi / 4.0)
    assert small == pytest.approx([0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)])
    # Within the cap the goal is reached exactly.
    tiny = step_orientation_toward(identity, small, math.pi / 4.0)
    assert tiny == pytest.approx(small)
    # Double-cover: approaching -goal yields the same rotation (up to sign).
    negative = step_orientation_toward(identity, [-value for value in half_turn], math.pi / 4.0)
    for expected, actual in zip(small, negative, strict=True):
        assert abs(expected - actual) < 1.0e-9 or abs(expected + actual) < 1.0e-9


def test_level_tcp_quaternion_matches_fk_golden_reference():
    golden = get_package_share_directory("alohamini_validation")
    with open(f"{golden}/config/fk_golden.yaml", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    matrix = data["samples"]["reference"]["transform"]
    # Rotation part of the validated "level grasp" reference FK; convert the
    # rotation matrix to an XYZW quaternion and compare with the constant.
    r00, r01, r02 = matrix[0][0], matrix[0][1], matrix[0][2]
    r10, r11, r12 = matrix[1][0], matrix[1][1], matrix[1][2]
    r20, r21, r22 = matrix[2][0], matrix[2][1], matrix[2][2]
    trace = r00 + r11 + r22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r21 - r12) / s
        y = (r02 - r20) / s
        z = (r10 - r01) / s
    else:
        if r00 >= r11 and r00 >= r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
            w = (r21 - r12) / s
        elif r11 >= r00 and r11 >= r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
            w = (r02 - r20) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s
            w = (r10 - r01) / s
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    quaternion = [value / norm for value in (x, y, z, w)]
    # Quaternions are double-cover; compare up to sign. The golden matrix
    # carries ~1e-12 rounding noise, so allow a small tolerance.
    for expected, actual in zip(LEVEL_TCP_QUATERNION, quaternion, strict=True):
        assert abs(expected - actual) < 1.0e-4 or abs(expected + actual) < 1.0e-4
    assert math.sqrt(sum(value * value for value in LEVEL_TCP_QUATERNION)) == pytest.approx(1.0)
