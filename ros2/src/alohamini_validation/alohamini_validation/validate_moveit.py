# SPDX-License-Identifier: Apache-2.0
# Migrated from alohamini_ros2/alohamini_validation.
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
from rcl_interfaces.msg import ParameterType
from rcl_interfaces.srv import GetParameters
from rclpy.node import Node

ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw_joint",
    "wrist_roll",
)
DEFAULT_STATE = {
    "root_x_axis_joint": 0.0,
    "root_y_axis_joint": 0.0,
    "root_z_rotation_joint": 0.0,
    "vertical_move": 0.2,
    "wheel1_joint": 0.0,
    "wheel2_joint": 0.0,
    "wheel3_joint": 0.0,
    "left_shoulder_pan": 0.0,
    "left_shoulder_lift": -1.571,
    "left_elbow_flex": 1.571,
    "left_wrist_flex": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_roll": 0.0,
    "left_gripper": -1.8030294104,
    "right_shoulder_pan": 0.0,
    "right_shoulder_lift": -1.571,
    "right_elbow_flex": 1.571,
    "right_wrist_flex": 0.0,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_roll": 0.0,
    "right_gripper": -1.8030294104,
}
VIRTUAL_BASE_JOINTS = {"root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"}


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
            x, y, z, w = (
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
            )
        elif index == 1:
            scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
            x, y, z, w = (
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
            )
        else:
            scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
            x, y, z, w = (
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
    quaternion = np.asarray([x, y, z, w], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def pose_matrix(pose) -> np.ndarray:
    x, y, z, w = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    matrix = np.eye(4)
    matrix[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    return matrix


def orientation_error(left: np.ndarray, right: np.ndarray) -> float:
    relative = left[:3, :3].T @ right[:3, :3]
    return math.acos(min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0)))


def model_context(description: str, semantic_description: str):
    """Use the served model's root and joints for both physical and offline graphs."""
    robot, semantic = ET.fromstring(description), ET.fromstring(semantic_description)
    if robot.tag != "robot" or semantic.tag != "robot":
        raise ValueError("MoveIt must publish URDF and SRDF robot descriptions")
    links = {link.attrib["name"] for link in robot.findall("link")}
    children = {joint.find("child").attrib["link"] for joint in robot.findall("joint")}
    roots = links - children
    if len(roots) != 1:
        raise ValueError("MoveIt robot description must have exactly one root frame")
    state = {
        joint.attrib["name"]: DEFAULT_STATE.get(joint.attrib["name"], 0.0)
        for joint in robot.findall("joint")
        if joint.attrib["type"] != "fixed" and joint.find("mimic") is None
    }
    return roots.pop(), state, semantic


class Validator(Node):
    def __init__(self, *, namespace="/alohamini_plan_only") -> None:
        # The default targets the isolated offline launch. ROS __ns remapping
        # can select another MoveIt graph, including the physical model at /.
        super().__init__("alohamini_moveit_validation", namespace=namespace)
        try:
            self.parameters_client = self.create_client(GetParameters, "move_group/get_parameters")
            self.fk_client = self.create_client(GetPositionFK, "compute_fk")
            self.ik_client = self.create_client(GetPositionIK, "compute_ik")
            self.validity_client = self.create_client(GetStateValidity, "check_state_validity")
            for client in (
                self.parameters_client, self.fk_client, self.ik_client, self.validity_client
            ):
                if not client.wait_for_service(timeout_sec=15.0):
                    raise RuntimeError(f"MoveIt service unavailable: {client.srv_name}")
            response = self.call(
                self.parameters_client,
                GetParameters.Request(names=["robot_description", "robot_description_semantic"]),
            )
            if len(response.values) != 2 or any(
                value.type != ParameterType.PARAMETER_STRING or not value.string_value
                for value in response.values
            ):
                raise RuntimeError("MoveIt did not provide its URDF and SRDF parameters")
            self.root_frame, self.default_state, self.semantic = model_context(
                *(value.string_value for value in response.values)
            )
            self.description_xml = response.values[0].string_value
            self.link_names = {
                link.attrib["name"]
                for link in ET.fromstring(response.values[0].string_value).findall("link")
            }
        except BaseException:
            self.destroy_node()
            raise

    def call(self, client, request, timeout: float = 5.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() is None:
            raise RuntimeError("MoveIt service call timed out")
        return future.result()

    def fill_state(self, robot_state, values: dict[str, float]) -> None:
        unknown = values.keys() - self.default_state.keys()
        if unknown:
            raise ValueError(f"Joints absent from the served MoveIt model: {sorted(unknown)}")
        complete = dict(self.default_state)
        complete.update(values)
        robot_state.joint_state.name = list(complete)
        robot_state.joint_state.position = list(complete.values())

    def fk_in_model(self, side: str, q_rad: list[float]) -> tuple[np.ndarray, np.ndarray]:
        request = GetPositionFK.Request()
        request.header.frame_id = self.root_frame
        request.fk_link_names = [f"{side}_Base", f"{side}_tcp"]
        values = dict(zip((f"{side}_{name}" for name in ARM_SUFFIXES), q_rad, strict=True))
        self.fill_state(request.robot_state, values)
        response = self.call(self.fk_client, request)
        assert response.error_code.val == response.error_code.SUCCESS, response.error_code.val
        assert len(response.pose_stamped) == 2
        return tuple(pose_matrix(item.pose) for item in response.pose_stamped)

    def fk(self, side: str, q_rad: list[float]) -> np.ndarray:
        base, tcp = self.fk_in_model(side, q_rad)
        return np.linalg.inv(base) @ tcp

    def ik(self, side: str, target: np.ndarray, seed: list[float]) -> list[float]:
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = f"{side}_arm"
        ik.ik_link_name = f"{side}_tcp"
        ik.pose_stamped.header.frame_id = self.root_frame
        base, _ = self.fk_in_model(side, seed)
        model_target = base @ target
        (
            ik.pose_stamped.pose.position.x,
            ik.pose_stamped.pose.position.y,
            ik.pose_stamped.pose.position.z,
        ) = (float(value) for value in model_target[:3, 3])
        (
            ik.pose_stamped.pose.orientation.x,
            ik.pose_stamped.pose.orientation.y,
            ik.pose_stamped.pose.orientation.z,
            ik.pose_stamped.pose.orientation.w,
        ) = matrix_to_quaternion(model_target)
        self.fill_state(
            ik.robot_state,
            dict(zip((f"{side}_{name}" for name in ARM_SUFFIXES), seed, strict=True)),
        )
        ik.avoid_collisions = False
        ik.timeout = Duration(sec=1)
        response = self.call(self.ik_client, request)
        assert response.error_code.val == response.error_code.SUCCESS, response.error_code.val
        solution = dict(
            zip(
                response.solution.joint_state.name,
                response.solution.joint_state.position,
                strict=True,
            )
        )
        return [float(solution[f"{side}_{name}"]) for name in ARM_SUFFIXES]

    def state_validity(self, values: dict[str, float]) -> tuple[bool, set[tuple[str, str]]]:
        request = GetStateValidity.Request()
        self.fill_state(request.robot_state, values)
        response = self.call(self.validity_client, request)
        pairs = {
            tuple(sorted((contact.contact_body_1, contact.contact_body_2)))
            for contact in response.contacts
        }
        return bool(response.valid), pairs


def check_collision_states(node, collision):
    from alohamini.model import get_robot_model

    model = get_robot_model("alohamini2pro")
    view = "planning" if VIRTUAL_BASE_JOINTS <= node.default_state.keys() else "collision"
    assert ET.canonicalize(node.description_xml, strip_text=True) == ET.canonicalize(
        model.description_xml(view), strip_text=True
    ), "MoveIt is serving a different model; rebuild and restart the ROS graph"

    for state_name, state in collision["state_validity"].items():
        values = {}
        if "named_state" in state:
            values = {
                joint.attrib["name"]: float(joint.attrib["value"])
                for group in node.semantic.findall("group_state")
                if group.attrib["name"] == state["named_state"]
                for joint in group.findall("joint")
            }
            assert values, f"Missing SRDF state: {state['named_state']}"
        values.update(state.get("joints", {}))
        valid, pairs = node.state_validity({key: float(value) for key, value in values.items()})
        assert valid is bool(state["valid"]), (
            f"{state_name}: valid={valid}, contacts={sorted(pairs)}"
        )
        expected_pairs = {tuple(pair) for pair in state.get("contact_pairs", [])}
        assert pairs == expected_pairs, f"{state_name}: contacts={sorted(pairs)}"
    return tuple(collision["state_validity"])


def main() -> None:
    validation = Path(get_package_share_directory("alohamini_validation"))
    golden = load_yaml(validation / "config/fk_golden.yaml")
    collision = load_yaml(validation / "config/collision_baseline.yaml")
    rclpy.init()
    node = None
    try:
        node = Validator()
        tolerance = float(golden["tolerance"])
        for sample_name, sample in golden["samples"].items():
            expected = np.asarray(sample["transform"], dtype=float)
            for side in ("left", "right"):
                actual = node.fk(side, sample["q_rad"])
                assert np.allclose(actual[:3, 3], expected[:3, 3], atol=tolerance), (
                    f"{sample_name}/{side} position"
                )
                assert orientation_error(actual, expected) <= 1e-7, (
                    f"{sample_name}/{side} orientation"
                )
        print("[PASS] MoveIt FK golden")

        reference = golden["samples"]["reference"]["q_rad"]
        target = np.asarray(golden["samples"]["offset_a"]["transform"], dtype=float)
        for side in ("left", "right"):
            solution = node.ik(side, target, reference)
            recovered = node.fk(side, solution)
            assert np.linalg.norm(recovered[:3, 3] - target[:3, 3]) < 1e-4
            assert orientation_error(recovered, target) < 1e-3
        print("[PASS] MoveIt IK reachable targets")

        check_collision_states(node, collision)
        print("[PASS] MoveIt collision state baseline")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
