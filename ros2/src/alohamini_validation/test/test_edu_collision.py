"""Check the EDU sensor housing in an isolated, nonexecuting MoveIt scene."""

import signal
import subprocess

import rclpy
from alohamini_validation.validate_moveit import Validator
from geometry_msgs.msg import Point, Pose, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

from alohamini.model import get_robot_model


def test_lidar_internal_contacts_do_not_disable_external_collision_checks(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_DOMAIN_ID", "86")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    model = get_robot_model("alohamini2edu")
    joint = model.description_tree("collision").find("./joint[@name='lidar_housing_joint']")
    x, y, z = map(float, joint.find("origin").get("xyz").split())
    node = None
    with (tmp_path / "moveit.log").open("w") as log:
        process = subprocess.Popen(
            [
                "ros2",
                "launch",
                "alohamini_moveit_config",
                "plan_only.launch.py",
                "robot_model:=alohamini2edu",
                "use_rviz:=false",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        rclpy.init(args=["--ros-args", "-p", "robot_model:=alohamini2edu"])
        try:
            node = Validator()
            valid, contacts = node.state_validity({})
            assert valid, contacts
            client = node.create_client(ApplyPlanningScene, "apply_planning_scene")
            assert client.wait_for_service(timeout_sec=10)
            obstacle = CollisionObject(
                header=Header(frame_id=joint.find("parent").get("link")),
                id="edu_lidar_external_test",
                primitives=[SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.005] * 3)],
                primitive_poses=[
                    Pose(position=Point(x=x, y=y, z=z), orientation=Quaternion(w=1.0))
                ],
                operation=CollisionObject.ADD,
            )
            scene = PlanningScene(is_diff=True)
            scene.world.collision_objects = [obstacle]
            assert node.call(client, ApplyPlanningScene.Request(scene=scene)).success
            valid, contacts = node.state_validity({})
            assert not valid
            assert tuple(sorted(("lidar_housing", obstacle.id))) in contacts
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.try_shutdown()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise
