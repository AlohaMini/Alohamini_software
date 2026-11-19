"""Launch Gazebo and validate its clock, controllers and lift trajectory."""

import argparse
import os
import signal
import subprocess
import time
from datetime import datetime, timezone

import rclpy
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from alohamini.paths import WorkspacePaths

CONTROLLERS = {
    "joint_state_broadcaster", "base_planar_controller", "wheel_velocity_controller",
    "lift_controller", "left_arm_controller", "right_arm_controller",
    "left_gripper_controller", "right_gripper_controller",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--domain", type=int, default=94)
    args = parser.parse_args()
    if args.timeout <= 0 or not 0 <= args.domain <= 101:
        parser.error("timeout must be positive; domain must be between 0 and 101")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    partition = f"alohamini_gazebo_check_{os.getpid()}"
    os.environ["IGN_PARTITION"] = partition
    os.environ["GZ_PARTITION"] = partition
    directory = WorkspacePaths().logs / "simulation"
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / ("gazebo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".log")
    rclpy.init()
    node = Node("gazebo_acceptance_observer")
    state = {"joints": {}, "clock": 0}

    def joint_state(message):
        state["joints"] = dict(zip(message.name, message.position, strict=True))

    def clock(_message):
        state["clock"] += 1

    node.create_subscription(JointState, "/alohamini_sim/joint_states", joint_state, 10)
    node.create_subscription(
        Clock, "/alohamini_sim/clock", clock,
        QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
    )
    manager = node.create_client(ListControllers, "/alohamini_sim/controller_manager/list_controllers")
    action = ActionClient(node, FollowJointTrajectory, "/alohamini_sim/lift_controller/follow_joint_trajectory")
    deadline = time.monotonic() + args.timeout
    process = None

    def until(predicate):
        while not predicate():
            if process.poll() is not None:
                raise RuntimeError("Gazebo launch exited before validation completed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Gazebo validation timed out")
            rclpy.spin_once(node, timeout_sec=0.05)

    success = False
    try:
        with log_path.open("w") as log:
            process = subprocess.Popen(
                ["ros2", "launch", "alohamini_gazebo", "simulation.launch.py", "headless:=true"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            until(lambda: state["clock"] >= 5 and "vertical_move" in state["joints"] and manager.service_is_ready())
            while True:
                future = manager.call_async(ListControllers.Request())
                until(future.done)
                active = {c.name for c in future.result().controller if c.state == "active"}
                if CONTROLLERS <= active:
                    break
                rclpy.spin_once(node, timeout_sec=0.05)
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Controllers did not activate: {sorted(CONTROLLERS - active)}")
            until(action.server_is_ready)
            start = state["joints"]["vertical_move"]
            target = start + 0.02
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = ["vertical_move"]
            goal.trajectory.points = [
                JointTrajectoryPoint(positions=[start], time_from_start=Duration()),
                JointTrajectoryPoint(positions=[target], time_from_start=Duration(sec=2)),
            ]
            goal.goal_time_tolerance = Duration(sec=3)
            request = action.send_goal_async(goal)
            until(request.done)
            handle = request.result()
            if not handle.accepted:
                raise RuntimeError("Simulated lift rejected its trajectory")
            result = handle.get_result_async()
            until(result.done)
            if result.result().status != GoalStatus.STATUS_SUCCEEDED:
                raise RuntimeError(f"Simulated lift trajectory failed: {result.result()}")
            until(lambda: abs(state["joints"]["vertical_move"] - target) < 0.005)
            success = True
            print(f"PASS: clock, {len(state['joints'])} joints, eight controllers and lift motion")
    except Exception as error:
        print(f"FAIL: {error}")
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                success = False
                print("FAIL: Gazebo did not stop cleanly")
        action.destroy()
        node.destroy_node()
        rclpy.shutdown()
    print(f"Log: {log_path}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
