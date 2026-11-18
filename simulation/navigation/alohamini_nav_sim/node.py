import json
import math
import time
import uuid
from pathlib import Path

import rclpy
import yaml
from alohamini_sensors.profile import sensor_profile
from geometry_msgs.msg import Twist, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool

from alohamini.kinematics import OmniBaseKinematics
from alohamini.model import get_robot_model
from alohamini.schema import BodyVelocity

from .world import World


class Simulator(Node):
    def __init__(self):
        super().__init__("alohamini_nav_sim")
        self.declare_parameter("world", "")
        self.declare_parameter("robot_model", "alohamini2edu")
        self.declare_parameter("sensor_profile", "")
        self.model = self.get_parameter("robot_model").value
        model = get_robot_model(self.model)
        self.kinematics = OmniBaseKinematics(model.wheel_radius_m, model.base_radius_m)
        self.wheel_positions = [0.0] * 3
        self.profile = sensor_profile(self.model, self.get_parameter("sensor_profile").value)
        self.world = World(yaml.safe_load(Path(self.get_parameter("world").value).read_text()))
        self.clock = self.create_publisher(Clock, "/clock", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_profile_sensor_data)
        self.imu_pub = self.create_publisher(Imu, "/imu/data_raw", qos_profile_sensor_data)
        self.wheel_pub = self.create_publisher(TwistWithCovarianceStamped, "/wheel/twist", 10)
        self.joint_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.truth_pub = self.create_publisher(Odometry, "/simulation/ground_truth", 10)
        self.state_pub = self.create_publisher(String, "/alohamini_nav_sim/state_json", 10)
        self.create_subscription(Twist, "/cmd_vel_safe", self.command, 1)
        self.create_service(SetBool, "~/command_enable", self.enable)
        self.enabled = False
        self.velocity = (0.0, 0.0, 0.0)
        self.last_command = 0.0
        self.steps = 0
        self.session = "simulation-" + uuid.uuid4().hex
        self.create_timer(0.02, self.tick)

    def enable(self, request, response):
        self.enabled = request.data
        self.velocity = (0.0, 0.0, 0.0)
        response.success = True
        response.message = "Simulation control enabled" if self.enabled else "Simulation stopped"
        return response

    def command(self, msg):
        values = (msg.linear.x, msg.linear.y, msg.angular.z)
        if self.enabled and all(math.isfinite(x) for x in values):
            self.velocity = tuple(
                max(-limit, min(limit, v)) for v, limit in zip(values, (0.4, 0.4, 1.0), strict=True)
            )
            self.last_command = time.monotonic()

    def tick(self):
        velocity = (
            self.velocity
            if self.enabled and time.monotonic() - self.last_command < 0.5
            else (0.0, 0.0, 0.0)
        )
        vx, vy, wz = self.world.step(*velocity, 0.02)
        self.steps += 1
        nanoseconds = 1_000_000_000 + self.steps * 20_000_000
        clock = Clock()
        clock.clock.sec, clock.clock.nanosec = divmod(nanoseconds, 10**9)
        self.clock.publish(clock)
        wheel = TwistWithCovarianceStamped()
        wheel.header.stamp, wheel.header.frame_id = clock.clock, "base_link"
        wheel.twist.twist.linear.x, wheel.twist.twist.linear.y = vx, vy
        wheel.twist.twist.angular.z = wz
        wheel.twist.covariance[0] = wheel.twist.covariance[7] = wheel.twist.covariance[35] = 0.001
        self.wheel_pub.publish(wheel)
        joints = JointState()
        joints.header = wheel.header
        joints.name = [f"wheel{i}_joint" for i in (1, 2, 3)]
        joints.velocity = list(self.kinematics.body_to_wheels(BodyVelocity(vx, vy, wz)))
        self.wheel_positions = [
            p + v * 0.02 for p, v in zip(self.wheel_positions, joints.velocity, strict=True)
        ]
        joints.position = self.wheel_positions
        self.joint_pub.publish(joints)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = clock.clock, self.profile["imu"]["frame_id"]
        imu.orientation_covariance[0] = -1.0
        imu.angular_velocity.z = wz
        imu.angular_velocity_covariance[0] = imu.angular_velocity_covariance[
            4
        ] = imu.angular_velocity_covariance[8] = 0.001
        imu.linear_acceleration.z = 9.80665
        imu.linear_acceleration_covariance[0] = -1.0
        self.imu_pub.publish(imu)
        truth = Odometry()
        truth.header.stamp, truth.header.frame_id, truth.child_frame_id = (
            clock.clock,
            "simulation_world",
            "base_link",
        )
        truth.pose.pose.position.x, truth.pose.pose.position.y = self.world.x, self.world.y
        truth.pose.pose.orientation.z, truth.pose.pose.orientation.w = (
            math.sin(self.world.yaw / 2),
            math.cos(self.world.yaw / 2),
        )
        truth.twist = wheel.twist
        self.truth_pub.publish(truth)
        self.state_pub.publish(
            String(
                data=json.dumps(
                    {
                        "_robot_metadata": {"robot_model": self.model},
                        "_safety": {"host_session_id": self.session},
                    }
                )
            )
        )
        if self.steps % 5 == 0:
            laser = self.profile["lidar"]
            scan = LaserScan()
            scan.header.stamp, scan.header.frame_id = clock.clock, laser["frame_id"]
            scan.angle_min, scan.angle_max, scan.angle_increment = (
                -math.pi,
                math.pi,
                2 * math.pi / 359,
            )
            scan.range_min, scan.range_max, scan.scan_time = 0.15, 8.0, 0.1
            scan.ranges = self.world.scan(laser["xyz"], laser["rpy"])
            self.scan_pub.publish(scan)


def main():
    rclpy.init()
    node = Simulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
