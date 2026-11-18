"""Keyboard mapping/navigation session with explicit motion enable and ordered exit."""

import argparse
import os
import select
import signal
import subprocess
import sys
import termios
import time
import tty
from datetime import datetime

from alohamini.paths import WorkspacePaths

from .base_teleop import KEYS


def launch_command(args):
    if args.simulation:
        command = [
            "ros2",
            "launch",
            "alohamini_nav_sim",
            "simulation.launch.py",
            f"mode:={args.mode}",
        ]
    else:
        command = [
            "ros2",
            "launch",
            "alohamini_navigation",
            args.mode + ".launch.py",
            f"host:={args.host}",
            f"start_sensors:={str(args.start_sensors).lower()}",
            f"enable_cameras:={str(not args.no_camera).lower()}",
        ]
    # The session owns RViz separately so it can close clients before servers.
    command += [f"robot_model:={args.robot_model}", "use_rviz:=false"]
    if args.mode == "navigation":
        command.append(f"map:={args.map}")
    return command


def close(process, *, process_group=False):
    if process.poll() is None:
        # ros2 launch forwards signals to its children; ros2 run requires a group
        # signal. Signalling launch's group twice can interrupt node destructors.
        if process_group:
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            print("Session process exceeded shutdown timeout; inspect its log", file=sys.stderr)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("mapping", "navigation"), default="mapping")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--robot-model", default="alohamini2edu")
    parser.add_argument("--map-name", default="map")
    parser.add_argument("--map", help="Saved map YAML for navigation")
    parser.add_argument("--start-sensors", action="store_true")
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-rviz", action="store_true")
    parser.add_argument("--record", help="New ROS recording name")
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        parser.error("An interactive terminal is required")
    if args.mode == "navigation" and not args.map:
        parser.error("Navigation requires --map")
    import rclpy
    from action_msgs.srv import CancelGoal
    from geometry_msgs.msg import Twist
    from rclpy.executors import ExternalShutdownException
    from rclpy.signals import SignalHandlerOptions
    from std_srvs.srv import SetBool

    from .map_session import call

    mode = "simulation" if args.simulation else "hardware"
    prefix = "/alohamini_nav_sim" if args.simulation else "/alohamini_lerobot_bridge"
    logs = WorkspacePaths().logs / "navigation" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    logs.mkdir(parents=True)
    # Keep the context alive while finally disables motion and closes children.
    # Python's KeyboardInterrupt owns Ctrl+C for this terminal application.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("alohamini_edu_session")
    publisher = node.create_publisher(Twist, "/cmd_vel", 1)
    settings = termios.tcgetattr(sys.stdin)
    processes, streams = [], []
    viewer = recorder = None
    status = 0

    def start(command, name):
        stream = (logs / name).open("w")
        streams.append(stream)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(process)
        return process

    def stop_motion():
        if rclpy.ok():
            try:
                publisher.publish(Twist())
            except RuntimeError as error:
                print(f"Stop publication failed: {error}", file=sys.stderr)

    def stop_navigation():
        call(node, SetBool, prefix + "/command_enable", SetBool.Request(data=False), 5)
        stop_motion()
        for action in ("navigate_to_pose", "navigate_through_poses"):
            service = f"/{action}/_action/cancel_goal"
            client = node.create_client(CancelGoal, service)
            try:
                if client.wait_for_service(timeout_sec=0.5):
                    future = client.call_async(CancelGoal.Request())
                    rclpy.spin_until_future_complete(node, future, timeout_sec=3)
                    if not future.done():
                        print(f"Cancel timed out for {action}; motion remains disabled")
            finally:
                node.destroy_client(client)

    def map_command(command):
        words = [
            "ros2",
            "run",
            "alohamini_navigation",
            "map_session",
            command,
            "--runtime-mode",
            mode,
            "--robot-model",
            args.robot_model,
        ]
        if command == "save":
            words.append(args.map_name)
        return subprocess.run(words, stdin=subprocess.DEVNULL, check=False).returncode == 0

    try:
        launch = start(launch_command(args), "launch.log")
        if not args.no_rviz:
            viewer = start(
                [
                    "ros2",
                    "launch",
                    "alohamini_navigation",
                    "rviz.launch.py",
                    f"use_sim_time:={str(args.simulation).lower()}",
                ],
                "rviz.log",
            )
        if args.record:
            recorder = start(
                [
                    "ros2",
                    "run",
                    "alohamini_navigation",
                    "map_session",
                    "record",
                    args.record,
                    "--runtime-mode",
                    mode,
                    "--robot-model",
                    args.robot_model,
                ],
                "record.log",
            )
        print(f"Logs: {logs}\nMotion starts disabled. e: enable  Space/0: stop")
        if args.mode == "mapping":
            print(
                "w/s: forward/back  a/d: turn  z/x: lateral  r/f: speed up/down\n"
                "Shift+S: stop and save  Shift+X: save and exit"
            )
        else:
            print("Send goals in RViz. Space/0: disable motion and cancel goals.")
        print("q/Ctrl+C: stop and exit")
        tty.setcbreak(sys.stdin.fileno())
        velocity, last_key, scale = Twist(), 0.0, 0.1
        while rclpy.ok() and launch.poll() is None:
            rclpy.spin_once(node, timeout_sec=0)
            if recorder is not None and recorder.poll() is not None:
                raise RuntimeError(f"Recording exited; inspect {logs / 'record.log'}")
            if select.select([sys.stdin], [], [], 0.03)[0]:
                key = sys.stdin.read(1)
                velocity = Twist()
                if key in ("q", "\x03"):
                    break
                if key == "e":
                    try:
                        response = call(
                            node, SetBool, prefix + "/command_enable", SetBool.Request(data=True), 5
                        )
                        print(response.message)
                    except RuntimeError as error:
                        print(error, file=sys.stderr)
                elif args.mode == "navigation":
                    if key in (" ", "0"):
                        stop_navigation()
                    elif key in KEYS:
                        print("Navigation uses RViz goals; keyboard driving is in mapping mode.")
                elif key in ("S", "X"):
                    stop_motion()
                    if args.mode != "mapping":
                        print("Map saving is available in mapping mode")
                    elif map_command("save"):
                        print("Saved. Press e to enable movement again.")
                        if key == "X":
                            break
                elif key in ("r", "f"):
                    scale = max(0.03, min(0.2, scale + (0.02 if key == "r" else -0.02)))
                    print(f"Linear speed: {scale:.2f} m/s")
                elif key in KEYS:
                    x, y, yaw = KEYS[key]
                    velocity.linear.x, velocity.linear.y, velocity.angular.z = (
                        x * scale,
                        y * scale,
                        yaw * 0.35,
                    )
                last_key = time.monotonic()
            if time.monotonic() - last_key > 0.35:
                velocity = Twist()
            if args.mode == "mapping":
                publisher.publish(velocity)
        if launch.poll() is not None:
            raise RuntimeError(f"Launch exited; inspect {logs / 'launch.log'}")
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as error:
        status = 1
        print(error, file=sys.stderr)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        stop_motion()
        if viewer is not None and viewer.poll() is None:
            try:
                # Nav2 clients may still be using lifecycle/action services.
                # Disable the motor entry before closing the UI, then stop servers.
                call(node, SetBool, prefix + "/command_enable", SetBool.Request(data=False), 5)
            except RuntimeError as error:
                print(error, file=sys.stderr)
                if processes and processes[0].poll() is None:
                    map_command("stop")
            close(viewer)
        if processes and processes[0].poll() is None:
            if not map_command("stop"):
                status = 1
                print("Ordered stop failed; inspect Host state before restarting", file=sys.stderr)
        for process in reversed(processes):
            close(process, process_group=process is recorder)
        for stream in streams:
            stream.close()
        node.destroy_node()
        rclpy.try_shutdown()
    return status
