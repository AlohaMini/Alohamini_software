"""Save grid/pose-graph revisions or record native ROS sensor topics."""

import argparse
import json
import math
import os
import sys
import time

from .maps import MapRevision


def call(node, service, name, request, timeout):
    import rclpy

    client = node.create_client(service, name)
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"Service unavailable: {name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
        if not future.done():
            future.cancel()
            raise TimeoutError(f"Service timed out: {name}; any partial artifacts stay incomplete")
        return future.result()
    finally:
        node.destroy_client(client)


def stop_and_wait(node, model, timeout, runtime_mode="hardware"):
    """Disable the Bridge first, then require new, measured stationary feedback."""
    import rclpy
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from std_msgs.msg import String
    from std_srvs.srv import SetBool

    if runtime_mode == "replay":
        return "replay"
    prefix = "/alohamini_lerobot_bridge" if runtime_mode == "hardware" else "/alohamini_nav_sim"
    response = call(
        node,
        SetBool,
        prefix + "/command_enable",
        SetBool.Request(data=False),
        timeout,
    )
    if not response.success:
        raise RuntimeError(response.message)
    # Subscribe after disable to avoid using an old idle sample as stop evidence.
    state = {"count": 0, "last_stamp": None, "host": None}

    def velocity(msg):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        age = (node.get_clock().now().nanoseconds - stamp) / 1e9
        v = msg.twist.twist
        values = (v.linear.x, v.linear.y, v.angular.z)
        valid = all(math.isfinite(x) for x in values) and -0.05 <= age <= 0.25
        valid &= state["last_stamp"] is None or stamp > state["last_stamp"]
        stopped = valid and math.hypot(v.linear.x, v.linear.y) < 0.01 and abs(v.angular.z) < 0.03
        state["count"] = state["count"] + 1 if stopped else 0
        state["last_stamp"] = stamp

    def host(msg):
        try:
            payload = json.loads(msg.data)
            if payload["_robot_metadata"]["robot_model"] == model:
                state["host"] = payload["_safety"]["host_session_id"]
        except (KeyError, TypeError, ValueError):
            pass

    subscriptions = [
        node.create_subscription(TwistWithCovarianceStamped, "/wheel/twist", velocity, 1),
        node.create_subscription(String, prefix + "/state_json", host, 1),
    ]
    try:
        deadline = time.monotonic() + timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["count"] >= 5 and state["host"]:
                return state["host"]
        raise TimeoutError(
            "No fresh stationary feedback for the requested model; map was not saved"
        )
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)


def save(args):
    import rclpy
    from nav2_msgs.srv import SaveMap
    from slam_toolbox.srv import SerializePoseGraph

    rclpy.init()
    from rclpy.parameter import Parameter

    node = rclpy.create_node(
        "alohamini_map_session",
        parameter_overrides=[Parameter("use_sim_time", value=args.runtime_mode != "hardware")],
    )
    revision = None
    try:
        # This precedes file I/O and potentially blocking service calls.
        session = stop_and_wait(node, args.robot_model, args.timeout, args.runtime_mode)
        revision = MapRevision(args.name, args.revision, model=args.robot_model)
        response = call(
            node,
            SaveMap,
            "/map_saver/save_map",
            SaveMap.Request(
                map_topic="/map",
                map_url=str(revision.staging / "map"),
                image_format="pgm",
                map_mode="trinary",
                free_thresh=0.25,
                occupied_thresh=0.65,
            ),
            args.timeout,
        )
        if not response.result:
            raise RuntimeError("Nav2 map saver failed")
        if not args.grid_only:
            response = call(
                node,
                SerializePoseGraph,
                "/slam_toolbox/serialize_map",
                SerializePoseGraph.Request(filename=str(revision.staging / "slam")),
                args.timeout,
            )
            if response.result != 0:
                raise RuntimeError(f"SLAM pose-graph save failed: {response.result}")
        # Snapshot nominal model configuration separately from live device calibration.
        import shutil

        shutil.copytree(revision.model.directory / "config", revision.staging / "model_config")
        result = revision.finish(
            pose_graph=not args.grid_only,
            metadata={
                "host_session_id": session,
                "map_frame": "map",
                "odom_frame": "odom",
                "base_frame": "base_link",
                "configuration": (
                    "model_config contains nominal assets; rosbag retains runtime observations"
                ),
                "saved_by_host": os.uname().nodename,
            },
        )
        print(result)
        print("ROS commands remain disabled. Enable again explicitly to continue driving.")
    except Exception as exc:
        if revision is not None and revision.staging.exists():
            revision.fail(exc)
            print(f"Incomplete artifacts: {revision.staging}", file=sys.stderr)
        raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def stop_runtime(args):
    """Stop commands, then clean lifecycle consumers while their TF providers still run."""
    import rclpy
    from nav2_msgs.srv import ManageLifecycleNodes

    rclpy.init()
    from rclpy.parameter import Parameter

    node = rclpy.create_node(
        "alohamini_navigation_stop",
        parameter_overrides=[Parameter("use_sim_time", value=args.runtime_mode != "hardware")],
    )
    try:
        stop_and_wait(node, args.robot_model, args.timeout, args.runtime_mode)
        for manager in (
            "lifecycle_manager_navigation",
            "lifecycle_manager_collision_monitor",
            "lifecycle_manager_localization",
            "lifecycle_manager_map_saver",
        ):
            name = f"/{manager}/manage_nodes"
            probe = node.create_client(ManageLifecycleNodes, name)
            try:
                present = probe.wait_for_service(timeout_sec=0.5)
            finally:
                node.destroy_client(probe)
            if not present:
                continue  # Mapping and navigation intentionally use different managers.
            response = call(
                node,
                ManageLifecycleNodes,
                name,
                ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.SHUTDOWN),
                args.timeout,
            )
            if not response.success:
                raise RuntimeError(
                    f"Lifecycle shutdown failed: {manager}; commands remain disabled"
                )
        print("Motion disabled; lifecycle consumers stopped. The launch process can now be closed.")
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main(argv=None):
    from .recording import CAMERAS, record

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    saver = commands.add_parser("save")
    saver.add_argument("name")
    saver.add_argument("--revision")
    saver.add_argument("--grid-only", action="store_true")
    saver.add_argument("--timeout", type=float, default=15.0)
    recorder = commands.add_parser("record")
    recorder.add_argument("name", help="New recording/episode name under workspace runs/")
    recorder.add_argument("--cameras", nargs="*", choices=CAMERAS, default=list(CAMERAS))
    recorder.add_argument("--duration", type=float)
    recorder.add_argument("--require-topics", nargs="*", default=[])
    recorder.add_argument("--dataset", help="Attach sensor samples to a completed native dataset")
    recorder.add_argument("--episode-index", type=int)
    attach = commands.add_parser("attach", help="Align a bag to one completed native episode")
    attach.add_argument("bag")
    attach.add_argument("dataset")
    attach.add_argument("--episode-index", type=int, required=True)
    stopper = commands.add_parser(
        "stop", help="Stop ROS control and lifecycle nodes before closing launch"
    )
    stopper.add_argument("--timeout", type=float, default=15.0)
    for command in (saver, recorder, stopper, attach):
        command.add_argument("--robot-model", default="alohamini2edu")
        command.add_argument(
            "--runtime-mode", choices=("hardware", "simulation", "replay"), default="hardware"
        )
    args = parser.parse_args(argv)
    if args.command == "record":
        if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
            parser.error("--duration must be finite and positive")
        if bool(args.dataset) != (args.episode_index is not None):
            parser.error("--dataset and --episode-index must be used together")
    if args.command in ("save", "stop") and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    try:
        if args.command == "attach":
            from pathlib import Path

            from .sensor_dataset import attach_episode

            print(
                attach_episode(
                    Path(args.bag), Path(args.dataset), args.episode_index, args.robot_model
                )
            )
        else:
            {"save": save, "record": record, "stop": stop_runtime}[args.command](args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"map_session: {exc}", file=sys.stderr)
        return 1
    return 0
