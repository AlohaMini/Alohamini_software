"""Save grid/pose-graph revisions or record native ROS sensor topics."""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

from alohamini.paths import WorkspacePaths

from .maps import MapRevision

RECORD_TOPICS = [
    "/scan",
    "/scan_filtered",
    "/imu/data_raw",
    "/imu/temperature",
    "/wheel/twist",
    "/odom",
    "/tf",
    "/tf_static",
    "/cmd_vel",
    "/cmd_vel_safe",
    "/alohamini_lerobot_bridge/state_json",
    "/diagnostics",
]


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


def stop_and_wait(node, model, timeout):
    """Disable the Bridge first, then require new, measured stationary feedback."""
    import rclpy
    from geometry_msgs.msg import TwistWithCovarianceStamped
    from std_msgs.msg import String
    from std_srvs.srv import SetBool

    response = call(
        node,
        SetBool,
        "/alohamini_lerobot_bridge/command_enable",
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
        node.create_subscription(String, "/alohamini_lerobot_bridge/state_json", host, 1),
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
    node = rclpy.create_node("alohamini_map_session")
    revision = None
    try:
        # This precedes file I/O and potentially blocking service calls.
        session = stop_and_wait(node, args.robot_model, args.timeout)
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


def record(args):
    directory = WorkspacePaths().run(args.name)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                episode_id=args.name,
                robot_model=args.robot_model,
                started_at=datetime.now(timezone.utc).isoformat(),
                topics=RECORD_TOPICS,
                format="rosbag2",
                training_dataset=False,
            ),
            indent=2,
        )
        + "\n"
    )
    os.execvp("ros2", ["ros2", "bag", "record", "-o", str(directory / "bag"), *RECORD_TOPICS])


def stop_runtime(args):
    """Stop commands, then clean lifecycle consumers while their TF providers still run."""
    import rclpy
    from nav2_msgs.srv import ManageLifecycleNodes

    rclpy.init()
    node = rclpy.create_node("alohamini_navigation_stop")
    try:
        stop_and_wait(node, args.robot_model, args.timeout)
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
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    saver = commands.add_parser("save")
    saver.add_argument("name")
    saver.add_argument("--revision")
    saver.add_argument("--grid-only", action="store_true")
    saver.add_argument("--timeout", type=float, default=15.0)
    recorder = commands.add_parser("record")
    recorder.add_argument("name", help="New recording/episode name under workspace runs/")
    stopper = commands.add_parser(
        "stop", help="Stop ROS control and lifecycle nodes before closing launch"
    )
    stopper.add_argument("--timeout", type=float, default=15.0)
    for command in (saver, recorder, stopper):
        command.add_argument("--robot-model", default="alohamini2edu")
    args = parser.parse_args(argv)
    if args.command in ("save", "stop") and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be finite and positive")
    try:
        {"save": save, "record": record, "stop": stop_runtime}[args.command](args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"map_session: {exc}", file=sys.stderr)
        return 1
    return 0
