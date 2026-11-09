"""Record ROS topics with a completed manifest and optional native episode alignment."""

import json
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from alohamini.paths import WorkspacePaths

CAMERAS = ("forward", "backward", "chest", "wrist_left", "wrist_right")
TOPICS = [
    "/scan",
    "/scan_filtered",
    "/scan_sector_marker",
    "/imu/data_raw",
    "/imu/temperature",
    "/wheel/twist",
    "/odom",
    "/tf",
    "/tf_static",
    "/joint_states",
    "/alohamini/robot_description",
    "/cmd_vel",
    "/cmd_vel_safe",
    "/collision_monitor_state",
    "/diagnostics",
    "/alohamini_lerobot_bridge/state_json",
    "/alohamini/camera_clock",
    "/clock",
    "/map",
    "/plan",
    "/local_plan",
    "/global_costmap/costmap",
    "/local_costmap/costmap",
]


def recording_topics(cameras):
    if any(name not in CAMERAS for name in cameras):
        raise ValueError("Unknown camera name")
    return TOPICS + [
        f"/alohamini/cameras/{name}/{suffix}"
        for name in cameras
        for suffix in ("image_raw/compressed", "camera_info")
    ]


def record(args):
    directory = WorkspacePaths().run(args.name)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(
        schema_version=2,
        episode_id=args.name,
        robot_model=args.robot_model,
        started_at=datetime.now(timezone.utc).isoformat(),
        status="recording",
        topics=recording_topics(args.cameras),
        format="rosbag2",
        runtime_mode=args.runtime_mode,
        dataset=args.dataset,
        episode_index=args.episode_index,
        training_dataset=False,
    )
    path = directory / "manifest.json"

    def write():
        temporary = directory / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)

    write()
    command = [
        "ros2",
        "bag",
        "record",
        "-s",
        "sqlite3",
        "-o",
        str(directory / "bag"),
        "--max-cache-size",
        "16777216",
    ]
    if args.runtime_mode != "hardware":
        command.append("--use-sim-time")
    command.extend(manifest["topics"])
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        deadline = time.monotonic() + args.duration if args.duration else float("inf")
        try:
            while process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("Recorder did not finalize before timeout") from None
        if process.returncode not in (0, -signal.SIGINT):
            raise RuntimeError(f"Recorder exited with status {process.returncode}")
        metadata = yaml.safe_load((directory / "bag/metadata.yaml").read_text())[
            "rosbag2_bagfile_information"
        ]
        counts = {
            item["topic_metadata"]["name"]: item["message_count"]
            for item in metadata["topics_with_message_count"]
        }
        if not sum(counts.values()):
            raise RuntimeError("No ROS messages were recorded")
        missing = [topic for topic in args.require_topics if not counts.get(topic)]
        manifest.update(
            message_counts=counts,
            missing_topics=[t for t in manifest["topics"] if not counts.get(t)],
        )
        if missing:
            raise RuntimeError(f"Required topics missing: {missing}")
        manifest.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat())
        write()
        if args.dataset:
            from .sensor_dataset import attach_episode

            manifest["sensor_dataset"] = str(
                attach_episode(
                    directory / "bag", Path(args.dataset), args.episode_index, args.robot_model
                )
            )
            write()
        print(directory)
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc))
        write()
        raise
