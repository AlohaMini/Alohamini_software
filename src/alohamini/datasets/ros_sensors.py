"""Read optional measured ROS sensor extensions alongside native episode frames."""

import hashlib
import json
from pathlib import Path


def iter_sensor_frames(dataset, episode_index):
    """Yield scan/IMU values and validity masks; does not change policy input dimensions."""
    root = Path(dataset).expanduser().resolve()
    directory = root / "ros_sensors" / f"episode_{episode_index:06d}"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported ROS sensor extension")
    safety = (root / manifest["source_safety"]).resolve()
    if (
        not safety.is_relative_to(root)
        or hashlib.sha256(safety.read_bytes()).hexdigest() != manifest["safety_sha256"]
    ):
        raise ValueError("Native episode changed after sensor alignment")
    if (
        hashlib.sha256((directory / "frames.jsonl").read_bytes()).hexdigest()
        != manifest["frames_sha256"]
    ):
        raise ValueError("ROS sensor extension checksum mismatch")
    with (directory / "frames.jsonl").open() as stream:
        count = 0
        for line in stream:
            row = json.loads(line)
            if row["episode_index"] != episode_index or row["frame_index"] != count:
                raise ValueError("Sensor frame indices do not match native episode")
            yield row
            count += 1
        if count != manifest["frames"]:
            raise ValueError("Truncated ROS sensor extension")
