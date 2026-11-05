"""Load model-specific mounts without depending on a ROS executor."""

import math
from pathlib import Path

import yaml

from alohamini.model import get_robot_model


def sensor_profile(model_id, path=""):
    path = (
        Path(path).expanduser()
        if path
        else get_robot_model(model_id).asset_path("config/sensors.yaml")
    )
    with path.open() as stream:
        data = yaml.safe_load(stream)
    if (
        not isinstance(data, dict)
        or data.get("schema_version") != 1
        or data.get("robot_model") != model_id
    ):
        raise ValueError("Sensor profile model/schema mismatch")
    frames = set()
    for name in ("lidar", "imu"):
        sensor = data[name]
        for key in ("xyz", "rpy"):
            vector = sensor[key]
            if (
                not isinstance(vector, list)
                or len(vector) != 3
                or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                    for x in vector
                )
            ):
                raise ValueError(f"Invalid {name}.{key}")
        parent, child = sensor["parent_frame"], sensor["frame_id"]
        if (
            parent != "base_link"
            or not isinstance(child, str)
            or not child
            or child == parent
            or child in frames
        ):
            raise ValueError("Sensors require unique child frames of base_link")
        frames.add(child)
        if type(sensor["serial_baudrate"]) is not int or sensor["serial_baudrate"] <= 0:
            raise ValueError("Invalid sensor baud rate")
    return data
