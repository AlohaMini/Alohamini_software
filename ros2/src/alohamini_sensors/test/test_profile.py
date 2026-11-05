import json
import math
import statistics
from pathlib import Path

import pytest
import yaml
from alohamini_sensors.profile import sensor_profile


def test_recorded_front_wall_keeps_its_direction_in_pro_base_frame():
    profile = sensor_profile("alohamini2edu")
    capture = json.loads((Path(__file__).parent / "fixtures/c1_front_wall.json").read_text())
    yaw = profile["lidar"]["rpy"][2]
    lower, upper = (profile["scan_filter"][k] for k in ("min_angle", "max_angle"))
    points = []
    for index, distance in capture["wall_samples"]:
        angle = capture["angle_min"] + index * capture["angle_increment"]
        assert angle >= lower or angle <= upper
        points.append((distance * math.cos(angle + yaw), distance * math.sin(angle + yaw)))
    mx, my = (statistics.mean(p[i] for p in points) for i in (0, 1))
    xx = statistics.mean((x - mx) ** 2 for x, y in points)
    yy = statistics.mean((y - my) ** 2 for x, y in points)
    xy = statistics.mean((x - mx) * (y - my) for x, y in points)
    normal = 0.5 * math.atan2(2 * xy, xx - yy) + math.pi / 2
    if mx * math.cos(normal) + my * math.sin(normal) < 0:
        normal += math.pi
    assert abs(math.atan2(math.sin(normal), math.cos(normal))) < math.radians(5)
    assert mx > 0


def test_wrong_model_or_nonfinite_mount_rejected(tmp_path):
    data = sensor_profile("alohamini2edu")
    data["imu"]["rpy"][0] = math.nan
    path = tmp_path / "sensors.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="imu.rpy"):
        sensor_profile("alohamini2edu", str(path))
    with pytest.raises(ValueError, match="model/schema"):
        sensor_profile("alohamini2pro", str(path))
