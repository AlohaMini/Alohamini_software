"""Attach measured ROS samples to native frames using session-scoped clock anchors."""

import bisect
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path


def stamp_ns(message):
    return message.header.stamp.sec * 10**9 + message.header.stamp.nanosec


def bag_messages(path):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {entry.name: entry.type for entry in reader.get_all_topics_and_types()}
    selected = {
        "/scan",
        "/imu/data_raw",
        "/alohamini/camera_clock",
        "/alohamini_lerobot_bridge/state_json",
    }
    reader.set_filter(rosbag2_py.StorageFilter(topics=sorted(selected & types.keys())))
    classes = {topic: get_message(types[topic]) for topic in selected & types.keys()}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        yield topic, deserialize_message(data, classes[topic])


def payload(message, kind):
    def vector(value):
        return [float(value.x), float(value.y), float(value.z)]

    result = dict(stamp_ns=stamp_ns(message), frame_id=message.header.frame_id)
    if kind == "scan":
        result.update(
            angle_min=message.angle_min,
            angle_increment=message.angle_increment,
            time_increment=message.time_increment,
            scan_time=message.scan_time,
            range_min=message.range_min,
            range_max=message.range_max,
            ranges=[
                float(x)
                if math.isfinite(x) and message.range_min <= x <= message.range_max
                else None
                for x in message.ranges
            ],
            intensities=[float(x) if math.isfinite(x) else None for x in message.intensities],
        )
    else:
        q = message.orientation
        result.update(
            angular_velocity=vector(message.angular_velocity),
            linear_acceleration=vector(message.linear_acceleration),
            orientation_xyzw=[q.x, q.y, q.z, q.w]
            if message.orientation_covariance[0] != -1
            else None,
            orientation_covariance=list(message.orientation_covariance),
            angular_velocity_covariance=list(message.angular_velocity_covariance),
            linear_acceleration_covariance=list(message.linear_acceleration_covariance),
        )
    json.dumps(result, allow_nan=False)
    return result


def collect(messages):
    anchors, samples = defaultdict(list), {"scan": [], "imu": []}
    for topic, message in messages:
        if topic in ("/scan", "/imu/data_raw"):
            kind = "scan" if topic == "/scan" else "imu"
            if message.header.frame_id and stamp_ns(message) > 0:
                samples[kind].append(payload(message, kind))
        else:
            data = json.loads(message.data)
            if topic == "/alohamini/camera_clock":
                session, mono, unix = (
                    data[k] for k in ("host_session_id", "capture_monotonic_s", "capture_unix_ns")
                )
            else:
                timing = data["_host_timing"]
                session = data["_safety"]["host_session_id"]
                mono, unix = (
                    timing["state_sample_finished_monotonic_s"],
                    timing["state_sample_unix_ns"],
                )
            if session and math.isfinite(mono) and mono >= 0 and type(unix) is int and unix > 0:
                anchors[session].append((mono, unix))
    for items in anchors.values():
        items.sort()
    for items in samples.values():
        items.sort(key=lambda item: item["stamp_ns"])
    return anchors, samples


def align_frame(record, anchors, samples, max_scan_age=0.15, max_imu_age=0.05, indexes=None):
    session = record["safety"]["host_session_id"]
    mono = record["host_timing"]["state_sample_monotonic_s"]
    if not isinstance(mono, (int, float)) or not math.isfinite(mono):
        raise ValueError("Native frame has no measured Host timestamp")
    clocks = anchors.get(session, [])
    if not clocks:
        raise ValueError(f"No clock anchor for Host session {session}")
    clock_times = indexes["anchors"][session] if indexes else [item[0] for item in clocks]
    at = bisect.bisect_left(clock_times, mono)
    nearest = min(clocks[max(0, at - 1) : at + 1], key=lambda item: abs(item[0] - mono))
    if abs(nearest[0] - mono) > 2:
        raise ValueError("Clock anchor is more than 2 seconds from native frame")
    nearby = clocks[
        bisect.bisect_left(clock_times, mono - 2) : bisect.bisect_right(clock_times, mono + 2)
    ]
    local = [unix - int(t * 1e9) for t, unix in nearby]
    if max(local) - min(local) > 20_000_000:
        raise ValueError("Host wall clock jumped; refusing ambiguous alignment")
    unix = nearest[1] + round((mono - nearest[0]) * 1e9)
    result = dict(
        episode_index=record["episode_index"],
        frame_index=record["frame_index"],
        host_session_id=session,
        sample_unix_ns=unix,
        sensors={},
    )
    for kind, age in (("scan", max_scan_age), ("imu", max_imu_age)):
        items = samples[kind]
        times = indexes[kind] if indexes else [x["stamp_ns"] for x in items]
        index = bisect.bisect_right(times, unix) - 1
        valid = index >= 0 and unix - items[index]["stamp_ns"] <= age * 1e9
        result["sensors"][kind] = dict(
            valid=valid,
            sample=items[index] if valid else None,
            age_ns=unix - items[index]["stamp_ns"] if valid else None,
        )
    return result


def attach_episode(bag, dataset, episode, model, *, messages=None):
    if type(episode) is not int or episode < 0:
        raise ValueError("episode-index must be a nonnegative integer")
    dataset = Path(dataset).expanduser().resolve()
    candidates = [
        dataset / f"meta/safety/episode_{episode:06d}.jsonl",
        dataset / f"episodes/episode_{episode:06d}/safety.jsonl",
    ]
    safety = next((p for p in candidates if p.is_file()), None)
    if safety is None:
        raise ValueError("Completed native episode safety.jsonl is required")
    expected_count = None
    if safety == candidates[0]:
        info = json.loads((dataset / "meta/info.json").read_text())
        if info.get("total_episodes", 0) <= episode:
            raise ValueError("Native episode has not been committed to dataset metadata")
    else:
        summary = json.loads((safety.parent / "episode.json").read_text())
        if summary.get("episode_index") != episode or summary.get("length", 0) <= 0:
            raise ValueError("Native episode completion metadata is invalid")
        expected_count = summary["length"]
    anchors, samples = collect(bag_messages(bag) if messages is None else messages)
    if not all(samples.values()):
        raise ValueError("Bag must contain both /scan and /imu/data_raw")
    indexes = {kind: [x["stamp_ns"] for x in items] for kind, items in samples.items()}
    indexes["anchors"] = {session: [x[0] for x in items] for session, items in anchors.items()}
    parent = dataset / "ros_sensors"
    parent.mkdir(exist_ok=True)
    target = parent / f"episode_{episode:06d}"
    lock = parent / f".{target.name}.lock"
    with lock.open("x"):
        try:
            if target.exists():
                raise FileExistsError(target)
            stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete-", dir=parent))
            count, valid = 0, {"scan": 0, "imu": 0}
            with (stage / "frames.jsonl").open("x") as stream, safety.open() as source:
                for line in source:
                    record = json.loads(line)
                    if record.get("frame_index") is None:
                        continue
                    if record["robot_metadata"]["robot_model"] != model:
                        raise ValueError("Dataset model does not match selected model")
                    if record["episode_index"] != episode or record["frame_index"] != count:
                        raise ValueError("Native episode frame indices are not contiguous")
                    row = align_frame(record, anchors, samples, indexes=indexes)
                    for kind in valid:
                        valid[kind] += int(row["sensors"][kind]["valid"])
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    count += 1
            if not count or not all(valid.values()):
                raise ValueError("No overlapping measured sensor data for native episode")
            if expected_count is not None and count != expected_count:
                raise ValueError("Native frame count differs from episode completion metadata")
            manifest = dict(
                schema_version=1,
                status="complete",
                robot_model=model,
                episode_index=episode,
                frames=count,
                valid_frames=valid,
                source_bag=str(Path(bag).resolve()),
                source_safety=str(safety.relative_to(dataset)),
                safety_sha256=hashlib.sha256(safety.read_bytes()).hexdigest(),
                alignment="causal samples; session clock anchors; synchronized wall clocks",
                frames_sha256=hashlib.sha256((stage / "frames.jsonl").read_bytes()).hexdigest(),
                units={
                    "ranges": "m",
                    "angles": "rad",
                    "angular_velocity": "rad/s",
                    "linear_acceleration": "m/s^2 including gravity",
                    "time": "ns",
                },
                null_ranges="invalid or out-of-range; never zero-filled",
            )
            (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            os.rename(stage, target)
            return target
        finally:
            lock.unlink()
