"""Read a committed recording index without decoding or scanning historical frames."""

from __future__ import annotations

import json
import math
import re
import stat
from collections import defaultdict
from pathlib import Path


def _path(root, relative, *, required=False, directory=False):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Recording path escapes its root")
    path = root
    for offset, part in enumerate(relative.parts):
        path /= part
        try:
            mode = path.lstat()
        except FileNotFoundError:
            if required:
                raise ValueError(f"Missing recording file or directory: {path}") from None
            continue
        if stat.S_ISLNK(mode.st_mode):
            raise ValueError(f"Recording path contains a symbolic link: {path}")
        if offset < len(relative.parts) - 1 or directory:
            if not stat.S_ISDIR(mode.st_mode):
                raise ValueError(f"Expected recording directory: {path}")
        elif not stat.S_ISREG(mode.st_mode) or (required and mode.st_size == 0):
            raise ValueError(f"Expected a nonempty recording file: {path}")
    return path


def _integer(value, label, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(
            f"Recording {label} must be a {'positive' if positive else 'nonnegative'} integer"
        )
    return value


def _coordinate(row, prefix):
    return {
        name: _integer(row[f"{prefix}/{name}"], f"{prefix}/{name}")
        for name in ("chunk_index", "file_index")
    }


def _index_files(root):
    directory = _path(root, "meta/episodes", required=True, directory=True)
    chunks = []
    for chunk in directory.glob("chunk-*"):
        match = re.fullmatch(r"chunk-(\d{3,})", chunk.name)
        if match is None or chunk.name != f"chunk-{int(match[1]):03d}":
            raise ValueError(f"Invalid recording index directory: {chunk}")
        _path(root, chunk.relative_to(root), required=True, directory=True)
        chunks.append((int(match[1]), chunk))
    for chunk_number, chunk in sorted(chunks):
        files = []
        for path in chunk.glob("*.parquet"):
            match = re.fullmatch(r"file-(\d{3,})\.parquet", path.name)
            if match is None or path.name != f"file-{int(match[1]):03d}.parquet":
                raise ValueError(f"Invalid recording index filename: {path}")
            files.append((int(match[1]), path))
        for file_number, path in sorted(files):
            yield (chunk_number, file_number), _path(root, path.relative_to(root), required=True)


def read_recording_index(root, source_info) -> tuple[dict, dict]:
    """Validate resume counters, references and schemas from indexes and footers.

    This is a startup check, not a full integrity scan: no numeric frame rows,
    safety-log contents, video contents or image statistics are read. The caller
    compares source_info to meta/alohamini.json and owns the recording lock.
    An explicitly selected root may itself be a symlink; internal paths may not.
    """
    import pyarrow.parquet as pq

    from alohamini.datasets.images import MAX_IMAGE_PIXELS, VIDEO_FORMAT
    from alohamini.datasets.lerobotv3 import DATA_PATH, VIDEO_PATH, _arrow_schema, _features
    from alohamini.datasets.record import StateSelection

    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Missing recording directory: {root}")
    for relative, pattern in (("meta/recovery", "*.json"), (".recording/episodes", "*")):
        directory = _path(root, relative, directory=True)
        if any(directory.glob(pattern)):
            raise ValueError("Interrupted recording retained; recover before resuming")
    info = json.loads(_path(root, "meta/info.json", required=True).read_text())
    if not isinstance(info, dict) or info.get("codebase_version") != "v3.0":
        raise ValueError("Recording resume requires LeRobot v3 metadata")
    count = _integer(info.get("total_episodes"), "total_episodes", positive=True)
    frames = _integer(info.get("total_frames"), "total_frames", positive=True)
    fps = info.get("fps")
    try:
        valid_fps = type(fps) in (int, float) and math.isfinite(fps) and fps > 0
    except OverflowError:
        valid_fps = False
    if not valid_fps:
        raise ValueError("Recording fps must be a finite positive number")
    if (
        fps != source_info["fps"]
        or info.get("robot_type") != source_info["robot_metadata"]["robot_model"]
        or source_info.get("image_format") != VIDEO_FORMAT
        or info.get("data_path") != DATA_PATH
        or info.get("video_path") != VIDEO_PATH
        or type(info.get("total_tasks")) is not int
        or info["total_tasks"] != 1
    ):
        raise ValueError("Recording schema, fps, task or storage layout changed")
    cameras = source_info["robot_metadata"].get("cameras", [])
    if (
        not isinstance(cameras, list)
        or any(
            not isinstance(camera, str) or re.fullmatch(r"[A-Za-z0-9_]+", camera) is None
            for camera in cameras
        )
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError("Invalid recording camera names")
    features = info.get("features")
    if not isinstance(features, dict):
        raise ValueError("Recording features must be an object")
    shapes = {}
    for camera in cameras:
        feature = features.get(f"observation.images.{camera}", {})
        shape = feature.get("shape") if isinstance(feature, dict) else None
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            or shape[2] != 3
            or shape[0] * shape[1] > MAX_IMAGE_PIXELS
        ):
            raise ValueError(f"Invalid recording image shape: {camera}")
        shapes[camera] = shape
    expected = _features(source_info, StateSelection(source_info), shapes)
    if features != expected:
        raise ValueError("Recording features differ from the source recording contract")
    schema = _arrow_schema(expected)
    # Stat the other required metadata, but leave its contents to explicit checks
    # and the deferred exact statistics reconstruction on the first save.
    for relative in ("meta/alohamini.json", "meta/stats.json", "meta/tasks.parquet"):
        _path(root, relative, required=True)
    columns = [
        "episode_index",
        "length",
        "tasks",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
        "meta/episodes/chunk_index",
        "meta/episodes/file_index",
        *(
            f"videos/observation.images.{camera}/{field}"
            for camera in cameras
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        ),
    ]
    expected_episode = offset = 0
    data_lengths = defaultdict(int)
    video_ends = {}
    for location, path in _index_files(root):
        with pq.ParquetFile(path) as index:
            if any(index.schema_arrow.names.count(name) != 1 for name in columns):
                raise ValueError(f"Recording index is missing required columns: {path}")
            if not 0 < index.metadata.num_rows <= count - expected_episode:
                raise ValueError("Recording index episode count disagrees with metadata")
            rows = index.read(columns=columns).to_pylist()
        for row in rows:
            episode = _integer(row["episode_index"], "episode_index")
            length = _integer(row["length"], "episode length", positive=True)
            start = _integer(row["dataset_from_index"], "dataset_from_index")
            stop = _integer(row["dataset_to_index"], "dataset_to_index")
            if episode != expected_episode or start != offset or stop != offset + length:
                raise ValueError("Recording episode indices or frame ranges are not contiguous")
            if row["tasks"] != [source_info["task"]]:
                raise ValueError("Recording episode task differs from the requested task")
            metadata_location = _coordinate(row, "meta/episodes")
            if tuple(metadata_location.values()) != location:
                raise ValueError("Recording episode index path disagrees with its metadata")
            relative = DATA_PATH.format(**_coordinate(row, "data"))
            data_lengths[relative] += length
            _path(root, f"meta/safety/episode_{episode:06d}.jsonl", required=True)
            for camera in cameras:
                key = f"observation.images.{camera}"
                relative = VIDEO_PATH.format(video_key=key, **_coordinate(row, f"videos/{key}"))
                _path(root, relative, required=True)
                first, last = (
                    row[f"videos/{key}/{field}"] for field in ("from_timestamp", "to_timestamp")
                )
                try:
                    valid = all(
                        type(value) in (int, float) and math.isfinite(value)
                        for value in (first, last)
                    )
                except OverflowError:
                    valid = False
                if (
                    not valid
                    or first < 0
                    or last <= first
                    or not math.isclose(last - first, length / fps, rel_tol=0, abs_tol=1e-4)
                    or not math.isclose(first, video_ends.get(relative, 0), rel_tol=0, abs_tol=1e-4)
                ):
                    raise ValueError(
                        f"Recording video interval disagrees with episode length: {camera}"
                    )
                video_ends[relative] = last
            expected_episode += 1
            offset = stop
    if expected_episode != count or offset != frames:
        raise ValueError("Recording total episode or frame count disagrees with its index")
    for relative, length in data_lengths.items():
        with pq.ParquetFile(_path(root, relative, required=True)) as data:
            if data.metadata.num_rows != length:
                raise ValueError(
                    f"Recording data footer row count disagrees with its index: {relative}"
                )
            if not data.schema_arrow.equals(schema, check_metadata=False):
                raise ValueError(
                    f"Recording data schema differs from its declared features: {relative}"
                )
    # Inspect only paths that the next publish may write, including atomic JSON
    # temporaries. Unreferenced files and unrelated directories are not traversed.
    chunk, file = divmod(count, 1000)
    future = [
        DATA_PATH.format(chunk_index=chunk, file_index=file),
        f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet",
        f"meta/safety/episode_{count:06d}.jsonl",
        *(
            VIDEO_PATH.format(
                video_key=f"observation.images.{camera}", chunk_index=chunk, file_index=file
            )
            for camera in cameras
        ),
        "recording.lock",
        ".recording/recording.lock",
        ".recording/meta/info.json",
    ]
    for name in ("info.json", "stats.json", "tasks.parquet", "alohamini.json"):
        future.append(f"meta/{name}")
        if name.endswith(".json"):
            future.append(f"meta/{name}.tmp")
    for relative in future:
        _path(root, relative)
    return info, shapes
