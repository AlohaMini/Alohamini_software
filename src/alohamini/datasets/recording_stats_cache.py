"""Atomic exact-statistics cache bound to committed recording metadata and files.

The caller owns the recording lock. File signatures detect ordinary changes and
replacement; they are not cryptographic verification of video or table contents.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import stat
import tempfile
import zipfile
import zlib
from pathlib import Path

import numpy as np

_CACHE = "meta/recording_stats.npz"
_METADATA = ("meta/info.json", "meta/alohamini.json", "meta/stats.json")
_MAX_MANIFEST_BYTES = 16 * 1024**2
_MAX_ARRAYS = 100_000


def _reject_constant(value):
    raise ValueError(f"Nonfinite recording cache JSON value: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate recording cache JSON key: {key}")
        result[key] = value
    return result


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _decode_json(data):
    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def _regular(root, relative):
    path = root
    for part in Path(relative).parts:
        path /= part
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Recording cache source is a symbolic link: {relative}")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode):
        raise ValueError(f"Recording cache source is not a regular file: {relative}")
    return path, details


def _capture_binding(root):
    hashes, metadata = {}, {}
    for relative in _METADATA:
        path, details = _regular(root, relative)
        if details.st_size > _MAX_MANIFEST_BYTES:
            raise ValueError(f"Recording metadata is too large: {relative}")
        data = path.read_bytes()
        if len(data) > _MAX_MANIFEST_BYTES:
            raise ValueError(f"Recording metadata is too large: {relative}")
        hashes[relative] = hashlib.sha256(data).hexdigest()
        metadata[relative] = _decode_json(data)
    files = []
    sources = (("data", ".parquet"), ("meta/episodes", ".parquet"), ("videos", ".mp4"))
    for directory, suffix in sources:
        base = root / directory
        if base.is_symlink():
            raise ValueError(f"Recording cache source directory is a symbolic link: {directory}")
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Recording cache source contains a symbolic link: {path}")
            if path.suffix != suffix or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            _, details = _regular(root, relative)
            files.append(
                {
                    "path": relative,
                    "size": details.st_size,
                    "mtime_ns": details.st_mtime_ns,
                    "ctime_ns": details.st_ctime_ns,
                }
            )
    files.sort(key=lambda value: value["path"])
    return {"metadata_sha256": hashes, "files": files}, metadata


def recording_statistics_binding(root) -> dict:
    """Fingerprint small committed metadata and stat formal files without reading their data."""
    return _capture_binding(Path(root).expanduser().resolve())[0]


def _committed(metadata):
    info = metadata["meta/info.json"]
    custom = metadata["meta/alohamini.json"]
    if not isinstance(info, dict) or not isinstance(custom, dict):
        raise ValueError("Invalid committed recording metadata")
    source = custom.get("source_info")
    if not isinstance(source, dict):
        raise ValueError("Invalid recording source metadata")
    frames, episodes, fps = (info.get(key) for key in ("total_frames", "total_episodes", "fps"))
    if (
        info.get("codebase_version") != "v3.0"
        or type(frames) is not int
        or not 1 <= frames <= np.iinfo(np.int64).max
        or type(episodes) is not int
        or episodes < 1
        or type(fps) not in (int, float)
        or not math.isfinite(fps)
        or fps <= 0
        or fps != source["fps"]
        or not isinstance(info.get("features"), dict)
        or not info["features"]
    ):
        raise ValueError("Recording cache requires consistent committed v3 metadata")
    for key, feature in info["features"].items():
        if (
            not isinstance(key, str)
            or not isinstance(feature, dict)
            or not isinstance(feature.get("dtype"), str)
        ):
            raise ValueError("Invalid recording feature contract")
        if feature["dtype"] not in ("image", "video"):
            _width(feature)
    return info


def _numeric(features):
    return {
        key: feature
        for key, feature in features.items()
        if feature["dtype"] not in ("image", "video")
    }


def _width(feature):
    shape = feature["shape"]
    if not isinstance(shape, list) or not shape or any(type(v) is not int or v < 1 for v in shape):
        raise ValueError("Invalid recording feature shape")
    width = math.prod(shape)
    if width > (_MAX_ARRAYS - 5) // 2:
        raise ValueError("Recording feature has too many cached coordinates")
    return width


def _archive_limits(features, frames):
    # Each coordinate can have at most one unique value/count pair per frame.
    # Bounds follow the committed dataset size, rather than imposing a fixed
    # cache-size ceiling on otherwise valid long recordings.
    widths = [_width(feature) for feature in _numeric(features).values()]
    arrays = 1 + sum(5 + 2 * width for width in widths)
    if arrays > _MAX_ARRAYS:
        raise ValueError("Recording statistics cache has too many arrays")
    data_bytes = sum(8 + 32 * width + 16 * frames * width for width in widths)
    return arrays, data_bytes + _MAX_MANIFEST_BYTES + arrays * 65536


def _validate_arrays(state, frames, width):
    keys = {"count", "mean", "m2", "min", "max"} | {
        f"{name}_{index}" for index in range(width) for name in ("values", "counts")
    }
    if set(state) != keys:
        raise ValueError("Unexpected recording accumulator arrays")
    for name, value in state.items():
        dtype = np.int64 if name == "count" or name.startswith("counts_") else np.float64
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise ValueError(f"Invalid recording accumulator array dtype: {name}")
    if state["count"].shape != () or state["count"].item() != frames:
        raise ValueError("Recording accumulator count differs from its commit")
    if any(state[name].shape != (width,) for name in ("mean", "m2", "min", "max")):
        raise ValueError("Recording accumulator dimensions differ from its feature")
    for index in range(width):
        values, counts = state[f"values_{index}"], state[f"counts_{index}"]
        if values.ndim != 1 or not 1 <= len(values) <= frames or counts.shape != values.shape:
            raise ValueError("Invalid recording accumulator distribution shape")


def _image_statistics(payload, features, frames):
    from alohamini.datasets.image_statistics import ImageStatistics

    cameras = {key for key, feature in features.items() if feature["dtype"] in ("image", "video")}
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
        or not isinstance(payload.get("sources"), list)
        or any(not isinstance(value, str) for value in payload["sources"])
        or not isinstance(payload.get("cameras"), dict)
        or set(payload["cameras"]) != cameras
    ):
        raise ValueError("Invalid recording image statistics cache")
    for key, values in payload["cameras"].items():
        if not isinstance(values, dict):
            raise ValueError(f"Invalid recording image statistics camera: {key}")
        count, histogram = values["frames"], values["histogram"]
        if type(count) is not int or not 1 <= count <= frames:
            raise ValueError(f"Invalid recording sampled frame count: {key}")
        if (
            not isinstance(histogram, list)
            or len(histogram) != 3
            or any(not isinstance(channel, list) or len(channel) != 256 for channel in histogram)
            or any(type(v) is not int or v < 0 for channel in histogram for v in channel)
        ):
            raise ValueError(f"Invalid recording RGB histogram: {key}")
        totals = [sum(channel) for channel in histogram]
        if min(totals) < count or len(set(totals)) != 1 or sum(totals) > np.iinfo(np.int64).max:
            raise ValueError(f"Invalid recording RGB histogram counts: {key}")
    return ImageStatistics(payload)


def _check_published_counts(metadata, features, frames, images):
    published = metadata["meta/stats.json"]
    if not isinstance(published, dict) or set(published) != set(features):
        raise ValueError("Published recording statistics differ from the feature contract")
    for key in features:
        expected = images.frames[key] if key in images.frames else frames
        if not isinstance(published[key], dict) or published[key].get("count") != [expected]:
            raise ValueError(f"Published recording statistics count differs: {key}")


def _check_history(previous, current):
    if previous is None:
        return
    files = {value["path"]: value for value in current["files"]}
    for value in previous["files"]:
        if files.get(value["path"]) != value:
            raise ValueError(f"Previously accumulated recording file changed: {value['path']}")


def save_recording_statistics(root, stats) -> None:
    """Atomically cache exact accumulators after a successful episode publication."""
    root = Path(root).expanduser().resolve()
    binding, metadata = _capture_binding(root)
    info = _committed(metadata)
    features = _decode_json(_json(stats.features))
    if type(stats.frames) is not int or stats.frames != info["total_frames"]:
        raise ValueError("Recording statistics frame count differs from its commit")
    if features != info["features"]:
        raise ValueError("Recording statistics features differ from its commit")
    _check_history(getattr(stats, "_recording_source_binding", None), binding)
    images = _image_statistics(stats.image_statistics.payload(), features, stats.frames)
    _check_published_counts(metadata, features, stats.frames, images)
    numeric = _numeric(features)
    if set(stats.trackers) != set(features):
        raise ValueError("Recording statistics trackers differ from features")
    arrays, indices = {}, {}
    for index, (key, feature) in enumerate(numeric.items()):
        state = stats.trackers[key].export_state()
        _validate_arrays(state, stats.frames, _width(feature))
        indices[key] = index
        arrays.update({f"tracker_{index}__{name}": value for name, value in state.items()})
    manifest = {
        "version": 1,
        "frames": stats.frames,
        "fps": info["fps"],
        "features": features,
        "trackers": indices,
        "image_stats": images.payload(),
        "binding": binding,
    }
    manifest_bytes = _json(manifest)
    if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
        raise ValueError("Recording statistics manifest exceeds its size limit")
    arrays["manifest"] = np.frombuffer(manifest_bytes, dtype=np.uint8)
    array_count, byte_limit = _archive_limits(features, stats.frames)
    if len(arrays) != array_count or sum(value.nbytes for value in arrays.values()) > byte_limit:
        raise ValueError("Recording statistics cache exceeds its size limit")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=".recording_stats-", suffix=".npz", dir=root / "meta", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        if recording_statistics_binding(root) != binding:
            raise ValueError("Recording changed while saving its statistics cache")
        os.replace(temporary, root / _CACHE)
        directory = os.open(root / "meta", os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        stats._recording_source_binding = binding
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_recording_statistics(root):
    """Restore validated cached accumulators, or return None for a legacy/stale cache."""
    root = Path(root).expanduser().resolve()
    path = root / _CACHE
    try:
        if not path.exists():
            return None
        _regular(root, _CACHE)
        binding, metadata = _capture_binding(root)
        info = _committed(metadata)
        array_count, byte_limit = _archive_limits(info["features"], info["total_frames"])
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if (
                len(entries) != array_count
                or len(set(names)) != len(names)
                or any("/" in name or not name.endswith(".npy") for name in names)
                or sum(entry.file_size for entry in entries) > byte_limit
                or any(
                    entry.filename == "manifest.npy"
                    and entry.file_size > _MAX_MANIFEST_BYTES + 65536
                    for entry in entries
                )
            ):
                raise ValueError("Invalid recording statistics archive size or entries")
        with np.load(path, allow_pickle=False, max_header_size=65536) as archive:
            encoded = archive["manifest"]
            if encoded.dtype != np.uint8 or encoded.ndim != 1 or encoded.size > _MAX_MANIFEST_BYTES:
                raise ValueError("Invalid recording statistics manifest array")
            manifest = _decode_json(encoded.tobytes())
            if (
                not isinstance(manifest, dict)
                or set(manifest)
                != {"version", "frames", "fps", "features", "trackers", "image_stats", "binding"}
                or type(manifest["version"]) is not int
                or manifest["version"] != 1
                or type(manifest["frames"]) is not int
                or manifest["frames"] != info["total_frames"]
                or type(manifest["fps"]) not in (int, float)
                or manifest["fps"] != info["fps"]
                or manifest["features"] != info["features"]
                or manifest["binding"] != binding
            ):
                raise ValueError("Recording statistics cache does not match the current commit")
            features, frames = manifest["features"], manifest["frames"]
            numeric, indices = _numeric(features), manifest["trackers"]
            if (
                not isinstance(indices, dict)
                or set(indices) != set(numeric)
                or any(type(value) is not int for value in indices.values())
                or set(indices.values()) != set(range(len(numeric)))
            ):
                raise ValueError("Invalid recording accumulator index")
            expected, restored = {"manifest"}, {}
            from alohamini.datasets.exact_distribution import ExactDistributionStats

            for key, feature in numeric.items():
                width, prefix = _width(feature), f"tracker_{indices[key]}__"
                keys = {"count", "mean", "m2", "min", "max"} | {
                    f"{name}_{index}" for index in range(width) for name in ("values", "counts")
                }
                expected.update(prefix + name for name in keys)
                state = {name: archive[prefix + name] for name in keys}
                _validate_arrays(state, frames, width)
                tracker = ExactDistributionStats.from_state(state)
                if tracker._count != frames or len(tracker._mean) != width:
                    raise ValueError(f"Cached recording accumulator count differs: {key}")
                restored[key] = tracker
            if set(archive.files) != expected:
                raise ValueError("Unexpected recording statistics cache arrays")
            images = _image_statistics(manifest["image_stats"], features, frames)
            _check_published_counts(metadata, features, frames, images)
        if recording_statistics_binding(root) != binding:
            raise ValueError("Recording changed while loading its statistics cache")
        from alohamini.datasets.lerobotv3 import _Stats

        result = _Stats(features, recording=True)
        result.trackers.update(restored)
        result.frames, result.image_statistics = frames, images
        result._recording_source_binding = binding
        return result
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        EOFError,
        zipfile.BadZipFile,
        zlib.error,
    ) as exc:
        logging.warning("Recording statistics cache ignored: %s", str(exc)[:256])
        return None
