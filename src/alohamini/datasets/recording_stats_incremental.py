"""Immutable cache ancestry and exact distribution replay; never reads robot data."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import uuid
import zipfile
from pathlib import PurePosixPath

import numpy as np

from alohamini.datasets import recording_stats_cache as cache


def compact_binding(binding):
    """Keep ancestry manifests bounded instead of repeating all historical paths."""
    return {
        "metadata_sha256": binding["metadata_sha256"],
        "files_sha256": hashlib.sha256(cache._json(binding["files"])).hexdigest(),
    }


def _validate_compact_binding(binding):
    if (
        not isinstance(binding, dict)
        or set(binding) != {"metadata_sha256", "files_sha256"}
        or not isinstance(binding["metadata_sha256"], dict)
        or set(binding["metadata_sha256"]) != set(cache._METADATA)
    ):
        raise ValueError("Invalid recording cache compact source binding")
    for digest in [binding["files_sha256"], *binding["metadata_sha256"].values()]:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Invalid recording cache source fingerprint")


def _signature(details):
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def checkpoint(root, frames, features, fps):
    """Remember the exact latest file whose distributions have been acknowledged."""
    _, details = cache._regular(root, cache._CACHE)
    return dict(frames=frames, features=features, fps=fps, signature=_signature(details))


def append_checkpoint(root, stats, features, fps):
    previous = getattr(stats, "_recording_cache_checkpoint", None)
    if (
        not isinstance(previous, dict)
        or type(previous.get("frames")) is not int
        or not 1 <= previous["frames"] < stats.frames
        or previous.get("features") != features
        or previous.get("fps") != fps
    ):
        return None
    try:
        _, details = cache._regular(root, cache._CACHE)
    except (OSError, ValueError):
        return None
    return previous if _signature(details) == previous.get("signature") else None


def preserve_parent(root, previous):
    """Keep the old latest inode durable before a new latest can reference it."""
    relative = f"meta/recording_stats/{uuid.uuid4().hex}.npz"
    directory = root / "meta/recording_stats"
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError("Recording cache archive directory is a symbolic link")
    source, details = cache._regular(root, cache._CACHE)
    if _signature(details) != previous["signature"]:
        raise ValueError("Recording cache changed before its incremental save")
    copied = False
    try:
        os.link(source, root / relative, follow_symlinks=False)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK}:
            raise
        # Some external filesystems cannot hard-link. Copy only the previous
        # immutable segment, not the entire accumulated distribution again.
        shutil.copyfile(source, root / relative, follow_symlinks=False)
        with (root / relative).open("rb") as stream:
            os.fsync(stream.fileno())
        copied = True
    _, linked = cache._regular(root, relative)
    _, after = cache._regular(root, cache._CACHE)
    unchanged = (
        _signature(after) == previous["signature"] and linked.st_size == previous["signature"][2]
        if copied
        else _signature(linked)[:4] == previous["signature"][:4]
        and _signature(after) == _signature(linked)
    )
    # Creating a hard link changes ctime, while the saved content is unchanged.
    if not unchanged:
        raise ValueError("Recording cache changed while preserving its parent")
    for path in (directory, root / "meta"):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"path": relative, "frames": previous["frames"]}


def _parent(value, frames):
    if not isinstance(value, dict) or set(value) != {"path", "frames"}:
        raise ValueError("Invalid recording cache parent")
    relative, count = value["path"], value["frames"]
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Invalid recording cache parent path")
    path = PurePosixPath(relative)
    if (
        path.as_posix() != relative
        or len(path.parts) != 3
        or path.parts[:2] != ("meta", "recording_stats")
        or not path.name.endswith(".npz")
        or path.name in (".npz", "..npz")
        or type(count) is not int
        or not 1 <= count < frames
    ):
        raise ValueError("Invalid recording cache parent path or frame count")
    return relative, count


def validate_state(state, frames, segment_frames, width):
    """Validate cumulative moments and one segment's sorted integer distribution."""
    cache._validate_arrays(state, frames, width)
    for name in ("mean", "m2", "min", "max"):
        if not np.isfinite(state[name]).all():
            raise ValueError("Nonfinite recording cache moments")
    if (state["m2"] < 0).any() or (state["min"] > state["max"]).any():
        raise ValueError("Invalid recording cache moments")
    for column in range(width):
        values, counts = state[f"values_{column}"], state[f"counts_{column}"]
        if (
            len(values) > segment_frames
            or not np.isfinite(values).all()
            or not (values[1:] > values[:-1]).all()
            or not (counts > 0).all()
            or (counts > segment_frames).any()
        ):
            raise ValueError("Invalid recording cache segment distribution")
        cumulative = np.cumsum(counts, dtype=np.uint64)
        if (
            (cumulative[1:] <= cumulative[:-1]).any()
            or cumulative[-1] != segment_frames
            or values[0] < state["min"][column]
            or values[-1] > state["max"][column]
        ):
            raise ValueError("Recording cache segment counts or bounds disagree")


def _read_layer(root, relative, frames, features, fps, current_binding):
    path, details = cache._regular(root, relative)
    signature = _signature(details)
    array_count, byte_limit = cache._archive_limits(features, frames)
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
                and entry.file_size > cache._MAX_MANIFEST_BYTES + 65536
                for entry in entries
            )
        ):
            raise ValueError("Invalid recording statistics archive size or entries")
    with np.load(path, allow_pickle=False, max_header_size=65536) as archive:
        encoded = archive["manifest"]
        if (
            encoded.dtype != np.uint8
            or encoded.ndim != 1
            or encoded.size > cache._MAX_MANIFEST_BYTES
        ):
            raise ValueError("Invalid recording statistics manifest array")
        manifest = cache._decode_json(encoded.tobytes())
        common = {"version", "frames", "fps", "features", "trackers", "image_stats", "binding"}
        if not isinstance(manifest, dict) or type(manifest.get("version")) is not int:
            raise ValueError("Invalid recording statistics manifest")
        version = manifest["version"]
        if (
            version not in (1, 2)
            or set(manifest) != (common if version == 1 else common | {"parent"})
            or type(manifest["frames"]) is not int
            or manifest["frames"] != frames
            or type(manifest["fps"]) not in (int, float)
            or manifest["fps"] != fps
            or manifest["features"] != features
        ):
            raise ValueError("Recording statistics cache does not match its commit")
        parent = _parent(manifest["parent"], frames) if version == 2 else None
        segment_frames = frames - parent[1] if parent else frames
        # A delta has the same moment vectors, but only segment-sized values/counts.
        _, segment_limit = cache._archive_limits(features, segment_frames)
        if sum(entry.file_size for entry in entries) > segment_limit:
            raise ValueError("Recording statistics segment exceeds its size limit")
        source = manifest["binding"]
        if version == 2:
            _validate_compact_binding(source)
            if relative == cache._CACHE and source != compact_binding(current_binding):
                raise ValueError("Recording statistics cache does not match the current commit")
        else:
            if (
                not isinstance(source, dict)
                or set(source) != {"metadata_sha256", "files"}
                or not isinstance(source["metadata_sha256"], dict)
                or not isinstance(source["files"], list)
                or any(
                    not isinstance(value, dict) or "path" not in value for value in source["files"]
                )
            ):
                raise ValueError("Invalid recording cache source binding")
            if relative == cache._CACHE:
                if source != current_binding:
                    raise ValueError("Recording statistics cache does not match the current commit")
            else:
                # There is only one v1 base. Latest v2 already binds every
                # current formal file; ancestry must not rescan that list.
                cache._check_history(source, current_binding)
        numeric, indices = cache._numeric(features), manifest["trackers"]
        if (
            not isinstance(indices, dict)
            or set(indices) != set(numeric)
            or any(type(value) is not int for value in indices.values())
            or set(indices.values()) != set(range(len(numeric)))
        ):
            raise ValueError("Invalid recording accumulator index")
        expected, states = {"manifest"}, {}
        for key, feature in numeric.items():
            width, prefix = cache._width(feature), f"tracker_{indices[key]}__"
            keys = {"count", "mean", "m2", "min", "max"} | {
                f"{name}_{column}" for name in ("values", "counts") for column in range(width)
            }
            expected.update(prefix + name for name in keys)
            state = {name: archive[prefix + name] for name in keys}
            validate_state(state, frames, segment_frames, width)
            states[key] = state
        if set(archive.files) != expected:
            raise ValueError("Unexpected recording statistics cache arrays")
    _, after = cache._regular(root, relative)
    if _signature(after) != signature:
        raise ValueError("Recording statistics cache changed during loading")
    return manifest, states, parent, signature


def load_chain(root, info, binding):
    """Replay distributions once; use the latest cumulative moments bit for bit."""
    from alohamini.datasets.exact_distribution import ExactDistributionStats

    features, frames = info["features"], info["total_frames"]
    relative, expected_frames, seen, segments = cache._CACHE, frames, set(), []
    latest = latest_states = latest_signature = None
    while True:
        if relative in seen or len(seen) >= frames:
            raise ValueError("Cycle or excessive depth in recording statistics cache")
        seen.add(relative)
        manifest, states, parent, signature = _read_layer(
            root, relative, expected_frames, features, info["fps"], binding
        )
        if latest is None:
            latest, latest_states, latest_signature = manifest, states, signature
        segments.append(states)
        if parent is None:
            break
        relative, expected_frames = parent
    restored = {}
    for key, feature in cache._numeric(features).items():
        final = {name: latest_states[key][name] for name in ("count", "mean", "m2", "min", "max")}
        for column in range(cache._width(feature)):
            name, count_name = f"values_{column}", f"counts_{column}"
            if len(segments) == 1:
                values, counts = segments[0][key][name], segments[0][key][count_name]
            else:
                # Matching values retain the oldest representative, including
                # the sign bit of zero, exactly as normal incremental updates.
                values = np.concatenate([part[key][name] for part in reversed(segments)])
                counts = np.concatenate([part[key][count_name] for part in reversed(segments)])
                _, first, inverse = np.unique(values, return_index=True, return_inverse=True)
                values = values[first]
                combined = np.zeros(len(values), dtype=np.int64)
                np.add.at(combined, inverse, counts)
                counts = combined
            final[name], final[count_name] = values, counts
        restored[key] = ExactDistributionStats.from_state(final)
    return latest, restored, latest_signature
