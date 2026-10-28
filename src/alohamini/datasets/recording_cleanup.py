"""Discard only a verified, interrupted recorder transaction under both recorder locks."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from alohamini.datasets.recording_index import _path, read_recording_index

_GLOBALS = {
    f"meta/{name}" for name in ("info.json", "stats.json", "tasks.parquet", "alohamini.json")
}


def _json_file(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate recording metadata key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid recording metadata: {path}")
    return value


def resume_source_info(root):
    """Read the source contract even if the first episode has not been committed."""
    root = Path(root).resolve()
    source = _path(root, "meta/alohamini.json")
    if source.is_file():
        info = _json_file(source).get("source_info")
        if not isinstance(info, dict):
            raise ValueError("Invalid recording source metadata")
        return info
    return _json_file(_path(root, ".recording/meta/info.json", required=True))


def _episode_paths(index, cameras):
    from alohamini.datasets.lerobotv3 import DATA_PATH, EPISODE_PATH, VIDEO_PATH

    chunk, file = divmod(index, 1000)
    return {
        DATA_PATH.format(chunk_index=chunk, file_index=file),
        EPISODE_PATH.format(chunk_index=chunk, file_index=file),
        f"meta/safety/episode_{index:06d}.jsonl",
        *(
            VIDEO_PATH.format(
                video_key=f"observation.images.{camera}", chunk_index=chunk, file_index=file
            )
            for camera in cameras
        ),
    }


def _regular_tree(directory, files, directories, *, cameras=()):
    """Check every object before deleting anything, without following symbolic links."""
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            if relative not in directories:
                raise ValueError(f"Unrecognized recording directory retained: {path}")
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Unsafe recording file retained: {path}")
        image = re.fullmatch(r"images/([^/]+)/frame_(\d{6,})\.(jpg|png)", relative)
        if relative not in files and not (
            image and image[1] in cameras and image[2] == f"{int(image[2]):06d}"
        ):
            raise ValueError(f"Unrecognized recording file retained: {path}")


def _transaction(root, cameras):
    staging = _path(root, ".recording", required=True, directory=True)
    for path in staging.iterdir():
        if path.name not in {"recording.lock", "meta", "episodes"}:
            raise ValueError(f"Unrecognized recording file retained: {path}")
        _path(root, path.relative_to(root), directory=path.name != "recording.lock")
    meta = _path(root, ".recording/meta", required=True, directory=True)
    _regular_tree(meta, {"info.json", "info.json.tmp"}, set())
    episodes = _path(root, ".recording/episodes", required=True, directory=True)
    entries = list(episodes.iterdir())
    if not entries:
        return None
    if len(entries) != 1:
        raise ValueError("Multiple interrupted recording entries retained; recover explicitly")
    episode = entries[0]
    match = re.fullmatch(r"episode_(\d{6,})(\.pending)?", episode.name)
    if match is None or match[1] != f"{int(match[1]):06d}":
        raise ValueError(f"Unrecognized recording entry retained: {episode}")
    _path(root, episode.relative_to(root), required=True, directory=True)
    index = int(match[1])
    new_files = _episode_paths(index, cameras)
    files = {
        "journal.jsonl",
        "frames.parquet",
        "frames.video.parquet",
        "safety.jsonl",
        "episode.json",
        "image_stats.json",
        "publish.json",
        "episode.json.tmp",
        "image_stats.json.tmp",
        "publish.json.tmp",
        *(f"videos/{camera}.mp4" for camera in cameras),
        *(f"v3/{name}" for name in new_files | _GLOBALS),
        *(f"v3/{name}.tmp" for name in _GLOBALS if name.endswith(".json")),
        *(f"backup/{name}" for name in _GLOBALS),
    }
    directories = {"images", "videos", "v3", "backup", "backup/meta"}
    directories.update(f"images/{camera}" for camera in cameras)
    directories.update(
        str(parent) for name in files for parent in Path(name).parents if str(parent) != "."
    )
    _regular_tree(episode, files, directories, cameras=cameras)
    for relative in new_files | _GLOBALS:
        _path(root, relative)
    return episode, index, new_files


def _first_episode(root, new_files, *, allow_new):
    # Missing metadata is not proof that an existing recording contains no history.
    for relative, suffix in (
        ("data", ".parquet"),
        ("videos", ".mp4"),
        ("meta/episodes", ".parquet"),
        ("meta/safety", ".jsonl"),
    ):
        directory = _path(root, relative, directory=True)
        for path in directory.rglob(f"*{suffix}"):
            _path(root, path.relative_to(root), required=True)
            if not allow_new or path.relative_to(root).as_posix() not in new_files:
                raise ValueError(
                    "Missing commit metadata with existing recording history; retained"
                )


def _check_tasks(path, task):
    import pyarrow.parquet as pq

    tasks = pq.read_table(path).to_pandas()
    texts = tasks["task"] if "task" in tasks else tasks.index
    if "task_index" not in tasks or list(tasks["task_index"]) != [0] or list(texts) != [task]:
        raise ValueError("Recording publication task metadata does not match its source")


def _check_stats(stats, info):
    if set(stats) != set(info["features"]):
        raise ValueError("Recording publication statistics features do not match")
    for key, feature in info["features"].items():
        if feature["dtype"] not in ("video", "image") and stats[key].get("count") != [
            info["total_frames"]
        ]:
            raise ValueError("Recording publication statistics counts do not match")


def _rollback_plan(root, episode, index, new_files, source_info, current):
    from alohamini.datasets.lerobotv3 import DATA_PATH, VIDEO_PATH, _features
    from alohamini.datasets.record import StateSelection, _recording_contract

    manifest = _json_file(episode / "publish.json")
    paths = manifest.get("files")
    if (
        set(manifest) != {"episode_index", "files"}
        or type(manifest.get("episode_index")) is not int
        or manifest["episode_index"] != index
        or not isinstance(paths, list)
        or any(not isinstance(path, str) for path in paths)
        or len(paths) != len(set(paths))
        or set(paths) != new_files | _GLOBALS
        or episode.name.endswith(".pending")
    ):
        raise ValueError("Invalid recording publication manifest; interrupted episode retained")
    summary = _json_file(episode / "episode.json")
    length = summary.get("length")
    shapes = summary.get("image_shapes")
    cameras = source_info["robot_metadata"]["cameras"]
    if (
        type(summary.get("episode_index")) is not int
        or summary["episode_index"] != index
        or type(length) is not int
        or length <= 0
        or not isinstance(shapes, dict)
        or set(shapes) != set(cameras)
        or any(
            not isinstance(shape, list)
            or len(shape) != 3
            or shape[2] != 3
            or any(type(value) is not int or value <= 0 for value in shape)
            for shape in shapes.values()
        )
    ):
        raise ValueError("Invalid interrupted episode summary; retained")
    staged = _json_file(episode / "v3/meta/info.json")
    expected_frames = (current["total_frames"] if current else 0) + length
    if (
        staged.get("codebase_version") != "v3.0"
        or type(staged.get("total_episodes")) is not int
        or staged["total_episodes"] != index + 1
        or type(staged.get("total_frames")) is not int
        or staged["total_frames"] != expected_frames
        or staged.get("total_tasks") != 1
        or staged.get("fps") != source_info["fps"]
        or staged.get("robot_type") != source_info["robot_metadata"]["robot_model"]
        or staged.get("data_path") != DATA_PATH
        or staged.get("video_path") != VIDEO_PATH
        or staged.get("features") != _features(source_info, StateSelection(source_info), shapes)
    ):
        raise ValueError("Invalid staged publication commit metadata; retained")
    if index:
        for relative in _GLOBALS:
            _path(root, (episode / "backup" / relative).relative_to(root), required=True)
        old_info = _json_file(episode / "backup/meta/info.json")
        source = _json_file(episode / "backup/meta/alohamini.json")
        if (
            old_info != current
            or old_info.get("total_episodes") != index
            or not isinstance(source.get("source_info"), dict)
            or _recording_contract(source["source_info"]) != _recording_contract(source_info)
        ):
            raise ValueError("Recording publication backups do not match the committed dataset")
        _check_stats(_json_file(episode / "backup/meta/stats.json"), old_info)
        _check_tasks(episode / "backup/meta/tasks.parquet", source_info["task"])
        read_recording_index(root, source_info, _interrupted_episode=index)
    else:
        if any((episode / "backup" / relative).exists() for relative in _GLOBALS):
            raise ValueError("First episode unexpectedly has publication backups; retained")
        _first_episode(root, new_files, allow_new=True)
        custom = root / "meta/alohamini.json"
        if custom.exists() and _recording_contract(_json_file(custom).get("source_info", {})) != (
            _recording_contract(source_info)
        ):
            raise ValueError("First publication source metadata does not match; retained")
    return index > 0


def _restore_metadata(root, episode, *, backups):
    for relative in [*sorted(_GLOBALS - {"meta/info.json"}), "meta/info.json"]:
        target = _path(root, relative)
        if not backups:
            target.unlink(missing_ok=True)
            continue
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".discard-", delete=False
            ) as f:
                temporary = Path(f.name)
                with (episode / "backup" / relative).open("rb") as source:
                    shutil.copyfileobj(source, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _sync_parents(paths):
    for parent in {path.parent for path in paths}:
        if not parent.is_dir():
            continue
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def discard_interrupted_recording(root, source_info):
    """The caller must own the root and staging recording locks and validate its request."""
    root = Path(root).resolve()
    transaction = _transaction(root, source_info["robot_metadata"]["cameras"])
    if transaction is None:
        return
    episode, index, new_files = transaction
    info_path = _path(root, "meta/info.json")
    current = _json_file(info_path) if info_path.is_file() else None
    count = current.get("total_episodes") if current else 0
    if type(count) is not int or count < 0:
        raise ValueError("Invalid recording commit counter; interrupted episode retained")
    if count == index + 1 and not episode.name.endswith(".pending"):
        read_recording_index(root, source_info, _interrupted_episode=index)
        shutil.rmtree(episode)
        logging.info(
            "Removed temporary files for saved episode %d; saved episodes unchanged", index
        )
        return
    if count != index:
        raise ValueError("Interrupted episode does not follow the committed dataset; retained")
    if (episode / "publish.json").exists():
        backups = _rollback_plan(root, episode, index, new_files, source_info, current)
        for relative in new_files:
            _path(root, relative).unlink(missing_ok=True)
        _restore_metadata(root, episode, backups=backups)
        # Make the rollback durable before discarding its only metadata backups.
        _sync_parents(root / relative for relative in new_files | _GLOBALS)
    else:
        if any((root / relative).exists() for relative in new_files):
            raise ValueError("Publication files exist without a valid manifest; retained")
        if index:
            read_recording_index(root, source_info, _interrupted_episode=index)
        else:
            if any((root / relative).exists() for relative in _GLOBALS):
                raise ValueError(
                    "First publication metadata exists without a valid manifest; retained"
                )
            _first_episode(root, new_files, allow_new=False)
    shutil.rmtree(episode)
    logging.info("Discarded interrupted episode %d; saved episodes unchanged", index)
