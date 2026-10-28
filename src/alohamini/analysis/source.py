"""Read selected numeric columns and lazy media references from native/v3 storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

IMAGE_PREFIX = "observation.images."
NUMERIC_COLUMNS = (
    "action",
    "observation.state",
    "observation.motor_current_ma",
    "motor_feedback.current_ma_valid",
)


@dataclass(frozen=True)
class Media:
    kind: str
    path: Path
    index: int = 0
    group: int = 0
    reference: object = None


@dataclass
class Episode:
    root: Path
    index: int
    info: dict
    source_info: dict
    rows: list[dict]
    sidecars: list[dict]
    media: dict[str, list[Media | None]]
    signatures: dict[Path, tuple[int, int]]
    warnings: list[str] = field(default_factory=list)

    def unchanged(self):
        for path, signature in self.signatures.items():
            if file_signature(path) != signature:
                raise RuntimeError(f"分析期间数据已改变，请停止录制后重新分析：{path}")


def file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dataset_path(root, template, **values):
    relative = Path(template.format(**values))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    return path


def load_episode(root: Path, episode: int, *, all_numeric=False, include_task=False) -> Episode:
    root = root.expanduser().resolve()
    info_path = root / "meta/info.json"
    signatures = {info_path: file_signature(info_path)}
    info = _json(info_path)
    source_info = info
    cameras = [key[len(IMAGE_PREFIX) :] for key in info["features"] if key.startswith(IMAGE_PREFIX)]
    if info.get("format") == "alohamini-episodes":
        cameras = list(info.get("cameras", info.get("robot_metadata", {}).get("cameras", cameras)))
    # Named, reproducible display order; unknown camera names retain metadata order.
    preferred = ["forward", "chest", "wrist_left", "wrist_right"]
    cameras = [c for c in preferred if c in cameras] + [c for c in cameras if c not in preferred]
    media = {camera: [] for camera in cameras}
    rows, warnings = [], []
    native = info.get("format") == "alohamini-episodes"
    if native:
        directory = root / "episodes" / f"episode_{episode:06d}"
        data_path = directory / "frames.parquet"
        sidecar_path = directory / "safety.jsonl"
        meta = None
    else:
        if not str(info.get("codebase_version", "")).startswith("v3"):
            raise ValueError("分析支持 AlohaMini 原生记录和 LeRobot v3 数据集")
        platform = root / "meta/alohamini.json"
        if platform.is_file():
            signatures[platform] = file_signature(platform)
            source_info = _json(platform).get("source_info", info)
        matches = []
        for path in sorted((root / "meta/episodes").rglob("*.parquet")):
            signatures[path] = file_signature(path)
            with pq.ParquetFile(path) as file:
                columns = [
                    k
                    for k in file.schema_arrow.names
                    if k == "episode_index"
                    or k == "length"
                    or k.startswith("data/")
                    or (k.startswith("videos/") and not k.startswith("stats/"))
                ]
                matches.extend(
                    r
                    for r in file.read(columns=columns).to_pylist()
                    if r["episode_index"] == episode
                )
        if len(matches) != 1:
            raise ValueError(f"Episode {episode} 的元信息不存在或重复")
        meta = matches[0]
        data_path = _dataset_path(
            root,
            info["data_path"],
            chunk_index=meta["data/chunk_index"],
            file_index=meta["data/file_index"],
        )
        sidecar_path = root / "meta/safety" / f"episode_{episode:06d}.jsonl"
    signatures[data_path] = file_signature(data_path)
    with pq.ParquetFile(data_path) as file:
        columns = [
            c
            for c in (*NUMERIC_COLUMNS, "timestamp", "frame_index", "episode_index")
            if c in file.schema_arrow.names
        ]
        if all_numeric:
            columns += [
                k
                for k in info["features"]
                if k not in columns
                and not k.startswith(IMAGE_PREFIX)
                and k in file.schema_arrow.names
                and info["features"][k].get("dtype") not in ("string", "image", "video")
            ]
        if include_task:
            columns += [key for key in ("task", "task_index") if key in file.schema_arrow.names]
        columns = list(dict.fromkeys(columns))
        if native:
            columns += [
                IMAGE_PREFIX + c for c in cameras if IMAGE_PREFIX + c in file.schema_arrow.names
            ]
        for group in range(file.num_row_groups):
            for offset, row in enumerate(file.read_row_group(group, columns=columns).to_pylist()):
                if row.get("episode_index") != episode:
                    continue
                rows.append(row)
                for camera in cameras:
                    key = IMAGE_PREFIX + camera
                    ref = None
                    if native:
                        from alohamini.datasets.images import image_path

                        value = row.pop(key, None)
                        if value is not None:
                            path = image_path(directory, camera, value)
                            ref = (
                                Media("video", path, value["frame_index"])
                                if isinstance(value, dict) and "frame_index" in value
                                else Media(
                                    "native_host_image"
                                    if info.get("image_color") == "opencv_imdecode_color_is_rgb"
                                    and not (isinstance(value, str) and value.endswith(".png"))
                                    else "native_image",
                                    directory,
                                    reference=value,
                                )
                            )
                    elif info["features"][key]["dtype"] == "video":
                        path = _dataset_path(
                            root,
                            info["video_path"],
                            video_key=key,
                            chunk_index=meta[f"videos/{key}/chunk_index"],
                            file_index=meta[f"videos/{key}/file_index"],
                        )
                        start = float(meta[f"videos/{key}/from_timestamp"]) * info["fps"]
                        if abs(start - round(start)) > 0.05:
                            raise ValueError(f"{camera}: 视频起点不在声明的帧率网格内")
                        ref = Media("video", path, round(start) + row["frame_index"])
                    elif info["features"][key]["dtype"] == "image":
                        ref = Media("v3_image", data_path, offset, group)
                    media[camera].append(ref)
    if not rows:
        raise ValueError(f"Episode {episode} 没有数据")
    if include_task and any(not isinstance(row.get("task"), str) for row in rows):
        tasks_path = root / "meta/tasks.parquet"
        tasks = {}
        if tasks_path.is_file():
            signatures[tasks_path] = file_signature(tasks_path)
            table = pq.read_table(tasks_path).to_pandas()
            # LeRobot stores task text as the pandas index, keyed by task_index.
            texts = table["task"] if "task" in table else table.index
            tasks = dict(zip(table["task_index"], texts, strict=True))
        for row in rows:
            if not isinstance(row.get("task"), str):
                text = tasks.get(row.get("task_index"), info.get("task") if native else None)
                if text is not None:
                    row["task"] = text
    frames = [r["frame_index"] for r in rows]
    if frames != sorted(set(frames)):
        raise ValueError("frame_index 重复或乱序，不能可靠对齐画面和动作")
    if frames != list(range(len(frames))):
        warnings.append("frame_index 不连续；保留原索引，跨缺口变化不代表单步变化。")
    if meta is not None and len(rows) != meta["length"]:
        warnings.append(f"元信息长度 {meta['length']} 与读取行数 {len(rows)} 不同。")
    sidecars = {}
    if sidecar_path.is_file():
        signatures[sidecar_path] = file_signature(sidecar_path)
        with sidecar_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                except ValueError:
                    warnings.append(f"sidecar 第 {line_number} 行损坏，已跳过。")
                    continue
                if entry.get("episode_index") != episode or entry.get("frame_index") is None:
                    continue
                frame = entry["frame_index"]
                # Recorder lifecycle events are not observations.
                if entry.get("event") and not entry.get("host_timing"):
                    continue
                if frame in sidecars:
                    raise ValueError(f"sidecar frame_index={frame} 重复，无法可靠对齐")
                sidecars[frame] = entry
    aligned = [sidecars.get(f, {}) for f in frames]
    if not all(aligned):
        warnings.append(f"{sum(not x for x in aligned)} 帧缺少保护/时间记录；相关诊断不可用。")
    result = Episode(root, episode, info, source_info, rows, aligned, media, signatures, warnings)
    result.unchanged()
    return result


def feature_names(data: Episode, column: str, width: int) -> list[str]:
    names = data.info["features"].get(column, {}).get("names")
    if isinstance(names, dict):
        names = names.get("motors")
    if not isinstance(names, list) or len(names) != width or len(set(names)) != width:
        return [f"{column}[{i}]" for i in range(width)]
    return names


def unit(data: Episode, name: str, *, column="action") -> str:
    feature = data.info["features"].get(column, {})
    declared = feature.get("units", {})
    if isinstance(declared, dict) and isinstance(declared.get(name), str):
        return declared[name]
    if isinstance(declared, list):
        names = feature_names(data, column, len(declared))
        if name in names and isinstance(declared[names.index(name)], str):
            return declared[names.index(name)]
    metadata = data.source_info.get("robot_metadata", {})
    motor = metadata.get("motors", {}).get(name.removesuffix(".pos"), {})
    if name.endswith(".pos") and motor.get("normalization"):
        return motor["normalization"]
    if name.endswith(".height_mm"):
        return "mm"
    if metadata.get("robot_model") and name in ("x.vel", "y.vel", "theta.vel"):
        return "deg/s" if name == "theta.vel" else "m/s"
    return "recorded units (unverified)"


def numeric(data: Episode, column: str) -> np.ndarray | None:
    if not any(row.get(column) is not None for row in data.rows):
        return None
    width = next(len(row[column]) for row in data.rows if row.get(column) is not None)
    if any(row.get(column) is not None and len(row[column]) != width for row in data.rows):
        raise ValueError(f"{column}: 数值维度不一致")
    return np.asarray([r.get(column) or [float("nan")] * width for r in data.rows], dtype=float)
