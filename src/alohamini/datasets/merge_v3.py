"""Merge local v3 datasets without resampling actions or re-encoding video."""

import json
import shutil
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alohamini.datasets.image_statistics import ImageStatistics
from alohamini.datasets.lerobot_tools import (
    IntegrityChecker,
    _dataset_path,
    aggregate_feature_stats,
)
from alohamini.datasets.lerobotv3 import DATA_PATH, EPISODE_PATH, VIDEO_PATH, _Stats
from alohamini.datasets.record import _recording_contract, _write_json
from alohamini.datasets.tools import _read_lock


def merge_v3(roots, output):
    from alohamini.datasets.edit import _destination, _output_path

    roots = [Path(root).expanduser().resolve() for root in roots]
    if len(set(roots)) != len(roots):
        raise ValueError("Merge sources must be distinct")
    output = _output_path(output, roots)
    with ExitStack() as resources:
        sources, annotations = [], []
        for root in roots:
            resources.enter_context(_read_lock(root))
            checker = IntegrityChecker(root, decode_images=True, decode_videos=True)
            report = checker.run()
            if not report["valid"] or not checker.episode_rows:
                raise ValueError(f"Source requires review before merging: {report}")
            sources.append(checker)
            annotation = root / "meta/alohamini.json"
            annotations.append(json.loads(annotation.read_text()) if annotation.is_file() else None)
        first = sources[0].info
        for source, annotation in zip(sources[1:], annotations[1:], strict=True):
            for key in ("fps", "features", "robot_type"):
                if source.info.get(key) != first.get(key):
                    raise ValueError(f"Cannot merge datasets with different {key}")
            if (annotation is None) != (annotations[0] is None):
                raise ValueError("Cannot mix known and unknown robot calibration")
            if annotation is not None:
                for key in ("state_groups", "state_units"):
                    if annotation.get(key) != annotations[0].get(key):
                        raise ValueError(f"Cannot merge different {key}")
                a, b = (item["source_info"] for item in (annotations[0], annotation))
                for key in ("robot_metadata", "features", "fixed_dimensions"):
                    if _recording_contract(a).get(key) != _recording_contract(b).get(key):
                        raise ValueError(f"Cannot merge datasets with different {key}")
        features = first["features"]
        numeric = {
            key: value
            for key, value in features.items()
            if value["dtype"] not in ("image", "video", "string")
        }
        global_stats = _Stats(numeric, recording=True)
        image_stats = ImageStatistics()
        complete_histograms = True
        image_keys = [
            key for key, value in features.items() if value["dtype"] in ("image", "video")
        ]
        tasks, episode_rows, provenance = [], [], []
        indexed = ("index", "episode_index", "task_index")
        indexed_stats = {}
        episode_offset = frame_offset = data_number = video_number = 0
        schema = None
        with _destination(output, [SimpleNamespace(root=root) for root in roots]) as stage:
            for source, annotation in zip(sources, annotations, strict=True):
                task_frame = pq.read_table(source.root / "meta/tasks.parquet").to_pandas()
                task_map = {}
                for caption, index in zip(task_frame.index, task_frame["task_index"], strict=True):
                    if caption not in tasks:
                        tasks.append(caption)
                    task_map[int(index)] = tasks.index(caption)
                data_map = {}
                for path in sorted(source.data_files):
                    chunk, file = divmod(data_number, 1000)
                    data_number += 1
                    data_map[path] = chunk, file
                    target = stage / DATA_PATH.format(chunk_index=chunk, file_index=file)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    parquet = pq.ParquetFile(path)
                    if schema is None:
                        schema = parquet.schema_arrow
                    if not schema.equals(parquet.schema_arrow, check_metadata=False):
                        raise ValueError("Cannot merge different Arrow schemas")
                    with pq.ParquetWriter(target, schema, compression="zstd") as writer:
                        for batch in parquet.iter_batches(batch_size=512):
                            rows = batch.to_pylist()
                            for row in rows:
                                # External image paths need relocation; embedded
                                # payloads and video references need no pixel rewrite.
                                for key in image_keys:
                                    if features[key]["dtype"] == "image" and not row[key].get(
                                        "bytes"
                                    ):
                                        raise ValueError("Merge requires embedded images or video")
                                row["index"] += frame_offset
                                row["episode_index"] += episode_offset
                                row["task_index"] = task_map[row["task_index"]]
                                if row["episode_index"] not in indexed_stats:
                                    indexed_stats[row["episode_index"]] = _Stats(
                                        {key: numeric[key] for key in indexed}, recording=True
                                    )
                                tracker = indexed_stats[row["episode_index"]]
                                tracker.update([row])
                            global_stats.update(rows)
                            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                videos = {}
                for original in sorted(source.episode_rows, key=lambda row: row["episode_index"]):
                    row = deepcopy(original)
                    old = row["episode_index"]
                    new = old + episode_offset
                    row.update(
                        episode_index=new,
                        dataset_from_index=row["dataset_from_index"] + frame_offset,
                        dataset_to_index=row["dataset_to_index"] + frame_offset,
                    )
                    old_data = _dataset_path(
                        source.root,
                        source.info["data_path"],
                        chunk_index=row["data/chunk_index"],
                        file_index=row["data/file_index"],
                    )
                    row["data/chunk_index"], row["data/file_index"] = data_map[old_data]
                    row["meta/episodes/chunk_index"] = new // 1000
                    row["meta/episodes/file_index"] = 0
                    for key in image_keys:
                        if features[key]["dtype"] != "video":
                            continue
                        prefix = f"videos/{key}"
                        video = _dataset_path(
                            source.root,
                            source.info["video_path"],
                            video_key=key,
                            chunk_index=row[f"{prefix}/chunk_index"],
                            file_index=row[f"{prefix}/file_index"],
                        )
                        if video not in videos:
                            location = divmod(video_number, 1000)
                            video_number += 1
                            videos[video] = location
                            target = stage / VIDEO_PATH.format(
                                video_key=key, chunk_index=location[0], file_index=location[1]
                            )
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(video, target)
                        row[f"{prefix}/chunk_index"], row[f"{prefix}/file_index"] = videos[video]
                    histogram = row.get("image_statistics")
                    if histogram:
                        image_stats.merge(ImageStatistics(json.loads(histogram)))
                    else:
                        complete_histograms = False
                    for key, stats in indexed_stats.pop(new).result().items():
                        row.update(
                            {f"stats/{key}/{name}": value.tolist() for name, value in stats.items()}
                        )
                    episode_rows.append(row)
                    if annotation is not None:
                        safety = source.root / "meta/safety" / f"episode_{old:06d}.jsonl"
                        target = stage / "meta/safety" / f"episode_{new:06d}.jsonl"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with safety.open() as reader, target.open("x") as writer:
                            for line in reader:
                                record = json.loads(line)
                                record["episode_index"] = new
                                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                provenance.append(
                    {
                        "root": str(source.root),
                        "episode_from": episode_offset,
                        "episode_to": episode_offset + len(source.episode_rows),
                    }
                )
                episode_offset += len(source.episode_rows)
                frame_offset += source.total_data_rows
            columns = list(dict.fromkeys(key for row in episode_rows for key in row))
            for number in range(0, len(episode_rows), 1000):
                target = stage / EPISODE_PATH.format(chunk_index=number // 1000, file_index=0)
                # Metadata locations must agree with the actual file containing each row.
                rows = [
                    {key: row.get(key) for key in columns}
                    for row in episode_rows[number : number + 1000]
                ]
                for row in rows:
                    row["meta/episodes/file_index"] = 0
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd")
            meta = stage / "meta"
            pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks).to_parquet(
                meta / "tasks.parquet"
            )
            info = deepcopy(first)
            info.update(
                total_episodes=episode_offset,
                total_frames=frame_offset,
                total_tasks=len(tasks),
                chunks_size=1000,
                data_path=DATA_PATH,
                video_path=VIDEO_PATH if videos else None,
                splits={"train": f"0:{episode_offset}"},
            )
            _write_json(meta / "info.json", info)
            stats = global_stats.result()
            if complete_histograms:
                stats.update(image_stats.result())
            else:
                for key in image_keys:
                    values = [
                        json.loads((source.root / "meta/stats.json").read_text())[key]
                        for source in sources
                    ]
                    stats[key] = {
                        name: value.tolist()
                        for name, value in aggregate_feature_stats(values).items()
                    }
            _write_json(meta / "stats.json", stats)
            if annotations[0] is not None:
                annotation = deepcopy(annotations[0])
                annotation.update(source=[str(root) for root in roots], merged_sources=provenance)
                annotation["image_quantiles"] = (
                    "linear quantiles of merged source RGB histograms"
                    if complete_histograms
                    else "omitted: source image histograms unavailable"
                )
                _write_json(meta / "alohamini.json", annotation)
            _write_json(
                meta / "merge.json",
                {
                    "sources": provenance,
                    "media": "copied without re-encoding",
                    "timing": "physical timestamps retained",
                },
            )
            report = IntegrityChecker(stage).run()
            if not report["valid"]:
                raise ValueError(f"Merged dataset failed validation: {report}")
        report["dataset_root"] = str(output)
        return report
