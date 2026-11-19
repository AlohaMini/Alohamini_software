#!/usr/bin/env python3
"""Copy a LeRobot v3 dataset into explicitly declared model coordinates.

The JSON plan records the source calibration, a verified destination coordinate
template and the physical reference assumption. No serial port is opened.
Requires numpy, pyarrow and PyAV for video input. Videos are hard linked;
numeric files are rewritten.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = "alohamini-coordinate-conversion"
FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
PRESERVED = {"observation.state": "observation.source_state", "action": "source_action"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def telemetry_name(name):
    if name == "observation.motor_position_raw":
        return "observation.source_motor_position_raw"
    if name == "motor_feedback.position_raw_valid":
        return "source_motor_feedback.position_raw_valid"
    return name


def check_media(source, info):
    videos = [key for key, value in info["features"].items() if value["dtype"] == "video"]
    if not videos:
        return 0
    import av

    inspected, problems = {}, []
    for path in sorted(source.glob("meta/episodes/**/*.parquet")):
        for episode in pq.read_table(path).to_pylist():
            for key in videos:
                filename = source / info["video_path"].format(
                    video_key=key,
                    chunk_index=episode[f"videos/{key}/chunk_index"],
                    file_index=episode[f"videos/{key}/file_index"],
                )
                if filename not in inspected:
                    try:
                        if not filename.resolve().is_relative_to(source) or filename.is_symlink():
                            raise ValueError("Video must be inside the source dataset")
                        with av.open(str(filename)) as container:
                            stream = container.streams.video[0]
                            if not stream.average_rate or not np.isclose(
                                float(stream.average_rate), info["fps"]
                            ):
                                raise ValueError("Video FPS differs from the dataset")
                            inspected[filename] = stream.frames
                    except (OSError, ValueError, IndexError) as error:
                        inspected[filename] = None
                        problems.append(f"{filename.relative_to(source)}: {error}")
                frames = inspected[filename]
                required = round(episode[f"videos/{key}/from_timestamp"] * info["fps"])
                required += episode["length"]
                if frames and frames < required:
                    problems.append(f"{filename.relative_to(source)}: video is too short")
    if problems:
        raise ValueError(
            f"Source media is incomplete ({len(problems)} errors); no output created: "
            + "; ".join(problems[:12])
        )
    return len(inspected)


def inverse_units(values, motor):
    mode = motor["normalization"]
    values = np.asarray(values, dtype=np.float64)
    if mode == "range_m100_100":
        fraction = ((-values if motor["drive_mode"] else values) + 100) / 200
    elif mode == "range_0_100":
        fraction = (100 - values if motor["drive_mode"] else values) / 100
    else:
        raise ValueError(f"Unsupported arm normalization: {mode}")
    return motor["range_min"] + fraction * (motor["range_max"] - motor["range_min"])


def forward_units(ticks, motor):
    fraction = (ticks - motor["range_min"]) / (motor["range_max"] - motor["range_min"])
    if motor["normalization"] == "range_m100_100":
        values = fraction * 200 - 100
        return -values if motor["drive_mode"] else values
    values = fraction * 100
    return 100 - values if motor["drive_mode"] else values


class Conversion:
    def __init__(self, plan, names):
        if (
            plan.get("format") != FORMAT
            or plan.get("version") != 1
            or plan.get("method") not in ("same_encoder", "legacy_2047", "legacy_home_contacts")
            or not isinstance(plan.get("basis"), str)
            or not plan["basis"].strip()
        ):
            raise ValueError("Provide a version 1 conversion plan and its physical basis")
        self.plan, self.names = plan, names
        self.source = plan["source_robot_metadata"]["motors"]
        self.target = plan["target_robot_metadata"]["motors"]
        self.arms = [
            (i, name.removesuffix(".pos"))
            for i, name in enumerate(names)
            if name.startswith("arm_") and name.endswith(".pos")
        ]
        if len(self.arms) != 14 or len(names) != 18 or len(set(names)) != len(names):
            raise ValueError("Conversion requires the named 18-dimensional dual-arm contract")
        for _, name in self.arms:
            for motor in (self.source[name], self.target[name]):
                if (
                    not 0 <= motor["range_min"] < motor["range_max"] <= 4095
                    or motor["drive_mode"] not in (0, 1)
                    or not -2047 <= motor["homing_offset"] <= 2047
                ):
                    raise ValueError(f"Invalid motor parameters: {name}")
            if any(self.source[name][k] != self.target[name][k] for k in ("id", "model")):
                raise ValueError(f"Source/destination motor identity differs: {name}")
        self.offsets = {}
        for _, name in self.arms:
            if plan["method"] == "same_encoder":
                shift = self.source[name]["homing_offset"] - self.target[name]["homing_offset"]
            elif plan["method"] == "legacy_home_contacts":
                anchor = plan["reference_anchors"][name]
                shift = anchor["target_tick"] - anchor["source_tick"]
            elif name.endswith("_gripper"):
                # A scanned closed gripper is an endpoint, never the arm's 2047 pose.
                shift = self.target[name]["range_min"] - self.source[name]["range_min"]
            else:
                shift = (
                    plan["target_robot_metadata"]["reference_calibration"]["reference_ticks"][name]
                    - 2047
                )
            self.offsets[name] = shift

    def transform(self, values, *, raw=None, raw_valid=None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.names) or not np.isfinite(values).all():
            raise ValueError("State/action must be finite vectors with declared names")
        result = values.copy()
        for index, name in self.arms:
            ticks = inverse_units(values[:, index], self.source[name])
            if raw is not None:
                valid = raw_valid[:, index] == 1
                measured = raw[:, index]
                if np.any(valid & (~np.isfinite(measured) | (measured < 0) | (measured > 4095))):
                    raise ValueError(f"Invalid recorded encoder feedback: {name}")
                reconstructed = forward_units(measured[valid], self.source[name])
                low = 0 if name.endswith("_gripper") else -100
                if not np.allclose(
                    np.clip(reconstructed, low, 100), values[valid, index], atol=2e-4, rtol=0
                ):
                    raise ValueError(f"Raw feedback and source calibration disagree: {name}")
                ticks = np.where(valid, measured, ticks)
            # Whole-turn branch changes are explicit, with no normalized clipping.
            shifted = ticks + self.offsets[name]
            if self.plan["method"] == "same_encoder":
                shifted %= 4096
            result[:, index] = forward_units(shifted, self.target[name])
        return result.astype(np.float32)

    def mapping(self, values):
        unknown = set(values) - set(self.names)
        if unknown:
            raise ValueError(f"Unknown position fields in safety metadata: {sorted(unknown)}")
        result = {}
        for name, value in values.items():
            # Dataset vectors contain float32 requests; preserve the same rounding
            # in the converted sidecar, even if the source logger used float64.
            value = float(np.float32(value))
            motor = name.removesuffix(".pos")
            if motor in self.offsets:
                old = self.source[motor]
                if old["normalization"] == "range_m100_100":
                    fraction = ((-value if old["drive_mode"] else value) + 100) / 200
                else:
                    fraction = (100 - value if old["drive_mode"] else value) / 100
                tick = old["range_min"] + fraction * (old["range_max"] - old["range_min"])
                tick += self.offsets[motor]
                if self.plan["method"] == "same_encoder":
                    tick %= 4096
                value = float(forward_units(tick, self.target[motor]))
            result[name] = float(np.float32(value))
        return result


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "min": values.min(0),
        "max": values.max(0),
        "mean": values.mean(0),
        "std": values.std(0),
        "count": np.array([len(values)], dtype=np.int64),
    }
    result.update(
        zip(
            ("q01", "q10", "q50", "q90", "q99"),
            np.quantile(values, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0),
            strict=True,
        )
    )
    return {key: value.tolist() for key, value in result.items()}


def converted_safety(source, output, conversion, target_metadata):
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as incoming, output.open("w") as outgoing:
        for line in incoming:
            record = json.loads(line)
            for key in ("robot_metadata", "motor_feedback"):
                if key in record:
                    record["source_" + key] = record.pop(key)
            if "source_robot_metadata" in record:
                record["robot_metadata"] = target_metadata
            if record.get("requested_action") is not None:
                record["source_requested_action"] = record["requested_action"]
                record["requested_action"] = conversion.mapping(record["requested_action"])
            safety = record.get("safety") or {}
            for key in ("requested_targets", "accepted_targets"):
                if safety.get(key) is not None:
                    safety["source_" + key] = safety[key]
                    safety[key] = conversion.mapping(safety[key])
            record["coordinate_conversion"] = "meta/coordinate_conversion.json"
            outgoing.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")


def convert(source, plan, *, output=None):
    source = Path(source).expanduser().resolve()
    info = json.loads((source / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Source must be a LeRobot v3 dataset")
    names = info["features"]["action"]["names"]
    if info["features"]["observation.state"]["names"] != names:
        raise ValueError("State and action must have identical explicit field names/order")
    if any(name in info["features"] for name in PRESERVED.values()):
        raise ValueError("Dataset already contains source-coordinate columns")
    if "observation.model_state_unclipped" in info["features"]:
        raise ValueError("Dataset already contains converted model coordinates")
    original_path = source / "meta/alohamini.json"
    annotation = json.loads(original_path.read_text()) if original_path.is_file() else None
    if annotation:
        recorded = annotation["source_info"]["robot_metadata"]["motors"]
        for name, motor in plan["source_robot_metadata"]["motors"].items():
            if any(
                recorded.get(name, {}).get(k) != motor[k]
                for k in (*FIELDS, "normalization", "model")
            ):
                raise ValueError(f"Dataset source calibration does not match plan: {name}")
    conversion = Conversion(plan, names)
    checked_media = check_media(source, info)
    metadata = deepcopy(plan["target_robot_metadata"])
    metadata["cameras"] = [
        k.removeprefix("observation.images.")
        for k in info["features"]
        if k.startswith("observation.images.")
    ]
    renames = {k: telemetry_name(k) for k in info["features"] if telemetry_name(k) != k}
    features = {renames.get(k, k): deepcopy(v) for k, v in info["features"].items()}
    for key, preserved in PRESERVED.items():
        features[preserved] = deepcopy(features[key])
    features["observation.model_state_unclipped"] = deepcopy(features["observation.state"])
    new_info = {**info, "robot_type": metadata["robot_model"], "features": features}
    output = Path(output).expanduser().absolute() if output is not None else None
    staging = None
    if output is not None:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Output already exists: {output}")
        if output.resolve().is_relative_to(source) or source.is_relative_to(output.resolve()):
            raise ValueError("Output must be separate from source")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".pending-{output.name}-", dir=output.parent))
    report = {
        "format": FORMAT,
        "version": 1,
        "converter_sha256": file_digest(__file__),
        "source_media_headers_checked": checked_media,
        "status": "preflight",
        "source": str(source),
        "plan": plan,
        "source_capture_metadata": "retained" if annotation else "unavailable",
        "target_contract_sha256": digest(metadata),
        "features_sha256": digest(features),
        "frames": 0,
        "episodes": 0,
        "bounds": {},
        "source_hashes": {},
        "raw_state_recovered": 0,
        "source_state_at_endpoint": [0] * 14,
        "media": {"method": "hard_link", "files": 0, "bytes": 0},
    }
    arrays = defaultdict(list)
    episodes = defaultdict(lambda: defaultdict(list))
    frame_indices = defaultdict(list)
    try:
        paths = sorted(source.glob("data/**/*.parquet"))
        if not paths:
            raise ValueError("No data parquet files")
        for number, path in enumerate(paths, 1):
            relative = path.relative_to(source)
            report["source_hashes"][str(relative)] = file_digest(path)
            table = pq.read_table(path)
            old = {k: np.array(table[k].to_pylist(), dtype=np.float64) for k in PRESERVED}
            if any(a.shape != (table.num_rows, 18) for a in old.values()):
                raise ValueError(f"Bad vector dimensions: {relative}")
            raw = valid = None
            if "observation.motor_position_raw" in table.column_names:
                feature_names = info["features"]["observation.motor_position_raw"]["names"]
                order = [
                    feature_names.index(n.removesuffix(".pos")) if n.endswith(".pos") else 0
                    for n in names
                ]
                raw = np.asarray(table["observation.motor_position_raw"].to_pylist())[:, order]
                valid = np.asarray(table["motor_feedback.position_raw_valid"].to_pylist())[:, order]
                report["raw_state_recovered"] += int(np.sum(np.all(valid[:, :14] == 1, axis=1)))
            measured = conversion.transform(old["observation.state"], raw=raw, raw_valid=valid)
            state = measured.copy()
            # Match the actual HostPositionUnits feedback channel. Preserve the
            # unclipped coordinate separately; submitted actions are not clipped.
            for index, name in conversion.arms:
                state[:, index] = np.clip(
                    state[:, index], 0 if name.endswith("_gripper") else -100, 100
                )
            converted = {
                "observation.state": state,
                "action": conversion.transform(old["action"]),
                "observation.model_state_unclipped": measured,
            }
            table = table.rename_columns([renames.get(k, k) for k in table.column_names])
            for key, kept in PRESERVED.items():
                table = table.append_column(kept, table[key])
                table = table.set_column(
                    table.schema.get_field_index(key),
                    key,
                    pa.array(converted[key].tolist(), type=table[key].type),
                )
            table = table.append_column(
                "observation.model_state_unclipped",
                pa.array(measured.tolist(), type=table["observation.state"].type),
            )
            # Arrow/Pandas/HF schema metadata can otherwise describe the old field list.
            table = table.replace_schema_metadata(None)
            ids = np.asarray(table["episode_index"].to_pylist())
            frames = np.asarray(table["frame_index"].to_pylist())
            values = {**converted, **{PRESERVED[k]: v.astype(np.float32) for k, v in old.items()}}
            for key, values_array in values.items():
                arrays[key].append(values_array)
                for ep in np.unique(ids):
                    episodes[int(ep)][key].append(values_array[ids == ep])
            for ep in np.unique(ids):
                frame_indices[int(ep)].extend(frames[ids == ep].tolist())
            for j, (index, name) in enumerate(conversion.arms):
                low = 0 if name.endswith("_gripper") else -100
                report["source_state_at_endpoint"][j] += int(
                    np.count_nonzero(
                        (old["observation.state"][:, index] <= low + 1e-5)
                        | (old["observation.state"][:, index] >= 100 - 1e-5)
                    )
                )
            report["frames"] += table.num_rows
            if staging:
                dest = staging / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, dest, compression="zstd")
            if number % 25 == 0 or number == len(paths):
                print(
                    f"Converted numeric files {number}/{len(paths)}; frames {report['frames']}",
                    flush=True,
                )
        report["episodes"] = len(episodes)
        if (report["frames"], len(episodes)) != (info["total_frames"], info["total_episodes"]):
            raise ValueError("Frame/episode counts differ from source metadata")
        for ep, indices in frame_indices.items():
            if indices != list(range(len(indices))):
                raise ValueError(f"Episode {ep}: missing/reordered frame indices")
        full = {k: np.concatenate(v) for k, v in arrays.items()}
        stats = json.loads((source / "meta/stats.json").read_text())
        stats = {renames.get(k, k): v for k, v in stats.items()}
        for key, value in full.items():
            stats[key] = statistics(value)
        for key in ("observation.state", "observation.model_state_unclipped", "action"):
            value = full[key]
            report["bounds"][key] = {}
            for index, name in conversion.arms:
                low = 0 if name.endswith("_gripper") else -100
                v = value[:, index]
                report["bounds"][key][name] = {
                    "min": float(v.min()),
                    "max": float(v.max()),
                    "below": int(np.count_nonzero(v < low - 1e-5)),
                    "above": int(np.count_nonzero(v > 100 + 1e-5)),
                }
        episode_paths = sorted(source.glob("meta/episodes/**/*.parquet"))
        seen = set()
        for path in episode_paths:
            report["source_hashes"][str(path.relative_to(source))] = file_digest(path)
            table = pq.read_table(path)
            rows = table.to_pylist()
            for row in rows:
                ep = row["episode_index"]
                if ep in seen or row["length"] != len(frame_indices[ep]):
                    raise ValueError(f"Episode {ep}: duplicate/mismatched metadata")
                seen.add(ep)
                for key in tuple(row):
                    if key.startswith("stats/"):
                        feature, stat = key[6:].rsplit("/", 1)
                        renamed = renames.get(feature, feature)
                        if renamed != feature:
                            row[f"stats/{renamed}/{stat}"] = row.pop(key)
                for key, values_array in episodes[ep].items():
                    for stat, result in statistics(np.concatenate(values_array)).items():
                        row[f"stats/{key}/{stat}"] = result
            if staging:
                target = staging / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(rows), target, compression="zstd")
        if seen != set(episodes):
            raise ValueError("Missing episode metadata")
        for path in (source / "meta").glob("*.json"):
            report["source_hashes"][str(path.relative_to(source))] = file_digest(path)
        if staging:
            # Preserve historical annotations separately. Never attach old encoders to new units.
            originals = staging / "meta/source_metadata"
            originals.mkdir(parents=True)
            for path in (source / "meta").glob("*.json"):
                shutil.copy2(path, originals / path.name)
            shutil.copy2(source / "meta/tasks.parquet", staging / "meta/tasks.parquet")
            for directory in ("videos", "images"):
                for path in sorted((source / directory).rglob("*")):
                    if path.is_symlink():
                        raise ValueError(f"Source media must not be symbolic links: {path}")
                    if path.is_file():
                        dest = staging / path.relative_to(source)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        os.link(path, dest)
                        report["media"]["files"] += 1
                        report["media"]["bytes"] += path.stat().st_size
            if annotation:
                for path in sorted(source.glob("meta/safety/*.jsonl")):
                    report["source_hashes"][str(path.relative_to(source))] = file_digest(path)
                    original = staging / "meta/source_safety" / path.name
                    original.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, original)
                    converted_safety(path, staging / path.relative_to(source), conversion, metadata)
            source_info = (
                deepcopy(annotation["source_info"])
                if annotation
                else {
                    "format": "alohamini-coordinate-import",
                    "version": 1,
                    "fps": info["fps"],
                    "capture_metadata": "not_recorded_in_source",
                }
            )
            source_info.update(robot_metadata=metadata, features=features)
            if source_info.get("fixed_dimensions"):
                source_info["fixed_dimensions"]["targets"] = conversion.mapping(
                    source_info["fixed_dimensions"]["targets"]
                )
            source_info["coordinate_conversion"] = {
                "method": plan["method"],
                "basis": plan["basis"],
                "source": str(source),
                "reference_quality": (
                    "same_unchanged_encoder_new_home"
                    if plan["method"] == "same_encoder"
                    else "operator_confirmed_home_contacts_and_legacy_manual_zero"
                    if plan["method"] == "legacy_home_contacts"
                    else "assumed_legacy_manual_zero"
                ),
                "state": "Host-normalized feedback; saturation matches current HostPositionUnits",
                "unclipped_state": (
                    "observation.model_state_unclipped retains converted measurements"
                ),
                "action": "unclipped converted submitted targets",
                "lift_height": (
                    "unchanged historical mm and zero; not a recovered cross-robot lift origin"
                ),
            }
            source_info.pop("training_only", None)
            result = deepcopy(annotation) if annotation else {"version": 1}
            result.update(
                source=str(source),
                source_info=source_info,
                action="Coordinate-converted submitted targets; no clipping or resampling",
                feedback_statistics=(
                    "Historical encoder positions retain source-prefixed names; "
                    "velocity, current and other offset-independent feedback are unchanged"
                ),
                training_review="required",
                capture_metadata={
                    "available": bool(annotation),
                    "reason": "retained" if annotation else "not_recorded_in_source",
                },
            )
            report["status"] = "complete"
            report["output"] = str(output)
            report["source_info_sha256"] = digest(source_info)
            write_json(staging / "meta/coordinate_conversion.json", report)
            result["coordinate_conversion_sha256"] = file_digest(
                staging / "meta/coordinate_conversion.json"
            )
            write_json(staging / "meta/alohamini.json", result)
            write_json(staging / "meta/info.json", new_info)
            write_json(staging / "meta/stats.json", stats)
            # Reject a changing source before publishing the completed directory.
            for relative, expected in report["source_hashes"].items():
                if file_digest(source / relative) != expected:
                    raise RuntimeError(f"Source changed during conversion: {relative}")
            staging.rename(output)
            staging = None
        return report
    finally:
        if staging is not None:
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Omit for a numeric preflight without dataset writes"
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.source, json.loads(args.plan.read_text()), output=args.output)
    write_json(args.report, report)
    print(
        f"{report['status']}: {report['episodes']} episodes, "
        f"{report['frames']} frames; {args.report}"
    )


if __name__ == "__main__":
    main()
