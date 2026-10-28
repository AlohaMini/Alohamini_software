"""Opt-in demonstration queries through each policy's deployment processor."""

from __future__ import annotations

import json
import random
from contextlib import contextmanager

import numpy as np

from .source import feature_names, file_signature, unit


def _offsets(windows):
    result = {}
    for key, values in windows.items():
        values = list(values)
        if (
            not values
            or any(type(value) is not int for value in values)
            or values != sorted(set(values))
        ):
            raise ValueError(f"{key}: 探针窗口必须是递增且不重复的整数行偏移")
        result[key] = values
    return result


def _spec(policy):
    from alohamini.policies.registry import algorithm

    manifest = policy.manifest
    adapter = getattr(policy, "algorithm", None) or algorithm(manifest["kind"])
    arguments = {"cameras": manifest["cameras"]}
    training = adapter.sample_spec(manifest["config"], **arguments)
    expected = _offsets(training.get("delta_indices", {}))
    saved = manifest.get("sample_windows")
    if saved is not None and _offsets(saved) != expected:
        raise ValueError("checkpoint 保存的采样窗口与策略声明不一致，无法可靠对齐探针")
    inference = getattr(adapter, "inference_spec", None)
    spec = inference(manifest["config"], **arguments) if inference else training
    windows = _offsets(spec.get("delta_indices", {}))
    actions = windows.get("action")
    if not actions or 0 not in actions or actions != expected.get("action"):
        raise ValueError("策略必须声明包含当前动作、与训练对齐的预测动作窗口")
    for key, values in windows.items():
        if key != "action" and any(value > 0 for value in values):
            raise ValueError(f"{key}: 推理窗口包含未来观测，需要策略声明因果 inference_spec")
        if key == "past_action" and any(value >= 0 for value in values):
            raise ValueError("past_action 只能包含严格过去的动作")
    return windows, bool(spec.get("include_task", adapter.include_task))


class _Inputs:
    def __init__(self, policy, reference, reader, windows, include_task):
        self.policy, self.reference, self.reader = policy, reference, reader
        self.windows, self.include_task = windows, include_task
        # Match training's row-offset windows: only the episode bounds padding.
        # Display segments/time ranges select queries, not new sample boundaries.
        self.start, self.end = 0, len(reference.data.rows)

    def sample(self, index):
        import torch
        from PIL import Image

        from alohamini.learning.processor import image_tensor

        policy, data = self.policy, self.reference.data
        cache, used, padding = {}, {}, {}

        def value(i, key):
            if (i, key) not in cache:
                row = data.rows[i]
                if key.startswith("observation.images."):
                    camera = key.removeprefix("observation.images.")
                    refs = data.media.get(camera)
                    if refs is None or refs[i] is None:
                        raise ValueError(f"示范缺少模型相机：{camera}, frame {row['frame_index']}")
                    with Image.open(self.reader.keyframe(refs[i], camera)) as image:
                        result = image_tensor(
                            np.array(image.convert("RGB")), policy.manifest["image_size"]
                        )
                elif key == "observation.state":
                    result = torch.from_numpy(policy.selection.frame(row))
                else:
                    result = torch.as_tensor(row[key], dtype=torch.float32)
                cache[i, key] = result
            return cache[i, key]

        def field(key, source=None):
            offsets = self.windows.get(key, [0])
            indices = [max(self.start, min(self.end - 1, index + delta)) for delta in offsets]
            used[key] = [data.rows[i]["frame_index"] for i in indices]
            padding[key] = [not self.start <= index + delta < self.end for delta in offsets]
            values = [value(i, source or key) for i in indices]
            return torch.stack(values) if key in self.windows else values[0]

        sample = {}
        if policy.selection:
            sample["observation.state"] = field("observation.state")
        for camera in policy.manifest["cameras"]:
            key = f"observation.images.{camera}"
            sample[key] = field(key)
        if "past_action" in self.windows:
            sample["past_action"] = field("past_action", "action")
            sample["sequence_index"] = torch.tensor(index)
        for key, values in padding.items():
            if key in self.windows:
                sample[f"{key}_is_pad"] = torch.tensor(values, dtype=torch.bool)
        if self.include_task:
            task = data.rows[index].get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError(
                    f"语言策略需要示范原始任务文本，frame {data.rows[index]['frame_index']}；"
                    "请用 load_episode(include_task=True) 读取 task/task_index"
                )
            sample["task"] = task
        return sample, used, padding


def _baseline(data, row, names):
    """Use observed coordinates only; never use an action label as a baseline."""
    result = np.full(len(names), np.nan, dtype=np.float32)
    state = row.get("observation.state")
    if state is None:
        return result
    state_names = feature_names(data, "observation.state", len(state))
    for j, name in enumerate(names):
        if name not in state_names:
            continue
        index = state_names.index(name)
        recorded_unit = unit(data, name)
        state_unit = unit(data, name, column="observation.state")
        if "unverified" in recorded_unit or state_unit != recorded_unit:
            continue
        if isinstance(state[index], (int, float)) and np.isfinite(state[index]):
            result[j] = state[index]
    return result


@contextmanager
def _random_state(torch, device):
    if str(device).startswith("cuda"):
        torch.cuda.init()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def probe_reference(context, reference, reader, output, *, samples, device, seed=0):
    import torch

    from alohamini.learning.policy import NativePolicy

    from .provenance import describe_checkpoint

    if type(samples) is not int or samples < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("探针 samples 必须为正整数，seed 须在 [0, 2^32) 内")
    description = context["checkpoint"]
    current = describe_checkpoint(description["path"], hash_weights=True)
    from pathlib import Path

    checkpoint_files = {
        Path(current["path"]) / name: file_signature(Path(current["path"]) / name)
        for name in ("policy.json", "model.safetensors")
    }
    for key in ("manifest_sha256", "weights_sha256"):
        if description.get(key) and current[key] != description[key]:
            raise ValueError("本机 checkpoint 已改变，与评估关联身份不一致")
    with _random_state(torch, device):
        policy = NativePolicy(description["path"], device=device, offline=True)
        if policy.fps != reference.data.info["fps"]:
            raise ValueError("示范帧率与 checkpoint 的动作窗口不一致")
        names = list(policy.names)
        if feature_names(reference.data, "action", len(names)) != names:
            raise ValueError("示范动作字段或顺序与 checkpoint 不一致")
        if policy.selection:
            columns = getattr(policy.selection, "columns", [("observation.state", 0, None, None)])
            features = policy.manifest["source_info"]["features"]
            for key in {
                field for source, _, _, mask in columns for field in (source, mask) if field
            }:
                expected, actual = features.get(key), reference.data.info["features"].get(key)
                if (
                    expected is None
                    or actual is None
                    or any(
                        expected.get(field) != actual.get(field)
                        for field in ("names", "shape", "units")
                    )
                ):
                    raise ValueError(f"{key}: 示范输入字段、顺序或单位与 checkpoint 不一致")
        windows, include_task = _spec(policy)
        offsets = windows["action"]
        inputs = _Inputs(policy, reference, reader, windows, include_task)
        positions = np.unique(
            np.linspace(0, len(reference.indices) - 1, min(samples, len(reference.indices)))
            .round()
            .astype(int)
        )
        predictions, targets, baselines, frames, rows, seeds = [], [], [], [], [], []
        target_frames, target_rows, observations, paddings, queries = [], [], [], [], []
        for p in positions:
            i = int(reference.indices[p])
            row = reference.data.rows[i]
            query_seed = (seed + int(row["frame_index"])) % 2**32
            random.seed(query_seed)
            np.random.seed(query_seed)
            torch.random.default_generator.manual_seed(query_seed)
            if torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(query_seed)
            policy.reset()
            sample, used, padding = inputs.sample(i)
            prediction = policy.predict(sample).numpy()
            if prediction.shape != (len(offsets), len(names)) or not np.isfinite(prediction).all():
                raise ValueError("模型预测的动作窗口、维度或有限值与声明不一致")
            truth = np.full_like(prediction, np.nan)
            frame_ids, row_ids = [], []
            for step, delta in enumerate(offsets):
                target = i + delta
                inside = inputs.start <= target < inputs.end
                frame_ids.append(reference.data.rows[target]["frame_index"] if inside else -1)
                row_ids.append(target if inside else -1)
                if inside:
                    values = np.asarray(reference.data.rows[target]["action"], dtype=np.float32)
                    if values.shape != (len(names),):
                        raise ValueError("示范动作维度与 checkpoint 不一致")
                    truth[step] = values
            predictions.append(prediction)
            targets.append(truth)
            baselines.append(np.broadcast_to(_baseline(reference.data, row, names), truth.shape))
            frames.append(row["frame_index"])
            rows.append(i)
            seeds.append(query_seed)
            target_frames.append(frame_ids)
            target_rows.append(row_ids)
            observations.append(json.dumps(used, ensure_ascii=False))
            paddings.append(json.dumps(padding, ensure_ascii=False))
            queries.append(
                dict(
                    frame_index=row["frame_index"],
                    row_index=i,
                    seed=query_seed,
                    task=sample.get("task"),
                    observation_frame_indices=used,
                    observation_padding=padding,
                )
            )
    prediction, truth, baseline = map(np.asarray, (predictions, targets, baselines))
    coordinate_valid = np.isfinite(truth) & (np.asarray(offsets)[None, :, None] >= 0)
    valid = coordinate_valid.all(axis=-1)
    fixed = set((policy.fixed_dimensions or {}).get("targets", {}))
    metrics, first = {}, offsets.index(0)

    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None

    for j, name in enumerate(names):
        if name in fixed:
            continue
        mask = coordinate_valid[:, :, j]
        errors = np.abs(prediction[:, :, j] - truth[:, :, j])
        baseline_errors = np.abs(baseline[:, :, j] - truth[:, :, j])
        baseline_mask = mask & np.isfinite(baseline_errors)
        metrics[name] = {
            "unit": reference.summary["actions"][name]["unit"],
            "chunk_mae": mean(errors, mask),
            "first_action_mae": mean(errors[:, first], mask[:, first]),
            "valid_action_steps": int(mask.sum()),
            "persistence_valid_action_steps": int(baseline_mask.sum()),
            "persistence_chunk_mae": mean(baseline_errors, baseline_mask),
            "persistence_first_action_mae": mean(
                baseline_errors[:, first], baseline_mask[:, first]
            ),
        }
    notes = [
        "首步对应行偏移 0；动作段误差只统计非负偏移，排除回合末尾 padding 和历史前缀。",
        "偏移按示范行顺序计数，仅以回合边界 padding；不按真实时间重采样。",
        "保持基线只使用当前同名观测值；状态缺失或单位无法确认时不计算该维基线。",
    ]
    if "past_action" in windows:
        notes.append("本策略以示范中严格过去的动作作为条件；不输入当前或未来动作标签。")
    notes.append("每次查询重置队列和潜变量；不复现连续运行中的融合、缓存或 RTC 行为。")
    result = {
        "schema_version": 2,
        "kind": current["kind"],
        "mode": "independent_chunks_checkpoint_config",
        "checkpoint_identity": {
            key: current[key] for key in ("path", "kind", "manifest_sha256", "weights_sha256")
        },
        "seed": seed,
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "samples": len(frames),
        "frame_indices": frames,
        "device": device,
        "split": context["reference_split"],
        "episode": reference.data.index,
        "action_names": names,
        "action_units": [unit(reference.data, name) for name in names],
        "window": {
            "predicted_steps": len(offsets),
            "executed_steps": min(
                current["config"].get("n_action_steps", len(offsets)),
                sum(delta >= 0 for delta in offsets),
            ),
            "fps": policy.fps,
            "action_offsets": offsets,
            "observation_offsets": {
                key: windows.get(key, [0])
                for key in (
                    (["observation.state"] if policy.selection else [])
                    + [f"observation.images.{camera}" for camera in policy.manifest["cameras"]]
                )
            },
        },
        "history_action_offsets": windows.get("past_action", []),
        "sample_boundary": "episode",
        "reference": {
            "dataset": str(reference.data.root),
            "episode": reference.data.index,
            "segment": reference.summary["segment"],
            "selected_range_s": reference.summary["selected_range_s"],
            "time_basis": reference.summary["time_basis"],
            "source_files": reference.summary["source_files"],
            "verified_reference_files": context.get("verified_reference_files", []),
        },
        "baseline_source": "current_observation_same_named_coordinates",
        "metrics": metrics,
        "queries": queries,
        "notes": notes,
        "excluded_fixed_dimensions": sorted(fixed),
        "artifacts": {
            "predictions_npz": "demonstration_predictions.npz",
            "probe_json": "demonstration_probe.json",
        },
        "interpretation": "在示范观测上按 checkpoint 配置独立预测，与同一回合的动作标签比较。"
        "这不是录制时的原始 Chunk，也不是闭环成功率；训练集误差不代表泛化。",
    }
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    reference.data.unchanged()
    if any(file_signature(path) != signature for path, signature in checkpoint_files.items()):
        raise RuntimeError("探针运行期间 checkpoint 已改变，结果未保存")
    np.savez_compressed(
        output / "demonstration_predictions.npz",
        metadata_json=encoded,
        frame_index=frames,
        row_index=rows,
        sample_seed=seeds,
        prediction=prediction,
        target=truth,
        baseline=baseline,
        valid=valid,
        coordinate_valid=coordinate_valid,
        action_names=names,
        action_units=result["action_units"],
        action_offsets=offsets,
        target_frame_index=target_frames,
        target_row_index=target_rows,
        observation_frame_indices=observations,
        observation_padding=paddings,
    )
    (output / "demonstration_probe.json").write_text(encoded + "\n", encoding="utf-8")
    return result
