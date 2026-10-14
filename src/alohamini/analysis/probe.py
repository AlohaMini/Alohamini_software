"""Opt-in ACT chunk predictions on known demonstrations, using deployment preprocessing."""

from __future__ import annotations

import random

import numpy as np


def probe_reference(context, reference, reader, output, *, samples, device, seed=0):
    import torch
    from PIL import Image

    from alohamini.learning.policy import NativePolicy
    from alohamini.learning.processor import image_tensor

    description = context["checkpoint"]
    from .provenance import describe_checkpoint

    current = describe_checkpoint(
        description["path"], hash_weights=bool(description.get("weights_sha256"))
    )
    for key in ("manifest_sha256", "weights_sha256"):
        if description.get(key) and current[key] != description[key]:
            raise ValueError("本机 checkpoint 已改变，与评估关联身份不一致")
    if description["kind"] not in ("act", "am_act"):
        raise ValueError("示范 Chunk 探针目前支持 ACT/AM-ACT；其他算法仍可查看示范对比")
    policy = NativePolicy(description["path"], device=device, offline=True)
    if policy.fps != reference.data.info["fps"]:
        raise ValueError("示范帧率与 checkpoint 的动作窗口不一致")
    # Independent observation-conditioned chunks. No demonstration actions enter the model.
    positions = np.unique(
        np.linspace(0, len(reference.indices) - 1, min(samples, len(reference.indices)))
        .round()
        .astype(int)
    )
    rows = reference.data.rows
    by_frame = {r["frame_index"]: r for r in rows}
    predictions, targets, masks, frames = [], [], [], []
    for p in positions:
        i = reference.indices[p]
        row = rows[i]
        random.seed(seed + int(row["frame_index"]))
        np.random.seed((seed + int(row["frame_index"])) % 2**32)
        torch.manual_seed(seed + int(row["frame_index"]))
        policy.reset()
        sample = {}
        if policy.selection:
            sample["observation.state"] = torch.from_numpy(policy.selection.frame(row))
        for camera in policy.manifest["cameras"]:
            if camera not in reference.data.media or reference.data.media[camera][i] is None:
                raise ValueError(f"示范缺少模型相机：{camera}, frame {row['frame_index']}")
            path = reader.keyframe(reference.data.media[camera][i], camera)
            with Image.open(path) as image:
                sample[f"observation.images.{camera}"] = image_tensor(
                    np.array(image.convert("RGB")), policy.manifest["image_size"]
                )
        prediction = policy.predict(sample).numpy()
        if prediction.ndim != 2 or prediction.shape[1] != len(policy.names):
            raise ValueError("Chunk 探针返回了意外的动作维度")
        if not np.isfinite(prediction).all():
            raise ValueError("模型预测包含非有限动作，不能作为有效预测统计")
        truth = np.full_like(prediction, np.nan)
        mask = np.zeros(len(prediction), dtype=bool)
        for step in range(len(prediction)):
            target = by_frame.get(row["frame_index"] + step)
            if target is not None:
                truth[step] = target["action"]
                mask[step] = np.isfinite(truth[step]).all()
        predictions.append(prediction)
        targets.append(truth)
        masks.append(mask)
        frames.append(row["frame_index"])
    prediction, truth, mask = np.asarray(predictions), np.asarray(targets), np.asarray(masks)
    fixed = set((policy.fixed_dimensions or {}).get("targets", {}))
    metrics = {}
    for j, name in enumerate(policy.names):
        if name in fixed:
            continue
        errors = np.abs(prediction[:, :, j] - truth[:, :, j])
        valid = mask & np.isfinite(errors)
        first = errors[:, 0][valid[:, 0]]
        metrics[name] = {
            "unit": reference.summary["actions"][name]["unit"],
            "chunk_mae": float(errors[valid].mean()) if valid.any() else None,
            "first_action_mae": float(first.mean()) if len(first) else None,
            "valid_action_steps": int(valid.sum()),
        }
    np.savez_compressed(
        output / "demonstration_predictions.npz",
        frame_index=frames,
        prediction=prediction,
        target=truth,
        valid=mask,
        action_names=policy.names,
    )
    return {
        "mode": "independent_prior_chunks_checkpoint_config",
        "seed": seed,
        "samples": len(frames),
        "frame_indices": frames,
        "device": device,
        "split": context["reference_split"],
        "episode": reference.data.index,
        "metrics": metrics,
        "excluded_fixed_dimensions": sorted(fixed),
        "interpretation": "在示范观测上重新预测，按同一 episode 的未来动作监督；"
        "每次重置队列和潜变量。"
        "这不是录制时的原始 Chunk，也不是闭环成功率；训练集误差不代表泛化。",
    }
