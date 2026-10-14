"""Demonstration comparisons use named coordinates and explicit stage anchors."""

from __future__ import annotations

import numpy as np

from .metrics import Analysis


def prepare(analysis: Analysis, context: dict, *, reference=None, anchors=None, diagnostics=False):
    analysis.summary["model_context"] = context
    analysis.summary["effective_policy_config"] = context.get("effective_policy_config")
    analysis.summary["warnings"].extend(context["warnings"])
    analysis.summary["control_diagnostics"] = diagnostics
    if not diagnostics:
        analysis.series = {
            key: {"recorded_action": value["recorded_action"]}
            for key, value in analysis.series.items()
            if key.startswith("actions/")
        }
        analysis.events = [
            e
            for e in analysis.events
            if e["kind"] in ("largest_recorded_action_change", "inferred_chunk_transition")
        ]
        analysis.summary["interpretation"] = [
            "评估 action 是模型执行时请求的动作，示范 action 是训练数据中的动作标签。",
            "不同运行的场景与进度可能不同；曲线并排不等于逐帧预测误差。",
            "默认展示 checkpoint 划分中的首条示范，未自动检索相似场景；可选择其他 episode。",
            "未标注任务成功或失败；动作变化排名不能直接判定任务成败。",
        ]
        for key in (
            "command_hz_estimate",
            "command_hz_assumption",
            "select_action_ms",
            "camera_skew_ms",
        ):
            analysis.summary.pop(key, None)
    analysis.reference = reference
    analysis.anchor_s = float(anchors[0]) if anchors else 0.0
    if reference is None:
        if anchors:
            raise ValueError("--align 需要可用的参考示范")
        return
    reference.anchor_s = float(anchors[1]) if anchors else 0.0
    for run in (analysis, reference):
        if anchors and (
            not np.isfinite(run.anchor_s) or not run.times[0] <= run.anchor_s <= run.times[-1]
        ):
            raise ValueError("对齐锚点必须在各自选中的时间范围内")
    reference.series = {
        key: {"recorded_action": value["recorded_action"]}
        for key, value in reference.series.items()
        if key.startswith("actions/")
    }
    common, incompatible = [], []
    for name, values in analysis.summary["actions"].items():
        other = reference.summary["actions"].get(name)
        if other is None or values["unit"] != other["unit"] or "unverified" in values["unit"]:
            incompatible.append(name)
        else:
            common.append(name)
    analysis.summary["comparison"] = {
        "dataset": str(reference.data.root),
        "episode": reference.data.index,
        "split": context["reference_split"],
        "reference_selection": "explicit_episode_or_first_in_split",
        "alignment": "user_stage_anchors" if anchors else "episode_start_only_not_stage_aligned",
        "anchors_s": [analysis.anchor_s, reference.anchor_s],
        "comparable_fields": common,
        "unverified_or_incompatible_fields": incompatible,
        "reference_actions": reference.summary["actions"],
        "missing_evaluation_cameras": [
            c for c in reference.data.media if c not in analysis.data.media
        ],
        "missing_reference_cameras": [
            c for c in analysis.data.media if c not in reference.data.media
        ],
        "pointwise_error": None,
    }
    if incompatible:
        analysis.summary["warnings"].append(
            "部分动作单位或字段不能确认一致；仅单独展示，不计算差值。"
        )
