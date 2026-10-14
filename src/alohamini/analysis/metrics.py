"""Clock-aware summaries. Observed discontinuities are evidence, not causal diagnoses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .source import Episode, feature_names, numeric, unit


@dataclass
class Analysis:
    data: Episode
    indices: np.ndarray
    times: np.ndarray
    summary: dict
    events: list[dict]
    series: dict[str, dict[str, np.ndarray]]
    reference: Analysis | None = None
    anchor_s: float = 0.0


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def command_key(command):
    if not isinstance(command, dict) or not finite(command.get("sequence")):
        return None
    fields = ("host_session_id", "control_epoch", "client_id")
    if any(command.get(k) is None for k in fields):
        return None
    return tuple(command[k] for k in fields)


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def timeline(data):
    host = [s.get("host_timing", {}).get("state_sample_monotonic_s") for s in data.sidecars]
    warnings = []
    if all(finite(v) for v in host):
        raw = np.asarray(host)
        basis = "host_state_monotonic"
        clocks = [
            s.get("host_timing", {}).get("clock_id") or s.get("safety", {}).get("host_session_id")
            for s in data.sidecars
        ]
    else:
        raw = [r.get("timestamp") for r in data.rows]
        if all(finite(v) for v in raw):
            basis = "dataset_timestamp"
        else:
            fps = data.info.get("fps")
            if not finite(fps) or fps <= 0:
                raise ValueError("缺少有效时间戳和 fps")
            raw = [r["frame_index"] / fps for r in data.rows]
            basis = "nominal_frame_index/fps"
        raw = np.asarray(raw)
        clocks = [None] * len(raw)
        warnings.append("Host 时间不完整，使用数据集时间；不能据此确认真实运行速度。")
    segments = np.zeros(len(raw), dtype=int)
    for i in range(1, len(raw)):
        segments[i] = segments[i - 1] + int(raw[i] < raw[i - 1] or clocks[i] != clocks[i - 1])
    repeated = int(np.sum(np.diff(raw) == 0))
    if repeated:
        warnings.append(f"{repeated} 对相邻记录复用同一观测时间；保留原时间，帧索引可区分。")
    if segments[-1]:
        warnings.append(f"发现时间重置或时钟变化，拆成 {segments[-1] + 1} 段；用 --segment 选择。")
    return raw, segments, basis, warnings


def analyze(
    data: Episode,
    *,
    segment=0,
    time_range=None,
    assumed_nas=None,
    first_command=None,
    no_resets=False,
) -> Analysis:
    raw, segments, basis, warnings = timeline(data)
    indices = np.flatnonzero(segments == segment)
    if not len(indices):
        raise ValueError(f"时间段 {segment} 不存在；可选 0..{segments[-1]}")
    origin = raw[indices[0]]
    times = raw[indices] - origin
    if time_range is not None:
        start, end = time_range
        if not finite(start) or not finite(end) or not 0 <= start < end:
            raise ValueError("--time-range 要求 0 <= 起点 < 终点，单位秒")
        keep = (times >= start) & (times <= end)
        indices, times = indices[keep], times[keep]
    if not len(indices):
        raise ValueError("所选时间范围没有记录帧")
    sides = [data.sidecars[i] for i in indices]
    frames = np.array([data.rows[i]["frame_index"] for i in indices])
    commands = [s.get("issued_command", {}) for s in sides]
    sequences = np.array([c.get("sequence", np.nan) for c in commands], dtype=float)
    keys = [command_key(c) for c in commands]
    contiguous_session = np.array(
        [keys[i] is not None and keys[i] == keys[i - 1] for i in range(1, len(keys))],
        dtype=bool,
    )
    sequence_delta = np.diff(sequences)
    comparable = contiguous_session & (sequence_delta > 0)
    delta_time = np.diff(times)
    events, series = [], {}

    def event(i, kind, field="", value=None, evidence="recorded", detail=""):
        events.append(
            dict(
                time_s=float(times[i]),
                frame_index=int(frames[i]),
                kind=kind,
                field=field,
                value=value,
                evidence=evidence,
                detail=detail,
            )
        )

    if assumed_nas is not None:
        if assumed_nas < 1 or first_command is None or not no_resets:
            raise ValueError(
                "推断 Chunk 需同时提供正整数 --assume-n-action-steps、"
                "--first-command-sequence 和 --assume-no-resets"
            )
        if not all(comparable) or not np.all(sequences >= first_command):
            raise ValueError("命令身份缺失、变化或序号不递增，不能按给定前提推断 Chunk")
        chunk = np.floor_divide(sequences - first_command, assumed_nas)
        for i in np.flatnonzero(np.diff(chunk) > 0) + 1:
            event(
                i,
                "inferred_chunk_transition",
                value=int(chunk[i]),
                evidence="user_assumption",
                detail="边界后首个录制帧；假设关闭融合、每 tick 发一个命令、全程无队列重置",
            )
    for i in np.flatnonzero(comparable & (sequence_delta > 1)) + 1:
        event(
            i,
            "recording_gap",
            value=int(sequence_delta[i - 1] - 1),
            detail="两个录制帧之间的命令序号缺口；不等于机器人未收到命令",
        )
    for i in np.flatnonzero(np.diff(frames) > 1) + 1:
        event(i, "frame_index_gap", value=int(frames[i] - frames[i - 1] - 1))
    commands_known = np.all(np.isfinite(sequences)) and all(comparable) and len(times) > 1
    command_hz = None
    if commands_known and basis == "host_state_monotonic":
        command_hz = float((sequences[-1] - sequences[0]) / (times[-1] - times[0]))
    latency = np.full(len(sides), np.nan)
    camera_skew = np.full(len(sides), np.nan)
    for i, side in enumerate(sides):
        timing = side.get("client_timing", {})
        a, b = (
            timing.get("action_sample_started_monotonic_s"),
            timing.get("action_sample_finished_monotonic_s"),
        )
        if finite(a) and finite(b) and b >= a:
            latency[i] = (b - a) * 1000
        captures = side.get("host_timing", {}).get("camera_capture_monotonic_s", {})
        if len(captures) > 1 and all(finite(v) for v in captures.values()):
            camera_skew[i] = (max(captures.values()) - min(captures.values())) * 1000
        safety = side.get("safety", {})
        previous = sides[i - 1].get("safety", {}) if i else {}
        for name in (
            "fault",
            "phase",
            "watchdog_active",
            "feedback_valid",
            "joint_holds",
            "gripper_holds",
            "last_command_error",
        ):
            if name in safety and safety.get(name) != previous.get(name):
                event(i, "safety_state", field=name, value=safety[name])
        for name in ("watchdog_events", "rejected_commands", "joint_hold_events"):
            a, b = previous.get(name), safety.get(name)
            same_host = (
                i > 0
                and safety.get("host_session_id") is not None
                and safety.get("host_session_id") == previous.get("host_session_id")
            )
            if same_host and finite(a) and finite(b) and b != a:
                event(
                    i,
                    "counter_increment" if b > a else "counter_reset",
                    name,
                    b - a,
                    detail=f"{a} → {b}；首帧累计值不计作本次新增",
                )
    series["timing/select_action_ms"] = {"value": latency}
    series["timing/camera_skew_ms"] = {"value": camera_skew}
    for camera in data.media:
        offsets = []
        for side in sides:
            timing = side.get("host_timing", {})
            capture = timing.get("camera_capture_monotonic_s", {}).get(camera)
            state_sample = timing.get("state_sample_monotonic_s")
            offsets.append(
                (state_sample - capture) * 1000
                if finite(state_sample) and finite(capture)
                else np.nan
            )
        series[f"camera_state_offset_ms/{camera}"] = {"value": np.asarray(offsets)}
    series["recording/command_sequence"] = {"value": sequences}
    series["recording/interval_s"] = {"value": np.r_[np.nan, delta_time]}
    action, state = numeric(data, "action"), numeric(data, "observation.state")
    metrics = {}
    state_names = (
        feature_names(data, "observation.state", state.shape[1]) if state is not None else []
    )
    if action is not None:
        names = feature_names(data, "action", action.shape[1])
        for j, name in enumerate(names):
            target = action[indices, j]
            difference = np.abs(np.diff(target))
            metrics[name] = {
                "unit": unit(data, name),
                "target": distribution(target),
                "adjacent_recorded_change": distribution(difference),
            }
            values = {"recorded_action": target}
            if name in state_names:
                values["observed_before_request"] = state[indices, state_names.index(name)]
            for field, path in (("host_accepted_target", "accepted_targets"),):
                accepted = np.array(
                    [s.get("safety", {}).get(path, {}).get(name, np.nan) for s in sides],
                    dtype=float,
                )
                if np.isfinite(accepted).any():
                    values[field] = accepted
            series[f"actions/{name}"] = values
            order = np.argsort(np.where(np.isfinite(difference), difference, -np.inf))[-3:][::-1]
            for previous in order:
                if not finite(difference[previous]) or difference[previous] <= 0:
                    continue
                i = previous + 1
                gap = int(sequence_delta[previous]) if comparable[previous] else None
                event(
                    i,
                    "largest_recorded_action_change",
                    name,
                    float(difference[previous]),
                    detail=f"unit={unit(data, name)}; dt={delta_time[previous]:.6f}s; "
                    f"command_interval={gap}; 每维前三大变化，不是故障判定",
                )
    currents = numeric(data, "observation.motor_current_ma")
    validity = numeric(data, "motor_feedback.current_ma_valid")
    if currents is not None:
        currents = currents[indices].copy()
        if validity is not None and validity.shape[1] == currents.shape[1]:
            currents[validity[indices] != 1] = np.nan
        for j, name in enumerate(
            feature_names(data, "observation.motor_current_ma", currents.shape[1])
        ):
            series[f"currents_ma/{name}"] = {"value": currents[:, j]}
    notes = [
        "action 与同帧 state 的差不是执行误差：state 在本次请求发出前采样。",
        "Host accepted target 属于反馈中的旧命令；这里只展示，不与新请求冒充配对。",
        "select_action 耗时包含预处理/队列取动作，不能当成每帧模型推理耗时。",
        "policy_action 可能已包含固定覆盖；此记录没有覆盖前网络输出或完整 Chunk。",
        "未提供任务成功标签；动作变化排名不判断任务成败或失败原因。",
    ]
    duration = float(times[-1] - times[0])
    fixed = next((s["fixed_dimensions"] for s in sides if "fixed_dimensions" in s), None)
    summary = dict(
        schema_version=1,
        dataset=str(data.root),
        episode=data.index,
        segment=int(segment),
        segment_count=int(segments[-1]) + 1,
        time_basis=basis,
        time_origin=float(origin),
        selected_range_s=[float(times[0]), float(times[-1])],
        episode_rows=len(data.rows),
        selected_rows=len(indices),
        duration_s=duration,
        recorded_frame_hz=(len(times) - 1) / duration if duration else None,
        command_hz_estimate=command_hz,
        command_hz_assumption="同一命令会话内序号每次发送递增 1；用观测时间近似发送时间",
        model_inference_hz=None,
        effective_policy_config="not_recorded",
        fixed_dimensions=fixed,
        cameras=list(data.media),
        select_action_ms=distribution(latency),
        camera_skew_ms=distribution(camera_skew),
        actions=metrics,
        warnings=data.warnings + warnings,
        interpretation=notes,
        chunk_assumption=(
            {
                "n_action_steps": assumed_nas,
                "first_sequence": first_command,
                "no_resets": True,
                "temporal_ensemble": "assumed_disabled",
            }
            if assumed_nas
            else None
        ),
        source_files=[
            {"path": str(p), "size": sig[0], "mtime_ns": sig[1]}
            for p, sig in data.signatures.items()
        ],
    )
    events.sort(key=lambda e: (e["time_s"], e["kind"], e["field"]))
    return Analysis(data, indices, times, summary, events, series)
