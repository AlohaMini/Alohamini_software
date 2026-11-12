"""Portable numeric evidence, plots and optional explicitly selected keyframes."""

from __future__ import annotations

import csv
import html
import json
import math
import shutil

import numpy as np

from .probe import probe_interpretation
from .provenance import display_config


def _probe_metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "不可用"
    return f"{value:.4g}"


def _probe_html(probe, checkpoint, output):
    if not probe:
        return ""
    esc = html.escape

    def text(value):
        return esc(str(value)) if value is not None else "不可用"

    def counts(metric):
        model = text(metric.get("valid_action_steps"))
        if "persistence_valid_action_steps" in metric:
            return f"模型{model} / 基线{text(metric['persistence_valid_action_steps'])}"
        return model

    facts = {
        "算法": probe.get("kind") or checkpoint.get("kind", "未知"),
        "示范划分": probe.get("split"),
        "Episode": probe.get("episode"),
        "采样观测数": probe.get("samples"),
    }
    window = probe.get("window") or {}
    if window:
        facts.update(
            {
                "预测窗口（步）": window.get("predicted_steps"),
                "执行窗口（步）": window.get("executed_steps"),
                "数据集帧率（Hz）": window.get("fps"),
            }
        )
    details = " · ".join(f"{esc(label)}：{text(value)}" for label, value in facts.items())
    metrics = probe["metrics"]
    names = [name for name in probe.get("action_names", []) if name in metrics]
    names.extend(name for name in metrics if name not in names)
    fields = (
        "first_action_mae",
        "chunk_mae",
        "persistence_first_action_mae",
        "persistence_chunk_mae",
    )
    rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{text(metrics[name].get('unit'))}</td>"
        + "".join(f"<td>{_probe_metric(metrics[name].get(field))}</td>" for field in fields)
        + f"<td>{counts(metrics[name])}</td></tr>"
        for name in names
    )
    defaults = {
        "predictions_npz": "demonstration_predictions.npz",
        "probe_json": "demonstration_probe.json",
    }
    artifacts = probe.get("artifacts") or defaults
    links = []
    root = output.resolve()
    for key, label in (("predictions_npz", "预测与标签 NPZ"), ("probe_json", "探针结果 JSON")):
        relative = artifacts.get(key)
        if not isinstance(relative, str):
            continue
        path = (output / relative).resolve()
        if path.is_relative_to(root) and path.is_file():
            links.append(
                f'<a href="{esc(path.relative_to(root).as_posix(), quote=True)}">{label}</a>'
            )
    downloads = f"<p>{' · '.join(links)}</p>" if links else ""
    excluded = probe.get("excluded_fixed_dimensions") or []
    fixed = f"<p>不计入误差的固定维度：{esc('、'.join(excluded))}</p>" if excluded else ""
    notes = "".join(f"<li>{esc(note)}</li>" for note in probe.get("notes", []))
    methods = (
        "<details><summary>计算方法与适用范围</summary>"
        "<p>MAE 沿用各动作的记录单位，各维度分别比较，不跨单位合并。"
        "首步对应当前动作；动作段统计当前及未来的有效标签，历史前缀与无效尾部不计入。"
        "表中的“不可用”表示缺少对应数据。</p>"
        f"{fixed}" + (f"<ul>{notes}</ul>" if notes else "") + "</details>"
    )
    # Compare only metrics with matching recorded denominators. Older artifacts
    # may lack baseline counts, so retain their table without inventing a ranking.
    comparable = [
        metrics[name]
        for name in names
        if type(metrics[name].get("valid_action_steps")) is int
        and metrics[name]["valid_action_steps"] > 0
        and metrics[name].get("valid_action_steps")
        == metrics[name].get("persistence_valid_action_steps")
        and all(
            _probe_metric(metrics[name].get(field)) != "不可用"
            for field in ("chunk_mae", "persistence_chunk_mae")
        )
    ]
    findings = ""
    if comparable:
        better = sum(m["chunk_mae"] < m["persistence_chunk_mae"] for m in comparable)
        findings = (
            f"<p>在统计范围一致的 {len(comparable)} 个动作维度中，"
            f"有 {better} 个维度的动作段 MAE 低于保持当前位置的基线。"
            "可结合下表查看改善集中在哪些关节，以及哪些维度仍接近或高于基线。</p>"
        )
    return (
        "<h2>模型在示范上的预测误差</h2>"
        f"<p>{details}</p><p>{esc(probe_interpretation(probe.get('split')))}</p>"
        "<p>首步 MAE 反映模型从当前状态开始时的预测偏差，动作段 MAE 则覆盖后续整段动作。"
        "两者一起看，有助于区分起始动作的偏差与后续轨迹的偏差。"
        "保持当前位置的基线将当前观测位置重复到整个窗口，用来判断模型预测后续运动带来的改善。</p>"
        f"{findings}"
        "<table><tr><th>动作</th><th>单位</th><th>模型首步 MAE</th><th>模型动作段 MAE</th>"
        "<th>保持当前动作首步 MAE</th><th>保持当前动作段 MAE</th>"
        "<th>有效步数（模型 / 基线）</th></tr>"
        f"{rows}</table>{downloads}{methods}"
    )


def write_summary(analysis, output):
    (output / "summary.json").write_text(
        json.dumps(analysis.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "events.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["time_s", "frame_index", "kind", "field", "value", "evidence", "detail"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for event in analysis.events:
            writer.writerow({**event, "value": json.dumps(event["value"], ensure_ascii=False)})


def export_html(analysis, output, reader, *, keyframe_times=(), cameras=None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    cameras = list(analysis.data.media) if cameras is None else cameras
    figures, keyframes = [], []
    plots = output / "plots"
    plots.mkdir()
    groups = {}
    for path in analysis.series:
        if path.startswith("actions/"):
            name = path.split("/", 1)[1]
            group = (
                "arm_left"
                if name.startswith("arm_left_")
                else "arm_right"
                if name.startswith("arm_right_")
                else "other"
            )
            groups.setdefault(group, []).append(path)
    for group, paths in groups.items():
        fig, axes = plt.subplots(
            len(paths), 1, figsize=(13, 2.1 * len(paths)), sharex=True, squeeze=False
        )
        for ax, path in zip(axes[:, 0], paths, strict=True):
            for label, values in analysis.series[path].items():
                # Each stored point is measured; no claim of intermediate control samples.
                ax.plot(
                    analysis.times - analysis.anchor_s,
                    values,
                    label="evaluation / " + label,
                    linewidth=0.8,
                )
            reference = analysis.reference
            comparable = analysis.summary.get("comparison", {}).get("comparable_fields", [])
            if reference is not None and path.split("/", 1)[1] in comparable:
                ax.plot(
                    reference.times - reference.anchor_s,
                    reference.series[path]["recorded_action"],
                    label="demonstration",
                    linewidth=0.8,
                    alpha=0.8,
                )
            name = path.split("/", 1)[1]
            ax.set_ylabel(analysis.summary["actions"][name]["unit"], fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.2)
            for event in analysis.events:
                if event["kind"] == "inferred_chunk_transition":
                    ax.axvline(
                        event["time_s"] - analysis.anchor_s,
                        color="tab:red",
                        alpha=0.2,
                        linewidth=0.7,
                    )
        axes[0, 0].legend(fontsize=7, loc="upper right")
        axes[-1, 0].set_xlabel("Time relative to alignment anchor (s)")
        fig.tight_layout()
        relative = f"plots/{group}.png"
        fig.savefig(output / relative, dpi=120)
        plt.close(fig)
        figures.append(relative)
    runs = [("evaluation", analysis)]
    if analysis.reference is not None:
        runs.append(("reference", analysis.reference))
    for requested in keyframe_times:
        for label, run in runs:
            aligned = requested - analysis.anchor_s + run.anchor_s
            if not run.times[0] <= aligned <= run.times[-1]:
                analysis.summary["warnings"].append(
                    f"{label}: 对应 {requested}s 的画面超出所选片段"
                )
                continue
            p = int(np.argmin(abs(run.times - aligned)))
            i = run.indices[p]
            frame = run.data.rows[i]["frame_index"]
            for camera in cameras:
                if camera not in run.data.media:
                    continue
                ref = run.data.media[camera][i]
                try:
                    if ref is None:
                        raise ValueError("图像引用缺失")
                    cached = reader.keyframe(ref, camera)
                    relative = f"keyframes/{label}_{frame:06d}_camera_{cameras.index(camera)}.png"
                    (output / "keyframes").mkdir(exist_ok=True)
                    shutil.copyfile(cached, output / relative)
                    keyframes.append(
                        dict(
                            run=label,
                            camera=camera,
                            requested_time_s=requested,
                            time_s=float(run.times[p]),
                            frame_index=frame,
                            path=relative,
                        )
                    )
                except (OSError, ValueError, RuntimeError) as exc:
                    analysis.summary["warnings"].append(f"关键帧 {label}/{camera}/{frame}: {exc}")
    analysis.summary["keyframes"] = keyframes
    write_html(analysis, output, figures)


def write_html(analysis, output, figures):
    """Render existing numerical results and media without rerunning a policy."""
    s = analysis.summary
    keyframes = s.get("keyframes", [])
    esc = html.escape
    facts = {
        "数据集": s["dataset"],
        "Episode / 时间段": f"{s['episode']} / {s['segment']}",
        "时间依据": s["time_basis"],
        "选中帧数": s["selected_rows"],
        "选中时长 (s)": round(s["duration_s"], 3),
        "录制频率 (Hz)": s["recorded_frame_hz"],
        "本次推理参数": json.dumps(
            display_config(s.get("effective_policy_config")), ensure_ascii=False
        ),
        "固定目标": json.dumps(s["fixed_dimensions"], ensure_ascii=False),
        "Chunk 推断前提": json.dumps(s["chunk_assumption"], ensure_ascii=False),
    }
    context = s.get("model_context", {})
    checkpoint = context.get("checkpoint") or {}
    comparison = s.get("comparison", {})
    facts.update(
        {
            "Checkpoint": checkpoint.get("path", "未关联"),
            "算法": checkpoint.get("kind", "未知"),
            "Checkpoint 默认参数": json.dumps(
                display_config(checkpoint.get("config")), ensure_ascii=False
            ),
            "模型相机顺序": checkpoint.get("cameras"),
            "Checkpoint 来源": context.get("checkpoint_source", "未知"),
            "训练数据集": context.get("training_root"),
            "参考示范": (
                f"{comparison.get('split', '')} / episode {comparison.get('episode', '不可用')}"
            ),
            "对齐方式": comparison.get("alignment", "无参考示范"),
            "对齐锚点 (评估/示范秒数)": comparison.get("anchors_s"),
        }
    )
    if s.get("control_diagnostics", True):
        facts["命令频率估计 (Hz)"] = s.get("command_hz_estimate")
    fact_html = "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in facts.items()
    )
    warnings = "".join(f"<li>{esc(v)}</li>" for v in s["warnings"])
    warnings = f"<aside><strong>数据提示</strong><ul>{warnings}</ul></aside>" if warnings else ""
    notes = "".join(f"<li>{esc(v)}</li>" for v in s["interpretation"])
    notes = (
        f"<details><summary>曲线与控制记录的阅读说明</summary><ul>{notes}</ul></details>"
        if notes
        else ""
    )
    if comparison:
        alignment = (
            "曲线按所选任务阶段的时间锚点对齐，可以对照接近、抓取和搬运时的动作顺序与幅度。"
            if comparison.get("alignment") == "user_stage_anchors"
            else "曲线按两个回合的起点并排展示，可先观察动作顺序与整体范围。"
            "需要细看某次抓取时，可用相应阶段的时间锚点对齐两条轨迹。"
        )
    else:
        alignment = "曲线展示本回合各关节的动作变化，可结合下方关键帧定位接近、抓取和搬运阶段。"
    if s.get("chunk_assumption"):
        alignment += "红线标出按所提供执行步数和队列假设推算的动作块交接位置。"
    changes = [e for e in analysis.events if e["kind"] != "recording_gap"]
    event_html = "".join(
        f"<tr><td>{e['time_s']:.3f}</td><td>{e['frame_index']}</td>"
        f"<td>{esc(e['kind'])}</td><td>{esc(e['field'])}</td>"
        f"<td>{esc(str(e['value']))}</td><td>{esc(e['detail'])}</td></tr>"
        for e in changes
    )
    images = "".join(f'<figure><img src="{p}" alt="Action timeline"></figure>' for p in figures)
    frames_html = "".join(
        f'<figure><img src="{f["path"]}" alt="{esc(f["camera"])}">'
        f"<figcaption>{esc(f['run'])} / {esc(f['camera'])} · {f['time_s']:.3f}s "
        f"· frame {f['frame_index']}"
        "</figcaption></figure>"
        for f in keyframes
    )
    probe_html = _probe_html(s.get("demonstration_probe"), checkpoint, output)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>评估记录分析 · {esc(analysis.data.root.name)}</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #dbe1e7;
padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#f4f7fa}}img{{max-width:100%}}
figure{{margin:20px 0}}.frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
li{{margin:8px 0}}a{{color:#175e9c}}details{{margin:20px 0;padding:12px;background:#f4f7fa}}
summary{{cursor:pointer}}aside{{padding:12px 18px;background:#fff8e8}}</style>
<h1>评估记录分析</h1><p>{esc(analysis.data.root.name)} · Episode {s["episode"]}</p>
<table>{fact_html}</table>{warnings}
<p><a href="summary.json">完整数值摘要</a> · <a href="events.csv">全部事件 CSV</a></p>
<h2>评估动作与参考示范</h2><p>{alignment}</p>{images}{notes}
{probe_html}<h2>评估 / 示范关键帧</h2><div class="frames">{frames_html}</div>
<h2>事件与逐维最大变化</h2><p>表格列出各动作维度变化最大的时刻及控制事件，便于回到相邻画面查看当时的运动阶段。</p>
<table><tr><th>秒</th><th>帧</th><th>事件</th>
<th>字段</th><th>值</th><th>说明</th></tr>{event_html}</table></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
