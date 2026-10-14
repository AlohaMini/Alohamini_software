"""Portable numeric evidence, plots and optional explicitly selected keyframes."""

from __future__ import annotations

import csv
import html
import json
import shutil

import numpy as np

from .provenance import display_config


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
        axes[-1, 0].set_xlabel("Seconds relative to anchor (not frame-wise ground truth)")
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
    s = analysis.summary
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
    notes = "".join(f"<li>{esc(v)}</li>" for v in s["warnings"] + s["interpretation"])
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
    probe = s.get("demonstration_probe")
    probe_html = ""
    if probe:
        probe_rows = "".join(
            f"<tr><td>{esc(name)}</td><td>{esc(m['unit'])}</td>"
            f"<td>{m['first_action_mae']}</td><td>{m['chunk_mae']}</td></tr>"
            for name, m in probe["metrics"].items()
        )
        probe_html = (
            f"<h2>模型在示范上的预测误差</h2><p>{esc(probe['interpretation'])}</p>"
            f"<p>{esc(probe['split'])} / episode {probe['episode']} · {probe['samples']} 个观测</p>"
            "<table><tr><th>动作</th><th>单位</th><th>首步 MAE</th><th>Chunk MAE</th></tr>"
            f"{probe_rows}</table>"
        )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>评估记录分析 · {esc(analysis.data.root.name)}</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #dbe1e7;
padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#f4f7fa}}img{{max-width:100%}}
figure{{margin:20px 0}}.frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
li{{margin:8px 0}}a{{color:#175e9c}}</style>
<h1>评估记录分析</h1><p>{esc(analysis.data.root.name)} · Episode {s["episode"]}</p>
<table>{fact_html}</table><ul>{notes}</ul>
<p><a href="summary.json">完整数值摘要</a> · <a href="events.csv">全部事件 CSV</a></p>
<h2>评估动作与参考示范</h2><p>曲线连接录制点。示范按对应阶段锚点对齐；未指定锚点时仅按起点并排，不计算逐帧误差。
红线是按用户前提推断的 Chunk 交接后的首个录制帧，不是模型日志。</p>{images}
{probe_html}<h2>评估 / 示范关键帧</h2><div class="frames">{frames_html}</div>
<h2>事件与逐维最大变化</h2><p>每个动作维度分别排名；不混合不同单位，不自动判断失败原因。
示范动作是参考，不能当作另一次运行的逐帧正确答案。</p><table><tr><th>秒</th><th>帧</th><th>事件</th>
<th>字段</th><th>值</th><th>说明</th></tr>{event_html}</table></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
