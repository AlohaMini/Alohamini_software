"""Portable numeric evidence, plots and optional explicitly selected keyframes."""

from __future__ import annotations

import csv
import html
import json
import shutil

import numpy as np


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
                ax.plot(analysis.times, values, label=label, linewidth=0.8)
            name = path.split("/", 1)[1]
            ax.set_ylabel(analysis.summary["actions"][name]["unit"], fontsize=8)
            ax.set_title(name, fontsize=10)
            ax.grid(alpha=0.2)
            for event in analysis.events:
                if event["kind"] == "inferred_chunk_transition":
                    ax.axvline(event["time_s"], color="tab:red", alpha=0.2, linewidth=0.7)
        axes[0, 0].legend(fontsize=7, loc="upper right")
        axes[-1, 0].set_xlabel(f"Seconds ({analysis.summary['time_basis']})")
        fig.tight_layout()
        relative = f"plots/{group}.png"
        fig.savefig(output / relative, dpi=120)
        plt.close(fig)
        figures.append(relative)
    for requested in keyframe_times:
        if not analysis.times[0] <= requested <= analysis.times[-1]:
            raise ValueError(f"关键帧 {requested}s 不在所选时间范围内")
        p = int(np.argmin(abs(analysis.times - requested)))
        i = analysis.indices[p]
        frame = analysis.data.rows[i]["frame_index"]
        for camera in cameras:
            ref = analysis.data.media[camera][i]
            try:
                if ref is None:
                    raise ValueError("图像引用缺失")
                cached = reader.keyframe(ref, camera)
                # Avoid using external camera names as filesystem path components.
                relative = f"keyframes/frame_{frame:06d}_camera_{cameras.index(camera)}.png"
                (output / "keyframes").mkdir(exist_ok=True)
                shutil.copyfile(cached, output / relative)
                keyframes.append(
                    dict(
                        camera=camera,
                        requested_time_s=requested,
                        time_s=float(analysis.times[p]),
                        frame_index=frame,
                        path=relative,
                    )
                )
            except (OSError, ValueError, RuntimeError) as exc:
                analysis.summary["warnings"].append(f"关键帧 {camera}/{frame}: {exc}")
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
        "命令频率估计 (Hz)": s["command_hz_estimate"],
        "模型推理频率": "未记录",
        "有效策略配置": "未记录",
        "固定目标": json.dumps(s["fixed_dimensions"], ensure_ascii=False),
        "Chunk 推断前提": json.dumps(s["chunk_assumption"], ensure_ascii=False),
    }
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
        f"<figcaption>{esc(f['camera'])} · {f['time_s']:.3f}s · frame {f['frame_index']}"
        "</figcaption></figure>"
        for f in keyframes
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>评估记录分析 · {esc(analysis.data.root.name)}</title>
<style>body{{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#203040}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #dbe1e7;
padding:8px;text-align:left;overflow-wrap:anywhere}}th{{background:#f4f7fa}}img{{max-width:100%}}
figure{{margin:20px 0}}.frames{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
li{{margin:8px 0}}a{{color:#175e9c}}</style>
<h1>评估记录分析</h1><p>{esc(analysis.data.root.name)} · Episode {s["episode"]}</p>
<table>{fact_html}</table><p>{esc(s["command_hz_assumption"])}</p><ul>{notes}</ul>
<p><a href="summary.json">完整数值摘要</a> · <a href="events.csv">全部事件 CSV</a></p>
<h2>动作与观测</h2><p>曲线连接录制点；缺口之间的控制动作未记录。
红线是按用户前提推断的 Chunk 交接后的首个录制帧，不是模型日志。</p>{images}
<h2>关键帧</h2><div class="frames">{frames_html}</div>
<h2>事件与逐维最大变化</h2><p>每个动作维度分别排名；不混合不同单位，不自动判断失败原因。
命令序号缺口见 events.csv。</p><table><tr><th>秒</th><th>帧</th><th>事件</th>
<th>字段</th><th>值</th><th>说明</th></tr>{event_html}</table></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
