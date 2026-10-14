"""Checkpoint-centered Rerun views: recorded evaluation and reference demonstration."""

from __future__ import annotations

import json

import numpy as np

from .provenance import display_config
from .source import file_signature


def show(analysis, reader, *, spawn=True, rrd=None, cameras=None, media_budget_mb=256):
    import rerun as rr
    import rerun.blueprint as rrb

    cameras = list(analysis.data.media) if cameras is None else cameras
    stream = rr.RecordingStream("alohamini-evaluation")
    if rrd:
        stream.save(str(rrd))
    elif spawn:
        stream.spawn(memory_limit="25%")
    runs = [("evaluation", "评估", analysis)]
    if analysis.reference is not None:
        runs.append(("reference", "示范", analysis.reference))
    images = rrb.Grid(
        *[
            rrb.Spatial2DView(origin=f"{prefix}/cameras/{camera}", name=f"{label} · {camera}")
            for prefix, label, run in runs
            for camera in cameras
            if camera in run.data.media
        ],
        grid_columns=len(cameras) or 1,
    )
    views = []
    for path in analysis.series:
        if path.startswith("actions/"):
            name = path.split("/", 1)[1]
            units = analysis.summary["actions"][name]["unit"]
            views.append(rrb.TimeSeriesView(origin=f"comparison/{path}", name=f"{name} [{units}]"))
    if analysis.reference is not None:
        for field in analysis.summary.get("comparison", {}).get(
            "unverified_or_incompatible_fields", []
        ):
            if f"actions/{field}" in analysis.reference.series:
                views.append(
                    rrb.TimeSeriesView(
                        origin=f"reference/actions/{field}", name=f"示范 · {field}（单独单位）"
                    )
                )
    if analysis.summary.get("control_diagnostics", True):
        views.extend(
            rrb.TimeSeriesView(origin=f"evaluation/{path}", name=path)
            for path in ("currents_ma", "timing/select_action_ms", "recording/command_sequence")
        )
    stream.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                images,
                rrb.Tabs(*views),
                rrb.Horizontal(
                    rrb.TextLogView(origin="evaluation/events", name="动作变化 / 当前帧"),
                    rrb.TextDocumentView(origin="summary", name="模型与示范"),
                ),
                row_shares=[4, 3, 2],
            ),
            rrb.TimePanel(timeline="comparison_time", duration_cursor=0.0),
        )
    )
    budget = int(media_budget_mb * 1024 * 1024)
    assets = {}
    for prefix, _, run in runs:
        selected_cameras = [c for c in cameras if c in run.data.media]
        budget = _log_run(
            stream,
            run,
            reader,
            prefix,
            selected_cameras,
            budget,
            analysis.summary.get("comparison", {}).get("comparable_fields", []),
            assets,
        )
    context = analysis.summary.get("model_context", {})
    checkpoint = context.get("checkpoint") or {}
    details = {
        "checkpoint": checkpoint.get("path"),
        "kind": checkpoint.get("kind"),
        "model_source": context.get("checkpoint_source"),
        "effective_policy_config": display_config(context.get("effective_policy_config")),
        "checkpoint_config": display_config(checkpoint.get("config")),
        "camera_order": checkpoint.get("cameras"),
        "demonstration_probe": analysis.summary.get("demonstration_probe"),
        "comparison": {
            k: v
            for k, v in analysis.summary.get("comparison", {}).items()
            if k != "reference_actions"
        },
        "warnings": analysis.summary["warnings"],
        "interpretation": analysis.summary["interpretation"],
    }
    stream.log(
        "summary", rr.TextDocument(json.dumps(details, ensure_ascii=False, indent=2)), static=True
    )
    stream.flush()
    return stream


def _log_run(stream, analysis, reader, prefix, cameras, budget, comparable, loaded):
    import rerun as rr

    stream.reset_time()
    column = rr.TimeColumn("comparison_time", duration=analysis.times - analysis.anchor_s)
    frame_column = rr.TimeColumn(
        f"{prefix}_frame", sequence=[analysis.data.rows[i]["frame_index"] for i in analysis.indices]
    )
    for path, series in analysis.series.items():
        for name, values in series.items():
            destination = (
                f"comparison/{path}/{prefix}_{name}"
                if path.startswith("actions/")
                and (prefix == "evaluation" or path.split("/", 1)[1] in comparable)
                else f"{prefix}/{path}/{name}"
            )
            stream.send_columns(
                destination,
                indexes=[column, frame_column],
                columns=rr.Scalars.columns(scalars=values),
            )
    for event in analysis.events:
        stream.set_time("comparison_time", duration=event["time_s"] - analysis.anchor_s)
        stream.set_time(f"{prefix}_frame", sequence=event["frame_index"])
        stream.log(f"{prefix}/events", rr.TextLog(json.dumps(event, ensure_ascii=False)))
    for p, i in enumerate(analysis.indices):
        stream.set_time("comparison_time", duration=float(analysis.times[p] - analysis.anchor_s))
        stream.set_time(f"{prefix}_frame", sequence=analysis.data.rows[i]["frame_index"])
        stream.log(
            f"{prefix}/events/frame", rr.TextLog(f"frame {analysis.data.rows[i]['frame_index']}")
        )
    for camera in cameras:
        if budget <= 0:
            analysis.summary["warnings"].append(f"{camera}: 媒体预算已用尽，未加载画面。")
            continue
        refs = analysis.data.media[camera]
        # Process each physical video once; one shared shard may contain several episodes.
        video_paths = dict.fromkeys(
            refs[i].path
            for i in analysis.indices
            if refs[i] is not None and refs[i].kind == "video"
        )
        for path in video_paths:
            try:
                if path not in loaded:
                    size = path.stat().st_size
                    if size > budget:
                        raise ValueError(
                            f"压缩视频 {size / 2**20:.1f} MiB 超过剩余媒体预算；"
                            "可选择更少相机或显式提高 --media-budget-mb"
                        )
                    signature = file_signature(path)
                    asset = rr.AssetVideo(path=path)
                    stamps = asset.read_frame_timestamps_nanos()
                    asset_entity = f"{prefix}/assets/video_{len(loaded)}"
                    stream.log(asset_entity, asset, static=True)
                    if file_signature(path) != signature:
                        raise ValueError("读取期间视频发生变化")
                    loaded[path] = (asset_entity, stamps)
                    budget -= size
                entity, stamps = loaded[path]
                selected = [
                    p
                    for p, i in enumerate(analysis.indices)
                    if refs[i] is not None and refs[i].kind == "video" and refs[i].path == path
                ]
                source_frames = np.array([refs[analysis.indices[p]].index for p in selected])
                if source_frames.min() < 0 or source_frames.max() >= len(stamps):
                    raise ValueError("引用的视频帧超出媒体索引")
                # Explicit source PTS, never host seconds substituted as media PTS.
                for p, pts in zip(selected, np.asarray(stamps)[source_frames], strict=True):
                    i = analysis.indices[p]
                    stream.set_time(
                        "comparison_time", duration=float(analysis.times[p] - analysis.anchor_s)
                    )
                    stream.set_time(
                        f"{prefix}_frame", sequence=analysis.data.rows[i]["frame_index"]
                    )
                    stream.log(
                        f"{prefix}/cameras/{camera}",
                        rr.VideoFrameReference(nanoseconds=int(pts), video_reference=entity),
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                analysis.summary["warnings"].append(f"{camera}: 视频未加载：{exc}")
        for p, i in enumerate(analysis.indices):
            ref = refs[i]
            if ref is not None and ref.kind == "video":
                continue
            stream.set_time(
                "comparison_time", duration=float(analysis.times[p] - analysis.anchor_s)
            )
            stream.set_time(f"{prefix}_frame", sequence=analysis.data.rows[i]["frame_index"])
            try:
                if ref is None:
                    raise ValueError("该帧没有图像引用")
                encoded = (
                    reader.keyframe(ref, camera).read_bytes()
                    if ref.kind == "native_host_image"
                    else reader.encoded(ref, camera)
                )
                if len(encoded) > budget:
                    budget = 0
                    raise ValueError("图像累计大小超过媒体预算，请缩短 --time-range")
                budget -= len(encoded)
                if encoded.startswith(b"\x89PNG\r\n\x1a\n"):
                    media_type = "image/png"
                elif encoded.startswith(b"\xff\xd8"):
                    media_type = "image/jpeg"
                else:
                    raise ValueError("Rerun 压缩图像显示仅支持此数据中的 PNG/JPEG")
                stream.log(
                    f"{prefix}/cameras/{camera}",
                    rr.EncodedImage(contents=encoded, media_type=media_type),
                )
            except (OSError, ValueError) as exc:
                stream.log(f"{prefix}/cameras/{camera}", rr.Clear(recursive=True))
                analysis.summary["warnings"].append(f"{camera} frame {i}: {exc}")
                if budget <= 0:
                    break
    stream.set_time(
        "comparison_time", duration=float(analysis.times[-1] - analysis.anchor_s + 1e-6)
    )
    stream.disable_timeline(f"{prefix}_frame")
    stream.log(f"{prefix}/cameras", rr.Clear(recursive=True))
    return budget
