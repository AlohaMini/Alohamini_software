"""Rerun 0.26 views using original compressed video assets and a measured timeline."""

from __future__ import annotations

import json

import numpy as np

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
    series_views = []
    for name in analysis.series:
        if name.startswith("actions/"):
            field = name.split("/", 1)[1]
            units = analysis.summary["actions"][field]["unit"]
            series_views.append(rrb.TimeSeriesView(origin=name, name=f"{field} [{units}]"))
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Grid(*[rrb.Spatial2DView(origin=f"cameras/{c}", name=c) for c in cameras]),
            rrb.Tabs(
                *series_views,
                rrb.TimeSeriesView(origin="currents_ma", name="Motor current (mA)"),
                rrb.TimeSeriesView(origin="timing/select_action_ms", name="Select action (ms)"),
                rrb.TimeSeriesView(origin="timing/camera_skew_ms", name="Camera skew (ms)"),
                rrb.TimeSeriesView(
                    origin="camera_state_offset_ms", name="State minus capture (ms)"
                ),
                rrb.TimeSeriesView(origin="recording/command_sequence", name="Command sequence"),
            ),
            rrb.Horizontal(
                rrb.TextLogView(origin="events", name="事件 / 当前帧"),
                rrb.TextDocumentView(origin="summary", name="记录与解释"),
            ),
            row_shares=[4, 3, 2],
        )
    )
    stream.send_blueprint(blueprint)
    column = rr.TimeColumn("episode_time", duration=analysis.times)
    frame_column = rr.TimeColumn(
        "frame", sequence=[analysis.data.rows[i]["frame_index"] for i in analysis.indices]
    )
    for path, series in analysis.series.items():
        for name, values in series.items():
            stream.send_columns(
                f"{path}/{name}",
                indexes=[column, frame_column],
                columns=rr.Scalars.columns(scalars=values),
            )
    for event in analysis.events:
        stream.set_time("episode_time", duration=event["time_s"])
        stream.set_time("frame", sequence=event["frame_index"])
        stream.log("events/diagnostics", rr.TextLog(json.dumps(event, ensure_ascii=False)))
    for position, i in enumerate(analysis.indices):
        side = analysis.data.sidecars[i]
        stream.set_time("episode_time", duration=float(analysis.times[position]))
        stream.set_time("frame", sequence=analysis.data.rows[i]["frame_index"])
        stream.log(
            "events/frame",
            rr.TextLog(
                json.dumps(
                    {
                        "frame": analysis.data.rows[i]["frame_index"],
                        "issued_command": side.get("issued_command"),
                        "feedback_command": side.get("safety", {}).get("command"),
                    },
                    ensure_ascii=False,
                )
            ),
        )
    budget = int(media_budget_mb * 1024 * 1024)
    loaded = {}
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
                    asset_entity = f"assets/video_{len(loaded)}"
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
                    stream.set_time("episode_time", duration=float(analysis.times[p]))
                    stream.set_time("frame", sequence=analysis.data.rows[i]["frame_index"])
                    stream.log(
                        f"cameras/{camera}",
                        rr.VideoFrameReference(nanoseconds=int(pts), video_reference=entity),
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                analysis.summary["warnings"].append(f"{camera}: 视频未加载：{exc}")
        for p, i in enumerate(analysis.indices):
            ref = refs[i]
            if ref is not None and ref.kind == "video":
                continue
            stream.set_time("episode_time", duration=float(analysis.times[p]))
            stream.set_time("frame", sequence=analysis.data.rows[i]["frame_index"])
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
                    f"cameras/{camera}", rr.EncodedImage(contents=encoded, media_type=media_type)
                )
            except (OSError, ValueError) as exc:
                stream.log(f"cameras/{camera}", rr.Clear(recursive=True))
                analysis.summary["warnings"].append(f"{camera} frame {i}: {exc}")
                if budget <= 0:
                    break
    stream.log(
        "summary",
        rr.TextDocument(json.dumps(analysis.summary, ensure_ascii=False, indent=2)),
        static=True,
    )
    stream.flush()
    return stream
