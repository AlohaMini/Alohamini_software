"""Offline analysis CLI, separate from model training and robot deployment."""

from __future__ import annotations

import argparse
import fcntl
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from alohamini.paths import WorkspacePaths


def parser():
    command = argparse.ArgumentParser(
        prog="alohamini dataset analyze", description="关联评估 checkpoint，对比训练示范与模型动作"
    )
    command.add_argument("root", type=Path, help="本地原生或 LeRobot v3 数据集目录")
    command.add_argument("--episode", type=int, default=0)
    command.add_argument(
        "--checkpoint", type=Path, help="旧评估补充模型关联，或重定位已记录的 checkpoint"
    )
    command.add_argument(
        "--training-root", type=Path, help="训练数据搬迁后的本地路径，须通过 checkpoint 指纹检查"
    )
    command.add_argument("--reference-split", choices=("train", "val"), default="train")
    command.add_argument(
        "--reference-episode", type=int, help="示范 episode；默认 checkpoint 划分中的首条"
    )
    command.add_argument("--reference-segment", type=int, default=0)
    command.add_argument("--reference-time-range", nargs=2, type=float, metavar=("START", "END"))
    command.add_argument(
        "--align",
        nargs=2,
        type=float,
        metavar=("EVAL_S", "DEMO_S"),
        help="把评估与示范中对应任务阶段的两个秒数对齐到 0；默认仅按各自起点并排",
    )
    command.add_argument(
        "--control-diagnostics",
        action="store_true",
        help="额外显示命令、Host 反馈、电流和耗时；默认隐藏",
    )
    command.add_argument(
        "--probe-samples",
        type=int,
        default=0,
        help="显式加载 checkpoint，按策略输入窗口抽取示范观测预测动作段并计算误差",
    )
    command.add_argument("--probe-device", default="cpu", choices=("cpu", "cuda"))
    command.add_argument("--probe-seed", type=int, default=0)
    command.add_argument("--segment", type=int, default=0, help="时钟重置后的时间段编号，默认 0")
    command.add_argument(
        "--time-range", nargs=2, type=float, metavar=("START", "END"), help="所选时间段内的起止秒数"
    )
    command.add_argument("--viewer", choices=("rerun", "none"), default="rerun")
    command.add_argument("--output", type=Path, help="数据集外的新报告目录")
    command.add_argument("--export-html", action="store_true", help="导出 HTML 报告和动作图表")
    command.add_argument(
        "--keyframes",
        nargs="+",
        type=float,
        default=[],
        metavar="SECONDS",
        help="导出指定秒数最近的画面；同时启用 HTML，缓存避免重复解码",
    )
    command.add_argument(
        "--save-rrd",
        action="store_true",
        help="保存可在 Rerun 打开的 recording.rrd（包含所选压缩媒体）",
    )
    command.add_argument("--cameras", nargs="+", help="要显示/导出的相机名称，按给定顺序排列")
    command.add_argument(
        "--media-budget-mb",
        type=int,
        default=256,
        help="Rerun 压缩媒体累计预算 MiB；共享 MP4 分片需整文件加载",
    )
    command.add_argument(
        "--assume-n-action-steps",
        type=int,
        help="仅为旧记录推断 Chunk 提供 NAS；必须确认关闭时序融合",
    )
    command.add_argument(
        "--first-command-sequence",
        type=int,
        help="Episode 首次预测动作对应的命令序号，不一定是首个录制帧",
    )
    command.add_argument(
        "--assume-no-resets",
        action="store_true",
        help="确认整段无队列重置、每 tick 一个命令、关闭时序融合",
    )
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    if args.media_budget_mb <= 0:
        raise ValueError("--media-budget-mb 必须大于 0")
    if args.probe_samples < 0 or not 0 <= args.probe_seed < 2**32:
        raise ValueError("probe-samples 必须非负，probe-seed 须在 [0, 2^32) 内")
    if args.assume_n_action_steps is None and (
        args.first_command_sequence is not None or args.assume_no_resets
    ):
        raise ValueError("Chunk 推断参数须配合 --assume-n-action-steps 使用")
    root = args.root.expanduser().resolve()
    workspace = WorkspacePaths().root
    identifier = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    output = (
        args.output.expanduser().resolve()
        if args.output
        else workspace / "reports" / root.name / f"episode_{args.episode:06d}" / identifier
    )
    if output.is_relative_to(root):
        raise ValueError("分析结果必须保存在源数据集之外")
    if output.exists():
        raise ValueError(f"输出目录已存在，请指定新目录：{output}")
    started = time.perf_counter()
    with ExitStack() as stack:
        if (root / "recording.lock").is_file():
            lock = stack.enter_context((root / "recording.lock").open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("数据集正在录制/编辑，请结束后分析") from exc
        from .comparison import prepare
        from .media import MediaReader
        from .metrics import analyze
        from .provenance import reference_episode, resolve_context, verify_reference
        from .report import export_html, write_summary
        from .source import load_episode

        context = resolve_context(
            root,
            checkpoint=args.checkpoint,
            training_root=args.training_root,
            reference_split=args.reference_split,
        )
        reference_index = reference_episode(context, args.reference_episode)
        if args.probe_samples:
            if not context["checkpoint"]:
                raise ValueError(
                    "模型探针需要关联 checkpoint；请用 --checkpoint 指定评估使用的模型"
                )
            if reference_index is None:
                split_key = f"{args.reference_split}_episodes"
                hint = (
                    "；可用 --reference-split train 检查训练拟合，但不能作为验证集误差"
                    if args.reference_split == "val"
                    else "；无法确认用于模型探针的训练示范"
                )
                raise ValueError(f"checkpoint 没有可用的 {split_key} 划分{hint}")
            if not context["training_available"]:
                raise ValueError(
                    f"模型探针对应的示范数据集不在本机：{context['training_root']}；"
                    "请用 --training-root 指定原数据集的本地副本"
                )
        reference = None
        if context["training_available"] and reference_index is not None:
            reference_root = Path(context["training_root"])
            if output.is_relative_to(reference_root):
                raise ValueError("分析结果必须保存在训练数据集之外")
            if (reference_root / "recording.lock").is_file() and reference_root != root:
                reference_lock = stack.enter_context((reference_root / "recording.lock").open("rb"))
                fcntl.flock(reference_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            reference_data = load_episode(
                reference_root,
                reference_index,
                all_numeric=args.probe_samples > 0,
                include_task=args.probe_samples > 0,
            )
            verify_reference(context, reference_data)
            reference = analyze(
                reference_data, segment=args.reference_segment, time_range=args.reference_time_range
            )

        data = load_episode(root, args.episode)
        if args.cameras and (
            len(set(args.cameras)) != len(args.cameras)
            or any(c not in data.media for c in args.cameras)
        ):
            raise ValueError(f"相机不存在或重复；可选：{', '.join(data.media)}")
        result = analyze(
            data,
            segment=args.segment,
            time_range=args.time_range,
            assumed_nas=args.assume_n_action_steps,
            first_command=args.first_command_sequence,
            no_resets=args.assume_no_resets,
        )
        prepare(
            result,
            context,
            reference=reference,
            anchors=args.align,
            diagnostics=args.control_diagnostics,
        )
        if any(not result.times[0] <= t <= result.times[-1] for t in args.keyframes):
            raise ValueError("关键帧时间必须位于所选时间范围内")
        output.mkdir(parents=True)
        reader = MediaReader(workspace / "cache/evaluation-keyframes")
        stack.callback(reader.close)
        if args.probe_samples:
            from .probe import probe_reference

            result.summary["demonstration_probe"] = probe_reference(
                context,
                reference,
                reader,
                output,
                samples=args.probe_samples,
                device=args.probe_device,
                seed=args.probe_seed,
            )
        if args.viewer == "rerun" or args.save_rrd:
            from .viewer import show

            show(
                result,
                reader,
                spawn=args.viewer == "rerun",
                rrd=output / "recording.rrd" if args.save_rrd else None,
                cameras=args.cameras,
                media_budget_mb=args.media_budget_mb,
            )
        if args.export_html or args.keyframes:
            export_html(result, output, reader, keyframe_times=args.keyframes, cameras=args.cameras)
        data.unchanged()
        if reference is not None:
            reference.data.unchanged()
            result.summary["warnings"].extend(reference.summary["warnings"])
        result.summary["performance"] = dict(
            elapsed_s=time.perf_counter() - started,
            keyframe_requests=reader.keyframe_requests,
            keyframe_cache_hits=reader.cache_hits,
            video_keyframe_reads=reader.video_frame_requests,
            video_reencodes=0,
            note="解码器 seek 可能解码关键帧前的 GOP；viewer 内部解码不计入该计数",
        )
        write_summary(result, output)
    print(f"分析结果：{output}")
    print(
        f"{result.summary['selected_rows']} 帧，{result.summary['duration_s']:.3f} 秒；"
        f"时间依据：{result.summary['time_basis']}"
    )
    for warning in dict.fromkeys(result.summary["warnings"]):
        print(f"提示：{warning}")
    if args.save_rrd:
        print(f"Rerun：rerun {output / 'recording.rrd'}")
    return output
