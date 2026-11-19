"""Run native PC checks and grouped macOS regressions with portable reports."""

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "serial": [
        "test_macos_serial.py", "test_calibration_procedure.py",
        "test_encoding_process.py::test_native_resident_memory_is_positive",
        "test_encoding_process.py::test_parent_pipe_closure_exits_child_without_linux_prctl",
    ],
    "communication": [
        "test_client.py", "test_protocol.py", "test_remote_protocol.py", "test_remote_link.py",
        "test_remote_integration.py", "test_remote_server.py", "test_camera_stream.py",
        "test_teleoperation.py", "test_visualization.py",
    ],
    "cache": [
        "test_data_store.py", "test_disk_dataset.py", "test_video_reader.py",
        "test_frozen_features.py", "test_bounded_numeric_statistics.py",
        "test_bounded_policy_statistics.py", "test_data_pipeline.py", "test_streaming.py",
        "test_streaming_config.py",
        "test_recording_stats_cache.py", "test_recording_stats_incremental.py",
    ],
    "data": [
        "test_dataset.py", "test_dataset_edit.py", "test_dataset_lerobot.py",
        "test_dataset_migration.py",
        "test_lerobot_source.py", "test_recording_storage.py", "test_recording_resume.py",
        "test_recording_interrupted_discard.py", "test_recording_stats_resume.py",
    ],
    "video": [
        "test_encoding_process.py", "test_recording_background_video.py",
        "test_recording_video_timing.py", "test_recording_video_validation.py",
        "test_recording_video_statistics.py", "test_video_backend_selection.py",
    ],
    "training": [
        "test_macos.py", "test_macos_hardware.py", "test_macos_training.py",
        "test_macos_coordinates.py",
        "test_training_precision.py", "test_loss_metrics.py", "test_validation_loading.py",
        "test_training_logging.py", "test_ema.py", "test_policy_normalization.py",
        "test_checkpoint_probe.py",
        "test_training.py::test_periodic_checkpoint_resume_exact_and_no_data_mutation[cpu-none-1-1-0]",
        "test_training.py::test_periodic_checkpoint_resume_exact_and_no_data_mutation[cpu-none-1-2-2]",
    ],
}


def summarize(path):
    """Count leaf test cases so nested JUnit suites cannot duplicate totals."""
    root = ET.parse(path).getroot()
    result = dict(passed=0, failed=0, skipped=0, skip_reasons=[])
    for case in root.iter("testcase"):
        skipped = case.find("skipped")
        if skipped is not None:
            result["skipped"] += 1
            reason = skipped.get("message", skipped.text or "")
            if reason not in result["skip_reasons"]:
                result["skip_reasons"].append(reason)
        elif case.find("failure") is not None or case.find("error") is not None:
            result["failed"] += 1
        else:
            result["passed"] += 1
    return result


def _run(command, name, output, environment):
    start = time.monotonic()
    log = output / f"{name}.log"
    with log.open("wb") as stream:
        try:
            process = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
                timeout=1200, check=False,
            )
            code = process.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return dict(
        status="passed" if code == 0 else "failed", returncode=code,
        duration_s=round(time.monotonic() - start, 3), log=str(log), command=command,
    )


def _host_check(host, model):
    """Read state and images only; the client never creates a command channel."""
    from alohamini.client import HostClient

    with HostClient(host, expected_model=model, timeout_s=3) as client:
        snapshot = client.refresh(include_images=True)
        return dict(
            status="passed", images={key: len(data) for key, data in snapshot.images.items()}
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="+", choices=("all", *SUITES), default=["all"])
    parser.add_argument("--output", type=Path, help="New directory for JSON, JUnit and logs")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--host", help="Ubuntu Host address for an optional read-only check")
    parser.add_argument("--model", default="alohamini2pro")
    parser.add_argument("--allow-other-platform", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if platform.system() != "Darwin" and not args.allow_other_platform:
        parser.error("Run this check on macOS")
    selected = list(SUITES) if "all" in args.sections else list(dict.fromkeys(args.sections))
    if platform.system() == "Darwin" and platform.machine() != "arm64" and selected != ["serial"]:
        parser.error(
            "Full PC checks require native arm64 Python; Intel serial checks: --sections serial"
        )
    label = f"{datetime.now():%Y%m%d-%H%M%S}-{time.time_ns() % 1000000:06d}"
    output = args.output or (
        Path(os.environ.get("ALOHAMINI_WORKSPACE", Path.home() / "Alohamini_workspace"))
        / "logs" / "macos-validation" / label
    )
    output = output.expanduser().resolve()
    if output.exists():
        parser.error(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    report = dict(
        system=platform.system(), architecture=platform.machine(), macos=platform.mac_ver()[0],
        python=sys.version, executable=sys.executable, native_macos=platform.system() == "Darwin",
        sections={},
    )
    try:
        report["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        report["commit"] = None
    report["packages"] = {}
    for name in ("torch", "torchvision", "torchcodec", "av", "opencv-python-headless", "pyzmq"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    environment = {
        **os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg",
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save()
    print(f"Reports: {output}", flush=True)
    if selected != ["serial"]:
        result = _run(
            [sys.executable, str(ROOT / "env/verify_pc.py"), "--device", args.device],
            "environment", output, environment,
        )
        report["sections"]["environment"] = result
        print(f"environment: {result['status']}", flush=True)
        save()
    for section in selected:
        junit = output / f"{section}.xml"
        command = [
            sys.executable, "-m", "pytest", "-q", "--tb=short", f"--junitxml={junit}",
            *[f"tests/{name}" for name in SUITES[section]],
            # macOS default is spawn; forkserver is separately supported. Native
            # library state inherited through an explicit fork is outside this run.
            "-k", "not fork or forkserver or rejects_explicit_fork",
        ]
        result = _run(command, section, output, environment)
        if junit.is_file():
            try:
                result.update(summarize(junit))
            except (ET.ParseError, OSError) as exc:
                result.update(status="failed", junit_error=str(exc))
        report["sections"][section] = result
        save()
        print(
            f"{section}: {result['status']}; {result.get('passed', 0)} passed, "
            f"{result.get('failed', 0)} failed, {result.get('skipped', 0)} skipped",
            flush=True,
        )
    if args.host:
        try:
            report["sections"]["host"] = _host_check(args.host, args.model)
        except Exception as exc:
            report["sections"]["host"] = dict(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
    return int(any(value["status"] != "passed" for value in report["sections"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
