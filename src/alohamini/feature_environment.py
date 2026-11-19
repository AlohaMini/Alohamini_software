"""按实际工作阶段检查和安装依赖，避免训练环境阻塞硬件操作。"""

from __future__ import annotations

import importlib
import io
import json
import platform
import subprocess
import sys
from contextlib import nullcontext, redirect_stdout

from alohamini.environment import (
    checkout,
    command_packages,
    dependency_snapshot,
    ensure_snapshot,
    environment_lock,
    installed_versions,
    lock_path,
    read_lock,
    sync_environment,
)

FEATURES = ("connection", "calibration", "recording", "training")


def feature_lock(root, feature):
    if feature == "connection":
        return root / "env/connection.lock"
    if feature == "calibration":
        return root / "env/hardware.lock"
    if feature == "recording":
        pc_lock = lock_path(root, "pc")
        return pc_lock.with_name(pc_lock.name.replace("pc-", "recording-"))
    if feature == "training":
        return lock_path(root, "pc")
    raise ValueError(f"未知功能：{feature}")


def inspect_feature_environment(feature):
    root = checkout()
    if root is None:
        raise ValueError("请从 software 源码检查功能环境。")
    result = {
        "is_ready": False,
        "installation_ready": False,
        "deployment_supported": True,
        "can_connect": False,
        "packages": [],
        "python_path": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": "",
        "compute_device": "",
        "status_message": "需要安装此功能所需的软件。",
    }
    try:
        expected, _ = read_lock(feature_lock(root, feature))
    except ValueError as error:
        result.update(deployment_supported=False, status_message=str(error))
        return result
    versions = installed_versions()
    result["packages"] = [
        {
            "name": name,
            "display_name": name,
            "installed": versions.get(name, "").split("+", 1)[0] == version,
            "version": versions.get(name, ""),
            "expected_version": version,
        }
        for name, version in sorted(expected.items())
    ]
    # 通信和串口锁不绑定学习环境的 Python 次版本，沿用项目支持范围。
    if feature in {"recording", "training"}:
        import tomllib

        python_expected = tomllib.loads((root / "env/dependencies.toml").read_text())["python"]
        if f"{sys.version_info.major}.{sys.version_info.minor}" != python_expected:
            result["status_message"] = f"需要 Python {python_expected}，请一键安装此功能环境。"
            return result
    if not all(row["installed"] for row in result["packages"]):
        return result
    names = {
        "connection": {"pyzmq", "pillow"},
        "calibration": command_packages(["calibrate"]),
        "recording": command_packages(["record"])
        | command_packages(["dataset"])
        | command_packages(["calibrate"])
        | {"pillow"},
        "training": command_packages(["train"]),
    }[feature]
    modules = {
        "pyzmq": "zmq",
        "pillow": "PIL.Image",
        "pyserial": "serial",
        "feetech-servo-sdk": "scservo_sdk.protocol_packet_handler",
        "opencv-python-headless": "cv2",
        "huggingface-hub": "huggingface_hub",
    }
    try:
        for name in sorted(names):
            if name == "opencv-python-headless" and platform.system() == "Darwin":
                # PyAV 与 OpenCV 的 macOS 轮子分别携带 FFmpeg，隔离探测原生库。
                subprocess.run(
                    [sys.executable, "-c", "import cv2"],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                continue
            module = importlib.import_module(modules.get(name, name))
            if name == "torch":
                result["torch_version"] = versions[name]
                result["compute_device"] = (
                    "cuda"
                    if module.cuda.is_available()
                    else "mps"
                    if module.backends.mps.is_available()
                    else "cpu"
                )
        if feature == "recording":
            importlib.import_module("alohamini.apps.recording")
            importlib.import_module("alohamini.datasets.lerobotv3")
        if feature == "training":
            from alohamini.learning.train_config import parse_training_args

            with redirect_stdout(io.StringIO()):
                try:
                    parse_training_args(["--help"])
                except SystemExit as error:
                    if error.code != 0:
                        raise RuntimeError("训练入口无法启动。") from error
    except (ImportError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result["status_message"] = f"功能检查未通过：{error}"
        return result
    result.update(
        is_ready=True,
        installation_ready=True,
        can_connect=feature == "connection",
        status_message="此功能环境已就绪。",
    )
    return result


def sync_feature_environment(root, feature, *, dry_run=False):
    if feature == "training":
        sync_environment(root, "pc", dry_run=dry_run)
        if not dry_run:
            subprocess.run([sys.executable, str(root / "env/verify_pc.py")], check=True)
        return
    from pathlib import Path

    if sys.prefix == sys.base_prefix and not (Path(sys.prefix) / "conda-meta").is_dir():
        raise ValueError("请在此功能的独立 venv 中安装，不在系统 Python 中安装。")
    lock = feature_lock(root, feature)
    # Windows 的串口和连接环境沿用独立 venv；桌面安装器串行管理这些任务。
    with environment_lock(exclusive=True) if sys.platform != "win32" else nullcontext():
        sources = dependency_snapshot(root, lock)
        command = [sys.executable, "-m", "pip", "install", "--require-hashes", "-r", str(lock)]
        subprocess.run([*command, "--dry-run"] if dry_run else command, check=True)
        ensure_snapshot(sources)
        if dry_run:
            return
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        result = inspect_feature_environment(feature)
        if not result["is_ready"]:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        print("功能环境安装完成。")
