"""阶段检查只依赖当前功能，平台与安装失败保持明确结果。"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from alohamini import feature_environment as features
from env.dependencies import ROOT, requirements


def test_lightweight_locks_exclude_training_dependencies():
    for name in ("connection", "recording-linux-64", "recording-osx-arm64"):
        locked = requirements(ROOT / f"env/{name}.lock")
        assert not {"torch", "torchvision", "transformers", "torchcodec"} & locked.keys()
    assert set(requirements(ROOT / "env/connection.lock")) == {"pillow", "pyzmq", "zeroconf", "ifaddr"}


@pytest.mark.parametrize("feature", ["connection", "calibration"])
def test_missing_dependencies_return_feature_local_details_without_loading_training(
    feature, tmp_path
):
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(tmp_path / "env")], check=True
    )
    python = tmp_path / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [
            str(python),
            "-c",
            "\n".join(
                [
                    "import json,sys",
                    "from alohamini.feature_environment import inspect_feature_environment",
                    f"print(json.dumps(inspect_feature_environment({feature!r})))",
                    "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
                ]
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    report = json.loads(result.stdout)
    assert report["deployment_supported"] and not report["is_ready"]
    assert report["packages"] and all(row["expected_version"] for row in report["packages"])
    assert not any(row["installed"] for row in report["packages"])


def test_platform_support_is_per_feature(monkeypatch):
    monkeypatch.setattr(features.platform, "system", lambda: "Windows")
    assert features.feature_lock(ROOT, "connection").name == "connection.lock"
    assert features.feature_lock(ROOT, "calibration").name == "hardware.lock"
    for feature in ("recording", "training"):
        report = features.inspect_feature_environment(feature)
        assert not report["deployment_supported"] and not report["is_ready"]


def test_failed_install_is_not_reported_as_ready(tmp_path):
    subprocess.run([sys.executable, "-m", "venv", str(tmp_path / "env")], check=True)
    python = tmp_path / "env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    result = subprocess.run(
        [str(python), "-m", "alohamini", "env", "sync", "--feature", "connection"],
        env={
            **os.environ,
            "PYTHONPATH": str(ROOT / "src"),
            "PIP_NO_INDEX": "1",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "功能环境安装完成" not in result.stdout


def test_invalid_feature_does_not_start_install():
    result = subprocess.run(
        [sys.executable, "-m", "alohamini", "env", "sync", "--feature", "unknown"],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "invalid choice" in result.stderr


def test_failed_lock_resolution_preserves_all_existing_feature_locks(tmp_path, monkeypatch):
    from env import dependencies

    shutil.copy(ROOT / "pyproject.toml", tmp_path)
    shutil.copytree(ROOT / "env", tmp_path / "env", ignore=shutil.ignore_patterns("__pycache__"))
    before = {path: path.read_bytes() for path in (tmp_path / "env").glob("*.lock")}

    def resolve(root, source, output, **options):
        if "recording" in output.name:
            raise RuntimeError("依赖解析失败")
        output.write_text("模拟已解析的连接依赖\n")

    monkeypatch.setattr(dependencies, "compile_lock", resolve)
    with pytest.raises(RuntimeError, match="依赖解析失败"):
        dependencies.refresh_features(tmp_path)
    assert all(path.read_bytes() == content for path, content in before.items())
