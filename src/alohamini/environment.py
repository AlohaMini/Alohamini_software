"""Inspect the current checkout's environment without importing model libraries."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import tomllib

_WARNED = set()


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def checkout():
    root = Path(__file__).resolve().parents[2]
    return root if (root / "env/dependencies.toml").is_file() else None


def lock_path(root, role):
    if platform.system() == "Darwin":
        arch = {"arm64": "osx-arm64", "x86_64": "osx-64"}.get(platform.machine())
        if role == "host":
            raise ValueError("Robot Host runs on Ubuntu; use the alohamini PC environment on macOS")
        if role == "pc" and arch == "osx-64":
            raise ValueError(
                "The current PyTorch deployment requires Apple Silicon and native arm64 Python; "
                "Intel Mac supports serial calibration, not the locked learning environment"
            )
        if arch is not None:
            return root / "env" / f"{role}-{arch}.lock"
    arch = {"x86_64": "linux-64", "aarch64": "linux-aarch64"}.get(platform.machine())
    if platform.system() != "Linux" or arch is None or (role == "pc" and arch != "linux-64"):
        raise ValueError(f"No {role} deployment lock for {platform.system()}/{platform.machine()}")
    return root / "env" / (f"host-{arch}.lock" if role == "host" else "pc-linux-64.lock")


def read_lock(path, seen=()):
    path = path.resolve()
    if path in seen:
        raise ValueError("Recursive environment lock")
    versions, sources = {}, {str(path): path.read_bytes()}
    for line in sources[str(path)].decode().replace("\\\n", " ").splitlines():
        line = line.strip()
        if line.startswith("-r "):
            child, files = read_lock(path.parent / line[3:].strip(), (*seen, path))
            sources.update(files)
        else:
            match = re.match(r"([\w.-]+)==([^\s;]+)", line)
            child = {canonical(match[1]): match[2]} if match else {}
        for name, version in child.items():
            if name in versions and versions[name] != version:
                raise ValueError(f"Conflicting lock entries for {name}")
            versions[name] = version
    if not versions:
        raise ValueError(f"Empty dependency lock: {path}")
    return versions, sources


def installed_distributions():
    result = {}
    for distribution in metadata.distributions():
        if distribution.metadata.get("Name"):
            result.setdefault(canonical(distribution.metadata["Name"]), distribution)
    return result


def installed_versions():
    return {name: distribution.version for name, distribution in installed_distributions().items()}


def inspect_environment(root, role, names=None):
    expected, sources = read_lock(lock_path(root, role))
    installed = installed_versions()
    selected = expected.keys() if names is None else set(names) & expected.keys()
    differences = [
        {"package": name, "installed": installed.get(name), "expected": expected[name]}
        for name in sorted(selected)
        if installed.get(name, "").split("+", 1)[0] != expected[name]
    ]
    python_expected = tomllib.loads((root / "env/dependencies.toml").read_text())["python"]
    python_actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if python_actual != python_expected:
        differences.insert(
            0, {"package": "python", "installed": python_actual, "expected": python_expected}
        )
    fingerprint = hashlib.sha256()
    for name, data in sorted(sources.items()):
        fingerprint.update(Path(name).name.encode())
        fingerprint.update(data)
    return {
        "role": role,
        "python": sys.executable,
        "prefix": sys.prefix,
        "lock_sha256": fingerprint.hexdigest(),
        "differences": differences,
        "matched": not differences,
    }


def command_packages(argv):
    """Check only packages used by this entry point; optional policies stay lazy."""
    if not argv or "--help" in argv or "-h" in argv or argv[0] in {"env", "paths", "inspect"}:
        return set()
    command = argv[0]
    if command == "host":
        return {"numpy", "opencv-python-headless", "pyzmq", "pyserial", "feetech-servo-sdk"}
    if command in {"train", "evaluate"} or argv[:2] == ["inference", "serve"]:
        return {
            "numpy",
            "torch",
            "torchvision",
            "transformers",
            "tokenizers",
            "huggingface-hub",
            "accelerate",
            "safetensors",
            "pyarrow",
            "torchcodec",
        }
    if argv[:2] == ["inference", "run"]:
        return {"numpy", "pyzmq", "opencv-python-headless"}
    if command == "dataset":
        packages = {"numpy", "pyarrow", "pandas", "opencv-python-headless", "av"}
        if argv[1:2] == ["analyze"]:
            packages |= {"rerun-sdk", "matplotlib"}
        return packages
    if command in {"record", "replay", "teleoperate"}:
        packages = {"numpy", "pyzmq", "opencv-python-headless", "pynput"}
        if command == "record":
            packages |= {"pyarrow", "av"}
        return packages
    if command in {"find-port", "calibrate"}:
        return {"pyserial", "feetech-servo-sdk"}
    if command == "find-cameras":
        return {"numpy", "opencv-python-headless"}
    return set()


def inferred_role():
    return "host" if Path(sys.prefix).name.endswith("_host") else "pc"


@contextmanager
def environment_lock(exclusive=False, prefix=None):
    key = hashlib.sha256(str(Path(prefix or sys.prefix).resolve()).encode()).hexdigest()
    directory = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    directory = directory / "alohamini" / "environments"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("环境正在使用或同步；请等待当前任务结束后重试。") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def runtime_environment(argv):
    root, names = checkout(), command_packages(argv)
    if root is None or not names:
        yield
        return
    if platform.system() == "Darwin" and names <= {"pyserial", "feetech-servo-sdk"}:
        # Serial-only commands also work on Intel, without a learning environment.
        with environment_lock():
            expected, _ = read_lock(root / "env/hardware.lock")
            installed = installed_versions()
            for name, version in expected.items():
                if installed.get(name) != version:
                    raise ValueError(
                        f"Serial tools require {name}=={version}; "
                        "install dependencies from env/hardware.lock"
                    )
            yield
        return
    role = "host" if argv[0] == "host" else inferred_role()
    # Reading metadata does not import optional frameworks or contact a package index.
    with environment_lock():
        report = inspect_environment(root, role, names)
        warning = (sys.prefix, role, json.dumps(report['differences'], sort_keys=True))
        if report["differences"] and warning not in _WARNED:
            changed = "; ".join(
                f"{r['package']}: {r['installed'] or '未安装'} → {r['expected']}"
                for r in report["differences"]
            )
            print(
                f"alohamini: 环境与当前代码不一致：{changed}\n"
                f"同步命令：alohamini env sync --role {role}",
                file=sys.stderr,
            )
            _WARNED.add(warning)
        yield


def active_python_processes(prefix=None, proc=Path("/proc")):
    """Also detect older entry points and notebooks which do not hold our lock."""
    prefix = Path(prefix or sys.prefix).resolve()
    if not proc.is_dir() and platform.system() == "Darwin":
        # /proc is Linux-specific. Ask ps for executable paths on macOS.
        output = subprocess.run(
            ["ps", "-axo", "pid=,comm="], check=True, capture_output=True, text=True
        ).stdout
        result = []
        for line in output.splitlines():
            parts = line.strip().split(None, 1)
            if len(parts) != 2 or not parts[0].isdigit() or int(parts[0]) == os.getpid():
                continue
            executable = Path(parts[1])
            if executable.name.startswith("python") and executable.is_relative_to(prefix):
                result.append(int(parts[0]))
        return sorted(result)
    result = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            executable = (entry / "exe").resolve(strict=True)
            invocation = (
                Path(os.fsdecode((entry / "cmdline").read_bytes().split(b"\0", 1)[0]))
                if (entry / "cmdline").exists()
                else executable
            )
            in_prefix = executable.is_relative_to(prefix) or (
                invocation.is_absolute() and invocation.is_relative_to(prefix)
            )
            if in_prefix and executable.name.startswith("python"):
                result.append(int(entry.name))
        except (OSError, RuntimeError):
            continue
    return sorted(result)


def projected_conflicts(plan):
    from packaging.markers import default_environment
    from packaging.requirements import Requirement

    versions = installed_versions()
    requirements = {name: d.requires or [] for name, d in installed_distributions().items()}
    for item in plan.get("install", []):
        info = item["metadata"]
        name = canonical(info["name"])
        versions[name], requirements[name] = info["version"], info.get("requires_dist", [])
    conflicts = []
    for name, rows in requirements.items():
        for row in rows:
            req = Requirement(row)
            if req.marker and not req.marker.evaluate({**default_environment(), "extra": ""}):
                continue
            installed = versions.get(canonical(req.name))
            if installed is None or not req.specifier.contains(installed, prereleases=True):
                conflicts.append(
                    f"{name} requires {req}; planned {req.name}={installed or 'missing'}"
                )
    return sorted(set(conflicts))


def dependency_snapshot(root, lock):
    _, sources = read_lock(lock)
    for path in (root / "pyproject.toml", root / "env/dependencies.toml"):
        if path.is_file():
            sources[str(path.resolve())] = path.read_bytes()
    return sources


def ensure_snapshot(sources):
    if any(
        not Path(path).is_file() or Path(path).read_bytes() != data
        for path, data in sources.items()
    ):
        raise RuntimeError("同步期间依赖文件发生变化；请按当前代码重新运行同步。")


def sync_environment(root, role, *, dry_run=False):
    prefix = Path(sys.prefix)
    if not (prefix / "conda-meta").is_dir() and sys.prefix == sys.base_prefix:
        raise ValueError("请先激活 Conda 或 venv 环境，不在系统 Python 中同步。")
    config = tomllib.loads((root / "env/dependencies.toml").read_text())
    if f"{sys.version_info.major}.{sys.version_info.minor}" != config["python"]:
        raise ValueError(f"需要 Python {config['python']}；请按安装文档重新创建环境。")
    lock = lock_path(root, role)
    from alohamini.paths import WorkspacePaths

    with environment_lock(exclusive=not dry_run):
        active = active_python_processes()
        if active and not dry_run:
            raise RuntimeError(f"当前环境仍有 Python 进程：{active}；请先结束任务/Notebook。")
        subprocess.run([sys.executable, str(root / "env/dependencies.py"), "check"], check=True)
        expected, _ = read_lock(lock)
        # Detect incompatible packages outside the platform lock before downloading.
        extras = projected_conflicts(
            {
                "install": [
                    {"metadata": {"name": name, "version": version, "requires_dist": []}}
                    for name, version in expected.items()
                ]
            }
        )
        if extras:
            raise ValueError(
                "锁文件之外的包与目标环境冲突；请先处理这些包或使用独立环境：\n" + "\n".join(extras)
            )
        sources = dependency_snapshot(root, lock)
        pip = [sys.executable, "-m", "pip"]
        with tempfile.TemporaryDirectory(prefix="alohamini-env-plan-") as temp:
            frozen = Path(temp) / "checkout"
            for name, data in sources.items():
                destination = frozen / Path(name).relative_to(root.resolve())
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            install = [
                *pip,
                "install",
                "--require-hashes",
                "--no-build-isolation",
                "-r",
                str(frozen / lock.relative_to(root)),
            ]
            plan_path = Path(temp) / "plan.json"
            subprocess.run([*install, "--dry-run", "--report", str(plan_path)], check=True)
            plan = json.loads(plan_path.read_text())
            conflicts = projected_conflicts(plan)
            if conflicts:
                raise ValueError(
                    "同步会遇到依赖冲突；请先处理额外安装的包或使用独立环境：\n"
                    + "\n".join(conflicts)
                )
            ensure_snapshot(sources)
            if dry_run:
                print(
                    json.dumps(
                        {
                            "install": [x["metadata"]["name"] for x in plan["install"]],
                            "active_processes": active,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return
            # Keep diagnostics only if installation fails or is interrupted.
            before = subprocess.run(
                [*pip, "freeze", "--all"], check=True, text=True, capture_output=True
            ).stdout
            try:
                subprocess.run(install, check=True)
                ensure_snapshot(sources)
                subprocess.run(
                    [*pip, "install", "--no-deps", "--no-build-isolation", "-e", f"{root}[{role}]"],
                    check=True,
                )
                subprocess.run([*pip, "check"], check=True)
                ensure_snapshot(sources)
                report = inspect_environment(root, role)
                if not report["matched"]:
                    raise RuntimeError("安装后的版本仍与锁文件不一致；请运行 alohamini env check。")
            except (Exception, KeyboardInterrupt) as exc:
                directory = Path(
                    tempfile.mkdtemp(prefix="sync-", dir=_logs_directory(WorkspacePaths()))
                )
                (directory / "before.txt").write_text(before)
                (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
                (directory / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
                print(f"同步未完成；原环境版本和安装计划保存在 {directory}", file=sys.stderr)
                raise
            print(f"环境已同步：{sys.prefix}")


def _logs_directory(paths):
    directory = paths.logs / "environment"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alohamini env", description="检查和同步当前代码的依赖")
    parser.add_argument("operation", choices=["check", "sync"])
    parser.add_argument("--role", choices=["pc", "host"], default=inferred_role())
    parser.add_argument("--dry-run", action="store_true", help="仅解析同步计划，不安装")
    parser.add_argument("--json", action="store_true", help="输出环境检查 JSON")
    args = parser.parse_args(argv)
    root = checkout()
    if root is None:
        parser.error("需要从源码仓库以 editable 方式安装 AlohaMini。")
    try:
        if args.operation == "sync":
            sync_environment(root, args.role, dry_run=args.dry_run)
            return 0
        with environment_lock():
            report = inspect_environment(root, args.role)
            manifests = subprocess.run(
                [sys.executable, str(root / "env/dependencies.py"), "check"],
                capture_output=True,
                text=True,
            )
        if manifests.returncode:
            report["matched"] = False
            report["manifest_error"] = (manifests.stderr or manifests.stdout).strip()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"环境：{sys.prefix}；角色：{args.role}")
            for row in report["differences"]:
                print(f"{row['package']}: {row['installed'] or '未安装'} → {row['expected']}")
            if report.get("manifest_error"):
                print(report["manifest_error"])
            print(
                "依赖与当前代码一致。"
                if report["matched"]
                else f"同步命令：alohamini env sync --role {args.role}"
            )
        return 0 if report["matched"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"alohamini env: {exc}", file=sys.stderr)
        return 1
