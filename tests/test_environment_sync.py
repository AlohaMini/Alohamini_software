"""Environment drift detection and guarded, lock-bound synchronization."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from alohamini import environment as env


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "env").mkdir(parents=True)
    (root / "env/dependencies.toml").write_text('python = "3.12"\n')
    (root / "env/pc-linux-64.lock").write_text("torch==2.15.0\ntransformers==5.22.1\n")
    (root / "env/host-linux-64.lock").write_text("numpy==2.2.6\n")
    monkeypatch.setattr(env.platform, "system", lambda: "Linux")
    monkeypatch.setattr(env.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(env, "checkout", lambda: root)
    return root


def test_read_lock_tracks_includes_and_rejects_conflicts_and_cycles(tmp_path):
    root, child = tmp_path / "root.lock", tmp_path / "child.lock"
    root.write_text("-r child.lock\nFoo_Bar==2.0 \\\n --hash=sha256:abc\n")
    child.write_text("baz==1.0\n")
    versions, sources = env.read_lock(root)
    assert versions == {"foo-bar": "2.0", "baz": "1.0"} and len(sources) == 2
    child.write_text("foo-bar==1.0\n")
    with pytest.raises(ValueError, match="Conflicting"):
        env.read_lock(root)
    child.write_text("-r root.lock\n")
    with pytest.raises(ValueError, match="Recursive"):
        env.read_lock(root)


def test_checks_actual_versions_after_checkout_changes(repository, monkeypatch):
    monkeypatch.setattr(env, "installed_versions", lambda: {"torch": "2.11.0+cu130"})
    old = env.inspect_environment(repository, "pc")
    assert [r["package"] for r in old["differences"]] == ["torch", "transformers"]
    (repository / "env/pc-linux-64.lock").write_text("torch==2.11.0\n")
    new = env.inspect_environment(repository, "pc")
    assert new["matched"] and new["lock_sha256"] != old["lock_sha256"]


def test_python_mismatch_is_reported(repository, monkeypatch):
    monkeypatch.setattr(env, "installed_versions", lambda: {"numpy": "2.2.6"})
    (repository / "env/dependencies.toml").write_text('python = "3.99"\n')
    assert env.inspect_environment(repository, "host")["differences"] == [
        {"package": "python", "installed": "3.12", "expected": "3.99"}
    ]


def test_runtime_scope_keeps_optional_models_out_of_recording():
    for command in ("record", "host", "teleoperate", "find-port"):
        assert not {"torch", "transformers"} & env.command_packages([command])
    assert "transformers" in env.command_packages(["train"])
    assert "torch" in env.command_packages(["inference", "serve"])
    assert "torch" not in env.command_packages(["inference", "run"])
    assert not env.command_packages(["evaluate", "--help"])


def test_host_launch_does_not_inspect_missing_learning_packages(repository, monkeypatch, capsys):
    monkeypatch.setattr(env, "installed_versions", lambda: {"numpy": "2.2.6"})
    monkeypatch.setattr(env.sys, "prefix", str(repository))
    with env.runtime_environment(["host"]):
        pass
    assert not capsys.readouterr().err


def test_runtime_drift_prints_actionable_sync_without_install(repository, monkeypatch, capsys):
    monkeypatch.setattr(env, "installed_versions", lambda: {"torch": "2.11.0"})
    monkeypatch.setattr(env.sys, "prefix", str(repository))
    monkeypatch.setattr(env.subprocess, "run", lambda *a, **kw: pytest.fail("unexpected process"))
    with env.runtime_environment(["train"]):
        pass
    output = capsys.readouterr().err
    assert "2.11.0 → 2.15.0" in output and "alohamini env sync --role pc" in output


def test_sync_lock_excludes_runtime_and_other_syncs(tmp_path):
    with env.environment_lock(prefix=tmp_path):
        with env.environment_lock(prefix=tmp_path):
            with pytest.raises(RuntimeError, match="正在使用"):
                with env.environment_lock(exclusive=True, prefix=tmp_path):
                    pass
    with env.environment_lock(exclusive=True, prefix=tmp_path):
        with pytest.raises(RuntimeError, match="正在使用"):
            with env.environment_lock(prefix=tmp_path):
                pass


def test_legacy_process_detection_is_environment_specific(tmp_path):
    prefix, other, proc = tmp_path / "env", tmp_path / "other", tmp_path / "proc"
    for root in (prefix, other):
        (root / "bin").mkdir(parents=True)
        (root / "bin/python3.12").touch()
    for pid, executable in [(123, prefix), (124, other), (os.getpid(), prefix)]:
        directory = proc / str(pid)
        directory.mkdir(parents=True)
        (directory / "exe").symlink_to(executable / "bin/python3.12")
    assert env.active_python_processes(prefix, proc) == [123]


def test_venv_process_detection_follows_invocation_not_only_shared_python_binary(tmp_path):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    python = tmp_path / "base/bin/python3.12"
    python.parent.mkdir(parents=True)
    python.touch()
    process = tmp_path / "proc/123"
    process.mkdir(parents=True)
    (process / "exe").symlink_to(python)
    (process / "cmdline").write_bytes(str(prefix / "bin/python").encode() + b"\0task.py")
    assert env.active_python_processes(prefix, process.parent) == [123]


def distribution(name, version, requires=()):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=list(requires))


def test_visible_distribution_takes_precedence_over_inherited_site_packages(monkeypatch):
    monkeypatch.setattr(
        env.metadata,
        "distributions",
        lambda: [
            distribution("torch", "2.15.0"),
            distribution("torch", "2.11.0"),
        ],
    )
    assert env.installed_versions()["torch"] == "2.15.0"


def test_extra_package_conflicts_use_planned_versions_and_markers(monkeypatch):
    distributions = [
        distribution("torch", "2.11.0"),
        distribution(
            "legacy",
            "1",
            ["torch<2.12", 'absent; extra == "optional"', 'windows-only; sys_platform == "win32"'],
        ),
    ]
    monkeypatch.setattr(env.metadata, "distributions", lambda: distributions)
    assert not env.projected_conflicts({"install": []})
    conflict = env.projected_conflicts(
        {"install": [{"metadata": {"name": "torch", "version": "2.15.0"}}]}
    )
    assert len(conflict) == 1 and "legacy requires torch<2.12" in conflict[0]


def test_editable_development_version_satisfies_unversioned_platform_requirement(monkeypatch):
    monkeypatch.setattr(
        env.metadata,
        "distributions",
        lambda: [
            distribution("alohamini-platform", "0.1.0.dev0"),
            distribution("alohamini-mujoco", "0.1.0", ["alohamini-platform[simulation]"]),
        ],
    )
    assert not env.projected_conflicts({"install": []})


@pytest.fixture
def sync_setup(repository, monkeypatch, tmp_path):
    prefix = tmp_path / "environment"
    (prefix / "conda-meta").mkdir(parents=True)
    monkeypatch.setattr(env.sys, "prefix", str(prefix))
    monkeypatch.setattr(env, "active_python_processes", lambda: [])
    monkeypatch.setenv("ALOHAMINI_WORKSPACE", str(tmp_path / "workspace"))
    distributions = [distribution("torch", "2.11.0"), distribution("transformers", "5.21.0")]
    monkeypatch.setattr(env.metadata, "distributions", lambda: distributions)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text(
                json.dumps(
                    {
                        "install": [
                            {"metadata": {"name": "torch", "version": "2.15.0"}},
                            {"metadata": {"name": "transformers", "version": "5.22.1"}},
                        ]
                    }
                )
            )
        elif "install" in command and "--require-hashes" in command:
            distributions[:] = [
                distribution("torch", "2.15.0+cu132"),
                distribution("transformers", "5.22.1"),
            ]
        return subprocess.CompletedProcess(command, 0, stdout="torch==2.11.0\n")

    monkeypatch.setattr(env.subprocess, "run", run)
    return repository, calls, distributions


def test_sync_checks_plan_installs_hashed_lock_and_verifies(sync_setup, tmp_path, capsys):
    root, calls, _ = sync_setup
    env.sync_environment(root, "pc")
    install = [c for c in calls if "install" in c]
    assert "--dry-run" in install[0] and "--require-hashes" in install[1]
    assert "--no-deps" in install[2] and str(root) + "[pc]" in install[2]
    assert all(c[0] == sys.executable for c in calls)
    assert calls[-1][-2:] == ["pip", "check"]
    assert env.inspect_environment(root, "pc")["matched"]
    assert not (tmp_path / "workspace").exists()
    output = capsys.readouterr()
    assert output.out == f"环境已同步：{sys.prefix}\n"
    assert not output.err


def test_already_matching_environment_leaves_no_record(sync_setup, monkeypatch, tmp_path):
    root, _, distributions = sync_setup
    distributions[:] = [
        distribution("torch", "2.15.0+cu132"),
        distribution("transformers", "5.22.1"),
    ]
    original = env.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if "--report" in command:
            Path(command[command.index("--report") + 1]).write_text('{"install": []}')
        return result

    monkeypatch.setattr(env.subprocess, "run", run)
    env.sync_environment(root, "pc")
    assert not (tmp_path / "workspace").exists()


def test_dry_run_does_not_install_or_create_success_record(sync_setup, tmp_path):
    root, calls, _ = sync_setup
    with env.environment_lock():
        env.sync_environment(root, "pc", dry_run=True)
    assert len(calls) == 2 and "--dry-run" in calls[-1]
    assert not (tmp_path / "workspace").exists()


def test_extra_conflict_stops_before_download_or_install(sync_setup):
    root, calls, distributions = sync_setup
    distributions.append(distribution("lerobot", "0.6.1", ["torch<2.12"]))
    with pytest.raises(ValueError, match="lerobot requires torch<2.12"):
        env.sync_environment(root, "pc")
    assert len(calls) == 1 and calls[0][-1] == "check"


def test_active_task_prevents_environment_changes(sync_setup, monkeypatch):
    root, calls, _ = sync_setup
    monkeypatch.setattr(env, "active_python_processes", lambda: [123])
    with pytest.raises(RuntimeError, match="123"):
        env.sync_environment(root, "pc")
    assert not calls


def test_checkout_change_during_resolution_prevents_install(sync_setup, monkeypatch, tmp_path):
    root, calls, _ = sync_setup
    original = env.subprocess.run

    def change(command, **kwargs):
        result = original(command, **kwargs)
        if "--dry-run" in command:
            (root / "env/pc-linux-64.lock").write_text("torch==2.16.0\n")
        return result

    monkeypatch.setattr(env.subprocess, "run", change)
    with pytest.raises(RuntimeError, match="依赖文件发生变化"):
        env.sync_environment(root, "pc")
    assert len(calls) == 2
    assert not (tmp_path / "workspace").exists()


@pytest.mark.parametrize("failure", ["install", "check", "versions", "interrupt"])
def test_failed_sync_preserves_diagnostics(
    sync_setup,
    monkeypatch,
    tmp_path,
    capsys,
    failure,
):
    root, _, distributions = sync_setup
    original = env.subprocess.run

    def fail(command, **kwargs):
        if (
            failure in {"install", "interrupt"}
            and "install" in command
            and "--dry-run" not in command
        ):
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise subprocess.CalledProcessError(1, command)
        if failure == "check" and command[-2:] == ["pip", "check"]:
            raise subprocess.CalledProcessError(1, command)
        result = original(command, **kwargs)
        if failure == "versions" and command[-2:] == ["pip", "check"]:
            distributions[:] = []
        return result

    monkeypatch.setattr(env.subprocess, "run", fail)
    error = {"versions": RuntimeError, "interrupt": KeyboardInterrupt}.get(
        failure, subprocess.CalledProcessError
    )
    with pytest.raises(error):
        env.sync_environment(root, "pc")
    snapshots = list((tmp_path / "workspace/logs/environment").glob("*/before.txt"))
    assert len(snapshots) == 1
    assert snapshots[0].read_text() == "torch==2.11.0\n"
    assert json.loads(snapshots[0].with_name("plan.json").read_text())["install"]
    assert error.__name__ in snapshots[0].with_name("error.txt").read_text()
    assert str(snapshots[0].parent) in capsys.readouterr().err
    assert not snapshots[0].with_name("after.json").exists()


def test_cli_check_returns_nonzero_for_drift(repository, monkeypatch, capsys):
    monkeypatch.setattr(env, "installed_versions", lambda: {})
    assert env.main(["check", "--role", "pc", "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["matched"]


def test_training_reports_drift_before_loading_broken_framework(repository, tmp_path):
    (repository / "env/pc-linux-64.lock").write_text("torch==999.0\n")
    (tmp_path / "torch.py").write_text("raise ImportError('simulated broken torch')\n")
    script = f"""
import sys, runpy
from pathlib import Path
from alohamini import environment
environment.checkout = lambda: Path({str(repository)!r})
environment.platform.system = lambda: 'Linux'
environment.platform.machine = lambda: 'x86_64'
sys.path.insert(0, {str(tmp_path)!r})
sys.argv = ['train']
runpy.run_module('alohamini.learning.train', run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", script], text=True,
                            capture_output=True, timeout=10)
    assert result.returncode != 0
    assert result.stderr.index("alohamini env sync") < result.stderr.index("simulated broken torch")
