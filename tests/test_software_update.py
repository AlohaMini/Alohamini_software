"""通过真实 Git 工作区验证增量同步及拒绝覆盖用户修改。"""

import subprocess
from pathlib import Path

import pytest

from alohamini.software_update import apply_bundle, git, prepare_bundle


@pytest.fixture
def repositories(tmp_path):
    pc, robot = tmp_path / "pc", tmp_path / "robot"
    pc.mkdir()
    git(pc, "init", "-b", "feat/macos")
    git(pc, "config", "user.email", "test@local")
    git(pc, "config", "user.name", "测试")
    package = pc / "src/alohamini"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n")
    git(pc, "add", ".")
    git(pc, "commit", "-m", "基础代码")
    base = git(pc, "rev-parse", "HEAD")
    subprocess.run(["git", "clone", str(pc), str(robot)], check=True, capture_output=True)
    git(pc, "switch", "-c", "feat/guided-workflow")
    (package / "__init__.py").write_text("VALUE = 2\n")
    git(pc, "commit", "-am", "图形工作流")
    return pc, robot, base


def test_incremental_bundle_preserves_original_branch_and_workspace(repositories, tmp_path):
    pc, robot, base = repositories
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "device.json").write_text("原有身份和标定")
    package = prepare_bundle(pc, base, tmp_path / "software.bundle")
    previous = apply_bundle(robot, package, tmp_path)
    assert previous == dict(commit=base, branch="feat/macos")
    assert git(robot, "rev-parse", "HEAD") == package["commit"]
    assert git(robot, "rev-parse", "feat/macos") == base
    assert (workspace / "device.json").read_text() == "原有身份和标定"
    assert not git(robot, "status", "--porcelain")


def test_rejects_dirty_and_changed_robot_and_bad_bundle(repositories, tmp_path):
    pc, robot, base = repositories
    package = prepare_bundle(pc, base, tmp_path / "software.bundle")
    (robot / "user-file").write_text("用户修改")
    with pytest.raises(ValueError, match="未提交修改"):
        apply_bundle(robot, package, tmp_path)
    (robot / "user-file").unlink()
    with pytest.raises(ValueError, match="代码已变化"):
        apply_bundle(robot, {**package, "base": "0" * 40}, tmp_path)
    with pytest.raises(subprocess.CalledProcessError):
        apply_bundle(robot, {**package, "bundle": "YmFk"}, tmp_path)
    assert git(robot, "rev-parse", "HEAD") == base
    (pc / "uncommitted").write_text("未提交")
    with pytest.raises(ValueError, match="未提交修改"):
        prepare_bundle(pc, base, tmp_path / "retry.bundle")
