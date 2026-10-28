"""Version identity follows the loaded package, including uncommitted source."""

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from alohamini import software_version as version


@pytest.fixture(autouse=True)
def clear_identity_cache():
    version.software_identity.cache_clear()
    yield
    version.software_identity.cache_clear()


def package(tmp_path):
    root = tmp_path / "source/src/alohamini"
    root.mkdir(parents=True)
    (root / "__init__.py").write_bytes(b'"""Package."""\n')
    (root / "worker.py").write_bytes(b"VALUE = 1\n")
    (root / "hardware").mkdir()
    (root / "hardware/device.py").write_bytes(b"READY = True\n")
    return root


def test_copied_source_has_same_digest_without_git_or_location_dependence(tmp_path):
    root = package(tmp_path)
    copied = tmp_path / "copy/alohamini"
    shutil.copytree(root, copied)
    for directory in (".git", "docs", "tests", "__pycache__"):
        (copied / directory).mkdir()
        (copied / directory / "ignored.py").write_bytes(b"different ignored source\n")
    (copied / "notes.txt").write_text("not Python source")
    digest = version._source_sha256(root)
    assert digest == version._source_sha256(copied)
    assert len(digest) == 64


def test_same_git_revision_does_not_hide_source_content_or_path_changes(tmp_path, monkeypatch):
    root = package(tmp_path)
    monkeypatch.setattr(version, "_package_root", lambda: root)
    monkeypatch.setattr(version, "_package_version", lambda: "0.1.0.dev0")
    monkeypatch.setattr(version, "_git_revision", lambda _: "a" * 40)
    first = version.software_identity().copy()
    (root / "worker.py").write_bytes(b"VALUE = 2\n")
    version.software_identity.cache_clear()
    edited = version.software_identity().copy()
    (root / "worker.py").rename(root / "renamed.py")
    version.software_identity.cache_clear()
    renamed = version.software_identity()
    assert first["git_revision"] == edited["git_revision"] == renamed["git_revision"]
    assert len({item["source_sha256"] for item in (first, edited, renamed)}) == 3


def test_identity_helpers_run_only_on_first_call(monkeypatch, tmp_path):
    root = package(tmp_path)
    helpers = {
        "_package_root": Mock(return_value=root),
        "_package_version": Mock(return_value="test-version"),
        "_source_sha256": Mock(return_value="b" * 64),
        "_git_revision": Mock(return_value=None),
    }
    for name, helper in helpers.items():
        monkeypatch.setattr(version, name, helper)
    expected = {
        "schema_version": 1,
        "package_version": "test-version",
        "source_sha256": "b" * 64,
        "git_revision": None,
    }
    assert version.software_identity() == expected
    assert version.software_identity() == expected
    for helper in helpers.values():
        helper.assert_called_once()
    assert version.software_identity.cache_info().hits == 1


def test_unreadable_source_keeps_available_version_and_git_revision(tmp_path, monkeypatch):
    root = package(tmp_path)
    read_bytes = Path.read_bytes

    def unreadable(path):
        if path == root / "worker.py":
            raise PermissionError("unreadable Python source")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    monkeypatch.setattr(version, "_package_root", lambda: root)
    monkeypatch.setattr(version, "_package_version", lambda: "installed")
    monkeypatch.setattr(version, "_git_revision", lambda _: "c" * 40)
    assert version.software_identity() == {
        "schema_version": 1,
        "package_version": "installed",
        "source_sha256": None,
        "git_revision": "c" * 40,
    }


def test_missing_metadata_and_git_do_not_hide_source_digest(tmp_path, monkeypatch):
    root = package(tmp_path)
    metadata_version = Mock(side_effect=version.metadata.PackageNotFoundError("not installed"))
    monkeypatch.setattr(version.metadata, "version", metadata_version)
    monkeypatch.setattr(version, "_package_root", lambda: root)
    monkeypatch.setattr(version, "_git_output", Mock(side_effect=FileNotFoundError("no git")))
    identity = version.software_identity()
    assert identity["source_sha256"] == version._source_sha256(root)
    assert identity["package_version"] is None and identity["git_revision"] is None
    metadata_version.assert_called_once_with("alohamini-platform")


def test_git_only_identifies_repository_tracking_the_actual_package(tmp_path, monkeypatch):
    root = package(tmp_path)
    repository = root.parents[1]
    revision = "d" * 40
    git = Mock(side_effect=[f"{repository}\n{revision}", "src/alohamini/__init__.py"])
    monkeypatch.setattr(version, "_git_output", git)
    assert version._git_revision(root) == revision
    assert all(call.args[0] == root for call in git.call_args_list)

    # Finding a parent repo is insufficient if these copied package files are untracked.
    git.side_effect = [
        f"{repository}\n{revision}",
        subprocess.CalledProcessError(1, "git ls-files"),
    ]
    assert version._git_revision(root) is None


def test_git_subprocess_is_bounded_and_uses_package_directory_without_changing_environment(
    tmp_path, monkeypatch
):
    root = package(tmp_path)
    run = Mock(return_value=SimpleNamespace(stdout="result\n"))
    monkeypatch.setattr(version.subprocess, "run", run)
    environment = dict(version.os.environ)
    assert version._git_output(root, "rev-parse", "HEAD") == "result"
    command = run.call_args.args[0]
    options = run.call_args.kwargs
    assert command == ["git", "-C", str(root), "rev-parse", "HEAD"]
    assert 0 < options["timeout"] <= 1
    assert options["check"] and options["stderr"] == subprocess.DEVNULL
    assert not any(name.startswith("GIT_") for name in options["env"])
    assert dict(version.os.environ) == environment


@pytest.mark.parametrize("failure", [OSError("no package path"), RuntimeError("link loop")])
def test_missing_package_directory_keeps_installed_version(monkeypatch, failure):
    monkeypatch.setattr(version, "_package_root", Mock(side_effect=failure))
    monkeypatch.setattr(version, "_package_version", lambda: "installed")
    assert version.software_identity() == {
        "schema_version": 1,
        "package_version": "installed",
        "source_sha256": None,
        "git_revision": None,
    }


def test_missing_empty_or_externally_linked_source_is_not_reported_as_equal(tmp_path):
    assert version._source_sha256(tmp_path / "missing") is None
    assert version._source_sha256(tmp_path) is None
    root = package(tmp_path)
    target = tmp_path / "outside"
    target.mkdir()
    (target / "module.py").write_text("outside = True\n")
    (root / "linked").symlink_to(target, target_is_directory=True)
    assert version._source_sha256(root) is None
