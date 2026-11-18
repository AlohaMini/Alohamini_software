"""Reject incompatible SDK stacks before loading native libraries."""

from importlib.metadata import PackageNotFoundError

import pytest

from alohamini.simulation import runtime


def installed(monkeypatch, requirements, versions):
    monkeypatch.setattr(runtime, "requires", lambda _: requirements)

    def version(name):
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(runtime, "version", version)


def test_supported_prerelease_and_optional_requirements(monkeypatch):
    installed(
        monkeypatch,
        [
            "alohamini-platform[simulation]",
            "sapien==3.0.0b1",
            "numpy>=1.26,<2",
            'extra-only; extra == "test"',
        ],
        {"alohamini-platform": "0.1.0.dev0", "sapien": "3.0.0b1", "numpy": "1.26.4"},
    )
    runtime.check_backend_runtime("alohamini-robotwin")


def test_reports_all_incompatible_dependencies_and_environment(monkeypatch):
    installed(
        monkeypatch,
        ["sapien==3.0.0b1", "numpy<2", "mplib==0.2.1"],
        {"sapien": "3.0.3", "numpy": "2.2.6"},
    )
    with pytest.raises(ImportError) as caught:
        runtime.check_backend_runtime("alohamini-robotwin")
    message = str(caught.value)
    assert "sapien==3.0.0b1 (found 3.0.3)" in message
    assert "numpy<2 (found 2.2.6)" in message
    assert "mplib==0.2.1 (found missing)" in message
    assert "docs/install.md" in message


def test_skips_requirements_for_other_platforms(monkeypatch):
    installed(monkeypatch, ['missing; sys_platform == "not-a-platform"'], {})
    runtime.check_backend_runtime("alohamini-maniskill")
