"""Dependency proposals preserve deployment constraints and executable lock integrity."""

import shutil
from datetime import UTC, datetime, timedelta

import pytest
from packaging.version import Version

from env.dependencies import (
    ROOT,
    check,
    declared,
    dependency_group,
    exact_version,
    latest_release,
    pc_input,
    policy,
    propose,
    refresh,
    requirements,
)


@pytest.fixture
def manifests(tmp_path):
    shutil.copy(ROOT / "pyproject.toml", tmp_path)
    shutil.copytree(ROOT / "env", tmp_path / "env", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def test_nested_extras_cover_native_algorithms_and_keep_host_light():
    pc = declared(ROOT, policy(ROOT)["pc_extras"])
    assert {"torch", "transformers", "timm", "ftfy", "protobuf", "pyyaml"} <= pc.keys()
    assert set(declared(ROOT, ["host"])) == {
        "numpy",
        "opencv-python-headless",
        "pyzmq",
        "pyserial",
        "feetech-servo-sdk",
    }


def test_project_pin_change_requires_lock_regeneration(manifests):
    path = manifests / "pyproject.toml"
    old = exact_version(requirements(manifests / "env/pc-linux-64.lock")["transformers"][0])
    version = Version(old)
    new = f"{version.major}.{version.minor + 1}.0"
    path.write_text(path.read_text().replace(f"transformers=={old}", f"transformers=={new}"))
    with pytest.raises(ValueError, match="stale"):
        check(manifests)
    (manifests / "env/pc-linux-64.in").write_text(pc_input(manifests))
    with pytest.raises(ValueError, match="violates project"):
        check(manifests)


def test_pin_cannot_override_project_range(manifests):
    path = manifests / "env/dependencies.toml"
    old = exact_version(requirements(manifests / "env/pc-linux-64.lock")["torch"][0])
    path.write_text(path.read_text().replace(f"torch=={old}", "torch==99999.0.0"))
    with pytest.raises(ValueError, match="violates project"):
        pc_input(manifests)


def test_lock_requires_hashes(manifests):
    path = manifests / "env/hardware.lock"
    path.write_text("pyserial==3.5\nfeetech-servo-sdk==1.0.0\n")
    with pytest.raises(ValueError, match="Missing SHA256"):
        check(manifests)


def test_lock_includes_reject_conflicts_and_cycles(tmp_path):
    child = tmp_path / "child.lock"
    root = tmp_path / "root.lock"
    child.write_text("example==1\n")
    root.write_text("-r child.lock\nexample==2\n")
    with pytest.raises(ValueError, match="Conflicting"):
        requirements(root)
    child.write_text("-r root.lock\n")
    with pytest.raises(ValueError, match="Recursive"):
        requirements(root)


def release(date, **kwargs):
    return [{"upload_time_iso_8601": date.isoformat(), "requires_python": ">=3.10", **kwargs}]


def test_candidate_excludes_yanked_fresh_prerelease_major_and_incompatible_python():
    cutoff = datetime(2026, 10, 25, tzinfo=UTC)
    old = cutoff - timedelta(days=10)
    data = {
        "releases": {
            "5.1.0": release(old),
            "5.2.0": release(old),
            "5.3.0": release(old, yanked=True),
            "5.4.0": release(cutoff + timedelta(days=1)),
            "5.5.0rc1": release(old),
            "6.0.0": release(old),
            "5.6.0": release(old, requires_python=">=3.13"),
            "5.7.0": [],
            "not-a-version": release(old),
        }
    }
    assert latest_release(data, "5.0.0", "3.12.0", cutoff) == "5.2.0"


def test_groups_do_not_move_hardware_or_compute_into_models():
    groups = policy(ROOT)["groups"]
    assert dependency_group("transformers", groups) == "models"
    assert dependency_group("nvidia-cudnn-cu13", groups) == "compute"
    assert dependency_group("numpy", groups) == "host"
    assert dependency_group("pyarrow", groups) == "tools"


def test_proposal_updates_matching_group_only_and_all_shared_declarations(manifests):
    before = (manifests / "pyproject.toml").read_text()
    old = exact_version(
        requirements(manifests / "env/pc-linux-64.lock")["opencv-python-headless"][0]
    )
    version = Version(old)
    new = f"{version.major}.{version.minor + 1}.0.1"

    def fetch(name):
        version = new if name == "opencv-python-headless" else "0.0.0"
        return {"releases": {version: release(datetime(2020, 1, 1, tzinfo=UTC))}}

    changes = propose(manifests, "host", fetch=fetch)
    after = (manifests / "pyproject.toml").read_text()
    assert changes == [{"package": "opencv-python-headless", "old": old, "new": new}]
    assert after == before.replace(
        f"opencv-python-headless=={old}", f"opencv-python-headless=={new}"
    )


def test_failed_resolution_restores_all_dependency_files(manifests, monkeypatch):
    paths = [manifests / "pyproject.toml", *sorted((manifests / "env").glob("*"))]
    before = {p: p.read_bytes() for p in paths if p.is_file()}

    def fail(*args, **kwargs):
        (manifests / "env/pc-linux-64.lock").write_text("incomplete")
        raise RuntimeError("resolver failed")

    monkeypatch.setattr("env.dependencies.compile_lock", fail)
    with pytest.raises(RuntimeError, match="resolver failed"):
        refresh(manifests)
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize(
    "selected,processes,error",
    [
        ("0", "", ValueError),
        ("0,0", "", ValueError),
        ("0,2", "", ValueError),
        ("0,1", "GPU-a, 1234\n", RuntimeError),
    ],
)
def test_gpu_validation_rejects_invalid_or_occupied_devices(selected, processes, error):
    from env.gpu_preflight import validate_devices

    with pytest.raises(error):
        validate_devices(selected, "0, GPU-a\n1, GPU-b\n", processes)


def test_gpu_validation_ignores_jobs_on_unselected_devices():
    from env.gpu_preflight import validate_devices

    validate_devices("0,1", "0, GPU-a\n1, GPU-b\n2, GPU-c\n", "GPU-c, 1234\n")
