"""Resume discards only an uncommitted recording transaction, never formal history."""

import contextlib
import fcntl
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from test_dataset_tools import hashes
from test_recording_stats_resume import no_history_content
from test_recording_storage import add_episode, create, metadata, rows

from alohamini.datasets.record import LocalDataset


def resume(root, **options):
    return LocalDataset(
        root,
        **{"fps": 30, "task": "pick", "robot_metadata": metadata(), **options},
        resume=True,
    )


def formal_hashes(root):
    return {
        name: value for name, value in hashes(root).items() if not name.startswith(".recording/")
    }


def tree_signature(root):
    result = {}
    for path in root.rglob("*"):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            result[relative] = ("link", str(path.readlink()))
        elif path.is_dir():
            result[relative] = ("directory",)
        else:
            result[relative] = ("file", hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def interrupt(root, stage, *, committed=1):
    """Make real writer artifacts, releasing locks as process exit would do."""
    if committed:
        if not root.exists():
            create(root, episodes=committed, cameras=("forward",))
        dataset = resume(root)
    else:
        dataset = LocalDataset(root, fps=30, task="pick", robot_metadata=metadata())
    add_episode(dataset, 3)
    if stage == "journal":
        dataset._writer._finish_writer()
        dataset._failed = dataset._writer.save_failed = True
    elif stage == "encoded":
        with (
            patch(
                "alohamini.datasets.lerobotv3.publish_recorded_episode",
                side_effect=OSError("interrupted before publication"),
            ),
            pytest.raises(OSError, match="interrupted before publication"),
        ):
            dataset.save_episode()
    elif stage == "partial_publish":
        replace = Path.replace

        def before_commit(path, destination):
            if path.name == "info.json" and path.parent.parent.name == "v3":
                raise OSError("interrupted before commit marker")
            return replace(path, destination)

        with (
            patch("pathlib.Path.replace", new=before_commit),
            pytest.raises(OSError, match="interrupted before commit marker"),
        ):
            dataset.save_episode()
    elif stage == "after_commit":
        with (
            patch(
                "alohamini.datasets.record.shutil.rmtree",
                side_effect=OSError("interrupted final cleanup"),
            ),
            pytest.raises(OSError, match="interrupted final cleanup"),
        ):
            dataset.save_episode()
    else:
        raise AssertionError(stage)
    dataset.close()
    episodes = root / ".recording/episodes"
    return next(episodes.iterdir())


@pytest.mark.parametrize("stage", ["journal", "encoded", "partial_publish"])
def test_resume_discards_uncommitted_attempt_and_appends_without_history_reads(tmp_path, stage):
    root = tmp_path / "record"
    create(root, episodes=1, cameras=("forward",))
    committed = formal_hashes(root)
    interrupt(root, stage)
    with no_history_content(root), contextlib.closing(resume(root)) as dataset:
        assert (dataset.num_episodes, dataset.total_frames) == (1, 3)
        assert not list((root / ".recording/episodes").iterdir())
        assert formal_hashes(root) == committed
        add_episode(dataset, 2)
        dataset.save_episode()
        assert (dataset.num_episodes, dataset.total_frames) == (2, 5)
    values = rows(root)
    assert [row["index"] for row in values] == list(range(5))
    assert [row["episode_index"] for row in values] == [0, 0, 0, 1, 1]
    assert [row["frame_index"] for row in values] == [0, 1, 2, 0, 1]
    current = formal_hashes(root)
    for name, digest in committed.items():
        if name.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
            assert current[name] == digest
    assert not (root / ".recording").exists()


def test_cleanup_failure_after_commit_preserves_saved_episode_and_all_formal_files(tmp_path):
    root = tmp_path / "record"
    interrupt(root, "after_commit")
    committed = formal_hashes(root)
    assert json.loads((root / "meta/info.json").read_text())["total_episodes"] == 2
    with no_history_content(root), contextlib.closing(resume(root)) as dataset:
        assert (dataset.num_episodes, dataset.total_frames) == (2, 6)
        assert not list((root / ".recording/episodes").iterdir())
        assert formal_hashes(root) == committed
        add_episode(dataset, 2)
        dataset.save_episode()
    assert [row["episode_index"] for row in rows(root)] == [0, 0, 0, 1, 1, 1, 2, 2]
    current = formal_hashes(root)
    for name, digest in committed.items():
        if name.startswith(("data/", "videos/", "meta/episodes/", "meta/safety/")):
            assert current[name] == digest


@pytest.mark.parametrize("stage", ["journal", "partial_publish"])
def test_first_uncommitted_episode_without_global_commit_restarts_at_zero(tmp_path, stage):
    root = tmp_path / "record"
    interrupt(root, stage, committed=0)
    assert not (root / "meta/info.json").exists()
    with no_history_content(root), contextlib.closing(resume(root)) as dataset:
        assert (dataset.num_episodes, dataset.total_frames) == (0, 0)
        add_episode(dataset, 2)
        dataset.save_episode()
        assert (dataset.num_episodes, dataset.total_frames) == (1, 2)
    assert [row["index"] for row in rows(root)] == [0, 1]
    assert [row["episode_index"] for row in rows(root)] == [0, 0]


def test_explicit_recovery_mode_keeps_interrupted_files_unchanged(tmp_path):
    root = tmp_path / "record"
    interrupt(root, "journal")
    before = tree_signature(root)
    with pytest.raises(RuntimeError, match="Interrupted|recover"):
        resume(root, discard_interrupted=False)
    assert tree_signature(root) == before


@pytest.mark.parametrize("changed", ["fps", "task", "calibration"])
def test_contract_mismatch_is_checked_before_removing_interrupted_data(tmp_path, changed):
    root = tmp_path / "record"
    interrupt(root, "journal")
    options = {"fps": 25} if changed == "fps" else {"task": "different task"}
    if changed == "calibration":
        robot = deepcopy(metadata())
        robot["motors"]["arm_left_shoulder_pan"]["range_min"] = 999
        options = {"robot_metadata": robot}
    before = tree_signature(root)
    with pytest.raises(ValueError, match="schema|calibration|changed"):
        resume(root, **options)
    assert tree_signature(root) == before


@pytest.mark.parametrize(
    "unsafe",
    [
        "unknown_path",
        "unknown_episode_file",
        "symlink",
        "missing_manifest",
        "bad_manifest",
        "missing_backup",
    ],
)
def test_unknown_or_unverifiable_transaction_is_retained_without_partial_deletion(tmp_path, unsafe):
    root = tmp_path / "record"
    stage = (
        "journal"
        if unsafe in {"unknown_path", "unknown_episode_file", "symlink"}
        else "partial_publish"
    )
    episode = interrupt(root, stage)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "sentinel").write_bytes(b"unrelated user data")
    if unsafe == "unknown_path":
        (root / ".recording/unrecognized.keep").write_text("keep")
    elif unsafe == "unknown_episode_file":
        (episode / "notes.keep").write_text("keep")
    elif unsafe == "symlink":
        (episode / "unrelated-link").symlink_to(outside, target_is_directory=True)
    elif unsafe == "missing_manifest":
        (episode / "publish.json").unlink()
    elif unsafe == "bad_manifest":
        manifest = json.loads((episode / "publish.json").read_text())
        manifest["files"].append("../external/sentinel")
        (episode / "publish.json").write_text(json.dumps(manifest))
    else:
        (episode / "backup/meta/stats.json").unlink()
    before, external = tree_signature(root), tree_signature(outside)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        resume(root)
    assert tree_signature(root) == before
    assert tree_signature(outside) == external


@pytest.mark.parametrize("value", [0, 1, "False", None])
def test_discard_interrupted_switch_requires_a_bool_without_deleting_files(tmp_path, value):
    root = tmp_path / "record"
    interrupt(root, "journal")
    before = tree_signature(root)
    with pytest.raises((TypeError, ValueError), match="discard_interrupted|bool"):
        resume(root, discard_interrupted=value)
    assert tree_signature(root) == before


def test_active_staging_writer_lock_prevents_cleanup_and_releases_outer_lock(tmp_path):
    root = tmp_path / "record"
    interrupt(root, "journal")
    before = tree_signature(root)
    with (root / ".recording/recording.lock").open("a") as live:
        fcntl.flock(live, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises((BlockingIOError, RuntimeError)):
            resume(root)
        assert tree_signature(root) == before
        with (root / "recording.lock").open("a") as outer:
            fcntl.flock(outer, fcntl.LOCK_EX | fcntl.LOCK_NB)
