"""Local dataset options preserve existing training and resume contracts."""

import json
from types import SimpleNamespace

import pytest
import torch

from alohamini.learning.loading import resolve_data_pipeline
from alohamini.learning.train_config import parse_training_args, resolve_dataset_options
from alohamini.learning.training_state import load_training_checkpoint

ARGS = ["--dataset.root=/tmp/data", "--output_dir=/tmp/run", "--foreground"]


def test_cli_defaults_keep_indexed_training_and_flat_dataset_path():
    cfg, background = parse_training_args(ARGS)
    assert not background
    assert cfg["dataset"] == "/tmp/data"
    assert cfg["dataset_streaming"] is False
    assert cfg["dataset_cache_size_mb"] == 64
    assert cfg["streaming_episode_pool_size"] == 8
    assert cfg["dataset_cache_block_rows"] == 128


def test_cli_overrides_saved_flat_dataset_options(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            dict(
                dataset="/tmp/data",
                output_dir="/tmp/run",
                dataset_streaming=False,
                dataset_cache_size_mb=32,
                streaming_episode_pool_size=2,
                dataset_cache_block_rows=64,
            )
        )
    )
    cfg, _ = parse_training_args(
        [
            "--config",
            str(path),
            "--dataset.streaming=true",
            "--dataset.cache_size_mb=12.5",
            "--dataset.streaming_episode_pool_size=3",
            "--dataset.cache_block_rows=16",
        ]
    )
    assert cfg["dataset"] == "/tmp/data"
    assert cfg["dataset_streaming"] is True
    assert cfg["dataset_cache_size_mb"] == 12.5
    assert cfg["streaming_episode_pool_size"] == 3
    assert cfg["dataset_cache_block_rows"] == 16
    assert not any(key.startswith("dataset.") for key in cfg)
    path.write_text(json.dumps(cfg))
    reloaded, _ = parse_training_args(["--config", str(path)])
    assert reloaded == cfg


@pytest.mark.parametrize(
    "option",
    [
        "--dataset.streaming=1",
        "--dataset.cache_size_mb=0",
        "--dataset.cache_size_mb=-1",
        "--dataset.cache_size_mb=nan",
        "--dataset.cache_size_mb=inf",
        "--dataset.streaming_episode_pool_size=0",
        "--dataset.streaming_episode_pool_size=1.5",
        "--dataset.cache_block_rows=-1",
    ],
)
def test_cli_rejects_invalid_dataset_options(option):
    with pytest.raises(SystemExit):
        parse_training_args([*ARGS, option])


@pytest.mark.parametrize(
    "name,value",
    [
        ("dataset_streaming", 0),
        ("dataset_streaming", 1),
        ("dataset_streaming", "false"),
        ("dataset_streaming", None),
        ("dataset_cache_size_mb", True),
        ("dataset_cache_size_mb", 0),
        ("dataset_cache_size_mb", -0.5),
        ("dataset_cache_size_mb", float("nan")),
        ("dataset_cache_size_mb", float("inf")),
        ("dataset_cache_size_mb", "64"),
        ("dataset_cache_size_mb", 10**1000),
        ("streaming_episode_pool_size", True),
        ("streaming_episode_pool_size", 0),
        ("streaming_episode_pool_size", 1.0),
        ("dataset_cache_block_rows", False),
        ("dataset_cache_block_rows", 0),
        ("dataset_cache_block_rows", 16.5),
    ],
)
def test_python_and_json_options_are_strict(name, value):
    with pytest.raises(ValueError, match=name):
        resolve_dataset_options({name: value})


def test_json_config_is_validated_without_mutating_or_changing_loader_defaults(tmp_path):
    source = {"dataset": "/tmp/data"}
    resolved = resolve_dataset_options(source)
    assert source == {"dataset": "/tmp/data"}
    assert resolved["dataset_streaming"] is False
    overrides = dict(
        video_backend="pyav",
        camera_workers=0,
        video_cache_size=0,
        return_uint8=False,
        dataloader_multiprocessing_context="spawn",
    )
    assert resolve_data_pipeline(overrides) == overrides
    path = tmp_path / "config.json"
    path.write_text(json.dumps({**source, "output_dir": "/tmp/run", "dataset_streaming": "false"}))
    with pytest.raises(SystemExit):
        parse_training_args(["--config", str(path)])


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "checkpoints/000001"
    (path / "training_state").mkdir(parents=True)
    (path / "pretrained_model").mkdir()
    torch.save({"step": 1}, path / "training_state/state.pt")
    (path / "pretrained_model/policy.json").write_text(
        json.dumps(dict(sample_filter="episode_windows_v2", table_sha256={"data": "same"}))
    )
    saved = dict(
        output_dir=str(tmp_path),
        steps=2,
        optimizer={},
        scheduler={"type": "none"},
        world_size=1,
        gradient_accumulation_steps=1,
        distributed_backend="ddp",
    )
    samples = SimpleNamespace(sample_filter="episode_windows_v2", table_sha256={"data": "same"})
    return path, saved, samples


def save_config(path, saved):
    (path / "pretrained_model/train_config.json").write_text(json.dumps(saved))


def test_old_checkpoint_missing_dataset_options_resumes_with_new_defaults(checkpoint):
    path, saved, samples = checkpoint
    save_config(path, saved)
    current = {**resolve_dataset_options(saved), "resume": True}
    assert load_training_checkpoint(path, current, samples, None)[2]["step"] == 1
    assert "dataset_streaming" not in saved


@pytest.mark.parametrize("streaming", [False, True])
def test_resume_allows_cache_changes_without_changing_sampling(checkpoint, streaming):
    path, saved, samples = checkpoint
    saved = {**resolve_dataset_options(saved), "dataset_streaming": streaming}
    save_config(path, saved)
    current = {
        **saved,
        "resume": True,
        "dataset_cache_size_mb": 0.5,
        "dataset_cache_block_rows": 1,
    }
    assert load_training_checkpoint(path, current, samples, None)[2]["step"] == 1


@pytest.mark.parametrize(
    "saved_streaming,changed_key,changed_value",
    [
        (False, "dataset_streaming", True),
        (True, "dataset_streaming", False),
        (True, "streaming_episode_pool_size", 4),
    ],
)
def test_resume_rejects_changed_stream_sampling(
    checkpoint, saved_streaming, changed_key, changed_value
):
    path, saved, samples = checkpoint
    saved = {**resolve_dataset_options(saved), "dataset_streaming": saved_streaming}
    save_config(path, saved)
    with pytest.raises(ValueError, match=changed_key):
        load_training_checkpoint(
            path, {**saved, "resume": True, changed_key: changed_value}, samples, None
        )


def test_old_checkpoint_cannot_silently_switch_to_streaming(checkpoint):
    path, saved, samples = checkpoint
    save_config(path, saved)
    with pytest.raises(ValueError, match="dataset_streaming"):
        load_training_checkpoint(path, {**saved, "dataset_streaming": True}, samples, None)


def test_resume_ignores_pool_size_when_streaming_is_disabled(checkpoint):
    path, saved, samples = checkpoint
    save_config(path, saved)
    current = {**saved, "resume": True, "streaming_episode_pool_size": 3}
    assert load_training_checkpoint(path, current, samples, None)[2]["step"] == 1


def test_cache_options_do_not_weaken_dataset_fingerprint_checks(checkpoint):
    path, saved, samples = checkpoint
    save_config(path, saved)
    samples.table_sha256 = {"data": "changed"}
    with pytest.raises(ValueError, match="Dataset changed"):
        load_training_checkpoint(path, {**saved, "dataset_cache_size_mb": 16}, samples, None)


def test_cache_directory_cli_defaults_and_explicit_path():
    cfg, _ = parse_training_args(ARGS)
    assert cfg["dataset_cache_dir"] is None
    cfg, _ = parse_training_args([*ARGS, "--dataset.cache_dir=/mnt/ssd/training-cache"])
    assert cfg["dataset_cache_dir"] == "/mnt/ssd/training-cache"


@pytest.mark.parametrize("directory", ["", "  ", False, 1, []])
def test_cache_directory_rejects_non_paths(directory):
    with pytest.raises(ValueError, match="dataset_cache_dir"):
        resolve_dataset_options({"dataset_cache_dir": directory})


def test_old_checkpoint_allows_new_cache_directory_on_resume(checkpoint):
    path, saved, samples = checkpoint
    save_config(path, saved)
    current = {**saved, "resume": True, "dataset_cache_dir": "/mnt/ssd/cache"}
    assert load_training_checkpoint(path, current, samples, None)[2]["step"] == 1
