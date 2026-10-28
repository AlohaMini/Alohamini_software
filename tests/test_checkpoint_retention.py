import pytest

from alohamini.learning.training_state import prune_checkpoints


def completed(root, name):
    path = root / "checkpoints" / name
    (path / "training_state").mkdir(parents=True)
    (path / "training_state/state.pt").touch()
    (path / "pretrained_model").mkdir()
    (path / "pretrained_model/policy.json").write_text("{}")
    return path


def test_retention_preserves_pending_external_and_current_checkpoints(tmp_path):
    old = completed(tmp_path, "000001")
    current = completed(tmp_path, "000002")
    newer = completed(tmp_path, "000003")
    pending = completed(tmp_path, "000004.pending-test")
    partial = tmp_path / "checkpoints/000005"
    partial.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / "checkpoints/000000").symlink_to(external)
    (tmp_path / "checkpoints/last").symlink_to(current)
    prune_checkpoints(tmp_path, 1)
    assert not old.exists()
    for path in (current, newer, pending, partial, external):
        assert path.exists()
    (tmp_path / "checkpoints/last").unlink()
    (tmp_path / "checkpoint").symlink_to(newer / "pretrained_model")
    prune_checkpoints(tmp_path, 1)
    assert not current.exists()
    assert (tmp_path / "checkpoint/policy.json").is_file()


def test_retention_disabled_and_invalid_values_do_not_delete(tmp_path):
    path = completed(tmp_path, "000001")
    prune_checkpoints(tmp_path, None)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            prune_checkpoints(tmp_path, value)
    assert path.exists()


def test_retention_cli():
    from alohamini.learning.train_config import parse_training_args

    config, _ = parse_training_args(
        ["--dataset.root=/tmp/data", "--output_dir=/tmp/run", "--keep_last_checkpoints=1"]
    )
    assert config["keep_last_checkpoints"] == 1


def test_failed_save_keeps_previous_until_replacement_is_complete(tmp_path, monkeypatch):
    import torch

    from alohamini.learning import training_state as state

    def save_model(path, *args, **kwargs):
        path.mkdir()
        (path / "policy.json").write_text("{}")

    monkeypatch.setattr(state, "save_checkpoint", save_model)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    cfg = {"steps": 2, "keep_last_checkpoints": 1}
    first = state.save_training_checkpoint(tmp_path, 1, cfg, model, optimizer, {}, None, {})
    original = torch.save

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        state.save_training_checkpoint(tmp_path, 2, cfg, model, optimizer, {}, None, {})
    assert first.is_dir()
    assert (tmp_path / "checkpoints/last").resolve() == first
    monkeypatch.setattr(torch, "save", original)
    second = state.save_training_checkpoint(tmp_path, 2, cfg, model, optimizer, {}, None, {})
    assert not first.exists()
    assert (tmp_path / "checkpoints/last").resolve() == second
