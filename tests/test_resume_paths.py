import json
from types import SimpleNamespace

import pytest
import torch

from alohamini.learning.training_state import load_training_checkpoint


@pytest.mark.parametrize("different_directory", [False, True])
def test_resume_compares_output_directory_identity(tmp_path, different_directory):
    storage = tmp_path / "data"
    storage.mkdir()
    workspace = tmp_path / "workspace"
    workspace.symlink_to(storage, target_is_directory=True)
    run = storage / "runs/task"
    run.mkdir(parents=True)
    # A checkpoint can live on another disk while the run and logs retain their paths.
    checkpoints = tmp_path / "system/checkpoints"
    checkpoint = checkpoints / "000001"
    (checkpoint / "training_state").mkdir(parents=True)
    (checkpoint / "pretrained_model").mkdir()
    (run / "checkpoints").symlink_to(checkpoints, target_is_directory=True)
    saved = dict(
        output_dir=str(workspace / "runs/task"),
        steps=2,
        optimizer={},
        scheduler={"type": "none"},
        world_size=1,
        gradient_accumulation_steps=1,
        distributed_backend="ddp",
    )
    manifest = dict(sample_filter={}, table_sha256="same", training={}, config={})
    (checkpoint / "pretrained_model/train_config.json").write_text(json.dumps(saved))
    (checkpoint / "pretrained_model/policy.json").write_text(json.dumps(manifest))
    torch.save({"step": 1}, checkpoint / "training_state/state.pt")
    current = {**saved, "resume": True, "output_dir": str(run.resolve())}
    samples = SimpleNamespace(sample_filter={}, table_sha256="same")
    if different_directory:
        current["output_dir"] = str(tmp_path / "other-run")
        with pytest.raises(ValueError, match="Resume configuration changed.*output_dir"):
            load_training_checkpoint(checkpoint, current, samples, None)
    else:
        path, _, state = load_training_checkpoint(checkpoint, current, samples, None)
        assert path == checkpoint.resolve()
        assert state["step"] == 1
