"""Architecture-independent EMA math, buffer handling and checkpoint contracts."""

import json
from copy import deepcopy

import pytest
import torch

from alohamini.learning.ema import EMAConfig, EMAManager, validate_ema_setup


def test_transformer_ema_tracks_trainable_parameters_only_and_resumes():
    model = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0, batch_first=True)
    model.linear1.requires_grad_(False)
    config = EMAConfig(enable=True, decay=0.5)
    ema = EMAManager(model, config)
    assert not any(name.startswith("linear1.") for name in ema.parameters)
    initial = {name: value.clone() for name, value in ema.parameters.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(2)
    ema.update(model)
    for name, value in ema.parameters.items():
        torch.testing.assert_close(value, initial[name] + 1)
    resumed = EMAManager(model, config)
    resumed.load_state_dict(ema.state_dict())
    ema.update(model)
    resumed.update(model)
    assert ema.num_updates == resumed.num_updates == 2
    for name, value in ema.parameters.items():
        torch.testing.assert_close(resumed.parameters[name], value, rtol=0, atol=0)
    with resumed.average_parameters(model), torch.no_grad():
        assert torch.isfinite(model(torch.ones(2, 3, 8))).all()


def test_ema_warmup_matches_diffusers():
    from diffusers.training_utils import EMAModel

    model = torch.nn.Linear(1, 1)
    config = EMAConfig(enable=True, update_after_step=2, min_decay=0.1, max_decay=0.95)
    ema = EMAManager(model, config)
    reference = EMAModel(
        model.parameters(),
        decay=0.95,
        min_decay=0.1,
        update_after_step=2,
        use_ema_warmup=True,
        inv_gamma=1.0,
        power=0.75,
    )
    for _ in range(12):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.1)
        ema.update(model)
        reference.step(model.parameters())
        for shadow, expected in zip(ema.parameters.values(), reference.shadow_params, strict=True):
            torch.testing.assert_close(shadow, expected)


def test_fp32_shadow_preserves_small_updates_from_bfloat16():
    model = torch.nn.Linear(1, 1, bias=False).bfloat16()
    with torch.no_grad():
        model.weight.fill_(1)
    ema = EMAManager(model, EMAConfig(enable=True, decay=0.999))
    with torch.no_grad():
        model.weight.fill_(2)
    ema.update(model)
    shadow = ema.parameters["weight"]
    assert shadow.dtype == torch.float32
    assert shadow.item() == pytest.approx(1.001)
    with ema.average_parameters(model):
        assert model.weight.dtype == torch.bfloat16
    assert model.weight.item() == 2


def test_batchnorm_buffers_are_copied_and_swap_restores_on_failure():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    model[1].eval()  # Preserve mixed per-module modes, too.
    ema = EMAManager(model, EMAConfig(enable=True, decay=0.5))
    model(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    ema.update(model)
    assert ema.buffers["0.num_batches_tracked"].item() == 1
    torch.testing.assert_close(ema.buffers["0.running_mean"], model[0].running_mean)
    with torch.no_grad():
        model[0].running_mean.add_(3)
        model[1].weight.add_(2)
    before = deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="evaluation failed"), ema.average_parameters(model):
        assert not model.training
        torch.testing.assert_close(model[0].running_mean, ema.buffers["0.running_mean"])
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with ema.average_parameters(model):
                pass
        raise RuntimeError("evaluation failed")
    assert model.training and model[0].training and not model[1].training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_ema_state_rejects_changed_schema_before_mutation():
    model = torch.nn.Linear(2, 1)
    ema = EMAManager(model, EMAConfig(enable=True))
    state = ema.state_dict()
    state["parameters"]["weight"].fill_(7)
    state["parameters"]["bias"] = torch.zeros(2)
    before = ema.parameters["weight"].clone()
    with pytest.raises(ValueError, match="shapes"):
        ema.load_state_dict(state)
    torch.testing.assert_close(ema.parameters["weight"], before)
    with pytest.raises(ValueError, match="configuration changed"):
        EMAManager(model, EMAConfig(enable=True, decay=0.5)).load_state_dict(ema.state_dict())
    model.bias.requires_grad_(False)
    with pytest.raises(ValueError, match="names or shapes"):
        ema.update(model)


@pytest.mark.parametrize(
    "settings",
    [
        {"enable": "false"},
        {"decay": 1},
        {"decay": float("nan")},
        {"power": 0},
        {"update_after_step": -1},
        {"unknown": 1},
        {"min_decay": 0.9, "max_decay": 0.8},
    ],
)
def test_invalid_ema_configuration(settings):
    with pytest.raises(ValueError):
        EMAConfig.from_dict(settings)


def test_ema_cli_and_backend_validation(tmp_path):
    from alohamini.learning.train_config import parse_training_args

    cfg, _ = parse_training_args(
        [
            "--dataset.root=/tmp/data",
            f"--output_dir={tmp_path / 'run'}",
            "--ema.enable=true",
            "--ema.decay=0.99",
            "--ema.use_for_eval=false",
        ]
    )
    assert cfg["ema"] == {"enable": True, "decay": 0.99, "use_for_eval": False}
    assert validate_ema_setup({**cfg, "distributed_backend": "fsdp2"}).enable


def test_legacy_dp_ema_checkpoint_is_rejected(tmp_path):
    from alohamini.learning.checkpoint import read_checkpoint
    from alohamini.learning.train_config import parse_training_args

    (tmp_path / "policy.json").write_text(
        json.dumps(
            {
                "format": "alohamini-policy",
                "version": 1,
                "kind": "diffusion",
                "config": {"use_ema": True},
            }
        )
    )
    with pytest.raises(ValueError, match="Legacy DP EMA checkpoints"):
        read_checkpoint(tmp_path)
    with pytest.raises(SystemExit):
        parse_training_args(
            [
                "--dataset.root=/tmp/data",
                "--output_dir=/tmp/run",
                "--policy.use_ema=true",
            ]
        )


def test_failed_ema_export_restores_online_model_and_does_not_publish(tmp_path, monkeypatch):
    import alohamini.learning.training_state as checkpointing

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ema = EMAManager(model, EMAConfig(enable=True, decay=0.5))
    ema.update(model)
    with torch.no_grad():
        model.weight.add_(2)
    online = model.weight.detach().clone()

    def save(path, model, *args, **kwargs):
        path.mkdir()
        if path.name == "ema_pretrained_model":
            torch.testing.assert_close(model.weight, ema.parameters["weight"])
            raise OSError("EMA export failed")

    monkeypatch.setattr(checkpointing, "save_checkpoint", save)
    with pytest.raises(OSError, match="EMA export failed"):
        checkpointing.save_training_checkpoint(
            tmp_path,
            1,
            {"steps": 2, "ema": {"enable": True}},
            model,
            optimizer,
            {},
            None,
            {},
            ema=ema,
        )
    torch.testing.assert_close(model.weight, online, rtol=0, atol=0)
    assert model.training
    assert not (tmp_path / "checkpoint").exists()
    assert not (tmp_path / "checkpoint-ema").exists()
    assert not (tmp_path / "checkpoints/last").exists()
