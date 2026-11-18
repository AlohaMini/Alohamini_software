"""An explicit compute policy must cover backward and checkpoint recomputation."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from alohamini.learning.precision import resolve_tf32, tf32_context


def test_opt_in_fp32_joint_attention_survives_outer_autocast(monkeypatch):
    from types import SimpleNamespace

    from alohamini.policies.pi05.gemma_pytorch import joint_attention

    original = torch.nn.functional.scaled_dot_product_attention
    seen = []

    def attention(query, key, value, **kwargs):
        seen.append((query.dtype, key.dtype, value.dtype, torch.is_autocast_enabled("cpu")))
        return original(query, key, value, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", attention)
    query = torch.randn(2, 2, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(2, 1, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output, _ = joint_attention(
            SimpleNamespace(num_key_value_groups=2),
            query,
            key,
            value,
            torch.zeros(2, 1, 3, 3),
            0.5,
            fp32=True,
        )
    assert seen == [(torch.float32, torch.float32, torch.float32, False)]
    assert output.dtype == torch.bfloat16
    output.float().square().mean().backward()
    assert all(torch.isfinite(tensor.grad).all() for tensor in (query, key, value))


def test_tf32_covers_forward_backward_recompute_and_restores_caller():
    events = []

    class Record(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            events.append(
                ("forward", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            )
            ctx.save_for_backward(value)
            return value.square()

        @staticmethod
        def backward(ctx, grad):
            events.append(
                ("backward", torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
            )
            return grad * ctx.saved_tensors[0] * 2

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    value = torch.tensor([2.0], requires_grad=True)
    with tf32_context(False):
        result = checkpoint(Record.apply, value, use_reentrant=False)
        result.sum().backward()
    assert [event[0] for event in events].count("forward") == 2
    assert all(flags == (False, False) for _, *flags in events for flags in [tuple(flags)])
    torch.testing.assert_close(value.grad, torch.tensor([4.0]))
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
    with pytest.raises(RuntimeError), tf32_context(False):
        raise RuntimeError("training failed")
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


def test_tf32_defaults_are_explicit_and_cli_overrides_are_validated():
    assert resolve_tf32({}) is True
    assert resolve_tf32({"cudnn_deterministic": True}) is False
    assert resolve_tf32({"tf32": False}) is False
    with pytest.raises(ValueError, match="tf32"):
        resolve_tf32({"tf32": "false"})
    from alohamini.learning.train_config import parse_training_args

    settings, _ = parse_training_args(
        [
            "--dataset.root=/tmp/data",
            "--output_dir=/tmp/run",
            "--tf32=false",
        ]
    )
    assert settings["tf32"] is False


@pytest.mark.parametrize("kind", ["eo1", "lingbot_va"])
def test_policy_storage_recipe_applies_before_generic_default_and_allows_override(kind):
    from alohamini.learning.train_config import parse_training_args

    arguments = [
        f"--policy.type={kind}",
        "--dataset.root=/tmp/data",
        "--output_dir=/tmp/run",
        "--image_size=[32,32]",
    ]
    settings, _ = parse_training_args(arguments)
    assert settings["trainable_dtype"] == "float32"
    assert settings["mixed_precision"] == "bfloat16"
    explicit, _ = parse_training_args(arguments + ["--trainable_dtype=preserve"])
    assert explicit["trainable_dtype"] == "preserve"


@pytest.mark.parametrize("kind", ["act", "pi05"])
def test_native_finetuning_preserves_storage_but_resolves_new_compute_policy(tmp_path, kind):
    import json

    from alohamini.learning.train_config import parse_training_args

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    precision = dict(trainable_dtype="float32", mixed_precision="bfloat16", tf32=False)
    manifest = dict(
        format="alohamini-policy",
        version=1,
        kind=kind,
        config={},
        training=precision,
        state="recorded",
        cameras=["forward"],
        image_size=[32, 32],
    )
    (checkpoint / "policy.json").write_text(json.dumps(manifest))
    (checkpoint / "model.safetensors").touch()  # This test resolves metadata, not tensor data.
    arguments = [f"--policy.path={checkpoint}", "--dataset.root=/tmp/data", "--output_dir=/tmp/run"]
    inherited, _ = parse_training_args(arguments)
    assert inherited["trainable_dtype"] == "float32"
    assert inherited.get("mixed_precision", "none") == "none"
    assert inherited["tf32"] is True
    override, _ = parse_training_args(
        arguments
        + [
            "--trainable_dtype=preserve",
            "--mixed_precision=none",
            "--tf32=true",
        ]
    )
    assert {key: override[key] for key in precision} == dict(
        trainable_dtype="preserve",
        mixed_precision="none",
        tf32=True,
    )
    manifest["training"] = {}
    (checkpoint / "policy.json").write_text(json.dumps(manifest))
    legacy, _ = parse_training_args(arguments)
    assert legacy["trainable_dtype"] == "preserve" and legacy["tf32"] is True
    assert legacy.get("mixed_precision", "none") == "none"
