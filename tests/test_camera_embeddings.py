"""Camera identity, legacy loading and explicit ACT-family fine-tuning upgrades."""

import json
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_native_learning import model_options

from alohamini.learning.checkpoint import initialize_policy, resolve_pretrained
from alohamini.learning.policy import make_policy
from alohamini.policies.registry import algorithm


def options(enabled=True, *, conditional=False):
    result = model_options(state=False, use_camera_embeddings=enabled)
    result["input_features"] = {
        f"observation.images.{name}": {"type": "VISUAL", "shape": (3, 64, 64)}
        for name in ("forward", "wrist_left", "wrist_right")
    }
    if conditional:
        result.update(posterior_condition="image_action", prior_type="conditional_gaussian")
    return result


def images():
    return {
        f"observation.images.{name}": torch.rand(1, 3, 64, 64)
        for name in ("forward", "wrist_left", "wrist_right")
    }


def predict(model, data):
    kwargs = (
        {"latent_sample": torch.zeros(1, model.config.latent_dim)} if model.name == "am_act" else {}
    )
    return model.model(
        {"observation.images": [data[key] for key in model.config.image_features]}, **kwargs
    )[0]


def checkpoint(path, model, *, legacy=False):
    path.mkdir()
    config = asdict(model.config)
    if legacy:
        config.pop("use_camera_embeddings")
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": model.name,
        "config": config,
        "state": "none",
        "image_size": [64, 64],
        "cameras": ["forward", "wrist_left", "wrist_right"],
    }
    (path / "policy.json").write_text(json.dumps(manifest))
    algorithm(model.name).save(path, model, manifest)


@pytest.mark.parametrize("kind,conditional", [("act", False), ("am_act", False), ("am_act", True)])
@pytest.mark.parametrize("enabled", [False, True])
def test_wrist_identity_changes_actions_with_fixed_latent(kind, conditional, enabled):
    torch.manual_seed(72)
    model = make_policy(kind, options(enabled, conditional=conditional)).eval()
    data = images()
    swapped = {
        **data,
        "observation.images.wrist_left": data["observation.images.wrist_right"],
        "observation.images.wrist_right": data["observation.images.wrist_left"],
    }
    calls = []
    handle = model.model.backbone.register_forward_hook(lambda *args: calls.append(1))
    original = predict(model, data)
    assert len(calls) == 3  # No repeated feature extraction for camera identity.
    handle.remove()
    exchanged = predict(model, swapped)
    if enabled:
        assert (original - exchanged).abs().max() > 1e-6
        original.square().mean().backward()
        grad = model.model.encoder_camera_embed.weight.grad
        assert torch.isfinite(grad).all()
        assert (grad.abs().sum(-1) > 0).all()
    else:
        torch.testing.assert_close(original, exchanged, rtol=0, atol=1e-6)


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_inference_binds_camera_names_not_input_dictionary_order(kind):
    torch.manual_seed(72)
    model = make_policy(kind, options()).eval()
    data = images()
    with torch.inference_mode():
        expected = model.predict_action_chunk(data)
        actual = model.predict_action_chunk(dict(reversed(list(data.items()))))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["act", "am_act"])
@pytest.mark.parametrize("legacy", [False, True])
def test_legacy_upgrade_is_explicit_preserves_outputs_and_learns(kind, legacy, tmp_path):
    torch.manual_seed(72)
    source = make_policy(kind, options(False)).eval()
    path = tmp_path / "old"
    checkpoint(path, source, legacy=legacy)
    component = algorithm(kind)
    cfg = resolve_pretrained({"pretrained_path": str(path)})
    assert cfg["model"]["use_camera_embeddings"] is False
    saved_options = json.loads((path / "policy.json").read_text())["config"]
    inference_options = component.checkpoint_options(path, saved_options, "cpu", None, "checkpoint")
    assert inference_options["use_camera_embeddings"] is False
    restored = component.build(inference_options).eval()
    component.load(path, restored)
    data = images()
    expected = predict(source, data).detach()
    torch.testing.assert_close(predict(restored, data), expected, rtol=0, atol=0)

    cfg = resolve_pretrained(
        {"pretrained_path": str(path), "model": {"use_camera_embeddings": True}}
    )
    upgraded = component.build(cfg["model"]).eval()
    initialize_policy(component, upgraded, cfg)
    embedding = upgraded.model.encoder_camera_embed.weight
    assert torch.count_nonzero(embedding) == 0
    prediction = predict(upgraded, data)
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
    prediction.square().mean().backward()
    assert torch.isfinite(embedding.grad).all()
    assert not torch.equal(embedding.grad[1], embedding.grad[2])
    torch.optim.SGD([embedding], lr=0.1).step()
    assert not torch.equal(embedding[1], embedding[2])

    new_path = tmp_path / "upgraded"
    checkpoint(new_path, upgraded)
    loaded = component.build(resolve_pretrained({"pretrained_path": str(new_path)})["model"]).eval()
    component.load(new_path, loaded)
    torch.testing.assert_close(predict(loaded, data), predict(upgraded, data), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["act", "am_act"])
def test_camera_load_stays_strict_and_rejects_reordering(kind, tmp_path):
    component = algorithm(kind)
    source = component.build(options())
    path = tmp_path / "new"
    checkpoint(path, source)
    reordered = options()
    reordered["input_features"] = dict(reversed(list(reordered["input_features"].items())))
    with pytest.raises(ValueError, match="camera names and order"):
        component.load(path, component.build(reordered))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        component.load(path, component.build(options(False)))
    state = load_file(path / "model.safetensors")
    del state["model.encoder_camera_embed.weight"]
    save_file(state, path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        component.load(path, component.build(options()))

    old_path = tmp_path / "old"
    checkpoint(old_path, component.build(options(False)), legacy=True)
    state = load_file(old_path / "model.safetensors")
    del state["model.action_head.weight"]
    save_file(state, old_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        component.load(old_path, component.build(options()))
