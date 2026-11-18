"""Local large-model loading must preserve aliases and fail before partial mutation."""

import json

import pytest
import torch
from safetensors.torch import save_file, save_model

from alohamini.learning.weights import load_weights


class TiedModel(torch.nn.Module):
    def __init__(self, width=3):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, width)
        self.output = torch.nn.Linear(width, 4, bias=False)
        self.output.weight = self.embedding.weight
        self.head = torch.nn.Linear(width, 2)


@pytest.mark.parametrize("sharded", [False, True])
def test_local_loading_restores_tied_parameters_without_changing_storage_dtype(tmp_path, sharded):
    reference = TiedModel().bfloat16()
    model = TiedModel()
    if sharded:
        state = dict(reference.state_dict())
        del state["output.weight"]
        keys = list(state)
        mapping = {}
        for i, key in enumerate(keys):
            filename = f"model-{i}.safetensors"
            save_file({key: state[key]}, tmp_path / filename)
            mapping[key] = filename
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    else:
        save_model(reference, tmp_path / "model.safetensors")
    assert load_weights(model, tmp_path) == []
    assert model.embedding.weight is model.output.weight
    for key, value in model.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference.state_dict()[key].float(), rtol=0, atol=0)


def test_wrong_later_shard_does_not_partially_update_model(tmp_path):
    reference, model = TiedModel(), TiedModel()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    save_file({"embedding.weight": reference.embedding.weight}, tmp_path / "first.safetensors")
    save_file({"head.weight": torch.ones(2, 7)}, tmp_path / "last.safetensors")
    mapping = {"embedding.weight": "first.safetensors", "head.weight": "last.safetensors"}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="incompatible"):
        load_weights(model, tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_only_explicit_embodiment_head_may_be_reinitialized(tmp_path):
    reference, model = TiedModel(), TiedModel()
    state = reference.state_dict()
    del state["head.weight"], state["head.bias"]
    state.pop("output.weight")
    save_file(state, tmp_path / "model.safetensors")
    head = model.head.weight.detach().clone()
    assert load_weights(model, tmp_path, reinitialize=("head.",)) == ["head.bias", "head.weight"]
    torch.testing.assert_close(model.head.weight, head, rtol=0, atol=0)
    torch.testing.assert_close(model.embedding.weight, reference.embedding.weight, rtol=0, atol=0)


@pytest.mark.parametrize("filename", ["../escape.safetensors", "/tmp/model.safetensors", "x.pt"])
def test_index_cannot_read_arbitrary_files(tmp_path, filename):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"embedding.weight": filename}})
    )
    with pytest.raises(ValueError, match="shard name"):
        load_weights(TiedModel(), tmp_path)


def test_explicit_checkpoint_prefix_preserves_tied_weights_and_rejects_wrong_namespace(tmp_path):
    wrapped = torch.nn.Module()
    wrapped.network = TiedModel()
    save_model(wrapped, tmp_path / "model.safetensors")
    model = TiedModel()
    assert load_weights(model, tmp_path, prefix="network.") == []
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, wrapped.network.state_dict()[name], rtol=0, atol=0)
    before = model.head.weight.detach().clone()
    with pytest.raises(ValueError, match="explicit prefix"):
        load_weights(model, tmp_path, prefix="model.")
    torch.testing.assert_close(model.head.weight, before, rtol=0, atol=0)


def test_explicit_key_mapping_restores_namespace_and_alias_metadata(tmp_path):
    old = torch.nn.Module()
    old.legacy = TiedModel()
    save_model(old, tmp_path / "model.safetensors")
    model = TiedModel()
    mapping = {"legacy." + key: key for key in model.state_dict()}
    assert load_weights(model, tmp_path, key_mapping=mapping) == []
    assert model.embedding.weight is model.output.weight
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, old.legacy.state_dict()[name], rtol=0, atol=0)


def test_colliding_checkpoint_key_mapping_fails_before_mutation(tmp_path):
    model = torch.nn.Linear(2, 2, bias=False)
    before = model.weight.detach().clone()
    save_file(
        {"old": torch.ones(2, 2), "weight": torch.zeros(2, 2)}, tmp_path / "model.safetensors"
    )
    with pytest.raises(ValueError, match="colliding"):
        load_weights(model, tmp_path, key_mapping={"old": "weight"})
    torch.testing.assert_close(model.weight, before, rtol=0, atol=0)


def test_explicit_removed_source_modules_preserve_strict_preflight(tmp_path):
    source = torch.nn.Module()
    source.trunk = TiedModel()
    source.teacher = torch.nn.Linear(3, 3)
    save_model(source, tmp_path / "model.safetensors")
    target = torch.nn.Module()
    target.trunk = TiedModel()
    with pytest.raises(ValueError, match="Unexpected"):
        load_weights(target, tmp_path)
    load_weights(target, tmp_path, exclude=("teacher.",))
    torch.testing.assert_close(target.trunk.head.weight, source.trunk.head.weight)
    with pytest.raises(ValueError, match="absent"):
        load_weights(target, tmp_path, exclude=("trunk.",))
    wrong = torch.nn.Module()
    wrong.trunk = TiedModel(width=5)
    before = wrong.trunk.head.weight.detach().clone()
    with pytest.raises(ValueError, match="incompatible"):
        load_weights(wrong, tmp_path, exclude=("teacher.",))
    torch.testing.assert_close(wrong.trunk.head.weight, before, rtol=0, atol=0)


def test_diffusers_weight_name_uses_same_preflight_and_shard_contract(tmp_path):
    model = TiedModel()
    reference = TiedModel()
    save_model(reference, tmp_path / "diffusion_pytorch_model.safetensors")
    load_weights(model, tmp_path, filename="diffusion_pytorch_model.safetensors")
    torch.testing.assert_close(model.head.weight, reference.head.weight)
    with pytest.raises(ValueError, match="basename"):
        load_weights(model, tmp_path, filename="../diffusion_pytorch_model.safetensors")
