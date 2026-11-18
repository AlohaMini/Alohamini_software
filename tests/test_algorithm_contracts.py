"""A newly registered adapter declares assets without changing checkpoint code."""

import sys
from types import ModuleType


def test_registered_asset_contract_relocates_and_accepts_a_new_optional_resource(
    tmp_path, monkeypatch
):
    from alohamini.learning.checkpoint import resolve_pretrained, write_checkpoint_metadata
    from alohamini.policies.registry import ALGORITHMS

    module = ModuleType("native_test_asset_adapter")
    module.Adapter = type("Adapter", (), {"asset_fields": ("encoder", "auxiliary_encoder")})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(ALGORITHMS, "test_asset_adapter", (module.__name__, "Adapter"))
    checkpoint = tmp_path / "checkpoint"
    encoder = checkpoint / "assets/encoder"
    encoder.mkdir(parents=True)
    (encoder / "vocab.json").write_text("{}")
    (checkpoint / "model.safetensors").touch()  # Metadata resolution does not read tensors.
    manifest = dict(
        format="alohamini-policy",
        version=1,
        kind="test_asset_adapter",
        config=dict(encoder="assets/encoder"),
        stats={},
        state="none",
        state_feature=None,
        state_units=[],
        cameras=[],
        image_size=[32, 32],
        source_info=dict(features=dict(action=dict(shape=[1], names=["joint"], units=["rad"]))),
    )
    write_checkpoint_metadata(checkpoint, manifest)
    assert manifest["assets"]["encoder"]["storage"] == "bundled"
    moved = checkpoint.rename(tmp_path / "relocated")
    resolved = resolve_pretrained(dict(pretrained_path=str(moved)))
    assert resolved["model"]["encoder"] == str(moved / "assets/encoder")
    auxiliary = tmp_path / "new-encoder"
    auxiliary.mkdir()
    monkeypatch.chdir(tmp_path)
    resolved = resolve_pretrained(
        dict(pretrained_path=str(moved), model=dict(auxiliary_encoder="new-encoder"))
    )
    assert resolved["model"]["auxiliary_encoder"] == str(auxiliary)
