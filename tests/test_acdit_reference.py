"""Optional parity against the pinned official source, without network downloads."""

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from alohamini.policies.acdit.transformer import CoordinationDiT


@pytest.mark.parametrize("body", [False, True])
def test_official_dit_forward_and_backward_parity(monkeypatch, body):
    source = os.environ.get("ALOHAMINI_ACDIT_SOURCE")
    if not source:
        pytest.skip("Set ALOHAMINI_ACDIT_SOURCE to pinned AC-DiT source for reference parity")
    root = Path(source)
    helper = ModuleType("helpers.common")
    helper.ACDIT_DEFAULT_LOGGER = SimpleNamespace(info=lambda *args: None)
    for name, module in {
        "helpers": ModuleType("helpers"),
        "helpers.common": helper,
        "models": ModuleType("models"),
        "models.rdt": ModuleType("models.rdt"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    load("models.rdt.blocks", root / "models/rdt/blocks.py")
    reference = load("acdit_reference_model", root / "models/rdt/model.py")
    torch.manual_seed(7)
    cls = reference.DiTCC if body else reference.DiT
    official = cls(
        output_dim=32,
        horizon=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        dtype=torch.float32,
        **({"num_latent_feature_tokens": 2} if body else {}),
    ).eval()
    with torch.no_grad():
        official.final_layer.ffn_final.fc2.weight.normal_(std=0.02)
    native = CoordinationDiT(32, 2, 32, 2, 4, 2 if body else 0).eval()
    native.load_state_dict(official.state_dict(), strict=True)
    x, frequency, timestep, context = (
        torch.randn(2, 3, 32),
        torch.tensor([30.0, 30.0]),
        torch.tensor([2, 7]),
        torch.randn(2, 32),
    )
    conditions = [torch.randn(2, length, 32) for length in (3, 4, 2)]
    masks = [torch.ones(c.shape[:2], dtype=torch.bool) for c in conditions]
    latent = torch.randn(2, 4, 32) if body else None
    kwargs = dict(
        x=x,
        freq=frequency,
        t=timestep,
        in_context_conditions_c=context,
        lang_c=conditions[0],
        img_c=conditions[1],
        pc_c=conditions[2],
        lang_mask=masks[0],
        img_mask=masks[1],
        pc_mask=masks[2],
    )
    if body:
        kwargs["lm_c"] = latent
    expected = official(**kwargs)
    actual = native(x, frequency, timestep, context[:, None], conditions, masks, latent)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(official.named_parameters())[name].grad, rtol=0, atol=0
        )
