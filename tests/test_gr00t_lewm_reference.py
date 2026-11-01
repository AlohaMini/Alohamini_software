"""Numerical checks against pinned official components; never download in tests."""

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import torch
from test_gr00t_lewm import groot_assets as groot_assets
from test_gr00t_lewm import options


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gr00t_official_head_loss_gradients_and_sampling(groot_assets):
    root = os.environ.get("ALOHAMINI_GR00T_SOURCE")
    if not root:
        pytest.skip("Set ALOHAMINI_GR00T_SOURCE to pinned official source")
    from transformers.feature_extraction_utils import BatchFeature

    from alohamini.policies.gr00t_n17._vendor.action_head import Gr00tN1d7ActionHead
    from alohamini.policies.gr00t_n17._vendor.configuration import Gr00tN1d7Config

    root = Path(root)
    dit = load(root / "gr00t/model/modules/dit.py", "gr00t_reference_dit")
    projection = load(
        root / "gr00t/model/modules/embodiment_conditioned_mlp.py", "gr00t_reference_mlp"
    )
    source = (root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py").read_text()
    source = source[: source.index("\ndef get_backbone_cls")]
    source = source.replace("from transformers import AutoConfig, AutoModel, PreTrainedModel", "")
    source = source.replace("import tree", "")
    source = source.replace("from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config", "")
    source = source.replace(
        "from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer", ""
    )
    source = source.replace(
        (
            "from gr00t.model.modules.embodiment_conditioned_mlp import ("
            "\n    CategorySpecificMLP,\n    MultiEmbodimentActionEncoder,\n"
            ")"
        ),
        "",
    )
    module = ModuleType("gr00t_reference_head")
    module.__dict__.update(
        Gr00tN1d7Config=Gr00tN1d7Config,
        AlternateVLDiT=dit.AlternateVLDiT,
        DiT=dit.DiT,
        SelfAttentionTransformer=dit.SelfAttentionTransformer,
        CategorySpecificMLP=projection.CategorySpecificMLP,
        MultiEmbodimentActionEncoder=projection.MultiEmbodimentActionEncoder,
    )
    exec(
        compile(source, str(root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py"), "exec"), module.__dict__
    )
    cfg = Gr00tN1d7Config(**options("gr00t_n17", groot_assets)["network"])
    official = module.Gr00tN1d7ActionHead(cfg)
    native = Gr00tN1d7ActionHead(cfg)
    native.load_state_dict(official.state_dict(), strict=True)
    features = torch.randn(2, 5, 32)
    state, action = torch.randn(2, 1, 32), torch.randn(2, 3, 32)
    mask = torch.zeros_like(action)
    mask[:, :2, :18] = 1

    def inputs():
        return (
            BatchFeature(
                dict(
                    backbone_features=features.clone(),
                    backbone_attention_mask=torch.ones(2, 5, dtype=torch.bool),
                    image_mask=torch.tensor([[False, True, True, False, False]]).expand(2, -1),
                )
            ),
            BatchFeature(
                dict(
                    state=state.clone(),
                    action=action.clone(),
                    action_mask=mask.clone(),
                    embodiment_id=torch.ones(2, dtype=torch.long),
                )
            ),
        )

    torch.manual_seed(9)
    expected = official(*inputs())["loss"]
    torch.manual_seed(9)
    actual = native(*inputs())["loss"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for name, parameter in native.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(official.named_parameters())[name].grad, rtol=0, atol=0
        )
    official.eval()
    native.eval()

    def observation():
        backbone, action_input = inputs()
        action_input.pop("action")
        return backbone, action_input

    torch.manual_seed(7)
    expected = official.get_action(*observation()).action_pred
    torch.manual_seed(7)
    actual = native.get_action(*observation()).action_pred
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_lewm_predictor_and_sigreg_official_parity():
    root = os.environ.get("ALOHAMINI_LEWM_SOURCE")
    if not root:
        pytest.skip("Set ALOHAMINI_LEWM_SOURCE to pinned official source")
    from alohamini.policies.lewm._vendor.module import ARPredictor
    from alohamini.policies.lewm.modeling_lewm import GlobalSIGReg

    reference = load(Path(root) / "module.py", "lewm_reference")
    kwargs = dict(
        num_frames=3,
        depth=2,
        heads=3,
        dim_head=8,
        mlp_dim=48,
        input_dim=24,
        hidden_dim=24,
        output_dim=24,
        dropout=0.0,
    )
    native, official = ARPredictor(**kwargs), reference.ARPredictor(**kwargs)
    native.load_state_dict(official.state_dict())
    x, c = torch.randn(2, 3, 24), torch.randn(2, 3, 24)
    torch.testing.assert_close(native(x, c), official(x, c), rtol=0, atol=0)
    z = torch.randn(3, 2, 24, requires_grad=True)
    torch.manual_seed(4)
    expected = reference.SIGReg(knots=5, num_proj=8)(z)
    torch.manual_seed(4)
    actual = GlobalSIGReg(knots=5, num_proj=8)(z)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, z)[0], torch.autograd.grad(expected, z)[0], rtol=0, atol=0
    )


def test_released_lewm_weights_load_strictly():
    path = os.environ.get("ALOHAMINI_LEWM_WEIGHTS")
    if not path:
        pytest.skip("Set ALOHAMINI_LEWM_WEIGHTS to the official PushT weights.pt")
    from alohamini.learning.policy import make_policy

    model = make_policy(
        "lewm",
        dict(
            input_features={
                "observation.images.forward": {"type": "VISUAL", "shape": [3, 224, 224]}
            },
            output_features={"action": {"type": "ACTION", "shape": [2]}},
        ),
    )
    model.world.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
