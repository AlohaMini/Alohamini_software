"""Preserve policy semantics across the supported Transformers 5 interfaces."""

import pytest
import torch
from transformers import PretrainedConfig
from transformers.cache_utils import DynamicCache

from alohamini.policies.transformers_compat import create_causal_mask, crop_cache_to_length


def test_cache_restores_mixed_self_and_cross_attention_layers():
    cache = DynamicCache()
    prefix = torch.randn(1, 2, 3, 4)
    for i in range(2):
        cache.update(prefix.clone(), prefix.clone(), i)
    for _ in range(3):
        # Only the self-attention layer appends suffix tokens during denoising.
        suffix = torch.randn(1, 2, 2, 4)
        cache.update(suffix, suffix, 0)
        crop_cache_to_length(cache, 3)
        for layer in cache.layers:
            torch.testing.assert_close(layer.keys, prefix)
            torch.testing.assert_close(layer.values, prefix)


def test_cached_causal_mask_preserves_padding_and_prefix_positions():
    config = PretrainedConfig()
    config._attn_implementation = "eager"
    cache = DynamicCache()
    prefix = torch.randn(1, 2, 3, 4)
    cache.update(prefix, prefix, 0)
    positions = torch.tensor([3, 4])
    result = create_causal_mask(
        config=config,
        input_embeds=torch.zeros(1, 2, 8),
        attention_mask=torch.tensor([[1, 1, 0, 1, 1]]),
        cache_position=positions,
        past_key_values=cache,
        position_ids=positions[None],
    )
    allowed = torch.tensor([[[[True, True, False, True, False], [True, True, False, True, True]]]])
    torch.testing.assert_close(result == 0, allowed)


def test_groot_config_factory_and_saved_overrides_are_preserved():
    from alohamini.policies.gr00t_n17._vendor.configuration import Gr00tN1d7Config

    first, second = Gr00tN1d7Config(), Gr00tN1d7Config()
    assert first.diffusion_model_cfg["num_layers"] == 16
    first.diffusion_model_cfg["num_layers"] = 2
    assert second.diffusion_model_cfg["num_layers"] == 16
    loaded = Gr00tN1d7Config(**first.to_filtered_dict())
    assert loaded.diffusion_model_cfg == first.diffusion_model_cfg
    assert loaded.action_horizon == first.action_horizon


@pytest.mark.parametrize(
    "options",
    [
        {"freeze_vision_encoder": False, "train_expert_only": True},
        {"attention_implementation": "invalid"},
    ],
)
def test_hyvla_config_validation_survives_base_class_initialization(options):
    from alohamini.policies.hyvla05._vendor.modeling_dual_tower import HyDualTowerConfig

    with pytest.raises(ValueError):
        HyDualTowerConfig(**options)
