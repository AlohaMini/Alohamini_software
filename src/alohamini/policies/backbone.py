"""Torchvision ResNet initialization with a visible, shared weights directory."""

import torch
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

from alohamini.paths import WorkspacePaths


def configure_decoder_depth(config, depth):
    """Keep layer count and HF per-layer metadata consistent before construction."""
    original = config.num_hidden_layers
    if depth is None:
        return original
    if type(depth) is not int or not 1 <= depth <= original:
        raise ValueError("Requested language depth exceeds the local backbone configuration")
    for key in ("layer_types", "mlp_layer_types"):
        values = getattr(config, key, None)
        if values is not None:
            if len(values) != original:
                raise ValueError(f"Backbone {key} disagrees with num_hidden_layers")
            setattr(config, key, list(values[:depth]))
    config.num_hidden_layers = depth
    return depth


def make_resnet(config, *, norm_layer=FrozenBatchNorm2d):
    """Keep torchvision initialization semantics; change only the cache directory."""
    weights = torchvision.models.get_model_weights(config.vision_backbone).verify(
        config.pretrained_backbone_weights
    )
    backbone = getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[
            False, False, getattr(config, "replace_final_stride_with_dilation", False)
        ],
        weights=None,
        norm_layer=norm_layer,
    )
    if weights is not None:
        state = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(WorkspacePaths().pretrained),
            map_location="cpu",
            check_hash=True,
            weights_only=True,
        )
        backbone.load_state_dict(state)
    return backbone
