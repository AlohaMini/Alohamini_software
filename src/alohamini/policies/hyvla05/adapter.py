"""Native HyVLA training, checkpoint and action interfaces."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import torch

from alohamini.paths import WorkspacePaths
from alohamini.policies.chunk import ChunkAlgorithm, ChunkPolicy, local_assets, padded_mean

from .configuration_hyvla05 import HyVLA05Config
from .preset import apply_preset


class HyVLA05Policy(ChunkPolicy):
    name = "hyvla05"

    def __init__(self, config):
        super().__init__()
        from ._vendor.configuration_hy_vla import HyVLAConfig
        from ._vendor.modeling_hy_vla import HyVLA

        self.config = config
        options = deepcopy(config.network)
        options.update(
            pretrained_path=str(local_assets(config.assets_path)),
            vlm_model_path="",
            image_features=config.image_features,
            chunk_size=config.chunk_size,
            n_action_steps=config.chunk_size,
        )
        inner = HyVLAConfig(**options)
        inner.action_feature = config.action_feature
        self.network = HyVLA(inner)
        # This adapter trains flow only. Unused LM heads must not enter DDP/EMA.
        tower = self.network.model.dual_tower
        tower.expert.lm_head = None
        embedding = tower.vlm.get_input_embeddings().weight
        head = tower.vlm.get_output_embeddings()
        if head is not None and head.weight is not embedding:
            head.requires_grad_(False)
        self.reset()

    def forward(self, batch):
        net = self.network
        images, masks = net.prepare_images(batch)
        tokens, token_mask, _ = net.prepare_language(batch)
        loss, _ = net.model(
            images,
            masks,
            tokens,
            token_mask,
            net.prepare_state(batch),
            net.prepare_action(batch),
        )
        loss = padded_mean(loss[..., : self.config.action_feature.shape[0]], batch["action_is_pad"])
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": loss.detach()}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.network.forward_evaluate(batch)["pred"]

    def save_assets(self, path):
        self.network.language_tokenizer.save_pretrained(path)


class HyVLA05Algorithm(ChunkAlgorithm):
    loss_terms = ("flow_loss",)
    config_class = HyVLA05Config
    apply_preset = staticmethod(apply_preset)
    # Decoder blocks are called functionally; only the ViT blocks have forward boundaries.
    fsdp_classes = ("_ViTBlock", "SpaceTimeBlock", "HunYuanVLMoTRMSNorm")

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return options
        from ._vendor.configuration_hy_vla import HyVLAConfig

        path = local_assets(
            options.get("assets_path")
            or settings.get("pretrained_path")
            or WorkspacePaths().pretrained / "hyvla05"
        )
        options["assets_path"] = str(path)
        if not options.get("network") or "vlm_config_dict" not in options["network"]:
            source = json.loads((path / "config.json").read_text())
            allowed = {f.name for f in fields(HyVLAConfig)}
            ignored = {
                "image_features",
                "pretrained_path",
                "vlm_model_path",
                "chunk_size",
                "n_action_steps",
            }
            options["network"] = {
                **{k: v for k, v in source.items() if k in allowed - ignored},
                **options.get("network", {}),
            }
        return options

    def sample_spec(self, options, *, cameras=()):
        if not cameras:
            raise ValueError("HyVLA requires cameras")
        count, stride = options.get("history_frames", 6), options.get("history_stride", 15)
        return dict(
            delta_indices={
                "action": list(range(options.get("chunk_size", 20))),
                **{
                    f"observation.images.{camera}": list(range(-(count - 1) * stride, 1, stride))
                    for camera in cameras
                },
            },
            include_task=True,
        )

    def build(self, options, stats=None):
        return HyVLA05Policy(self.config_class(**deepcopy(options)))

    def live_history_spec(self, options, *, cameras=()):
        return {
            key: value
            for key, value in self.sample_spec(options, cameras=cameras)["delta_indices"].items()
            if key != "action"
        }

    def initialize(self, model, settings):
        from safetensors.torch import load_model

        path = Path(settings.get("pretrained_path") or model.config.assets_path).expanduser()
        load_model(model.network, path / "model.safetensors", strict=True)
