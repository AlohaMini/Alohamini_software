"""X-VLA recorded-coordinate training through platform-owned execution and scheduling."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_xvla import XVLAConfig


class XVLAAlgorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = XVLAConfig
    fsdp_classes = (
        "TransformerBlock",
        "BartEncoderLayer",
        "Florence2VisionBlock",
        "DomainAwareLinear",
    )
    loss_terms = ("joints_loss",)

    @staticmethod
    def apply_preset(settings):
        cfg = deepcopy(settings)
        for name, value in dict(
            state="auto",
            batch_size=16,
            mixed_precision="none",
            ema={"enable": False},
            return_uint8=True,
        ).items():
            cfg.setdefault(name, value)
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        schedule = dict(
            type="constant_with_warmup",
            start_step=1000,
            warmup_steps=0,
            before_start=dict(vlm=0.0, transformer_core=0.0),
        )
        if "scheduler" not in cfg:
            cfg["scheduler"] = schedule
        elif isinstance(cfg["scheduler"], dict) and cfg["scheduler"].get(
            "type", "constant_with_warmup"
        ) in ("constant_with_warmup", "staged_cosine"):
            cfg["scheduler"] = {**schedule, **cfg["scheduler"]}
        cfg["paper_preset"] = dict(
            name="xvla-recorded-v1",
            repository="https://github.com/2toINF/X-VLA",
            revision="6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2",
            images="ordered RGB; local Florence image processor; saved pixel configuration",
            language="BART, right-padded 64-token instruction, author all-token attention",
            state="named recorded fields with train-only mean/std; pad to 20D",
            action="named recorded commands, mean/std; AUTO MSE, no guessed gripper/rotation slots",
            architecture=(
                "24 action Transformer blocks, width1024,16heads,32 soft prompts; "
                "Florence layers saved"
            ),
            training=(
                "first1000 successful updates: only prompts/action projections have nonzero LR; "
                "then all groups"
            ),
            paper_protocol=(
                "EE6D20: metric XYZ, rotation6D, normalized grippers; "
                "not inferred from recorded width"
            ),
            adaptations=[
                "recorded robot coordinate MSE/normalization instead of canonical EE6D task loss",
                "platform data/trainer/checkpoint and grouped scheduler; no internal optimizer",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return cfg

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if not settings.get("resume"):
            path = Path(settings.get("pretrained_path") or ".").expanduser()
            if (path / "config.json").is_file() and not (path / "policy.json").is_file():
                base = json.loads((path / "config.json").read_text())
                allowed = {item.name for item in fields(XVLAConfig)} - {
                    "input_features",
                    "output_features",
                    "action_mode",
                }
                result = {**{key: value for key, value in base.items() if key in allowed}, **result}
                result.setdefault("assets_path", str(path / "assets"))
            result["assets_path"] = str(local_assets(result.get("assets_path", "")))
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 30)))}, include_task=True
        )

    def build(self, options, stats=None):
        from .modeling_xvla import XVLAPolicy

        return XVLAPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import XVLAProcessor

        return XVLAProcessor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(model, settings["pretrained_path"])
