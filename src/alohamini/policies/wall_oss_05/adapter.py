"""WALL flow adaptation through the shared reader, trainer and checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.processor import quantile_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_wall_oss_05 import WallOSS05Config


class WallOSS05Algorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = WallOSS05Config
    supports_field_transforms = True
    fsdp_classes = (
        "WallOSS05DecoderLayer",
        "Qwen2_5_VLVisionBlock",
        "WallOSS05RMSNorm",
        "Embedding",
    )

    @staticmethod
    def apply_preset(settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto",
            batch_size=8,
            steps=200000,
            save_freq=5000,
            mixed_precision="none",
            return_uint8=True,
            ema={"enable": False},
        ).items():
            cfg.setdefault(key, value)
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="wall-oss-05-recorded-flow-v1",
            repository="https://github.com/X-Square-Robot/wall-x",
            revision="ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f",
            architecture="Qwen2.5-VL36 layers,36 action-expert layers; Qwen vision32 blocks",
            images="ordered RGB; long edge448 aspect resize; local Qwen vision processor",
            language="original flow chat prompt; normalized state digitized into512 bins",
            numeric="named robot fields normalized before explicit26D slot placement",
            normalization="recorded:q01/q99+1e-6,no clipping; wall:original clipped quantiles",
            time="32 future row offsets; repeated episode tails excluded from target loss",
            training="author LIBERO AdamW recipe5e-5;1000 warmup;200k cosine;clip1",
            precision="BF16 model;FP32 robot heads and RMSNorms;platform AMP/TF32",
            adaptations=[
                "native reader/trainer/checkpoint; train-only exact statistics",
                "new embodiment resets robot input/output heads by default",
                "explicit state/action mappings saved; no anatomical inference",
                "flow robot fine-tuning only; language supervision and DMuon not included",
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
            path = Path(
                settings.get("pretrained_path") or result.get("assets_path", ".")
            ).expanduser()
            if (path / "config.json").is_file() and not (path / "policy.json").is_file():
                raw = json.loads((path / "config.json").read_text())
                allowed = {item.name for item in fields(WallOSS05Config)} - {
                    "input_features",
                    "output_features",
                    "camera_key_mapping",
                }
                result = {**{k: v for k, v in raw.items() if k in allowed}, **result}
                result.setdefault("core_config", raw.get("core_config", raw.get("vlm_config", {})))
                result.setdefault("assets_path", str(path))
            result["assets_path"] = str(local_assets(result.get("assets_path", "")))
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 32)))}, include_task=True
        )

    @staticmethod
    def prepare_options(options, samples):
        model_features = getattr(samples, "model_features", {})
        state = model_features.get(
            "observation.state", samples.selection.feature if samples.selection else {}
        )
        action = model_features.get("action", samples.info["features"]["action"])
        for kind, feature in (("state", state), ("action", action)):
            names = feature.get("names", [])
            if not names or len(set(names)) != len(names):
                raise ValueError("WALL requires distinct recorded/transformed coordinate names")
            if options.get(f"{kind}_names", names) != names:
                raise ValueError(f"WALL {kind} names differ from the dataset representation")
            options[f"{kind}_names"] = list(names)
            if not options.get(f"{kind}_slots"):
                if not options.get("reinitialize_robot_heads", True):
                    raise ValueError(
                        "Retaining WALL robot heads requires explicit named state/action slots"
                    )
                options[f"{kind}_slots"] = {name: i for i, name in enumerate(names)}

    @staticmethod
    def statistics(samples, options=None):
        return quantile_statistics(samples)

    @staticmethod
    def loss_counts(batch):
        mask = batch.get("action_is_pad")
        width = batch["action"].shape[-1]
        count = int((~mask).sum()) if mask is not None else batch["action"].numel() // width
        return {"_flow_weight": count * width}

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        return [
            MetricSpec(
                "flow_loss",
                "flow_loss",
                "coordinates",
                loss_count("_flow_weight"),
                console="flow_loss",
            )
        ]

    def build(self, options, stats=None):
        from .modeling_wall_oss_05 import WallOSS05Policy

        return WallOSS05Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import WallOSS05Processor

        return WallOSS05Processor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(model, settings["pretrained_path"])
        if model.config.reinitialize_robot_heads:
            model.model.action_preprocessor.w1.reset_parameters()
            model.model.action_preprocessor.action_proj_back.reset_parameters()

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from transformers import Qwen2_5_VLProcessor

        from alohamini.learning.checkpoint import save_weights

        Qwen2_5_VLProcessor.from_pretrained(
            model.config.assets_path, local_files_only=True
        ).save_pretrained(path / "assets")
        manifest["config"]["assets_path"] = "assets"
        save_weights(path / "model.safetensors", model, model_state)
