"""EO-1 robot fine-tuning with platform dataset, optimizer and checkpoint management."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_eo1 import EO1Config


class EO1Algorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = EO1Config
    fsdp_classes = ("Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock")
    loss_terms = ("flow_loss",)

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto",
            mixed_precision="bfloat16",
            batch_size=2,
            trainable_dtype="float32",
            ema={"enable": False},
        ).items():
            cfg.setdefault(key, value)
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="eo1-robot-flow-v1",
            repository="https://github.com/EO-Robotics/EO1",
            protocol="robot-only flow fine-tuning; no invented text/CoT supervision",
            images="Qwen2.5-VL native processor, ordered RGB camera tokens; pixel budget saved",
            state="named recorded coordinates, MEAN_STD, padded to 32D; one state token",
            action="named recorded commands, MEAN_STD, 16 future action placeholders",
            language="official physical-assistant system prompt; task with <|vla|> marker",
            time="row offsets at recorded FPS; padded tail targets included",
            architecture="backbone layers from saved vlm_config; action projector 2 layers",
            adaptations=[
                "platform FP32 trainable storage with BF16 compute by default",
                "language-only/CoT multi-dataset training is outside this robot preset",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return cfg

    def options(self, settings, device):
        result = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return result
        path = Path(settings.get("pretrained_path") or result.get("assets_path", ".")).expanduser()
        if (path / "config.json").is_file() and not (path / "policy.json").is_file():
            base = json.loads((path / "config.json").read_text())
            names = {f.name for f in fields(EO1Config)} - {"input_features", "output_features"}
            result = {**{k: v for k, v in base.items() if k in names}, **result}
            result.setdefault("assets_path", str(path / "assets"))
        result["assets_path"] = str(local_assets(result.get("assets_path", "")))
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 16)))}, include_task=True
        )

    def build(self, options, stats=None):
        from .modeling_eo1 import EO1Policy

        return EO1Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import EO1Processor

        return EO1Processor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(model, settings["pretrained_path"])

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from transformers import Qwen2_5_VLProcessor

        from alohamini.learning.checkpoint import save_weights

        Qwen2_5_VLProcessor.from_pretrained(
            model.config.assets_path, local_files_only=True
        ).save_pretrained(path / "assets")
        manifest["config"]["assets_path"] = "assets"
        save_weights(path / "model.safetensors", model, model_state)
