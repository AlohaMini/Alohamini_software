"""EVO1 stage-specific native training with saved InternVL architecture and assets."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.policies.chunk import ChunkAlgorithm, TaskProcessor, local_assets

from .configuration_evo1 import Evo1Config


class Evo1Processor(TaskProcessor):
    def __init__(self, config, stats, device):
        super().__init__(config, stats, device)
        self.action_dim = config.action_feature.shape[0]

    def action(self, tensor, *, context=None):
        return self.numeric.action(tensor[..., : self.action_dim])


class Evo1Algorithm(ChunkAlgorithm):
    asset_fields = ("vlm_model_name",)
    config_class = Evo1Config
    fsdp_classes = ("InternVLVisionLayer", "Qwen2DecoderLayer", "BasicTransformerBlock")

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto", batch_size=8, mixed_precision="bfloat16", ema={"enable": False}
        ).items():
            cfg.setdefault(key, value)
        cfg.setdefault("scheduler", {"type": "diffusers_cosine", "warmup_steps": 300})
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="evo1-recorded-v1",
            repository="https://github.com/MINT-SJTU/Evo-1",
            images=(
                "ordered views, resize/quantize using InternVL3 bicubic pixels; ImageNet mean/std"
            ),
            state="named recorded fields; train-only MIN_MAX then zero pad to 24D",
            action="recorded targets; MIN_MAX then 24D mask; repeated episode tails excluded",
            language="InternVL image placeholders and task, 1024 tokens",
            time="50 future row offsets; 32 flow integration steps; uniform [-1,1] action noise",
            architecture="14 InternVL language layers; 8 action Transformer layers, width 896",
            training="stage1 frozen VLM; stage2 full VLM; explicit branch flags take precedence",
            adaptations=[
                "native platform datasets/trainer/checkpoints and local-only assets",
                "SDPA replaces optional FlashAttention dependency; no internal global AMP override",
                "recorded gripper commands; no implicit LIBERO sign/binary conversion",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return cfg

    @staticmethod
    def finetuning_options(options, saved, overrides):
        if "training_stage" in overrides and overrides["training_stage"] != saved["training_stage"]:
            for key in (
                "finetune_vlm",
                "finetune_language_model",
                "finetune_vision_model",
                "finetune_action_head",
            ):
                if key not in overrides:
                    options[key] = None
        return options

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if not settings.get("resume"):
            path = Path(settings.get("pretrained_path") or ".").expanduser()
            if (path / "config.json").is_file() and not (path / "policy.json").is_file():
                base = json.loads((path / "config.json").read_text())
                allowed = {f.name for f in fields(Evo1Config)} - {
                    "input_features",
                    "output_features",
                    "device",
                }
                result = {**{k: v for k, v in base.items() if k in allowed}, **result}
                if "vlm_model_name" not in settings.get("model", {}):
                    result["vlm_model_name"] = str(path / "assets")
            result["vlm_model_name"] = str(local_assets(result.get("vlm_model_name", "")))
        result["device"] = "cpu"
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 50)))}, include_task=True
        )

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": int((~batch["action_is_pad"]).sum())}

    @staticmethod
    def metric_specs(config):
        return [
            MetricSpec(
                "flow_loss",
                "loss",
                "valid_action_elements",
                loss_count("_loss_weight", config.action_feature.shape[0]),
            )
        ]

    def build(self, options, stats=None):
        from .modeling_evo1 import Evo1Policy

        return Evo1Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        return Evo1Processor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(model, settings["pretrained_path"])

    @staticmethod
    def checkpoint_options(path, options, device, n_action_steps, temporal_ensemble_coeff):
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("EVO1 does not use ACT temporal ensembling")
        result = {**deepcopy(options), "device": "cpu", "vlm_model_name": str(path / "assets")}
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        return result

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        model.model.embedder.model.config.save_pretrained(path / "assets")
        model.model.embedder.tokenizer.save_pretrained(path / "assets")
        manifest["config"]["vlm_model_name"] = "assets"
        save_weights(path / "model.safetensors", model, model_state)
