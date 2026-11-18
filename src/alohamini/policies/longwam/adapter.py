"""Long-WAM through native datasets, global loss reduction and portable checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import limits_statistics
from alohamini.policies.chunk import ChunkAlgorithm

from .configuration_longwam import LongWAMConfig


class LongWAMAlgorithm(ChunkAlgorithm):
    asset_fields = ("tokenizer_path",)
    config_class = LongWAMConfig
    # Every control tick must reach the model's history, even while executing queued actions.
    queues_online_observations = True
    supports_remote_chunks = False
    # Joint video/action layers call projections directly: block parents aren't invoked.
    fsdp_classes = ("WanRMSNorm",)

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto", mixed_precision="bfloat16", batch_size=2, seed=42, ema={"enable": False}
        ).items():
            cfg.setdefault(key, value)
        # Pixel geometry is owned by the model processor. Preserve source dimensions.
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="longwam-recorded-v1",
            repository="https://github.com/NVlabs/LongLive",
            revision="31c70d9071cd7851d82939806dc9fac645fcabe5",
            reference="Long-WAM/configs/data/libero.yaml",
            images="camera order explicit; history stride 4; training Resize, online PIL crop",
            state="selected named recorded coordinates; training limits fit before normalization",
            action="absolute recorded coordinates; no implicit LIBERO delta or gripper transform",
            language="UMT5 local tokenizer, instruction prompt, 128 tokens",
            normalization="train-only MIN_MAX; clamp normalized inputs [-5,5]",
            time="past 48, future 32 control frames; strided RGB and 32 action targets",
            architecture="video and action experts share layer count from video_config.num_layers",
            adaptations=[
                "native platform dataset/trainer/checkpoint; no semantic inference from width",
                "recorded robot coordinates replace LIBERO pose delta protocol",
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
        path = Path(settings.get("pretrained_path") or ".").expanduser()
        if (path / "config.json").is_file() and not (path / "policy.json").is_file():
            base = json.loads((path / "config.json").read_text())
            allowed = {f.name for f in fields(LongWAMConfig)} - {
                "input_features",
                "output_features",
            }
            result = {**{k: v for k, v in base.items() if k in allowed}, **result}
            result.setdefault("tokenizer_path", str(path / "tokenizer"))
        if not all(result.get(k) for k in ("video_config", "vae_config", "text_encoder_config")):
            raise ValueError(
                "Long-WAM requires a local converted base with backbone configurations"
            )
        if not Path(result.get("tokenizer_path", "")).expanduser().is_dir():
            raise FileNotFoundError(
                "Long-WAM requires --policy.tokenizer_path to local UMT5 assets"
            )
        result["tokenizer_path"] = str(Path(result["tokenizer_path"]).expanduser().resolve())
        return result

    @staticmethod
    def prepare_options(options, samples):
        options["camera_keys"] = [f"observation.images.{c}" for c in samples.cameras]
        options["action_dim"] = samples.output_features["action"].shape[0]
        options["proprio_dim"] = samples.input_features["observation.state"].shape[0]

    @staticmethod
    def sample_spec(options, *, cameras=()):
        past, future = options.get("past_obs_steps", 48), options.get("future_video_steps", 32)
        offsets = list(range(-past, future + 1, 4))
        return dict(
            delta_indices={
                "action": list(range(options.get("action_horizon", 32))),
                "observation.state": offsets,
                **{f"observation.images.{c}": offsets for c in cameras},
            },
            include_task=True,
        )

    @staticmethod
    def inference_spec(options, *, cameras=()):
        offsets = list(range(-options.get("past_obs_steps", 48), 1, 4))
        return dict(
            delta_indices={
                "action": list(range(options.get("action_horizon", 32))),
                "observation.state": offsets,
                **{f"observation.images.{c}": offsets for c in cameras},
            },
            include_task=True,
        )

    statistics = staticmethod(lambda samples, options=None: limits_statistics(samples))

    @staticmethod
    def action_horizon(config):
        return config.action_horizon

    @staticmethod
    def metric_specs(config):
        return [
            MetricSpec(f"loss_{key}", f"loss_{key}", "samples", loss_count("_loss_weight"))
            for key in ("video", "action")
        ]

    def build(self, options, stats=None):
        from .modeling_longwam import LongWAMPolicy

        return LongWAMPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        from .processor import LongWAMProcessor

        return LongWAMProcessor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        allowed = (
            ("proprio_encoder.", "dit.action.action_encoder.", "dit.action.head.")
            if model.config.reinitialize_embodiment_heads
            else ()
        )
        missing = load_weights(model, settings["pretrained_path"], reinitialize=allowed)
        if missing:
            import logging

            logging.getLogger(__name__).info("Initialized Long-WAM embodiment heads: %s", missing)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("Long-WAM does not use ACT temporal ensembling")
        result["tokenizer_path"] = str(path / "tokenizer")
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        return result

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from transformers import AutoTokenizer

        from alohamini.learning.checkpoint import save_weights

        AutoTokenizer.from_pretrained(
            model.config.tokenizer_path, local_files_only=True
        ).save_pretrained(path / "tokenizer")
        manifest["config"]["tokenizer_path"] = "tokenizer"
        save_weights(path / "model.safetensors", model, model_state)
