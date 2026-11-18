# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for native Alohamini integration in 2026.
"""G0.5 through platform windows, statistics, training and checkpoint interfaces."""

import json
from copy import deepcopy
from pathlib import Path

from alohamini.learning.processor import quantile_statistics
from alohamini.learning.window_statistics import window_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_g05 import G05Config


class G05Algorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = G05Config
    supports_field_transforms = True
    fsdp_classes = ("Qwen3_5DecoderLayer", "G05ActionDecoderLayer", "Qwen3_5VisionBlock")

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        for key, value in dict(
            state="auto",
            batch_size=8,
            steps=100000,
            save_freq=5000,
            mixed_precision="bfloat16",
            trainable_dtype="float32",
            return_uint8=True,
            ema={"enable": False},
        ).items():
            result.setdefault(key, value)
        if "image_size" not in result and result.get("dataset"):
            from alohamini.learning.processor import source_image_size

            result["image_size"] = source_image_size(result)
        result["paper_preset"] = dict(
            name="g05-recorded-flow-v1",
            repository="https://github.com/OpenGalaxea/GalaxeaVLA",
            revision="89f2322b4ad016e192437adc1a2c253b05bab246",
            reference="configs/task/libero.yaml",
            architecture="24-layer Qwen3.5 +24-layer action expert; geometry from local base",
            images="ordered RGB; uint8 bilinear resize256; [-1,1]; saved camera history",
            language="author FMOnly prompt; task text and normalized proprio embeddings",
            numeric="named recorded/transformed coordinates; explicit model slots",
            normalization="train-only q01/q99; per-future-offset action stats; recorded:no clip",
            temporal="32 future actions; selected historical camera/state frames",
            precision="FP32 master/trainable parameters; BF16 AMP; author FP32 islands",
            training="LIBERO flow fine-tuning lr1e-5,batch8,100k,1000 warmup,8 flow draws,joint",
            adaptations=[
                "native data/trainer/checkpoint; recorded robot heads reset by default",
                "no guessed anatomy, coordinate frames or binary gripper slots",
                "decode complete temporal action normalization before execution queue",
                "optional source clipping protocol; explicit semantic transforms remain external",
                "safe non-reentrant checkpointing is optional, not enabled by the LIBERO recipe",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return result

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return result
        path = Path(settings.get("pretrained_path") or result.get("assets_path", ".")).expanduser()
        config_path = path / "config.json"
        if config_path.is_file() and not (path / "policy.json").is_file():
            raw = json.loads(config_path.read_text())
            keys = {
                "author_model_config",
                "n_obs_steps",
                "chunk_size",
                "n_action_steps",
                "image_mean",
                "image_std",
                "use_stepwise_action_norm",
            }
            result = {**{k: v for k, v in raw.items() if k in keys}, **result}
            if result.get("data_protocol") == "g05" and "normalization_mapping" in raw:
                result.setdefault("normalization_mapping", raw["normalization_mapping"])
        result.setdefault("assets_path", str(path))
        result["assets_path"] = str(local_assets(result["assets_path"]))
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        history = list(range(1 - options.get("n_obs_steps", 1), 1))
        result = dict(
            delta_indices={
                "action": list(range(options.get("chunk_size", 32))),
                "observation.state": history,
                **{"observation.images." + key: history for key in cameras},
            },
            include_task=True,
        )
        if options.get("train_discrete_actions") or options.get("predict_cot"):
            result["loss_metadata_fields"] = ("observation.state", "task")
        if options.get("use_language_recipe"):
            from alohamini.datasets.language import LANGUAGE_COLUMNS
            from alohamini.datasets.recipe import TrainingRecipe

            columns = TrainingRecipe.from_dict(options.get("recipe", {})).annotation_columns()
            result["annotation_fields"] = {key: key in columns for key in LANGUAGE_COLUMNS}
        return result

    @classmethod
    def live_history_spec(cls, options, *, cameras=()):
        return {
            key: offsets
            for key, offsets in cls.sample_spec(options, cameras=cameras)["delta_indices"].items()
            if key != "action"
        }

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
                raise ValueError("G0.5 requires distinct recorded/transformed coordinate names")
            if options.get(kind + "_names", names) != names:
                raise ValueError(f"G0.5 {kind} names differ from the selected representation")
            options[kind + "_names"] = list(names)
            if not options.get(kind + "_slots"):
                if not options.get("reinitialize_robot_heads", True):
                    raise ValueError(
                        "Retaining G0.5 heads requires explicit named state/action slots"
                    )
                options[kind + "_slots"] = {name: i for i, name in enumerate(names)}
        selected = ["observation.images." + key for key in samples.cameras]
        options.setdefault("camera_order", selected)
        options.setdefault("camera_sizes", {key: [256, 256] for key in options["camera_order"]})
        if options.get("use_stepwise_action_norm", True):
            options.setdefault("normalization_windows", {}).setdefault(
                "action", list(range(options.get("chunk_size", 32)))
            )
        options.setdefault("processor_metadata", {}).setdefault("frequency", samples.info["fps"])

    @staticmethod
    def statistics(samples, options=None):
        options = options or {}
        windows = options.get("normalization_windows", {})
        if options.get("use_stepwise_action_norm", True):
            windows = {"action": samples.delta_indices["action"], **windows}
        return window_statistics(samples, windows, base=quantile_statistics(samples))

    def build(self, options, stats=None):
        from .modeling_g05 import G05Policy

        return G05Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import G05Processor

        return G05Processor(model, stats, device)

    @staticmethod
    def make_loss_counter(config, stats):
        from .processor import G05LossCounter

        return G05LossCounter(config, stats)

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        specs = []
        if config.train_flow_actions:
            specs.append(
                MetricSpec(
                    "fm_loss",
                    "fm_loss",
                    "weighted_coordinates",
                    loss_count("_flow_weight"),
                    console="fm_loss",
                )
            )
        if config.train_discrete_actions or config.predict_cot:
            specs.append(
                MetricSpec(
                    "ce_loss",
                    "ce_loss",
                    "text_tokens",
                    loss_count("_text_weight"),
                    console="ce_loss",
                )
            )
        return specs

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        heads = (
            "model.proprio_embedder.",
            "model.action_expert.input_proj.",
            "model.action_expert.output_proj.",
        )
        # Published converted bases use backend.model.* tensor names.
        load_weights(
            model.backend,
            settings["pretrained_path"],
            prefix="backend.",
            reinitialize=heads if model.config.reinitialize_robot_heads else (),
        )
        if model.config.reinitialize_robot_heads:
            for module in model.backend.model.proprio_embedder.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            model.backend.model.action_expert.input_proj.reset_parameters()
            model.backend.model.action_expert.output_proj.reset_parameters()
