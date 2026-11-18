"""FineART through the native data reader, loss counting, trainer and checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.processor import quantile_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_fineart_vla import FineARTVLAConfig


class FineARTVLAAlgorithm(ChunkAlgorithm):
    asset_fields = ("assets_path", "fast_tokenizer_path")
    config_class = FineARTVLAConfig
    fsdp_classes = ("GemmaRMSNorm", "Embedding")
    language_query_kinds = ("subtask", "vqa")
    supports_field_transforms = True

    @staticmethod
    def query_language(model, processor, batch, **kwargs):
        from .query import query_language

        return query_language(model, processor, batch, **kwargs)

    @staticmethod
    def apply_preset(settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto",
            steps=30000,
            batch_size=32,
            save_freq=1000,
            mixed_precision="none",
            return_uint8=True,
            ema={"enable": False},
        ).items():
            cfg.setdefault(key, value)
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="fineart-recorded-subtask-v1",
            source="https://arxiv.org/abs/2609.36416",
            implementation="https://github.com/huggingface/lerobot",
            revision="7910c20a75834ad3fa6a9792df4d44a614d1bba6",
            architecture="PaliGemma2B18 layers + Gemma300M18 layers; timestep adaRMS; padded32D",
            images="selected recorded RGB, aspect-preserving224 padding; [-1,1]",
            language="recorded timestamped subtasks; 30% task->subtask text,70% subtask->action",
            numeric="named recorded units; q01/q99+1e-6; no clipping or zeroing",
            time="50 future row offsets; recorded cadence; repeated episode tails retained",
            losses="flow10,text1,FAST1; KI;5 independent flow draws; text z-loss1e-4",
            precision="BF16 core, FP32 projections/norms; platform AMP/TF32 policy",
            adaptations=[
                "robot fields stay named recorded coordinates; no guessed relative joint slots",
                "no silent missing-label fallback or token truncation",
                "rank0 train-only FAST fitting, held-out reconstruction checks, immutable cache",
                "complete byte alphabet prevents unseen coefficient symbols being dropped",
                "camera-bound normalized xy/xyxy VQA uses reserved PaliGemma loc tokens",
                "optional platform subtask controller; tools are not included",
                "native global token/sample loss denominators for accumulation and validation",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides",
                    {key: value for key, value in settings.items() if key != "paper_preset"},
                )
            ),
        )
        return cfg

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if not settings.get("resume"):
            base = Path(settings.get("pretrained_path") or ".").expanduser()
            if (base / "config.json").is_file() and not (base / "policy.json").is_file():
                raw = json.loads((base / "config.json").read_text())
                allowed = {item.name for item in fields(FineARTVLAConfig)} - {
                    "input_features",
                    "output_features",
                    "assets_path",
                    "fast_tokenizer_path",
                }
                result = {**{key: value for key, value in raw.items() if key in allowed}, **result}
                core = {
                    key: raw[key]
                    for key in ("dtype", "paligemma_variant", "action_expert_variant")
                    if key in raw
                }
                for source, target in (
                    ("chunk_size", "action_horizon"),
                    ("max_action_dim", "action_dim"),
                    ("tokenizer_max_length", "max_token_len"),
                ):
                    if source in raw:
                        core[target] = raw[source]
                result["network"] = {**core, **result.get("network", {})}
                if (base / "action_tokenizer").is_dir():
                    result.setdefault("fast_tokenizer_path", str(base / "action_tokenizer"))
            result["assets_path"] = str(local_assets(result.get("assets_path", "")))
            if result.get("fast_tokenizer_path"):
                result["fast_tokenizer_path"] = str(local_assets(result["fast_tokenizer_path"]))
            if not settings.get("pretrained_path"):
                raise ValueError("FineART requires --policy.path to a local converted PyTorch base")
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        from alohamini.datasets.language import LANGUAGE_COLUMNS
        from alohamini.datasets.recipe import TrainingRecipe

        from .configuration_fineart_vla import default_recipe

        columns = TrainingRecipe.from_dict(
            options.get("recipe", default_recipe())
        ).annotation_columns()
        if options.get("vqa_cameras"):
            columns.add("language_events")
        return dict(
            delta_indices={
                "action": list(range(options.get("network", {}).get("action_horizon", 50)))
            },
            include_task=True,
            exclude_fixed_state=False,
            annotation_fields={key: key in columns for key in LANGUAGE_COLUMNS},
            loss_metadata_fields=("observation.state", "task"),
        )

    @staticmethod
    def statistics(samples, options=None):
        return quantile_statistics(samples)

    def build(self, options, stats=None):
        from .modeling_fineart_vla import FineARTVLAPolicy

        return FineARTVLAPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import FineARTProcessor

        result = FineARTProcessor(model.config, stats, device)
        width = (
            model.network.paligemma_with_expert.paligemma.language_model.embed_tokens.num_embeddings
        )
        if len(result.text.tokenizer) > width:
            raise ValueError(
                "FineART tokenizer IDs exceed the model's vocabulary; use matching local assets"
            )
        return result

    @staticmethod
    def make_loss_counter(config, stats):
        from .processor import FineARTTextEncoder

        return FineARTTextEncoder(config, stats).counts

    @staticmethod
    def prepare_training_assets(options, samples, stats, *, validation_samples=None):
        from .assets import prepare_fast_assets

        return prepare_fast_assets(options, samples, stats)

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        return [
            MetricSpec(f"loss_{name}", f"{name}_loss", unit, loss_count(weight), console=name)
            for name, unit, weight in [
                ("flow", "action_samples", "_flow_weight"),
                ("text", "text_tokens", "_text_weight"),
                ("fast", "action_samples", "_fast_weight"),
            ]
        ]

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(
            model.network, settings["pretrained_path"], prefix=model.config.base_weights_prefix
        )

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = super().checkpoint_options(
            path, options, device, n_action_steps, temporal_ensemble_coeff
        )
        if result.get("fast_tokenizer_path"):
            result["fast_tokenizer_path"] = str(path / "assets/fast")
        return result

    def save(self, path, model, manifest, model_state=None):
        super().save(path, model, manifest, model_state)
        if model.config.fast_tokenizer_path:
            manifest["config"]["fast_tokenizer_path"] = "assets/fast"

    @staticmethod
    def supports_action_prediction(config):
        # Mixed high/low-level training validation reports objectives, not oracle subtask MAE.
        return False
