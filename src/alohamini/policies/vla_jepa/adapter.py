"""VLA-JEPA assembly using common data windows, loss accounting and checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.processor import limits_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_vla_jepa import VLAJEPAConfig


class VLAJEPAAlgorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = VLAJEPAConfig
    supports_field_transforms = True
    fsdp_classes = (
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
        "ACBlock",
    )

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        for key, value in dict(
            state="auto",
            batch_size=8,
            steps=120000,
            save_freq=10000,
            mixed_precision="bfloat16",
            return_uint8=True,
            ema={"enable": False},
        ).items():
            result.setdefault(key, value)
        if "image_size" not in result and result.get("dataset"):
            from alohamini.learning.processor import source_image_size

            result["image_size"] = source_image_size(result)
        result["paper_preset"] = dict(
            name="vlajepa-recorded-causal-v1",
            repository="https://github.com/ginwind/VLA-JEPA",
            revision="ec907778b4af46a45742baf5ebfb2e055560346e",
            reference="scripts/configs/vlajepa_libero_ft.yaml",
            architecture="local Qwen3VL2B geometry; action DiT16 layers; latent predictor12 layers",
            images="current RGB for actions;8 teacher rows;local Qwen/VJEPA processors",
            language="original temporal/action-readout prompt;pre-final-norm Qwen block features",
            state="named recorded coordinates,MEAN_STD;explicit dimensional head adaptation",
            action="named commands,MIN_MAX,7 future rows;explicit pose/gripper representation",
            temporal="video tubelets2;future targets never enter the action-observation branch",
            world="selected cameras;causal prefix teacher;shared selects original context",
            precision="BF16 Qwen,FP32 DiT/predictor;frozen FP32 teacher;platform AMP/TF32",
            training="author AdamW LR groups1e-5/1e-4/5e-4;120k cosine,20k warmup",
            adaptations=[
                "native reader,train-only stats,trainer and checkpoint",
                "valid temporal WM masks available without decoding images",
                "no silent camera duplication/trimming or fabricated semantic labels",
                "causal teacher context default;shared is explicit paper-protocol selection",
                "frozen teacher stays eval;all-frozen/readout-unused parameters excluded",
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
        base = Path(settings.get("pretrained_path") or result.get("assets_path", ".")).expanduser()
        if (base / "config.json").is_file() and not (base / "policy.json").is_file():
            raw = json.loads((base / "config.json").read_text())
            allowed = {
                f.name
                for f in fields(VLAJEPAConfig)
                if not f.name.startswith(("optimizer_", "scheduler_"))
            } - {
                "input_features",
                "output_features",
            }
            result = {**{k: v for k, v in raw.items() if k in allowed}, **result}
            if raw.get("type") == "vla_jepa":
                result.setdefault("base_weights_mode", "policy")
                result.setdefault("retain_unused_predictor_heads", True)
                result.setdefault("world_model_rope_legacy", True)
        assets = local_assets(result.get("assets_path", base))
        result["assets_path"] = str(assets)
        for name, folder in (("qwen_config", "qwen"), ("jepa_config", "vjepa2")):
            path = assets / folder / "config.json"
            if name not in result and path.is_file():
                result[name] = json.loads(path.read_text())
        if result.get("jepa_config"):
            result.setdefault("jepa_tubelet_size", result["jepa_config"].get("tubelet_size", 2))
        return result

    @staticmethod
    def prepare_options(options, samples):
        options["state_dim"] = samples.input_features["observation.state"].shape[0]
        options["action_dim"] = samples.output_features["action"].shape[0]
        if options.get("enable_world_model", True):
            selected = list(options["input_features"])
            cameras = options.get("world_model_cameras", []) or [
                key for key in selected if key.startswith("observation.images.")
            ]
            options["world_model_cameras"] = [
                key if key.startswith("observation.images.") else "observation.images." + key
                for key in cameras
            ]

    @staticmethod
    def sample_spec(options, *, cameras=()):
        horizon = options.get("chunk_size", 7)
        windows = {"action": list(range(horizon))}
        if options.get("enable_world_model", True):
            frames = options.get("num_video_frames", 8)
            stride = max(1, (horizon - 1) // (frames - 1))
            windows.update(
                {
                    "observation.images." + key: [i * stride for i in range(frames)]
                    for key in cameras
                }
            )
        return dict(delta_indices=windows, include_task=True)

    @staticmethod
    def inference_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 7)))}, include_task=True
        )

    @staticmethod
    def statistics(samples, options=None):
        return limits_statistics(samples)

    @staticmethod
    def make_loss_counter(config, stats):
        def counts(batch):
            result = {"_loss_weight": len(batch["action"])}
            if config.enable_world_model:
                from .modeling_vla_jepa import world_validity

                result["_world_weight"] = int(world_validity(config, batch).any(-1).sum())
            return result

        return counts

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        result = [MetricSpec("action_loss", "action_loss", "samples", loss_count("_loss_weight"))]
        if config.enable_world_model:
            result.append(MetricSpec("wm_loss", "wm_loss", "samples", loss_count("_world_weight")))
        return result

    def build(self, options, stats=None):
        from .modeling_vla_jepa import VLAJEPAPolicy

        return VLAJEPAPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import VLAJEPAProcessor

        return VLAJEPAProcessor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        cfg = model.config
        if cfg.base_weights_mode == "backbones":
            load_weights(model.model.qwen.model, Path(cfg.assets_path) / "qwen")
            if cfg.enable_world_model:
                load_weights(model.model.video_encoder, Path(cfg.assets_path) / "vjepa2")
        else:
            heads = (
                (
                    "model.action_model.action_encoder.layer1.",
                    "model.action_model.action_decoder.layer2.",
                    "model.action_model.state_encoder.",
                )
                if cfg.reinitialize_robot_heads
                else ()
            )
            if cfg.reinitialize_world_heads:
                heads += (
                    "model.video_predictor.predictor_embed.",
                    "model.video_predictor.predictor_proj.",
                )
            excluded = (
                () if cfg.enable_world_model else ("model.video_encoder.", "model.video_predictor.")
            )
            load_weights(model, settings["pretrained_path"], reinitialize=heads, exclude=excluded)
            if cfg.reinitialize_robot_heads:
                model.model.action_model.action_encoder.layer1.reset_parameters()
                model.model.action_model.action_decoder.layer2.reset_parameters()
                for layer in model.model.action_model.state_encoder.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
            if cfg.reinitialize_world_heads:
                model.model.video_predictor.predictor_embed.reset_parameters()
                model.model.video_predictor.predictor_proj.reset_parameters()

    @staticmethod
    def load(path, model):
        from alohamini.learning.weights import load_weights

        excluded = (
            ()
            if model.config.enable_world_model
            else ("model.video_encoder.", "model.video_predictor.")
        )
        load_weights(model, path, exclude=excluded)

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from transformers import AutoProcessor, AutoVideoProcessor

        from alohamini.learning.checkpoint import save_weights

        assets = path / "assets"
        AutoProcessor.from_pretrained(
            Path(model.config.assets_path) / "qwen", local_files_only=True
        ).save_pretrained(assets / "qwen")
        if model.config.enable_world_model:
            AutoVideoProcessor.from_pretrained(
                Path(model.config.assets_path) / "vjepa2", local_files_only=True
            ).save_pretrained(assets / "vjepa2")
        manifest["config"]["assets_path"] = "assets"
        save_weights(path / "model.safetensors", model, model_state)
