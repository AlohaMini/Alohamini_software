"""Shared Nano/Edge protocol adapters; all data and execution use platform interfaces."""

import json
from copy import deepcopy
from pathlib import Path

from alohamini.learning.ema import ema_settings
from alohamini.learning.processor import quantile_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_cosmos3 import Cosmos3EdgeConfig, Cosmos3NanoConfig


class Cosmos3Algorithm(ChunkAlgorithm):
    asset_fields = ("tokenizer_path",)
    config_class = Cosmos3NanoConfig
    supports_field_transforms = True
    loss_terms = ("video_loss", "action_loss")
    fsdp_classes = (
        "Qwen3VLMoTDecoderLayer",
        "Qwen3VLVisionBlock",
        "Nemotron3DenseVLMoTDecoderLayer",
        "Nemotron3DenseVLVisionEncoderLayer",
    )

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        # EDM2 sigma_rel=.1: largest real root of [1,7,16-s^-2,12-s^-2], plus1.
        result["ema"] = ema_settings(
            dict(
                enable=True,
                schedule="power_law",
                power=7.937203937601809,
                max_decay=0.9999999999999999,
            ),
            result.get("ema", {}),
        )
        for key, value in dict(
            state="auto",
            batch_size=1,
            steps=16000,
            save_freq=2000,
            mixed_precision="bfloat16",
            trainable_dtype="float32",
            return_uint8=True,
        ).items():
            result.setdefault(key, value)
        if "image_size" not in result and result.get("dataset"):
            from alohamini.learning.processor import source_image_size

            result["image_size"] = source_image_size(result)
        result["paper_preset"] = dict(
            name="cosmos3-recorded-v1",
            repository="https://github.com/NVIDIA/cosmos-framework",
            revision="0822f1d1f79f0e75a249ddddcb77c929932e881e",
            reference=(
                "cosmos_framework/configs/base/experiment/action/posttrain_config/"
                "action_policy_libero_nano.py"
            ),
            architecture="Nano36-layer Qwen3-VL; Edge28-layer Nemotron; local config authoritative",
            images="ordered RGB; native resize/quantization/concat/reflection canvas; Wan2.2 VAE",
            language="local tokenizer chat template; JSON task/video caption; CFG dropout0.1",
            state="optional explicit action-layout state; no dimension-based pose inference",
            action="named commands; quantiles; first D of max64 capacity with fresh robot heads",
            temporal="16 future commands; current+16 future frames; current-only inference",
            precision="FP32 master parameters; BF16 AMP/frozen backbone; FP32 time/rotary math",
            training="joint video/action losses10/10; generation stream; action LR multiplier5",
            adaptations=[
                "native reader/trainer/checkpoints and linear warmup/decay",
                "recorded commands replace LIBERO rot6d convention; no gripper/rotation guesses",
                "idle count omitted without declared physical motion semantics; prompt uses0",
                "camera order/roles explicit; no wrist/scene role inferred from dimensionality",
                "FP32 parameters provide master weights with standard AdamW; no Apex dependency",
                "global TF32/compilation settings are not changed by model construction",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return result

    def options(self, settings, device):
        result = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return result
        path = Path(
            settings.get("pretrained_path") or result.get("tokenizer_path", ".")
        ).expanduser()
        if (path / "config.json").is_file() and not (path / "policy.json").is_file():
            raw = json.loads((path / "config.json").read_text())
            # Inherit geometry; source DROID data/optimizer choices do not define this robot.
            keys = {
                "mot_backbone",
                "vae",
                "max_action_dim",
                "latent_patch_size",
                "timestep_scale",
                "timestep_embedding_dim",
                "num_embodiment_domains",
                "base_fps",
                "temporal_modality_margin",
            }
            result = {**{k: v for k, v in raw.items() if k in keys}, **result}
        result.setdefault("tokenizer_path", str(path / "tokenizer"))
        result["tokenizer_path"] = str(local_assets(result["tokenizer_path"]))
        result["variant"] = self.config_class.__dataclass_fields__["variant"].default
        return result

    @staticmethod
    def prepare_options(options, samples):
        features = {**samples.info["features"], **getattr(samples, "model_features", {})}
        names = features["action"].get("names", [])
        if options.get("action_feature_names", names) != names:
            raise ValueError("Cosmos3 action names differ from the dataset representation")
        options["action_feature_names"] = list(names)
        if options.get("use_state"):
            state_names = features["observation.state"].get("names", [])
            if options.get("state_action_names", state_names) != state_names:
                raise ValueError("Cosmos3 state names differ from the dataset representation")
            options["state_action_names"] = list(state_names)
        selected = ["observation.images." + c for c in samples.cameras]
        options["camera_keys"] = [
            k if k.startswith("observation.images.") else "observation.images." + k
            for k in options.get("camera_keys") or selected
        ]
        options.setdefault("fps", samples.info["fps"])

    @staticmethod
    def sample_spec(options, *, cameras=()):
        count = options.get("chunk_size", 16)
        return dict(
            delta_indices={
                "action": list(range(count)),
                **{"observation.images." + c: list(range(count + 1)) for c in cameras},
            },
            include_task=True,
            drop_n_last_frames=count,
        )

    @staticmethod
    def inference_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 16)))}, include_task=True
        )

    @staticmethod
    def statistics(samples, options=None):
        return quantile_statistics(samples)

    def build(self, options, stats=None):
        from .modeling_cosmos3 import Cosmos3Policy

        return Cosmos3Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import Cosmos3Processor

        return Cosmos3Processor(model, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        heads = tuple("model.network." + n + "." for n in model.model.network._action_modules)
        load_weights(
            model,
            settings["pretrained_path"],
            reinitialize=heads if model.config.reinitialize_robot_heads else (),
        )
        if model.config.reinitialize_robot_heads:
            import torch

            network = model.model.network
            network.action2llm._init_weights(network.action2llm.input_size**-0.5)
            network.llm2action._init_weights(network.llm2action.input_size**-0.5)
            std = network.action_modality_embed.weight.numel() ** -0.5
            torch.nn.init.trunc_normal_(
                network.action_modality_embed.weight, std=std, a=-3 * std, b=3 * std
            )

    @staticmethod
    def checkpoint_options(path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("Cosmos3 does not use ACT temporal ensembling")
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        result["tokenizer_path"] = str(path / "assets/tokenizer")
        return result

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        model.save_assets(path / "assets")
        manifest["config"]["tokenizer_path"] = "assets/tokenizer"
        save_weights(path / "model.safetensors", model, model_state)


class Cosmos3NanoAlgorithm(Cosmos3Algorithm):
    config_class = Cosmos3NanoConfig


class Cosmos3EdgeAlgorithm(Cosmos3Algorithm):
    config_class = Cosmos3EdgeConfig
