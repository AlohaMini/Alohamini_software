"""LingBot-VA through platform data, optimizer execution and safetensors checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

from alohamini.learning.processor import quantile_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_lingbot_va import LingBotVAConfig


class LingBotVAAlgorithm(ChunkAlgorithm):
    asset_fields = ("wan_pretrained_path", "text_feature_store_path")
    config_class = LingBotVAConfig
    queues_online_observations = True
    supports_remote_chunks = False
    supports_field_transforms = True
    fsdp_classes = ("WanTransformerBlock", "RMSNorm")
    loss_terms = ("latent_loss", "action_loss")

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        for key, value in dict(
            state="auto",
            batch_size=1,
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
            name="lingbot-va-recorded-v1",
            repository="https://github.com/Robbyant/lingbot-va",
            revision="7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb",
            reference="wan_va/configs/va_libero_cfg.py",
            architecture="shared Wan2.2 transformer30 layers; separate video/action time stems",
            images="ordered RGB,128x128 bilinear,[-1,1]; local frozen Wan VAE",
            language="UMT5 local assets;512 tokens; cleaned task; zero-pad embeddings",
            state="recorded for generic field transforms; not consumed by this model",
            action="named commands mapped into30D after q01/q99+1e-6 normalization",
            normalization=(
                "recorded preserves outliers; lingbot clips normalized targets to [-1.5,1.5]"
            ),
            temporal=(
                "initial visual latent +4 future latent frames; 16 future commands; "
                "physical-zero bootstrap group"
            ),
            training=(
                "full future clips; episode end trim; timestep weighting; block-causal dual streams"
            ),
            precision=(
                "FP32 trainable storage; BF16 compute/frozen assets; FP32 norms; float64 RoPE"
            ),
            adaptations=[
                "platform data/trainer/checkpoint; exact train-only statistics",
                "normalize commands during both training and execution",
                "named slots and optional fresh robot heads; no guessed pose/quaternion semantics",
                "explicit masks per forward; checkpointing cannot read another microbatch mask",
                "CPU SDPA/reference and GPU Flex use the same causal visibility",
                "offline chunks independent; online per-tick observation feedback retained",
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
        path = Path(
            settings.get("pretrained_path") or result.get("wan_pretrained_path", ".")
        ).expanduser()
        base = path / "transformer/config.json"
        if base.is_file():
            raw = json.loads(base.read_text())
            allowed = {f.name for f in fields(LingBotVAConfig)} - {
                "input_features",
                "output_features",
                "attn_mode",
            }
            result = {**{k: v for k, v in raw.items() if k in allowed}, **result}
        result["wan_pretrained_path"] = str(local_assets(result.get("wan_pretrained_path", path)))
        vae = Path(result["wan_pretrained_path"]) / "vae/config.json"
        if vae.is_file():
            cfg = json.loads(vae.read_text())
            factor = 2 ** (len(cfg.get("dim_mult", [1, 2, 4, 4])) - 1) * (
                cfg.get("patch_size") or 1
            )
            result.setdefault("vae_spatial_factor", factor)
        return result

    @staticmethod
    def prepare_options(options, samples):
        selected = ["observation.images." + camera for camera in samples.cameras]
        declared = options.get("obs_cam_keys") or selected
        options["obs_cam_keys"] = [
            key if key.startswith("observation.images.") else "observation.images." + key
            for key in declared
        ]
        feature = getattr(samples, "model_features", {}).get(
            "action", samples.info["features"]["action"]
        )
        names = feature.get("names", [])
        if options.get("action_names", names) != names:
            raise ValueError("LingBot-VA command names differ from dataset representation")
        options["action_names"] = list(names)
        if not options.get("action_slots"):
            if not options.get("reinitialize_robot_heads", True):
                raise ValueError(
                    "Keeping LingBot-VA robot heads requires explicit named action slots"
                )
            options["action_slots"] = {name: i for i, name in enumerate(names)}

    @staticmethod
    def sample_spec(options, *, cameras=()):
        count = options.get("frame_chunk_size", 4) * options.get("action_per_frame", 4)
        stride = options.get("action_per_frame", 4) // 4
        return dict(
            delta_indices={
                "action": list(range(count)),
                **{"observation.images." + c: list(range(0, count + 1, stride)) for c in cameras},
            },
            include_task=True,
            drop_n_last_frames=count,
        )

    @staticmethod
    def inference_spec(options, *, cameras=()):
        return dict(
            delta_indices={
                "action": list(
                    range(options.get("frame_chunk_size", 4) * options.get("action_per_frame", 4))
                )
            },
            include_task=True,
        )

    @staticmethod
    def prepare_training_assets(options, samples, stats, *, validation_samples=None):
        from .assets import prepare_text_features

        return prepare_text_features(options, samples, validation_samples=validation_samples)

    @staticmethod
    def statistics(samples, options=None):
        return quantile_statistics(samples)

    def build(self, options, stats=None):
        from .modeling_lingbot_va import LingBotVAPolicy

        return LingBotVAPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import LingBotVAProcessor

        return LingBotVAProcessor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        path = Path(settings["pretrained_path"]) / "transformer"
        # Published weights retain an obsolete Conv3d; current author code uses only the MLP.
        load_weights(
            model.transformer,
            path,
            filename="diffusion_pytorch_model.safetensors",
            exclude=("patch_embedding.",),
        )
        if model.config.reinitialize_robot_heads:
            model.transformer.action_embedder.reset_parameters()
            model.transformer.action_proj_out.reset_parameters()

    @staticmethod
    def checkpoint_options(path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("LingBot-VA does not use ACT temporal ensembling")
        if n_action_steps not in (
            None,
            result.get("frame_chunk_size", 4) * result.get("action_per_frame", 4),
        ):
            raise ValueError("LingBot-VA streams complete chunks with per-tick feedback")
        result["wan_pretrained_path"] = str(local_assets(result["wan_pretrained_path"]))
        if result.get("text_feature_store_path"):
            result["text_feature_store_path"] = str(path / "assets/text_features")
        return result

    @staticmethod
    def save(path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        if model._text_features.path is not None:
            model._text_features.copy_to(path / "assets/text_features")
            manifest["config"]["text_feature_store_path"] = "assets/text_features"
        save_weights(path / "model.safetensors", model, model_state)
