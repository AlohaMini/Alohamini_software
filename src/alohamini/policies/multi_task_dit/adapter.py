"""Native Multi-Task DiT recipe, temporal contract and portable CLIP assets."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import limits_statistics, validate_statistics

from .processor_multi_task_dit import MultiTaskDiTProcessor

UPSTREAM_REVISION = "53feaef37b67e8f4c23b6953e880ff535e39d6fe"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class MultiTaskDiTAlgorithm:
    include_task = True
    fsdp_classes = ()  # FSDP2 export has not been adapted; single device/DDP are supported.

    @property
    def config_class(self):
        from .configuration_multi_task_dit import MultiTaskDiTConfig

        return MultiTaskDiTConfig

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        if cfg.get("resume") and cfg.get("paper_preset"):
            return cfg
        cfg.setdefault("state", "auto")
        cfg.setdefault("seed", 42)
        cfg.setdefault("batch_size", 64)
        cfg.setdefault("image_size", [256, 256])
        cfg.setdefault("model", {})
        cfg["optimizer"] = {"type": "adam", **cfg.get("optimizer", {})}
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": 0,
            **(cfg.get("scheduler") or {}),
        }
        cfg["paper_preset"] = {
            "name": "lerobot-multi-task-dit-robot",
            "source": (
                "https://brysonkjones.substack.com/p/dissecting-and-open-sourcing-multitask-"
                "diffusion-transformer-policy"
            ),
            "implementation": "https://github.com/huggingface/lerobot/tree/"
            + UPSTREAM_REVISION
            + "/src/lerobot/policies/multi_task_dit",
            "reference_revision": UPSTREAM_REVISION,
            "protocol": (
                "Bryson Jones community implementation; not an official TRI paper "
                "reproduction or LIBERO benchmark preset"
            ),
            "observation": (
                "history [-1,0], actions [-1,30] by default; episode-boundary repeat padding;"
                " dataset fps, no timestamp resampling"
            ),
            "action": (
                "recorded named action fields in dataset units/order; no inferred TCP, delta,"
                " rotation or gripper conversion"
            ),
            "language": (
                "per-row dataset task; CLIP tokenizer 77 tokens, right padding and "
                "truncation; frozen text encoder with trainable projection"
            ),
            "images": (
                "RGB [0,1], resize to 256x256 then random 224x224 crop in training/center "
                "crop in inference; fixed CLIP mean/std"
            ),
            "normalization": (
                "training split numeric min/max to [-1,1]; checkpoint stats reused for "
                "validation and inference"
            ),
            "reference_recipe": {
                "n_obs_steps": 2,
                "horizon": 32,
                "n_action_steps": 24,
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "use_rope": True,
                "objective": "diffusion",
                "noise_scheduler_type": "DDPM",
                "num_train_timesteps": 100,
                "optimizer": "adam",
                "lr": 2e-5,
                "vision_lr_multiplier": 0.1,
                "betas": [0.95, 0.999],
                "weight_decay": 0.0,
                "scheduler": "cosine",
                "do_mask_loss_for_padding": False,
            },
            "adaptation": (
                "batch size 64 and resize 256 chosen for the local robot pipeline; fixed CLIP"
                " RGB statistics replace dataset RGB statistics; EMA remains trainer-"
                "controlled (off unless enabled)"
            ),
            "overrides": deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides",
                    {key: value for key, value in settings.items() if key != "paper_preset"},
                )
            ),
        }
        return cfg

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        self.mask_loss = options.get("do_mask_loss_for_padding", False)
        return options

    def sample_spec(self, options, *, cameras=()):
        n = options.get("n_obs_steps", self.config_class.n_obs_steps)
        horizon = options.get("horizon", self.config_class.horizon)
        actions = options.get("n_action_steps", self.config_class.n_action_steps)
        drop = options.get("drop_n_last_frames")
        return {
            "delta_indices": {
                "observation.state": list(range(1 - n, 1)),
                **{f"observation.images.{c}": list(range(1 - n, 1)) for c in cameras},
                "action": list(range(1 - n, 1 - n + horizon)),
            },
            "include_task": True,
            "drop_n_last_frames": horizon - actions - n + 1 if drop is None else drop,
        }

    def statistics(self, samples, options=None):
        stats = limits_statistics(samples)
        for key, feature in samples.input_features.items():
            if feature.type == "VISUAL":
                stats[key].update(mean=[[[v]] for v in CLIP_MEAN], std=[[[v]] for v in CLIP_STD])
        return stats

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        validate_statistics(stats, expected, names)
        for key, values in stats.items():
            for field in ("min", "max"):
                if (
                    field not in values
                    or np.asarray(values[field]).shape != np.asarray(values["mean"]).shape
                ):
                    raise ValueError(f"{key}: missing or invalid {field} statistics")

    @staticmethod
    def metric_specs(config):
        masked = config.do_mask_loss_for_padding
        return [
            MetricSpec(
                "loss_" + config.objective,
                "objective_loss",
                "valid_action_elements" if masked else "action_elements_including_repeated_tails",
                loss_count(
                    "_loss_weight",
                    config.action_feature.shape[0] * (1 if masked else config.horizon),
                ),
                console=config.objective,
            )
        ]

    def loss_counts(self, batch):
        return {
            "_loss_weight": int((~batch["action_is_pad"]).sum())
            if getattr(self, "mask_loss", False)
            else len(batch["action"])
        }

    def build(self, options, stats=None):
        from .modeling_multi_task_dit import MultiTaskDiTPolicy

        return MultiTaskDiTPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        return MultiTaskDiTProcessor(model.config, stats, model.tokenizer, device)

    def initialize(self, model, settings):
        if settings.get("pretrained_path"):
            self.load(Path(settings["pretrained_path"]).expanduser(), model)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to Multi-Task DiT")
        options = deepcopy(options)
        for key in ("vision_encoder_name", "text_encoder_name"):
            asset = (path / options[key]).resolve()
            if not asset.is_relative_to(path.resolve()) or not asset.is_dir():
                raise ValueError(f"Multi-Task DiT checkpoint is missing bundled {key} assets")
            options[key] = str(asset)
        options["load_pretrained_encoders"] = False
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        encoder = model.observation_encoder
        vision = (
            encoder.vision_encoder
            if encoder.vision_encoder is not None
            else encoder.vision_encoders[0]
        )
        vision.model.config.save_pretrained(path / "clip_vision")
        encoder.text_encoder.text_encoder.config.save_pretrained(path / "clip_text")
        model.tokenizer.save_pretrained(path / "clip_text")
        manifest["config"].update(
            vision_encoder_name="clip_vision",
            text_encoder_name="clip_text",
            load_pretrained_encoders=False,
        )
        state = model.state_dict() if model_state is None else model_state
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in state.items()},
            path / "model.safetensors",
        )

    def load(self, path, model):
        model.load_state_dict(load_file(path / "model.safetensors"), strict=True)
