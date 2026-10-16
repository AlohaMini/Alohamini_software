"""DiT-Block recipe and native dataset/checkpoint integration."""

from copy import deepcopy

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.policies.diffusion.adapter import DiffusionAlgorithm

REVISION = "ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73"


class DiTPolicyAlgorithm(DiffusionAlgorithm):
    @property
    def config_class(self):
        from .configuration_dit_policy import DiTPolicyConfig

        return DiTPolicyConfig

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        if cfg.get("resume") and cfg.get("paper_preset"):
            return cfg
        cfg.setdefault("state", "auto")
        cfg.setdefault("seed", 292285)
        cfg.setdefault("batch_size", 150)
        cfg.setdefault("steps", 150000)
        cfg.setdefault("image_size", [256, 256])
        cfg["optimizer"] = {"type": "adamw", **cfg.get("optimizer", {})}
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": 2000,
            **(cfg.get("scheduler") or {}),
        }
        cfg["paper_preset"] = {
            "name": "dit-policy-rgb-robot",
            "source": "https://github.com/SudeepDasari/dit-policy",
            "reference_revision": REVISION,
            "reference": (
                "README ac_chunk=100; experiments/agent/diffusion.yaml; trainer/bc_cos_sched.yaml"
            ),
            "reference_data": {
                "format": "Robobuf: state, enc_cam_<index>, action; ac_norm.json loc/scale",
                "aloha": {"state_dim": 14, "action_dim": 14, "camera_indexes": [0, 1, 3]},
                "readme_end_effector_r6": {
                    "state_dim": 7,
                    "action_dim": 10,
                    "camera_indexes": [0],
                },
                "units": "defined by external converter; no dimension-based coordinate inference",
                "split": "500 held-out transitions in reference; native uses held-out episodes",
                "padding": "repeat episode boundary; padded future actions excluded from loss",
                "language": "unused",
            },
            "observation": (
                "current state; image history ending at current row, default one frame; "
                "newest-first image tokens within each camera"
            ),
            "action": (
                "named dataset actions and units; no inferred rotation, delta or TCP conversion"
            ),
            "language": "unused",
            "images": (
                "RGB 256; train random resized crop 224 scale [0.9,1] and Gaussian blur; eval "
                "resize 224; ImageNet mean/std; ResNet18 GN(16), spatial tokens"
            ),
            "normalization": (
                "reference buffer is preprocessed externally; native state IDENTITY, action "
                "training-split MIN_MAX [-1,1]; explicit mapping and statistics saved"
            ),
            "reference_recipe": {
                "hidden_dim": 512,
                "num_layers": 6,
                "num_heads": 8,
                "train_diffusion_steps": 100,
                "eval_diffusion_steps": 8,
                "prediction_type": "epsilon",
                "lr": 3e-4,
                "batch_size": 150,
                "warmup_steps": 2000,
                "ema": False,
            },
            "adaptation": (
                "native episode split and named robot coordinates; action normalization "
                "fitted locally; torchvision ImageNet V1 BN affine/conv weights initialize GN "
                "instead of the separately distributed author GN checkpoint; default spatial "
                "tokens follow agent YAML (README pooled backbone selectable with "
                "pooled_vision=true)"
            ),
            "overrides": deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        }
        return cfg

    def options(self, settings, device):
        return deepcopy(settings.get("model", {}))

    def sample_spec(self, options, *, cameras=()):
        n = options.get("n_obs_steps", self.config_class.n_obs_steps)
        h = options.get("horizon", self.config_class.horizon)
        return {
            "delta_indices": {
                "observation.state": [0],
                **{f"observation.images.{c}": list(range(1 - n, 1)) for c in cameras},
                "action": list(range(h)),
            },
            "drop_n_last_frames": options.get("drop_n_last_frames", 0),
        }

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": len(batch["action"])}

    @staticmethod
    def metric_specs(config):
        return [
            MetricSpec(
                "loss_diffusion",
                "diffusion_loss",
                "batch_action_dimensions",
                loss_count("_loss_weight", config.action_feature.shape[0]),
                console="diffusion",
            )
        ]

    def build(self, options, stats=None):
        from .modeling_dit_policy import DiTPolicy

        return DiTPolicy(self.config_class(**deepcopy(options)))

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("dit_policy does not support ACT temporal ensembling")
        options = deepcopy(options)
        options["pretrained_backbone_weights"] = None
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options
