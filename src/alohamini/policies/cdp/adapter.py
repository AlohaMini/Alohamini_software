"""CDP2 recipe, past-action normalization and native checkpoint integration."""

from copy import deepcopy

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import Processor
from alohamini.policies.diffusion.adapter import DiffusionAlgorithm

REVISION = "58b23af1e9ac246a11001d7d5388a1de4f53daa5"


class CDPProcessor(Processor):
    def __call__(self, batch):
        result = super().__call__(batch)
        if "past_action" in batch:
            result["past_action"] = super().__call__({"action": batch["past_action"]})["action"]
            result["past_action"] = result["past_action"].masked_fill(
                result["past_action_is_pad"][..., None],
                0,
            )
        return result


class CDPAlgorithm(DiffusionAlgorithm):
    @property
    def config_class(self):
        from .configuration_cdp import CDPConfig

        return CDPConfig

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        if cfg.get("resume") and cfg.get("paper_preset"):
            return cfg
        cfg.setdefault("state", "auto")
        cfg.setdefault("seed", 44)
        cfg.setdefault("batch_size", 64)
        cfg.setdefault("steps", 100000)
        cfg.setdefault("image_size", [84, 84])
        cfg["optimizer"] = {"type": "adamw", **cfg.get("optimizer", {})}
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": 1000,
            **(cfg.get("scheduler") or {}),
        }
        cfg["ema"] = {"enable": True, **(cfg.get("ema") or {})}
        cfg["paper_preset"] = {
            "name": "cdp2-rgb-robot",
            "source": "https://github.com/GaavaMa/Causal-Diffusion-Policy",
            "reference_revision": REVISION,
            "reference": "Causal-Diffusion-Policy/diffusion_policy/config/cdp2.yaml",
            "reference_data": {
                "task": "adroit_hammer",
                "format": "expert Zarr: img -> image, state -> agent_pos, action",
                "shapes": {"image": [3, 84, 84], "agent_pos": [24], "action": [26]},
                "units": "environment-native; field-level physical units unspecified by task YAML",
                "split": "episode seed 42, val_ratio 0.02, max_train_episodes 90",
                "padding": {"before": 31, "after": 7},
                "language": "unused",
            },
            "observation": (
                "32 RGB/state frames ending at current row; episode-local row offsets, no "
                "resampling"
            ),
            "action": (
                "past_action [-32,-1], target [0,7]; recorded names/units; no inferred "
                "coordinate conversion; missing history is zero after normalization"
            ),
            "language": "unused",
            "images": (
                "RGB 84, random crop 76 in training, center crop in inference; ImageNet "
                "mean/std; ResNet18 GN groups=channels/16, no pretrained weights"
            ),
            "normalization": "training-split state/action MIN_MAX [-1,1]; saved statistics reused",
            "reference_recipe": {
                "horizon": 40,
                "n_obs_steps": 32,
                "n_action_steps": 8,
                "num_layers": 8,
                "hidden_dim": 256,
                "num_heads": 4,
                "diffusion_steps": 100,
                "prediction_type": "sample",
                "noise_regime_weights": [0.1, 0, 0, 0.9],
                "history_noise_scale": 0.02,
                "lr": 1e-4,
                "ema_power": 0.75,
                "num_epochs": 3000,
            },
            "adaptation": (
                "CDP2 RGB only, no CDP3 point clouds; native robot observations include the "
                "current frame while history actions are strictly past; episode-relative "
                "cyclic positions; complete chunks only; zero buffer initialization; EMA "
                "owned by trainer; optional tail loss masking; offline queries fully denoise "
                "future window independently, online uses persistent rolling buffer and KV "
                "caches; default native budget 100000 optimizer steps instead of 3000 epochs; "
                "RGB [0,1] uses ImageNet normalization directly, replacing upstream's stacked "
                "image range and ImageNet normalizers"
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
                "observation.state": list(range(1 - n, 1)),
                **{f"observation.images.{c}": list(range(1 - n, 1)) for c in cameras},
                "past_action": list(range(-n, 0)),
                "action": list(range(h - n)),
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
                "full_trajectory_elements",
                loss_count("_loss_weight", config.horizon * config.action_feature.shape[0]),
                console="diffusion",
            )
        ]

    def processor(self, model, stats, device):
        return CDPProcessor.from_config(model.config, stats, device)

    def build(self, options, stats=None):
        from .modeling_cdp import CDPPolicy

        return CDPPolicy(self.config_class(**deepcopy(options)))

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("CDP does not support ACT temporal ensembling")
        if n_action_steps is not None and n_action_steps != options["n_action_steps"]:
            raise ValueError("CDP action chunk size is part of the trained causal mask")
        options = deepcopy(options)
        options["pretrained_backbone_weights"] = None
        return options
