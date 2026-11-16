"""Native BSP training protocol and physical-time inference adaptation."""

from copy import deepcopy

import numpy as np

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import Processor
from alohamini.policies.diffusion.adapter import DiffusionAlgorithm
from alohamini.trajectory_execution import PlanCapabilities

from .configuration_bspline import BSplineConfig

REVISION = "61ed5f42fced971d50a89b46417493790876ccd1"


class BSplineProcessor(Processor):
    def __init__(self, config, stats, device):
        modes = {
            key: config.normalization_mapping.get(ft.type, "IDENTITY")
            for key, ft in {**config.input_features, **config.output_features}.items()
        }
        super().__init__(stats, device, modes={**modes, "trajectory": "MIN_MAX"})

    def parameters(self, values):
        return self.unnormalize("trajectory", values)


class BSplineAlgorithm(DiffusionAlgorithm):
    config_class = BSplineConfig
    supports_remote_chunks = True
    plan_capabilities = PlanCapabilities(
        kinds=("bspline",), continuity_methods=("none", "phase"), default_continuity="phase"
    )

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        if cfg.get("resume") and cfg.get("paper_preset"):
            return cfg
        defaults = dict(state="auto", seed=42, batch_size=64, image_size=[84, 84])
        for name, value in defaults.items():
            cfg.setdefault(name, value)
        model = cfg.setdefault("model", {})
        for name, value in dict(
            horizon=16,
            n_obs_steps=2,
            n_action_steps=1,
            degree=3,
            fit_tolerance=0.01,
            crop_shape=[76, 76],
            use_group_norm=True,
            pretrained_backbone_weights=None,
            noise_scheduler_type="DDIM",
            num_inference_steps=16,
            down_dims=[256, 512, 1024],
            drop_n_last_frames=0,
        ).items():
            model.setdefault(name, value)
        cfg.setdefault("optimizer", {"type": "adamw"})
        cfg.setdefault("scheduler", {"type": "diffusers_cosine", "warmup_steps": 500})
        cfg.setdefault("ema", {"enable": True, "power": 0.75})
        cfg["paper_preset"] = dict(
            name="bsp-diffusion-recorded-v1",
            source="https://arxiv.org/abs/2607.09648v1",
            repository="https://github.com/B-spline-policy/bspline-policy",
            revision=REVISION,
            reference="train_diffusion_unet_real_hybrid_bspline_workspace.yaml",
            images="RGB [0,1], 84x84, random 76x76 crop; original camera names retained",
            language="unused",
            state="selected named recorded coordinates and dataset units",
            action="absolute recorded command coordinates; joint/TCP semantics remain explicit",
            representation="degree-3 knots + control points; 16 parameter rows",
            time="episode frame indices at recorded FPS; first valid knot anchors each plan",
            normalization="training-only knot/coefficient MIN_MAX; physical actions separate",
            reference_protocol="YAM TCP position+rot6d+gripper; source max_error=0.002",
            adaptation="recorded platform coordinates; no inferred YAM transform; max_error=0.01",
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides",
                    {key: value for key, value in settings.items() if key != "paper_preset"},
                )
            ),
        )
        return cfg

    def sample_spec(self, options, *, cameras=()):
        n = options.get("n_obs_steps", 2)
        offsets = list(range(1 - n, 1))
        return dict(
            delta_indices={
                "observation.state": offsets,
                **{f"observation.images.{c}": offsets for c in cameras},
                "action": [0],
            },
            drop_n_last_frames=0,
        )

    def live_history_spec(self, options, *, cameras=()):
        return {
            k: v
            for k, v in self.sample_spec(options, cameras=cameras)["delta_indices"].items()
            if k != "action"
        }

    @staticmethod
    def prepare_dataset(samples, options):
        from .dataset import BSplineDataset

        return BSplineDataset(samples, options)

    def statistics(self, samples, options=None):
        stats = super().statistics(samples, options)
        stats["trajectory"] = samples.trajectory_statistics()
        return stats

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        basic = {k: v for k, v in stats.items() if k != "trajectory"}
        DiffusionAlgorithm.validate_statistics(basic, expected, names, manifest=manifest)
        values = stats.get("trajectory", {})
        for key in ("mean", "std", "min", "max"):
            x = np.asarray(values.get(key))
            if x.shape != (len(names) + 1,) or not np.isfinite(x).all():
                raise ValueError("Missing or invalid BSP parameter statistics")
        if (np.asarray(values["max"]) < np.asarray(values["min"])).any():
            raise ValueError("Invalid BSP parameter limits")

    @staticmethod
    def metric_specs(config):
        return [
            MetricSpec(
                "loss_diffusion",
                "diffusion_loss",
                "samples",
                loss_count("_loss_weight"),
                console="bspline",
            )
        ]

    def loss_counts(self, batch):
        return {"_loss_weight": len(batch["action"])}

    def build(self, options, stats=None):
        from .modeling_bspline import BSplinePolicy

        return BSplinePolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        return BSplineProcessor(model.config, stats, device)

    @staticmethod
    def action_horizon(config):
        return config.horizon

    @staticmethod
    def predict_action_samples(model, processor, batch, *, fps):
        """Decode physical commands at dataset offsets for held-out MAE."""
        import torch

        from .codec import parameter_plan

        parameters = processor.parameters(model.predict_parameters(batch)).detach().float().cpu()
        names = tuple(str(i) for i in range(model.config.action_feature.shape[0]))
        count = batch["action"].shape[1]
        actions = []
        for values in parameters.numpy():
            plan = parameter_plan(
                values, names, degree=model.config.degree, fps=fps, speed=model.config.time_scale
            )
            actions.append([plan.action_at(i / fps) for i in range(count)])
        return torch.tensor(actions, dtype=torch.float32)

    def predict_plan(self, policy, batch, *, prefix=None, continuity="none"):
        from .codec import parameter_plan

        if continuity not in ("none", "phase"):
            raise ValueError("BSP has not enabled a parameter-space continuity strategy")
        params = policy.processor.parameters(policy.model.predict_parameters(batch))[0]
        params = params.detach().float().cpu().numpy()
        # Deployment scaling and constant Host targets act linearly on coefficients.
        scale = getattr(policy.config, "inference_action_scale", 1.0)
        dims = getattr(policy.config, "inference_action_scale_dims", ())
        if dims:
            params[:, np.asarray(dims) + 1] *= scale
        for name, value in (policy.fixed_dimensions or {}).get("targets", {}).items():
            params[:, 1 + policy.names.index(name)] = value
        plan = parameter_plan(
            params,
            policy.names,
            degree=policy.config.degree,
            fps=policy.fps,
            speed=policy.config.time_scale,
        )
        if continuity == "phase" and prefix is not None:
            from .codec import align_phase

            plan = align_phase(plan, prefix)
        return plan
