"""Native VQ-BeT windows, two-stage training and strict checkpoint loading."""

from copy import deepcopy

from alohamini.learning.processor import Processor, limits_statistics, validate_statistics
from alohamini.policies.diffusion.adapter import DiffusionAlgorithm

from .configuration_vqbet import VQBeTConfig


class VQBeTAlgorithm(DiffusionAlgorithm):
    config_class = VQBeTConfig
    include_task = False
    queues_online_observations = True
    supports_remote_chunks = False
    fsdp_classes = ("Block",)
    validate_statistics = staticmethod(validate_statistics)

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto",
            image_size=[96, 96],
            batch_size=64,
            mixed_precision="none",
            ema={"enable": False},
        ).items():
            cfg.setdefault(key, value)
        cfg["optimizer"] = {"type": "adam", **cfg.get("optimizer", {})}
        cfg["paper_preset"] = dict(
            name="vqbet-recorded-v1",
            paper="https://arxiv.org/abs/2403.03181",
            repository="https://github.com/jayLEE0301/vq_bet_official",
            revision="09d4851288ca5deaaa1ab367a208e520f8ee9a84",
            images=(
                "RGB 96x96, author ImageNet normalization (.229,.224,.255 std); random 84x84 crop"
            ),
            state="selected recorded coordinates with explicit names and units",
            action="absolute recorded commands; history-aligned complete action windows",
            language="unused",
            normalization="train-only state/action limits [-1,1]",
            time="5 past/current observations; 3 prediction tokens; 5 actions per token",
            training="RVQ 20000 successful optimizer updates, then GPT; count saved as buffer",
            adaptations=[
                "native platform dataset/trainer/checkpoint",
                "complete action windows avoid silently training on episode padding",
                "multiple camera tokens follow the source multi-view model path",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return cfg

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            result["pretrained_backbone_weights"] = None
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        observations = options.get("n_obs_steps", 5)
        tokens, chunk = options.get("n_action_pred_token", 3), options.get("action_chunk_size", 5)
        history = list(range(1 - observations, 1))
        return dict(
            delta_indices={
                "observation.state": history,
                **{f"observation.images.{c}": history for c in cameras},
                "action": list(range(1 - observations, tokens + chunk - 1)),
            },
            drop_n_first_frames=observations - 1,
            drop_n_last_frames=tokens + chunk - 2,
        )

    @staticmethod
    def action_horizon(config):
        return config.action_chunk_size

    @staticmethod
    def action_offset(config):
        return 0

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": len(batch["action"])}

    @staticmethod
    def metric_specs(config):
        # The available diagnostics change at the stage boundary. The total
        # objective is always logged by the shared trainer; do not fill missing terms.
        return []

    @staticmethod
    def statistics(samples, options=None):
        result = limits_statistics(samples)
        for key in samples.input_features:
            if key.startswith("observation.images."):
                result[key]["mean"] = [[[v]] for v in (0.485, 0.456, 0.406)]
                result[key]["std"] = [[[v]] for v in (0.229, 0.224, 0.255)]
        return result

    def processor(self, model, stats, device):
        return Processor.from_config(model.config, stats, device)

    def build(self, options, stats=None):
        from .modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def predict_action_samples(model, processor, batch, *, fps):
        import torch

        windows = model.predict_action_chunk(batch, all_windows=True)
        # Each query supplies its aligned first action; the final query supplies
        # the remaining tail. Every value is predicted without future observations.
        result = torch.cat((windows[:, :, 0], windows[:, -1, 1:]), dim=1)
        return processor.action(result, context=batch.get("_action_context"))

    @staticmethod
    def checkpoint_options(path, options, device, n_action_steps, temporal_ensemble_coeff):
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("VQ-BeT does not use ACT temporal ensembling")
        if n_action_steps is not None and n_action_steps != options.get("action_chunk_size", 5):
            raise ValueError("VQ-BeT executes action_chunk_size commands per chunk")
        return {**deepcopy(options), "pretrained_backbone_weights": None}
