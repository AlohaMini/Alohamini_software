"""LeWM uses the native trainer; action generation requires an explicit visual goal."""

from copy import deepcopy

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import IMAGENET_MEAN, IMAGENET_STD, limits_statistics
from alohamini.policies.chunk import ChunkAlgorithm

from .configuration_lewm import LeWMConfig
from .preset import apply_preset


class LeWMAlgorithm(ChunkAlgorithm):
    config_class = LeWMConfig
    apply_preset = staticmethod(apply_preset)
    include_task = False
    supports_field_transforms = True
    supports_remote_chunks = False
    fsdp_classes = ("ViTLayer", "ConditionalBlock")

    def options(self, settings, device):
        if settings.get("data", {}).get("action_representation") == "tcp_delta":
            raise ValueError(
                "LeWM transition action blocks require recorded or tcp_absolute coordinates"
            )
        options = deepcopy(settings.get("model", {}))
        self.block = options.get("action_block", 5)
        return options

    @staticmethod
    def sample_spec(options, *, cameras=()):
        history, block = options.get("history_size", 3), options.get("action_block", 5)
        start = -(history - 1) * block
        return dict(
            delta_indices={
                "action": list(range(start, block + 1)),
                **{
                    f"observation.images.{c}": list(range(start, block + 1, block)) for c in cameras
                },
            },
            drop_n_last_frames=block,
        )

    @staticmethod
    def prepare_options(options, samples):
        size = options.get("action_block", 5) * options.get("planning_horizon", 5)
        options.setdefault("chunk_size", size)
        options.setdefault("n_action_steps", size)

    @staticmethod
    def evaluation_sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={
                "action": list(range(options["chunk_size"])),
                **{f"observation.images.{c}": [0] for c in cameras},
            }
        )

    def statistics(self, samples, options=None):
        stats = limits_statistics(samples)
        for key in samples.input_features:
            if key.startswith("observation.images."):
                stats[key]["mean"] = [[[v]] for v in IMAGENET_MEAN]
                stats[key]["std"] = [[[v]] for v in IMAGENET_STD]
        return stats

    def loss_counts(self, batch):
        # The final action row is only the endpoint validity marker. It never
        # enters a transition action block or the predictor's input.
        pad = batch["action_is_pad"]
        # sample_spec fixes the block/history; options() is called before counts.
        block = getattr(self, "block", 5)
        history = (pad.shape[1] - 1) // block
        valid = ~pad[:, :-1].reshape(len(pad), history, block).any(-1)
        valid = valid & ~pad[:, block::block]
        return {"_prediction_weight": int(valid.sum()), "_sigreg_weight": len(pad)}

    def build(self, options, stats=None):
        from .modeling_lewm import LeWMPolicy

        self.block = options.get("action_block", 5)
        return LeWMPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        from .processor_lewm import LeWMProcessor

        return LeWMProcessor(model.config, stats, device)

    def metric_specs(self, config):
        self.block = config.action_block
        return [
            MetricSpec(
                "prediction_loss",
                "prediction_loss",
                "valid_transitions",
                loss_count("_prediction_weight"),
                console="pred",
            ),
            MetricSpec(
                "sigreg_loss",
                "sigreg_loss",
                "samples",
                loss_count("_sigreg_weight"),
                console="sigreg",
            ),
        ]

    @staticmethod
    def supports_action_prediction(config):
        # The trainer cannot choose a user's task goal; imitation MAE is not a
        # world-model control benchmark. Validation still reports both losses.
        return False

    def initialize(self, model, settings):
        if settings.get("pretrained_path"):
            from pathlib import Path

            import torch

            path = Path(settings["pretrained_path"]).expanduser()
            weights = torch.load(path / "weights.pt", map_location="cpu", weights_only=True)
            model.world.load_state_dict(weights, strict=True)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("LeWM does not use ACT temporal ensembling")
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        return result

    def save(self, path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        save_weights(path / "model.safetensors", model, model_state)
