"""FAST uses the platform data reader, trainer and native safetensors checkpoints."""

from copy import deepcopy
from pathlib import Path

from alohamini.policies.pi05.adapter import PI05Algorithm
from alohamini.policies.pi05.processor_pi05 import PI05Processor

from .configuration_pi0_fast import PI0FastConfig


class PI0FastAlgorithm(PI05Algorithm):
    asset_fields = ("tokenizer_path", "fast_tokenizer_path")
    from alohamini.trajectory_execution import PlanCapabilities

    config_class = PI0FastConfig
    plan_capabilities = PlanCapabilities()
    prediction_arguments = None
    fsdp_classes = ("GemmaDecoderLayer", "SiglipEncoderLayer", "Embedding")

    @staticmethod
    def action_horizon(config):
        return config.chunk_size

    @staticmethod
    def supports_action_prediction(config):
        return config.evaluate_action_predictions

    @staticmethod
    def apply_preset(settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            steps=30000,
            batch_size=32,
            save_freq=1000,
            state="auto",
            mixed_precision="none",
            ema={"enable": False},
        ).items():
            cfg.setdefault(key, value)
        cfg["paper_preset"] = dict(
            name="openpi-pi0-fast-alohamini-v1",
            repository="https://github.com/Physical-Intelligence/openpi",
            revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
            images="ordered RGB, aspect-preserving 224x224 padding, [-1,1]",
            state="18 named recorded coordinates, quantiles, 256-bin state text, no zero padding",
            action="12 relative arm joints; absolute gripper/base/lift; quantiles; DCT/BPE FAST",
            normalization="window q01/q99 plus 1e-6, no clipping",
            language="lowercase task/state prefix; Action: + mapped FAST codes + | and EOS",
            time="50 future row offsets, repeated tails included in codec",
            architecture="PaliGemma 2B, 18 layers; no separate flow action expert",
            adaptations=["named AlohaMini coordinates; strict malformed-action rejection"],
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
        for name in ("tokenizer_path", "fast_tokenizer_path"):
            if not result.get(name):
                raise ValueError(f"FAST requires local {name}")
            result[name] = str(Path(result[name]).expanduser().resolve())
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError("FAST requires --policy.path to a converted local PyTorch base")
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            exclude_fixed_state=False,
            include_task=True,
            delta_indices={"action": list(range(options.get("chunk_size", 50)))},
        )

    def build(self, options, stats=None):
        from .modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import PI0FastProcessor

        return PI0FastProcessor(model.config, stats, device)

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        return [
            MetricSpec(
                "loss_fast",
                "fast_loss",
                "samples_with_token_mean",
                loss_count("_loss_weight"),
                console="fast",
            )
        ]

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        load_weights(
            model.network, settings["pretrained_path"], prefix=model.config.base_weights_prefix
        )

    def initialize_checkpoint(self, path, model, manifest):
        self.load(path, model)

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        if manifest and (manifest.get("state_feature") or {}).get("names") != names:
            raise ValueError("FAST state coordinates must match named action coordinates")
        PI05Processor(stats, tokenizer=None, action_dim=18)
