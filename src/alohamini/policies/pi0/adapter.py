"""Native Pi0 uses the platform reader, trainer, weight storage and checkpoint paths."""

from copy import deepcopy
from pathlib import Path

from alohamini.policies.pi05.adapter import PI05Algorithm, PI05TrainModel

from .configuration_pi0 import PI0Config, PI0TrainingConfig
from .processor_pi0 import PI0BatchProcessor, PI0Processor


class PI0TrainModel(PI05TrainModel):
    name = "pi0"
    core_config_class = PI0Config


class PI0Algorithm(PI05Algorithm):
    asset_fields = ("tokenizer_path",)
    config_class = PI0TrainingConfig

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
            name="openpi-pi0-alohamini-v1",
            repository="https://github.com/Physical-Intelligence/openpi",
            revision="15a9616a00943ada6c20a0f158e3adb39df2ccac",
            images="RGB; aspect-preserving 224x224 padding; [-1,1]",
            state="named 18D recorded coordinates, continuous action-expert token",
            action="arm joints relative to current state; absolute gripper/base/lift",
            language="PaliGemma prompt + newline, 48 tokens; no discretized state in text",
            normalization="training-window mean/std; additive epsilon 1e-6; no clipping",
            time="50 future row offsets; repeated episode tails retained in padded-width flow loss",
            architecture="PaliGemma 2B and Gemma 300M, 18 layers each; no timestep adaRMSNorm",
            adaptations=[
                "native platform reader/trainer/checkpoint; named AlohaMini coordinates",
                "reuse the existing native OpenPI dual-stream model with pi05=false",
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
        if result.get("tokenizer_path"):
            result["tokenizer_path"] = str(Path(result["tokenizer_path"]).expanduser().resolve())
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError("Pi0 requires --policy.path to a local OpenPI PyTorch base")
        result.setdefault("n_action_steps", result.get("network", {}).get("action_horizon", 50))
        return result

    def build(self, options, stats=None):
        return PI0TrainModel(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        return PI0BatchProcessor(model.config, stats, device)

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        if manifest and (manifest.get("state_feature") or {}).get("names") != names:
            raise ValueError("Pi0 requires named state coordinates matching actions")
        PI0Processor(stats, tokenizer=None)
