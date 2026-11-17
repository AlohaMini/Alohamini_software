"""DM0.5 assembly; platform owns readers, optimization, statistics and checkpoints."""

import json
from copy import deepcopy
from pathlib import Path

from alohamini.learning.processor import quantile_statistics
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_dm05 import DM05Config


class DM05Algorithm(ChunkAlgorithm):
    asset_fields = ("assets_path",)
    config_class = DM05Config
    supports_field_transforms = True
    fsdp_classes = (
        "PatchedGemma3DecoderLayer",
        "DM05ActionExpert",
        "SiglipEncoderLayer",
        "Embedding",
    )
    loss_terms = ("flow_loss",)

    @staticmethod
    def apply_preset(settings):
        cfg = deepcopy(settings)
        for key, value in dict(
            state="auto",
            batch_size=8,
            mixed_precision="none",
            steps=100000,
            save_freq=5000,
            ema={"enable": False},
            return_uint8=True,
        ).items():
            cfg.setdefault(key, value)
        if "image_size" not in cfg and cfg.get("dataset"):
            from alohamini.learning.processor import source_image_size

            cfg["image_size"] = source_image_size(cfg)
        cfg["paper_preset"] = dict(
            name="opendm-dm05-recorded-v1",
            repository="https://github.com/Dexmal/OpenDM",
            revision="03f799120d425ead51d9482bb1da950b6dabade9",
            images="ordered RGB, centered square padding then local Gemma3 image processor",
            language="official task/speed/camera labels; discretized actual state before padding",
            state="named recorded fields, q01/q99 +1e-6; state bins saturate to256 bins",
            action="recorded targets, actual dimensions masked inside32D expert; tails excluded",
            normalization=(
                "recorded preserves outliers and zero-quantile coordinates; "
                "opendm explicitly clips and zeroes"
            ),
            architecture="saved Gemma3/VLM and expert depths must match; released base34/34 layers",
            precision="saved dtype, eager text and SDPA vision/action; platform compute policy",
            adaptations=[
                "native dataset/trainer/checkpoint; exact train-only quantiles",
                "recorded normalization preserves rare valid robot actions",
                "optional FlashAttention/Liger/compiled suffix are not enabled by default",
            ],
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides",
                    {key: value for key, value in settings.items() if key != "paper_preset"},
                )
            ),
        )
        return cfg

    @staticmethod
    def options(settings, device):
        result = deepcopy(settings.get("model", {}))
        if not settings.get("resume"):
            path = Path(settings.get("pretrained_path") or ".").expanduser()
            if (path / "config.json").is_file() and not (path / "policy.json").is_file():
                base = json.loads((path / "config.json").read_text())
                result.setdefault("core_config", base.get("core_config", base))
                result.setdefault("assets_path", str(path))
            result["assets_path"] = str(local_assets(result.get("assets_path", "")))
        return result

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 50)))},
            include_task=True,
            exclude_fixed_state=False,
        )

    @staticmethod
    def statistics(samples, options=None):
        return quantile_statistics(samples)

    def build(self, options, stats=None):
        from .modeling_dm05 import DM05Policy

        return DM05Policy(self.config_class(**deepcopy(options)))

    @staticmethod
    def processor(model, stats, device):
        from .processor import DM05Processor

        return DM05Processor(model.config, stats, device)

    @staticmethod
    def initialize(model, settings):
        from alohamini.learning.weights import load_weights

        # Raw OpenDM checkpoints carry core keys; native checkpoints use ChunkAlgorithm.load.
        current = "model.vlm.model.vision_tower."
        legacy = current + "vision_model."
        mapping = {
            legacy + name.removeprefix(current): name
            for name in model.model.state_dict()
            if name.startswith(current)
        }
        load_weights(model.model, settings["pretrained_path"], key_mapping=mapping)
