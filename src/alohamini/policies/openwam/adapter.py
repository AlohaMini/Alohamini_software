"""OpenWAM-alpha registry adapter; all training infrastructure stays shared."""

from copy import deepcopy
from pathlib import Path

from alohamini.paths import WorkspacePaths
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_openwam import OpenWAMConfig


class OpenWAMAlgorithm(ChunkAlgorithm):
    loss_terms = ("loss_video", "loss_action")
    config_class = OpenWAMConfig
    # The joint driver calls stream projections directly, bypassing DiTBlock.forward.
    fsdp_classes = ("RMSNorm", "T5LayerNorm")

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        cfg.setdefault("state", "auto")
        cfg.setdefault("image_size", [256, 320])
        cfg.setdefault("mixed_precision", "bfloat16")
        cfg.setdefault("batch_size", 16)
        cfg.setdefault("steps", 30000)
        cfg["scheduler"] = {
            "type": "diffusers_cosine",
            "warmup_steps": int(cfg["steps"] * 0.05),
            "decay_steps": cfg["steps"],
            "decay_lr": 1e-6,
            **(cfg.get("scheduler") or {}),
        }
        cfg["paper_preset"] = {
            "name": "openwam-alpha-recorded-v1",
            "paper": "https://arxiv.org/abs/2609.07398v1",
            "source": "https://github.com/OpenWAM-Official/OpenWAM",
            "revision": "3dfc9a7d9230f0e1382e6b22639278f1a86e5eb5",
            "reference_recipe": {
                "model": "Wan2.2-TI2V-5B + ActionDiT 1024/4096, 30 paired layers, 24x128 attention",
                "images": "L canvas 384x320: head 256x320; wrists each 128x160; pixels [-1,1]",
                "time": "actions 32 rows; video offsets 0,4,...,32; causal VAE temporal factor 4",
                "state_action": "80D explicit slots: xyz(m), rotation6d, gripper, hand, reserved",
                "normalization": (
                    "RoboTwin min/max; mixed pretraining uses source-specific quantiles"
                ),
                "language": "frozen umT5-XXL, 512 tokens, whitespace cleaning",
                "flow": "independent training times, shift 5; joint 10-step inference",
                "training": "AdamW lr=1e-4, betas=[0.9,0.95], wd=.01, clip=1; 5% warmup",
            },
            "adaptations": {
                "coordinates": (
                    "recorded state/action names and units; no inferred 80D semantic mapping"
                ),
                "heads": "reinitialize action input/output and proprio projections on base import",
                "normalization": "training split limits in recorded coordinates",
                "cameras": "ordered selected views or explicit camera_layout, maximum three",
                "schedule": "explicit step budget replaces dataset-dependent epochs",
            },
            "overrides": deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        }
        return cfg

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return options
        source = Path(settings.get("pretrained_path") or WorkspacePaths().pretrained / "openwam")
        options["assets_path"] = str(
            local_assets(options.get("assets_path") or source / "tokenizer/google/umt5-xxl")
        )
        return options

    def sample_spec(self, options, *, cameras=()):
        if not cameras:
            raise ValueError("OpenWAM requires cameras")
        horizon, stride = options.get("chunk_size", 32), options.get("video_stride", 4)
        return dict(
            delta_indices={
                "action": list(range(horizon)),
                **{
                    f"observation.images.{camera}": list(range(0, horizon + 1, stride))
                    for camera in cameras
                },
            },
            include_task=True,
        )

    def inference_spec(self, options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 32)))}, include_task=True
        )

    def build(self, options, stats=None):
        from .modeling_openwam import OpenWAMPolicy

        return OpenWAMPolicy(self.config_class(**deepcopy(options)))

    def initialize(self, model, settings):
        from safetensors.torch import load_file

        path = Path(
            settings.get("pretrained_path") or WorkspacePaths().pretrained / "openwam"
        ).expanduser()
        files = list(path.glob("checkpoint_step_*.safetensors"))
        if len(files) != 1:
            raise ValueError(
                "OpenWAM base directory must contain exactly one checkpoint_step_*.safetensors"
            )
        state = load_file(files[0])
        # The native recorded-coordinate head is a different contract even at width 80.
        heads = (
            "action_backbone.action_encoder.",
            "action_backbone.action_decoder.",
            "proprio_encoder.",
        )
        selected = {k: v for k, v in state.items() if not k.startswith(heads)}
        own = model.network.state_dict()
        missing = own.keys() - selected.keys()
        unexpected = selected.keys() - own.keys()
        if unexpected or any(not k.startswith(heads) for k in missing):
            raise ValueError(
                f"OpenWAM base mismatch: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        model.network.load_state_dict(selected, strict=False)
