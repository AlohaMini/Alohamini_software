"""OpenWAM-alpha registry adapter; all training infrastructure stays shared."""

from copy import deepcopy
from pathlib import Path

from alohamini.paths import WorkspacePaths
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_openwam import OpenWAMConfig


class OpenWAMAlgorithm(ChunkAlgorithm):
    supports_field_transforms = True
    loss_terms = ("loss_video", "loss_action")
    config_class = OpenWAMConfig
    # The joint driver calls stream projections directly, bypassing DiTBlock.forward.
    fsdp_classes = ("RMSNorm", "T5LayerNorm")

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        semantic = cfg.get("model", {}).get("action_representation", "recorded") == "alpha80"
        if semantic:
            from .semantic import ROTATION

            data = cfg.setdefault("data", {})
            for name, value in dict(
                state_representation="tcp",
                action_representation="tcp_absolute",
                rotation_representation=ROTATION,
            ).items():
                if name in data and data[name] != value:
                    raise ValueError(f"OpenWAM alpha80 requires data.{name}={value}")
                data[name] = value
        elif any(
            cfg.get("data", {}).get(name, "recorded") != "recorded"
            for name in ("state_representation", "action_representation")
        ):
            raise ValueError("OpenWAM TCP fields require policy.action_representation=alpha80")
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
        if semantic:
            cfg["paper_preset"]["name"] = "openwam-alpha-semantic80-v1"
            cfg["paper_preset"]["adaptations"].update(
                coordinates=(
                    "calibrated absolute FK; official left 0:10, right 34:44; "
                    "rotation matrix columns; declared arm frames; "
                    "platform extension x/y/yaw velocity and lift in reserved slots 68:72"
                ),
                heads="retain all official 80D action input/output and proprio projections",
                normalization=(
                    "train min/max with input clipping; rotation6d identity; "
                    "gripper closed=-1/open=+1; exact constant labels restored on decode"
                ),
                masking="unused action slots excluded from loss; analytic noise path at inference",
            )
        return cfg

    def prepare_options(self, options, samples):
        if options.get("action_representation", "recorded") != "alpha80":
            return
        from .semantic import make_layout

        transform = getattr(samples, "field_transform", None)
        if transform is None:
            raise ValueError("alpha80 requires calibrated TCP data fields")
        layout = make_layout(transform, samples.model_features)
        if options.get("alpha80_layout") and options["alpha80_layout"] != layout:
            raise ValueError("alpha80 checkpoint layout differs from the selected data fields")
        options["alpha80_layout"] = layout

    def statistics(self, samples, options=None):
        stats = super().statistics(samples, options)
        if (options or {}).get("action_representation") == "alpha80":
            from .semantic import semantic_statistics

            return semantic_statistics(stats, options["alpha80_layout"])
        return stats

    def processor(self, model, stats, device):
        if model.config.action_representation == "alpha80":
            from .semantic import Alpha80Processor

            return Alpha80Processor(model.config, stats, device)
        return super().processor(model, stats, device)

    def save(self, path, model, manifest, model_state=None):
        super().save(path, model, manifest, model_state)
        if model.config.action_representation == "alpha80":
            manifest.setdefault("field_transforms", []).append(
                dict(
                    operation="openwam_alpha80",
                    layout=deepcopy(model.config.alpha80_layout),
                    model_width=80,
                    inactive_action_loss="masked",
                    normalization="train_min_max_rotation_identity_gripper_endpoints",
                    constant_action_decode="training_constant",
                )
            )

    def validate_statistics(self, stats, expected, names, *, manifest=None):
        super().validate_statistics(stats, expected, names, manifest=manifest)
        config = (manifest or {}).get("config", {})
        if config.get("action_representation") != "alpha80":
            return
        from .semantic import make_layout, validate_layout, validate_limits

        layout = config["alpha80_layout"]
        validate_layout(
            layout, state_width=expected["observation.state"][1][0], action_width=len(names)
        )
        if layout["features"]["action"]["names"] != list(names):
            raise ValueError("alpha80 action names differ from the data contract")
        validate_limits(stats, layout)
        if "format" in manifest:
            from alohamini.datasets.field_transforms import FieldTransforms

            if not manifest.get("data_fields"):
                raise ValueError("alpha80 checkpoint is missing frozen FK/IK data fields")
            fields = FieldTransforms(manifest["data_fields"])
            features = fields.features(
                manifest["state_feature"],
                manifest["state_units"],
                manifest["source_info"]["features"]["action"],
            )
            if make_layout(fields, features) != layout:
                raise ValueError("alpha80 layout differs from frozen FK/IK data fields")

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return options
        source = Path(settings.get("pretrained_path") or WorkspacePaths().pretrained / "openwam")
        source = source.expanduser()
        if (source / "policy.json").is_file():
            from alohamini.learning.checkpoint import read_checkpoint

            _, manifest = read_checkpoint(source)
            if manifest["config"].get("action_representation", "recorded") != options.get(
                "action_representation", "recorded"
            ):
                raise ValueError(
                    "Cannot switch an OpenWAM checkpoint between recorded and alpha80; "
                    "start alpha80 fine-tuning from the official 80D base"
                )
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
        if model.config.action_representation == "alpha80":
            # Same semantic boundary: retain the pretrained I/O and state projections.
            model.network.load_state_dict(state, strict=True)
            return
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
