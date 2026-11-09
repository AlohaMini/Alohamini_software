"""Native LingBot registry, dataset protocol, initialization and checkpoints."""

import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import numpy as np

from alohamini.datasets.statistics import ExactQuantileStats
from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.policies.chunk import ChunkAlgorithm, local_assets, local_bpe_assets

from .configuration import LingBotConfig
from .semantic import JOINTS, configure, relative_actions, validate_layout


class LingBotAlgorithm(ChunkAlgorithm):
    config_class = LingBotConfig
    # Dual-stream attention directly calls decoder projections; wrapping decoder
    # parents does not gather them. These leaf module forwards are actually invoked.
    fsdp_classes = (
        "Qwen3VLVisionBlock",
        "Qwen2FusedExperts",
        "Qwen2RMSNorm",
        "FixQwen2RMSNorm",
        "AdaRMSNorm",
    )

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        cfg.setdefault("state", "auto")
        cfg.setdefault("cameras", ["forward", "wrist_left", "wrist_right"])
        cfg.setdefault("image_size", [256, 256])
        cfg.setdefault("mixed_precision", "bfloat16")
        cfg.setdefault("return_uint8", True)
        cfg.setdefault("ema", {"enable": False})
        cfg["paper_preset"] = dict(
            name="lingbot-vla-v2-alohamini-v1",
            source="https://github.com/Robbyant/lingbot-vla-v2",
            revision="be969b8fd117fb70550c5d4bf4bc328211b5b1b6",
            protocol={
                "images": "global,left wrist,right wrist; 256x256; Qwen3-VL 65536 pixels",
                "language": "Qwen chat user turn, 72 tokens",
                "time": "50 actions, current/future teacher frames",
                "actions": (
                    "joint position relative to current state; absolute gripp"
                    "ers and base velocities"
                ),
                "normalization": (
                    "train split mean/std after relative conversion, before 55D packing"
                ),
                "slots": [list(x) for x in JOINTS],
            },
            adaptations={
                "optimizer": "platform AdamW with official MoE expert LR multiplier; not Muon",
                "lift": cfg.get("model", {}).get("lift_mode", "hold"),
            },
            overrides=deepcopy(
                settings.get("paper_preset", {}).get(
                    "overrides", {k: v for k, v in settings.items() if k != "paper_preset"}
                )
            ),
        )
        return cfg

    def options(self, settings, device):
        result = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return result
        source = Path(settings.get("pretrained_path") or result.get("assets_path", "")).expanduser()
        if (source / "policy.json").is_file():
            return result
        if not (source / "config.json").is_file():
            raise FileNotFoundError(
                "LingBot --policy.path must contain the released LeRobot "
                "base config.json and model.safetensors"
            )
        base = json.loads((source / "config.json").read_text())
        allowed = {f.name for f in fields(LingBotConfig)} - {
            "input_features",
            "output_features",
            "state_slots",
            "action_slots",
            "relative_exclude_joints",
            "action_feature_names",
            "layout",
            "canonical_joints",
            "dtype",
            "use_relative_actions",
        }
        result = {**{k: v for k, v in base.items() if k in allowed}, **result}
        configured = Path(result.get("tokenizer_path", "")).expanduser()
        backbone = configured if (configured / "config.json").is_file() else source / configured
        if not (backbone / "config.json").is_file():
            backbone = source / "assets"
        result["tokenizer_path"] = str(local_assets(backbone))
        text = result.get("text_tokenizer_path") or (
            source / "chat_template_tokenizer"
            if (source / "chat_template_tokenizer").is_dir()
            else backbone
        )
        image = result.get("image_processor_path") or (
            source / "image_processor" if (source / "image_processor").is_dir() else backbone
        )
        result["text_tokenizer_path"] = str(local_bpe_assets(text))
        result["image_processor_path"] = str(local_assets(image))
        result["canonical_joints"] = dict(JOINTS)
        result.setdefault("use_relative_actions", True)
        if result.get("align_params"):
            root = Path(result.get("teacher_root") or source).expanduser().resolve()
            result["teacher_root"] = str(root)
            for section, key, relative in (
                ("depth", "moge_path", "moge/model.pt"),
                ("depth", "morgbd_path", "depth/model.pt"),
                ("video", "ckpt_path", "dino_video/teacher_step_10000.pth"),
                ("video", "config_path", "dino_video/config.yaml"),
            ):
                if section in result["align_params"]:
                    path = root / relative
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    result["align_params"][section][key] = str(path)
        return result

    prepare_options = staticmethod(configure)

    def sample_spec(self, options, *, cameras=()):
        if len(cameras) != 3:
            raise ValueError("LingBot requires ordered global, left wrist and right wrist cameras")
        chunk = options.get("chunk_size", 50)
        delta = {"action": list(range(chunk))}
        if options.get("align_params"):
            delta.update({f"observation.images.{c}": [0, chunk - 1] for c in cameras})
        return dict(delta_indices=delta, include_task=True, exclude_fixed_state=False)

    def inference_spec(self, options, *, cameras=()):
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 50)))},
            include_task=True,
            exclude_fixed_state=False,
        )

    evaluation_sample_spec = inference_spec

    def statistics(self, samples, options=None):
        stats = samples.statistics(keys=("observation.state",))
        tracker = ExactQuantileStats()
        for i, row in enumerate(samples.sample_indices):
            state = samples.selection.frame(samples.rows[row])
            entry = samples.action_metadata(i)
            values = relative_actions(state, entry["action"].numpy(), options["layout"])
            tracker.update(values[~entry["action_is_pad"].numpy()])
        stats["action"] = {k: v.tolist() for k, v in tracker.get_statistics().items()}
        return stats

    def validate_statistics(self, stats, expected, names, *, manifest=None):
        for key in ("observation.state", "action"):
            for field in ("mean", "std"):
                value = np.asarray(stats[key][field])
                if (
                    value.shape != (18,)
                    or not np.isfinite(value).all()
                    or (field == "std" and (value < 0).any())
                ):
                    raise ValueError("Invalid LingBot named-coordinate statistics")
        if manifest and manifest.get("config", {}).get("layout"):
            layout = manifest["config"]["layout"]
            validate_layout(layout)
            if layout["names"] != list(names):
                raise ValueError("LingBot checkpoint field names differ")

    @staticmethod
    def loss_counts(batch):
        return {
            "_loss_weight": int((~batch["action_is_pad"]).sum()),
            "_aux_loss_weight": len(batch["action"]),
        }

    def metric_specs(self, config):
        specs = [
            MetricSpec(
                "flow_loss",
                "flow_loss",
                "valid_actions",
                loss_count("_loss_weight", sum(s is not None for s in config.layout["slots"])),
                console="flow",
            )
        ]
        names = ["seq_wise_loss", "router_z_loss"]
        if config.align_params:
            names += ["depth_loss"]
            if config.align_params["depth"].get("use_future_depth"):
                names += ["future_depth_loss"]
            if config.align_params.get("use_future_video"):
                names += ["future_video_loss"]
                if config.align_params.get("video", {}).get("use_current_patch_loss"):
                    names += ["current_video_loss"]
        return specs + [MetricSpec(n, n, "samples", loss_count("_aux_loss_weight")) for n in names]

    def build(self, options, stats=None):
        from .modeling import LingBotPolicy

        return LingBotPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        from .processor import LingBotProcessor

        return LingBotProcessor(model.config, stats, device)

    def initialize(self, model, settings):
        from safetensors.torch import load_model

        load_model(
            model, Path(settings["pretrained_path"]).expanduser() / "model.safetensors", strict=True
        )

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        result = deepcopy(options)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("LingBot does not use ACT temporal ensembling")
        result["tokenizer_path"] = str(path / "assets")
        result["text_tokenizer_path"] = result["image_processor_path"] = result["tokenizer_path"]
        if n_action_steps is not None:
            result["n_action_steps"] = n_action_steps
        return result

    def save(self, path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        model.save_assets(path / "assets")
        manifest["config"]["tokenizer_path"] = "assets"
        manifest["config"]["text_tokenizer_path"] = "assets"
        manifest["config"]["image_processor_path"] = "assets"
        manifest.setdefault("field_transforms", []).append(
            dict(operation="lingbot_slots", layout=deepcopy(model.config.layout))
        )
        save_weights(path / "model.safetensors", model, model_state)
