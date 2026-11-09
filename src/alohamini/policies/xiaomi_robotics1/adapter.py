"""XR1 data protocol and model integration under the platform trainer."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from alohamini.datasets.statistics import ExactQuantileStats
from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.policies.chunk import ChunkAlgorithm, local_bpe_assets

from .configuration import XiaomiConfig
from .semantic import XiaomiFields


class XiaomiAlgorithm(ChunkAlgorithm):
    config_class = XiaomiConfig
    supports_field_transforms = True
    fsdp_classes = ("Qwen3VLVisionBlock", "Qwen3VLTextDecoderLayer", "DecoderLayer")

    def apply_preset(self, settings):
        cfg = deepcopy(settings)
        data = cfg.setdefault("data", {})
        for key, value in dict(
            state_representation="recorded",
            action_representation="tcp_delta",
            rotation_representation="rotvec",
            tcp_delta_frame="tool",
        ).items():
            if key in data and data[key] != value:
                raise ValueError(f"Xiaomi requires data.{key}={value}")
            data[key] = value
        cfg.setdefault("state", "auto")
        cfg.setdefault("cameras", ["forward", "wrist_left", "wrist_right"])
        cfg.setdefault("mixed_precision", "bfloat16")
        cfg.setdefault("return_uint8", True)
        cfg.setdefault("ema", {"enable": False})
        cfg["paper_preset"] = dict(
            name="xiaomi-robotics1-alohamini-v1",
            source="https://github.com/XiaomiRobotics/Xiaomi-Robotics-1",
            revision="0dd7aef8dc87296246aae812a1f59ccb708e5546",
            protocol={
                "images": (
                    "ego,left wrist,right wrist; resize to multiples of 32, max 160000 pixels"
                ),
                "language": (
                    "official multi-view /no_cot prompt; auxiliary "
                    "state/action/score turns only for losses"
                ),
                "state": "60D, joint radians plus calibrated grippers; separate state/action slots",
                "action": (
                    "30x60; tool-frame TCP delta, axis-angle radians, gripper delta, base velocity"
                ),
                "normalization": (
                    "action mean/std per horizon step; state q01/q99; train split only"
                ),
                "loss": "flow MSE, frequency, action choice and score with separate denominators",
                "training": "official synchronous mode; noise repeat 4; flow integration 5 steps",
            },
            adaptations={
                "calibration": "saved platform FK/IK descriptor",
                "lift": "hold observed height; waist slot inactive",
                "base": "platform SI velocity convention; yaw converted deg/s to rad/s",
                "execution": (
                    "platform synchronous chunks; native PyTorch SDPA/RMSNorm"
                    " in place of FlashAttention/Liger"
                ),
                "constant_actions": "exact constant restoration",
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
        from transformers import Qwen3VLConfig

        source = Path(settings.get("pretrained_path") or "").expanduser()
        path = local_bpe_assets(result.get("assets_path") or source / "assets")
        result["assets_path"] = str(path)
        result.setdefault(
            "vlm_config", Qwen3VLConfig.from_pretrained(path, local_files_only=True).to_dict()
        )
        return result

    def prepare_options(self, options, samples):
        fields = getattr(samples, "field_transform", None)
        if fields is None or samples.selection.feature["names"] != fields.geometry.names:
            raise ValueError("Xiaomi requires full calibrated joint observations")
        saved = fields.geometry.descriptor
        if options.get("kinematic_fields") and options["kinematic_fields"] != saved:
            raise ValueError("Xiaomi checkpoint kinematics differ from selected data")
        XiaomiFields(saved)
        options["kinematic_fields"] = deepcopy(saved)

    def sample_spec(self, options, *, cameras=()):
        if len(cameras) != 3:
            raise ValueError("Xiaomi requires ordered ego, left wrist and right wrist cameras")
        return dict(
            delta_indices={"action": list(range(options.get("chunk_size", 30)))},
            include_task=True,
            exclude_fixed_state=False,
        )

    inference_spec = sample_spec

    def statistics(self, samples, options=None):
        codec = XiaomiFields(options["kinematic_fields"])
        chunk = options.get("chunk_size", 30)
        states = ExactQuantileStats()
        actions = [ExactQuantileStats() for _ in range(chunk)]
        seen = [False] * chunk
        for index, row in enumerate(samples.sample_indices):
            raw = samples.field_transform.context(samples.rows[row])
            states.update(codec.state(raw)[None])
            entry = samples.action_metadata(index)
            values = codec.action(raw, entry["action"].numpy())
            for step, valid in enumerate(~entry["action_is_pad"].numpy()):
                if valid:
                    actions[step].update(values[step : step + 1])
                    seen[step] = True
        qs = states.get_statistics()
        fitted = [
            t.get_statistics() if used else {"mean": np.zeros(60), "std": np.zeros(60)}
            for t, used in zip(actions, seen, strict=True)
        ]
        # Ordinary feature statistics remain available for shared validation/UI.
        stats = samples.statistics(keys=("observation.state", "action"))
        stats["xiaomi"] = {
            "state": {k: qs[k][None].tolist() for k in ("q01", "q99")},
            "action": {k: np.stack([s[k] for s in fitted]).tolist() for k in ("mean", "std")},
        }
        return stats

    def validate_statistics(self, stats, expected, names, *, manifest=None):
        data = stats.get("xiaomi", {})
        if set(data) != {"state", "action"}:
            raise ValueError("Xiaomi requires packed state and per-step action statistics")
        mean, std = (np.asarray(data["action"][k]) for k in ("mean", "std"))
        low, high = (np.asarray(data["state"][k]) for k in ("q01", "q99"))
        if (
            mean.ndim != 2
            or mean.shape[-1] != 60
            or std.shape != mean.shape
            or low.shape != (1, 60)
            or high.shape != low.shape
        ):
            raise ValueError("Invalid Xiaomi statistic dimensions")
        if (
            not all(np.isfinite(v).all() for v in (mean, std, low, high))
            or (std < 0).any()
            or (high < low).any()
        ):
            raise ValueError("Invalid Xiaomi statistic values")
        config = (manifest or {}).get("config", {})
        if config and mean.shape[0] != config["chunk_size"]:
            raise ValueError("Xiaomi horizon differs from saved statistics")
        if config.get("kinematic_fields"):
            codec = XiaomiFields(config["kinematic_fields"])
            relative = codec.geometry.relative_actions(reference_frame="tool")
            if relative.feature(codec.geometry.feature())["names"] != list(names):
                raise ValueError("Xiaomi TCP field names differ from saved kinematics")

    @staticmethod
    def loss_counts(batch):
        valid = ~batch["action_is_pad"]
        return {
            "_loss_weight": int(valid.sum()),
            "_freq_weight": int(valid.all(-1).sum()),
            "_aux_loss_weight": len(valid),
        }

    def metric_specs(self, config):
        count = len(XiaomiFields(config.kinematic_fields).action_slots)
        return [
            MetricSpec(
                "loss_mse",
                "loss_mse",
                "valid_coordinates",
                loss_count("_loss_weight", count),
                console="mse",
            ),
            MetricSpec("loss_freq", "loss_freq", "complete_chunks", loss_count("_freq_weight")),
            MetricSpec("loss_l1", "loss_l1", "samples", loss_count("_aux_loss_weight")),
            MetricSpec("loss_score", "loss_score", "samples", loss_count("_aux_loss_weight")),
        ]

    def build(self, options, stats=None):
        from .modeling import XiaomiPolicy

        return XiaomiPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        from .processor import XiaomiProcessor

        return XiaomiProcessor(model.config, stats, device)

    def initialize(self, model, settings):
        import torch

        path = Path(settings["pretrained_path"]).expanduser()
        if path.is_dir():
            path = path / "model_states.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("state_dict", state.get("module", state))
        # Released runner export names the policy model; normalize only a uniform prefix.
        for prefix in ("model.", "module."):
            if state and all(k.startswith(prefix) for k in state):
                state = {k.removeprefix(prefix): v for k, v in state.items()}
        model.network.load_state_dict(state, strict=True)
