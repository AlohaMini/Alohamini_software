"""AC-DiT recipes, named coordinate binding, staged initialization and export."""

import math
from copy import deepcopy

from safetensors.torch import load_file

from alohamini.learning.checkpoint import read_checkpoint
from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import validate_statistics
from alohamini.policies.chunk import ChunkAlgorithm

from .configuration_acdit import ACDiTConfig
from .preset import apply_preset
from .processor_acdit import ACDiTProcessor


class ACDiTAlgorithm(ChunkAlgorithm):
    config_class = ACDiTConfig
    apply_preset = staticmethod(apply_preset)
    supports_field_transforms = True
    supports_remote_chunks = False
    fsdp_classes = ("RDTBlock", "SiglipEncoderLayer", "LoRALinear", "Linear1Layer", "Linear2Layer")

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        self.mask_loss = options.get("mask_padding_loss", False)
        if options.get("stage", "full") == "full" and not settings.get("resume"):
            if not settings.get("pretrained_path"):
                raise ValueError(
                    "Train AC-DiT stage=mobility first; initialize stage=full with "
                    "--policy.path=<mobility checkpoint>"
                )
            _, source = read_checkpoint(settings["pretrained_path"])
            if source["kind"] != "acdit":
                raise ValueError("AC-DiT initialization requires a native AC-DiT checkpoint")
            if settings.get("normalization") != "checkpoint":
                raise ValueError(
                    "Full-stage initialization must retain mobility normalization; use "
                    "--normalization=checkpoint"
                )
        return options

    @staticmethod
    def sample_spec(options, *, cameras=()):
        history = list(range(1 - options.get("n_obs_steps", 2), 1))
        windows = {"observation.state": [0], "action": list(range(options.get("chunk_size", 2)))}
        windows.update({f"observation.images.{c}": history for c in cameras})
        if options.get("use_pointcloud", True):
            windows[options.get("pointcloud_field", "observation.point_cloud")] = history
        windows.update({key: [0] for key in options.get("context_fields", ())})
        return {"delta_indices": windows, "include_task": True}

    @classmethod
    def live_history_spec(cls, options, *, cameras=()):
        return {
            key: offsets
            for key, offsets in cls.sample_spec(options, cameras=cameras)["delta_indices"].items()
            if key != "action"
        }

    @staticmethod
    def prepare_options(options, samples):
        if samples.selection is None:
            raise ValueError("AC-DiT requires recorded numeric state")
        state_names = samples.input_features["observation.state"].shape[0]
        model_features = getattr(samples, "model_features", {})
        names = model_features.get("observation.state", samples.selection.feature)["names"]
        action_names = model_features.get("action", samples.info["features"]["action"]).get("names")
        if not action_names or state_names != len(names):
            raise ValueError("AC-DiT needs declared state/action field names")
        if (options.get("mobility_pretrained_path") or options.get("body_pretrained_path")) and (
            not options.get("state_slot_map") or not options.get("action_slot_map")
        ):
            raise ValueError(
                "RDT weight initialization requires explicit semantic "
                "state_slot_map/action_slot_map"
            )
        for key, values in (("state_names", names), ("action_names", action_names)):
            if options.get(key) and list(options[key]) != list(values):
                raise ValueError(f"AC-DiT {key} differs from the recorded field order")
            options[key] = list(values)
        options.setdefault("state_slot_map", {name: i for i, name in enumerate(names)})
        options.setdefault("action_slot_map", {name: i for i, name in enumerate(action_names)})
        fps = samples.info["fps"]
        if "control_frequency" in options and not math.isclose(options["control_frequency"], fps):
            raise ValueError("AC-DiT control_frequency must equal recorded dataset FPS")
        options["control_frequency"] = fps
        if options.get("use_pointcloud", True):
            field = options.get("pointcloud_field", "observation.point_cloud")
            feature = samples.info["features"][field]
            frame = feature.get("coordinate_frame", options.get("pointcloud_frame"))
            units = feature.get("units", options.get("pointcloud_units"))
            if not isinstance(frame, str) or not frame or units != "m":
                raise ValueError(
                    "Point cloud needs a declared coordinate_frame and units=m in feature "
                    "metadata or policy options"
                )
            if (
                options.get("pointcloud_frame", frame) != frame
                or options.get("pointcloud_units", units) != units
            ):
                raise ValueError("Point-cloud units/frame differ from recorded metadata")
            options.update(pointcloud_frame=frame, pointcloud_units=units)
        context = options.get("context_fields", ())
        if any(
            not key.startswith("observation.") or samples.input_features[key].type != "ENV"
            for key in context
        ):
            raise ValueError("Context inputs must be explicitly recorded numeric observations")
        width = sum(math.prod(samples.input_features[key].shape) for key in context)
        if options.get("context_dim", width) != width:
            raise ValueError("context_dim differs from the declared context fields")
        options["context_dim"] = width

    def build(self, options, stats=None):
        from .modeling_acdit import ACDiTPolicy

        return ACDiTPolicy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        return ACDiTProcessor(model.config, stats, model.tokenizer, device)

    def statistics(self, samples, options=None):
        stats = samples.statistics(keys=["observation.state", "action"])
        for key, feature in samples.input_features.items():
            if feature.type == "VISUAL":
                stats[key] = {"mean": [[[0.0]]] * 3, "std": [[[1.0]]] * 3}
        return stats

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        validate_statistics(
            stats,
            {key: value for key, value in expected.items() if value[0] != "ENV"},
            names,
            manifest=manifest,
        )

    def metric_specs(self, config):
        return [
            MetricSpec(
                "loss_diffusion",
                "diffusion_loss",
                "valid_padded_action_elements"
                if config.mask_padding_loss
                else "padded_action_elements",
                loss_count(
                    "_loss_weight",
                    config.padded_dim * (1 if config.mask_padding_loss else config.chunk_size),
                ),
                console="diffusion",
            )
        ]

    def loss_counts(self, batch):
        return {
            "_loss_weight": int((~batch["action_is_pad"]).sum())
            if getattr(self, "mask_loss", False)
            else len(batch["action"])
        }

    @staticmethod
    def supports_action_prediction(config):
        return config.stage == "full"

    def initialize(self, model, settings):
        if settings.get("pretrained_path"):
            raise ValueError(
                "--policy.path accepts native AC-DiT checkpoints; use the named RDT "
                "initializer options for external weights"
            )
        if model.config.mobility_pretrained_path:
            model.mobility.load_rdt(model.config.mobility_pretrained_path)

    def initialize_checkpoint(self, path, model, manifest):
        source, target = manifest["config"], model.config
        if source["stage"] != target.stage and (
            source["stage"] != "mobility" or target.stage != "full"
        ):
            raise ValueError(
                "AC-DiT supports mobility-to-full initialization; resume cannot switch stages"
            )
        keys = (
            "n_obs_steps",
            "chunk_size",
            "padded_dim",
            "mobility_hidden_dim",
            "mobility_num_layers",
            "mobility_num_heads",
            "num_inference_steps",
            "num_train_timesteps",
            "beta_schedule",
            "prediction_type",
            "use_pointcloud",
            "pointcloud_field",
            "pointcloud_frame",
            "pointcloud_units",
            "pointcloud_points",
            "pointcloud_knn",
            "pointcloud_stages",
            "pointcloud_embed_dim",
            "pointcloud_sampling",
            "pointcloud_projection_size",
            "lora_rank",
            "lora_alpha",
            "context_fields",
            "context_dim",
            "action_names",
            "state_names",
            "base_action_names",
            "action_slot_map",
            "state_slot_map",
            "control_frequency",
            "normalization_mapping",
        )
        from dataclasses import asdict

        resolved = asdict(target)
        import json

        changed = [
            key
            for key in keys
            if json.loads(json.dumps(source[key])) != json.loads(json.dumps(resolved[key]))
        ]
        if changed:
            raise ValueError(f"Mobility branch contract changed: {changed}")
        if source["stage"] == target.stage:
            self.load(path, model)
            return
        missing, unexpected = model.load_state_dict(
            load_file(path / "model.safetensors"), strict=False
        )
        if unexpected or any(not key.startswith(("body.", "perception.")) for key in missing):
            raise ValueError(
                "Mobility checkpoint does not contain the complete encoder/projector/head state"
            )
        if target.body_pretrained_path:
            model.body.load_rdt(target.body_pretrained_path)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = super().checkpoint_options(
            path, options, device, n_action_steps, temporal_ensemble_coeff
        )
        options.update(
            load_pretrained_encoders=False, mobility_pretrained_path=None, body_pretrained_path=None
        )
        return options

    def save(self, path, model, manifest, model_state=None):
        super().save(path, model, manifest, model_state)
        manifest["config"].update(
            load_pretrained_encoders=False, mobility_pretrained_path=None, body_pretrained_path=None
        )
