"""Spirit v1.5 registration through the shared native training/checkpoint APIs."""

import json
import shutil
from collections import deque
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, load_model

from alohamini.learning.metrics import MetricSpec, loss_count

from .configuration_spirit import SpiritConfig
from .preset import apply_preset
from .processor_spirit import SpiritFields, SpiritProcessor, fit_statistics


def _set_statistics(network, stats):
    with torch.no_grad():
        for module, key, field in (
            (network.normalize_inputs, "observation.state", "observation_state"),
            (network.normalize_targets, "action", "action"),
            (network.unnormalize_outputs, "action", "action"),
        ):
            buffer = getattr(module, "buffer_" + field)
            for bound in ("min", "max"):
                values = (
                    stats[key][bound]
                    if stats
                    else (torch.full_like(buffer[bound], -1 if bound == "min" else 1))
                )
                buffer[bound].copy_(torch.as_tensor(values, dtype=buffer[bound].dtype))


class SpiritTrainModel(torch.nn.Module):
    name = "spirit"

    def __init__(self, config, stats=None):
        super().__init__()
        from .modeling_spirit_vla import SpiritVLAConfig, SpiritVLAPolicy

        self.config = config
        options = {
            k: v
            for k, v in config.network.items()
            if k in {f.name for f in fields(SpiritVLAConfig)}
        }
        options.update(
            backbone=config.backbone_path,
            load_backbone_weights=False,
            device="cpu",
            n_action_steps=config.network["chunk_size"],
            action_horizon=config.network["chunk_size"],
            input_features={
                "observation.state": asdict(config.robot_state_feature),
                **{
                    f"observation.images.{slot}": dict(type="VISUAL", shape=[3, 240, 320])
                    for slot in ("cam_high", "cam_left_wrist", "cam_right_wrist")
                },
            },
            output_features={"action": asdict(config.action_feature)},
        )
        self.network = SpiritVLAPolicy(SpiritVLAConfig(**options))
        # The flow objective uses hidden states, not vocabulary logits. Exclude
        # the otherwise stateless optimizer entries; shared embeddings still train.
        embeddings = {id(p) for p in self.network.qwen.get_input_embeddings().parameters()}
        for parameter in self.network.qwen.get_output_embeddings().parameters():
            if id(parameter) not in embeddings:
                parameter.requires_grad_(False)
        _set_statistics(self.network, stats)
        if config.gradient_checkpointing:
            self.network.qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.reset()

    def reset(self):
        self._actions = deque()
        self.action_context = None

    def get_optim_params(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, batch):
        loss, values = self.network(batch)
        return loss * batch.get("_loss_weight", 1.0), {"flow_loss": values["loss/flow_mse"]}

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.network.select_action(batch)

    @torch.no_grad()
    def select_action(self, batch):
        if not self._actions:
            self.action_context = batch.get("_action_context")
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._actions.extend(actions.transpose(0, 1))
        return self._actions.popleft()


class SpiritAlgorithm:
    supports_remote_chunks = True

    @staticmethod
    def action_horizon(config):
        return config.chunk_size

    config_class = SpiritConfig
    include_task = True
    fsdp_classes = ("Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock", "BasicTransformerBlock")
    apply_preset = staticmethod(apply_preset)

    @staticmethod
    def metric_specs(config):
        return [
            MetricSpec(
                "loss_flow",
                "flow_loss",
                "valid_action_elements",
                loss_count("_loss_weight"),
                console="flow",
            )
        ]

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": int((~batch["action_is_pad"]).sum()) * batch["action"].shape[-1]}

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            from alohamini.learning.checkpoint import read_checkpoint

            checkpoint, manifest = read_checkpoint(settings["_checkpoint"])
            for key in ("backbone_path", "kinematics_path", "kinematic_fields"):
                options.pop(key, None)
            options = {**manifest["config"], **options}
            return self.checkpoint_options(checkpoint, options, device, None, "checkpoint")
        path = (
            Path(settings["pretrained_path"]).expanduser()
            if settings.get("pretrained_path")
            else None
        )
        if not settings.get("resume") and path is None:
            raise ValueError("Spirit fine-tuning requires --policy.path to local base weights")
        if path and not (path / "policy.json").exists() and (path / "config.json").is_file():
            external = json.loads((path / "config.json").read_text())
            from .configuration_spirit import NETWORK_DEFAULTS

            options["network"] = {
                **{k: external[k] for k in NETWORK_DEFAULTS if k in external},
                **options.get("network", {}),
            }
        if options.get("kinematics_path"):
            descriptor = json.loads(Path(options["kinematics_path"]).expanduser().read_text())
            options["kinematic_fields"] = descriptor
            options["kinematics_path"] = ""
        if (
            path
            and (path / "policy.json").is_file()
            and settings.get("normalization") == "checkpoint"
        ):
            saved = json.loads((path / "policy.json").read_text())["config"]
            for key in ("action_representation", "kinematic_fields"):
                if options.get(key, saved.get(key)) != saved.get(key):
                    raise ValueError(f"Checkpoint statistics require unchanged {key}")
        if options.get("backbone_path"):
            assets = Path(options["backbone_path"]).expanduser().resolve()
            if not (assets / "config.json").is_file():
                raise ValueError(
                    "backbone_path must contain local Qwen config/tokenizer/image assets"
                )
            options["backbone_path"] = str(assets)
        return options

    def sample_spec(self, options, *, cameras=()):
        horizon = options.get("network", {}).get("chunk_size", 60)
        return dict(
            exclude_fixed_state=False,
            delta_indices={"action": list(range(horizon))},
            include_task=True,
        )

    def statistics(self, samples, options=None):
        return fit_statistics(samples, self.config_class(**options))

    @staticmethod
    def validate_statistics(stats, expected, names, *, manifest=None):
        if set(stats) != {"observation.state", "action"}:
            raise ValueError("Spirit requires separate state/action quantile statistics")
        for key, values in stats.items():
            shape = (len(names),) if key == "action" else expected["observation.state"][1]
            lo, hi = (np.asarray(values.get(k, [])) for k in ("min", "max"))
            if (
                lo.shape != tuple(shape)
                or hi.shape != tuple(shape)
                or not np.isfinite(lo).all()
                or not np.isfinite(hi).all()
                or (hi < lo).any()
            ):
                raise ValueError(f"Invalid Spirit quantile statistics: {key}")
        if manifest and manifest["config"].get("action_representation") == "eef_delta":
            descriptor = manifest["config"]["kinematic_fields"]
            if descriptor["source_names"] != names or manifest["state_feature"]["names"] != names:
                raise ValueError("Checkpoint kinematics field names/order mismatch")

    def build(self, options, stats=None):
        return SpiritTrainModel(self.config_class(**deepcopy(options)), stats)

    def processor(self, model, stats, device):
        _set_statistics(model.network, stats)
        processor = SpiritProcessor(model.config, stats, device)
        processor.is_training = lambda: model.training
        return processor

    def initialize(self, model, settings):
        path = Path(settings["pretrained_path"]).expanduser()
        weights = load_file(path / "model.safetensors" if path.is_dir() else path, device="cpu")
        normalization = {
            k for k in model.network.state_dict() if k.startswith(("normalize_", "unnormalize_"))
        }
        weights = {k: v for k, v in weights.items() if k not in normalization}
        missing, unexpected = model.network.load_state_dict(weights, strict=False)
        if set(missing) != normalization or unexpected:
            raise RuntimeError(
                f"Spirit base mismatch: missing={sorted(set(missing) - normalization)}, unexpected={unexpected}"
            )

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        assets = (path / options["backbone_path"]).resolve()
        if not assets.is_relative_to(path.resolve()) or not assets.is_dir():
            raise ValueError("Spirit checkpoint is missing bundled local Qwen assets")
        options["backbone_path"] = str(assets)
        if temporal_ensemble_coeff not in ("checkpoint", None):
            raise ValueError("ACT temporal ensembling does not apply to Spirit")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        from alohamini.learning.checkpoint import save_weights

        fields = SpiritFields(model.config)
        if fields.geometry:
            absolute = fields.geometry.feature()
            manifest["statistics_features"] = {
                "observation.state": absolute,
                "action": fields.relative.feature(absolute),
            }
            manifest["field_transforms"] = [
                {
                    "operation": "forward_kinematics",
                    "fields": ["observation.state", "action"],
                    "descriptor": fields.geometry.descriptor,
                },
                {
                    "operation": "relative_actions",
                    "field": "action",
                    "anchor": "observation.state",
                    "configuration": fields.relative.configuration(),
                },
            ]
        else:
            state = manifest["state_feature"]
            units = dict(zip(state["names"], manifest["state_units"], strict=True))
            action = manifest["source_info"]["features"]["action"]
            manifest["statistics_features"] = {
                "observation.state": {**state, "units": manifest["state_units"]},
                "action": {
                    **action,
                    "units": [
                        units.get(name, "recorded (unit unverified)") for name in action["names"]
                    ],
                },
            }
            manifest["field_transforms"] = []
        manifest["field_transforms"].append(
            {
                "operation": "quantile_normalization",
                "fields": ["observation.state", "action"],
                "quantiles": [0.01, 0.99],
                "range": [-1, 1],
                "epsilon": 1e-8,
            }
        )
        assets = path / "qwen_assets"
        model.network.qwen.config.save_pretrained(assets)
        model.network.language_tokenizer.save_pretrained(assets)
        model.network.image_processor.save_pretrained(assets)
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(
            model.config.backbone_path, local_files_only=True, use_fast=False
        ).save_pretrained(assets)
        provenance = Path(model.config.backbone_path) / "sources.json"
        if provenance.is_file():
            shutil.copyfile(provenance, assets / "sources.json")
        manifest["config"]["backbone_path"] = "qwen_assets"
        save_weights(path / "model.safetensors", model, model_state)

    def load(self, path, model):
        load_model(model, path / "model.safetensors", strict=True)
