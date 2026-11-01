"""GR00T N1.7 named coordinates, local official weights and shared training runtime."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from safetensors.torch import load_file

from alohamini.learning.metrics import MetricSpec, loss_count
from alohamini.learning.processor import limits_statistics
from alohamini.paths import WorkspacePaths
from alohamini.policies.chunk import ChunkAlgorithm, local_assets

from .configuration_gr00t_n17 import GR00TN17Config
from .preset import apply_preset


class GR00TN17Algorithm(ChunkAlgorithm):
    config_class = GR00TN17Config
    apply_preset = staticmethod(apply_preset)
    supports_field_transforms = True
    fsdp_classes = (
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionModel",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
        "CategorySpecificLinear",
        "Embedding",
    )

    def options(self, settings, device):
        options = deepcopy(settings.get("model", {}))
        if settings.get("resume"):
            return options
        base = Path(
            settings.get("pretrained_path") or WorkspacePaths().pretrained / "gr00t_n17"
        ).expanduser()
        assets = local_assets(options.get("assets_path") or base / "backbone")
        options["assets_path"] = str(assets)
        source = base / "config.json"
        if source.is_file() and not (base / "policy.json").exists():
            network = json.loads(source.read_text())
            if network.get("model_type") != "Gr00tN1d7":
                raise ValueError("Expected an official Gr00tN1d7 base config")
            for key, value in options.get("network", {}).items():
                network[key] = (
                    {**network[key], **value}
                    if isinstance(network.get(key), dict) and isinstance(value, dict)
                    else value
                )
            options["network"] = network
        elif not options.get("network"):
            raise FileNotFoundError(source)
        for name in (
            "crop_fraction",
            "shortest_image_edge",
            "formalize_language",
            "use_percentiles",
        ):
            if options["network"].get(name) is not None:
                options.setdefault(name, options["network"][name])
        if not options.get("backbone_config"):
            options["backbone_config"] = json.loads((assets / "config.json").read_text())
        # Preserve official tensor shapes, while keeping padded coordinates and
        # tokens out of newly trained native policies.
        options["network"].setdefault("strict_action_padding_mask", True)
        options["network"].setdefault("strict_vl_padding_mask", True)
        options.setdefault("chunk_size", options["network"].get("action_horizon", 40))
        options.setdefault("n_action_steps", options["chunk_size"])
        return options

    @staticmethod
    def sample_spec(options, *, cameras=()):
        return dict(
            delta_indices={
                "observation.state": [0],
                "action": list(range(options.get("chunk_size", 40))),
                **{f"observation.images.{c}": [0] for c in cameras},
            },
            include_task=True,
        )

    @staticmethod
    def prepare_options(options, samples):
        if samples.selection is None:
            raise ValueError("GR00T requires named state")
        explicit_slots = "state_slot_map" in options and "action_slot_map" in options
        transformed = getattr(samples, "model_features", {})
        for field, name, fallback in (
            ("observation.state", "state", samples.selection.feature),
            ("action", "action", samples.info["features"]["action"]),
        ):
            names = transformed.get(field, fallback).get("names")
            if not names or (
                options.get(name + "_names") and list(options[name + "_names"]) != names
            ):
                raise ValueError("GR00T recorded field names/order changed")
            options[name + "_names"] = list(names)
            options.setdefault(name + "_slot_map", {n: i for i, n in enumerate(names)})
        if not options.get("reinitialize_embodiment", True) and not explicit_slots:
            raise ValueError("Reusing a pretrained embodiment requires explicit coordinate slots")

    def statistics(self, samples, options=None):
        stats = limits_statistics(samples)
        if (options or {}).get("use_percentiles", True):
            for key in ("observation.state", "action"):
                if "q01" in stats[key] and "q99" in stats[key]:
                    stats[key].update(min=stats[key]["q01"], max=stats[key]["q99"])
                    continue
                values = torch.stack(
                    [samples._value(i, key).double() for i in sorted(samples._used_rows[key])]
                )
                low, high = torch.quantile(
                    values, torch.tensor([0.01, 0.99], dtype=torch.float64), dim=0
                )
                stats[key].update(
                    min=low.tolist(), max=high.tolist(), q01=low.tolist(), q99=high.tolist()
                )
        return stats

    def build(self, options, stats=None):
        from .modeling_gr00t_n17 import GR00TN17Policy

        return GR00TN17Policy(self.config_class(**deepcopy(options)))

    def processor(self, model, stats, device):
        from .processor_gr00t_n17 import GR00TN17Processor

        return GR00TN17Processor(model.config, stats, model.vlm_processor, device, model)

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": int((~batch["action_is_pad"]).sum())}

    def metric_specs(self, config):
        return [
            MetricSpec(
                "flow_loss",
                "flow_loss",
                "valid_action_coordinates",
                loss_count("_loss_weight", config.action_feature.shape[0]),
                console="flow",
            )
        ]

    def initialize(self, model, settings):
        if not model.config.initialize_from_base:
            return
        root = Path(
            settings.get("pretrained_path") or WorkspacePaths().pretrained / "gr00t_n17"
        ).expanduser()
        index = root / "model.safetensors.index.json"
        files = (
            sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if index.exists()
            else ["model.safetensors"]
        )
        expected, loaded = model.state_dict(), set()
        for name in files:
            values = load_file(root / name)
            unknown = values.keys() - expected.keys()
            if unknown:
                raise ValueError(f"Unexpected official GR00T tensors: {sorted(unknown)[:5]}")
            for key, tensor in values.items():
                if key in loaded or expected[key].shape != tensor.shape:
                    raise ValueError(f"GR00T tensor shape/name mismatch: {key}")
            model.load_state_dict(values, strict=False)
            loaded.update(values)
        if loaded != expected.keys():
            raise ValueError(f"Incomplete GR00T weights: {sorted(expected.keys() - loaded)[:5]}")
        if model.config.reinitialize_embodiment:
            from ._vendor.embodiment_conditioned_mlp import CategorySpecificLinear

            with torch.no_grad():
                for module in model.action_head.modules():
                    if isinstance(module, CategorySpecificLinear):
                        module.W[model.config.embodiment_id].normal_(std=0.02)
                        module.b[model.config.embodiment_id].zero_()
