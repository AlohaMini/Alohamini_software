"""Training/checkpoint adapter around the OpenPI model and transforms."""

import logging
import math
import shutil
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_model, save_model

from alohamini.policies.configuration import PolicyConfig

from .configuration_pi05 import PI05Config
from .finetuning import FINETUNING_MODES, configure_finetuning, load_base_weights
from .processor_pi05 import ARM_DELTA_DIMS, PI05Processor, fit_statistics, validate_camera_names


@dataclass
class PI05TrainingConfig(PolicyConfig):
    network: dict = field(default_factory=dict)
    camera_order: tuple[str, ...] | None = None
    tokenizer_path: str = ""
    n_action_steps: int = 50
    num_inference_steps: int = 10
    delta_dims: tuple = ARM_DELTA_DIMS
    span_floors: dict | None = None
    training_recipe: str = "flow"
    finetuning_mode: str = "full"
    a_vision_mode: str = "frozen"
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: float = 16.0
    expert_lora_rank: int = 32
    expert_lora_alpha: float = 32.0
    fast_tokenizer_path: str = ""
    fast_max_tokens: int = 256
    ce_chunk_size: int = 32
    flow_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    vlm_data_path: str = ""
    vlm_data_sha256: str = ""
    vlm_max_tokens: int = 256
    vlm_loss_weight: float = 1.0
    optimizer_lr: float = 2.5e-5
    optimizer_weight_decay: float = 1e-10
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "warmup_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        core = PI05Config(**self.network)
        self.network = asdict(core)
        if self.training_recipe not in ("flow", "ki"):
            raise ValueError("training_recipe must be flow or ki")
        if self.finetuning_mode not in FINETUNING_MODES:
            raise ValueError(f"finetuning_mode must be one of {tuple(FINETUNING_MODES)}")
        if self.a_vision_mode not in ("frozen", "full"):
            raise ValueError("a_vision_mode must be frozen or full")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        for key in (
            "lora_rank",
            "expert_lora_rank",
            "fast_max_tokens",
            "ce_chunk_size",
            "vlm_max_tokens",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "lora_alpha",
            "expert_lora_alpha",
            "flow_loss_weight",
            "fast_loss_weight",
            "vlm_loss_weight",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if self.training_recipe == "ki":
            if self.finetuning_mode == "b":
                raise ValueError(
                    "Mode b freezes the VLM; use training_recipe=flow for this baseline"
                )
            if not self.fast_tokenizer_path:
                raise ValueError("KI requires a local fast_tokenizer_path")
        elif self.vlm_data_path:
            raise ValueError("VLM co-training requires training_recipe=ki")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (18,):
            raise ValueError("PI0.5 AlohaMini requires the original 18-D position state")
        if self.action_feature.shape != (18,) or core.action_dim < 18:
            raise ValueError("PI0.5 requires 18 physical action coordinates")
        if any(not key.startswith("observation.images.") for key in self.image_features):
            raise ValueError("PI0.5 visual features must use observation.images.<camera>")
        cameras = validate_camera_names(
            key.removeprefix("observation.images.") for key in self.image_features
        )
        self.camera_order = validate_camera_names(
            cameras if self.camera_order is None else self.camera_order
        )
        if set(self.camera_order) != set(cameras):
            raise ValueError("PI0.5 camera_order must match the selected image features")
        if not 1 <= self.n_action_steps <= core.action_horizon or self.num_inference_steps < 1:
            raise ValueError("Invalid PI0.5 execution horizon or sampling steps")

    @property
    def action_delta_indices(self):
        return list(range(self.network["action_horizon"]))


class PI05BatchProcessor:
    def __init__(self, config, stats, device):
        from .tokenizer import PaligemmaTokenizer

        self.config, self.device = config, device
        self.is_training = lambda: False
        self._fast = None
        self._vlm_data = None
        self.transform = PI05Processor(
            stats,
            PaligemmaTokenizer(
                config.tokenizer_path,
                config.network["max_token_len"],
                config.training_recipe == "ki",
            ),
            action_dim=config.network["action_dim"],
            delta_dims=config.delta_dims,
            span_floors=config.span_floors,
            cameras=config.camera_order,
        )

    def __call__(self, batch):
        states = batch["observation.state"]
        tasks = batch.get("task")
        if not isinstance(tasks, (list, tuple)) or len(tasks) != len(states):
            raise ValueError("PI0.5 requires one task prompt per observation")
        observations, targets = [], []
        for i, state in enumerate(states):
            images = {}
            for key in self.config.image_features:
                rgb = batch[key][i].detach().cpu()
                if rgb.dtype != torch.uint8:
                    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
                        raise ValueError(f"{key}: expected CHW RGB in [0, 1]")
                    rgb = rgb.mul(255).round().byte()
                # OpenPI owns resize/padding and [-1, 1] scaling; raw bytes need
                # neither a float round trip nor ImageNet normalization.
                images[key.removeprefix("observation.images.")] = rgb.permute(1, 2, 0).numpy()
            observation, target = self.transform.prepare(
                state.detach().cpu().numpy(),
                images,
                tasks[i],
                batch["action"][i].detach().cpu().numpy() if "action" in batch else None,
                device=self.device,
            )
            observations.append(observation)
            if target is not None:
                targets.append(target)
        first = observations[0]
        values = {}
        for key, value in vars(first).items():
            if isinstance(value, dict):
                values[key] = {
                    name: torch.cat([getattr(o, key)[name] for o in observations]) for name in value
                }
            else:
                values[key] = (
                    None if value is None else torch.cat([getattr(o, key) for o in observations])
                )
        result = {
            "observation": SimpleNamespace(**values),
            "_action_context": states.to(self.device),
        }
        if targets:
            result["action"] = torch.cat(targets)
            if self.config.training_recipe == "ki":
                from .fast_tokenizer import FASTTokenizer

                if self._fast is None:
                    self._fast = FASTTokenizer(
                        self.config.fast_tokenizer_path,
                        self.transform.tokenizer._tokenizer,
                        self.config.fast_max_tokens,
                    )
                result["fast_tokens"], result["fast_mask"] = self._fast.targets(
                    result["action"][..., :18].detach().cpu().numpy(),
                    device=self.device,
                )
                if self.config.vlm_data_path and self.is_training():
                    from .vlm_data import VLMCoTrainingData

                    if self._vlm_data is None:
                        self._vlm_data = VLMCoTrainingData(self.config, self.transform.tokenizer)
                    result["vlm_batch"] = self._vlm_data.sample(len(states), device=self.device)
        return result

    def action(self, tensor, *, context=None):
        if context is None:
            raise ValueError("PI0.5 delta actions require their original observation state")
        single = tensor.ndim == 2
        result = self.transform.restore_actions(tensor[:, None] if single else tensor, context)
        return result[:, 0] if single else result


class PI05TrainModel(torch.nn.Module):
    name = "pi05"

    def __init__(self, config):
        super().__init__()
        from .modeling_pi05 import PI0Pytorch

        self.config = config
        self.network = PI0Pytorch(PI05Config(**config.network))
        self.trainable_parameters = configure_finetuning(self.network, config)
        logging.getLogger(__name__).info(
            "PI0.5 recipe=%s finetuning=%s parameters=%s",
            config.training_recipe,
            config.finetuning_mode,
            self.trainable_parameters,
        )
        if config.gradient_checkpointing:
            self.network.gradient_checkpointing_enable()
        else:
            self.network.gradient_checkpointing_disable()
        self.reset()

    def get_optim_params(self):
        return (param for param in self.parameters() if param.requires_grad)

    def forward(self, batch):
        # Author's mean includes padded coordinates and repeated tail targets.
        if self.config.training_recipe == "flow":
            flow = self.network(batch["observation"], batch["action"]).mean()
            loss, outputs = self.config.flow_loss_weight * flow, {"flow_loss": flow.detach()}
        else:
            flow, fast = self.network(
                batch["observation"],
                batch["action"],
                fast_tokens=batch["fast_tokens"],
                fast_mask=batch["fast_mask"],
                knowledge_insulation=True,
                ce_chunk_size=self.config.ce_chunk_size,
            )
            flow, fast = flow.mean(), fast.mean()
            loss = self.config.flow_loss_weight * flow + self.config.fast_loss_weight * fast
            outputs = {
                "flow_loss": flow.detach(),
                "fast_loss": fast.detach(),
                "vlm_loss": flow.detach() * 0,
            }
            if self.training and "vlm_batch" in batch:
                vlm = self.network.forward_language(
                    **batch["vlm_batch"],
                    ce_chunk_size=self.config.ce_chunk_size,
                ).mean()
                loss = loss + self.config.vlm_loss_weight * vlm
                outputs["vlm_loss"] = vlm.detach()
        return loss * batch.get("_loss_weight", 1.0), outputs

    def reset(self):
        self._actions = deque()
        self.action_context = None

    @torch.no_grad()
    def predict_action_chunk(self, batch):
        return self.network.sample_actions(
            next(self.parameters()).device,
            batch["observation"],
            num_steps=self.config.num_inference_steps,
        )

    @torch.no_grad()
    def select_action(self, batch):
        if not self._actions:
            self.action_context = batch["_action_context"].detach().clone()
            chunk = self.predict_action_chunk(batch)
            self._actions.extend(chunk[:, : self.config.n_action_steps].transpose(0, 1))
        return self._actions.popleft()


class PI05Algorithm:
    config_class = PI05TrainingConfig
    state_stats_key = "state"
    include_task = True
    fsdp_classes = ()

    @staticmethod
    def metric_specs(config):
        from alohamini.learning.metrics import MetricSpec, loss_count

        # Each sample mean retains the network's padded action width and tails.
        specs = [
            MetricSpec(
                "loss_flow",
                "flow_loss",
                "samples_with_padded_action_mean",
                loss_count("_loss_weight"),
                console="flow",
            )
        ]
        if config is not None and config.training_recipe == "ki":
            specs.append(
                MetricSpec(
                    "loss_fast",
                    "fast_loss",
                    "samples_with_token_mean",
                    loss_count("_loss_weight"),
                    console="fast",
                )
            )
            if config.vlm_data_path:
                specs.append(
                    MetricSpec(
                        "loss_vlm",
                        "vlm_loss",
                        "samples_with_token_mean",
                        loss_count("_loss_weight"),
                        phases=("train",),
                    )
                )
        return specs

    @staticmethod
    def apply_preset(settings):
        result = deepcopy(settings)
        defaults = dict(
            steps=30000,
            batch_size=32,
            save_freq=1000,
            log_freq=100,
            state="joint_position,base_velocity,lift_height",
            # OpenPI casts selected model weights to BF16 and retains FP32
            # projections/norms; its trainer does not wrap forward in autocast.
            mixed_precision="none",
        )
        for key, value in defaults.items():
            result.setdefault(key, value)
        model = result.setdefault("model", {})
        model.setdefault("gradient_checkpointing", True)
        model.setdefault("network", {}).setdefault("pytorch_compile_mode", "max-autotune")
        result.setdefault("ema", {"enable": False})
        result.setdefault(
            "paper_preset",
            {
                "name": "openpi-pi05-alohamini-v1",
                "source": "https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac",
                "protocol": (
                    "Released OpenPI ALOHA transforms and PyTorch training defaults; "
                    "not paper pretraining"
                ),
                "data": {
                    "state": "Original 18-D AlohaMini coordinates, encoded into language tokens",
                    "action": (
                        "12 arm joints relative to the current state; "
                        "grippers/base/lift remain absolute targets/velocities"
                    ),
                    "images": "RGB; resize with aspect-preserving padding to 224x224; [-1,1]",
                    "language": "PaliGemma task + quantized state, length 200 by default",
                    "normalization": (
                        "Train-window q01/q99; float64 transforms; additive 1e-6; "
                        "no automatic clipping"
                    ),
                    "time": "50 future recorded rows by default; no timestamp resampling",
                    "loss": (
                        "Mean over batch/time/padded action width, "
                        "retaining repeated episode-tail targets"
                    ),
                },
                "adaptations": [
                    "No Trossen signs/gripper geometry",
                    "Dataset-selected camera names/order; no fixed ALOHA camera slots",
                    "Configured dataset image size is saved explicitly",
                    "Trainer EMA is optional and disabled by default, matching the PyTorch recipe",
                    "Explicit overrides saved in train_config.json",
                ],
            },
        )
        if model.get("training_recipe", "flow") == "ki":
            result["paper_preset"]["ki"] = {
                "source": "https://www.pi.website/download/pi05_KI.pdf",
                "method": (
                    "FAST CE + flow MSE; per-layer stop-gradient K/V; "
                    "mutually masked action branches"
                ),
                "weights": "flow=1, FAST=1, optional VLM=1; mean tokens per sample",
                "fast": (
                    "physical-intelligence/fast@ec4d7aa71691cac0b8bed6942be45684db2110f4; "
                    "physical 18-D normalized delta actions"
                ),
                "adaptations": (
                    "Retain released Pi05 prompt; optional local VQA/planning JSONL, "
                    "one VLM example per robot sample; not the unreleased generalist data mixture"
                ),
            }
        return result

    def options(self, settings, device):
        options = dict(settings.get("model", {}))
        if options.get("training_recipe") == "ki" and options.get("finetuning_mode") == "b":
            raise ValueError("Mode b freezes the VLM; use training_recipe=flow for this baseline")
        if options.get("training_recipe") == "ki" and not options.get("fast_tokenizer_path"):
            from alohamini.paths import WorkspacePaths

            options["fast_tokenizer_path"] = str(
                WorkspacePaths().pretrained / "pi05_tokenizers" / "fast"
            )
        for key in ("tokenizer_path", "fast_tokenizer_path"):
            if options.get(key):
                options[key] = str(Path(options[key]).expanduser().resolve())
        options.setdefault("n_action_steps", options.get("network", {}).get("action_horizon", 50))
        if options.get("vlm_data_path"):
            from .vlm_data import manifest_digest

            options["vlm_data_path"] = str(Path(options["vlm_data_path"]).expanduser().resolve())
            digest = manifest_digest(options["vlm_data_path"])
            if options.get("vlm_data_sha256") and options["vlm_data_sha256"] != digest:
                raise ValueError("VLM co-training manifest changed")
            options["vlm_data_sha256"] = digest
        if not settings.get("resume") and not settings.get("pretrained_path"):
            raise ValueError("PI0.5 requires --policy.path to local converted PyTorch base weights")
        return options

    def sample_spec(self, options, *, cameras=()):
        return dict(
            # State tokens retain all named coordinates, including fixed axes.
            exclude_fixed_state=False,
            delta_indices={
                "action": list(range(options.get("network", {}).get("action_horizon", 50)))
            },
            include_task=True,
        )

    def statistics(self, samples, options=None):
        return fit_statistics(samples, delta_dims=(options or {}).get("delta_dims", ARM_DELTA_DIMS))

    def build(self, options, stats=None):
        return PI05TrainModel(self.config_class(**options))

    def processor(self, model, stats, device):
        processor = PI05BatchProcessor(model.config, stats, device)
        processor.is_training = lambda: model.training
        return processor

    @staticmethod
    def loss_counts(batch):
        return {"_loss_weight": len(batch["action"])}

    def initialize(self, model, settings):
        path = Path(settings["pretrained_path"]).expanduser()
        weights = path / "model.safetensors" if path.is_dir() else path
        load_base_weights(model.network, weights)

    def checkpoint_options(self, path, options, device, n_action_steps, temporal_ensemble_coeff):
        options = deepcopy(options)
        if "camera_order" not in options:
            # Older processors always used this order, regardless of feature order.
            options["camera_order"] = tuple(
                name for name in ("forward", "wrist_left", "wrist_right")
                if f"observation.images.{name}" in options["input_features"]
            )
        assets = (path / options["tokenizer_path"]).resolve()
        if not assets.is_relative_to(path) or not assets.is_file():
            raise ValueError("PI0.5 checkpoint is missing its local tokenizer")
        options["tokenizer_path"] = str(assets)
        if options.get("fast_tokenizer_path"):
            fast_assets = (path / options["fast_tokenizer_path"]).resolve()
            if not fast_assets.is_relative_to(path) or not fast_assets.is_dir():
                raise ValueError("PI0.5 checkpoint is missing its local FAST tokenizer")
            options["fast_tokenizer_path"] = str(fast_assets)
        if temporal_ensemble_coeff not in (None, "checkpoint"):
            raise ValueError("ACT temporal ensembling does not apply to PI0.5")
        if n_action_steps is not None:
            options["n_action_steps"] = n_action_steps
        return options

    def save(self, path, model, manifest, model_state=None):
        if model_state is not None:
            raise ValueError("PI0.5 sharded export is not adapted yet")
        shutil.copyfile(Path(model.config.tokenizer_path).expanduser(), path / "tokenizer.model")
        manifest["config"]["tokenizer_path"] = "tokenizer.model"
        if model.config.fast_tokenizer_path:
            from .fast_tokenizer import FAST_FILES

            (path / "fast_tokenizer").mkdir()
            for filename in FAST_FILES:
                shutil.copyfile(
                    Path(model.config.fast_tokenizer_path).expanduser() / filename,
                    path / "fast_tokenizer" / filename,
                )
            manifest["config"]["fast_tokenizer_path"] = "fast_tokenizer"
        save_model(model.network, path / "model.safetensors")

    def load(self, path, model):
        load_model(model.network, path / "model.safetensors", strict=True)

    def initialize_checkpoint(self, path, model, manifest):
        # A full native checkpoint can seed fresh adapters. Resuming and loading
        # existing adapter checkpoints always retain strict tensor matching.
        if manifest["config"].get("finetuning_mode", "full") in ("full", "b"):
            load_base_weights(model.network, path / "model.safetensors")
        else:
            self.load(path, model)

    def validate_statistics(self, stats, expected, names, *, manifest=None):
        if manifest is not None and (manifest.get("state_feature") or {}).get("names") != names:
            raise ValueError("PI0.5 requires original position state matching action coordinates")
        # PI05Processor validates quantile shapes, ordering and finiteness.
        PI05Processor(stats, tokenizer=None)
