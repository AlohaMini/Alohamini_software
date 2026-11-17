"""Serializable platform settings for Cosmos3 Nano and Edge."""

import math
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import torch

from alohamini.policies.configuration import PolicyConfig

from .backbone_configuration import (
    Nemotron3DenseVLMoTConfig,
    Qwen3VLMoTConfig,
    Wan2pt2VAEConfig,
)


@dataclass
class Cosmos3Config(PolicyConfig):
    variant: str = "nano"
    mot_backbone: dict = field(default_factory=dict)
    vae: dict = field(default_factory=lambda: asdict(Wan2pt2VAEConfig()))
    tokenizer_path: str = ""
    dtype: str = "bfloat16"
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    max_action_dim: int = 64
    latent_patch_size: int = 2
    timestep_scale: float = 0.001
    timestep_embedding_dim: int = 256
    num_embodiment_domains: int = 32
    base_fps: float = 24.0
    temporal_modality_margin: int = 15000
    video_temporal_causal: bool = False
    train_gen_stream_only: bool = True
    freeze_vision_encoder: bool = True
    reinitialize_robot_heads: bool = True
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    compile_model: bool = False
    compile_mode: str = "default"
    num_train_timesteps: int = 1000
    video_time_distribution: str = "waver"
    training_shift: float | dict = field(
        default_factory=lambda: {"256": 3.0, "480": 5.0, "720": 10.0}
    )
    independent_action_schedule: bool = False
    action_time_distribution: str = "logitnormal"
    action_training_shift: float | None = None
    video_loss_weight: float = 10.0
    action_loss_weight: float = 10.0
    num_inference_steps: int = 30
    guidance_scale: float = 1.0
    guidance_interval: tuple | None = None
    sampling_shift: float = 5.0
    sampler: str = "unipc"
    save_predicted_video: bool = False
    mode: str = "wam"
    use_state: bool = False
    state_action_names: list = field(default_factory=list)
    domain_id: int = 0
    fps: float = 30.0
    view_size: tuple = (256, 256)
    camera_keys: list = field(default_factory=list)
    camera_roles: dict = field(default_factory=dict)
    canvas_hw: tuple | None = None
    view_description: str = ""
    action_feature_names: list = field(default_factory=list)
    pretrained_action_names: list = field(default_factory=list)
    cfg_dropout_rate: float = 0.1
    normalization_mapping: dict = field(
        default_factory=lambda: dict(VISUAL="IDENTITY", STATE="IDENTITY", ACTION="QUANTILES")
    )
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple = (0.9, 0.99)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.05
    optimizer_grad_clip_norm: float = 1.0
    optimizer_action_lr_multiplier: float = 5.0
    scheduler_type: str = "warmup_linear"
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 16000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in ("nano", "edge") or self.dtype not in ("float32", "bfloat16"):
            raise ValueError("Cosmos3 requires nano/edge and float32/bfloat16")
        if not self.tokenizer_path:
            raise ValueError("Cosmos3 requires a local tokenizer_path")
        if not self.mot_backbone:
            self.mot_backbone = self.backbone_class().to_dict()
        if self.mot_backbone.get("model_type") != self.backbone_class.model_type:
            raise ValueError("Cosmos3 backbone type differs from the selected variant")
        if self.n_obs_steps != 1 or self.mode != "wam" or self.video_temporal_causal:
            raise ValueError("Cosmos3 supports wam mode with one current observation")
        if not self.freeze_vision_encoder:
            raise ValueError(
                "Cosmos3 cameras use the generation VAE; the unused vision tower stays frozen"
            )
        for name in (
            "chunk_size",
            "n_action_steps",
            "max_action_dim",
            "latent_patch_size",
            "timestep_embedding_dim",
            "num_embodiment_domains",
            "num_train_timesteps",
            "num_inference_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Cosmos3 {name} must be a positive integer")
        if (
            self.n_action_steps > self.chunk_size
            or not 0 <= self.domain_id < self.num_embodiment_domains
        ):
            raise ValueError("Cosmos3 execution horizon or embodiment row is outside its capacity")
        vae = Wan2pt2VAEConfig(**self.vae)
        self.vae = asdict(vae)
        if self.chunk_size % (vae.scale_factor_temporal or 1):
            raise ValueError("Cosmos3 horizon must divide into complete temporal VAE groups")
        if self.original_action_dim > self.max_action_dim:
            raise ValueError("Cosmos3 recorded actions exceed max_action_dim")
        if len(self.action_feature_names) != self.original_action_dim:
            raise ValueError("Cosmos3 requires the recorded action coordinate names")
        if len(set(self.action_feature_names)) != self.original_action_dim:
            raise ValueError("Cosmos3 action names must be unique")
        if (
            not self.reinitialize_robot_heads
            and self.pretrained_action_names != self.action_feature_names
        ):
            raise ValueError(
                "Keeping Cosmos3 robot heads requires explicit pretrained action names in order"
            )
        if self.use_state:
            if (
                self.robot_state_feature is None
                or self.robot_state_feature.shape != self.action_feature.shape
            ):
                raise ValueError("Cosmos3 state conditioning requires state in the action layout")
            if self.state_action_names != self.action_feature_names:
                raise ValueError(
                    "Cosmos3 state coordinates must explicitly match action coordinates"
                )
            if self.normalization_mapping.get("STATE") != self.normalization_mapping.get("ACTION"):
                raise ValueError(
                    "Cosmos3 state and action layout require identical normalization modes"
                )
        if not self.camera_keys:
            self.camera_keys = list(self.image_features)
        if not 1 <= len(self.camera_keys) <= 3 or len(set(self.camera_keys)) != len(
            self.camera_keys
        ):
            raise ValueError("Cosmos3 supports one to three ordered camera views")
        if set(self.camera_keys) != set(self.image_features):
            raise ValueError("Cosmos3 camera order must cover exactly the selected cameras")
        if self.camera_roles.keys() - set(self.camera_keys):
            raise ValueError("Cosmos3 camera roles refer to unselected cameras")
        for name in ("view_size", "canvas_hw"):
            size = getattr(self, name)
            if size is not None and (
                len(size) != 2 or any(type(n) is not int or n < 1 for n in size)
            ):
                raise ValueError(f"Cosmos3 {name} must be positive pixel dimensions")
        if self.canvas_hw and any(n % (vae.scale_factor_spatial or 1) for n in self.canvas_hw):
            raise ValueError("Cosmos3 canvas must contain complete VAE cells")
        for name in ("fps", "base_fps", "sampling_shift", "timestep_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Cosmos3 {name} must be positive and finite")
        for name in ("video_loss_weight", "action_loss_weight", "optimizer_action_lr_multiplier"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Cosmos3 {name} must be nonnegative and finite")
        if (
            not 0 <= self.cfg_dropout_rate <= 1
            or not math.isfinite(self.guidance_scale)
            or self.guidance_scale < 1
        ):
            raise ValueError("Cosmos3 dropout or guidance scale is invalid")
        if self.sampler != "unipc" or self.attn_implementation not in ("eager", "sdpa"):
            raise ValueError("Cosmos3 currently supports UniPC and eager/SDPA attention")
        if self.compile_model and self.compile_mode != "default":
            raise ValueError("Cosmos3 compiled decoder layers currently use default mode")
        if self.video_time_distribution not in (
            "uniform",
            "logitnormal",
            "waver",
        ) or self.action_time_distribution not in ("uniform", "logitnormal", "waver"):
            raise ValueError("Cosmos3 noise distribution is invalid")
        shifts = (
            self.training_shift.values()
            if isinstance(self.training_shift, dict)
            else [self.training_shift]
        )
        if any(not math.isfinite(v) or v <= 0 for v in shifts):
            raise ValueError("Cosmos3 training shifts must be positive and finite")
        if (
            self.independent_action_schedule
            and self.action_training_shift is None
            and isinstance(self.training_shift, dict)
        ):
            raise ValueError("Cosmos3 independent action schedule needs action_training_shift")

    @property
    def original_action_dim(self):
        return self.action_feature.shape[0]

    @property
    def backbone_class(self):
        return Qwen3VLMoTConfig if self.variant == "nano" else Nemotron3DenseVLMoTConfig

    def backend(self):
        values = dict(self.__dict__)
        values.update(
            dtype=getattr(torch, self.dtype),
            mot_backbone=self.backbone_class.from_dict(self.mot_backbone),
            vae=Wan2pt2VAEConfig(**self.vae),
            original_action_dim=self.original_action_dim,
        )
        return SimpleNamespace(**values)


@dataclass
class Cosmos3NanoConfig(Cosmos3Config):
    variant: str = "nano"


@dataclass
class Cosmos3EdgeConfig(Cosmos3Config):
    variant: str = "edge"
