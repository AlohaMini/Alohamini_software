"""Explicit VLA-JEPA geometry, camera bindings and temporal training protocol."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class VLAJEPAConfig(PolicyConfig):
    assets_path: str = ""
    qwen_config: dict = field(default_factory=dict)
    jepa_config: dict = field(default_factory=dict)
    dtype: str = "bfloat16"
    base_weights_mode: str = "backbones"
    reinitialize_robot_heads: bool = True
    reinitialize_world_heads: bool = False
    gradient_checkpointing: bool = True
    freeze_qwen: bool = False
    enable_world_model: bool = True
    world_model_cameras: list = field(default_factory=list)
    world_model_context: str = "causal"
    world_model_rope_legacy: bool = False
    retain_unused_predictor_heads: bool = False
    resize_images_to: tuple | None = (224, 224)
    chunk_size: int = 7
    n_action_steps: int = 7
    num_video_frames: int = 8
    jepa_tubelet_size: int = 2
    num_action_tokens_per_timestep: int = 8
    num_embodied_action_tokens_per_instruction: int = 32
    num_inference_timesteps: int = 4
    special_action_token: str = "<|action_{}|>"
    embodied_action_token: str = "<|embodied_action|>"
    prompt_template: str = (
        "Your task is {instruction}. Infer the temporal dynamics from frames {actions} "
        "and produce the corresponding policy actions {e_actions}."
    )
    tokenizer_max_length: int = 4096
    action_dim: int = 7
    state_dim: int = 8
    action_hidden_size: int = 1024
    action_model_type: str = "DiT-B"
    action_num_layers: int = 16
    action_num_heads: int | None = None
    action_attention_head_dim: int | None = None
    action_dropout: float = 0.2
    action_num_timestep_buckets: int = 1000
    action_noise_beta_alpha: float = 1.5
    action_noise_beta_beta: float = 1.0
    action_noise_s: float = 0.999
    action_max_seq_len: int = 1024
    predictor_depth: int = 12
    predictor_num_heads: int = 8
    predictor_mlp_ratio: float = 4.0
    predictor_dropout: float = 0.0
    lr_scales: dict = field(default_factory=lambda: dict(qwen=1 / 3, action=10 / 3, world=50 / 3))
    world_model_loss_weight: float = 0.1
    repeated_diffusion_steps: int = 8
    normalization_mapping: dict = field(
        default_factory=lambda: dict(VISUAL="IDENTITY", STATE="MEAN_STD", ACTION="MIN_MAX")
    )
    clip_normalized_actions: bool = True
    optimizer_lr: float = 3e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "diffusers_cosine"
    scheduler_warmup_steps: int = 20000
    scheduler_decay_steps: int = 120000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if not self.assets_path or not self.qwen_config:
            raise ValueError("VLA-JEPA requires local Qwen assets and saved qwen_config")
        if self.dtype not in ("bfloat16", "float32"):
            raise ValueError("VLA-JEPA dtype must be bfloat16 or float32")
        if self.base_weights_mode not in ("backbones", "policy"):
            raise ValueError("VLA-JEPA base_weights_mode must be backbones or policy")
        if self.world_model_context not in ("causal", "shared"):
            raise ValueError("VLA-JEPA world_model_context must be causal or shared")
        if self.clip_normalized_actions and self.normalization_mapping.get("ACTION") != "MIN_MAX":
            raise ValueError(
                "VLA-JEPA output clipping requires MIN_MAX; otherwise disable it explicitly"
            )
        if set(self.lr_scales) != {"qwen", "action", "world"} or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            for v in self.lr_scales.values()
        ):
            raise ValueError("VLA-JEPA requires positive finite qwen/action/world LR scales")
        for name in ("action_noise_beta_alpha", "action_noise_beta_beta", "action_noise_s"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"VLA-JEPA {name} must be finite and positive")
        for name in ("action_dropout", "predictor_dropout"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError(f"VLA-JEPA {name} must be in [0,1)")
        if not math.isfinite(self.world_model_loss_weight) or self.world_model_loss_weight < 0:
            raise ValueError("VLA-JEPA world loss weight must be finite and nonnegative")
        for name in (
            "chunk_size",
            "n_action_steps",
            "num_video_frames",
            "jepa_tubelet_size",
            "num_action_tokens_per_timestep",
            "num_embodied_action_tokens_per_instruction",
            "num_inference_timesteps",
            "action_hidden_size",
            "action_num_layers",
            "action_max_seq_len",
            "predictor_depth",
            "predictor_num_heads",
            "repeated_diffusion_steps",
            "tokenizer_max_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"VLA-JEPA {name} must be a positive integer")
        if self.n_action_steps > self.chunk_size or not self.image_features:
            raise ValueError("VLA-JEPA requires RGB and n_action_steps <= chunk_size")
        if (
            not self.robot_state_feature
            or self.action_dim != self.action_feature.shape[0]
            or (self.state_dim != self.robot_state_feature.shape[0])
        ):
            raise ValueError("VLA-JEPA robot dimensions must match named dataset fields")
        if (
            self.num_video_frames % self.jepa_tubelet_size
            or self.num_video_frames < 2 * self.jepa_tubelet_size
        ):
            raise ValueError("VLA-JEPA video must contain at least two whole temporal tubelets")
        if self.num_video_frames // self.jepa_tubelet_size - 1 > self.chunk_size * 4:
            raise ValueError("VLA-JEPA temporal readouts exceed the action-token vocabulary")
        if self.freeze_qwen and self.enable_world_model:
            raise ValueError(
                "Frozen Qwen cannot learn JEPA readouts; explicitly disable world model"
            )
        if self.reinitialize_world_heads and not self.enable_world_model:
            raise ValueError("Cannot reset world-model heads when the world model is disabled")
        if self.enable_world_model:
            if (
                not self.jepa_config
                or self.jepa_config.get("tubelet_size", 2) != self.jepa_tubelet_size
            ):
                raise ValueError("VLA-JEPA requires matching local video encoder geometry")
            if not self.world_model_cameras:
                self.world_model_cameras = list(self.image_features)
            if len(set(self.world_model_cameras)) != len(self.world_model_cameras) or (
                set(self.world_model_cameras) - set(self.image_features)
            ):
                raise ValueError("VLA-JEPA world cameras must be distinct selected dataset cameras")

    @property
    def video_offsets(self):
        stride = max(1, (self.chunk_size - 1) // (self.num_video_frames - 1))
        return [i * stride for i in range(self.num_video_frames)]
