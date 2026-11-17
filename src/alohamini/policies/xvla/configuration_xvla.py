"""X-VLA recorded-coordinate adaptation with explicit architecture and domain ID."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class XVLAConfig(PolicyConfig):
    assets_path: str = ""
    florence_config: dict = field(default_factory=dict)
    dtype: str = "float32"
    chunk_size: int = 30
    n_action_steps: int = 30
    tokenizer_max_length: int = 64
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_domains: int = 30
    default_domain_id: int = 0
    len_soft_prompts: int = 32
    dim_time: int = 32
    max_len_seq: int = 512
    use_hetero_proj: bool = False
    action_mode: str = "auto"
    num_denoising_steps: int = 10
    use_proprio: bool = True
    max_state_dim: int = 20
    max_action_dim: int = 20
    freeze_vision_encoder: bool = False
    freeze_language_encoder: bool = False
    train_policy_transformer: bool = True
    train_soft_prompts: bool = True
    gradient_checkpointing: bool = True
    normalization_mapping: dict = field(
        default_factory=lambda: dict(
            VISUAL="IDENTITY",
            STATE="MEAN_STD",
            ACTION="MEAN_STD",
        )
    )
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    learning_coef: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        for key in (
            "chunk_size",
            "n_action_steps",
            "depth",
            "num_heads",
            "hidden_size",
            "num_domains",
            "max_action_dim",
            "max_state_dim",
            "dim_time",
            "max_len_seq",
            "tokenizer_max_length",
            "num_denoising_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"X-VLA {key} must be a positive integer")
        if not math.isfinite(self.learning_coef) or self.learning_coef <= 0:
            raise ValueError("X-VLA learning_coef must be finite and positive")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("X-VLA dtype must be float32 or bfloat16")
        if not self.florence_config or not self.assets_path:
            raise ValueError("X-VLA requires local Florence configuration, tokenizer and processor")
        if self.action_mode != "auto":
            raise ValueError("Native X-VLA currently uses named recorded-coordinate auto actions")
        if not 1 <= self.n_action_steps <= self.chunk_size or self.num_denoising_steps < 1:
            raise ValueError("Invalid X-VLA execution horizon or denoising steps")
        if not self.image_features or self.action_feature.shape[0] > self.max_action_dim:
            raise ValueError("X-VLA requires views and actions within max_action_dim")
        if self.use_proprio and (
            self.robot_state_feature is None
            or self.robot_state_feature.shape[0] > self.max_state_dim
        ):
            raise ValueError("X-VLA proprioception does not fit max_state_dim")
        if not 0 <= self.default_domain_id < self.num_domains:
            raise ValueError("X-VLA default_domain_id must select a configured soft-prompt domain")
        if self.depth < 1 or self.hidden_size % self.num_heads:
            raise ValueError("Invalid X-VLA action Transformer depth or attention geometry")

    @property
    def torch_dtype(self):
        import torch

        return getattr(torch, self.dtype)

    def get_florence_config(self):
        from transformers import Florence2Config

        from .florence import _translate_vision_config

        config = dict(self.florence_config)
        config["vision_config"] = _translate_vision_config(config["vision_config"])
        text = dict(config["text_config"])
        if text.get("model_type") == "florence2_language":
            text["model_type"] = "bart"
        config["text_config"] = text
        return Florence2Config(**config)
