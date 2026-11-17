"""OpenPI FAST geometry; no flow-matching action expert is allocated."""

from dataclasses import dataclass

from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.pi05.configuration_pi05 import get_config
from alohamini.policies.pi05.processor_pi05 import ARM_DELTA_DIMS, validate_camera_names


@dataclass
class PI0FastConfig(PolicyConfig):
    paligemma_variant: str = "gemma_2b"
    dtype: str = "bfloat16"
    chunk_size: int = 50
    n_action_steps: int = 50
    tokenizer_path: str = ""
    fast_tokenizer_path: str = ""
    base_weights_prefix: str = "model.paligemma_with_expert."
    max_token_len: int = 200
    fast_max_tokens: int = 256
    max_decoding_steps: int = 256
    ce_chunk_size: int = 32
    camera_order: tuple | None = None
    delta_dims: tuple = ARM_DELTA_DIMS
    gradient_checkpointing: bool = True
    freeze_vision_encoder: bool = False
    evaluate_action_predictions: bool = False
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
        get_config(self.paligemma_variant)
        if not isinstance(self.base_weights_prefix, str):
            raise ValueError("FAST base_weights_prefix must be a string")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("FAST dtype must be float32 or bfloat16")
        for key in (
            "chunk_size",
            "n_action_steps",
            "max_token_len",
            "fast_max_tokens",
            "max_decoding_steps",
            "ce_chunk_size",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("FAST n_action_steps cannot exceed chunk_size")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (18,):
            raise ValueError("FAST AlohaMini preset requires named 18D state")
        if self.action_feature.shape != (18,):
            raise ValueError("FAST AlohaMini preset requires named 18D actions")
        cameras = validate_camera_names(
            key.removeprefix("observation.images.") for key in self.image_features
        )
        self.camera_order = validate_camera_names(
            cameras if self.camera_order is None else self.camera_order
        )
        if set(cameras) != set(self.camera_order):
            raise ValueError("FAST camera_order differs from selected dataset cameras")
        for key in (
            "gradient_checkpointing",
            "freeze_vision_encoder",
            "evaluate_action_predictions",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
