"""Xiaomi-Robotics-1 model options and frozen robot protocol."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class XiaomiConfig(PolicyConfig):
    assets_path: str = ""
    vlm_config: dict = field(default_factory=dict)
    kinematic_fields: dict = field(default_factory=dict)
    chunk_size: int = 30
    n_action_steps: int = 30
    hidden_size: int = 1024
    num_layers: int | None = None
    training_repeat: int = 4
    num_steps: int = 5
    freq_coefficient: float = 1.0
    async_train: bool = False
    gradient_checkpointing: bool = True
    dtype: str = "float32"
    attention_implementation: str = "sdpa"
    image_max_pixels: int = 160000
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_weight_decay: float = 0.01
    optimizer_eps: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if not 1 <= self.n_action_steps <= self.chunk_size <= 60:
            raise ValueError("Xiaomi requires 1 <= n_action_steps <= chunk_size <= 60")
        if self.training_repeat < 1 or self.num_steps < 1:
            raise ValueError("training_repeat and num_steps must be positive")
        if self.async_train:
            raise ValueError(
                "The platform Xiaomi recipe uses official synchronous training; "
                "asynchronous prefix execution is not yet supported"
            )
        if not self.assets_path or not self.vlm_config or not self.kinematic_fields:
            raise ValueError(
                "Xiaomi requires local Qwen assets, model config and calibrated kinematic fields"
            )
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if len(self.image_features) != 3:
            raise ValueError("Xiaomi requires ordered ego, left wrist and right wrist cameras")
