"""EO-1 model geometry and robot-only fine-tuning protocol."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class EO1Config(PolicyConfig):
    assets_path: str = ""
    vlm_config: dict = field(default_factory=dict)
    chunk_size: int = 16
    n_action_steps: int = 16
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_denoise_steps: int = 10
    num_action_layers: int = 2
    action_act: str = "linear"
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    supervise_padding_action_dims: bool = True
    supervise_padding_actions: bool = True
    mask_repeated_tail_actions: bool = True
    dtype: str = "bfloat16"
    force_fp32_autocast: bool = True
    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    tokenizer_max_length: int = 1000
    image_min_pixels: int = 64 * 28 * 28
    image_max_pixels: int = 128 * 28 * 28
    flow_loss_weight: float = 1.0
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.1
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "diffusers_cosine"
    scheduler_warmup_steps: int = 900
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if not self.vlm_config or not self.assets_path:
            raise ValueError("EO-1 requires local Qwen configuration and processor assets")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("EO-1 dtype must be float32 or bfloat16")
        if (
            self.robot_state_feature is None
            or self.robot_state_feature.shape[0] > self.max_state_dim
        ):
            raise ValueError("EO-1 state exceeds its declared state dimension")
        if not self.image_features or self.action_feature.shape[0] > self.max_action_dim:
            raise ValueError("EO-1 needs cameras and actions within max_action_dim")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("EO-1 requires 1 <= n_action_steps <= chunk_size")
        if self.attn_implementation not in ("sdpa", "eager"):
            raise ValueError("EO-1 currently supports sdpa or eager attention")
        if not self.supervise_padding_actions or not self.supervise_padding_action_dims:
            raise ValueError(
                "EO-1 native sample reduction currently uses the released padded-width objective"
            )
