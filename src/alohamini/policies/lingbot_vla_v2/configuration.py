"""Native LingBot configuration; model fields follow the pinned upstream port."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig

from .semantic import JOINTS, validate_layout


@dataclass
class LingBotConfig(PolicyConfig):
    n_obs_steps: int = 1
    chunk_size: int = 50  # action_horizon in LingBot-VLA
    n_action_steps: int = 50

    max_action_dim: int = 55
    max_state_dim: int = 55

    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )

    state_slots: dict | None = None
    action_slots: dict | None = None
    use_relative_actions: bool = True
    relative_exclude_joints: list[str] = field(default_factory=lambda: ["gripper"])
    action_feature_names: list[str] | None = None

    tokenizer_path: str = ""
    tokenizer_max_length: int = 72

    resize_imgs_with_padding: tuple[int, int] = (256, 256)
    image_max_pixels: int = 65536
    image_min_pixels: int = 65536
    num_steps: int = 10
    dtype: str = "float32"
    canonical_joints: dict[str, int] = field(default_factory=lambda: dict(JOINTS))

    qwen3vl_use_vision_boundaries: bool = True
    precompute_grid_thw: bool = False

    expert_hidden_size: int = 768
    expert_intermediate_size: int = 2752
    action_num_attention_heads: int = 32
    action_num_key_value_heads: int = 8
    action_head_dim: int = 128
    action_fp32: bool = False

    use_moe: bool = True
    token_moe_layers: list = field(default_factory=lambda: list(range(36)))
    token_num_experts: int = 32
    token_top_k: int = 4
    token_moe_intermediate_size: int = 512
    token_shared_intermediate_size: int = 704
    sequence_wise_loss_coeff: float = 1e-3
    sequence_wise_mode: str = "per_sequence"
    router_z_loss_coeff: float = 1e-4
    router_activation: str = "sigmoid"
    routed_scaling_factor: float = 4.0
    use_shared_expert_gate: bool = False
    moe_implementation: str = "fused"

    attention_implementation: str = "sdpa"
    vit_attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = False
    compile_predict_velocity: bool = False
    compile_predict_velocity_mode: str = "default"
    use_cudagraph_denoise: bool = False
    use_cudagraph_prefix: bool = False
    use_cudagraph_prefix_full: bool = False
    rtc_config: dict | None = None
    moe_metrics_interval: int = 50
    use_cache: bool = True
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_state_proj: bool = True
    vlm_causal: bool = True
    vocab_size: int = 0
    use_lm_head: bool = False
    loss_type: str = "L1_fm"

    adanorm_time: bool = True
    final_norm_adanorm: bool = False

    align_params: dict = field(default_factory=dict)

    use_moe_expert_lr: bool = True
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0

    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-5  # constant lr schedule (decay_lr == peak_lr)

    layout: dict = field(default_factory=dict)
    dataset_fps: float = 30.0
    teacher_root: str = ""
    text_tokenizer_path: str = ""
    image_processor_path: str = ""
    lift_mode: str = "hold"

    def __post_init__(self):
        super().__post_init__()
        if self.n_obs_steps != 1 or self.num_steps < 1:
            raise ValueError(
                "LingBot requires one current observation and positive denoising steps"
            )
        if self.normalization_mapping != {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }:
            raise ValueError("LingBot's named-coordinate protocol requires mean/std normalization")
        if self.relative_exclude_joints != ["gripper"] or self.action_feature_names is not None:
            raise ValueError("Use the saved named layout for LingBot relative-action coordinates")
        if not 1 <= self.n_action_steps <= self.chunk_size:
            raise ValueError("Require 1 <= n_action_steps <= chunk_size")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if self.rtc_config is not None:
            raise ValueError("RTC guidance is not yet part of the platform LingBot executor")
        if self.attention_implementation not in ("sdpa", "eager"):
            raise ValueError("LingBot requires sdpa or eager attention")
        if self.lift_mode not in ("hold", "reserved"):
            raise ValueError("lift_mode must be hold or reserved")
        if not self.tokenizer_path or not self.layout:
            raise ValueError("LingBot requires local Qwen assets and an explicit semantic layout")
        validate_layout(self.layout)
        if list(self.canonical_joints.items()) != list(JOINTS):
            raise ValueError("LingBot canonical slot order must match the released upstream layout")
        if self.state_slots is not None or self.action_slots is not None:
            raise ValueError(
                "Platform LingBot uses the saved named layout instead of LeRobot slot overrides"
            )
        if (
            self.layout["lift_mode"] != self.lift_mode
            or bool(self.layout["relative"]) != self.use_relative_actions
        ):
            raise ValueError("LingBot action modes differ from the saved semantic layout")
        if (
            self.robot_state_feature is None
            or self.robot_state_feature.shape != (18,)
            or self.action_feature.shape != (18,)
        ):
            raise ValueError(
                "LingBot AlohaMini preset requires the complete named 18D state/action"
            )
        if self.max_action_dim != 55 or self.max_state_dim != 55 or len(self.image_features) != 3:
            raise ValueError("LingBot requires the released 55D boundary and three ordered cameras")
