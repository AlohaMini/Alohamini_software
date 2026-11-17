"""EVO1 stage geometry; compute precision is controlled by the platform."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import NormalizationMode, PolicyConfig


@dataclass
class Evo1Config(PolicyConfig):
    device: str = "cpu"
    attention_implementation: str = "sdpa"
    training_stage: str = "stage1"
    use_amp: bool = False
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 24
    max_action_dim: int = 24
    max_views: int = 3
    image_resolution: tuple[int, int] = (448, 448)
    empty_cameras: int = 0
    postprocess_action_dim: int | None = None
    binarize_gripper: bool = False
    gripper_index: int = 6
    gripper_threshold: float = 0.5
    gripper_below_threshold_value: float = 1.0
    gripper_above_threshold_value: float = -1.0
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    vlm_model_name: str = ""
    vlm_num_layers: int | None = 14
    dtype: str = "bfloat16"
    vlm_dtype: str | None = None
    max_text_length: int = 1024
    use_flash_attn: bool = False
    action_head: str = "flowmatching"
    embed_dim: int = 896
    hidden_dim: int = 1024
    state_hidden_dim: int = 1024
    num_heads: int = 8
    num_layers: int = 8
    dropout: float = 0.0
    num_inference_timesteps: int = 32
    num_categories: int = 1
    return_cls_only: bool = False
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    finetune_vlm: bool | None = None
    finetune_language_model: bool | None = None
    finetune_vision_model: bool | None = None
    finetune_action_head: bool | None = None
    task_field: str = "task"
    embodiment_id_field: str | None = None
    default_embodiment_id: int = 0
    rtc_config: dict | None = None
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 300

    def __post_init__(self):
        super().__post_init__()
        if self.training_stage not in ("stage1", "stage2"):
            raise ValueError("EVO1 training_stage must be stage1 or stage2")
        default = self.training_stage == "stage2"
        if self.finetune_vlm is None:
            self.finetune_vlm = default
        for key in ("finetune_language_model", "finetune_vision_model"):
            if getattr(self, key) is None:
                setattr(self, key, self.finetune_vlm)
        if self.finetune_action_head is None:
            self.finetune_action_head = True
        if self.dtype not in ("float32", "bfloat16") or self.use_amp or self.use_flash_attn:
            raise ValueError("EVO1 uses platform AMP and explicit sdpa/eager attention")
        if self.attention_implementation not in ("sdpa", "eager"):
            raise ValueError("EVO1 attention_implementation must be sdpa or eager")
        if self.rtc_config is not None or self.binarize_gripper:
            raise ValueError(
                "EVO1 native execution currently uses recorded gripper commands without RTC"
            )
        if not self.vlm_model_name:
            raise ValueError("EVO1 requires a local InternVL3 configuration/tokenizer")
        if self.vlm_dtype is not None:
            raise ValueError("Use EVO1 dtype instead of deprecated vlm_dtype")
        self.validate_features()

    def validate_features(self):
        if self.postprocess_action_dim not in (None, self.action_feature.shape[0]):
            raise ValueError("EVO1 output width must match the saved named action fields")
        if (
            self.robot_state_feature is None
            or self.robot_state_feature.shape[0] > self.max_state_dim
        ):
            raise ValueError("EVO1 state does not fit max_state_dim")
        if self.action_feature.shape[0] > self.max_action_dim or not self.image_features:
            raise ValueError("EVO1 requires camera observations and actions within max_action_dim")
        if (
            len(self.image_features) > self.max_views
            or not 1 <= self.n_action_steps <= self.chunk_size
        ):
            raise ValueError("EVO1 view count or execution horizon exceeds configuration")
