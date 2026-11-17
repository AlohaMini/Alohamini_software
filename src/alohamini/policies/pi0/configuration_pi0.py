"""Pi0 uses the shared PaliGemma expert with continuous state conditioning."""

from dataclasses import asdict, dataclass, field

from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.pi05.configuration_pi05 import PI05Config, validate_dimensions
from alohamini.policies.pi05.finetuning import FINETUNING_MODES
from alohamini.policies.pi05.processor_pi05 import ARM_DELTA_DIMS, validate_camera_names


@dataclass(frozen=True)
class PI0Config(PI05Config):
    pi05: bool = False
    discrete_state_input: bool = False
    max_token_len: int = 48

    def __post_init__(self):
        if self.pi05 or self.discrete_state_input:
            raise ValueError("Pi0 requires continuous state input without timestep adaRMSNorm")
        validate_dimensions(self)


@dataclass
class PI0TrainingConfig(PolicyConfig):
    network: dict = field(default_factory=dict)
    tokenizer_path: str = ""
    camera_order: tuple | None = None
    n_action_steps: int = 50
    num_inference_steps: int = 10
    delta_dims: tuple = ARM_DELTA_DIMS
    finetuning_mode: str = "full"
    a_vision_mode: str = "frozen"
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: float = 16.0
    expert_lora_rank: int = 32
    expert_lora_alpha: float = 32.0
    flow_loss_weight: float = 1.0
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
        core = PI0Config(**self.network)
        self.network = asdict(core)
        if self.finetuning_mode not in FINETUNING_MODES or self.a_vision_mode not in (
            "frozen",
            "full",
        ):
            raise ValueError("Invalid Pi0 fine-tuning configuration")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (18,):
            raise ValueError("Pi0 AlohaMini adaptation requires named 18D state")
        if self.action_feature.shape != (18,) or core.action_dim < 18:
            raise ValueError("Pi0 AlohaMini adaptation requires named 18D actions")
        cameras = validate_camera_names(
            k.removeprefix("observation.images.") for k in self.image_features
        )
        self.camera_order = validate_camera_names(
            cameras if self.camera_order is None else self.camera_order
        )
        if set(cameras) != set(self.camera_order):
            raise ValueError("Pi0 camera_order differs from the dataset cameras")
        if not 1 <= self.n_action_steps <= core.action_horizon or self.num_inference_steps < 1:
            raise ValueError("Invalid Pi0 sampling steps or execution horizon")

    @property
    def training_recipe(self):
        return "flow"

    @property
    def span_floors(self):
        return None

    @property
    def fast_tokenizer_path(self):
        return ""
