"""AC-DiT paper architecture and explicit native data contracts."""

import math
from dataclasses import dataclass, field

from alohamini.paths import WorkspacePaths
from alohamini.policies.configuration import NormalizationMode, PolicyConfig


@dataclass
class ACDiTConfig(PolicyConfig):
    stage: str = "full"
    recipe: str = "paper"
    assets_path: str = field(
        default_factory=lambda: str(WorkspacePaths().pretrained / "siglip-so400m-patch14-384")
    )
    load_pretrained_encoders: bool = True
    mobility_pretrained_path: str | None = None
    body_pretrained_path: str | None = None
    n_obs_steps: int = 2
    chunk_size: int = 2
    n_action_steps: int = 2
    padded_dim: int = 128
    hidden_dim: int = 2048
    num_layers: int = 28
    num_heads: int = 32
    mobility_hidden_dim: int = 1024
    mobility_num_layers: int = 14
    mobility_num_heads: int = 32
    num_train_timesteps: int = 1000
    num_inference_steps: int = 5
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "sample"
    clip_sample: bool = False
    tokenizer_max_length: int = 64
    use_pointcloud: bool = True
    pointcloud_field: str = "observation.point_cloud"
    pointcloud_frame: str = ""
    pointcloud_units: str = "m"
    pointcloud_points: int = 1024
    pointcloud_knn: int = 64
    pointcloud_stages: int = 3
    pointcloud_embed_dim: int = 96
    pointcloud_sampling: str = "random"
    pointcloud_projection_size: int = 224
    lora_rank: int = 2
    lora_alpha: float = 1.0
    lora_dropout: float = 0.25
    context_fields: tuple[str, ...] = ()
    context_dim: int = 0
    action_names: tuple[str, ...] = ()
    state_names: tuple[str, ...] = ()
    base_action_names: tuple[str, ...] = ("x.vel", "y.vel", "theta.vel")
    action_slot_map: dict[str, int] = field(default_factory=dict)
    state_slot_map: dict[str, int] = field(default_factory=dict)
    control_frequency: float = 30.0
    condition_mask_prob: float = 0.1
    state_noise_snr: float | None = 40.0
    image_augmentation: bool = True
    gradient_checkpointing: bool = False
    mask_padding_loss: bool = False
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        for key in (
            "load_pretrained_encoders",
            "use_pointcloud",
            "clip_sample",
            "image_augmentation",
            "gradient_checkpointing",
            "mask_padding_loss",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in ("context_fields", "action_names", "state_names", "base_action_names"):
            setattr(self, key, tuple(getattr(self, key)))
        if self.stage not in ("mobility", "full") or self.recipe not in ("paper", "released"):
            raise ValueError("AC-DiT stage must be mobility/full; recipe must be paper/released")
        for key in (
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "padded_dim",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "mobility_hidden_dim",
            "mobility_num_layers",
            "mobility_num_heads",
            "num_train_timesteps",
            "num_inference_steps",
            "tokenizer_max_length",
            "pointcloud_points",
            "pointcloud_knn",
            "pointcloud_stages",
            "pointcloud_embed_dim",
            "lora_rank",
            "pointcloud_projection_size",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for width, heads in (
            (self.hidden_dim, self.num_heads),
            (self.mobility_hidden_dim, self.mobility_num_heads),
        ):
            if width % heads or width % 4:
                raise ValueError("AC-DiT widths must divide into heads and four position channels")
        if (
            self.n_action_steps > self.chunk_size
            or self.num_inference_steps > self.num_train_timesteps
        ):
            raise ValueError("Action steps and inference steps exceed their configured horizons")
        if self.prediction_type not in ("sample", "epsilon"):
            raise ValueError("AC-DiT supports sample or epsilon diffusion prediction")
        if self.pointcloud_sampling not in ("random", "fps"):
            raise ValueError("pointcloud_sampling must be random or fps")
        if (
            self.pointcloud_points // 2** self.pointcloud_stages < 1
            or self.pointcloud_knn > self.pointcloud_points // 2 ** (self.pointcloud_stages - 1)
        ):
            raise ValueError("Point hierarchy needs enough points for its kNN neighborhoods")
        if self.pointcloud_embed_dim % 6:
            raise ValueError("Point geometry width must be divisible by six")
        if not math.isfinite(self.control_frequency) or self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if not 0 <= self.condition_mask_prob < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Condition mask and LoRA dropout must be in [0,1)")
        if self.state_noise_snr is not None and not math.isfinite(self.state_noise_snr):
            raise ValueError("state_noise_snr must be finite or null")
        if len(set(self.context_fields)) != len(self.context_fields) or self.context_dim < 0:
            raise ValueError("Context fields must be unique and have a nonnegative width")
        if not self.base_action_names or len(set(self.base_action_names)) != len(
            self.base_action_names
        ):
            raise ValueError("base_action_names must explicitly identify unique base controls")
        if self.input_features:
            if self.robot_state_feature is None or not self.image_features:
                raise ValueError("AC-DiT needs numeric state and at least one RGB camera")
            for names, slots, width in (
                (self.action_names, self.action_slot_map, self.action_feature.shape[0]),
                (self.state_names, self.state_slot_map, self.robot_state_feature.shape[0]),
            ):
                if len(names) != width or len(set(names)) != width or set(slots) != set(names):
                    raise ValueError(
                        "AC-DiT requires recorded state/action names and explicit slot maps"
                    )
                if len(set(slots.values())) != width or any(
                    type(i) is not int or not 0 <= i < self.padded_dim for i in slots.values()
                ):
                    raise ValueError(
                        "AC-DiT slot maps must select distinct padded-vector coordinates"
                    )
            if not set(self.base_action_names) <= set(self.action_names):
                raise ValueError("base_action_names are absent from the recorded action fields")
            if self.use_pointcloud:
                feature = self.input_features.get(self.pointcloud_field)
                if (
                    feature is None
                    or feature.type != "ENV"
                    or len(feature.shape) != 2
                    or feature.shape[-1] not in (3, 6)
                ):
                    raise ValueError("AC-DiT requires a recorded Nx3/Nx6 point cloud ENV field")
                if not self.pointcloud_frame or self.pointcloud_units != "m":
                    raise ValueError("Declare the point-cloud coordinate frame and meter units")
