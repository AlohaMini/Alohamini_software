"""DiT-Block configuration (SudeepDasari/dit-policy)."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class DiTPolicyConfig(PolicyConfig):
    n_obs_steps: int = 1
    horizon: int = 100
    n_action_steps: int = 100
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 2048
    time_dim: int = 256
    dropout: float = 0.1
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    share_rgb_encoder: bool = False
    pooled_vision: bool = False
    crop_shape: tuple[int, int] | None = (224, 224)
    random_resized_crop: bool = True
    gaussian_blur: bool = True
    num_train_timesteps: int = 100
    num_inference_steps: int = 8
    clip_sample: bool = True
    drop_n_last_frames: int = 0
    optimizer_lr: float = 3e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    optimizer_grad_clip_norm: float = 0.0
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "MEAN_STD",
            "STATE": "IDENTITY",
            "ACTION": "MIN_MAX",
        }
    )

    def __post_init__(self):
        super().__post_init__()
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("dit_policy requires RGB cameras and observation.state")
        for key in (
            "n_obs_steps",
            "horizon",
            "n_action_steps",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "feedforward_dim",
            "num_train_timesteps",
            "num_inference_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps exceeds horizon")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if self.time_dim < 4 or self.time_dim % 2:
            raise ValueError("time_dim must be even and >= 4")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 1 <= self.num_inference_steps <= self.num_train_timesteps:
            raise ValueError("Invalid num_inference_steps")
        if self.vision_backbone not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError("Unsupported ResNet backbone")
        if self.drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be nonnegative")
        if self.crop_shape is not None:
            if len(self.crop_shape) != 2 or min(self.crop_shape) < 1:
                raise ValueError("crop_shape must contain two positive sizes")
