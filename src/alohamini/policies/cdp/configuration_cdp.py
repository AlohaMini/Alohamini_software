"""RGB Causal Diffusion Policy (CDP2) and rolling diffusion configuration."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class CDPConfig(PolicyConfig):
    n_obs_steps: int = 32
    horizon: int = 40  # Includes the historical action prefix.
    n_action_steps: int = 8
    hidden_dim: int = 256
    num_layers: int = 8
    num_heads: int = 4
    dropout: float = 0.3
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = None
    share_rgb_encoder: bool = False
    crop_shape: tuple[int, int] | None = (76, 76)
    num_train_timesteps: int = 100
    num_inference_steps: int = 100
    prediction_type: str = "sample"
    clip_sample: bool = True
    history_noise_scale: float = 0.02
    noise_regime_weights: tuple = (0.1, 0.0, 0.0, 0.9)
    do_mask_loss_for_padding: bool = False
    drop_n_last_frames: int = 0
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    optimizer_grad_clip_norm: float = 0.0
    vision_weight_decay: float = 1e-6
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX",
        }
    )

    @property
    def future_steps(self):
        return self.horizon - self.n_obs_steps

    def __post_init__(self):
        super().__post_init__()
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("CDP2 requires RGB cameras and observation.state")
        for key in (
            "n_obs_steps",
            "horizon",
            "n_action_steps",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "num_train_timesteps",
            "num_inference_steps",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_obs_steps % self.n_action_steps or self.future_steps < self.n_action_steps:
            raise ValueError(
                "CDP history must contain whole chunks and horizon must include a future chunk"
            )
        if self.future_steps % self.n_action_steps:
            raise ValueError("CDP future window must contain whole action chunks")
        chunks = self.future_steps // self.n_action_steps
        if self.num_train_timesteps % chunks:
            raise ValueError("Diffusion steps must be divisible by the number of future chunks")
        if self.num_inference_steps != self.num_train_timesteps:
            raise ValueError("CDP rolling DDPM requires num_inference_steps == num_train_timesteps")
        if self.hidden_dim < 4 or self.hidden_dim % 2 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be even, >= 4 and divisible by num_heads")
        if self.prediction_type not in ("sample", "epsilon"):
            raise ValueError("prediction_type must be sample or epsilon")
        if not 0 <= self.dropout < 1 or not 0 <= self.history_noise_scale < 1:
            raise ValueError("Invalid dropout or history_noise_scale")
        if (
            len(self.noise_regime_weights) != 4
            or min(self.noise_regime_weights) < 0
            or sum(self.noise_regime_weights) <= 0
        ):
            raise ValueError(
                "noise_regime_weights requires four nonnegative weights with positive sum"
            )
        if self.vision_backbone not in ("resnet18", "resnet34", "resnet50"):
            raise ValueError("Unsupported ResNet backbone")
        if self.drop_n_last_frames < 0:
            raise ValueError("drop_n_last_frames must be nonnegative")
        if self.crop_shape is not None:
            if len(self.crop_shape) != 2 or min(self.crop_shape) < 1:
                raise ValueError("crop_shape must contain two positive sizes")
            if any(
                any(c > s for c, s in zip(self.crop_shape, f.shape[1:], strict=True))
                for f in self.image_features.values()
            ):
                raise ValueError("crop_shape must fit the image dimensions")
