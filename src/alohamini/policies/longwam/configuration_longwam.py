"""Long-WAM architecture and explicit recorded-coordinate adaptation."""

from dataclasses import dataclass, field

import torch

from alohamini.policies.configuration import PolicyConfig


@dataclass
class LongWAMConfig(PolicyConfig):
    past_obs_steps: int = 48
    video_stride: int = 4
    future_video_steps: int = 32
    action_horizon: int = 32
    inference_action_horizon: int | None = None
    n_action_steps: int = 10
    denoising: str = "idm"
    num_inference_steps: int = 20
    imagine_sigma: float = 0.9
    imagine_steps: int = 4
    flow_shift: float = 5.0
    inference_seed: int | None = 1
    camera_keys: list[str] = field(default_factory=list)
    camera_size: tuple = (224, 224)
    action_dim: int = 7
    proprio_dim: int = 8
    dtype: str = "bfloat16"
    tokenizer_path: str = ""
    tokenizer_max_length: int = 128
    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    video_config: dict = field(default_factory=dict)
    vae_config: dict = field(default_factory=dict)
    text_encoder_config: dict = field(default_factory=dict)
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    use_gradient_checkpointing: bool = True
    reinitialize_embodiment_heads: bool = False
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX",
        }
    )
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.0
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_grad_clip_norm: float = 1.0
    loss_lambda_video: float = 1.0
    loss_lambda_action: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.video_stride != 4 or self.past_obs_steps < 0 or self.past_obs_steps % 16:
            raise ValueError("Long-WAM history must be a nonnegative multiple of 16; stride=4")
        if self.future_video_steps <= 0 or self.future_video_steps % 16:
            raise ValueError("Long-WAM future_video_steps must be a positive multiple of 16")
        if self.action_horizon < 1 or not 1 <= self.n_action_steps <= self.prediction_steps:
            raise ValueError("Invalid Long-WAM action horizon")
        if self.denoising not in ("idm", "cod"):
            raise ValueError("denoising must be idm or cod")
        if not 0 <= self.imagine_sigma <= 1 or (
            self.denoising == "cod" and self.imagine_sigma != 0
        ):
            raise ValueError("imagine_sigma must be in [0,1]; COD requires zero")
        if min(self.num_inference_steps, self.imagine_steps, self.flow_shift) <= 0:
            raise ValueError("Sampling steps and flow shift must be positive")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("Long-WAM dtype must be float32 or bfloat16")
        if not self.camera_keys or list(self.image_features) != list(self.camera_keys):
            raise ValueError("Long-WAM cameras must match their explicit checkpoint order")
        if any(n <= 0 or n % 16 for n in self.camera_size):
            raise ValueError("Long-WAM camera_size must be divisible by 16")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (
            self.proprio_dim,
        ):
            raise ValueError("Long-WAM proprio_dim differs from selected state")
        if self.action_feature.shape != (self.action_dim,):
            raise ValueError("Long-WAM action_dim differs from selected action fields")
        if not all((self.video_config, self.vae_config, self.text_encoder_config)):
            raise ValueError("Long-WAM requires saved video, VAE and text encoder configurations")
        if self.normalization_mapping != {
            "VISUAL": "IDENTITY",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX",
        }:
            raise ValueError("Long-WAM recorded-coordinate adaptation uses limits normalization")

    @property
    def torch_dtype(self):
        return getattr(torch, self.dtype)

    @property
    def prediction_steps(self):
        return (
            self.action_horizon
            if self.inference_action_horizon is None
            else self.inference_action_horizon
        )

    @property
    def observation_delta_indices(self):
        return list(range(-self.past_obs_steps, self.future_video_steps + 1, self.video_stride))
