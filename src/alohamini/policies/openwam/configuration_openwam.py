"""OpenWAM-alpha architecture and explicit recorded-coordinate fine-tuning."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig

WAN_DEFAULTS = dict(
    has_image_input=False,
    patch_size=(1, 2, 2),
    in_dim=48,
    dim=3072,
    ffn_dim=14336,
    freq_dim=256,
    text_dim=4096,
    out_dim=48,
    num_heads=24,
    num_layers=30,
    eps=1e-6,
    seperated_timestep=True,
    require_clip_embedding=False,
    require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True,
)


@dataclass
class OpenWAMConfig(PolicyConfig):
    assets_path: str = ""
    wan: dict = field(default_factory=lambda: dict(WAN_DEFAULTS))
    text_encoder: dict = field(default_factory=dict)
    vae: dict = field(default_factory=dict)
    action_hidden_dim: int = 1024
    action_ffn_dim: int = 4096
    chunk_size: int = 32
    n_action_steps: int = 32
    video_stride: int = 4
    canvas_size: tuple = (384, 320)
    camera_layout: list = field(default_factory=list)
    action_representation: str = "recorded"
    num_steps: int = 10
    shift: float = 5.0
    lambda_video: float = 1.0
    lambda_action: float = 1.0
    gradient_checkpointing: bool = True
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MIN_MAX",
            "ACTION": "MIN_MAX",
        }
    )
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not self.assets_path or self.robot_state_feature is None or not self.image_features:
            raise ValueError("OpenWAM requires local assets, state and cameras")
        if self.action_representation != "recorded":
            raise ValueError(
                "Native OpenWAM fine-tuning uses recorded coordinates, not inferred 80D slots"
            )
        for name in ("chunk_size", "n_action_steps", "video_stride", "num_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_action_steps > self.chunk_size or self.chunk_size > 512:
            raise ValueError("OpenWAM requires n_action_steps <= chunk_size <= 512")
        if self.chunk_size % (4 * self.video_stride):
            raise ValueError("chunk_size must be divisible by 4 * video_stride for the causal VAE")
        self.canvas_size = tuple(self.canvas_size)
        if len(self.canvas_size) != 2 or any(
            type(n) is not int or n <= 0 or n % 32 for n in self.canvas_size
        ):
            raise ValueError("canvas_size must contain positive multiples of 32")
        if self.canvas_size[0] % 3:
            raise ValueError("L-shaped canvas height must be divisible by 3")
        if not self.camera_layout:
            self.camera_layout = list(self.image_features)
            self.camera_layout += [None] * (3 - len(self.camera_layout))
        present = [key for key in self.camera_layout if key is not None]
        if (
            len(self.camera_layout) != 3
            or len(set(present)) != len(present)
            or set(present) != set(self.image_features)
        ):
            raise ValueError(
                "camera_layout must assign each selected camera once to head/left/right"
            )
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.shift, self.lambda_video, self.lambda_action)
        ):
            raise ValueError("OpenWAM joint training requires positive shift and both loss weights")
        if not self.wan.get("fuse_vae_embedding_in_latents"):
            raise ValueError("OpenWAM-alpha requires the clean first-frame TI2V path")
