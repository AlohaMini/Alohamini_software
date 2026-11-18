"""LingBot-VA shared Wan2.2 geometry and explicit robot/data protocol."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.slots import NamedSlots


@dataclass
class LingBotVAConfig(PolicyConfig):
    patch_size: tuple = (1, 2, 2)
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    in_channels: int = 48
    out_channels: int = 48
    action_dim: int = 30
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 14336
    num_layers: int = 30
    cross_attn_norm: bool = True
    eps: float = 1e-6
    rope_max_seq_len: int = 1024
    attn_mode: str = "flex"
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"
    wan_pretrained_path: str = ""
    text_encoder_device: str = "cpu"
    precompute_text_features: bool = True
    text_feature_cache_dir: str = ""
    text_feature_store_path: str = ""
    text_feature_cache_bytes: int = 256 * 1024 * 1024
    bootstrap_condition: str = "physical_zero"
    obs_cam_keys: list = field(default_factory=list)
    image_hflip: bool = False
    camera_layout: str = "width_concat"
    height: int = 128
    width: int = 128
    vae_spatial_factor: int = 16
    action_per_frame: int = 4
    frame_chunk_size: int = 4
    attn_window: int = 30
    num_inference_steps: int = 20
    video_exec_step: int = -1
    action_num_inference_steps: int = 50
    guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    snr_shift: float = 5.0
    action_snr_shift: float = 0.05
    max_sequence_length: int = 512
    save_predicted_video: bool = False
    return_predicted_video: bool = False
    action_names: list = field(default_factory=list)
    action_slots: dict = field(default_factory=dict)
    reinitialize_robot_heads: bool = True
    normalization_recipe: str = "recorded"
    normalization_mapping: dict = field(
        default_factory=lambda: dict(VISUAL="IDENTITY", STATE="IDENTITY", ACTION="QUANTILES")
    )
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "constant_with_warmup"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "num_attention_heads",
            "attention_head_dim",
            "in_channels",
            "out_channels",
            "action_dim",
            "text_dim",
            "freq_dim",
            "ffn_dim",
            "num_layers",
            "rope_max_seq_len",
            "height",
            "width",
            "vae_spatial_factor",
            "action_per_frame",
            "frame_chunk_size",
            "attn_window",
            "num_inference_steps",
            "action_num_inference_steps",
            "max_sequence_length",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"LingBot-VA {name} must be a positive integer")
        if not self.wan_pretrained_path or self.dtype not in ("float32", "bfloat16"):
            raise ValueError("LingBot-VA requires local frozen assets and float32/bfloat16 dtype")
        if self.bootstrap_condition not in ("physical_zero", "normalized_zero"):
            raise ValueError(
                "LingBot-VA bootstrap_condition must be physical_zero or normalized_zero"
            )
        if type(self.text_feature_cache_bytes) is not int or self.text_feature_cache_bytes < 0:
            raise ValueError("LingBot-VA feature cache budget must be nonnegative integer bytes")
        if self.attention_head_dim % 2 or (self.attention_head_dim // 3) % 2:
            raise ValueError("LingBot-VA rotary head partitions must all have even width")
        if self.attn_mode not in ("flex", "torch") or self.camera_layout not in (
            "width_concat",
            "robotwin_tshape",
        ):
            raise ValueError("LingBot-VA supports flex/torch attention and declared camera layouts")
        if not self.image_features or self.action_per_frame % 4:
            raise ValueError(
                "LingBot-VA needs cameras and action_per_frame divisible by VAE temporal factor4"
            )
        if not self.obs_cam_keys:
            self.obs_cam_keys = list(self.image_features)
        if len(set(self.obs_cam_keys)) != len(self.obs_cam_keys) or set(self.obs_cam_keys) != set(
            self.image_features
        ):
            raise ValueError("LingBot-VA camera order must cover exactly the selected cameras")
        if self.camera_layout == "robotwin_tshape" and len(self.obs_cam_keys) != 3:
            raise ValueError(
                "LingBot-VA T layout requires explicitly ordered head/left/right views"
            )
        if self.return_predicted_video:
            raise ValueError(
                "Native LingBot-VA returns commands; use save_predicted_video for diagnostics"
            )
        if (
            len(self.patch_size) != 3
            or self.patch_size[0] != 1
            or any(type(n) is not int or n < 1 for n in self.patch_size)
        ):
            raise ValueError("LingBot-VA requires a three-axis patch with temporal factor1")
        if self.attn_window // 2 * self.frame_chunk_size < self.frame_chunk_size + 1:
            raise ValueError("LingBot-VA cache must fit the first chunk and its conditioning frame")
        if self.height % self.vae_spatial_factor or self.width % self.vae_spatial_factor:
            raise ValueError("LingBot-VA image dimensions must divide into VAE latent cells")
        h, w = self.height // self.vae_spatial_factor, self.width // self.vae_spatial_factor
        if self.camera_layout == "robotwin_tshape":
            if h % 2 or w % 2:
                raise ValueError("LingBot-VA half-resolution views must fit complete VAE cells")
            h = h * 3 // 2
        else:
            w *= len(self.obs_cam_keys)
        if h % self.patch_size[1] or w % self.patch_size[2]:
            raise ValueError("LingBot-VA tiled latent dimensions must fit complete patches")
        if self.in_channels != self.out_channels:
            raise ValueError("LingBot-VA velocity output must match input latent channels")
        for name in ("eps", "snr_shift", "action_snr_shift"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"LingBot-VA {name} must be finite and positive")
        for name in ("guidance_scale", "action_guidance_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 1:
                raise ValueError(f"LingBot-VA {name} must be at least1;1 disables guidance")
        if type(self.video_exec_step) is not int or not (
            self.video_exec_step == -1 or 1 <= self.video_exec_step <= self.num_inference_steps + 1
        ):
            raise ValueError("LingBot-VA video_exec_step must be -1 or a valid denoising prefix")
        NamedSlots(self.action_names, self.action_slots, self.action_dim)
        if len(self.action_names) != self.action_feature.shape[0]:
            raise ValueError("LingBot-VA names must match actual command coordinates")
        if (
            self.normalization_recipe not in ("recorded", "lingbot")
            or self.normalization_mapping.get("ACTION") != "QUANTILES"
        ):
            raise ValueError("LingBot-VA requires recorded/lingbot quantile normalization")

    @property
    def chunk_size(self):
        return self.frame_chunk_size * self.action_per_frame

    @property
    def n_action_steps(self):
        return self.chunk_size

    @property
    def used_action_channel_ids(self):
        return [self.action_slots[name] for name in self.action_names]

    @property
    def quantile_options(self):
        return {"action": dict(epsilon=1e-6)}

    @property
    def normalization_clip(self):
        return {"action": [-1.5, 1.5]} if self.normalization_recipe == "lingbot" else {}
