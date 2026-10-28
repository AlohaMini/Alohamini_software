"""HyVLA network settings and the platform's recorded-action adaptation."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class HyVLA05Config(PolicyConfig):
    assets_path: str = ""
    network: dict = field(default_factory=dict)
    chunk_size: int = 20
    n_action_steps: int = 20
    history_frames: int = 6
    history_stride: int = 15
    action_representation: str = "recorded"
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "warmup_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 40000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if not self.assets_path or not self.network.get("vlm_config_dict"):
            raise ValueError("HyVLA requires local tokenizer assets and an embedded VLM config")
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("HyVLA requires cameras and observation.state")
        if self.action_representation != "recorded":
            raise ValueError("HyVLA native adaptation requires recorded action coordinates")
        for key in ("chunk_size", "n_action_steps", "history_frames", "history_stride"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must not exceed chunk_size")
        for feature, maximum in (
            (self.robot_state_feature, "max_state_dim"),
            (self.action_feature, "max_action_dim"),
        ):
            if feature.shape[0] > self.network.get(maximum, 32):
                raise ValueError(f"Recorded feature width exceeds HyVLA {maximum}")
        if self.history_frames > 1 and not self.network.get("use_video_encoder", False):
            raise ValueError("Historical images require the MEM video encoder")
        if self.history_frames > 18:
            raise ValueError("Released MEM supports at most 18 history frames")
        if self.network.get("attention_implementation", "eager") != "eager":
            raise ValueError("The native HyVLA adapter uses the eager shared-attention path")
