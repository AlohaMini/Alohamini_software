"""WALL-OSS-0.5 geometry, named robot slots and flow fine-tuning settings."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.slots import NamedSlots


@dataclass
class WallOSS05Config(PolicyConfig):
    core_config: dict = field(default_factory=dict)
    assets_path: str = ""
    dtype: str = "bfloat16"
    chunk_size: int = 32
    n_action_steps: int = 32
    num_inference_timesteps: int = 10
    tokenizer_max_length: int = 1000
    image_long_edge: int = 448
    gradient_checkpointing: bool = True
    reinitialize_robot_heads: bool = True
    state_names: list = field(default_factory=list)
    action_names: list = field(default_factory=list)
    state_slots: dict = field(default_factory=dict)
    action_slots: dict = field(default_factory=dict)
    camera_key_mapping: dict = field(default_factory=dict)
    normalization_recipe: str = "recorded"
    normalization_mapping: dict = field(
        default_factory=lambda: dict(VISUAL="IDENTITY", STATE="QUANTILES", ACTION="QUANTILES")
    )
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "diffusers_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 200000
    scheduler_decay_lr: float = 1e-6

    def __post_init__(self):
        super().__post_init__()
        if not self.core_config or not self.assets_path:
            raise ValueError("WALL requires local processor assets and saved core_config")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("WALL dtype must be float32 or bfloat16")
        for name in (
            "chunk_size",
            "n_action_steps",
            "num_inference_timesteps",
            "tokenizer_max_length",
            "image_long_edge",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"WALL {name} must be a positive integer")
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("WALL requires named state and RGB cameras")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("WALL n_action_steps must not exceed chunk_size")
        if self.normalization_recipe not in ("recorded", "wall"):
            raise ValueError("WALL normalization_recipe must be recorded or wall")
        for kind in ("STATE", "ACTION"):
            if self.normalization_mapping.get(kind) != "QUANTILES":
                raise ValueError("WALL numeric recipes require QUANTILES")
        for names, slots, feature in (
            (self.state_names, self.state_slots, self.robot_state_feature),
            (self.action_names, self.action_slots, self.action_feature),
        ):
            NamedSlots(names, slots, 26)
            if len(names) != feature.shape[0]:
                raise ValueError("WALL named coordinates must match feature width")
        if not self.camera_key_mapping:
            self.camera_key_mapping = {
                key: key.removeprefix("observation.images.") for key in self.image_features
            }
        if set(self.camera_key_mapping) != set(self.image_features):
            raise ValueError("WALL camera mapping must cover exactly the selected cameras")
        if any(not isinstance(v, str) or not v.strip() for v in self.camera_key_mapping.values()):
            raise ValueError("WALL camera labels must be nonempty strings")
        if len(set(self.camera_key_mapping.values())) != len(self.camera_key_mapping):
            raise ValueError("WALL camera labels must be distinct")
        core = self.core_config
        if core.get("use_x_pred", False) or not core.get("use_x_loss", True):
            raise ValueError("WALL native flow path requires velocity output and x-prediction loss")
        if not core.get("causal_action_attention_mask", True):
            raise ValueError("WALL native flow path requires causal action attention")
        widths = core.get("dim_inputs", [])
        if len(widths) != 2 or widths[0] < widths[1] or widths[1] < 4:
            raise ValueError("WALL requires VLM and action-expert input widths")
        if core.get("action_hidden_size") != widths[1]:
            raise ValueError("WALL action_hidden_size must match its expert width")
        if widths[1] % 2:
            raise ValueError("WALL action expert width must be even for timestep embeddings")
        scheduler = core.get("noise_scheduler", {})
        for name, default in (("beta_alpha", 1.5), ("beta_beta", 1.0), ("time_shift", 1.0)):
            value = scheduler.get(name, default)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"WALL noise scheduler {name} must be finite and positive")
        scale = scheduler.get("s", 0.999)
        if not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale < 1:
            raise ValueError("WALL noise scheduler s must be inside (0,1)")
        experts = core.get("experts", [])
        if len(experts) != 2 or [v.get("hidden_size") for v in experts] != widths:
            raise ValueError("WALL expert dimensions must match dim_inputs")
        text = core.get("text_config", core)
        if text.get("hidden_size") != widths[0]:
            raise ValueError("WALL text geometry must match the VLM expert")

    @property
    def quantile_options(self):
        paper = self.normalization_recipe == "wall"
        return {
            key: dict(epsilon=0.0 if paper else 1e-6, clip_to_bounds=paper)
            for key, feature in {**self.input_features, **self.output_features}.items()
            if feature.type in ("STATE", "ACTION")
        }
