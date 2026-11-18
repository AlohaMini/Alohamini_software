"""DM0.5 local Gemma3 geometry and explicit numeric preprocessing protocol."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class DM05Config(PolicyConfig):
    core_config: dict = field(default_factory=dict)
    assets_path: str = ""
    dtype: str = "bfloat16"
    chunk_size: int = 50
    n_action_steps: int = 50
    diffusion_steps: int = 10
    tokenizer_max_length: int = 1024
    normalization_recipe: str = "recorded"
    add_state: bool = True
    freeze_vlm_embedding: bool = True
    llm_attn_implementation: str = "eager"
    vision_attn_implementation: str = "sdpa"
    action_attn_implementation: str = "sdpa"
    vlm_gradient_checkpointing: bool = True
    ae_gradient_checkpointing: bool = True
    ae_gradient_checkpointing_layers: int = 1
    normalization_mapping: dict = field(
        default_factory=lambda: dict(
            VISUAL="IDENTITY",
            STATE="QUANTILES",
            ACTION="QUANTILES",
        )
    )
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "diffusers_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("DM05 dtype must be float32 or bfloat16")
        if self.normalization_recipe not in ("recorded", "opendm"):
            raise ValueError("DM05 normalization_recipe must be recorded or opendm")
        if not self.assets_path or not self.core_config:
            raise ValueError("DM05 requires local Gemma3 assets and saved core_config")
        for key in (
            "chunk_size",
            "n_action_steps",
            "diffusion_steps",
            "tokenizer_max_length",
            "ae_gradient_checkpointing_layers",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"DM05 {key} must be a positive integer")
        if self.n_action_steps > self.chunk_size or not self.image_features:
            raise ValueError("DM05 requires camera views and n_action_steps <= chunk_size")
        for key in ("vlm_config", "action_config"):
            if not isinstance(self.core_config.get(key), dict):
                raise ValueError("DM05 nested model configurations must be serialized dictionaries")
        text = self.core_config["vlm_config"].get("text_config", {})
        expert = self.core_config["action_config"]
        if text.get("num_hidden_layers") != expert.get("num_hidden_layers"):
            raise ValueError("DM05 VLM and action-expert layer counts must match")
        width = self.core_config.get("action_dim", 32)
        if self.action_feature.shape[0] > width or self.robot_state_feature is None:
            raise ValueError("DM05 requires named state and actions within the core action width")
        if any(self.normalization_mapping.get(kind) != "QUANTILES" for kind in ("STATE", "ACTION")):
            raise ValueError("DM05 normalization recipes require QUANTILES for state/actions")

    @property
    def quantile_options(self):
        paper = self.normalization_recipe == "opendm"
        return {
            key: dict(epsilon=1e-6, clip_to_bounds=paper, zero_if_zero_bounds=paper)
            for key, feature in {**self.input_features, **self.output_features}.items()
            if feature.type in ("STATE", "ACTION")
        }
