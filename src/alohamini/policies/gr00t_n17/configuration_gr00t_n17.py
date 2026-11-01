"""Native GR00T N1.7 configuration with explicit local assets and robot slots."""

from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class GR00TN17Config(PolicyConfig):
    assets_path: str = ""
    backbone_config: dict = field(default_factory=dict)
    network: dict = field(default_factory=dict)
    chunk_size: int = 40
    n_action_steps: int = 40
    embodiment_id: int = 31
    reinitialize_embodiment: bool = True
    initialize_from_base: bool = True
    tune_llm: bool = False
    tune_visual: bool = False
    tune_top_llm_layers: int = 0
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True
    backbone_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    formalize_language: bool = True
    image_augmentation: bool = True
    crop_fraction: float = 0.95
    shortest_image_edge: int = 256
    use_percentiles: bool = True
    state_names: tuple = ()
    action_names: tuple = ()
    state_slot_map: dict = field(default_factory=dict)
    action_slot_map: dict = field(default_factory=dict)
    normalization_mapping: dict = field(
        default_factory=lambda: {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    )
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    optimizer_grad_clip_norm: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if not self.assets_path or not self.backbone_config:
            raise ValueError(
                "GR00T N1.7 requires local processor assets and an embedded Qwen3-VL config"
            )
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("GR00T N1.7 requires RGB cameras and named numeric state")
        for key in ("chunk_size", "n_action_steps", "shortest_image_edge"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.n_action_steps > self.chunk_size or not 0 < self.crop_fraction <= 1:
            raise ValueError("Invalid execution horizon or crop_fraction")
        if self.backbone_dtype not in ("float32", "bfloat16"):
            raise ValueError("backbone_dtype must be float32 or bfloat16")
        count = self.network.get("max_num_embodiments", 32)
        if type(self.embodiment_id) is not int or not 0 <= self.embodiment_id < count:
            raise ValueError("embodiment_id is outside the checkpoint's projector bank")
        if not 0 <= self.tune_top_llm_layers <= self.network.get("select_layer", 12):
            raise ValueError("tune_top_llm_layers exceeds the retained LLM layers")
        for names, slots, feature, maximum in (
            (self.state_names, self.state_slot_map, self.robot_state_feature, "max_state_dim"),
            (self.action_names, self.action_slot_map, self.action_feature, "max_action_dim"),
        ):
            if (
                len(names) != feature.shape[0]
                or len(set(names)) != len(names)
                or set(slots) != set(names)
            ):
                raise ValueError("GR00T requires ordered recorded names and explicit slot maps")
            if len(set(slots.values())) != len(names) or any(
                type(i) is not int or not 0 <= i < self.network.get(maximum, 132)
                for i in slots.values()
            ):
                raise ValueError("GR00T coordinate slots must be distinct and in range")
        text = self.backbone_config.get("text_config", {})
        if text.get("hidden_size") != self.network.get("backbone_embedding_dim", 2048):
            raise ValueError("Qwen3-VL hidden width differs from the action head's input width")
        for key in (
            "reinitialize_embodiment",
            "initialize_from_base",
            "tune_llm",
            "tune_visual",
            "tune_projector",
            "tune_diffusion_model",
            "tune_vlln",
            "gradient_checkpointing",
            "formalize_language",
            "image_augmentation",
            "use_percentiles",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
