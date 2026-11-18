# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for native Alohamini integration in 2026.
"""G0.5 geometry, named robot slots and saved data/precision settings."""

import math
from copy import deepcopy
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.slots import NamedSlots

from .templates import make_g05_cot_prompt_template, make_g05_prompt_template


@dataclass
class G05Config(PolicyConfig):
    author_model_config: dict = field(default_factory=dict)
    assets_path: str = ""
    dtype: str = "bfloat16"
    embodiment: str = "recorded"
    data_protocol: str = "recorded"
    action_head: str = "flow"
    train_flow_actions: bool = True
    train_discrete_actions: bool = False
    joint_training: bool = True
    num_flow_samples: int = 8
    num_inference_steps: int = 10
    flow_loss_weight: float = 1.0
    text_loss_weight: float = 0.0
    predict_cot: bool = False
    runtime_system: str = "system1"
    runtime_cot_fields: tuple = ("subtask",)
    proprio_dropout_p: float | None = None
    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 16
    state_names: list = field(default_factory=list)
    action_names: list = field(default_factory=list)
    state_slots: dict = field(default_factory=dict)
    action_slots: dict = field(default_factory=dict)
    reinitialize_robot_heads: bool = True
    camera_order: list = field(default_factory=list)
    camera_sizes: dict = field(default_factory=dict)
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)
    quantize_rgb_before_resize: bool = True
    gradient_checkpointing: bool = False
    freeze_vision: bool = False
    freeze_vlm: bool = False
    use_stepwise_action_norm: bool = True
    normalization_windows: dict = field(default_factory=dict)
    normalization_mapping: dict = field(
        default_factory=lambda: dict(VISUAL="IDENTITY", STATE="QUANTILES", ACTION="QUANTILES")
    )
    normalization_clip: dict = field(default_factory=dict)
    use_language_recipe: bool = False
    recipe: dict = field(default_factory=dict)
    prompt_template: str = ""
    processor_metadata: dict = field(default_factory=dict)
    optimizer_lr: float = 1e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    optimizer_backbone_lr_multiplier: float = 1.0
    optimizer_vision_lr_multiplier: float = 1.0
    optimizer_apply_decay_on_norm_and_bias: bool = False
    scheduler_type: str = "constant_with_warmup"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 100000
    scheduler_decay_lr: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if not self.assets_path or not self.author_model_config:
            raise ValueError("G0.5 requires local assets and the resolved author_model_config")
        self.author_model_config = deepcopy(self.author_model_config)
        self.author_model_config["fm"].update(
            joint_training=self.joint_training,
            num_flow_samples=self.num_flow_samples,
            horizon_steps=self.chunk_size,
            num_inference_steps=self.num_inference_steps,
            fm_weight=self.flow_loss_weight,
        )
        self.author_model_config.setdefault("ar", {})["ce_weight"] = self.text_loss_weight
        if self.data_protocol not in ("recorded", "g05"):
            raise ValueError("G0.5 data_protocol must be recorded or g05")
        if self.data_protocol == "g05":
            self.normalization_clip.setdefault("observation.state", [-5.0, 5.0])
            self.normalization_clip.setdefault("action", [-5.0, 5.0])
        for name in ("flow_loss_weight", "text_loss_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"G0.5 {name} must be finite and nonnegative")
        if not self.train_flow_actions and not self.train_discrete_actions:
            raise ValueError("G0.5 needs an enabled action training head")
        if self.dtype not in ("float32", "bfloat16") or self.action_head not in (
            "flow",
            "actioncodec",
        ):
            raise ValueError("G0.5 supports float32/bfloat16 and flow/actioncodec heads")
        for name in (
            "n_obs_steps",
            "chunk_size",
            "n_action_steps",
            "num_flow_samples",
            "num_inference_steps",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"G0.5 {name} must be a positive integer")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("G0.5 execution horizon exceeds its predicted chunk")
        if self.robot_state_feature is None:
            raise ValueError("G0.5 requires recorded or explicitly transformed state")
        if (
            len(self.action_names) != self.action_feature.shape[0]
            or len(self.state_names) != self.robot_state_feature.shape[0]
        ):
            raise ValueError("G0.5 coordinate names must match its actual numeric inputs/outputs")
        NamedSlots(self.action_names, self.action_slots, self.policy_action_dim)
        NamedSlots(self.state_names, self.state_slots, self.policy_state_dim)
        expert, vlm = (self.author_model_config[key] for key in ("action_expert", "vlm"))
        if any(
            expert[key] != vlm[key]
            for key in ("num_hidden_layers", "num_key_value_heads", "head_dim")
        ):
            raise ValueError("G0.5 expert and VLM must have matching cache geometry")
        if (
            expert["input_dim"] != self.policy_action_dim
            or expert["output_dim"] != self.policy_action_dim
        ):
            raise ValueError("G0.5 action heads must match the configured flow width")
        if self.runtime_system not in ("system1", "system2") or (
            self.runtime_system == "system2" and not self.predict_cot
        ):
            raise ValueError("G0.5 System2 requires a CoT-capable configuration")
        if self.action_head == "actioncodec" and not self.train_discrete_actions:
            raise ValueError("G0.5 ActionCodec inference requires train_discrete_actions=true")
        if self.action_head == "flow" and not self.train_flow_actions:
            raise ValueError("G0.5 flow inference requires train_flow_actions=true")
        if (
            not self.joint_training
            and self.reinitialize_robot_heads
            and not (self.train_discrete_actions or self.predict_cot)
        ):
            raise ValueError(
                "Fresh G0.5 proprio heads need joint flow gradients or text supervision"
            )
        if self.train_discrete_actions and self.reinitialize_robot_heads:
            raise ValueError(
                "G0.5 pretrained ActionCodec requires explicit compatible robot slots and heads"
            )
        encoder = self.author_model_config.get("proprio_encoder", "mlp")
        mode = encoder.get("mode", "mlp") if isinstance(encoder, dict) else encoder
        if mode not in ("mlp", "mlp_dropout"):
            raise ValueError("Native G0.5 requires an MLP proprioception checkpoint")
        if self.proprio_dropout_p is None:
            self.proprio_dropout_p = (
                (encoder.get("dropout_p", 0.2) if isinstance(encoder, dict) else 0.2)
                if mode == "mlp_dropout"
                else 0.0
            )
        if not 0 <= self.proprio_dropout_p <= 1:
            raise ValueError("G0.5 proprio dropout must be in [0,1]")
        if self.n_obs_steps > 1 and self.author_model_config["vision"].get("temporal_freq", 0) <= 0:
            raise ValueError("Multi-frame G0.5 needs a temporal vision checkpoint")
        if not self.camera_order:
            self.camera_order = list(self.image_features)
        if (
            not self.camera_order
            or len(set(self.camera_order)) != len(self.camera_order)
            or set(self.camera_order) != set(self.image_features)
        ):
            raise ValueError("G0.5 camera order must cover exactly the selected views")
        if set(self.camera_sizes) != set(self.camera_order):
            raise ValueError("G0.5 requires an explicit size for every selected camera")
        vision = self.author_model_config["vision"]
        patch = vision["patch_size"] * vision["spatial_merge_size"]
        if any(
            len(size) != 2 or any(type(n) is not int or n < patch or n % patch for n in size)
            for size in self.camera_sizes.values()
        ):
            raise ValueError("G0.5 camera sizes must fit whole merged vision patches")
        if (
            len(self.image_mean) != 3
            or len(self.image_std) != 3
            or any(not math.isfinite(n) for n in (*self.image_mean, *self.image_std))
            or any(n <= 0 for n in self.image_std)
        ):
            raise ValueError("G0.5 image mean/std must contain three finite channels")
        if self.use_stepwise_action_norm:
            declared = self.normalization_windows.get("action", list(range(self.chunk_size)))
            if declared != list(range(self.chunk_size)):
                raise ValueError("G0.5 temporal action normalization must match its future horizon")
            self.normalization_windows["action"] = list(range(self.chunk_size))
        if not self.prompt_template:
            self.prompt_template = (
                make_g05_cot_prompt_template(
                    self.num_prompt_images,
                    fields=tuple(self.runtime_cot_fields),
                    flow_only=not self.train_discrete_actions,
                )
                if self.predict_cot
                else make_g05_prompt_template(
                    self.num_prompt_images,
                    predict_cot=False,
                    flow_only=not self.train_discrete_actions,
                )
            )
        if self.train_discrete_actions and "<action_action" not in self.prompt_template:
            raise ValueError("G0.5 discrete training requires an ActionCodec target template")
        if self.use_language_recipe:
            from alohamini.datasets.recipe import TrainingRecipe

            recipe = TrainingRecipe.from_dict(self.recipe)
            if recipe.referenced_cameras() - set(self.image_features):
                raise ValueError("G0.5 language recipe references unselected camera views")

    @property
    def policy_action_dim(self):
        return self.author_model_config["fm"]["action_dim"]

    @property
    def policy_state_dim(self):
        return self.author_model_config["proprio_dim"]

    @property
    def num_input_images(self):
        return len(self.camera_order) * self.n_obs_steps

    @property
    def num_prompt_images(self):
        return len(self.camera_order)

    @property
    def continuous_action(self):
        return self.train_flow_actions

    @property
    def discrete_action(self):
        return self.train_discrete_actions
