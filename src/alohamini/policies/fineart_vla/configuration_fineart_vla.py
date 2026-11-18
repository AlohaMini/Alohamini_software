"""FineART-VLA geometry and robot adaptation; source revisions in SOURCE.json."""

import math
from dataclasses import asdict, dataclass, field

from alohamini.datasets.recipe import TrainingRecipe
from alohamini.policies.configuration import PolicyConfig
from alohamini.policies.pi05.configuration_pi05 import PI05Config


def default_recipe():
    return {
        "blend": {
            "high_level_subtask": {
                "weight": 0.30,
                "messages": [
                    {"role": "user", "content": "${task}", "stream": "high_level"},
                    {
                        "role": "assistant",
                        "content": "${subtask}",
                        "stream": "high_level",
                        "target": True,
                        "if_present": "subtask",
                    },
                ],
            },
            "low_level_execution": {
                "weight": 0.70,
                "messages": [
                    {
                        "role": "user",
                        "content": "${subtask}",
                        "stream": "low_level",
                        "if_present": "subtask",
                    },
                ],
            },
        }
    }


@dataclass
class FineARTVLAConfig(PolicyConfig):
    network: dict = field(default_factory=dict)
    assets_path: str = ""
    fast_tokenizer_path: str = ""
    auto_fit_fast_tokenizer: bool = True
    fast_tokenizer_cache_dir: str = ""
    fast_fit_samples: int = 1024
    fast_validation_samples: int = 256
    fast_fit_vocab_size: int = 1024
    fast_fit_scale: float = 10.0
    fast_fit_seed: int = 0
    fast_max_reconstruction_rmse: float = 0.1
    fast_max_dim_rmse: float = 0.2
    fast_fit_report: dict = field(default_factory=dict)
    base_weights_prefix: str = "model."
    n_action_steps: int = 50
    num_inference_steps: int = 10
    max_action_tokens: int = 256
    fast_skip_tokens: int = 1152
    ce_chunk_size: int = 32
    flow_num_repeats: int = 5
    max_decoding_steps: int = 128
    knowledge_insulation: bool = True
    fp32_joint_attention: bool = True
    gradient_checkpointing: bool = True
    freeze_vision_encoder: bool = False
    flow_loss_weight: float = 10.0
    text_loss_weight: float = 1.0
    fast_action_loss_weight: float = 1.0
    text_ce_z_loss_weight: float = 1e-4
    recipe: dict = field(default_factory=default_recipe)
    vqa_cameras: tuple[str, ...] = ()
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        }
    )
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 1.0
    scheduler_type: str = "warmup_cosine"
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self):
        super().__post_init__()
        core = PI05Config(**self.network)
        self.network = asdict(core)
        recipe = TrainingRecipe.from_dict(self.recipe)
        if self.vqa_cameras:
            self.vqa_cameras = tuple(
                camera
                if camera.startswith("observation.images.")
                else f"observation.images.{camera}"
                for camera in self.vqa_cameras
            )
            if (
                len(set(self.vqa_cameras)) != len(self.vqa_cameras)
                or not set(self.vqa_cameras) <= self.image_features.keys()
            ):
                raise ValueError("vqa_cameras must name distinct selected camera fields")
            if recipe.blend is None:
                raise ValueError("Automatic VQA camera routes require a blend recipe")
            for camera in self.vqa_cameras:
                if f"vqa:{camera}" in recipe.blend:
                    continue
                recipe.blend[f"vqa:{camera}"] = TrainingRecipe.from_dict(
                    {
                        "weight": 1.0,
                        "route": "vqa",
                        "bindings": {
                            "vqa": f"emitted_at(t, style=vqa, role=assistant, camera={camera})",
                            "vqa_query": f"emitted_at(t, style=vqa, role=user, camera={camera})",
                        },
                        "messages": [
                            {
                                "role": "user",
                                "content": f"Image: {camera}\n${{vqa_query}}",
                                "stream": "high_level",
                                "if_present": ["vqa", "vqa_query"],
                            },
                            {
                                "role": "assistant",
                                "content": "${vqa}",
                                "stream": "high_level",
                                "target": True,
                                "if_present": ["vqa", "vqa_query"],
                            },
                        ],
                    }
                )
        if not recipe.referenced_cameras() <= self.image_features.keys():
            raise ValueError("Recipe camera bindings must name selected image fields")
        self.recipe = asdict(recipe)
        for key in (
            "n_action_steps",
            "num_inference_steps",
            "max_action_tokens",
            "ce_chunk_size",
            "flow_num_repeats",
            "max_decoding_steps",
            "fast_fit_samples",
            "fast_validation_samples",
            "fast_fit_vocab_size",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.fast_skip_tokens) is not int or self.fast_skip_tokens < 0:
            raise ValueError("fast_skip_tokens must be a nonnegative integer")
        for key in (
            "knowledge_insulation",
            "fp32_joint_attention",
            "gradient_checkpointing",
            "freeze_vision_encoder",
            "auto_fit_fast_tokenizer",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        for key in (
            "flow_loss_weight",
            "text_loss_weight",
            "fast_action_loss_weight",
            "text_ce_z_loss_weight",
        ):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if not any((self.flow_loss_weight, self.text_loss_weight, self.fast_action_loss_weight)):
            raise ValueError("FineART requires at least one enabled objective")
        for key in ("fast_fit_scale", "fast_max_reconstruction_rmse", "fast_max_dim_rmse"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if type(self.fast_fit_seed) is not int or self.fast_fit_seed < 0:
            raise ValueError("fast_fit_seed must be a nonnegative integer")
        if not self.assets_path or (
            self.fast_action_loss_weight
            and not self.fast_tokenizer_path
            and not self.auto_fit_fast_tokenizer
        ):
            raise ValueError(
                "FineART requires local tokenizer assets and a FAST codec when enabled"
            )
        if not self.image_features or self.robot_state_feature is None:
            raise ValueError("FineART requires selected cameras and named state fields")
        if max(self.action_feature.shape[0], self.robot_state_feature.shape[0]) > core.action_dim:
            raise ValueError("Recorded state/actions exceed the configured core width")
        if self.n_action_steps > core.action_horizon:
            raise ValueError("n_action_steps exceeds the trained horizon")
        if any(self.normalization_mapping[k] != "QUANTILES" for k in ("STATE", "ACTION")):
            raise ValueError("FineART state/action normalization requires QUANTILES")

    @property
    def chunk_size(self):
        return self.network["action_horizon"]

    @property
    def quantile_options(self):
        return {
            key: {"epsilon": 1e-6}
            for key, feature in {**self.input_features, **self.output_features}.items()
            if feature.type in ("STATE", "ACTION")
        }
