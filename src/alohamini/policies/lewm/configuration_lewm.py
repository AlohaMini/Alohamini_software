"""LeWorldModel architecture, recorded transitions and goal-conditioned planning."""

import math
from dataclasses import dataclass, field

from alohamini.policies.configuration import PolicyConfig


@dataclass
class LeWMConfig(PolicyConfig):
    history_size: int = 3
    action_block: int = 5
    planning_horizon: int = 5
    chunk_size: int = 25
    n_action_steps: int = 25
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_mlp_dim: int = 768
    patch_size: int = 14
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    predictor_mlp_dim: int = 2048
    projector_hidden_dim: int = 2048
    dropout: float = 0.1
    sigreg_weight: float = 0.09
    sigreg_knots: int = 17
    sigreg_projections: int = 1024
    cem_samples: int = 300
    cem_elites: int = 30
    cem_iterations: int = 30
    cem_initial_std: float = 1.0
    constrain_actions: bool = True
    gradient_checkpointing: bool = False
    normalization_mapping: dict = field(
        default_factory=lambda: {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    )
    optimizer_lr: float = 5e-5
    optimizer_betas: tuple = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-3
    optimizer_grad_clip_norm: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        for key in (
            "history_size",
            "action_block",
            "planning_horizon",
            "chunk_size",
            "n_action_steps",
            "embed_dim",
            "encoder_depth",
            "encoder_heads",
            "encoder_mlp_dim",
            "patch_size",
            "predictor_depth",
            "predictor_heads",
            "predictor_head_dim",
            "predictor_mlp_dim",
            "projector_hidden_dim",
            "sigreg_projections",
            "sigreg_knots",
            "cem_samples",
            "cem_elites",
            "cem_iterations",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.sigreg_knots < 2 or self.sigreg_weight < 0 or self.cem_initial_std <= 0:
            raise ValueError("Invalid SIGReg or CEM settings")
        if not all(math.isfinite(v) for v in (self.sigreg_weight, self.cem_initial_std)):
            raise ValueError("SIGReg and CEM settings must be finite")
        if not 0 <= self.dropout < 1 or self.embed_dim % self.encoder_heads:
            raise ValueError("Invalid dropout or encoder heads")
        if self.chunk_size != self.action_block * self.planning_horizon:
            raise ValueError("chunk_size must equal action_block * planning_horizon")
        if self.n_action_steps > self.chunk_size or self.cem_elites > self.cem_samples:
            raise ValueError("Execution horizon or CEM elites exceed available candidates")
        if not self.image_features:
            raise ValueError("LeWM needs at least one recorded RGB camera")
        if self.normalization_mapping.get("ACTION") != "MEAN_STD":
            raise ValueError("LeWM uses z-score action normalization")
        shapes = {f.shape for f in self.image_features.values()}
        if len(shapes) != 1 or next(iter(shapes))[1] != next(iter(shapes))[2]:
            raise ValueError("LeWM cameras must have one square input size")
        if next(iter(shapes))[1] % self.patch_size:
            raise ValueError("Image size must be divisible by patch_size")
        for key in ("constrain_actions", "gradient_checkpointing"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
