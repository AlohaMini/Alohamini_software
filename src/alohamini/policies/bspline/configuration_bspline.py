"""Physical action metadata is distinct from the generated parameter matrix."""

from dataclasses import dataclass

from alohamini.policies.diffusion.configuration_diffusion import DiffusionConfig


@dataclass
class BSplineConfig(DiffusionConfig):
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 1
    drop_n_last_frames: int | None = 0
    degree: int = 3
    fit_tolerance: float = 0.01
    knot_stride: int = 1
    time_scale: float = 1.0
    parameter_cache_dir: str | None = None
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int | None = 16

    def __post_init__(self):
        super().__post_init__()
        import math

        if type(self.degree) is not int or not 1 <= self.degree <= 5:
            raise ValueError("BSP degree must be an integer in [1,5]")
        if self.horizon < 2 * self.degree + 2:
            raise ValueError("BSP parameter window is too short for its degree")
        if type(self.knot_stride) is not int or self.knot_stride < 1:
            raise ValueError("BSP knot_stride must be positive")
        if any(not math.isfinite(v) or v <= 0 for v in (self.fit_tolerance, self.time_scale)):
            raise ValueError("BSP tolerance and time_scale must be finite and positive")
