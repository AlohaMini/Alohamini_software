"""Runtime RTC settings; safe to import on a Pi without ML dependencies."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RTCConfig:
    enabled: bool = False
    replan_steps: int = 10
    inference_delay_steps: int = 3
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: str = "linear"
    max_prediction_age_s: float = 1.0

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("RTC enabled must be a boolean")
        for name, minimum in (("replan_steps", 1), ("inference_delay_steps", 0)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= 512:
                raise ValueError(f"RTC {name} must be an integer in [{minimum}, 512]")
        for name in ("max_guidance_weight", "max_prediction_age_s"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"RTC {name} must be finite and positive")
        if self.prefix_attention_schedule not in ("linear", "exp", "ones", "zeros"):
            raise ValueError("Unknown RTC prefix attention schedule")

    def validate_horizon(self, horizon):
        if horizon <= self.replan_steps + self.inference_delay_steps:
            raise ValueError("RTC chunk must exceed replan_steps + inference_delay_steps")

    def to_dict(self):
        return asdict(self)


def add_rtc_arguments(parser):
    parser.add_argument("--rtc", action="store_true", help="Enable guided real-time chunking")
    parser.add_argument("--rtc.replan_steps", dest="rtc_replan_steps", type=int, default=10)
    parser.add_argument(
        "--rtc.inference_delay_steps", dest="rtc_inference_delay_steps", type=int, default=3
    )
    parser.add_argument(
        "--rtc.max_guidance_weight", dest="rtc_max_guidance_weight", type=float, default=10.0
    )
    parser.add_argument(
        "--rtc.prefix_attention_schedule", dest="rtc_prefix_attention_schedule",
        choices=("linear", "exp", "ones", "zeros"), default="linear",
    )
    parser.add_argument(
        "--rtc.max_prediction_age_s", dest="rtc_max_prediction_age_s", type=float, default=1.0
    )


def pop_rtc_options(options):
    enabled = options.pop("rtc", False)
    values = {
        name: options.pop("rtc_" + name)
        for name in RTCConfig.__dataclass_fields__
        if "rtc_" + name in options
    }
    config = RTCConfig(enabled=enabled, **values)
    if not enabled and config != RTCConfig():
        raise ValueError("RTC options require --rtc")
    return config
