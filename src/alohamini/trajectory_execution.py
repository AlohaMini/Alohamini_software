"""Trajectory scheduling options, independent of policy names and tensor runtimes."""

import math
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class ExecutionConfig:
    enabled: bool = False
    replan_interval_s: float | None = None
    prediction_lead_s: float = 0.1
    max_prediction_age_s: float = 5.0
    startup_timeout_s: float = 15.0
    continuity: str = "none"
    tcp_tracking: dict | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("Execution enabled must be a boolean")
        for name in (
            "prediction_lead_s",
            "max_prediction_age_s",
            "startup_timeout_s",
            "replan_interval_s",
        ):
            v = getattr(self, name)
            if name == "replan_interval_s" and v is None:
                continue
            if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"inference.{name} must be finite and positive")
        if not isinstance(self.continuity, str) or not self.continuity:
            raise ValueError("Execution continuity must name a strategy capability")
        if self.tcp_tracking is not None:
            from .tcp_execution import TCPTrackingConfig

            if not isinstance(self.tcp_tracking, dict):
                raise ValueError("inference.tcp_tracking must be a configuration object")
            TCPTrackingConfig(**self.tcp_tracking)

    def to_dict(self):
        values = asdict(self)
        if self.tcp_tracking is None:
            values.pop("tcp_tracking")
        return values


@dataclass(frozen=True)
class PlanCapabilities:
    kinds: tuple[str, ...] = ("dense",)
    continuity_methods: tuple[str, ...] = ("none",)
    gradient_methods: tuple[str, ...] = ()
    default_continuity: str = "none"
    default_replan_interval_s: float | None = None

    def resolve(self, settings):
        """Resolve adapter defaults before execution or the remote handshake."""
        if settings.continuity == "auto":
            settings = replace(settings, continuity=self.default_continuity)
        if settings.replan_interval_s is None and self.default_replan_interval_s is not None:
            settings = replace(settings, replan_interval_s=self.default_replan_interval_s)
        if self.kinds == ("tcp",) and settings.tcp_tracking is None:
            from .tcp_execution import TCPTrackingConfig

            settings = replace(settings, tcp_tracking=asdict(TCPTrackingConfig()))
        self.validate(settings)
        return settings

    def validate(self, settings):
        if settings.tcp_tracking is not None and "tcp" not in self.kinds:
            raise ValueError("TCP tracking settings require a TCP action checkpoint")
        if settings.continuity not in self.continuity_methods:
            raise ValueError(
                f"Policy does not support {settings.continuity!r} trajectory continuity; "
                f"available: {', '.join(self.continuity_methods)}"
            )


def prediction_timing(settings, plan, *, fps, latency_s, observation_age_s):
    """Budget inference and leave usable committed samples for continuity.

    The trigger can be crossed between control ticks, and prefix sampling drops
    the expired endpoint. Dense RTC also rounds delay to the model's sample grid.
    Start early enough to retain those samples instead of relaxing prefix checks.
    """
    delay = max(latency_s, observation_age_s) + 1 / fps
    margin = 0.0
    if settings.continuity != "none":
        margin = 2 / fps + getattr(plan, "period_s", 0.0)
    return delay, max(settings.prediction_lead_s, delay + margin)


def add_inference_arguments(parser, *, default="sync"):
    import json
    from argparse import SUPPRESS

    parser.add_argument(
        "--inference.tcp_tracking",
        dest="inference_tcp_tracking",
        type=json.loads,
        default=SUPPRESS,
        help="TCP 在线跟踪配置的 JSON 对象，仅 async",
    )

    parser.add_argument(
        "--inference.type",
        dest="inference_type",
        choices=("sync", "async", "rtc"),
        default=default,
        help=f"推理模式，默认 {default}；rtc 要求策略支持前缀引导",
    )
    parser.add_argument(
        "--inference.replan_interval_s",
        dest="inference_replan_interval_s",
        type=float,
        default=SUPPRESS,
    )
    parser.add_argument(
        "--inference.prediction_lead_s",
        dest="inference_prediction_lead_s",
        type=float,
        default=SUPPRESS,
    )
    parser.add_argument(
        "--inference.max_prediction_age_s",
        dest="inference_max_prediction_age_s",
        type=float,
        default=SUPPRESS,
    )
    parser.add_argument(
        "--inference.continuity",
        dest="inference_continuity",
        choices=("none", "phase"),
        default=SUPPRESS,
        help="仅 async：覆盖策略默认衔接方式",
    )


def pop_inference_options(options, *, remote=False):
    mode = options.pop("inference_type", "async" if remote else "sync")
    if mode not in ("sync", "async", "rtc"):
        raise ValueError("Unknown inference type")
    fields = {
        n: options.pop("inference_" + n)
        for n in ExecutionConfig.__dataclass_fields__
        if "inference_" + n in options
    }
    if mode == "sync":
        if remote:
            raise ValueError("Remote inference requires --inference.type=async or rtc")
        if fields:
            raise ValueError("Inference tuning requires --inference.type=async or rtc")
        return ExecutionConfig()
    if mode == "rtc" and "continuity" in fields:
        raise ValueError("--inference.continuity only applies to --inference.type=async")
    if mode == "rtc" and "tcp_tracking" in fields:
        raise ValueError("TCP reference tracking requires --inference.type=async")
    fields.setdefault("continuity", "rtc" if mode == "rtc" else "auto")
    return ExecutionConfig(enabled=True, **fields)
