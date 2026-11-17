"""Physical-time action plans shared by local and remote execution.

No ML runtime is needed to validate, transmit or query a plan. Times are seconds
relative to its source observation, never a diffusion timestep or a remote clock.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Protocol


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _layout(names):
    names = tuple(names)
    if not names or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("A trajectory requires named action coordinates")
    if len(set(names)) != len(names):
        raise ValueError("Trajectory action names must be unique")
    return names


def _matrix(rows, width):
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    result = tuple(tuple(_number(v, "action value") for v in row) for row in rows)
    if not result or any(len(row) != width for row in result):
        raise ValueError("Trajectory values do not match their action layout")
    return result


class ActionPlan(Protocol):
    names: tuple[str, ...]
    start_s: float
    end_s: float

    def action_at(self, elapsed_s: float) -> tuple[float, ...]: ...

    def to_dict(self) -> dict: ...


@dataclass(frozen=True)
class DenseActionPlan:
    """Sample-and-hold absolute commands; interpolation is deliberately explicit."""

    names: tuple[str, ...]
    actions: tuple[tuple[float, ...], ...]
    period_s: float
    start_s: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "names", _layout(self.names))
        object.__setattr__(self, "actions", _matrix(self.actions, len(self.names)))
        period = _number(self.period_s, "period_s")
        start = _number(self.start_s, "start_s")
        if period <= 0 or len(self.actions) > 16384:
            raise ValueError("Invalid dense trajectory period or length")
        object.__setattr__(self, "period_s", period)
        object.__setattr__(self, "start_s", start)

    @property
    def end_s(self):
        return self.start_s + len(self.actions) * self.period_s

    def action_at(self, elapsed_s):
        elapsed_s = _number(elapsed_s, "elapsed_s")
        if not self.start_s <= elapsed_s < self.end_s:
            raise ValueError("Requested action lies outside the trajectory time range")
        index = min(int((elapsed_s - self.start_s) / self.period_s + 1e-9), len(self.actions) - 1)
        return self.actions[index]

    def to_dict(self):
        return dict(
            version=1,
            kind="dense",
            names=list(self.names),
            actions=[list(row) for row in self.actions],
            period_s=self.period_s,
            start_s=self.start_s,
        )


@dataclass(frozen=True)
class BSplineActionPlan:
    """An absolute-command B-spline queried with de Boor's algorithm on the CPU."""

    names: tuple[str, ...]
    knots_s: tuple[float, ...]
    control_points: tuple[tuple[float, ...], ...]
    degree: int = 3

    def __post_init__(self):
        object.__setattr__(self, "names", _layout(self.names))
        points = _matrix(self.control_points, len(self.names))
        knots = tuple(_number(u, "knot") for u in self.knots_s)
        p = self.degree
        if type(p) is not int or not 1 <= p <= 5 or len(points) < p + 1:
            raise ValueError("Invalid B-spline degree or control-point count")
        if len(points) > 4096 or len(knots) != len(points) + p + 1:
            raise ValueError("B-spline knot and control-point counts disagree")
        if any(a > b for a, b in zip(knots, knots[1:], strict=False)) or knots[p] >= knots[-p - 1]:
            raise ValueError("B-spline knots require a nonempty nondecreasing time domain")
        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "knots_s", knots)

    @property
    def start_s(self):
        return self.knots_s[self.degree]

    @property
    def end_s(self):
        return self.knots_s[-self.degree - 1]

    def action_at(self, elapsed_s):
        t = _number(elapsed_s, "elapsed_s")
        if not self.start_s <= t < self.end_s:
            raise ValueError("Requested action lies outside the trajectory time range")
        p, knots = self.degree, self.knots_s
        span = min(bisect_right(knots, t) - 1, len(self.control_points) - 1)
        values = [list(row) for row in self.control_points[span - p : span + 1]]
        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                i = span - p + j
                width = knots[i + p - r + 1] - knots[i]
                alpha = (t - knots[i]) / width if width else 0.0
                values[j] = [
                    (1 - alpha) * a + alpha * b
                    for a, b in zip(values[j - 1], values[j], strict=True)
                ]
        return tuple(values[p])

    def to_dict(self):
        return dict(
            version=1,
            kind="bspline",
            names=list(self.names),
            degree=self.degree,
            knots_s=list(self.knots_s),
            control_points=[list(row) for row in self.control_points],
        )


def plan_from_dict(value, *, names=None):
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported trajectory plan version")
    kind = value.get("kind")
    if kind == "tcp":
        from .tcp_trajectory import TCPActionPlan

        keys = {
            "version",
            "kind",
            "names",
            "actions",
            "period_s",
            "start_s",
            "descriptor",
            "fixed_targets",
        }
        if set(value) != keys:
            raise ValueError("Invalid TCP trajectory plan fields")
        plan = TCPActionPlan(**{k: v for k, v in value.items() if k not in ("version", "kind")})
        if names is not None and plan.names != tuple(names):
            raise ValueError("Trajectory action layout changed")
        return plan
    keys = (
        {"version", "kind", "names", "actions", "period_s", "start_s"}
        if kind == "dense"
        else {"version", "kind", "names", "knots_s", "control_points", "degree"}
    )
    if kind not in ("dense", "bspline") or set(value) != keys:
        raise ValueError("Invalid trajectory plan fields")
    args = {k: v for k, v in value.items() if k not in ("version", "kind")}
    plan = DenseActionPlan(**args) if kind == "dense" else BSplineActionPlan(**args)
    if names is not None and plan.names != tuple(names):
        raise ValueError("Trajectory action layout changed")
    return plan


@dataclass(frozen=True)
class ActionPrefix:
    """Already committed commands at times relative to a new observation."""

    names: tuple[str, ...]
    times_s: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    inference_delay_s: float

    def __post_init__(self):
        object.__setattr__(self, "names", _layout(self.names))
        object.__setattr__(self, "actions", _matrix(self.actions, len(self.names)))
        times = tuple(_number(t, "prefix time") for t in self.times_s)
        if len(times) != len(self.actions) or times[0] < 0:
            raise ValueError("Invalid action-prefix timestamps")
        if any(a >= b for a, b in zip(times, times[1:], strict=False)):
            raise ValueError("Action-prefix timestamps must increase")
        delay = _number(self.inference_delay_s, "inference_delay_s")
        if delay < 0:
            raise ValueError("Inference delay must be nonnegative")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "inference_delay_s", delay)


def committed_prefix(plan, offset_s, fps, delay_s):
    """Sample the remaining accepted plan in execution space, never parameter rows."""
    start = max(0.0, plan.start_s - offset_s)
    count = min(16384, max(0, math.ceil((plan.end_s - offset_s - start) * fps - 1e-9)))
    if count == 0:
        return None
    times = tuple(start + i / fps for i in range(count))
    return ActionPrefix(
        plan.names, times, tuple(plan.action_at(offset_s + t) for t in times), delay_s
    )
