"""BSP episode fitting and parameter packing; see SOURCE.json and LICENSE.bspline.

The upstream row layout is retained: the first column contains knots and the
remaining columns contain control points, padded to the knot-vector length.
Knot units here are recorded frame indices, converted to seconds only at execution.
"""

import numpy as np
from scipy.interpolate import generate_knots, make_lsq_spline

from alohamini.trajectory import BSplineActionPlan


def fit_episode(actions, *, degree=3, parameter_rows=16, max_error=0.01, stride=1):
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or not np.isfinite(actions).all() or len(actions) < degree + 1:
        raise ValueError("BSP fitting requires a finite episode with at least degree+1 rows")
    if parameter_rows < 2 * degree + 2 or stride < 1 or max_error <= 0:
        raise ValueError("Invalid BSP parameter window or fitting tolerance")
    time = np.arange(len(actions), dtype=np.float64)
    spline = None
    for knots in generate_knots(time, actions, k=degree, s=1e-12):
        spline = make_lsq_spline(time, actions, knots, k=degree)
        if np.max(np.abs(spline(time) - actions)) <= max_error:
            break
    if spline is None or np.max(np.abs(spline(time) - actions)) > max_error + 1e-7:
        raise ValueError("Episode cannot satisfy the declared BSP fitting tolerance")
    span_count = parameter_rows - 2 * degree
    unique = spline.t[degree:-degree]
    chunks = []
    for start in range(0, len(unique) - 1, stride):
        knots = spline.t[start : start + parameter_rows]
        points = spline.c[start : start + parameter_rows]
        if len(knots) < parameter_rows:
            knots = np.pad(knots, (0, parameter_rows - len(knots)), mode="edge")
        if len(points) < parameter_rows:
            points = np.pad(points, ((0, parameter_rows - len(points)), (0, 0)), mode="edge")
        if knots[degree] < knots[-degree - 1]:
            chunks.append(np.column_stack((knots, points)))
        if start + span_count >= len(unique):
            break
    if not chunks:
        # A single polynomial still has a valid fixed-size representation.
        while len(spline.t) < parameter_rows:
            spans = np.diff(spline.t)
            i = int(np.argmax(spans))
            spline = spline.insert_knot(float((spline.t[i] + spline.t[i + 1]) / 2))
        knots = spline.t[:parameter_rows]
        points = np.pad(spline.c, ((0, parameter_rows - len(spline.c)), (0, 0)), mode="edge")
        chunks = [np.column_stack((knots, points))]
    return np.asarray(chunks, dtype=np.float32)


def encode_sample(chunks, frame, degree):
    anchors = chunks[:, degree, 0]
    index = min(int(np.searchsorted(anchors, frame, side="left")), len(chunks) - 1)
    result = chunks[index].copy()
    result[:, 0] -= frame
    return result


def parameter_plan(parameters, names, *, degree, fps, speed=1.0):
    params = np.asarray(parameters, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != len(names) + 1 or not np.isfinite(params).all():
        raise ValueError("Invalid predicted BSP parameters")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("BSP execution requires positive FPS and time scale")
    # Noise prediction does not enforce knot ordering. Preserve order with a
    # minimal forward projection rather than sorting coefficients independently.
    knots = params[:, 0].copy()
    for i in range(1, len(knots)):
        if knots[i] < knots[i - 1]:
            knots[i] = knots[i - 1] + 1e-5
    if knots[-degree - 1] <= knots[degree] + 1e-6:
        raise ValueError("Predicted BSP has an empty execution time range")
    # Segment execution starts at its first valid knot. Shape and span durations
    # are retained; asynchronous source-time compensation is owned by the executor.
    knots = (knots - knots[degree]) / (fps * speed)
    points = params[: -(degree + 1), 1:]
    return BSplineActionPlan(tuple(names), knots.tolist(), points.tolist(), degree)


def align_phase(plan, prefix):
    """Match the committed command at the estimated handover time (BSP Eq. 2).

    Remote execution uses the transmitted delay estimate, never the cloud's clock
    minus a Pi timestamp. The executor retains the old plan until the new domain starts.
    """
    from bisect import bisect_right

    from scipy.optimize import minimize_scalar

    if prefix.names != plan.names:
        raise ValueError("BSP phase alignment requires identical command coordinates")
    delay = prefix.inference_delay_s
    if delay > prefix.times_s[-1] + 1e-9:
        raise ValueError("BSP handover delay exceeds the committed prefix")
    index = max(0, bisect_right(prefix.times_s, delay + 1e-9) - 1)
    target = np.asarray(prefix.actions[index])
    times = np.linspace(plan.start_s, (plan.start_s + plan.end_s) / 2, 65)

    def error(t):
        return float(np.square(np.asarray(plan.action_at(float(t))) - target).sum())

    best = int(np.argmin([error(t) for t in times]))
    lo, hi = times[max(0, best - 1)], times[min(len(times) - 1, best + 1)]
    refined = minimize_scalar(error, bounds=(lo, hi), method="bounded")
    candidates = [float(times[best]), float(refined.x)]
    phase = min(candidates, key=error)
    shift = phase - delay
    return BSplineActionPlan(
        plan.names, tuple(t - shift for t in plan.knots_s), plan.control_points, plan.degree
    )
