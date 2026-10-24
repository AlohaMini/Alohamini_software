"""Reject internally inconsistent native convex distance queries.

These checks detect contradictions, not every possible GJK/EPA error. They
never repair a distance or switch collision engines. Projection intervals use
the native convex hull or analytic primitive, independently of GJK/EPA.

Upstream engine regression: https://github.com/google-deepmind/mujoco/issues/3383
"""

from __future__ import annotations

import mujoco
import numpy as np


class InvalidNativeDistance(ValueError):
    """A native distance contradicts its witnesses or a separating projection."""


def _projections(model, data, geom, directions):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local = directions @ rotation
    center = directions @ data.geom_xpos[geom]
    kind, size = model.geom_type[geom], model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        projected = model.mesh_vert[start : start + count].astype(float) @ local.T
        return center + projected.min(0), center + projected.max(0)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        radius = abs(local) @ size
    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = np.repeat(size[0], len(directions))
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = size[0] * np.linalg.norm(local[:, :2], axis=1) + size[1] * abs(local[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = size[0] + size[1] * abs(local[:, 2])
    elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radius = np.linalg.norm(local * size, axis=1)
    else:
        return None
    return center - radius, center + radius


def checked_geom_distance(model, data, first, second, maximum):
    """Return distance and witnesses, raising on a geometric contradiction.

    The cutoff must be positive. Censored distances have no required witnesses.
    Uncensored distances must agree with witness length and support projections
    along world axes, each geom's axes, the center difference and witness segment.
    For unsupported shapes only witness consistency is checked. Passing is not an
    independent clearance certificate or proof of collision-solver correctness.
    """
    if isinstance(maximum, (bool, np.bool_)) or not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Native distance queries need a positive finite cutoff in metres")
    witness = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, first, second, maximum, witness))
    tolerance = max(1e-9, 4 * model.opt.ccd_tolerance)
    reason = None
    if not np.isfinite(distance) or not np.isfinite(witness).all() or distance > maximum:
        reason = "nonfinite result or distance beyond cutoff"
    elif distance < maximum:
        segment = witness[3:] - witness[:3]
        length = float(np.linalg.norm(segment))
        if abs(length - abs(distance)) > tolerance:
            reason = f"witness length {length:.17g} m disagrees with distance"
        else:
            directions = np.vstack(
                [
                    np.eye(3),
                    data.geom_xmat[first].reshape(3, 3).T,
                    data.geom_xmat[second].reshape(3, 3).T,
                    data.geom_xpos[second] - data.geom_xpos[first],
                    segment,
                ]
            )
            lengths = np.linalg.norm(directions, axis=1)
            directions = directions[lengths > 1e-15] / lengths[lengths > 1e-15, None]
            a, b = (_projections(model, data, geom, directions) for geom in (first, second))
            if a is not None and b is not None:
                gap = float(np.maximum(a[0] - b[1], b[0] - a[1]).max())
                # A positive gap proves separation. A negative gap gives an
                # upper bound on the translation needed to separate this pair.
                if distance + tolerance < gap:
                    reason = f"projection lower bound {gap:.17g} m contradicts distance"
    if reason is not None:
        names = (model.geom(first).name, model.geom(second).name)
        raise InvalidNativeDistance(
            f"Invalid MuJoCo {mujoco.__version__} distance {names}: {distance:.17g} m; {reason}"
        )
    return distance, witness
