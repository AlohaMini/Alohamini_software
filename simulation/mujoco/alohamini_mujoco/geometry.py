"""Triangle proximity and conservative continuous surface comparison in metres."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


def _squared_distances(points, triangles):
    """Paired, broadcastable point-to-triangle distances, including degenerate faces."""
    a, b, c = (triangles[..., i, :] for i in range(3))
    u, v, w = b - a, c - a, points - a
    normal = np.cross(u, v)
    uu, vv, uv = (np.sum(x * y, axis=-1) for x, y in ((u, u), (v, v), (u, v)))
    determinant = np.sum(normal * normal, axis=-1)
    safe = np.where(determinant > 0, determinant, 1)
    wu, wv = np.sum(w * u, axis=-1), np.sum(w * v, axis=-1)
    alpha, beta = (vv * wu - uv * wv) / safe, (uu * wv - uv * wu) / safe
    inside = (determinant > 0) & (alpha >= 0) & (beta >= 0) & (alpha + beta <= 1)
    squared = np.where(inside, np.sum(w * normal, axis=-1) ** 2 / safe, np.inf)
    for first, last in ((a, b), (b, c), (c, a)):
        edge = last - first
        norm = np.sum(edge * edge, axis=-1)
        t = np.clip(np.sum((points - first) * edge, axis=-1) / np.where(norm > 0, norm, 1), 0, 1)
        squared = np.minimum(squared, np.sum((points - first - t[..., None] * edge) ** 2, axis=-1))
    return squared


def _triangles(values):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 3
        or values.shape[1:] != (3, 3)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Use finite nonempty triangle coordinates [T,3,3]")
    return values


class SurfaceIndex:
    """Exact nearest triangles using radius-bucketed centroid trees.

    A triangle lies inside its centroid sphere. Querying all spheres that can
    beat an existing surface distance prunes work without approximate nearest
    neighbors, triangle reduction, or a fixed number of candidate faces.
    """

    def __init__(self, triangles):
        self.triangles = _triangles(triangles).copy()
        centers = self.triangles.mean(axis=1)
        radii = np.linalg.norm(self.triangles - centers[:, None], axis=2).max(axis=1)
        # Separate large triangles from small ones; a single global radius
        # otherwise makes a fine CAD mesh degenerate into an all-pairs query.
        exponents = np.frexp(radii)[1]
        exponents[radii == 0] = np.iinfo(exponents.dtype).min
        self.buckets = []
        for exponent in np.unique(exponents):
            indices = np.flatnonzero(exponents == exponent)
            self.buckets.append((indices, cKDTree(centers[indices]), radii[indices].max()))
        self.guard = 64 * np.finfo(float).eps * max(1.0, np.abs(self.triangles).max())

    def closest(self, points, *, batch_size=32):
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or not np.isfinite(points).all()
            or type(batch_size) is not int
            or batch_size <= 0
        ):
            raise ValueError("Use finite points [N,3] and a positive integer batch size")
        distances, nearest = np.empty(len(points)), np.empty(len(points), dtype=np.int64)
        for start in range(0, len(points), batch_size):
            p = points[start : start + batch_size]
            best, selected = np.full(len(p), np.inf), np.zeros(len(p), dtype=np.int64)
            for indices, tree, _ in self.buckets:
                _, local = tree.query(p, k=1)
                candidate = indices[local]
                squared = _squared_distances(p, self.triangles[candidate])
                update = squared < best
                best[update], selected[update] = squared[update], candidate[update]
            for indices, tree, radius in self.buckets:
                query_guard = max(self.guard, 64 * np.finfo(float).eps * max(1.0, np.abs(p).max()))
                query_radius = np.nextafter(np.sqrt(best) + radius + query_guard, np.inf)
                candidates = tree.query_ball_point(p, query_radius, eps=0)
                owners = np.repeat(np.arange(len(p)), [len(c) for c in candidates])
                candidate = indices[np.concatenate(candidates).astype(np.int64)]
                # Bound the temporary paired-distance arrays even when many
                # overlapping triangles share a centroid sphere.
                for offset in range(0, len(candidate), 8192):
                    ids, rows = candidate[offset : offset + 8192], owners[offset : offset + 8192]
                    squared = _squared_distances(p[rows], self.triangles[ids])
                    starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
                    minima = np.minimum.reduceat(squared, starts)
                    lengths = np.diff(np.r_[starts, len(rows)])
                    winners = np.minimum.reduceat(
                        np.where(
                            squared == np.repeat(minima, lengths),
                            np.arange(len(rows)),
                            len(rows),
                        ),
                        starts,
                    )
                    update = squared[winners] < best[rows[winners]]
                    winners = winners[update]
                    best[rows[winners]], selected[rows[winners]] = squared[winners], ids[winners]
            distances[start : start + len(p)] = np.sqrt(best)
            nearest[start : start + len(p)] = selected
        return distances, nearest


def directed_surface_bounds(
    source_triangles, target, *, tolerance_m, max_samples=1_000_000, progress=None
):
    """Bound the maximum source-surface distance to a target triangle surface.

    Every source triangle is covered. At its centroid, distance is a lower
    bound. The 1-Lipschitz distance function supplies an upper bound using the
    triangle radius. Distance to any *one* target triangle is convex, so its
    largest distance at the three source vertices is another upper bound.
    Unresolved triangles split along their longest edge. A budget exhaustion
    is explicitly inconclusive; it can never turn a sample check into a pass.

    Bounds use float64 with a scale-dependent rounding guard. They compare
    the supplied meshes, not the original analytic CAD or physical hardware.
    """
    frontier = _triangles(source_triangles).copy()
    if (
        not math.isfinite(tolerance_m)
        or tolerance_m <= 0
        or type(max_samples) is not int
        or max_samples < len(frontier)
    ):
        raise ValueError("Use a positive metric tolerance and budget for every source triangle")
    if not isinstance(target, SurfaceIndex):
        target = SurfaceIndex(target)
    guard = max(target.guard, 64 * np.finfo(float).eps * max(1.0, np.abs(frontier).max()))
    lower, accepted_upper, samples, levels = 0.0, 0.0, 0, 0
    while True:
        centers = frontier.mean(axis=1)
        distance, closest = target.closest(centers)
        radius = np.linalg.norm(frontier - centers[:, None], axis=2).max(axis=1)
        vertex_bound = np.sqrt(_squared_distances(frontier, target.triangles[closest, None])).max(
            axis=1
        )
        bounds = np.minimum(distance + radius, vertex_bound) + guard
        lower = max(lower, float(distance.max()) - guard)
        upper = max(accepted_upper, float(bounds.max()))
        samples += len(frontier)
        levels += 1
        unresolved = bounds > tolerance_m
        if progress is not None:
            progress(
                {
                    "centroid_queries": samples,
                    "lower_bound_m": max(0.0, lower),
                    "upper_bound_m": upper,
                    "unresolved_triangles": int(unresolved.sum()),
                }
            )
        if lower > tolerance_m:
            status = "failed"
            break
        if not unresolved.any():
            status = "passed"
            break
        if samples + 2 * int(unresolved.sum()) > max_samples:
            status = "inconclusive"
            break
        if (~unresolved).any():
            accepted_upper = max(accepted_upper, float(bounds[~unresolved].max()))
        triangles = frontier[unresolved]
        lengths = np.sum((triangles - np.roll(triangles, -1, axis=1)) ** 2, axis=2)
        first = lengths.argmax(axis=1)
        rows = np.arange(len(triangles))
        a, b, c = (triangles[rows, (first + i) % 3] for i in range(3))
        midpoint = (a + b) / 2
        frontier = np.concatenate((np.stack((a, midpoint, c), 1), np.stack((midpoint, b, c), 1)))
    return {
        "status": status,
        "lower_bound_m": max(0.0, lower),
        "upper_bound_m": upper,
        "tolerance_m": tolerance_m,
        "centroid_queries": samples,
        "subdivision_levels": levels,
        "max_samples": max_samples,
        "rounding_guard_m": guard,
        "method": "adaptive-triangle-cover-v1",
        "scope": "continuous triangle surfaces in float64; excludes CAD tessellation error",
    }


def measure_surface_distance(
    source_triangles, target, *, resolution_m, max_samples=1_000_000, progress=None
):
    """Bound a directed surface maximum to a requested interval width.

    This measures geometry; it does not choose or change an admission tolerance.
    The same continuous triangle covers as directed_surface_bounds are used.
    A witness identifies a sampled lower bound, never an exact maximum point.
    """
    frontier = _triangles(source_triangles).copy()
    if (
        not math.isfinite(resolution_m)
        or resolution_m <= 0
        or type(max_samples) is not int
        or max_samples < len(frontier)
    ):
        raise ValueError("Use a positive metric resolution and budget for every source triangle")
    if not isinstance(target, SurfaceIndex):
        target = SurfaceIndex(target)
    guard = max(target.guard, 64 * np.finfo(float).eps * max(1.0, np.abs(frontier).max()))
    source_ids = np.arange(len(frontier))
    lower, accepted_upper, samples, levels, witness = 0.0, 0.0, 0, 0, None
    while True:
        centers = frontier.mean(axis=1)
        distance, closest = target.closest(centers)
        radius = np.linalg.norm(frontier - centers[:, None], axis=2).max(axis=1)
        vertex_bound = np.sqrt(_squared_distances(frontier, target.triangles[closest, None])).max(
            axis=1
        )
        bounds = np.minimum(distance + radius, vertex_bound) + guard
        row = int(distance.argmax())
        if witness is None or distance[row] - guard > lower:
            lower = max(0.0, float(distance[row]) - guard)
            witness = {
                "source_triangle": int(source_ids[row]),
                "target_triangle": int(closest[row]),
                "source_point_m": centers[row].tolist(),
                "sample_distance_m": float(distance[row]),
            }
        upper = max(accepted_upper, float(bounds.max()))
        samples += len(frontier)
        levels += 1
        unresolved = bounds > lower + resolution_m
        if progress is not None:
            progress({"centroid_queries": samples, "lower_bound_m": lower, "upper_bound_m": upper})
        if upper - lower <= resolution_m:
            status = "bounded"
            break
        if samples + 2 * int(unresolved.sum()) > max_samples:
            status = "inconclusive"
            break
        if (~unresolved).any():
            accepted_upper = max(accepted_upper, float(bounds[~unresolved].max()))
        triangles = frontier[unresolved]
        first = np.sum((triangles - np.roll(triangles, -1, axis=1)) ** 2, axis=2).argmax(axis=1)
        rows = np.arange(len(triangles))
        a, b, c = (triangles[rows, (first + i) % 3] for i in range(3))
        midpoint = (a + b) / 2
        frontier = np.concatenate((np.stack((a, midpoint, c), 1), np.stack((midpoint, b, c), 1)))
        source_ids = np.tile(source_ids[unresolved], 2)
    return {
        "status": status,
        "lower_bound_m": lower,
        "upper_bound_m": upper,
        "resolution_m": resolution_m,
        "centroid_queries": samples,
        "subdivision_levels": levels,
        "max_samples": max_samples,
        "rounding_guard_m": guard,
        "witness": witness,
        "scope": "directed continuous mesh distance; no admission or analytic CAD accuracy claim",
    }
