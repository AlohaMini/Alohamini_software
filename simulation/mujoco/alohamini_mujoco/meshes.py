"""Lossless mesh transport and physical convex-piece cross sections."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def triangles(self):
        return self.vertices[self.faces]

    def msh_bytes(self):
        # MSH has no STL decoder's 200,000-face restriction. Indexing exact
        # repeated float32 coordinates changes neither triangles nor winding.
        header = struct.pack("<4i", len(self.vertices), 0, 0, len(self.faces))
        return header + self.vertices.astype("<f4").tobytes() + self.faces.astype("<i4").tobytes()


def binary_stl(data):
    if len(data) < 84:
        raise ValueError("Truncated binary STL")
    count = struct.unpack_from("<I", data, 80)[0]
    if not count or len(data) != 84 + count * 50:
        raise ValueError("Expected a complete binary STL with its declared face count")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    triangles = np.frombuffer(data, dtype=dtype, count=count, offset=84)["vertices"]
    if not np.isfinite(triangles).all():
        raise ValueError("Nonfinite STL vertices")
    vertices, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    if len(vertices) < 4:
        raise ValueError("A rigid mesh needs at least four distinct vertices")
    return Mesh(vertices, indices.reshape(-1, 3))


def binary_msh(data):
    if len(data) < 16:
        raise ValueError("Truncated MSH")
    vertices, normals, textures, faces = struct.unpack_from("<4i", data)
    if (
        vertices < 4
        or faces < 4
        or normals not in (0, vertices)
        or textures not in (0, vertices)
        or len(data) != 16 + 12 * (vertices + normals + faces) + 8 * textures
    ):
        raise ValueError("Invalid MSH sizes")
    points = np.frombuffer(data, dtype="<f4", offset=16, count=3 * vertices).reshape(-1, 3)
    offset = 16 + 12 * (vertices + normals) + 8 * textures
    indices = np.frombuffer(data, dtype="<i4", offset=offset, count=3 * faces).reshape(-1, 3)
    if not np.isfinite(points).all() or indices.min() < 0 or indices.max() >= vertices:
        raise ValueError("Invalid MSH coordinates or indices")
    return Mesh(points.copy(), indices.copy())


def ray_x_intersections(triangles, y, z):
    """Intersect the infinite line (x,y,z) with a triangle surface, in metres.

    Duplicate edge/vertex hits are merged geometrically. This does not pretend
    a visual triangle mesh is the convex collision hull used by the engine.
    """
    t = np.asarray(triangles, dtype=float)
    p, a, b = t[:, 0], t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    det = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    valid = np.abs(det) > 1e-16
    p, a, b, det = p[valid], a[valid], b[valid], det[valid]
    u = ((y - p[:, 1]) * b[:, 2] - (z - p[:, 2]) * b[:, 1]) / det
    v = (a[:, 1] * (z - p[:, 2]) - a[:, 2] * (y - p[:, 1])) / det
    valid = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
    hits = np.sort((p[:, 0] + u * a[:, 0] + v * b[:, 0])[valid])
    return hits[np.r_[True, np.diff(hits) > 1e-8]] if len(hits) else hits


def surface_distances(points, triangles, *, batch_size=32):
    """Exact distances at supplied points, not a continuous surface-error bound.

    Projections onto faces, edges and vertices all participate. Batching bounds
    temporary memory without reducing the input mesh or the sample set.
    """
    points, triangles = np.asarray(points, dtype=float), np.asarray(triangles, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or not len(triangles)
        or not np.isfinite(points).all()
        or not np.isfinite(triangles).all()
        or type(batch_size) is not int
        or batch_size <= 0
    ):
        raise ValueError("Use finite points [N,3], triangles [T,3,3], and a positive batch size")
    a, b, c = (triangles[:, i] for i in range(3))
    u, v = b - a, c - a
    normal = np.cross(u, v)
    uu, vv, uv = np.sum(u * u, axis=1), np.sum(v * v, axis=1), np.sum(u * v, axis=1)
    determinant = np.sum(normal * normal, axis=1)
    safe = np.where(determinant > 0, determinant, 1)
    result = np.empty(len(points))
    for start in range(0, len(points), batch_size):
        p = points[start : start + batch_size, None]
        w = p - a
        wu, wv = np.sum(w * u, axis=2), np.sum(w * v, axis=2)
        alpha, beta = (vv * wu - uv * wv) / safe, (uu * wv - uv * wu) / safe
        inside = (determinant > 0) & (alpha >= 0) & (beta >= 0) & (alpha + beta <= 1)
        squared = np.where(inside, np.sum(w * normal, axis=2) ** 2 / safe, np.inf)
        for first, last in ((a, b), (b, c), (c, a)):
            edge = last - first
            norm = np.sum(edge * edge, axis=1)
            t = np.clip(np.sum((p - first) * edge, axis=2) / np.where(norm > 0, norm, 1), 0, 1)
            squared = np.minimum(squared, np.sum((p - first - t[:, :, None] * edge) ** 2, axis=2))
        result[start : start + len(p)] = np.sqrt(squared.min(axis=1))
    return result
