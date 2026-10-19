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
