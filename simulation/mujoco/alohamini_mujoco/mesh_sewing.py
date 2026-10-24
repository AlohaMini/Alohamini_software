"""Explicitly budgeted sewing of existing, uniquely paired mesh boundary edges."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .collision import topology
from .meshes import Mesh


def sew_boundary_edges(mesh, budget_m):
    """Identify reverse boundary edges, keeping every original triangle.

    This is an explicit geometry revision, not a float32 encoding repair. Each
    whole directed boundary edge needs exactly one reciprocal, reverse partner.
    Neither caps, edge subdivisions, nearest-surface snapping nor arbitrary
    proximity welds participate. Every triangle retains its original orientation
    and has a corresponding source triangle, giving a continuous surface bound.
    """
    if type(budget_m) not in (int, float) or not math.isfinite(budget_m) or budget_m <= 0:
        raise ValueError("Boundary sewing needs an explicit positive finite metre budget")
    vertices, faces = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or not len(faces)
        or faces.dtype.kind not in "iu"
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Boundary sewing requires finite indexed triangles")
    source = Mesh(vertices, faces)
    original_topology = topology(source)
    if any(
        original_topology[key]
        for key in (
            "nonmanifold_edges",
            "repeated_directed_edges",
            "zero_area_faces",
        )
    ):
        raise ValueError("Boundary sewing requires an oriented surface without branched edges")
    directed = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1),
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    uses = np.flatnonzero(counts[inverse] == 1)
    boundary = directed[uses]
    starts, ends = vertices[boundary[:, 0]], vertices[boundary[:, 1]]
    tree, matches = cKDTree(starts), []
    guard = 128 * np.finfo(float).eps * max(1.0, float(np.abs(vertices).max()))
    # Endpoints may each move by at most budget; do not search unrelated
    # vertices or pair only a fraction of a source edge.
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        candidates = [
            other
            for other in tree.query_ball_point(end, 2 * budget_m)
            if other != index and np.linalg.norm(ends[other] - start) <= 2 * budget_m
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Boundary edge has no unique complete reverse partner; "
                "missing faces and ambiguous seams are not filled"
            )
        matches.append(candidates[0])
    parent = np.arange(len(vertices))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    maximum_gap, pairs = 0.0, []
    for first, last in enumerate(matches):
        if matches[last] != first:
            raise ValueError("Boundary edge partners must be reciprocal")
        if last < first:
            continue
        for a, b in zip(boundary[first], boundary[last][::-1], strict=True):
            maximum_gap = max(maximum_gap, float(np.linalg.norm(vertices[a] - vertices[b])))
            parent[find(int(a))] = find(int(b))
        pairs.append([int(uses[first]), int(uses[last])])
    roots, mapping = np.unique([find(i) for i in range(len(vertices))], return_inverse=True)
    # Stable vertex order makes the representative independent of the edge
    # traversal. Only matched boundary vertices can enter the same group.
    order = np.argsort(mapping, kind="stable")
    groups = np.split(order, np.bincount(mapping).cumsum()[:-1])
    sewn_vertices = np.array([vertices[group].mean(0) for group in groups])
    result = Mesh(sewn_vertices, mapping[faces])
    original_triangles, sewn_triangles = source.triangles, result.triangles
    movement = np.linalg.norm(sewn_triangles - original_triangles, axis=2)
    bound = float(movement.max() + guard) if pairs else 0.0
    if bound > budget_m:
        raise ValueError("The combined seam displacement exceeds the declared surface budget")
    normals = np.cross(
        original_triangles[:, 1] - original_triangles[:, 0],
        original_triangles[:, 2] - original_triangles[:, 0],
    )
    new_normals = np.cross(
        sewn_triangles[:, 1] - sewn_triangles[:, 0],
        sewn_triangles[:, 2] - sewn_triangles[:, 0],
    )
    if (np.einsum("ij,ij->i", normals, new_normals) <= 0).any():
        raise ValueError("Boundary sewing would collapse or reverse a source triangle")
    _, coordinate_ids = np.unique(result.vertices, axis=0, return_inverse=True)
    if len(np.unique(np.sort(coordinate_ids[result.faces], axis=1), axis=0)) != len(faces):
        raise ValueError("Boundary sewing would produce duplicated triangle surfaces")
    if any(topology(result).values()):
        raise ValueError("Paired boundaries did not produce a closed oriented surface")
    return result, {
        "algorithm": "unique reciprocal oriented boundary-edge sewing-v1",
        "geometry_revision": True,
        "budget_m": float(budget_m),
        "boundary_edges": len(boundary),
        "paired_edges": len(pairs),
        "edge_use_pairs": pairs,
        "source_triangles": len(faces),
        "output_triangles": len(result.faces),
        "merged_boundary_vertices": len(vertices) - len(roots),
        "maximum_source_endpoint_gap_m": maximum_gap,
        "maximum_vertex_displacement_m": float(movement.max()),
        "continuous_surface_distance_bound_m": bound,
        "surface_bound_method": "same-triangle barycentric correspondence, both directions",
        "float64_rounding_guard_m": guard,
        "source_face_order_preserved": True,
        "source_orientations_preserved": True,
        "new_vertices": 0,
        "new_triangles": 0,
        "hole_filling": False,
        "per_coordinate_rounding_intervals_checked": False,
        "accuracy_scope": "bounded revision of stored mesh; no analytic or physical accuracy claim",
    }
