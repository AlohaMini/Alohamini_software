"""Conform interior CAD-face mesh edges without welding or filling holes."""

from __future__ import annotations

import numpy as np

from .collision import topology
from .meshes import Mesh


def conform_face_mesh(mesh, triangle_faces, protected, face_tolerances, guard):
    """Subdivide coarse edges to match existing nodes on the same native face.

    Only closed boundary loops confined to a face interior and a native-tolerance
    corridor around a straight segment qualify. Original CAD boundary nodes are
    protected. Every replacement subdivides an existing triangle; no new vertex,
    cap, proximity weld or inferred trim surface is introduced.
    """
    directed = np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    _, inverse, counts = np.unique(
        np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, counts.cumsum()[:-1]]
    uses = order[starts[counts == 1]]
    boundary = directed[uses]
    owners = uses % len(mesh.faces)
    adjacent = {}
    for index, (a, b) in enumerate(boundary):
        adjacent.setdefault(int(a), []).append(index)
        adjacent.setdefault(int(b), []).append(index)
    remaining, loops, replacements, reports = set(adjacent), [], {}, []
    while remaining:
        pending, vertices, edges = [min(remaining)], set(), set()
        while pending:
            vertex = pending.pop()
            if vertex in vertices:
                continue
            vertices.add(vertex)
            for edge in adjacent[vertex]:
                edges.add(edge)
                pending.extend(int(v) for v in boundary[edge] if v not in vertices)
        remaining -= vertices
        loops.append((np.array(sorted(vertices)), np.array(sorted(edges))))
    for vertices, edges in loops:
        face_ids = np.unique(np.asarray(triangle_faces)[owners[edges]])
        row = {
            "boundary_vertices": len(vertices),
            "boundary_edges": len(edges),
            "cad_faces": face_ids.tolist(),
            "subdivided_triangles": 0,
        }
        reports.append(row)
        if (
            len(face_ids) != 1
            or np.asarray(protected)[vertices].any()
            or any(len(adjacent[int(v)]) != 2 for v in vertices)
        ):
            row["status"] = "unresolved: protected boundary, mixed faces or branched loop"
            continue
        face = int(face_ids[0])
        tolerance = float(face_tolerances[face])
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Native face tolerances must be finite and nonnegative")
        points = mesh.vertices[vertices]
        axis = int(np.ptp(points, axis=0).argmax())
        first, last = points[points[:, axis].argmin()], points[points[:, axis].argmax()]
        direction = last - first
        length_squared = float(direction @ direction)
        if length_squared <= guard**2:
            row["status"] = "unresolved: segment is below arithmetic resolution"
            continue
        parameters = (points - first) @ direction / length_squared
        residual = np.linalg.norm(points - (first + parameters[:, None] * direction), axis=1)
        row.update(
            native_face_tolerance_m=tolerance,
            arithmetic_guard_m=float(guard),
            maximum_line_residual_m=float(residual.max()),
            maximum_line_distance_bound_m=float(residual.max() + guard),
        )
        if residual.max() + guard > tolerance:
            row["status"] = "unresolved: loop is not an interior edge subdivision"
            continue
        pending = {}
        maximum_offset = 0.0
        for edge in edges:
            a, b = boundary[edge]
            triangle_id = int(owners[edge])
            first, last = mesh.vertices[[a, b]]
            lo, hi = sorted((first[axis], last[axis]))
            inserted = vertices[(points[:, axis] > lo + guard) & (points[:, axis] < hi - guard)]
            if not len(inserted):
                continue
            vector = last - first
            t = (mesh.vertices[inserted] - first) @ vector / (vector @ vector)
            offsets = np.linalg.norm(
                mesh.vertices[inserted] - (first + t[:, None] * vector), axis=1
            )
            if (
                offsets.max() + guard > tolerance
                or (t <= 0).any()
                or (t >= 1).any()
                or triangle_id in pending
                or triangle_id in replacements
            ):
                pending = None
                break
            inserted = inserted[np.argsort(t, kind="stable")]
            chain = np.r_[a, inserted, b]
            if (np.linalg.norm(np.diff(mesh.vertices[chain], axis=0), axis=1) <= guard).any():
                pending = None
                break
            original = mesh.faces[triangle_id]
            third = original[(int(np.flatnonzero(original == a)[0]) + 2) % 3]
            new = np.column_stack([chain[:-1], chain[1:], np.full(len(chain) - 1, third)])
            triangle = mesh.vertices[original]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            new_points = mesh.vertices[new]
            new_normals = np.cross(
                new_points[:, 1] - new_points[:, 0], new_points[:, 2] - new_points[:, 0]
            )
            if (new_normals @ normal <= 0).any():
                pending = None
                break
            pending[triangle_id] = new
            maximum_offset = max(maximum_offset, float(offsets.max() + guard))
        if not pending:
            row["status"] = "unresolved: ambiguous or nonconforming triangle subdivision"
            continue
        replacements.update(pending)
        row.update(
            status="subdivided",
            subdivided_triangles=len(pending),
            maximum_subdivision_surface_change_bound_m=maximum_offset,
        )
    faces = np.vstack([replacements.get(i, f[None]) for i, f in enumerate(mesh.faces)])
    return Mesh(mesh.vertices, faces), {
        "algorithm": "native face-interior boundary conformity by existing-node subdivision",
        "source_triangles": len(mesh.faces),
        "output_triangles": len(faces),
        "new_vertices": 0,
        "vertex_coordinate_displacement_m": 0.0,
        "protected_CAD_boundaries_changed": False,
        "hole_filling": False,
        "maximum_subdivision_surface_change_bound_m": max(
            (r.get("maximum_subdivision_surface_change_bound_m", 0) for r in reports), default=0
        ),
        "loops": reports,
    }


def split_index_fans(mesh):
    """Separate corner fans while keeping every oriented triangle unchanged.

    Coincident or oppositely duplicated faces are ambiguous material input and
    are rejected. This changes connectivity only; no faces, vertices, normals,
    thickness or solid interiors are invented. Native solid checks follow.
    """
    _, coordinate_ids = np.unique(mesh.vertices, axis=0, return_inverse=True)
    triangles = np.sort(coordinate_ids[mesh.faces], axis=1)
    if len(np.unique(triangles, axis=0)) != len(triangles):
        raise ValueError("Duplicated triangle surfaces require explicit source ownership")
    edges = {}
    for face_id, face in enumerate(mesh.faces):
        for corner in range(3):
            a, b = int(face[corner]), int(face[(corner + 1) % 3])
            edges.setdefault(tuple(sorted((a, b))), []).append((face_id, a < b))
    parent = np.arange(mesh.faces.size)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    branched = 0
    for vertices, uses in edges.items():
        forward = [f for f, direction in uses if direction]
        reverse = [f for f, direction in uses if not direction]
        if not forward or len(forward) != len(reverse):
            raise ValueError("Index splitting cannot repair open or inconsistently oriented edges")
        branched += len(uses) > 2
        for first, last in zip(forward, reverse, strict=True):
            for vertex in vertices:
                a = first * 3 + int(np.flatnonzero(mesh.faces[first] == vertex)[0])
                b = last * 3 + int(np.flatnonzero(mesh.faces[last] == vertex)[0])
                parent[find(a)] = find(b)
    roots, inverse = np.unique([find(i) for i in range(len(parent))], return_inverse=True)
    result = Mesh(mesh.vertices[mesh.faces.ravel()[roots]], inverse.reshape(-1, 3))
    if not np.array_equal(result.triangles, mesh.triangles) or any(topology(result).values()):
        raise ValueError("Corner connectivity remains ambiguous without changing source surfaces")
    return result, {
        "algorithm": "oriented edge pairing and separate vertex corner fans",
        "pairing": "stable original face-index order",
        "branched_edges": int(branched),
        "input_vertices": len(mesh.vertices),
        "output_vertices": len(result.vertices),
        "source_oriented_triangles_identical": True,
        "vertex_coordinate_displacement_m": 0.0,
        "solid_interiors_inferred": False,
    }
