"""Offline convex-cover selection with continuous CAD material certificates."""

from __future__ import annotations

import hashlib
import math
from importlib.metadata import version

import numpy as np
from scipy.spatial import ConvexHull

from .collision import quantized_convex, validate_convex
from .frames import rigid_transform, rotation_axis
from .meshes import surface_distances
from .solid_geometry import (
    _solid,
    _spatial_union,
    clip_convex_halfspace,
    convex_boundary,
    metric_kernel,
)


def _require_boolean(solid):
    import manifold3d as manifold

    if solid.status() != manifold.Error.NoError:
        raise ValueError(f"Native convex-cover Boolean failed: {solid.status()}")
    return solid


def _expanded(piece, kernel):
    return convex_boundary((piece.vertices[:, None] + kernel.vertices).reshape(-1, 3))


def bounded_convex_mesh(piece, max_vertices=128):
    """Build an inner convex approximation with a measured Euclidean error.

    Every retained point is an original float32 vertex. Distance to a convex
    set is convex, so its maximum on the original hull occurs at a vertex.
    Testing all source vertices therefore bounds the complete convex solid,
    including its triangle interiors. The complete CAD certificate still
    decides whether this candidate can be used in the final cover.
    """
    validate_convex(piece)
    if type(max_vertices) is not int or max_vertices < 6:
        raise ValueError("The convex mesh vertex budget must be an integer of at least six")
    if len(piece.vertices) <= max_vertices:
        return piece, 0.0
    vertices = piece.vertices
    selected = set(np.argmin(vertices, axis=0)) | set(np.argmax(vertices, axis=0))
    if np.linalg.matrix_rank(vertices[sorted(selected)] - vertices[min(selected)]) < 3:
        from scipy.linalg import qr

        _, _, pivots = qr((vertices - vertices[0]).T, pivoting=True)
        selected.update([0, *pivots[:3]])
    while len(selected) < max_vertices:
        hull = ConvexHull(vertices[sorted(selected)])
        violation = np.max(vertices @ hull.equations[:, :3].T + hull.equations[:, 3], axis=1)
        violation[list(selected)] = -np.inf
        next_vertex = int(np.argmax(violation))
        if violation[next_vertex] <= 0:
            break
        selected.add(next_vertex)
    result = quantized_convex(convex_boundary(vertices[sorted(selected)]))[0]
    hull = ConvexHull(result.vertices)
    outside = np.any(vertices @ hull.equations[:, :3].T + hull.equations[:, 3] > 0, axis=1)
    error = (
        float(surface_distances(vertices[outside], result.triangles).max())
        if outside.any()
        else 0.0
    )
    return result, error


def _prism(vertices, normal, first, last, *, inset=0):
    result = convex_boundary(np.vstack((vertices + first * normal, vertices + last * normal)))
    if inset:
        # Only the footprint shrinks, by the float32 arithmetic guard. The two
        # planes parallel to the contact face retain their declared offsets.
        for equation in ConvexHull(result.vertices).equations:
            if abs(equation[:3] @ normal) < 0.5:
                result = clip_convex_halfspace(result, equation[:3], -equation[3] - inset)
                if result is None:
                    raise ValueError("The contact footprint is below the arithmetic guard")
    return result


def _subtract_convex(piece, region, guard_m):
    """Partition C minus P into convex cells without cutting material outside P."""
    inside = piece
    outside = []
    for equation in ConvexHull(region.vertices).equations:
        signed = inside.vertices @ equation[:3] + equation[3]
        if signed.max() <= -guard_m:
            continue
        if signed.min() >= guard_m:
            outside.append(inside)
            break
        if signed.max() > guard_m:
            fragment = clip_convex_halfspace(inside, -equation[:3], equation[3])
            if fragment is not None:
                outside.append(quantized_convex(fragment)[0])
        if signed.min() >= -guard_m:
            break
        inside = clip_convex_halfspace(inside, equation[:3], -equation[3])
        if inside is None:
            break
    return outside


def _contact_candidates(source, candidates, contacts, tolerance_m, guard_m):
    import manifold3d as manifold

    reference = _solid(source)
    depth = 2 * np.linalg.norm(np.ptp(source.vertices, axis=0)) + tolerance_m
    layers, reports, approach_regions = [], [], []
    for contact in contacts:
        vertices = np.asarray(contact["vertices"], dtype=float)
        triangles = np.asarray(contact["faces"], dtype=int)
        normal = np.asarray(contact["normal"], dtype=float)
        accuracy = float(contact["tolerance_m"])
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or triangles.ndim != 2
            or triangles.shape[1:] != (3,)
            or not len(triangles)
            or normal.shape != (3,)
            or not np.isfinite(vertices).all()
            or not np.isfinite(normal).all()
            or not np.isfinite(accuracy)
            or not guard_m < accuracy <= tolerance_m
            or not np.isclose(normal @ normal, 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("Provide a planar SI contact footprint and a unit outward normal")
        offset = float(vertices[0] @ normal)
        if np.ptp(vertices @ normal) > guard_m:
            raise ValueError("Contact footprint is not planar in the declared link frame")
        area = (
            np.linalg.norm(
                np.cross(
                    vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                    vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
                ),
                axis=1,
            ).sum()
            / 2
        )
        layer = _prism(vertices, normal, -tolerance_m, 0)
        if not np.isclose(_solid(layer).volume() / tolerance_m, area, rtol=1e-7, atol=0):
            raise ValueError("Contact footprints must be convex and contain no holes")
        free = _solid(_prism(vertices, normal, accuracy, depth, inset=guard_m))
        overlap = _require_boolean(
            manifold.Manifold.batch_boolean([reference, free], manifold.OpType.Intersect)
        )
        if not overlap.is_empty():
            raise ValueError("The declared contact approach region contains CAD material")
        setback = tolerance_m + guard_m
        reserved_mesh = _prism(vertices, normal, -setback, depth, inset=guard_m)
        reserved = _solid(reserved_mesh)
        adapted = []
        for piece in candidates:
            if np.max(piece.vertices @ normal) > offset - setback:
                overlap = _require_boolean(
                    manifold.Manifold.batch_boolean(
                        [_solid(piece), reserved], manifold.OpType.Intersect
                    )
                )
                if not overlap.is_empty():
                    adapted.extend(_subtract_convex(piece, reserved_mesh, guard_m))
                    continue
            adapted.append(piece)
        candidates = adapted
        layer = quantized_convex(layer)[0]
        layers.append(layer)
        approach_regions.append(
            _solid(_prism(vertices, normal, accuracy, depth, inset=2 * guard_m))
        )
        reports.append(
            {
                "name": contact["name"],
                "normal": normal.tolist(),
                "plane_offset_m": offset,
                "tolerance_m": accuracy,
                "footprint_arithmetic_inset_m": guard_m,
                "area_m2": float(area),
                "source_approach_region_clear": True,
                "structural_setback_m": setback,
                "contact_layer_thickness_m": tolerance_m,
                "maximum_contact_plane_error_m": float(
                    np.max(
                        np.abs(
                            layer.vertices[
                                np.abs(layer.vertices @ normal - offset) < tolerance_m / 2
                            ]
                            @ normal
                            - offset
                        )
                    )
                ),
            }
        )
    return candidates, layers, reports, approach_regions


def select_convex_cover(
    source, candidates, *, tolerance_m, contacts=(), max_iterations=32, prepare_mesh=None
):
    """Select overlapping convex cells and certify the complete reference solid.

    Witness coverage proposes a subset. Native solid differences certify that
    subset continuously; residual vertices and face centres become additional
    witnesses until the certificate passes. Candidate generation and greedy
    selection do not claim a globally minimum convex-cover cardinality.

    Reverse inclusion is checked per convex piece, so numerical degeneracies
    at internal Boolean seams cannot replace the physical material check.
    This function is for asset preparation and never runs in a physics step.
    """
    import manifold3d as manifold

    reference = _solid(source)
    if (
        not np.isfinite(tolerance_m)
        or tolerance_m <= 0
        or type(max_iterations) is not int
        or max_iterations <= 0
        or not candidates
    ):
        raise ValueError("Provide convex candidates, positive metric accuracy and a work budget")
    candidates = list(candidates)
    for piece in candidates:
        validate_convex(piece)
    rounding = float(np.linalg.norm(np.spacing(np.abs(source.vertices).max(0).astype(np.float32))))
    guard = max(8 * rounding, 32 * reference.get_tolerance())
    if tolerance_m <= guard:
        raise ValueError("Geometry accuracy is below the source arithmetic guard")
    input_count = len(candidates)
    bounded = [bounded_convex_mesh(piece) for piece in candidates]
    candidates = [piece for piece, _ in bounded]
    approximation = max(error for _, error in bounded)
    candidates, layers, contact_report, approach_regions = _contact_candidates(
        source, candidates, contacts, tolerance_m, guard
    )
    bounded = [bounded_convex_mesh(piece) for piece in candidates]
    candidates = [piece for piece, _ in bounded]
    contact_approximation = max((error for _, error in bounded), default=0.0)
    if prepare_mesh is not None:
        candidates = [p for piece in candidates for p in prepare_mesh(piece)]
        prepared_layers = []
        for layer, contact in zip(layers, contact_report, strict=True):
            prepared = prepare_mesh(layer, tolerance_m=contact["tolerance_m"] - guard)
            # The runtime mesh preparation is part of the material certificate,
            # including the more precise functional contact footprint.
            pad_kernel = metric_kernel(contact["tolerance_m"] - guard)
            expanded_pad = _require_boolean(
                manifold.Manifold.batch_boolean(
                    [_solid(_expanded(p, pad_kernel)) for p in prepared], manifold.OpType.Add
                )
            )
            if not _require_boolean(_solid(layer) - expanded_pad).is_empty():
                raise ValueError("Mesh preparation exceeds the contact surface tolerance")
            original_envelope = _solid(_expanded(layer, pad_kernel))
            for actual in prepared:
                if not _require_boolean(_solid(actual) - original_envelope).is_empty():
                    raise ValueError("Mesh preparation exceeds the contact surface tolerance")
            normal = np.asarray(contact["normal"])
            contact["maximum_contact_plane_error_m"] = max(
                abs(float(np.max(p.vertices @ normal)) - contact["plane_offset_m"])
                for p in prepared
            )
            contact["piece_indices"] = list(
                range(len(prepared_layers), len(prepared_layers) + len(prepared))
            )
            prepared_layers.extend(prepared)
        layers = prepared_layers
    for piece in [*candidates, *layers]:
        validate_convex(piece)
        if len(piece.vertices) > 128:
            raise ValueError("Prepared convex mesh exceeds the shared backend vertex budget")
    unique = {hashlib.sha256(p.msh_bytes()).hexdigest(): p for p in candidates}
    candidates = list(unique.values())
    selected = list(range(len(candidates), len(candidates) + len(layers)))
    candidates.extend(layers)
    kernel = metric_kernel(tolerance_m - guard)
    expanded, equations, bounds = [], [], []
    for piece in candidates:
        mesh = _expanded(piece, kernel)
        expanded.append(_solid(mesh))
        equations.append(ConvexHull(mesh.vertices).equations)
        bounds.append((mesh.vertices.min(0), mesh.vertices.max(0)))

    def coverage(points):
        result = np.zeros((len(candidates), len(points)), dtype=bool)
        for i, (planes, (lower, upper)) in enumerate(zip(equations, bounds, strict=True)):
            ids = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
            result[i, ids] = np.all(points[ids] @ planes[:, :3].T + planes[:, 3] <= 0, axis=1)
        return result

    points = np.unique(np.vstack((source.vertices, source.triangles.mean(1))), axis=0)
    matrix = coverage(points)
    history = []
    for _ in range(max_iterations):
        uncovered = (
            ~np.any(matrix[selected], axis=0) if selected else np.ones(len(points), dtype=bool)
        )
        while uncovered.any():
            scores = matrix[:, uncovered].sum(1)
            scores[selected] = 0
            best = int(np.argmax(scores))
            if not scores[best]:
                example = points[np.flatnonzero(uncovered)[0]].tolist()
                raise ValueError(
                    f"The admissible convex cells do not cover the CAD witnesses at {example} m"
                )
            selected.append(best)
            uncovered &= ~matrix[best]
        envelope = _require_boolean(
            manifold.Manifold.batch_boolean([expanded[i] for i in selected], manifold.OpType.Add)
        )
        residual = _require_boolean(reference - envelope)
        history.append({"pieces": len(selected), "uncovered_volume_m3": float(residual.volume())})
        if residual.is_empty():
            break
        mesh = residual.to_mesh64()
        vertices = np.asarray(mesh.vert_properties)[:, :3]
        triangles = np.asarray(mesh.tri_verts)
        centres = vertices[triangles].mean(1)
        normals = np.cross(
            vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
            vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
        )
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0
        inward_step = max(
            8 * residual.get_tolerance(),
            64 * np.finfo(float).eps * max(1, np.abs(vertices).max()),
        )
        interior = centres[valid] - normals[valid] / lengths[valid, None] * inward_step
        extra = np.unique(np.vstack((vertices, centres, interior)), axis=0)
        matrix = np.concatenate((matrix, coverage(extra)), axis=1)
        points = np.vstack((points, extra))
    else:
        raise ValueError("Convex-cover refinement did not reach a continuous certificate")
    result = [candidates[i] for i in selected]
    enclosure_kernel = metric_kernel(tolerance_m - reference.get_tolerance())
    source_envelope = _require_boolean(reference.minkowski_sum(_solid(enclosure_kernel)))
    for piece in result:
        if not _require_boolean(_solid(piece) - source_envelope).is_empty():
            raise ValueError("A selected collider exceeds the CAD material envelope")
    for contact, free in zip(contact_report, approach_regions, strict=True):
        normal = np.asarray(contact["normal"])
        front = contact["plane_offset_m"] + contact["tolerance_m"]
        for piece in result:
            if np.max(piece.vertices @ normal) <= front:
                continue
            overlap = _require_boolean(
                manifold.Manifold.batch_boolean([_solid(piece), free], manifold.OpType.Intersect)
            )
            if not overlap.is_empty():
                raise ValueError("A selected collider obstructs the contact approach region")
        contact["approach_region_clear"] = True
    return result, {
        "algorithm": "counterexample-guided convex-cover selection",
        "boolean_kernel": "manifold3d",
        "boolean_kernel_version": version("manifold3d"),
        "input_candidates": input_count,
        "admissible_candidates": len(candidates),
        "output_pieces": len(result),
        "maximum_convex_vertices": 128,
        "vertex_reduction": {
            "initial_maximum_error_m": approximation,
            "after_contact_maximum_error_m": contact_approximation,
        },
        "tolerance_m": tolerance_m,
        "arithmetic_guard_m": guard,
        "native_boolean_tolerance_m": float(
            max(
                reference.get_tolerance(), source_envelope.get_tolerance(), envelope.get_tolerance()
            )
        ),
        "source_inside_expanded_collision": True,
        "each_collider_inside_expanded_source": True,
        "contacts": contact_report,
        "refinement": history,
    }


def certify_rotating_clearance(
    fixed, moving, T_fixed_joint, axis, limits, *, accuracy_m=0.00005, max_evaluations=20000
):
    """Bound clearance over a complete revolute-joint interval.

    A point at radius R moves at most 2R sin(delta/2) under a rotation delta.
    This bounds the change in minimum boundary distance between samples. An
    initial solid-intersection check and positive interval bounds certify that
    the two printed parts stay disjoint, conditional on native kernel accuracy.
    """
    import manifold3d as manifold

    pose = rigid_transform(T_fixed_joint)
    axis = np.asarray(axis, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (
        axis.shape != (3,)
        or limits.shape != (2,)
        or not np.isfinite(axis).all()
        or not np.isfinite(limits).all()
        or not np.isclose(axis @ axis, 1, rtol=0, atol=1e-12)
        or limits[0] >= limits[1]
        or not np.isfinite(accuracy_m)
        or accuracy_m <= 0
        or type(max_evaluations) is not int
        or max_evaluations <= 0
    ):
        raise ValueError(
            "Provide a rigid joint frame, unit axis, radian limits and metric accuracy"
        )
    fixed_solid = _spatial_union([_solid(piece) for piece in fixed])
    moving_solid = _spatial_union([_solid(piece) for piece in moving])
    vertices = np.vstack([piece.vertices for piece in moving])
    radial = vertices - np.outer(vertices @ axis, axis)
    radius = float(np.linalg.norm(radial, axis=1).max())
    native_budget = max(fixed_solid.get_tolerance(), moving_solid.get_tolerance())

    def placed(angle):
        turn = np.eye(4)
        turn[:3, :3] = rotation_axis(axis, angle)
        return moving_solid.transform((pose @ turn)[:3])

    initial = placed(limits[0])
    overlap = _require_boolean(
        manifold.Manifold.batch_boolean([fixed_solid, initial], manifold.OpType.Intersect)
    )
    if not overlap.is_empty():
        raise ValueError("The printed jaws intersect at the joint limit")
    extent = (
        np.linalg.norm(np.vstack([p.vertices for p in fixed]), axis=1).max()
        + np.linalg.norm(pose[:3, 3])
        + np.linalg.norm(vertices, axis=1).max()
        + accuracy_m
    )
    initial_gap = fixed_solid.min_gap(initial, float(extent))
    search = initial_gap + accuracy_m
    sampled = [(initial_gap, float(limits[0]))]
    intervals, bounds = [(float(limits[0]), float(limits[1]))], []
    while intervals:
        if len(sampled) >= max_evaluations:
            raise ValueError("Joint-clearance proof exceeded its evaluation budget")
        lower, upper = intervals.pop()
        angle = (lower + upper) / 2
        gap = fixed_solid.min_gap(placed(angle), search)
        uncertainty = 2 * radius * math.sin(min((upper - lower) / 4, math.pi / 2))
        bound = gap - uncertainty - 2 * native_budget
        sampled.append((gap, angle))
        if gap <= 2 * native_budget:
            raise ValueError(f"Printed-jaw clearance is not positive at {angle} rad")
        if bound > 0 and uncertainty <= accuracy_m:
            bounds.append(bound)
        else:
            intervals.extend(((lower, angle), (angle, upper)))
    gap, angle = min(sampled)
    return {
        "algorithm": "rotational Hausdorff interval bound",
        "q_range_rad": limits.tolist(),
        "axis": axis.tolist(),
        "T_fixed_moving_zero": pose.tolist(),
        "rotation_radius_bound_m": radius,
        "sample_count": len(sampled),
        "interval_count": len(bounds),
        "minimum_sampled_gap_m": gap,
        "minimum_at_q_rad": angle,
        "certified_gap_lower_bound_m": min(bounds),
        "distance_uncertainty_m": accuracy_m,
        "native_boolean_tolerance_m": native_budget,
        "passed": True,
        "scope": "continuous printed-jaw clearance; hardware proxies are retained separately",
    }
