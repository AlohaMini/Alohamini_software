"""Offline structural envelopes with protected contact and motion volumes.

Structural envelopes may fill mounting holes and reinforcing pockets. They
retain material coverage, the declared contact patch and articulation clearance;
they do not claim to reproduce every CAD cavity.
"""

import heapq
import itertools
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from .collision import quantized_convex, validate_convex
from .convex_cover import _expanded, _require_boolean
from .interop import obj_bytes
from .meshes import Mesh, surface_distances
from .solid_geometry import _solid, clip_convex_halfspace, convex_boundary, metric_kernel


def prepare_shared_convex(piece, *, tolerance_m=None):
    """Bake PhysX preparation into the shared source and check its fixed point.

    Cooking may merge near-coplanar points even below the vertex limit. The
    resulting source must be certified against CAD before either backend loads
    it. Millimetre transport matches the generated OBJ files and URDF scales.
    """
    import sapien

    original = piece
    for _ in range(8):
        # Use the actual OBJ import path: the file loader can reorder vertices,
        # which changes near-coplanar QuickHull choices compared with array input.
        with tempfile.TemporaryDirectory(prefix="alohamini-convex-") as temporary:
            path = Path(temporary) / "piece.obj"
            path.write_bytes(obj_bytes(Mesh(piece.vertices.astype(float) * 1000, piece.faces)))
            shape = sapien.physx.PhysxCollisionShapeConvexMesh(
                str(path), [0.001] * 3, sapien.physx.get_default_material()
            )
        actual = quantized_convex(
            convex_boundary(np.asarray(shape.get_vertices()).astype(float) / 1000)
        )[0]
        error = max(
            float(surface_distances(first.vertices, second.triangles).max())
            for first, second in ((piece, actual), (actual, piece))
        )
        if error < 0.000001:
            break
        piece = actual
    else:
        raise ValueError("PhysX convex preparation did not reach a stable shared mesh")
    if (
        tolerance_m is None
        or max(
            float(surface_distances(first.vertices, second.triangles).max())
            for first, second in ((original, piece), (piece, original))
        )
        < tolerance_m
    ):
        return [piece]
    # A large flat contact pad can lose curved boundary detail during cooking.
    # Split its footprint along its longest extent until each cooked patch
    # meets the declared contact tolerance; retain the original plane.
    axis = np.eye(3)[np.argmax(np.ptp(original.vertices, axis=0))]
    offset = float((original.vertices @ axis).min() + (original.vertices @ axis).max()) / 2
    halves = [clip_convex_halfspace(original, sign * axis, sign * offset) for sign in (1, -1)]
    return [
        p
        for half in halves
        for p in prepare_shared_convex(quantized_convex(half)[0], tolerance_m=tolerance_m)
    ]


def _clip_polygon(polygon, normal, offset):
    if not len(polygon):
        return polygon
    result = []
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        a, b = first @ normal - offset, second @ normal - offset
        if a <= 0:
            result.append(first)
        if (a < 0 < b) or (b < 0 < a):
            result.append(first + (second - first) * (a / (a - b)))
    return np.asarray(result).reshape(-1, 3)


def _outside_convex(polygon, equations):
    remainder, result = polygon, []
    for normal, offset in ((row[:3], -row[3]) for row in np.asarray(equations, dtype=float)):
        outside = _clip_polygon(remainder, -normal, -offset)
        if len(outside):
            result.append(outside)
        remainder = _clip_polygon(remainder, normal, offset)
        if not len(remainder):
            return result
    return result


def _outside_box(polygon, lower, upper):
    return _outside_convex(
        polygon,
        [
            [*list(np.eye(3)[axis] * sign), offset]
            for axis in range(3)
            for sign, offset in ((-1, lower[axis]), (1, -upper[axis]))
        ],
    )


def cylinder_distance(points, radius, z_range):
    """Signed distance to a source-Z cylinder, in metres."""
    lower, upper = z_range
    distance = np.c_[
        np.linalg.norm(points[:, :2], axis=1) - radius,
        np.abs(points[:, 2] - (lower + upper) / 2) - (upper - lower) / 2,
    ]
    return np.linalg.norm(np.maximum(distance, 0), axis=1) + np.minimum(distance.max(1), 0)


def certify_cylinder_box_cover(source, radius, z_range, boxes, *, hulls=(), tolerance_m):
    """Cover complete triangles by an axial cylinder, boxes and convex hulls.

    Subtract each polyhedron from the triangle polygon. Distance to a convex
    solid is convex, so its maximum on each residual polygon is bounded by
    its vertex values. Original triangle vertices alone are insufficient to
    establish coverage by a union of primitives.
    """
    vertices = source.vertices.astype(float)
    bounds = np.asarray(boxes, dtype=float).reshape(-1, 2, 3)
    if not (
        np.isfinite(vertices).all()
        and np.isfinite([radius, *z_range, tolerance_m]).all()
        and radius > 0
        and tolerance_m > 0
        and z_range[0] < z_range[1]
        and np.isfinite(bounds).all()
        and np.all(bounds[:, 0] < bounds[:, 1])
    ):
        raise ValueError("Require a positive radius, tolerance and ordered cylinder ends")
    covered = (cylinder_distance(vertices, radius, z_range) <= 0)[source.faces].all(1)
    for lower, upper in bounds:
        inside = np.all(vertices >= lower, axis=1) & np.all(vertices <= upper, axis=1)
        covered |= inside[source.faces].all(1)
    planes = [ConvexHull(hull.vertices).equations for hull in hulls]
    for equations in planes:
        inside = np.all(vertices @ equations[:, :3].T + equations[:, 3] <= 0, axis=1)
        covered |= inside[source.faces].all(1)
    candidates = np.flatnonzero(~covered)
    maximum, count = 0.0, 0
    for index in candidates:
        polygons = [vertices[source.faces[index]]]
        for lower, upper in bounds:
            polygons = [
                piece for polygon in polygons for piece in _outside_box(polygon, lower, upper)
            ]
        for equations in planes:
            polygons = [
                piece for polygon in polygons for piece in _outside_convex(polygon, equations)
            ]
        # A residual boundary polygon may span both a cylinder end and an
        # adjacent support. Partition at the actual end planes before bounding
        # the distance to either solid.
        for height in z_range:
            polygons = [
                piece
                for polygon in polygons
                for piece in (
                    _clip_polygon(polygon, np.array([0, 0, 1]), height),
                    _clip_polygon(polygon, np.array([0, 0, -1]), -height),
                )
                if len(piece)
            ]
        for polygon in polygons:
            count += 1
            bound = max(0.0, float(cylinder_distance(polygon, radius, z_range).max()))
            # Coplanar subtraction can leave boundary polygons. Measure their
            # distance to the convex solid as well as to the cylinder; either
            # convex distance bounds every point of the residual polygon.
            for hull, equations in zip(hulls, planes, strict=True):
                outside = np.any(polygon @ equations[:, :3].T + equations[:, 3] > 0, axis=1)
                distance = (
                    float(surface_distances(polygon[outside], hull.triangles).max())
                    if outside.any()
                    else 0.0
                )
                bound = min(bound, distance)
            maximum = max(maximum, bound)
    guard = 256 * np.finfo(float).eps * max(1.0, float(np.abs(vertices).max()))
    if maximum + guard > tolerance_m:
        raise ValueError(f"Primitive union leaves source surfaces uncovered by {maximum:g} m")
    return {
        "algorithm": "triangle-polyhedron subtraction and convex cylinder distance bound",
        "source_triangles": len(source.faces),
        "residual_polygons_checked": count,
        "maximum_source_outside_distance_bound_m": maximum + guard,
        "tolerance_m": tolerance_m,
        "arithmetic_guard_m": guard,
        "passed": True,
    }


def axis_aligned_box(piece):
    lower, upper = piece.vertices.min(0), piece.vertices.max(0)
    corners = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    if len(piece.vertices) == 8 and set(map(tuple, corners)) == set(map(tuple, piece.vertices)):
        pose = np.eye(4)
        pose[:3, 3] = (lower + upper) / 2
        return upper - lower, pose
    return None


def certify_material_coverage(source, pieces, *, tolerance_m):
    """Certify CAD material coverage; intentional support cavity filling is allowed."""
    import manifold3d as manifold

    reference = _solid(source)
    rounding = np.linalg.norm(np.spacing(np.abs(source.vertices).max(0).astype(np.float32)))
    guard = max(32 * reference.get_tolerance(), 8 * float(rounding))
    kernel = metric_kernel(tolerance_m - guard)
    envelope = _require_boolean(
        manifold.Manifold.batch_boolean(
            [_solid(_expanded(piece, kernel)) for piece in pieces], manifold.OpType.Add
        )
    )
    if not _require_boolean(reference - envelope).is_empty():
        raise ValueError("Support envelopes omit CAD material beyond the coverage tolerance")
    return {"source_inside_expanded_collision": True, "tolerance_m": tolerance_m}


def merge_support_envelopes(
    pieces, *, protected_indices, forbidden_regions, obstacles, clearance_m, adjacency_m
):
    """Prefer native boxes, then adjacent convex envelopes, outside protected volumes.

    Obstacles are proposal samples of the opposite jaw's motion. The caller
    must certify the complete joint interval after both jaws are prepared.
    """
    protected = set(protected_indices)
    active = {i: piece for i, piece in enumerate(pieces) if i not in protected}
    for piece in pieces:
        validate_convex(piece)
    bounds = {i: (p.vertices.min(0), p.vertices.max(0)) for i, p in active.items()}
    volumes = {i: ConvexHull(p.vertices).volume for i, p in active.items()}
    obstacle_bounds = [np.asarray(s.bounding_box()).reshape(2, 3) for s in obstacles]
    queue, tested, accepted, invalid = [], 0, 0, 0
    next_id = len(pieces)

    def enqueue(first, last):
        amin, amax = bounds[first]
        bmin, bmax = bounds[last]
        if np.any(amin > bmax + adjacency_m) or np.any(bmin > amax + adjacency_m):
            return
        points = np.vstack((active[first].vertices, active[last].vertices))
        ratio = ConvexHull(points).volume / (volumes[first] + volumes[last])
        heapq.heappush(queue, (ratio, first, last))

    def admissible(piece):
        solid = _solid(piece)
        for forbidden in forbidden_regions:
            if not _require_boolean(solid ^ forbidden).is_empty():
                return False
        lower, upper = piece.vertices.min(0), piece.vertices.max(0)
        for obstacle, box in zip(obstacles, obstacle_bounds, strict=True):
            if np.any(lower > box[1] + clearance_m) or np.any(box[0] > upper + clearance_m):
                continue
            if solid.min_gap(obstacle, clearance_m) < clearance_m:
                return False
            # Surface distance alone cannot reject containment.
            if not _require_boolean(solid ^ obstacle).is_empty():
                return False
        return True

    for first, last in itertools.combinations(active, 2):
        enqueue(first, last)
    while queue:
        _, first, last = heapq.heappop(queue)
        if first not in active or last not in active:
            continue
        points = np.vstack((active[first].vertices, active[last].vertices))
        lower, upper = points.min(0), points.max(0)
        box = convex_boundary(np.array(list(itertools.product(*zip(lower, upper, strict=True)))))
        choices = [quantized_convex(box)[0]]
        try:
            hull = quantized_convex(convex_boundary(points))[0]
        except ValueError:
            # A numerically invalid proposal does not remove either source cell.
            invalid += 1
        else:
            if len(hull.vertices) <= 128:
                choices.append(hull)
        tested += 1
        candidate = next((piece for piece in choices if admissible(piece)), None)
        if candidate is None:
            continue
        del active[first], active[last]
        active[next_id] = candidate
        bounds[next_id] = (candidate.vertices.min(0), candidate.vertices.max(0))
        volumes[next_id] = ConvexHull(candidate.vertices).volume
        for other in active:
            if other != next_id:
                enqueue(other, next_id)
        next_id += 1
        accepted += 1
    result = [pieces[i] for i in sorted(protected)] + list(active.values())
    return result, {
        "algorithm": "adjacent support envelopes with box preference and protected volumes",
        "input_pieces": len(pieces),
        "output_pieces": len(result),
        "native_boxes": sum(axis_aligned_box(p) is not None for p in result),
        "protected_contact_indices": list(range(len(protected))),
        "tested_pairs": tested,
        "accepted_merges": accepted,
        "invalid_proposals": invalid,
        "sampled_clearance_m": clearance_m,
        "cavity_policy": "mounting holes and support pockets may be filled",
        "continuous_motion_certificate_required": True,
    }
