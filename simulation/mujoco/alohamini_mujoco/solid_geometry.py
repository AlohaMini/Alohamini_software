"""Verify closed CAD material and simplify transported convex collision unions."""

from __future__ import annotations

import heapq
import itertools
from importlib.metadata import version

import numpy as np
from scipy.spatial import ConvexHull

from .collision import signed_volume, topology
from .geometry import SurfaceIndex, directed_surface_bounds
from .meshes import Mesh


class InvalidUnionBoundary(ValueError):
    """A native Boolean result cannot be exported as a nondegenerate closed mesh."""


def _solid(mesh):
    try:
        import manifold3d as manifold
    except ImportError as exc:
        raise ValueError("Install the simulation geometry extra for solid union checks") from exc
    if (
        mesh.vertices.ndim != 2
        or mesh.vertices.shape[1:] != (3,)
        or not np.isfinite(mesh.vertices).all()
        or mesh.faces.ndim != 2
        or mesh.faces.shape[1:] != (3,)
        or not np.issubdtype(mesh.faces.dtype, np.integer)
        or not len(mesh.faces)
        or mesh.faces.min() < 0
        or mesh.faces.max() >= len(mesh.vertices)
        or any(topology(mesh).values())
        or signed_volume(mesh) <= 0
    ):
        raise ValueError("Solid comparison requires a closed, consistently oriented positive mesh")
    solid = manifold.Manifold(
        manifold.Mesh64(
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.uint64),
            tolerance=0,
        )
    )
    if solid.status() != manifold.Error.NoError or solid.is_empty() or solid.volume() <= 0:
        raise ValueError(f"Native solid construction failed: {solid.status()}")
    return solid


def union_boundary(pieces):
    """Remove internal partitions and overlaps without filling source openings.

    No vertex merge, remeshing, hole filling or requested simplification is
    performed. Native floating-point Boolean tolerances are returned explicitly.
    A closed output is not a proof that an arbitrary input is self-intersection
    free; source solids still require their own validity and ownership checks.
    """
    if not pieces:
        raise ValueError("A collision union needs at least one complete closed piece")
    solids = [_solid(piece) for piece in pieces]
    import manifold3d as manifold

    union = manifold.Manifold.batch_boolean(solids, manifold.OpType.Add)
    if union.status() != manifold.Error.NoError or union.is_empty() or union.volume() <= 0:
        raise ValueError(f"Native collision union failed: {union.status()}")
    exported = union.to_mesh64()
    boundary = Mesh(
        np.asarray(exported.vert_properties, dtype=np.float64)[:, :3].copy(),
        np.asarray(exported.tri_verts, dtype=np.int64).copy(),
    )
    if any(topology(boundary).values()) or signed_volume(boundary) <= 0:
        raise InvalidUnionBoundary(
            f"Native collision union has an invalid exterior boundary: {topology(boundary)}"
        )
    return (
        boundary,
        union,
        {
            "algorithm": "Manifold float64 Boolean union of complete closed pieces",
            "algorithm_version": version("manifold3d"),
            "input_pieces": len(pieces),
            "input_triangles": sum(len(p.faces) for p in pieces),
            "boundary_triangles": len(boundary.faces),
            "native_tolerance_m": float(
                max([union.get_tolerance()] + [s.get_tolerance() for s in solids])
            ),
            "source_repair": "none; no merge, remesh, hole fill or explicit simplification",
            "union_volume_m3": float(union.volume()),
        },
    )


def compare_solid_union(source, pieces, *, tolerance_m, max_samples=1_000_000, progress=None):
    """Continuous, bidirectional *boundary* distance; never compare internal cuts.

    The bound concerns supplied triangle surfaces in one explicit metric frame.
    Boolean kernel tolerances are included in the reported error budget, not
    treated as a proof of exact real-arithmetic operations or physical accuracy.
    Difference volumes are diagnostics, not substitutes for distance checks.
    """
    if not np.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Use a finite positive metric surface tolerance")
    reference = _solid(source)
    import manifold3d as manifold

    boundary, union, metadata = union_boundary(pieces)
    native_budget = max(metadata["native_tolerance_m"], reference.get_tolerance())
    if tolerance_m <= native_budget:
        raise ValueError("Requested surface tolerance is below the native Boolean tolerance")
    directions = {}
    for name, first, second in (
        ("source_to_union", source, boundary),
        ("union_to_source", boundary, source),
    ):
        callback = None if progress is None else lambda value, name=name: progress(name, value)
        directions[name] = directed_surface_bounds(
            first.triangles,
            SurfaceIndex(second.triangles),
            tolerance_m=tolerance_m - native_budget,
            max_samples=max_samples,
            progress=callback,
        )
    difference_volumes = {}
    for name, first, second in (
        ("source_minus_union_m3", reference, union),
        ("union_minus_source_m3", union, reference),
    ):
        residual = first - second
        if residual.status() != manifold.Error.NoError:
            raise ValueError(f"Native solid difference failed: {residual.status()}")
        difference_volumes[name] = float(residual.volume())
    return boundary, {
        "passed": all(d["status"] == "passed" for d in directions.values()),
        "scope": (
            "continuous triangle boundary comparison; not analytic CAD or physical calibration"
        ),
        "tolerance_m": float(tolerance_m),
        "native_boolean_tolerance_m": float(native_budget),
        "directions": directions,
        "union": metadata,
        "difference_volume_diagnostics": difference_volumes,
    }


def convex_boundary(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    used, indexed = np.unique(faces, return_inverse=True)
    return Mesh(vertices[used], indexed.reshape(-1, 3))


def metric_kernel(radius_m):
    """A convex polytope strictly inside the requested Euclidean ball.

    Inclusion after dilation by this kernel is sufficient, not necessary, for
    the corresponding solid Hausdorff bound. A nonempty difference is therefore
    a failed certificate, not a measured lower bound on the actual distance.
    """
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Use a finite positive metric kernel radius")
    vectors = np.array([v for v in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(v)])
    points = vectors / np.linalg.norm(vectors, axis=1)[:, None]
    points *= np.nextafter(float(radius_m), 0)
    # Direction normalization and the norm itself can each round outward.
    # Move only offending coordinates one representable value toward zero.
    outside = np.linalg.norm(points, axis=1) > radius_m
    while outside.any():
        points[outside] = np.nextafter(points[outside], 0)
        outside = np.linalg.norm(points, axis=1) > radius_m
    return convex_boundary(points)


def certify_solid_distance(source, pieces, *, tolerance_m):
    """Check both solid inclusions using an inscribed metric dilation kernel.

    This handles internal sub-resolution cracks between convex pieces. Boundary
    Hausdorff distance can be large at such a crack even when the represented
    material differs only by nanometres. Genuine filled cavities still fail the
    reverse inclusion. All results are conditional on native Boolean tolerances.
    """
    from .collision import validate_convex

    reference = _solid(source)
    boundary, union, metadata = union_boundary(pieces)
    native_budget = max(reference.get_tolerance(), metadata["native_tolerance_m"])
    if not np.isfinite(tolerance_m) or tolerance_m <= native_budget:
        raise ValueError("Use a metric tolerance above the native Boolean tolerance")
    kernel_mesh = metric_kernel(tolerance_m - native_budget)
    kernel = _solid(kernel_mesh)
    reference_envelope = reference.minkowski_sum(kernel)
    import manifold3d as manifold

    if reference_envelope.status() != manifold.Error.NoError:
        raise ValueError("Native reference metric dilation failed")
    expanded = []
    for piece in pieces:
        validate_convex(piece)
        expanded.append(
            _solid(convex_boundary((piece.vertices[:, None] + kernel_mesh.vertices).reshape(-1, 3)))
        )
    candidate_envelope = manifold.Manifold.batch_boolean(expanded, manifold.OpType.Add)
    if candidate_envelope.status() != manifold.Error.NoError:
        raise ValueError("Native collider metric dilation failed")
    checks = {}
    for name, first, second in (
        ("source_inside_expanded_collision", reference, candidate_envelope),
        ("collision_inside_expanded_source", union, reference_envelope),
    ):
        residual = first - second
        if residual.status() != manifold.Error.NoError:
            raise ValueError("Native metric inclusion difference failed")
        checks[name] = {
            "passed": residual.is_empty(),
            "residual_volume_m3": float(residual.volume()),
            "residual_bounds_m": None if residual.is_empty() else list(residual.bounding_box()),
        }
    return boundary, {
        "passed": all(c["passed"] for c in checks.values()),
        "criterion": "bidirectional solid inclusion in inscribed Euclidean dilation kernels",
        "tolerance_m": float(tolerance_m),
        "kernel_vertices": len(kernel_mesh.vertices),
        "kernel_maximum_radius_m": float(np.linalg.norm(kernel_mesh.vertices, axis=1).max()),
        "native_boolean_tolerance_m": float(
            max(
                native_budget,
                reference_envelope.get_tolerance(),
                candidate_envelope.get_tolerance(),
            )
        ),
        "checks": checks,
        "union": metadata,
        "scope": (
            "sufficient solid Hausdorff certificate under native Boolean tolerances; "
            "failed inclusion is inconclusive about the exact maximum distance; "
            "does not certify contact normals, dynamics, analytic CAD or physical calibration"
        ),
    }


def clip_convex_halfspace(piece, normal, offset_m):
    """Intersect a convex polyhedron with n.x <= offset in its declared frame."""
    from .collision import validate_convex

    validate_convex(piece)
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(offset_m):
        raise ValueError("Use a finite plane normal and metric offset")
    norm = np.linalg.norm(normal)
    if not norm:
        raise ValueError("A clipping plane needs a nonzero normal")
    normal, offset_m = normal / norm, float(offset_m) / norm
    signed = piece.vertices @ normal - offset_m
    kept = piece.vertices[signed <= 0]
    edges = np.unique(
        np.sort(
            np.vstack((piece.faces[:, [0, 1]], piece.faces[:, [1, 2]], piece.faces[:, [2, 0]])),
            axis=1,
        ),
        axis=0,
    )
    crossing = edges[
        (signed[edges[:, 0]] < 0) & (signed[edges[:, 1]] > 0)
        | (signed[edges[:, 0]] > 0) & (signed[edges[:, 1]] < 0)
    ]
    if len(crossing):
        first, last = crossing.T
        t = signed[first] / (signed[first] - signed[last])
        kept = np.vstack(
            (
                kept,
                piece.vertices[first] + t[:, None] * (piece.vertices[last] - piece.vertices[first]),
            )
        )
    if len(kept) < 4 or np.linalg.matrix_rank(kept - kept[0]) < 3:
        return None
    return convex_boundary(kept)


def refine_convex_union(source, pieces, *, tolerance_m, progress=None):
    """Clip only out-of-budget convex cells and certify the transported result.

    The clipping envelope is contracted by a derived float32 rounding bound.
    The six axis planes bounding each proven excess region are considered.
    Among admissible convex clips, retain the greatest material volume. No
    decomposition threshold or metric error is changed; the complete result
    must pass bidirectional material coverage after float32 transport.
    """
    from .collision import quantized_convex, validate_convex

    reference = _solid(source)
    for piece in pieces:
        validate_convex(piece)
    largest = np.max([np.abs(p.vertices).max(0) for p in pieces], axis=0).astype(np.float32)
    half_ulp = (np.nextafter(largest, np.float32(np.inf)).astype(float) - largest) / 2
    rounding = float(np.linalg.norm(half_ulp))
    unit_kernel = metric_kernel(1.0)
    minimum_support = float((-ConvexHull(unit_kernel.vertices).equations[:, 3]).min())
    contraction = (rounding + reference.get_tolerance()) / minimum_support
    if not np.isfinite(tolerance_m) or tolerance_m <= contraction:
        raise ValueError("Metric budget is smaller than the derived transport contraction")
    envelope = reference.minkowski_sum(_solid(metric_kernel(tolerance_m - contraction)))
    import manifold3d as manifold

    if envelope.status() != manifold.Error.NoError:
        raise ValueError("Native refinement envelope failed")
    result, records = [], []
    for index, piece in enumerate(pieces):
        current = _solid(piece)
        excess = current - envelope
        if excess.status() != manifold.Error.NoError:
            raise ValueError("Native refinement difference failed")
        if excess.is_empty():
            result.append(piece)
            continue
        bounds = np.array(excess.bounding_box()).reshape(2, 3)
        candidates = []
        margin = rounding + envelope.get_tolerance()
        for axis in range(3):
            for sign in (1, -1):
                normal = np.eye(3)[axis] * sign
                offset = bounds[0, axis] - margin if sign == 1 else -bounds[1, axis] - margin
                clipped = clip_convex_halfspace(piece, normal, offset)
                if clipped is None:
                    continue
                transported, _ = quantized_convex(clipped)
                trial = _solid(transported)
                residue = trial - envelope
                if residue.status() != manifold.Error.NoError:
                    raise ValueError("Native clipping candidate difference failed")
                if residue.is_empty():
                    candidates.append((trial.volume(), transported, normal, offset))
        if not candidates:
            raise ValueError("No resolved convex clip satisfies the reference envelope")
        retained_volume, transported, normal, offset = max(candidates, key=lambda c: c[0])
        result.append(transported)
        records.append(
            {
                "input_piece": index,
                "out_of_envelope_volume_m3": float(excess.volume()),
                "out_of_envelope_bounds_m": list(excess.bounding_box()),
                "plane_normal": normal.tolist(),
                "plane_offset_m": float(offset),
                "selection": "greatest retained volume among six admissible excess-bound planes",
                "retained_volume_m3": float(retained_volume),
            }
        )
        if progress is not None:
            progress(records[-1])
    boundary, certificate = certify_solid_distance(source, result, tolerance_m=tolerance_m)
    return (
        result,
        boundary,
        {
            "passed": certificate["passed"],
            "algorithm": ("verified local convex halfspace clipping of out-of-envelope cells"),
            "input_pieces": len(pieces),
            "output_pieces": len(result),
            "float32_position_rounding_bound_m": rounding,
            "metric_envelope_contraction_m": contraction,
            "changed_pieces": records,
            "final_certificate": certificate,
        },
    )


def merge_verified_convex(source, pieces, *, tolerance_m, max_tests=20000, progress=None):
    """Merge touching cells only when their convex hull remains in the CAD envelope.

    Greedy ordering prefers the smallest relative addition of material. There
    is no requested hull count or claim of global optimality. A work budget
    bounds the number of native inclusion tests; remaining candidates and the
    final bidirectional solid certificate are recorded explicitly.
    """
    import manifold3d as manifold

    from .collision import quantized_convex, validate_convex

    if type(max_tests) is not int or max_tests <= 0:
        raise ValueError("Use a positive integer merge work budget")
    reference = _solid(source)
    if not np.isfinite(tolerance_m) or tolerance_m <= reference.get_tolerance():
        raise ValueError("Use a metric tolerance above the native Boolean tolerance")
    active = {}
    for number, piece in enumerate(pieces):
        validate_convex(piece)
        if not np.array_equal(piece.vertices, piece.vertices.astype(np.float32).astype(float)):
            raise ValueError("Verified merging requires float32-transported input vertices")
        active[number] = piece
    if not active:
        raise ValueError("Verified merging needs at least one convex piece")
    largest = np.max([np.abs(p.vertices).max(0) for p in pieces], axis=0).astype(np.float32)
    adjacency_padding = float(
        np.linalg.norm(np.nextafter(largest, np.float32(np.inf)).astype(float) - largest)
    )
    envelope = reference.minkowski_sum(
        _solid(metric_kernel(tolerance_m - reference.get_tolerance()))
    )
    if envelope.status() != manifold.Error.NoError:
        raise ValueError("Native merge envelope failed")
    bounds = {i: (p.vertices.min(0), p.vertices.max(0)) for i, p in active.items()}
    volumes = {i: ConvexHull(p.vertices).volume for i, p in active.items()}
    queue = []

    def enqueue(first, last):
        amin, amax = bounds[first]
        bmin, bmax = bounds[last]
        if np.any(amin > bmax + adjacency_padding) or np.any(bmin > amax + adjacency_padding):
            return
        points = np.vstack((active[first].vertices, active[last].vertices))
        ratio = ConvexHull(points).volume / (volumes[first] + volumes[last])
        heapq.heappush(queue, (ratio, first, last))

    for first, last in itertools.combinations(active, 2):
        enqueue(first, last)
    tested, merged, next_id = 0, 0, len(active)
    history = []
    while queue and tested < max_tests:
        _, first, last = heapq.heappop(queue)
        if first not in active or last not in active:
            continue
        hull = convex_boundary(np.vstack((active[first].vertices, active[last].vertices)))
        candidate, _ = quantized_convex(hull)
        excess = _solid(candidate) - envelope
        if excess.status() != manifold.Error.NoError:
            raise ValueError("Native merge inclusion test failed")
        tested += 1
        if excess.is_empty():
            history.append((next_id, first, active[first], last, active[last]))
            del active[first], active[last]
            active[next_id] = candidate
            bounds[next_id] = (candidate.vertices.min(0), candidate.vertices.max(0))
            volumes[next_id] = ConvexHull(candidate.vertices).volume
            for other in active:
                if other != next_id:
                    enqueue(other, next_id)
            next_id += 1
            merged += 1
        if progress is not None and (tested % 100 == 0 or not queue):
            progress({"tested_pairs": tested, "accepted_merges": merged, "pieces": len(active)})
    reverted = []
    while True:
        result = list(active.values())
        try:
            boundary, certificate = certify_solid_distance(source, result, tolerance_m=tolerance_m)
            if certificate["passed"]:
                break
            reason = "final bidirectional inclusion failed"
        except InvalidUnionBoundary as exc:
            reason = str(exc)
        if not history:
            raise ValueError(f"Input geometry could not retain a valid certificate: {reason}")
        merged_id, first, first_piece, last, last_piece = history.pop()
        del active[merged_id]
        active[first], active[last] = first_piece, last_piece
        reverted.append({"merge_id": merged_id, "reason": reason})
    live_queue = sum(first in active and last in active for _, first, last in queue)
    return (
        result,
        boundary,
        {
            "passed": certificate["passed"],
            "algorithm": "greedy adjacent convex merging with native CAD envelope inclusion",
            "input_pieces": len(pieces),
            "output_pieces": len(result),
            "adjacency_padding_m": adjacency_padding,
            "tested_pairs": tested,
            "accepted_merges": merged - len(reverted),
            "reverted_merges": reverted,
            "maximum_tests": max_tests,
            "remaining_candidate_pairs": live_queue,
            "queue_exhausted": live_queue == 0 and not reverted,
            "search_stop": "final-certificate rollback" if reverted else "queue or work budget",
            "optimality": "local greedy search; no global minimum hull-count claim",
            "final_certificate": certificate,
        },
    )
