import numpy as np
import pytest

pytest.importorskip("manifold3d")

from alohamini_mujoco.collision import quantized_convex
from alohamini_mujoco.convex_cover import (
    bounded_convex_mesh,
    certify_rotating_clearance,
    select_convex_cover,
)
from alohamini_mujoco.solid_geometry import _solid, convex_boundary, union_boundary


def box(lower, upper):
    import itertools

    vertices = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    return quantized_convex(convex_boundary(vertices))[0]


def test_overlapping_cover_preserves_a_reentrant_slot_and_reduces_cells():
    pieces = [
        box([0, 0, 0], [0.01, 0.03, 0.01]),
        box([0.02, 0, 0], [0.03, 0.03, 0.01]),
        box([0, 0, 0], [0.03, 0.01, 0.01]),
    ]
    reference, _, _ = union_boundary(pieces)
    redundant = [box([0, 0, 0], [0.005, 0.005, 0.005])] * 4
    result, certificate = select_convex_cover(reference, pieces + redundant, tolerance_m=0.0005)
    assert len(result) == 3
    assert certificate["source_inside_expanded_collision"]
    assert certificate["each_collider_inside_expanded_source"]
    _, solid, _ = union_boundary(result)
    assert solid.min_gap(_solid(box([0.012, 0.02, 0], [0.018, 0.025, 0.01])), 0.001) > 0


def test_surface_witnesses_cannot_hide_missing_interior_material():
    reference = box([0, 0, 0], [0.03, 0.03, 0.03])
    walls = []
    for axis in range(3):
        for lower_side in [True, False]:
            lower, upper = np.zeros(3), np.full(3, 0.03)
            if lower_side:
                upper[axis] = 0.002
            else:
                lower[axis] = 0.028
            walls.append(box(lower, upper))
    with pytest.raises(ValueError, match="do not cover the CAD witnesses"):
        select_convex_cover(reference, walls, tolerance_m=0.0005)


def test_filling_a_functional_slot_fails_the_material_certificate():
    pieces = [
        box([0, 0, 0], [0.01, 0.03, 0.01]),
        box([0.02, 0, 0], [0.03, 0.03, 0.01]),
        box([0, 0, 0], [0.03, 0.01, 0.01]),
    ]
    reference, _, _ = union_boundary(pieces)
    with pytest.raises(ValueError, match="exceeds the CAD material envelope"):
        select_convex_cover(reference, [box([0, 0, 0], [0.03, 0.03, 0.01])], tolerance_m=0.0005)


def test_contact_plane_has_one_continuous_face_and_structural_cells_stay_behind_it():
    reference = box([0, 0, 0], [0.02, 0.02, 0.02])
    contact = {
        "name": "pad",
        "vertices": [[0, 0, 0.02], [0.02, 0, 0.02], [0.02, 0.02, 0.02], [0, 0.02, 0.02]],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "normal": [0, 0, 1],
        "tolerance_m": 0.00005,
    }
    result, certificate = select_convex_cover(
        reference, [reference], tolerance_m=0.0005, contacts=[contact]
    )
    assert len(result) == 2
    layer, structure = result
    assert np.max(layer.vertices[:, 2]) == pytest.approx(0.02, abs=1e-8)
    assert np.max(structure.vertices[:, 2]) <= 0.02 - 0.000499
    assert certificate["contacts"][0]["approach_region_clear"]
    top = layer.vertices[np.isclose(layer.vertices[:, 2], 0.02, atol=1e-8)]
    assert len(top) == 4


def test_nonplanar_or_wrongly_oriented_contact_footprints_are_rejected():
    reference = box([0, 0, 0], [0.02, 0.02, 0.02])
    contact = {
        "name": "pad",
        "vertices": [[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0.02]],
        "faces": [[0, 1, 2]],
        "normal": [0, 0, 1],
        "tolerance_m": 0.00005,
    }
    with pytest.raises(ValueError, match="not planar"):
        select_convex_cover(reference, [reference], tolerance_m=0.0005, contacts=[contact])


def test_backend_preparation_cannot_recess_the_contact_face_within_the_coarse_cad_budget():
    reference = box([0, 0, 0], [0.02, 0.02, 0.02])
    contact = {
        "name": "pad",
        "vertices": [[0, 0, 0.02], [0.02, 0, 0.02], [0.02, 0.02, 0.02], [0, 0.02, 0.02]],
        "faces": [[0, 1, 2], [0, 2, 3]],
        "normal": [0, 0, 1],
        "tolerance_m": 0.00005,
    }

    def prepare(piece, **kwargs):
        return [quantized_convex(convex_boundary(piece.vertices - [0, 0, 0.00008]))[0]]

    with pytest.raises(ValueError, match="contact surface tolerance"):
        select_convex_cover(
            reference, [reference], tolerance_m=0.0005, contacts=[contact], prepare_mesh=prepare
        )


def test_convex_vertex_reduction_has_a_solid_distance_bound():
    angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    ring = np.column_stack((0.01 * np.cos(angles), 0.01 * np.sin(angles)))
    vertices = np.vstack(
        (np.column_stack((ring, np.zeros(256))), np.column_stack((ring, np.full(256, 0.01))))
    )
    original = quantized_convex(convex_boundary(vertices))[0]
    reduced, error = bounded_convex_mesh(original, max_vertices=64)
    assert len(reduced.vertices) <= 64
    assert error < 0.0001
    assert error > 0
    assert _solid(reduced).volume() <= _solid(original).volume()


def test_clearance_proof_covers_the_full_joint_interval():
    fixed = [box([-0.005] * 3, [0.005] * 3)]
    moving = [box([0.015, -0.005, -0.005], [0.025, 0.005, 0.005])]
    report = certify_rotating_clearance(
        fixed, moving, np.eye(4), [0, 0, 1], [-np.pi, np.pi], accuracy_m=0.001
    )
    assert report["passed"]
    assert report["certified_gap_lower_bound_m"] > 0.005
    assert report["interval_count"] > 2


def test_clearance_proof_rejects_collision_between_clear_endpoints():
    fixed = [box([-0.005, 0.015, -0.005], [0.005, 0.025, 0.005])]
    moving = [box([0.015, -0.005, -0.005], [0.025, 0.005, 0.005])]
    with pytest.raises(ValueError, match="clearance is not positive"):
        certify_rotating_clearance(
            fixed, moving, np.eye(4), [0, 0, 1], [0, np.pi], accuracy_m=0.001
        )
