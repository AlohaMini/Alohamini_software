import itertools

import numpy as np
import pytest

pytest.importorskip("manifold3d")

from alohamini_mujoco.collision import quantized_convex
from alohamini_mujoco.functional_geometry import (
    axis_aligned_box,
    certify_cylinder_box_cover,
    certify_material_coverage,
    merge_support_envelopes,
)
from alohamini_mujoco.meshes import Mesh
from alohamini_mujoco.solid_geometry import _solid, convex_boundary, union_boundary


def box(lower, upper):
    corners = np.array(list(itertools.product(*zip(lower, upper, strict=True))))
    return quantized_convex(convex_boundary(corners))[0]


def test_support_cells_merge_to_a_native_box_while_the_pad_remains_independent():
    pad = box([0, 0, 0.0195], [0.02, 0.02, 0.02])
    parts = [pad, box([0, 0, 0], [0.01, 0.02, 0.0194]), box([0.01, 0, 0], [0.02, 0.02, 0.0194])]
    result, report = merge_support_envelopes(
        parts,
        protected_indices=[0],
        forbidden_regions=[],
        obstacles=[],
        clearance_m=0.0003,
        adjacency_m=0.001,
    )
    assert len(result) == 2
    assert result[0] is pad
    size, pose = axis_aligned_box(result[1])
    np.testing.assert_allclose(size, [0.02, 0.02, 0.0194], atol=1e-9)
    np.testing.assert_allclose(pose[:3, 3], [0.01, 0.01, 0.0097], atol=1e-9)
    assert report["accepted_merges"] == 1


def test_functional_slot_is_kept_while_mounting_pockets_can_be_approximated():
    parts = [
        box([0, 0, 0], [0.01, 0.03, 0.01]),
        box([0.02, 0, 0], [0.03, 0.03, 0.01]),
        box([0, 0, 0], [0.03, 0.01, 0.01]),
    ]
    free = _solid(box([0.012, 0.02, -0.001], [0.018, 0.031, 0.011]))
    result, _ = merge_support_envelopes(
        parts,
        protected_indices=[],
        forbidden_regions=[free],
        obstacles=[],
        clearance_m=0.0003,
        adjacency_m=0.001,
    )
    for piece in result:
        assert (_solid(piece) ^ free).is_empty()
    source, _, _ = union_boundary(parts)
    assert certify_material_coverage(source, result, tolerance_m=0.0005)[
        "source_inside_expanded_collision"
    ]


def test_merge_cannot_encapsulate_a_moving_part_without_boundary_intersection():
    parts = [
        box([-0.03, -0.01, -0.01], [-0.02, 0.01, 0.01]),
        box([0.02, -0.01, -0.01], [0.03, 0.01, 0.01]),
    ]
    obstacle = _solid(box([-0.001] * 3, [0.001] * 3))
    result, _ = merge_support_envelopes(
        parts,
        protected_indices=[],
        forbidden_regions=[],
        obstacles=[obstacle],
        clearance_m=0.001,
        adjacency_m=0.05,
    )
    assert len(result) == 2


def test_material_coverage_rejects_a_missing_support():
    source = box([0, 0, 0], [0.03, 0.03, 0.03])
    with pytest.raises(ValueError, match="omit CAD material"):
        certify_material_coverage(source, [box([0, 0, 0], [0.01, 0.03, 0.03])], tolerance_m=0.0005)


def test_primitive_coverage_checks_triangle_interiors_across_shape_boundaries():
    source = Mesh(np.array([[-0.5, 0, 0], [0.5, 0, 0], [2, 0, 0.5]]), np.array([[0, 1, 2]]))
    bounds = [np.array([[0.3, -0.2, -0.5], [2.1, 0.2, 0.6]])]
    assert certify_cylinder_box_cover(source, 1, [-1, 1], bounds, tolerance_m=1e-8)["passed"]
    hull = box(*bounds[0])
    assert certify_cylinder_box_cover(source, 1, [-1, 1], [], hulls=[hull], tolerance_m=1e-8)[
        "passed"
    ]
    # Every original vertex is covered, but there is a gap inside the triangle.
    source = Mesh(np.array([[0, 0, 0], [2, 2, 0], [2, 2, 0.5]]), np.array([[0, 1, 2]]))
    bounds = [np.array([[1.8, 1.8, -0.5], [2.2, 2.2, 0.6]])]
    with pytest.raises(ValueError, match="source surfaces uncovered"):
        certify_cylinder_box_cover(source, 1, [-1, 1], bounds, tolerance_m=1e-8)
    with pytest.raises(ValueError, match="source surfaces uncovered"):
        certify_cylinder_box_cover(
            source, 1, [-1, 1], [], hulls=[box(*bounds[0])], tolerance_m=1e-8
        )
