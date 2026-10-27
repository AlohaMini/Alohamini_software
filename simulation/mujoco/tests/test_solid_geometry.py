import numpy as np
import pytest
from alohamini_mujoco.meshes import Mesh
from alohamini_mujoco.solid_geometry import (
    certify_solid_distance,
    clip_convex_halfspace,
    compare_solid_union,
    merge_verified_convex,
    metric_kernel,
    refine_convex_union,
    union_boundary,
)

manifold = pytest.importorskip("manifold3d")


def box(size, center=(0, 0, 0)):
    solid = manifold.Manifold.cube(size, center=True).translate(center).to_mesh64()
    return Mesh(np.array(solid.vert_properties)[:, :3], np.array(solid.tri_verts, dtype=np.int64))


def test_overlap_and_interior_partition_faces_are_removed_before_comparison():
    source = box([0.02] * 3)
    parts = [source, box([0.002] * 3)]
    boundary, report = compare_solid_union(source, parts, tolerance_m=0.0001)
    assert report["passed"]
    assert len(boundary.faces) == len(source.faces)
    assert report["union"]["input_triangles"] == 24
    assert report["union"]["boundary_triangles"] == 12
    assert report["difference_volume_diagnostics"]["source_minus_union_m3"] == 0
    assert report["difference_volume_diagnostics"]["union_minus_source_m3"] == 0


def test_filling_a_through_hole_fails_even_when_all_outer_faces_match():
    walls = [box([0.002, 0.04, 0.1], [x, 0, 0]) for x in (-0.009, 0.009)] + [
        box([0.016, 0.002, 0.1], [0, y, 0]) for y in (-0.019, 0.019)
    ]
    source, _, _ = union_boundary(walls)
    _, identical = compare_solid_union(source, walls, tolerance_m=0.0005)
    assert identical["passed"]
    _, filled = compare_solid_union(source, [box([0.02, 0.04, 0.1])], tolerance_m=0.0005)
    assert not filled["passed"]
    assert filled["directions"]["source_to_union"]["lower_bound_m"] > 0.0005
    assert filled["difference_volume_diagnostics"]["union_minus_source_m3"] == pytest.approx(
        0.016 * 0.036 * 0.1
    )


@pytest.mark.parametrize("damage", ["open", "inward", "nonfinite"])
def test_invalid_shells_are_rejected_without_repair_or_solid_inference(damage):
    source = box([0.02] * 3)
    if damage == "open":
        source = Mesh(source.vertices, source.faces[:-1])
    elif damage == "inward":
        source = Mesh(source.vertices, source.faces[:, ::-1])
    else:
        source.vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="closed, consistently oriented"):
        union_boundary([source])


def test_metric_inclusion_tolerates_an_internal_rounding_crack_but_rejects_a_filled_hole():
    source = box([0.02] * 3)
    pieces = [box([0.00999999, 0.02, 0.02], [x, 0, 0]) for x in (-0.005000005, 0.005000005)]
    _, report = certify_solid_distance(source, pieces, tolerance_m=0.0001)
    assert report["passed"]
    assert report["kernel_maximum_radius_m"] <= 0.0001
    walls = [box([0.002, 0.04, 0.1], [x, 0, 0]) for x in (-0.009, 0.009)] + [
        box([0.016, 0.002, 0.1], [0, y, 0]) for y in (-0.019, 0.019)
    ]
    source, _, _ = union_boundary(walls)
    _, filled = certify_solid_distance(source, [box([0.02, 0.04, 0.1])], tolerance_m=0.0005)
    assert not filled["passed"]
    assert not filled["checks"]["collision_inside_expanded_source"]["passed"]


def test_convex_clipping_uses_declared_metric_planes_and_preserves_the_retained_half():
    source = box([0.02] * 3)
    clipped = clip_convex_halfspace(source, [1, 0, 0], 0.0)
    np.testing.assert_allclose(clipped.vertices.min(0), [-0.01, -0.01, -0.01])
    np.testing.assert_allclose(clipped.vertices.max(0), [0, 0.01, 0.01])
    assert clip_convex_halfspace(source, [1, 0, 0], -0.02) is None


def test_local_refinement_removes_only_the_identified_excess_and_rechecks_full_coverage():
    source = box([0.02] * 3)
    extended = box([0.0208, 0.02, 0.02], [0.0004, 0, 0])
    result, _, report = refine_convex_union(source, [extended], tolerance_m=0.0005)
    assert report["passed"] and len(result) == 1
    assert report["changed_pieces"][0]["plane_normal"] == [1, 0, 0]
    assert report["final_certificate"]["checks"]["source_inside_expanded_collision"]["passed"]
    assert result[0].vertices[:, 0].max() <= 0.0105


def test_fully_enclosed_cavities_are_not_reinterpreted_as_separate_positive_solids():
    hollow = manifold.Manifold.cube([0.02] * 3, center=True) - manifold.Manifold.cube(
        [0.01] * 3, center=True
    )
    exported = hollow.to_mesh64()
    source = Mesh(
        np.array(exported.vert_properties)[:, :3], np.array(exported.tri_verts, dtype=np.int64)
    )
    _, report = certify_solid_distance(source, [box([0.02] * 3)], tolerance_m=0.0005)
    assert not report["passed"]


@pytest.mark.parametrize("radius", [0, -1, np.nan])
def test_metric_kernel_rejects_invalid_radii(radius):
    with pytest.raises(ValueError):
        metric_kernel(radius)


def test_verified_merging_reduces_flat_partitions_without_filling_a_notch():
    from alohamini_mujoco.collision import quantized_convex

    halves = [box([0.01, 0.02, 0.02], [x, 0, 0]) for x in (-0.005, 0.005)]
    parts = [quantized_convex(p)[0] for p in halves]
    merged, _, report = merge_verified_convex(box([0.02] * 3), parts, tolerance_m=0.0001)
    assert report["passed"] and len(merged) == 1
    assert report["queue_exhausted"]
    walls = [box([0.01, 0.03, 0.02]), box([0.01, 0.01, 0.02], [0.01, -0.01, 0])]
    source, _, _ = union_boundary(walls)
    pieces = [quantized_convex(p)[0] for p in walls]
    merged, _, report = merge_verified_convex(source, pieces, tolerance_m=0.0001)
    assert report["passed"] and len(merged) == 2
    assert report["accepted_merges"] == 0


def test_verified_merging_reports_an_exhausted_work_budget():
    from alohamini_mujoco.collision import quantized_convex

    parts = [quantized_convex(box([0.01, 0.02, 0.02], [x, 0, 0]))[0] for x in (-0.01, 0, 0.01)]
    merged, _, report = merge_verified_convex(
        box([0.03, 0.02, 0.02]), parts, tolerance_m=0.0001, max_tests=1
    )
    assert report["passed"] and len(merged) == 2
    assert not report["queue_exhausted"]
    assert report["remaining_candidate_pairs"] == 1


def test_merging_reverts_a_candidate_that_cannot_export_a_valid_final_union(monkeypatch):
    from alohamini_mujoco import solid_geometry
    from alohamini_mujoco.collision import quantized_convex

    original = solid_geometry.certify_solid_distance

    def reject_merged_boundary(source, pieces, **kwargs):
        if len(pieces) == 1:
            raise solid_geometry.InvalidUnionBoundary("unresolved degenerate Boolean boundary")
        return original(source, pieces, **kwargs)

    monkeypatch.setattr(solid_geometry, "certify_solid_distance", reject_merged_boundary)
    parts = [quantized_convex(box([0.01, 0.02, 0.02], [x, 0, 0]))[0] for x in (-0.005, 0.005)]
    merged, _, report = merge_verified_convex(box([0.02] * 3), parts, tolerance_m=0.0001)
    assert report["passed"] and len(merged) == 2
    assert report["accepted_merges"] == 0 and len(report["reverted_merges"]) == 1
    assert not report["queue_exhausted"]
