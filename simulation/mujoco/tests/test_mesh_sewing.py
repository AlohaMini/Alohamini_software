import numpy as np
import pytest
from alohamini_mujoco.collision import topology
from alohamini_mujoco.mesh_sewing import sew_boundary_edges
from alohamini_mujoco.meshes import Mesh


def ring_with_cap_seam(gap=2e-8):
    """An annular prism: the top cap has independent, displaced boundary nodes."""
    angles = np.arange(16) * (2 * np.pi / 16)
    circle = np.column_stack([np.cos(angles), np.zeros(16), np.sin(angles)])
    vertices = np.vstack(
        [circle * radius + [0, y, 0] for y in [0, 0.0021] for radius in [0.0096, 0.0045]]
    )
    faces = []
    for i in range(16):
        j = (i + 1) % 16
        for a, b, c, d in [
            (i, j, 32 + j, 32 + i),
            (16 + j, 16 + i, 48 + i, 48 + j),
            (j, i, 16 + i, 16 + j),
            (32 + i, 32 + j, 48 + j, 48 + i),
        ]:
            faces.extend([[a, b, c], [a, c, d]])
    faces = np.array(faces)
    # All cap faces remain; move only their separately indexed copies.
    cap = np.array([i % 8 in (6, 7) for i in range(len(faces))])
    vertices = np.vstack([vertices, vertices[32:] + [0, gap, 0]])
    faces[cap] += 32
    return Mesh(vertices, faces[:, [0, 2, 1]])


def test_sewing_retains_through_bore_all_faces_and_certifies_continuous_error():
    pytest.importorskip("manifold3d")
    from alohamini_mujoco.contact_probe import _winding
    from alohamini_mujoco.solid_geometry import _solid

    original = ring_with_cap_seam()
    assert topology(original)["boundary_edges"] == 64
    sewn, report = sew_boundary_edges(original, 1e-6)
    assert not any(topology(sewn).values())
    assert report["paired_edges"] == 32 and not report["hole_filling"]
    assert report["new_triangles"] == 0 and report["new_vertices"] == 0
    assert len(sewn.faces) == len(original.faces) == 128
    assert report["maximum_vertex_displacement_m"] == pytest.approx(1e-8)
    bound = report["continuous_surface_distance_bound_m"]
    weights = np.random.default_rng(0).dirichlet([1, 1, 1], size=200)
    source_points = np.einsum("pk,fkc->pfc", weights, original.triangles)
    target_points = np.einsum("pk,fkc->pfc", weights, sewn.triangles)
    assert np.linalg.norm(target_points - source_points, axis=2).max() <= bound
    solid = _solid(sewn)
    expected = 16 * np.sin(2 * np.pi / 16) / 2 * (0.0096**2 - 0.0045**2) * 0.00210001
    assert solid.volume() == pytest.approx(expected, rel=1e-10)
    assert abs(_winding(sewn, [0, 0.001, 0])) < 1e-10  # through bore stays empty
    assert abs(_winding(sewn, [0.007, 0.001, 0])) == pytest.approx(1)


@pytest.mark.parametrize("damage", ["missing_face", "insufficient_budget", "real_gap"])
def test_missing_faces_and_seams_outside_budget_are_rejected(damage):
    mesh = ring_with_cap_seam(1e-4 if damage == "real_gap" else 2e-8)
    if damage == "missing_face":
        mesh = Mesh(mesh.vertices, mesh.faces[:-1])
    with pytest.raises(ValueError, match="no unique complete reverse partner"):
        sew_boundary_edges(mesh, 1e-10 if damage == "insufficient_budget" else 1e-6)


def test_ambiguous_reverse_partners_and_zero_thickness_surfaces_are_rejected():
    triangle = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])
    vertices = np.vstack([triangle, triangle + [0, 0, 1e-8], triangle + [0, 0, -1e-8]])
    mesh = Mesh(vertices, np.array([[0, 1, 2], [5, 4, 3], [8, 7, 6]]))
    with pytest.raises(ValueError, match="no unique complete reverse partner"):
        sew_boundary_edges(mesh, 1e-6)
    sheet = Mesh(vertices[:6], np.array([[0, 1, 2], [5, 4, 3]]))
    with pytest.raises(ValueError, match="duplicated triangle"):
        sew_boundary_edges(sheet, 1e-6)


def test_combined_endpoint_groups_cannot_exceed_the_budget_even_when_each_pair_fits():
    budget = 1e-6
    vertices = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    triangles = vertices[faces].copy()
    for face, angle in zip([0, 1, 3], [0, 2 * np.pi / 3, 4 * np.pi / 3], strict=True):
        corner = np.flatnonzero(faces[face] == 0)[0]
        triangles[face, corner] += 1.1 * budget * np.array([np.cos(angle), np.sin(angle), 0])
    mesh = Mesh(triangles.reshape(-1, 3), np.arange(12).reshape(-1, 3))
    with pytest.raises(ValueError, match="combined seam displacement exceeds"):
        sew_boundary_edges(mesh, budget)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), True])
def test_sewing_budget_must_be_explicit_finite_and_positive(budget):
    with pytest.raises(ValueError, match="explicit positive finite"):
        sew_boundary_edges(ring_with_cap_seam(), budget)
