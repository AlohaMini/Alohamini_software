import numpy as np
import pytest
from alohamini_mujoco._cad_conform import conform_face_mesh
from alohamini_mujoco.collision import signed_volume, topology
from alohamini_mujoco.meshes import Mesh


def face_fixture(*, two_chains=False, offset=0.0, hole=False):
    vertices = np.array(
        [
            [0, 0, 0],
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.01],
            [0.002, 0.002, 0],
            [0.006, 0.002, 0],
            [0.002, 0.006, 0],
            [0.004 if not two_chains else 0.005, 0.002 + offset, 0],
            [0.003, 0.002, 0],
        ]
    )
    faces = [[5, 1, 0], [4, 5, 0], [1, 6, 2], [1, 5, 6], [2, 4, 0], [2, 6, 4]]
    if two_chains:
        faces[1:2] = [[4, 8, 0], [8, 5, 0]]
    if not hole:
        faces.extend([[5, 7, 6], [7, 4, 6]])
    triangle_faces = [0] * len(faces)
    faces.extend([[0, 1, 3], [1, 2, 3], [2, 0, 3]])
    triangle_faces.extend([1, 2, 3])
    return Mesh(vertices, np.array(faces)), np.array(triangle_faces)


@pytest.mark.parametrize("two_chains", [False, True])
def test_interior_subdivision_closes_t_junctions_without_new_vertices_or_caps(two_chains):
    source, face_ids = face_fixture(two_chains=two_chains, offset=2e-12)
    assert topology(source)["boundary_edges"] in (3, 4)
    protected = np.zeros(len(source.vertices), dtype=bool)
    protected[:4] = True
    result, report = conform_face_mesh(source, face_ids, protected, [1e-10] * 4, 1e-14)
    assert not any(topology(result).values())
    assert signed_volume(result) == pytest.approx(0.01**3 / 6, abs=1e-18)
    np.testing.assert_array_equal(result.vertices, source.vertices)
    assert len(result.faces) > len(source.faces)
    assert report["new_vertices"] == 0 and not report["hole_filling"]
    assert not report["protected_CAD_boundaries_changed"]
    assert report["maximum_subdivision_surface_change_bound_m"] <= 1e-10
    assert all(r["status"] == "subdivided" for r in report["loops"])


@pytest.mark.parametrize("damage", ["hole", "protected", "mixed_faces", "beyond_local_tolerance"])
def test_real_holes_boundaries_and_unresolved_gaps_cannot_be_promoted(damage):
    source, face_ids = face_fixture(
        hole=damage == "hole",
        offset=1e-9 if damage == "beyond_local_tolerance" else 0,
    )
    protected = np.zeros(len(source.vertices), dtype=bool)
    if damage == "protected":
        protected[4] = True
    if damage == "mixed_faces":
        face_ids[1] = 1
    result, report = conform_face_mesh(
        source, face_ids, protected, [1e-10, 0.001, 0.001, 0.001], 1e-14
    )
    np.testing.assert_array_equal(result.vertices, source.vertices)
    np.testing.assert_array_equal(result.faces, source.faces)
    assert topology(result)["boundary_edges"] > 0
    assert all(r["status"].startswith("unresolved:") for r in report["loops"])
