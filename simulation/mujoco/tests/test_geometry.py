import json
import struct
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.geometry import SurfaceIndex, directed_surface_bounds
from alohamini_mujoco.meshes import surface_distances
from alohamini_mujoco.registration import register_part


def test_surface_index_matches_all_triangles_at_mixed_scales_and_degenerate_faces():
    rng = np.random.default_rng(42)
    triangles = rng.normal(size=(120, 3, 3)) * 10.0 ** rng.integers(-5, 2, size=(120, 1, 1))
    triangles[0] = 0
    triangles[1, 2] = triangles[1, 1]
    points = np.vstack((rng.normal(size=(45, 3)), [[0, 0, 0], [10000, 10000, -10000]]))
    index = SurfaceIndex(triangles)
    actual, nearest = index.closest(points, batch_size=7)
    np.testing.assert_allclose(actual, surface_distances(points, triangles), atol=1e-10)
    for point, distance, face in zip(points, actual, nearest, strict=True):
        assert surface_distances(point[None], triangles[face : face + 1])[0] == pytest.approx(
            distance, abs=1e-10
        )
    np.testing.assert_allclose(index.closest(points, batch_size=1)[0], actual)
    assert SurfaceIndex(np.zeros((1, 3, 3))).closest([[1, 0, 0]])[0][0] == 1


def test_continuous_surface_check_detects_a_hole_despite_matching_vertices():
    a, b, c = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    target = np.array([[a, ab, ca], [ab, b, bc], [ca, bc, c]])
    source = np.array([[a, b, c]])
    np.testing.assert_array_equal(SurfaceIndex(target).closest(source[0])[0], [0, 0, 0])
    report = directed_surface_bounds(source, target, tolerance_m=0.05)
    assert report["status"] == "failed"
    assert report["lower_bound_m"] > 0.05


def test_surface_check_bounds_offset_and_never_passes_on_budget_exhaustion():
    square = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]])
    offset = square + [0, 0, 0.001]
    report = directed_surface_bounds(square, offset, tolerance_m=0.00101, max_samples=2)
    assert report["status"] == "passed"
    assert report["lower_bound_m"] <= 0.001 <= report["upper_bound_m"]
    # The opposite diagonal needs subdivision to establish coverage by target
    # triangles, even though these two meshes describe the very same square.
    other = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]])
    limited = directed_surface_bounds(square, other, tolerance_m=0.01, max_samples=2)
    assert limited["status"] == "inconclusive"
    assert limited["lower_bound_m"] <= 0.01 < limited["upper_bound_m"]
    assert directed_surface_bounds(square, other, tolerance_m=0.01)["status"] == "passed"


def test_subdivision_finds_a_gap_away_from_initial_vertices_and_centroids():
    def rectangle(x0, y0, x1, y1):
        return [
            [[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]],
            [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]],
        ]

    source = np.array(rectangle(0, 0, 1, 1))
    target = np.array(
        rectangle(0, 0, 1, 0.1)
        + rectangle(0, 0.2, 1, 1)
        + rectangle(0, 0.1, 0.1, 0.2)
        + rectangle(0.2, 0.1, 1, 0.2)
    )
    samples = np.vstack((source.reshape(-1, 3), source.mean(axis=1)))
    np.testing.assert_array_equal(SurfaceIndex(target).closest(samples)[0], np.zeros(len(samples)))
    report = directed_surface_bounds(source, target, tolerance_m=0.01)
    assert report["status"] == "failed"
    assert report["lower_bound_m"] > 0.01
    assert report["subdivision_levels"] > 1


@pytest.mark.parametrize("joined", [False, True])
def test_registration_binds_explicit_component_transform_and_rejects_tampering(tmp_path, joined):
    vertices = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0], [0, 0, 0.04]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    triangles = vertices[faces]
    if joined:
        triangles = np.vstack((triangles, (vertices * [1, -1, -1])[faces]))
    stl = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        stl += struct.pack("<12fH", *([0, 0, 0] + triangle.reshape(-1).tolist()), 0)
    (tmp_path / "part.stl").write_bytes(stl)
    link = ET.fromstring(
        '<link><visual><geometry><mesh filename="part.stl"/></geometry></visual></link>'
    )
    description = SimpleNamespace(
        robot_model=SimpleNamespace(directory=tmp_path, model_id="fixture"),
        links={"part": link},
        mesh_path=lambda _: tmp_path / "part.stl",
        source_bytes={},
    )
    source = tmp_path / "mesh"
    source.mkdir()
    np.savez(source / "solid.npz", vertices=vertices - [0.1, 0.2, 0.3], faces=faces)
    manifest = {
        "schema_version": 1,
        "source_cad_sha256": "original-step",
        "units": "m",
        "coordinates": "CAD part definition frame, float64",
        "definition": "part-definition",
        "name": "fixture",
        "brep_sha256": "original-brep",
        "linear_deflection_m": 5e-5,
        "angular_deflection_rad": 0.1,
        "solids": [{"file": "solid.npz", "sha256": file_sha256(source / "solid.npz")}],
    }
    (source / "mesh.json").write_text(json.dumps(manifest))
    transform = np.eye(4)
    transform[:3, 3] = [0.1, 0.2, 0.3]
    report = register_part(
        description,
        source,
        tmp_path / "registered",
        link="part",
        component_index=0,
        surface_index=0 if joined else None,
        T_link_part=transform,
        tolerance_m=1e-7,
    )
    assert report["passed"]
    if joined:
        assert report["target"]["surface_partition"]["surface_count"] == 2
        assert not report["target"]["surface_partition"]["geometry_changed"]
        assert report["target"]["surface_partition"]["component_triangles"] == 8
    assert report["target"]["visual"]["sha256"] == file_sha256(tmp_path / "part.stl")
    for invalid_surface in (True, -1, 2):
        with pytest.raises(ValueError, match="manifold surface"):
            register_part(
                description,
                source,
                tmp_path / "invalid",
                link="part",
                component_index=0,
                surface_index=invalid_surface,
                T_link_part=transform,
                tolerance_m=1e-7,
            )
    for name, expected in report["files"].items():
        assert file_sha256(tmp_path / "registered" / name) == expected
    transform[0, 3] += 0.01
    bad = register_part(
        description,
        source,
        tmp_path / "wrong",
        link="part",
        component_index=0,
        surface_index=0 if joined else None,
        T_link_part=transform,
        tolerance_m=1e-7,
    )
    assert not bad["passed"]
    (source / "solid.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        register_part(
            description,
            source,
            tmp_path / "tampered",
            link="part",
            component_index=0,
            surface_index=0 if joined else None,
            T_link_part=transform,
            tolerance_m=1e-7,
        )
