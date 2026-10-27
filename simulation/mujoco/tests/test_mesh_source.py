import json
import struct
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
from alohamini_mujoco.cad_collision import _source, build_cad_collision
from alohamini_mujoco.mesh_source import extract_mesh_source
from alohamini_mujoco.meshes import binary_stl
from alohamini_mujoco.registration import register_part
from alohamini_mujoco.solid_geometry import convex_boundary

pytest.importorskip("manifold3d")


def fixture(tmp_path, *, steps=1, remove_face=False, tall=False):
    vertices = np.array(
        [[0, 0, 0], [0.03125, 0, 0], [0, 0.0625, 0], [0, 0, 4 if tall else 0.125]], dtype=np.float32
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    triangles = vertices[faces].copy()
    value = triangles[0, 2, 0]
    for _ in range(steps):
        value = np.nextafter(value, np.float32(np.inf))
    triangles[0, 2, 0] = value
    if remove_face:
        triangles = triangles[:-1]
    return description_for_triangles(tmp_path, triangles)


def description_for_triangles(tmp_path, triangles):
    data = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        data += struct.pack("<12fH", *([0, 0, 0] + triangle.ravel().tolist()), 0)
    (tmp_path / "part.stl").write_bytes(data)
    link = ET.fromstring(
        '<link><visual><origin xyz=".1 .2 .3" rpy="0 0 1.5707963267948966"/>'
        '<geometry><mesh filename="part.stl" scale="2 3 4"/></geometry></visual></link>'
    )
    return SimpleNamespace(
        robot_model=SimpleNamespace(directory=tmp_path, model_id="fixture"),
        links={"part": link},
        mesh_path=lambda _: tmp_path / "part.stl",
        source_bytes={},
    )


def test_source_closes_only_encoding_seams_and_keeps_scaled_visual_coordinates(tmp_path):
    description = fixture(tmp_path)
    output = tmp_path / "source"
    report = extract_mesh_source(description, output, link="part", component_index=0)
    assert report["source_format"] == "urdf-binary-stl"
    assert report["brep_sha256"] is None and report["linear_deflection_m"] is None
    assert report["seam_repair"]["merged_vertices"] == 1
    assert report["seam_repair"]["collapsed_faces"] == 0
    assert report["seam_repair"]["per_coordinate_rounding_intervals_checked"]
    mesh, metadata, _ = _source(output)
    assert len(mesh.faces) == 4
    np.testing.assert_allclose(mesh.vertices.max(0), [0.0625, 0.1875, 0.5], atol=4e-9, rtol=0)
    assert register_part(
        description,
        output,
        tmp_path / "registered",
        link="part",
        component_index=0,
        T_link_part=report["T_link_part_hint"],
        tolerance_m=report["surface_reference_error_m"] * 2,
    )["passed"]
    assert metadata == json.loads((output / "mesh.json").read_text())
    with pytest.raises(FileExistsError):
        extract_mesh_source(description, output, link="part", component_index=0)


@pytest.mark.parametrize("damage", ["hole", "outside_coordinate_rounding"])
def test_source_rejects_missing_faces_and_a_seam_beyond_its_own_coordinate_precision(
    tmp_path, damage
):
    description = fixture(
        tmp_path,
        steps=2 if damage == "outside_coordinate_rounding" else 0,
        tall=damage == "outside_coordinate_rounding",
        remove_face=damage == "hole",
    )
    output = tmp_path / "rejected"
    with pytest.raises(ValueError):
        extract_mesh_source(description, output, link="part", component_index=0)
    assert not output.exists()


def test_mesh_reference_reuses_the_qualified_collision_pipeline_without_claiming_brep_accuracy(
    tmp_path,
):
    pytest.importorskip("coacd")
    description = fixture(tmp_path, steps=0)
    source, registration = tmp_path / "source", tmp_path / "registration"
    metadata = extract_mesh_source(description, source, link="part", component_index=0)
    assert register_part(
        description,
        source,
        registration,
        link="part",
        component_index=0,
        T_link_part=metadata["T_link_part_hint"],
        tolerance_m=1e-7,
    )["passed"]
    output = tmp_path / "component"
    report = build_cad_collision(
        description,
        source,
        [registration],
        output,
        tolerance_m=0.0005,
        component_only=True,
    )
    assert report["passed"]
    assert not (output / "collision.json").exists()
    manifest = json.loads((output / "component.json").read_text())
    assert manifest["source_mesh_manifest"]["source_format"] == "urdf-binary-stl"
    assert manifest["source_mesh_manifest"]["brep_sha256"] is None


def test_shared_edge_corner_fans_preserve_all_oriented_triangles_and_solid_volume(tmp_path):
    first = np.array([[0, 0, 0], [0, 0, 0.03125], [0.03125, 0, 0], [0, 0.03125, 0]])
    second = first * [-1, -1, 1]
    triangles = np.vstack([convex_boundary(v).triangles for v in [first, second]])
    description = description_for_triangles(tmp_path, triangles)
    output = tmp_path / "fans"
    metadata = extract_mesh_source(description, output, link="part", component_index=0)
    assert metadata["index_fan_repair"]["branched_edges"] == 1
    assert metadata["index_fan_repair"]["source_oriented_triangles_identical"]
    assert metadata["seam_repair"]["maximum_vertex_displacement_m"] == 0
    mesh, _, _ = _source(output)
    original = binary_stl((tmp_path / "part.stl").read_bytes())
    np.testing.assert_array_equal(mesh.triangles, original.triangles * [2, 3, 4])


def test_coincident_duplicate_faces_cannot_be_silently_promoted_to_a_material_reference(tmp_path):
    tetrahedron = convex_boundary(
        np.array([[0, 0, 0], [0.03125, 0, 0], [0, 0.03125, 0], [0, 0, 0.03125]])
    )
    triangles = np.vstack([tetrahedron.triangles, tetrahedron.triangles[:1]])
    description = description_for_triangles(tmp_path, triangles)
    output = tmp_path / "duplicates"
    with pytest.raises(ValueError, match="Duplicated triangle"):
        extract_mesh_source(description, output, link="part", component_index=0)
    assert not output.exists()


def test_explicit_sewing_is_a_separate_geometry_revision_with_a_surface_error_budget(tmp_path):
    description = fixture(tmp_path, steps=2)
    strict = tmp_path / "strict"
    with pytest.raises(ValueError):
        extract_mesh_source(description, strict, link="part", component_index=0)
    assert not strict.exists()
    output = tmp_path / "sewn"
    metadata = extract_mesh_source(
        description,
        output,
        link="part",
        component_index=0,
        seam_budget_m=1e-6,
    )
    assert metadata["source_format"] == "urdf-binary-stl-sewn"
    assert metadata["brep_sha256"] is None
    repair = metadata["seam_repair"]
    assert repair["geometry_revision"] and not repair["per_coordinate_rounding_intervals_checked"]
    assert repair["source_triangles"] == repair["output_triangles"] == 4
    assert repair["continuous_surface_distance_bound_m"] <= 1e-6
    assert metadata["surface_reference_error_m"] >= repair["continuous_surface_distance_bound_m"]
    mesh, _, _ = _source(output)
    assert len(mesh.faces) == 4
    assert register_part(
        description,
        output,
        tmp_path / "registered-sewn",
        link="part",
        component_index=0,
        T_link_part=metadata["T_link_part_hint"],
        tolerance_m=metadata["surface_reference_error_m"] * 2,
    )["passed"]


def test_explicit_sewing_still_cannot_publish_a_real_hole(tmp_path):
    description = fixture(tmp_path, steps=0, remove_face=True)
    output = tmp_path / "hole"
    with pytest.raises(ValueError, match="no unique complete reverse partner"):
        extract_mesh_source(
            description,
            output,
            link="part",
            component_index=0,
            seam_budget_m=1e-6,
        )
    assert not output.exists()
