import hashlib
import json
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.audit import convexification_report, gripper_sections, validate_model
from alohamini_mujoco.collision import (
    components,
    generate_bundle,
    load_bundle,
    merge_bundles,
    quantized_convex,
    topology,
    validate_convex,
    visual_sources,
    weld_precision_seams,
)
from alohamini_mujoco.meshes import Mesh, binary_msh
from alohamini_mujoco.model import ModelProfile, build_model


def tetrahedron():
    return Mesh(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=np.float32),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
    )


def fixture_bundle(description, directory):
    directory.mkdir()
    mesh = tetrahedron()
    data = mesh.msh_bytes()
    (directory / "piece.msh").write_bytes(data)
    record = {
        "frame": "URDF link-local",
        "source_visuals": visual_sources(description, "right_Moving_Jaw"),
        "pieces": [
            {
                "path": "piece.msh",
                "sha256": hashlib.sha256(data).hexdigest(),
                "triangles": len(mesh.faces),
            }
        ],
    }
    manifest = {
        "schema_version": 1,
        "units": "m",
        "robot_model": description.robot_model.model_id,
        "links": {"right_Moving_Jaw": record},
    }
    (directory / "collision.json").write_text(json.dumps(manifest))
    return manifest, record


def test_merge_keeps_every_hull_and_rejects_overlapping_link_replacements(robot, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    fixture_bundle(robot.description, first)
    metadata, record = fixture_bundle(robot.description, second)
    record["source_visuals"] = visual_sources(robot.description, "left_Moving_Jaw")
    metadata["links"] = {"left_Moving_Jaw": record}
    (second / "collision.json").write_text(json.dumps(metadata))
    result = merge_bundles(robot.description, [first, second], tmp_path / "combined")
    loaded, provenance = load_bundle(robot.description, result)
    assert set(loaded) == {"left_Moving_Jaw", "right_Moving_Jaw"}
    assert sum(map(len, loaded.values())) == 2
    assert len(provenance["definition"]["source_manifests"]) == 2
    with pytest.raises(ValueError, match="overlapping"):
        merge_bundles(robot.description, [first, first], tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_welding_precision_seam_preserves_closed_surface_and_frame():
    mesh = tetrahedron()
    points = np.vstack((mesh.vertices, mesh.vertices[0] + [1e-9, 0, 0]))
    faces = mesh.faces.copy()
    faces[0, 0] = 4
    cracked = Mesh(points, faces)
    assert topology(cracked)["boundary_edges"] > 0
    welded, report = weld_precision_seams(cracked, 5e-8)
    assert not any(topology(welded).values())
    assert report["maximum_vertex_displacement_m"] <= 5e-8
    assert report["collapsed_faces"] == 0 and report["merged_vertices"] == 1
    assert len(components(welded)) == 1
    np.testing.assert_allclose(welded.vertices.min(axis=0), mesh.vertices.min(axis=0), atol=1e-9)
    # A real missing face remains missing; welding does not cap a hole.
    open_mesh, _ = weld_precision_seams(Mesh(points, faces[:-1]), 5e-8)
    assert topology(open_mesh)["boundary_edges"] == 3
    with pytest.raises(ValueError):
        validate_convex(open_mesh)


def test_convex_piece_rejects_concavity_and_bad_winding():
    mesh = tetrahedron()
    validate_convex(mesh)
    bad_faces = mesh.faces.copy()
    bad_faces[0] = bad_faces[0, ::-1]
    with pytest.raises(ValueError, match="closed oriented"):
        validate_convex(Mesh(mesh.vertices, bad_faces))
    # Dent the sloping face inward, preserving a closed oriented manifold.
    vertices = np.vstack((mesh.vertices, [0.001, 0.001, 0.001]))
    faces = np.vstack((mesh.faces[:-1], [[1, 2, 4], [2, 3, 4], [3, 1, 4]]))
    assert not any(topology(Mesh(vertices, faces)).values())
    with pytest.raises(ValueError, match="concave"):
        validate_convex(Mesh(vertices, faces))


def test_msh_rejects_corruption_and_preserves_indexed_geometry():
    mesh = tetrahedron()
    loaded = binary_msh(mesh.msh_bytes())
    np.testing.assert_array_equal(loaded.triangles, mesh.triangles)
    with pytest.raises(ValueError, match="MSH sizes"):
        binary_msh(mesh.msh_bytes()[:-1])
    invalid = replace(mesh, faces=mesh.faces + 4)
    with pytest.raises(ValueError, match="indices"):
        binary_msh(invalid.msh_bytes())


def test_float32_transport_retriangulates_thin_convex_pieces_without_dropping_them():
    # A real thin CAD decomposition cap. Rounding makes the original planar
    # triangle fan nonconvex even though every vertex moves less than 2 nm.
    vertices = np.array(
        [
            [-0.013425659226014852, -0.01074566962001334, 0.0352502885335286],
            [-0.013425659226014852, -0.004929391914802551, 0.03524934675798333],
            [-0.012752842527027674, -0.01074566962001334, 0.035249375393050585],
            [-0.012731715415323325, -0.01074566962001334, 0.03524934675798333],
            [-0.013425659226014852, 0.010001220888813474, 0.0352502885335286],
            [-0.013425659226014852, -0.0023685236472027592, 0.03524934675798333],
            [-0.012731715415323325, 0.010001220888813474, 0.03524934675798333],
            [-0.013340945561215466, 0.010001220888813474, 0.03525017240242251],
        ]
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 6],
            [0, 6, 4],
            [0, 4, 5],
            [0, 5, 1],
            [1, 3, 2],
            [1, 5, 6],
            [1, 6, 3],
            [4, 6, 7],
            [4, 7, 5],
            [5, 7, 6],
        ]
    )
    validate_convex(Mesh(vertices, faces))
    with pytest.raises(ValueError, match="concave"):
        validate_convex(Mesh(vertices.astype(np.float32), faces))
    mesh, transport = quantized_convex(Mesh(vertices, faces))
    assert transport["maximum_vertex_quantization_m"] < 2e-9
    validate_convex(binary_msh(mesh.msh_bytes()))
    # All support extrema of the transported points remain identical.
    directions = np.random.default_rng(0).normal(size=(128, 3))
    np.testing.assert_allclose(
        (mesh.vertices @ directions.T).max(axis=0),
        (vertices.astype(np.float32) @ directions.T).max(axis=0),
        atol=1e-15,
    )


def test_bundle_rejects_wrong_frames_source_hashes_corruption_and_path_escape(robot, tmp_path):
    directory = tmp_path / "candidate"
    manifest, record = fixture_bundle(robot.description, directory)
    loaded, metadata = load_bundle(robot.description, directory)
    assert len(loaded["right_Moving_Jaw"]) == 1
    assert metadata["manifest_sha256"]
    for malformed in ([], {**manifest, "schema_version": True}):
        (directory / "collision.json").write_text(json.dumps(malformed))
        with pytest.raises(ValueError, match="coordinate contract"):
            load_bundle(robot.description, directory)
    for field, value, message in (
        ("frame", "world", "CAD frames"),
        ("source_visuals", [], "CAD frames"),
        ("pieces", [], "Empty collision"),
        ("pieces", [{}], "Invalid collision piece"),
    ):
        changed = json.loads(json.dumps(manifest))
        changed["links"]["right_Moving_Jaw"][field] = value
        (directory / "collision.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            load_bundle(robot.description, directory)
    record["pieces"][0]["path"] = "../outside.msh"
    (directory / "collision.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        load_bundle(robot.description, directory)
    record["pieces"][0]["path"] = "piece.msh"
    (directory / "collision.json").write_text(json.dumps(manifest))
    (directory / "piece.msh").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(robot.description, directory)


def test_candidate_changes_only_explicit_link_colliders(robot, tmp_path):
    directory = tmp_path / "candidate"
    fixture_bundle(robot.description, directory)
    candidate = build_model(profile=ModelProfile(collision_bundle=directory))
    assert candidate.manifest["collision_profile"] == "cad-bundle"
    assert candidate.model.ngeom == robot.model.ngeom - 6 + 1
    assert candidate.model.nexclude == robot.model.nexclude
    assert candidate.model.nu == robot.model.nu
    np.testing.assert_array_equal(candidate.model.body_mass, robot.model.body_mass)
    np.testing.assert_array_equal(candidate.model.body_inertia, robot.model.body_inertia)
    np.testing.assert_array_equal(candidate.model.jnt_range, robot.model.jnt_range)
    assert validate_model(candidate, samples=2)["passed"]
    assert sum(g["kind"] == "visual" for g in candidate.geometry.values()) == 26
    assert (
        candidate.model.geom_contype[
            mujoco.mj_name2id(
                candidate.model, mujoco.mjtObj.mjOBJ_GEOM, "right_Moving_Jaw__collision_0"
            )
        ]
        == 1
    )


def test_existing_concave_collision_meshes_are_reported(robot):
    report = convexification_report(robot)
    assert report["meshes"]
    ratios = [m["convex_to_source_volume_ratio"] for m in report["meshes"]]
    assert any(r is None or r > 1.2 for r in ratios)
    for record in report["meshes"]:
        if any(record["source_topology"].values()):
            assert record["convex_to_source_volume_ratio"] is None


def test_missing_collision_surface_is_reported_instead_of_discarded_from_gap_error(robot):
    report = gripper_sections(robot, positions=[0.0], depths_m=[0.08, 10.0])
    assert report["probe_counts"] == {
        "total": 2,
        "paired_gaps": 1,
        "visual_gap_without_collision_gap": 0,
        "collision_gap_without_visual_gap": 0,
        "neither_gap": 1,
    }
    geometry = {name: dict(entry) for name, entry in robot.geometry.items()}
    for entry in geometry.values():
        if entry["link"] == "right_Moving_Jaw" and entry["kind"] == "collision":
            entry["T_link_geometry"] = entry["T_link_geometry"].copy()
            entry["T_link_geometry"][1, 3] += 100
    damaged = replace(robot, geometry=geometry)
    report = gripper_sections(damaged, positions=[0.0], depths_m=[0.08])
    assert report["maximum_absolute_gap_difference_m"] is None
    assert report["probe_counts"]["visual_gap_without_collision_gap"] == 1
    assert report["probe_counts"]["paired_gaps"] == 0
    assert report["contact_surface_probe_counts"]["moving"]["visual_only"] == 1


def test_common_contact_surface_displacement_cannot_hide_behind_unchanged_aperture(robot):
    original = gripper_sections(robot, positions=[0.0], depths_m=[0.08])["sections"][0]
    displacement = np.array([0.001, 0, 0])
    rotation = robot.description.joints["right_gripper"].T_parent_joint[:3, :3]
    geometry = {name: dict(entry) for name, entry in robot.geometry.items()}
    for entry in geometry.values():
        if entry["kind"] != "collision":
            continue
        if entry["link"] in ("right_Fixed_Jaw", "right_Moving_Jaw"):
            entry["T_link_geometry"] = entry["T_link_geometry"].copy()
            entry["T_link_geometry"][:3, 3] += (
                displacement if entry["link"] == "right_Fixed_Jaw" else rotation.T @ displacement
            )
    shifted = gripper_sections(replace(robot, geometry=geometry), positions=[0.0], depths_m=[0.08])
    row = shifted["sections"][0]
    assert row["collision_gap_m"] == pytest.approx(original["collision_gap_m"], abs=1e-12)
    for finger in ("fixed", "moving"):
        assert row[f"{finger}_contact_error_m"] - original[
            f"{finger}_contact_error_m"
        ] == pytest.approx(0.001, abs=1e-12)


def test_generator_rejects_dirty_cad_before_producing_geometry(robot, tmp_path):
    pytest.importorskip("coacd")
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="not a closed oriented manifold"):
        generate_bundle(robot.description, ["right_Fixed_Jaw"], output)
    assert not output.exists()


def test_initial_position_cannot_silently_override_a_lock():
    with pytest.raises(ValueError, match="conflicts with locked"):
        ModelProfile(initial_positions={"root_x_axis_joint": 0.1})
