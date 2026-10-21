import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from alohamini_mujoco.audit import validate_model
from alohamini_mujoco.collision import (
    load_bundle,
    merge_bundles,
    ordered_convex_parts,
    visual_sources,
)
from alohamini_mujoco.description import Description
from alohamini_mujoco.interop import export_collision_urdf
from alohamini_mujoco.meshes import Mesh
from alohamini_mujoco.model import ModelProfile, build_model
from alohamini_mujoco.organization import organize_collisions

from alohamini.model import get_robot_model


def tetrahedron():
    return Mesh(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]]),
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]),
    )


def test_recovery_preserves_all_faces_despite_a_shared_internal_face():
    first = tetrahedron()
    second_vertices = first.vertices * [-1, 1, 1]
    points, inverse = np.unique(
        np.vstack((first.vertices, second_vertices)), axis=0, return_inverse=True
    )
    faces = inverse[np.vstack((first.faces, first.faces[:, ::-1] + 4))]
    source = Mesh(points, faces)
    parts, record = ordered_convex_parts(source)
    assert len(parts) == 2
    assert record["face_ranges"] == [[0, 4], [4, 8]]
    assert not record["geometry_changed"]
    np.testing.assert_array_equal(np.concatenate([p.triangles for p in parts]), source.triangles)


def test_recovery_refuses_incomplete_inward_and_concave_source_shells():
    mesh = tetrahedron()
    with pytest.raises(ValueError, match="Incomplete"):
        ordered_convex_parts(Mesh(mesh.vertices, mesh.faces[:-1]))
    with pytest.raises(ValueError, match="Inward"):
        ordered_convex_parts(Mesh(mesh.vertices, mesh.faces[:, ::-1]))
    vertices = np.vstack((mesh.vertices, [0.001, 0.001, 0.001]))
    faces = np.vstack((mesh.faces[[0, 1, 3]], [[1, 2, 4], [2, 3, 4], [3, 1, 4]]))
    with pytest.raises(ValueError):
        ordered_convex_parts(Mesh(vertices, faces))


@pytest.fixture(scope="module")
def organized(tmp_path_factory):
    path = tmp_path_factory.mktemp("organization") / "bundle"
    description = Description(get_robot_model("alohamini2pro"))
    report = organize_collisions(description, path)
    return description, path, report


def test_every_link_and_retained_primitive_has_explicit_provenance(organized):
    description, path, report = organized
    assert report["links"].keys() == description.links.keys()
    assert report["physical_links"] == 26 and report["coordinate_links"] == 6
    assert report["source_collision_shapes"] == 50
    assert report["organized_collision_shapes"] == 120
    meshes, provenance = load_bundle(description, path)
    assert len(meshes) == 14 and sum(map(len, meshes.values())) == 84
    for side in ("left", "right"):
        assert [
            len(meshes[f"{side}_{suffix}"])
            for suffix in (
                "Base",
                "Rotation_Pitch",
                "Upper_Arm",
                "Lower_Arm",
                "Wrist_Pitch_Roll",
                "wrist_yaw",
                "camera",
            )
        ] == [8, 6, 8, 8, 4, 6, 2]
    assert report["links"]["base_cad_link"]["requires_authored_geometry"]
    assert report["links"]["vertical_link"]["representation"] == "retained_source"
    assert provenance["definition"]["assembly_inventory"] == report


@pytest.mark.parametrize(
    "mutation", ["omit_link", "retained_primitive", "omit_piece", "source_mesh"]
)
def test_loading_refuses_incomplete_inventory_or_changed_retained_geometry(
    organized, tmp_path, mutation
):
    description, source, _ = organized
    path = tmp_path / "changed"
    shutil.copytree(source, path)
    manifest = path / "collision.json"
    data = json.loads(manifest.read_text())
    if mutation == "omit_link":
        del data["assembly_inventory"]["links"]["root"]
    elif mutation == "retained_primitive":
        data["assembly_inventory"]["links"]["wheel1"]["source_collisions"][0]["parameters"][
            "radius"
        ] = "1"
    elif mutation == "omit_piece":
        data["links"]["left_Upper_Arm"]["pieces"].pop()
    else:
        data["links"]["left_Upper_Arm"]["source_collisions"][0]["mesh"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_bundle(description, path)


def test_native_organized_robot_preserves_frames_inertias_limits_and_contact_policy(organized):
    _, path, _ = organized
    robot = build_model(
        profile=ModelProfile(collision_bundle=path, initial_positions={"vertical_move": 0.2})
    )
    report = validate_model(robot, samples=4, seed=13)
    assert report["passed"], report
    assert report["robot_links"] == 32 and report["active_joints"] == 15
    assert report["collision_geoms"] == 120
    assert report["imported_mass_kg"] == pytest.approx(16.0, abs=1e-12)
    assert report["max_inertia_error_kg_m2"] < 1e-14
    assert robot.model.nexclude == 38
    assert report["initial_contacts"] == []


def test_composition_keeps_whole_robot_source_checks(organized, tmp_path):
    description, source, _ = organized
    extra = tmp_path / "extra"
    extra.mkdir()
    data = tetrahedron().msh_bytes()
    (extra / "piece.msh").write_bytes(data)
    (extra / "collision.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "robot_model": description.robot_model.model_id,
                "units": "m",
                "links": {
                    "chest_camera": {
                        "frame": "URDF link-local",
                        "source_visuals": visual_sources(description, "chest_camera"),
                        "pieces": [
                            {
                                "path": "piece.msh",
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "triangles": 4,
                            }
                        ],
                    }
                },
            }
        )
    )
    combined = merge_bundles(description, [source, extra], tmp_path / "combined")
    pieces, _ = load_bundle(description, combined)
    assert len(pieces) == 15
    manifest = combined / "collision.json"
    metadata = json.loads(manifest.read_text())
    inventory = metadata["source_manifests"][0]["generator"]["assembly_inventory"]
    inventory["links"]["wheel1"]["source_collisions"][0]["parameters"]["radius"] = "1"
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source changed"):
        load_bundle(description, combined)


def test_portable_export_keeps_primitives_and_every_recovered_shell(organized, tmp_path):
    description, path, _ = organized
    export = export_collision_urdf(description, path, tmp_path / "portable")
    tree = ET.parse(export / "robot.urdf").getroot()
    assert sum(len(link.findall("collision")) for link in tree.findall("link")) == 120
    assert len(tree.find("link[@name='left_Upper_Arm']").findall("collision")) == 8
    assert (
        tree.find("link[@name='wheel1']/collision/geometry/cylinder").get("radius") == "0.0634601"
    )
    for name, node in description.links.items():
        source = node.find("inertial")
        if source is not None:
            assert (
                ET.tostring(tree.find(f"link[@name='{name}']/inertial")).split()
                == ET.tostring(source).split()
            )
