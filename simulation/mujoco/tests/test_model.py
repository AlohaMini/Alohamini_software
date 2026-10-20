import hashlib
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import replace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.audit import contacts, validate_model
from alohamini_mujoco.cli import export_robot
from alohamini_mujoco.frames import points_in_frame, rigid_transform, rotation_rpy, transform
from alohamini_mujoco.meshes import binary_stl, ray_x_intersections
from alohamini_mujoco.model import ModelProfile, build_model


def test_urdf_rpy_and_frame_direction():
    T_world_root = transform([1, 2, 3], [np.pi / 2, 0, np.pi / 2])
    # Roll first maps local Y to Z; yaw then leaves that Z unchanged.
    np.testing.assert_allclose(points_in_frame(T_world_root, [[0, 1, 0]]), [[1, 2, 4]])
    np.testing.assert_allclose(rotation_rpy([0, 0, np.pi / 2]) @ [1, 0, 0], [0, 1, 0], atol=1e-15)
    for matrix in [np.diag([1, -1, 1, 1]), np.diag([1.000001, 1, 1, 1])]:
        with pytest.raises(ValueError, match="right-handed"):
            rigid_transform(matrix)


def test_native_frames_inertias_meshes_and_mass_matrix_match_urdf(robot):
    report = validate_model(robot, samples=24, seed=9)
    assert report["passed"], report
    assert report["robot_links"] == 32
    assert report["active_joints"] == 15
    assert report["visual_geoms"] == 26
    assert report["collision_geoms"] == 50
    assert report["imported_mass_kg"] == pytest.approx(16.0, abs=1e-12)
    assert report["max_inertia_error_kg_m2"] < 1e-14
    assert report["minimum_mass_matrix_eigenvalue"] > 0
    assert report["initial_contacts"] == []
    assert any(m["source_triangles"] > 200_000 for m in report["meshes"])
    assert all(m["triangles_preserved"] for m in report["meshes"])
    for path, digest in robot.manifest["source_sha256"].items():
        assert (
            hashlib.sha256(
                (robot.description.robot_model.directory / path).read_bytes()
            ).hexdigest()
            == digest
        )


def test_gravity_generalized_forces_match_urdf_potential_energy(robot):
    positions = dict(robot.initial_positions)
    data = robot.data()
    gravity = np.asarray(robot.profile.gravity)

    def potential(values):
        poses = robot.description.forward(values, robot.profile.T_world_root)
        return -sum(
            mass * np.dot(gravity, (poses[name] @ com)[:3, 3])
            for name, (mass, com, _) in robot.description.inertials.items()
        )

    for name, joint in robot.description.joints.items():
        if joint.kind == "fixed" or name in robot.profile.locked_positions:
            continue
        value = positions.get(name, 0)
        epsilon = 1e-6
        derivative = (
            potential({**positions, name: value + epsilon})
            - potential({**positions, name: value - epsilon})
        ) / (2 * epsilon)
        dof = robot.model.jnt_dofadr[robot.model.joint(name).id]
        assert data.qfrc_bias[dof] == pytest.approx(derivative, abs=1e-7)


def test_collision_parts_and_explicit_srdf_filters_are_retained(robot):
    for side in ("left", "right"):
        assert (
            sum(
                g["link"] == f"{side}_Fixed_Jaw" and g["kind"] == "collision"
                for g in robot.geometry.values()
            )
            == 8
        )
        assert (
            sum(
                g["link"] == f"{side}_Moving_Jaw" and g["kind"] == "collision"
                for g in robot.geometry.values()
            )
            == 6
        )
    assert robot.model.nexclude == len(robot.description.exclusions) == 38
    assert int(robot.model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    for index in range(robot.model.ngeom):
        name = robot.model.geom(index).name
        collision = robot.geometry[name]["kind"] == "collision"
        assert robot.model.geom_contype[index] == int(collision)
        assert robot.model.geom_conaffinity[index] == int(collision)


def test_external_obstacle_still_contacts_the_finger(robot):
    root = ET.fromstring(robot.xml)
    position = robot.data().geom("right_Fixed_Jaw__collision_7").xpos
    ET.SubElement(
        root.find("worldbody"),
        "geom",
        name="external_probe",
        type="sphere",
        size="0.003",
        pos=" ".join(map(str, position)),
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), robot.assets)
    augmented = replace(robot, model=model)
    touching = contacts(augmented, augmented.data())
    assert any("external_probe" in (c["geom1"], c["geom2"]) for c in touching)


def test_export_is_relocatable_and_does_not_overwrite(robot, tmp_path):
    destination = export_robot(robot, {"checked": True}, tmp_path / "before")
    destination.rename(tmp_path / "after")
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "after/model.xml"))
    np.testing.assert_allclose(model.body_mass, robot.model.body_mass, atol=1e-12)
    np.testing.assert_allclose(model.jnt_range, robot.model.jnt_range, atol=1e-12)
    manifest = json.loads((tmp_path / "after/manifest.json").read_text())
    for name, digest in manifest["derived_asset_sha256"].items():
        assert hashlib.sha256((tmp_path / "after/assets" / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_robot(robot, {}, tmp_path / "after")


def test_profiles_reject_fabricated_inertia_and_bad_locks():
    with pytest.raises(ValueError, match="no measured/declared inertia"):
        build_model(profile=ModelProfile(locked_positions={}))
    with pytest.raises(ValueError, match="Unknown profile joint"):
        build_model(profile=ModelProfile(locked_positions={"guessed_joint": 0}))
    with pytest.raises(ValueError, match="outside URDF limits"):
        build_model(profile=ModelProfile(initial_positions={"vertical_move": 2}))


def stl_bytes(triangles):
    triangles = np.asarray(triangles, dtype="<f4")
    dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
    values = np.zeros(len(triangles), dtype=dtype)
    values["vertices"] = triangles
    return bytes(80) + struct.pack("<I", len(triangles)) + values.tobytes()


def test_large_stl_conversion_preserves_triangles_winding_and_coordinates():
    tetrahedron = np.array(
        [[[0, 0, 0], [0, 1, 0], [1, 0, 0]], [[0, 0, 0], [1, 0, 0], [0, 0, 1]]], dtype=np.float32
    )
    triangles = np.tile(tetrahedron, (100001, 1, 1))
    mesh = binary_stl(stl_bytes(triangles))
    data = mesh.msh_bytes()
    nvertices, nnormals, ntexture, nfaces = struct.unpack_from("<4i", data)
    assert nfaces == 200002 and nnormals == ntexture == 0
    vertices = np.frombuffer(data, dtype="<f4", count=3 * nvertices, offset=16).reshape(-1, 3)
    faces = np.frombuffer(data, dtype="<i4", offset=16 + nvertices * 12).reshape(-1, 3)
    np.testing.assert_array_equal(vertices[faces], triangles)
    with pytest.raises(ValueError, match="complete binary STL"):
        binary_stl(data[:100])


def test_ray_intersection_handles_shared_triangle_edges():
    triangles = np.array(
        [[[2, -1, -1], [2, 1, -1], [2, 1, 1]], [[2, -1, -1], [2, 1, 1], [2, -1, 1]]]
    )
    np.testing.assert_allclose(ray_x_intersections(triangles, 0, 0), [2])
    assert len(ray_x_intersections(triangles, 3, 0)) == 0
