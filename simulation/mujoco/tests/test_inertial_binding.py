import copy
import json
from dataclasses import replace

import numpy as np
import pytest
from alohamini_mujoco.audit import validate_model
from alohamini_mujoco.control import DynamicFollower
from alohamini_mujoco.frames import transform
from alohamini_mujoco.inertia import MassProperties, combine
from alohamini_mujoco.inertial_binding import load_inertial_bundle
from alohamini_mujoco.model import build_model
from alohamini_mujoco.organization import description_hashes


def specification(robot):
    main = MassProperties("vertical_link", 1.2, [0.01, -0.04, 0.4], np.eye(3) * 0.01)
    camera = MassProperties("vertical_link", 0.08, [-0.02, -0.07, 0.45], np.eye(3) * 1e-4)
    properties = combine([main, camera])
    T_main = transform([0.5, 1.2, -0.2], [0.3, 0.4, -0.7])
    T_camera = transform([-0.1, 0.7, 0.1], [-0.2, 0.8, 0.6])
    return {
        "schema_version": 1,
        "robot_model": robot.description.robot_model.model_id,
        "units": "kg, m, kg*m^2",
        "source_description_sha256": description_hashes(robot.description),
        "required_locked_positions": {},
        "link_groups": [
            {
                "link": "vertical_link",
                "replaces": ["vertical_link", "chest_camera"],
                "sources": [
                    {
                        "properties": main.in_frame(np.linalg.inv(T_main), "report/main").to_dict(),
                        "T_link_source": T_main.tolist(),
                        "component_ids": ["carriage"],
                    },
                    {
                        "properties": camera.in_frame(
                            np.linalg.inv(T_camera), "report/camera"
                        ).to_dict(),
                        "T_link_source": T_camera.tolist(),
                        "component_ids": ["chest_camera"],
                    },
                ],
            }
        ],
        "component_inventory": ["carriage", "chest_camera", "unbound_arm"],
        "unresolved": ["Other inertias retain their original declarations"],
    }, properties


def save(tmp_path, specification):
    path = tmp_path / "inertia.json"
    path.write_text(json.dumps(specification))
    return path


def test_binding_replaces_a_fixed_group_without_double_counting_or_moving_geometry(robot, tmp_path):
    definition, properties = specification(robot)
    path = save(tmp_path, definition)
    T_world_root = transform([0.3, -0.7, 0.2], [0.1, 0.2, 0.3])
    bound = build_model(
        profile=replace(robot.profile, inertial_bundle=path, T_world_root=T_world_root)
    )
    metadata = bound.manifest["inertial_binding"]
    assert not metadata["complete"]
    assert metadata["unassigned_cad_components"] == ["unbound_arm"]
    assert set(metadata["removed_urdf_inertias"]) == {"vertical_link", "chest_camera"}
    assert "chest_camera" not in bound.description.inertials
    assert bound.model.body("chest_camera").mass == 0
    assert bound.model.body("vertical_link").mass == pytest.approx(properties.mass_kg)
    old_mass = sum(p[0] for p in robot.description.inertials.values())
    removed_mass = sum(robot.description.inertials[n][0] for n in ["vertical_link", "chest_camera"])
    assert bound.model.body_mass.sum() == pytest.approx(
        old_mass - removed_mass + properties.mass_kg
    )
    assert bound.manifest["source_sha256"] == robot.manifest["source_sha256"]
    assert bound.model.ngeom == robot.model.ngeom
    assert bound.model.njnt == robot.model.njnt
    np.testing.assert_array_equal(bound.model.jnt_range, robot.model.jnt_range)
    np.testing.assert_array_equal(bound.model.geom_pos, robot.model.geom_pos)
    np.testing.assert_array_equal(bound.model.geom_quat, robot.model.geom_quat)
    np.testing.assert_array_equal(bound.model.actuator_forcerange, robot.model.actuator_forcerange)
    report = validate_model(bound, samples=3)
    assert report["passed"], report
    assert report["mass_matrix_max_absolute_error"] < 1e-12
    data = bound.data()
    pose = bound.description.forward(bound.initial_positions, T_world_root)["vertical_link"]
    np.testing.assert_allclose(
        data.body("vertical_link").xipos, pose[:3, :3] @ properties.com_m + pose[:3, 3]
    )
    body = bound.model.body("vertical_link").id
    rotation = data.ximat[body].reshape(3, 3)
    np.testing.assert_allclose(
        rotation @ np.diag(bound.model.body_inertia[body]) @ rotation.T,
        pose[:3, :3] @ properties.central_inertia_kg_m2 @ pose[:3, :3].T,
        rtol=0,
        atol=1e-13,
    )
    with pytest.raises(ValueError, match="Partial inertial binding"):
        DynamicFollower(bound)


def test_binding_rejects_combining_material_across_an_active_joint(robot, tmp_path):
    definition, _ = specification(robot)
    definition["link_groups"][0]["replaces"].append("left_Upper_Arm")
    with pytest.raises(ValueError, match="across an active joint"):
        load_inertial_bundle(
            robot.description, save(tmp_path, definition), robot.profile.locked_positions
        )


def test_a_collapsed_wheel_requires_its_explicit_locked_pose(robot, tmp_path):
    definition, properties = specification(robot)
    group = definition["link_groups"][0]
    group["link"] = "base_cad_link"
    group["replaces"] = ["base_cad_link", "wheel1"]
    path = save(tmp_path, definition)
    with pytest.raises(ValueError, match="across an active joint"):
        load_inertial_bundle(robot.description, path, robot.profile.locked_positions)
    definition["required_locked_positions"] = {"wheel1_joint": 0.0}
    path = save(tmp_path, definition)
    effective, _ = load_inertial_bundle(robot.description, path, robot.profile.locked_positions)
    assert "wheel1" not in effective
    with pytest.raises(ValueError, match="declared locked joint positions"):
        load_inertial_bundle(robot.description, path, {})


def test_binding_rejects_duplicate_materials_and_a_changed_description(robot, tmp_path):
    original, properties = specification(robot)
    definition = copy.deepcopy(original)
    definition["link_groups"].append(
        {
            "link": "left_Upper_Arm",
            "replaces": ["left_Upper_Arm"],
            "sources": [
                {
                    "properties": properties.to_dict(),
                    "T_link_source": np.eye(4).tolist(),
                    "component_ids": ["carriage"],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="exactly one inertial owner"):
        load_inertial_bundle(
            robot.description, save(tmp_path, definition), robot.profile.locked_positions
        )
    original["source_description_sha256"] = {}
    with pytest.raises(ValueError, match="current robot description"):
        load_inertial_bundle(
            robot.description, save(tmp_path, original), robot.profile.locked_positions
        )


def complete_specification(robot):
    groups = []
    for name, (mass, pose, inertia) in robot.description.inertials.items():
        properties = MassProperties(
            name, mass, pose[:3, 3], pose[:3, :3] @ inertia @ pose[:3, :3].T
        )
        groups.append(
            {
                "link": name,
                "replaces": [name],
                "sources": [
                    {
                        "properties": properties.to_dict(),
                        "T_link_source": np.eye(4).tolist(),
                        "component_ids": [name],
                    }
                ],
            }
        )
    definition, _ = specification(robot)
    definition.update(
        required_locked_positions=dict(robot.profile.locked_positions),
        link_groups=groups,
        component_inventory=[*robot.description.inertials, "stationary_electronics"],
        fixed_unassigned_components={"base_cad_link": ["stationary_electronics"]},
        unresolved=[],
    )
    return definition


def test_stationary_mass_gap_preserves_active_dynamics_and_its_scope(robot, tmp_path):
    definition = complete_specification(robot)
    bound = build_model(profile=replace(robot.profile, inertial_bundle=save(tmp_path, definition)))
    metadata = bound.manifest["inertial_binding"]
    assert not metadata["complete"]
    assert metadata["dynamics_ready"]
    assert metadata["unassigned_cad_components"] == ["stationary_electronics"]
    assert metadata["unassigned_active_cad_components"] == []
    np.testing.assert_allclose(bound.data().qfrc_bias, robot.data().qfrc_bias, rtol=0, atol=1e-12)
    DynamicFollower(bound)
    definition["required_locked_positions"].pop("root_x_axis_joint")
    with pytest.raises(ValueError, match="stationary owner"):
        load_inertial_bundle(
            robot.description, save(tmp_path, definition), robot.profile.locked_positions
        )
    definition = complete_specification(robot)
    definition["fixed_unassigned_components"] = {"left_Upper_Arm": ["stationary_electronics"]}
    with pytest.raises(ValueError, match="stationary owner"):
        load_inertial_bundle(
            robot.description, save(tmp_path, definition), robot.profile.locked_positions
        )


def test_armature_adds_only_declared_joint_inertia_and_is_audited(robot, tmp_path):
    definition = complete_specification(robot)
    definition["joint_armature_kg_m2"] = {"left_shoulder_pan": 3.9e-4}
    bound = build_model(profile=replace(robot.profile, inertial_bundle=save(tmp_path, definition)))
    baseline, modified = (
        np.empty((robot.model.nv, robot.model.nv)),
        np.empty((bound.model.nv, bound.model.nv)),
    )
    import mujoco

    mujoco.mj_fullM(robot.model, robot.data(), baseline)
    mujoco.mj_fullM(bound.model, bound.data(), modified)
    address = bound.model.jnt_dofadr[bound.model.joint("left_shoulder_pan").id]
    expected = np.zeros_like(baseline)
    expected[address, address] = 3.9e-4
    np.testing.assert_allclose(modified - baseline, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bound.data().qfrc_bias, robot.data().qfrc_bias, rtol=0, atol=1e-12)
    assert validate_model(bound, samples=1)["passed"]
    for name, value in [
        ("vertical_move", 1e-3),
        ("missing", 1e-3),
        ("wheel1_joint", 1e-3),
        ("left_shoulder_pan", -1),
        ("left_shoulder_pan", float("nan")),
    ]:
        definition["joint_armature_kg_m2"] = {name: value}
        with pytest.raises(ValueError, match="Joint armature"):
            load_inertial_bundle(
                robot.description, save(tmp_path, definition), robot.profile.locked_positions
            )
