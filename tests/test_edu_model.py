"""EDU uses Pro mechanics while navigation has a single base TF parent."""

import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

from alohamini.kinematics.arm import AlohaMiniArmKinematics
from alohamini.model import get_robot_model


def test_edu_mechanics_match_pro_without_borrowing_device_calibration():
    pro, edu = (get_robot_model(name) for name in ("alohamini2pro", "alohamini2edu"))
    original = ET.fromstring(
        pro.description_path("collision").read_text().replace("alohamini2pro", "alohamini2edu")
    )
    updated = ET.parse(edu.description_path("collision")).getroot()
    updated.remove(updated.find("./link[@name='lidar_housing']"))
    updated.remove(updated.find("./joint[@name='lidar_housing_joint']"))
    assert ET.canonicalize(ET.tostring(original), strip_text=True) == ET.canonicalize(
        ET.tostring(updated), strip_text=True
    )
    assert pro.actuators == edu.actuators
    assert (pro.wheel_radius_m, pro.base_radius_m, pro.lift_lead_m_per_rev) == (
        edu.wheel_radius_m,
        edu.base_radius_m,
        edu.lift_lead_m_per_rev,
    )
    assert not json.loads(edu.asset_path("model.json").read_text())["validation"][
        "device_calibration_included"
    ]
    preset = json.loads(edu.kinematic_preset_path(edu.default_kinematic_preset).read_text())
    assert preset["robot_model"] == edu.model_id
    dh_path = edu.asset_path("config/kinematics/right_arm_kinematics.yaml")
    dh = yaml.safe_load(dh_path.read_text())
    assert (
        hashlib.sha256((dh_path.parent / dh["source_urdf"]).read_bytes()).hexdigest()
        == dh["source_urdf_sha256"]
    )
    for description in edu.descriptions:
        root = ET.fromstring(edu.description_xml(description))
        assert root.tag == "robot"


def test_lidar_box_is_present_in_planning_and_base_views_at_the_sensor_mount():
    from alohamini.simulation.frames import rotation_rpy

    model = get_robot_model("alohamini2edu")
    sensor = yaml.safe_load(model.asset_path("config/sensors.yaml").read_text())["lidar"]
    housing = json.loads(model.asset_path("config/lidar_collision.json").read_text())
    expected_center = np.asarray(sensor["xyz"]) + rotation_rpy(sensor["rpy"]) @ np.asarray(
        housing["center_in_reference_m"]
    )
    for name in ("collision", "planning", "navigation", "base"):
        tree = model.description_tree(name)
        link = tree.find("./link[@name='lidar_housing']")
        box = link.find("collision/geometry/box")
        np.testing.assert_allclose(list(map(float, box.get("size").split())), housing["size_m"])
        joint = tree.find("./joint[@name='lidar_housing_joint']")
        assert joint.find("parent").get("link") == sensor["parent_frame"]
        np.testing.assert_allclose(
            list(map(float, joint.find("origin").get("xyz").split())), expected_center
        )
        # Sensor drivers publish the calibrated scan frame independently.
        assert tree.find("./link[@name='laser_frame']") is None


@pytest.mark.parametrize("description", ["base", "navigation"])
def test_navigation_root_does_not_compete_with_ekf(description):
    root = ET.fromstring(get_robot_model("alohamini2edu").description_xml(description))
    links = {link.get("name") for link in root.findall("link")}
    children = [joint.find("child").get("link") for joint in root.findall("joint")]
    assert len(children) == len(set(children))
    assert links - set(children) == {"base_link"}
    assert "root" not in links and "odom" not in links and "map" not in links
    joints = {joint.get("name") for joint in root.findall("joint")}
    assert {"wheel1_joint", "wheel2_joint", "wheel3_joint"} <= joints
    if description == "base":
        assert "vertical_move" not in joints and "left_shoulder_pan" not in joints


@pytest.mark.parametrize("side", ["left", "right"])
def test_edu_fk_ik_reuses_pro_geometry_and_keeps_model_identity(side):
    pro = AlohaMiniArmKinematics.from_model("alohamini2pro", side)
    edu = AlohaMiniArmKinematics.from_model("alohamini2edu", side)
    q = np.array([0.1, -0.5, 0.7, 0.2, -0.1, 0.15])
    np.testing.assert_allclose(pro.forward(q), edu.forward(q), atol=1e-12)
    assert edu.model_reference["model_id"] == "alohamini2edu"
    # Known target with a nearby seed exercises the shared inverse solver.
    result = edu.inverse(edu.forward(q), q + 0.01)
    assert result is not None
    np.testing.assert_allclose(edu.forward(result), edu.forward(q), atol=1e-4)
