"""Planning and FK views must retain the authoritative physical robot unchanged."""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from alohamini.model import get_robot_model, load_robot_model
from alohamini.simulation.description import Description
from alohamini.simulation.frames import transform


def elements(tree, tag):
    return {node.attrib["name"]: ET.tostring(node) for node in tree.findall(tag)}


def test_planning_adds_only_nonphysical_base_coordinates_and_shares_all_robot_elements():
    model = get_robot_model("alohamini2pro")
    physical, planning = model.description_tree("collision"), model.description_tree("planning")
    for tag in ("link", "joint", "material"):
        source, derived = elements(physical, tag), elements(planning, tag)
        assert all(derived[name] == value for name, value in source.items())
    physical_links, planning_links = elements(physical, "link"), elements(planning, "link")
    assert planning_links.keys() - physical_links.keys() == {"root", "root_x_link", "root_y_link"}
    assert len(planning.findall(".//collision")) == len(physical.findall(".//collision")) == 1115
    assert model.description_path("semantic") == model.description_path("semantic_planning")
    assert model.description_source_path("planning") == model.description_path("collision")
    with pytest.raises(ValueError, match="generated"):
        model.description_path("planning")


def test_generated_base_motion_matches_physical_world_pose_for_every_retained_frame():
    model = get_robot_model("alohamini2pro")
    physical, planning = Description(model), Description(model, name="planning")
    assert physical.prepared == planning.prepared
    assert physical.inertials.keys() == planning.inertials.keys()
    assert physical.exclusions == planning.exclusions
    rng = np.random.default_rng(716)
    for _ in range(6):
        pose = rng.uniform([-1, -1, -2], [1, 1, 2])
        q = {
            n: float(rng.uniform(j.lower, j.upper))
            for n, j in physical.joints.items() if j.lower is not None
        }
        expected = physical.forward(q, transform([pose[0], pose[1], 0], [0, 0, pose[2]]))
        actual = planning.forward({**q, **dict(zip(
            ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint"), pose,
            strict=True,
        ))})
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], atol=1e-14, rtol=0)


@pytest.mark.parametrize("view,root,count", [
    ("kinematic", "root", 32), ("left_kinematic", "left_Base", 10),
    ("right_kinematic", "right_Base", 10),
])
def test_kinematic_views_keep_source_joints_without_geometry_or_inertias(view, root, count):
    model = get_robot_model("alohamini2pro")
    source, derived = model.description_tree("planning"), model.description_tree(view)
    parents = {j.find("child").attrib["link"] for j in derived.findall("joint")}
    assert {n.attrib["name"] for n in derived.findall("link")} - parents == {root}
    assert len(derived.findall("link")) == count
    assert not any(derived.findall(".//" + tag) for tag in ("visual", "collision", "inertial"))
    authoritative = elements(source, "joint")
    assert all(authoritative[name] == value for name, value in elements(derived, "joint").items())


def test_view_rendering_is_independent_and_does_not_write_assets():
    model = get_robot_model("alohamini2pro")
    files = {p: p.read_bytes() for p in model.description_sources("planning")}
    first = model.description_tree("planning")
    first.find("link[@name='base_link']").set("name", "changed")
    assert model.description_tree("planning").find("link[@name='base_link']") is not None
    output = ET.fromstring(model.description_xml("planning"))
    assert all(n.attrib["filename"].startswith("file:") for n in output.findall(".//mesh"))
    assert all(p.read_bytes() == contents for p, contents in files.items())


@pytest.mark.parametrize("mutation", ["inertia", "axis", "cycle", "shadow"])
def test_planar_overlay_rejects_physical_content_and_wrong_coordinate_chains(mutation):
    model = get_robot_model("alohamini2pro")
    source = model.description_path("collision")
    # Keep source mesh references valid and alter only the nonphysical overlay.
    overlay = ET.parse(model.asset_path("config/planar_base.urdf"))
    if mutation == "inertia":
        ET.SubElement(overlay.find("link"), "inertial")
    elif mutation == "axis":
        overlay.find("joint/axis").set("xyz", "0 0 1")
    elif mutation == "cycle":
        overlay.find("joint/child").set("link", "root")
    else:
        overlay.find("joint").set("name", "vertical_move")
    from alohamini.model.views import attach_planar_base

    with pytest.raises(ValueError, match="Planar-base"):
        attach_planar_base(ET.parse(source).getroot(), overlay.getroot())


def test_view_declaration_cannot_shadow_assets_or_reference_another_view(tmp_path):
    import json

    model = get_robot_model("alohamini2pro")
    metadata = json.loads(model.asset_path("model.json").read_text())
    metadata["descriptions"] = {"collision": "robot.urdf"}
    metadata.pop("kinematic_chains", None)
    metadata.pop("kinematic_presets", None)
    metadata.pop("default_kinematic_preset", None)
    (tmp_path / "robot.urdf").write_text('<robot name="fixture"><link name="base_link"/></robot>')
    for views in (
        {"collision": {"source": "collision"}}, {"a": {"source": "b"}, "b": {"source": "a"}}
    ):
        metadata["description_views"] = views
        (tmp_path / "model.json").write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="view"):
            load_robot_model(tmp_path)
