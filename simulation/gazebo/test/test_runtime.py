import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from alohamini_gazebo.runtime import gazebo_runtime, world_description
from alohamini_gazebo.sim_description import make_sim_description

from alohamini.model import get_robot_model


@pytest.mark.parametrize(
    "distro,executable,version,prefix,namespace",
    [
        ("humble", "ign", "6", "ignition-gazebo", "ignition::gazebo::systems"),
        ("jazzy", "gz", "8", "gz-sim", "gz::sim::systems"),
    ],
)
def test_world_and_robot_load_plugins_from_the_same_gazebo_version(
    tmp_path, distro, executable, version, prefix, namespace
):
    runtime = gazebo_runtime(distro)
    assert runtime.command[0] == executable
    assert runtime.command[-1] == version
    source = Path(__file__).parents[1] / "worlds/lift_pick_place.sdf"
    original = source.read_bytes()
    world = ET.fromstring(world_description(source, runtime))
    plugins = world.findall("./world/plugin")
    assert {p.attrib["name"] for p in plugins} == {
        f"{namespace}::{name}"
        for name in ("Physics", "UserCommands", "SceneBroadcaster", "Contact")
    }
    assert all(p.attrib["filename"].startswith(prefix + "-") for p in plugins)
    pose_publisher = world.find(
        f"./world/model[@name='demo_object']/plugin[@name='{namespace}::PosePublisher']"
    )
    assert pose_publisher is not None
    assert pose_publisher.attrib["filename"] == f"{prefix}-pose-publisher-system"
    assert pose_publisher.findtext("publish_model_pose") == "true"
    assert pose_publisher.findtext("topic") == "/model/demo_object/pose"
    model = get_robot_model("alohamini2pro")
    robot = ET.fromstring(
        make_sim_description(
            model.directory / "urdf/alohamini2pro.urdf",
            tmp_path / "controllers.yaml",
            runtime=runtime,
        )
    )
    grasp = robot.find(f"./gazebo/plugin[@name='{namespace}::DetachableJoint']")
    assert grasp is not None
    assert grasp.attrib["filename"] == f"{prefix}-detachable-joint-system"
    assert grasp.findtext("child_model") == "demo_object"
    assert source.read_bytes() == original


def test_unsupported_distribution_fails_before_starting_gazebo():
    with pytest.raises(ValueError, match="Unsupported ROS distribution"):
        gazebo_runtime("unknown")
