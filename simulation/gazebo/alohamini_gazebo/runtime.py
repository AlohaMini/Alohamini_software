"""Gazebo commands and system plugins for the supported ROS distributions."""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GazeboRuntime:
    command: tuple[str, ...]
    library_prefix: str
    namespace: str

    def system_plugin(self, component: str, class_name: str) -> dict[str, str]:
        return {
            "filename": f"{self.library_prefix}-{component}-system",
            "name": f"{self.namespace}::{class_name}",
        }


def gazebo_runtime(ros_distro: str | None = None) -> GazeboRuntime:
    distro = ros_distro or os.environ.get("ROS_DISTRO", "jazzy")
    if distro == "humble":
        return GazeboRuntime(
            ("ign", "gazebo", "--force-version", "6"),
            "ignition-gazebo",
            "ignition::gazebo::systems",
        )
    if distro == "jazzy":
        return GazeboRuntime(
            ("gz", "sim", "--force-version", "8"), "gz-sim", "gz::sim::systems"
        )
    raise ValueError(f"Unsupported ROS distribution for Gazebo: {distro}")


def world_description(path: Path, runtime: GazeboRuntime) -> str:
    path = path.resolve()
    tree = ET.parse(path)
    world = tree.getroot().find("world")
    # sdformat12 ignores plugins attached to <include>. Expand local models
    # that carry plugins so Fortress and Harmonic load the same systems.
    for include in list(world.findall("include")):
        if include.find("plugin") is None:
            continue
        uri = include.findtext("uri", "")
        resources = (path.parent.parent / "models").resolve()
        model_path = (resources / uri.removeprefix("model://") / "model.sdf").resolve()
        if not uri.startswith("model://") or not model_path.is_relative_to(resources):
            raise ValueError(f"Expected a local model resource: {uri}")
        model = ET.parse(model_path).getroot().find("model")
        model.set("name", include.findtext("name", model.get("name")))
        pose = include.find("pose")
        if pose is not None:
            old_pose = model.find("pose")
            if old_pose is not None:
                model.remove(old_pose)
            model.append(pose)
        model.extend(include.findall("plugin"))
        index = list(world).index(include)
        world.remove(include)
        world.insert(index, model)
    for plugin in tree.findall(".//plugin"):
        if not plugin.get("filename", "").startswith("gz-sim-"):
            continue
        component = plugin.attrib["filename"].removeprefix("gz-sim-").removesuffix("-system")
        class_name = plugin.attrib["name"].rsplit("::", 1)[-1]
        plugin.attrib.update(runtime.system_plugin(component, class_name))
    return ET.tostring(tree.getroot(), encoding="unicode", xml_declaration=True)
