"""Shared primitive research scenes; task logic lives in task adapters."""

import xml.etree.ElementTree as ET

import numpy as np

from alohamini.simulation.scene import load_scene

from .model import pose_attributes, text


class ResearchScene:
    def __init__(self, path):
        self.configuration = load_scene(path)
        self.manifest = {"benchmark": "alohamini-research-v1", "configuration": self.configuration}

    def augment(self, root, assets):
        world = root.find("worldbody")
        if self.configuration["base_mode"] == "kinematic":
            world.find("body[@name='base_link']").set("mocap", "true")
        for obj in self.configuration["objects"]:
            name, size, shape = "scene__" + obj["name"], obj["size"], obj["shape"]
            body = ET.SubElement(
                world, "body", name=name, **pose_attributes(obj.get("T_world_object", np.eye(4)))
            )
            if shape == "box":
                dimensions = np.asarray(size) / 2
            elif shape == "cylinder":
                dimensions = [size[0], size[1] / 2]
            else:
                dimensions = size
            mass = obj.get("mass_kg", 0)
            if mass:
                ET.SubElement(body, "freejoint", name=name + "_joint")
                if shape == "box":
                    x, y, z = size
                    tensor = mass / 12 * np.array([y * y + z * z, x * x + z * z, x * x + y * y])
                elif shape == "sphere":
                    tensor = np.full(3, 0.4 * mass * size[0] ** 2)
                else:
                    r, length = size
                    tensor = [mass * (3 * r * r + length * length) / 12] * 2 + [mass * r * r / 2]
                ET.SubElement(
                    body, "inertial", mass=text([mass]), pos="0 0 0", diaginertia=text(tensor)
                )
            attributes = {"type": shape, "size": text(dimensions), "mass": "0"}
            ET.SubElement(
                body,
                "geom",
                name=name + "_collision",
                **attributes,
                group="3",
                friction=text([obj.get("friction", 1), 0.005, 0.0001]),
                rgba=".1 .8 .3 .3",
            )
            ET.SubElement(
                body,
                "geom",
                name=name + "_visual",
                **attributes,
                group="2",
                contype="0",
                conaffinity="0",
                rgba=text(obj.get("color", [0.7, 0.7, 0.7, 1])),
            )

    def initialize(self, robot, data):
        pass

    def settle(self, follower):
        # Research scenes start at their declared state. Tasks can provide
        # explicit settling and acceptance checks in their own adapter.
        return
