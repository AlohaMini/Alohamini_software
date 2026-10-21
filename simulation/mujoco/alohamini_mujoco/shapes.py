"""Native collision primitives in explicit SI link frames.

Box sizes and cylinder lengths are full dimensions. Cylinders use local Z,
as in URDF; backend axis conversions belong to each backend's loader.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .frames import rigid_transform, rotation_rpy


def _text(values):
    return " ".join(format(float(x), ".17g") for x in np.asarray(values).ravel())


@dataclass(frozen=True)
class Primitive:
    kind: str
    parameters: dict
    T_link_geometry: np.ndarray

    def __post_init__(self):
        expected = {"box": {"size"}, "sphere": {"radius"}, "cylinder": {"radius", "length"}}
        if (
            not isinstance(self.kind, str)
            or self.kind not in expected
            or not isinstance(self.parameters, dict)
        ):
            raise ValueError("Unsupported native collision primitive")
        if self.parameters.keys() != expected[self.kind]:
            raise ValueError("Primitive dimensions must be explicit; no inferred units or axes")
        parameters = {}
        for name, value in self.parameters.items():
            size = 3 if name == "size" else None
            sequence = value if isinstance(value, (tuple, list)) else [value]
            if any(isinstance(item, (bool, np.bool_)) for item in sequence):
                raise ValueError("Primitive dimensions must be positive finite numbers")
            raw = np.asarray(value)
            if raw.dtype.kind not in "ifu" or raw.shape != ((size,) if size else ()):
                raise ValueError("Primitive dimensions must be positive finite numbers")
            array = raw.astype(float)
            if not np.isfinite(array).all() or (array <= 0).any():
                raise ValueError("Primitive dimensions must be positive finite numbers")
            parameters[name] = array.tolist() if size else float(array)
        pose = rigid_transform(self.T_link_geometry)
        pose.flags.writeable = False
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "T_link_geometry", pose)

    def record(self):
        return {
            "type": self.kind,
            "parameters": dict(self.parameters),
            "T_link_geometry": self.T_link_geometry.tolist(),
        }

    @classmethod
    def from_record(cls, record):
        if (
            not isinstance(record, dict)
            or {"type", "parameters", "T_link_geometry"} != record.keys()
        ):
            raise ValueError("Incomplete native collision primitive")
        return cls(record["type"], record["parameters"], record["T_link_geometry"])

    def collision_node(self, name):
        rotation = self.T_link_geometry[:3, :3]
        cosine = math.hypot(rotation[0, 0], rotation[1, 0])
        pitch = math.atan2(-rotation[2, 0], cosine)
        if cosine > 1e-12:
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            roll, yaw = math.atan2(-rotation[1, 2], rotation[1, 1]), 0.0
        rpy = [roll, pitch, yaw]
        if not np.allclose(rotation_rpy(rpy), rotation, rtol=0, atol=2e-12):
            raise ValueError("Cannot encode the declared primitive frame in URDF RPY")
        node = ET.Element("collision", name=name)
        ET.SubElement(node, "origin", xyz=_text(self.T_link_geometry[:3, 3]), rpy=_text(rpy))
        ET.SubElement(
            ET.SubElement(node, "geometry"),
            self.kind,
            **{k: _text(v) for k, v in self.parameters.items()},
        )
        return node
