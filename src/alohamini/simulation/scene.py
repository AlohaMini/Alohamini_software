"""Small, explicit scene configuration shared by research backends."""

import json
from pathlib import Path

import numpy as np

from .frames import rigid_transform


def load_scene(path=None):
    config = {} if path is None else json.loads(Path(path).expanduser().read_text())
    if not isinstance(config, dict) or config.get("schema_version", 1) != 1:
        raise ValueError("Scene configuration requires schema_version 1")
    allowed = {
        "schema_version",
        "T_world_base",
        "initial_positions",
        "objects",
        "timestep_s",
        "control_frequency_hz",
        "base_mode",
        "seed",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unknown scene fields: {sorted(config.keys() - allowed)}")
    result = {
        "schema_version": 1,
        "T_world_base": np.eye(4).tolist(),
        "initial_positions": {},
        "objects": [],
        "timestep_s": 0.001,
        "control_frequency_hz": 20,
        "base_mode": "fixed",
        "seed": 0,
        **config,
    }
    result["T_world_base"] = rigid_transform(result["T_world_base"]).tolist()
    dt, frequency = result["timestep_s"], result["control_frequency_hz"]
    if (
        not np.isfinite([dt, frequency]).all()
        or not 0 < dt <= 0.01
        or frequency <= 0
        or frequency != round(frequency)
        or not np.isclose(1 / dt, round(1 / dt), rtol=0, atol=1e-9)
        or not np.isclose(1 / dt / frequency, round(1 / dt / frequency), rtol=0, atol=1e-9)
    ):
        raise ValueError("Use integer physics/control frequencies with integer substeps")
    if result["base_mode"] not in ("fixed", "kinematic"):
        raise ValueError("base_mode must be fixed or kinematic")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(result["initial_positions"], dict):
        raise ValueError("initial_positions must map URDF joint names to rad/m values")
    if not isinstance(result["objects"], list):
        raise ValueError("objects must be a list")
    names = set()
    for obj in result["objects"]:
        if not isinstance(obj, dict) or set(obj) - {
            "name",
            "shape",
            "size",
            "T_world_object",
            "mass_kg",
            "color",
            "friction",
        }:
            raise ValueError("Invalid scene object fields")
        name = obj.get("name")
        if not isinstance(name, str) or not name.isidentifier() or name in names:
            raise ValueError("Scene objects need distinct identifier names")
        names.add(name)
        shape, size = obj.get("shape"), np.asarray(obj.get("size"), dtype=float)
        count = {"box": 3, "sphere": 1, "cylinder": 2}.get(shape)
        if (
            count is None
            or size.shape != (count,)
            or not np.isfinite(size).all()
            or (size <= 0).any()
        ):
            raise ValueError(
                "Use box full XYZ dimensions, sphere radius, or cylinder radius/length"
            )
        rigid_transform(obj.get("T_world_object", np.eye(4)))
        mass, friction = obj.get("mass_kg", 0), obj.get("friction", 1)
        if not np.isfinite([mass, friction]).all() or mass < 0 or friction < 0:
            raise ValueError("Use nonnegative mass and friction; mass zero denotes a fixed fixture")
        color = np.asarray(obj.get("color", [0.7, 0.7, 0.7, 1]), dtype=float)
        if (
            color.shape != (4,)
            or not np.isfinite(color).all()
            or (color < 0).any()
            or (color > 1).any()
        ):
            raise ValueError("Color requires four finite RGBA values in [0,1]")
    return result
