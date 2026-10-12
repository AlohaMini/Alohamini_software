# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable model descriptions, separate from installed-device calibration."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ActuatorSpec:
    """Motor IDs are unique within one bus, not across the robot."""

    name: str
    bus: str
    motor_id: int
    motor_model: str


def asset_path(directory: Path, relative: str) -> Path:
    """Resolve a model-local file without following paths outside the asset root."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Asset path must be a nonempty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Asset path must stay inside the model directory")
    root = directory.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Asset path escapes the model directory")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


@dataclass(frozen=True)
class ArmRomBand:
    """One joint's expected range of motion, in encoder ticks.

    Mirrors the vocabulary of the installed geometry it is derived from
    (`joint_limits.yaml`: `type`, `calibrated`), so the two can be compared
    side by side. Derived from installed geometry, never from a device's own
    calibration: the expected span is what the mechanism can physically travel,
    so a measured span outside this band means the calibration sampling is wrong,
    not the mechanism. Absolute ticks are deliberately absent because they depend
    on each unit's homing offset; only the span is portable across units.
    """

    type: str
    span_tick: int
    span_min_tick: int
    span_max_tick: int

    def __post_init__(self) -> None:
        if self.type not in ("revolute", "continuous"):
            raise ValueError("type must be 'revolute' or 'continuous'")
        for field_name in ("span_tick", "span_min_tick", "span_max_tick"):
            value = getattr(self, field_name)
            if type(value) is not int or not 0 <= value <= 4095:
                raise ValueError(f"{field_name} must be an integer in [0, 4095]")
        if not self.span_min_tick <= self.span_tick <= self.span_max_tick:
            raise ValueError("span_tick must lie within the accepted span band")

    @property
    def continuous(self) -> bool:
        """A joint with no mechanical end stop accepts a full turn."""
        return self.type == "continuous"

    def accepts(self, span_tick: int) -> bool:
        """Report whether a measured span is physically plausible for this joint."""
        return self.span_min_tick <= span_tick <= self.span_max_tick


@dataclass(frozen=True)
class ArmRom:
    """Model-level expected arm travel, and where it came from.

    `source` and `calibrated` carry the same provenance vocabulary as
    `joint_limits.yaml` so a reviewer can trace every band back to its origin.
    """

    source: str
    calibrated: str
    joints: Mapping[str, ArmRomBand]


@dataclass(frozen=True)
class RobotModel:
    """Nominal layout and description assets; not a physical safety configuration."""

    model_id: str
    actuators: tuple[ActuatorSpec, ...]
    wheel_radius_m: float
    base_radius_m: float
    lift_lead_m_per_rev: float
    asset_revision: int
    descriptions: Mapping[str, str]
    directory: Path = field(repr=False, compare=False)
    arm_rom: ArmRom | None = None

    def rom_band(self, joint: str) -> ArmRomBand | None:
        """Expected travel for one joint, or None when the model declares none."""
        return None if self.arm_rom is None else self.arm_rom.joints.get(joint)

    def asset_path(self, relative: str) -> Path:
        return asset_path(self.directory, relative)

    def description_path(self, name: str) -> Path:
        """Return a description file; no fallback to another model or description."""
        if name not in self.descriptions:
            raise ValueError(f"Model {self.model_id!r} has no description {name!r}")
        return self.asset_path(self.descriptions[name])

    def description_xml(self, name: str) -> str:
        """Render a description with contained mesh file URIs; leave assets unchanged."""
        path = self.description_path(name)
        root = ET.parse(path).getroot()
        for mesh in root.findall(".//mesh"):
            resource = (path.parent / mesh.attrib["filename"]).resolve()
            if not resource.is_relative_to(self.directory.resolve()) or not resource.is_file():
                raise ValueError("Robot mesh must exist inside the model directory")
            mesh.set("filename", resource.as_uri())
        return ET.tostring(root, encoding="unicode")
