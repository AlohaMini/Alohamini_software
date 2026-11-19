# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable model descriptions, separate from installed-device calibration."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .cameras import DEFAULT_CAMERAS


@dataclass(frozen=True)
class ActuatorSpec:
    """Motor IDs are unique within one bus, not across the robot."""

    name: str
    bus: str
    motor_id: int
    motor_model: str


@dataclass(frozen=True)
class DescriptionView:
    source: str
    overlay: str | None = None
    root_link: str | None = None
    geometry: bool = True
    exclude_joints: tuple[str, ...] = ()


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
    kinematic_chains: Mapping[str, str] = field(default_factory=dict)
    kinematic_presets: Mapping[str, str] = field(default_factory=dict)
    default_kinematic_preset: str | None = None
    description_views: Mapping[str, DescriptionView] = field(default_factory=dict)
    default_cameras: tuple[str, ...] = DEFAULT_CAMERAS

    def asset_path(self, relative: str) -> Path:
        return asset_path(self.directory, relative)

    def description_path(self, name: str) -> Path:
        """Return a description file; no fallback to another model or description."""
        if name in self.description_views:
            raise ValueError(f"Description {name!r} is generated; use description_xml()")
        if name not in self.descriptions:
            raise ValueError(f"Model {self.model_id!r} has no description {name!r}")
        return self.asset_path(self.descriptions[name])

    @property
    def description_names(self) -> tuple[str, ...]:
        return (*self.descriptions, *self.description_views)

    def description_source_path(self, name: str) -> Path:
        """Return the authoritative source, including for an in-memory view."""
        view = self.description_views.get(name)
        return self.description_path(view.source if view is not None else name)

    def description_sources(self, name: str) -> tuple[Path, ...]:
        sources = [self.description_source_path(name)]
        view = self.description_views.get(name)
        if view is not None and view.overlay is not None:
            sources.append(self.asset_path(view.overlay))
        return tuple(sources)

    def description_tree(self, name: str, *, _sources=None) -> ET.Element:
        """Build an independent tree from a source and a declared nonphysical overlay."""
        sources = (
            {path: path.read_bytes() for path in self.description_sources(name)}
            if _sources is None
            else _sources
        )
        source = self.description_source_path(name)
        tree = ET.fromstring(sources[source])
        view = self.description_views.get(name)
        if view is not None:
            from .views import derive_tree

            overlay = (
                ET.fromstring(sources[self.asset_path(view.overlay)]) if view.overlay else None
            )
            tree = derive_tree(tree, view, overlay)
        return tree

    def kinematic_chain_path(self, name: str) -> Path:
        """Resolve an explicitly registered chain; never borrow another model's geometry."""
        if name not in self.kinematic_chains:
            raise ValueError(f"Model {self.model_id!r} has no kinematic chain {name!r}")
        return self.asset_path(self.kinematic_chains[name])

    def kinematic_preset_path(self, name: str) -> Path:
        """Resolve a named recording-coordinate protocol independently of device files."""
        if name not in self.kinematic_presets:
            raise ValueError(f"Model {self.model_id!r} has no kinematic preset {name!r}")
        return self.asset_path(self.kinematic_presets[name])

    def description_xml(self, name: str, *, resolve_meshes: bool = True) -> str:
        """Render a description with contained mesh file URIs; leave assets unchanged."""
        path = self.description_source_path(name)
        root = self.description_tree(name)
        for mesh in root.findall(".//mesh"):
            resource = (path.parent / mesh.attrib["filename"]).resolve()
            if not resource.is_relative_to(self.directory.resolve()) or not resource.is_file():
                raise ValueError("Robot mesh must exist inside the model directory")
            if resolve_meshes:
                mesh.set("filename", resource.as_uri())
        return ET.tostring(root, encoding="unicode")
