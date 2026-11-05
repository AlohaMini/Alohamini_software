"""Apply prepared link-local inertias without changing geometry or kinematics.

CAD reading, report parsing and registration belong to asset preparation.
Runtime loading consumes only the prepared numeric JSON and its provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .inertia import MassProperties, combine
from .organization import description_hashes


def load_inertial_bundle(description, path, locked_positions):
    path = Path(path).expanduser().resolve()
    contents = path.read_bytes()
    definition = json.loads(contents)
    if (
        definition.get("schema_version") != 1
        or definition.get("robot_model") != description.robot_model.model_id
        or definition.get("units") != "kg, m, kg*m^2"
        or definition.get("source_description_sha256") != description_hashes(description)
    ):
        raise ValueError("Inertial binding requires the current robot description and SI units")
    required_locks = definition["required_locked_positions"]
    if any(locked_positions.get(joint) != value for joint, value in required_locks.items()):
        raise ValueError("Inertial binding requires its declared locked joint positions")

    # Only links connected by fixed/explicitly locked joints form one rigid group.
    # Absorbing a child across an active joint changes the mechanism's dynamics.
    parent = {name: name for name in description.links}

    def representative(name):
        while parent[name] != name:
            name = parent[name]
        return name

    for joint in description.joints.values():
        if joint.kind == "fixed" or joint.name in required_locks:
            parent[representative(joint.child)] = representative(joint.parent)

    moving = set()

    def visit(link, inherited=False):
        if inherited:
            moving.add(link)
        for joint in description.by_parent.get(link, []):
            visit(
                joint.child,
                inherited or (joint.kind != "fixed" and joint.name not in required_locks),
            )

    visit(description.root)

    groups = definition["link_groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError("Inertial binding needs explicit rigid link groups")
    replacements, removed, assigned_components = {}, set(), set()
    for group in groups:
        link, replaces = group["link"], group["replaces"]
        if (
            link not in description.links
            or link in replacements
            or not isinstance(replaces, list)
            or link not in replaces
            or len(set(replaces)) != len(replaces)
            or set(replaces) - description.inertials.keys()
            or removed.intersection(replaces)
        ):
            raise ValueError("Inertial groups need distinct existing inertias and a named owner")
        if any(representative(other) != representative(link) for other in replaces):
            raise ValueError(f"Cannot combine inertias across an active joint: {link}")
        sources = group["sources"]
        if not isinstance(sources, list) or not sources:
            raise ValueError("Each rigid group needs explicit source tensors and frame transforms")
        parts = []
        for source in sources:
            properties = MassProperties(**source["properties"])
            parts.append(properties.in_frame(source["T_link_source"], link))
            components = source["component_ids"]
            if (
                not isinstance(components, list)
                or not components
                or any(not isinstance(c, str) or not c for c in components)
                or len(set(components)) != len(components)
                or assigned_components.intersection(components)
            ):
                raise ValueError("CAD material instances must have exactly one inertial owner")
            assigned_components.update(components)
        properties = combine(parts)
        removed.update(replaces)
        com_frame = np.eye(4)
        com_frame[:3, 3] = properties.com_m
        replacements[link] = (
            properties.mass_kg,
            com_frame,
            properties.central_inertia_kg_m2,
        )

    inventory = definition["component_inventory"]
    if (
        not isinstance(inventory, list)
        or len(set(inventory)) != len(inventory)
        or assigned_components - set(inventory)
    ):
        raise ValueError("Inertial assignments must belong to the declared CAD inventory")
    retained = sorted(description.inertials.keys() - removed)
    unassigned = sorted(set(inventory) - assigned_components)
    unresolved = definition["unresolved"]
    fixed_unassigned = definition.get("fixed_unassigned_components", {})
    fixed_components = set()
    for link, components in fixed_unassigned.items():
        if (
            link not in description.links
            or link in moving
            or not isinstance(components, list)
            or not components
            or any(not isinstance(c, str) or not c for c in components)
            or len(set(components)) != len(components)
            or fixed_components.intersection(components)
            or set(components) - set(unassigned)
        ):
            raise ValueError(
                "Unassigned fixed components need a stationary owner and unique inventory entries"
            )
        fixed_components.update(components)
    retained_active = sorted(set(retained) & moving)
    unassigned_active = sorted(set(unassigned) - fixed_components)
    dynamics_ready = not (retained_active or unassigned_active or unresolved)
    complete = not (retained or unassigned or unresolved)
    armature = definition.get("joint_armature_kg_m2", {})
    for name, value in armature.items():
        joint = description.joints.get(name)
        if (
            joint is None
            or joint.kind not in ("revolute", "continuous")
            or name in locked_positions
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "Joint armature needs an active revolute joint and finite nonnegative inertia"
            )
    effective = {
        **{name: values for name, values in description.inertials.items() if name not in removed},
        **replacements,
    }
    return effective, {
        "bundle_sha256": hashlib.sha256(contents).hexdigest(),
        "definition": definition,
        "complete": complete,
        "dynamics_ready": dynamics_ready,
        "applied_links": sorted(replacements),
        "removed_urdf_inertias": sorted(removed),
        "retained_urdf_inertias": retained,
        "unassigned_cad_components": unassigned,
        "retained_active_urdf_inertias": retained_active,
        "unassigned_active_cad_components": unassigned_active,
        "fixed_unassigned_components": fixed_unassigned,
        "joint_armature_kg_m2": armature,
        "effective_mass_kg": sum(value[0] for value in effective.values()),
        "scope": (
            "complete declared CAD binding; hardware identification remains separate"
            if complete
            else (
                "active mechanism covered under required locks; "
                "stationary mass subtotal is incomplete"
                if dynamics_ready
                else "partial active inertial binding; dynamics/collection disabled"
            )
        ),
    }
