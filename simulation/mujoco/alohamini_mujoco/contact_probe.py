"""Native sphere contacts against explicitly registered CAD surface locations."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .cad import file_sha256
from .cad_collision import _file, _source, load_cad_components
from .compilation import compile_xml
from .frames import points_in_frame, vector
from .geometry import SurfaceIndex
from .meshes import Mesh
from .native_distance import checked_geom_distance


def _winding(mesh, point):
    a, b, c = np.moveaxis(mesh.triangles - point, 1, 0)
    la, lb, lc = (np.linalg.norm(p, axis=1) for p in (a, b, c))
    denominator = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    return float(np.arctan2(numerator, denominator).sum() / (2 * math.pi))


def _component_fixture_inputs(robot, directory, *, name_prefix="cad_fixture"):
    if "rigid_assembly_fixture" in robot.manifest:
        raise ValueError("Add collision components before assigning their rigid owners")
    if robot.scene is not None:
        raise ValueError("Use a robot-only model for isolated CAD component fixtures")
    pieces, metadata = load_cad_components(robot.description, directory)
    root = ET.fromstring(robot.xml)
    assets, geometry, selected, bindings = dict(robot.assets), dict(robot.geometry), {}, {}
    for link, meshes in pieces.items():
        body = root.find(f'.//body[@name="{link}"]')
        if body is None:
            raise ValueError("CAD component fixture link is absent from the native model")
        selected[link] = []
        target = metadata["definition"]["links"][link]["cad_geometry"]["registration"]["target"]
        for i, mesh in enumerate(meshes):
            name = f"{name_prefix}_{link}_{i}"
            if name in geometry or name + ".msh" in assets:
                raise ValueError("CAD component fixture asset name is already in use")
            assets[name + ".msh"] = mesh.msh_bytes()
            ET.SubElement(root.find("asset"), "mesh", name=name, file=name + ".msh")
            ET.SubElement(body, "geom", name=name, type="mesh", mesh=name, mass="0", group="4")
            geometry[name] = {"kind": "collision", "link": link}
            selected[link].append(name)
            bindings[name] = {
                "source_link": link,
                "visual_index": target["visual_index"],
                "component_index": target["component_index"],
            }
    xml = ET.tostring(root, encoding="unicode")
    return dict(
        xml=xml,
        assets=assets,
        geometry=geometry,
        manifest={
            **robot.manifest,
            "cad_component_fixture": metadata,
            "cad_component_geoms": selected,
            "cad_component_bindings": bindings,
        },
    )


def attach_component_fixture(robot, directory):
    """Compile an isolated query fixture, retaining every original collider.

    Same-body overlaps are intentional; this model cannot run following or
    collection. Probe construction combines components and the sphere before
    compiling, avoiding retention of an extra complete robot model.
    """
    inputs = _component_fixture_inputs(robot, directory)
    return replace(robot, model=compile_xml(inputs["xml"], assets=inputs["assets"]), **inputs)


def _component_fixtures_inputs(robot, directories):
    """Assemble registered components before native compilation.

    This retains all original geometry and never promotes partial coverage to
    a link replacement. The same source component cannot be added twice.
    """
    if not directories:
        raise ValueError("Select at least one registered component bundle")
    current, bundles, selected, covered, bindings = robot, [], {}, set(), {}
    for index, directory in enumerate(directories):
        inputs = _component_fixture_inputs(current, directory, name_prefix=f"cad_fixture{index}")
        bundle = inputs["manifest"]["cad_component_fixture"]
        for link, record in bundle["definition"]["links"].items():
            target = record["cad_geometry"]["registration"]["target"]
            identity = (link, target["visual_index"], target["component_index"])
            if identity in covered:
                raise ValueError("A visual component is already present in the inspection assembly")
            covered.add(identity)
            selected.setdefault(link, []).extend(inputs["manifest"]["cad_component_geoms"][link])
        bundles.append(bundle)
        bindings.update(inputs["manifest"]["cad_component_bindings"])
        current = replace(current, **inputs)
    manifest = {
        **robot.manifest,
        "cad_component_fixture": {
            "coverage": "multiple visual components; complete-link coverage unverified",
            "components": bundles,
        },
        "cad_component_geoms": selected,
        "cad_component_bindings": bindings,
    }
    return dict(
        xml=current.xml, assets=current.assets, geometry=current.geometry, manifest=manifest
    )


def attach_component_fixtures(robot, directories):
    """Attach a scoped inspection assembly to an existing robot, compiling once."""
    inputs = _component_fixtures_inputs(robot, directories)
    return replace(robot, model=compile_xml(inputs["xml"], assets=inputs["assets"]), **inputs)


def probe_cad_contacts(robot, mesh_bundle, configuration, *, component_bundle=None):
    if "rigid_assembly_fixture" in robot.manifest:
        raise ValueError("Source-frame contact qualification requires the original rigid owners")
    inputs = (
        _component_fixture_inputs(robot, component_bundle)
        if component_bundle is not None
        else dict(
            xml=robot.xml, assets=robot.assets, geometry=robot.geometry, manifest=robot.manifest
        )
    )
    component_scope = "cad_component_fixture" in inputs["manifest"]
    directory, configuration = (
        Path(mesh_bundle).expanduser().resolve(),
        Path(configuration).expanduser().resolve(),
    )
    config_hash = file_sha256(configuration)
    config = json.loads(configuration.read_text())
    source, metadata, fingerprint = _source(directory)
    radius, tolerance = float(config["sphere_radius_m"]), float(config["tolerance_m"])
    clearances = np.asarray(config["clearances_m"], dtype=float)
    if (
        config.get("schema_version") != 1
        or config.get("source_mesh_manifest_sha256") != fingerprint
        or not math.isfinite(radius)
        or radius <= 0
        or not math.isfinite(tolerance)
        or tolerance <= 0
        or clearances.ndim != 1
        or not len(clearances)
        or not np.isfinite(clearances).all()
        or (clearances <= -radius).any()
        or not config.get("cases")
    ):
        raise ValueError("Contact probes need explicit CAD identity, SI dimensions and named cases")
    if robot.scene is not None:
        raise ValueError("Use a robot-only model for the independent CAD contact fixture")
    bundle = inputs["manifest"].get("cad_component_fixture") or inputs["manifest"].get(
        "collision_bundle"
    )
    if not bundle:
        raise ValueError("Contact probes require an explicitly qualified CAD collision bundle")
    if "components" in bundle:
        raise ValueError("Probe one source component bundle at a time")
    root = ET.fromstring(inputs["xml"])
    if root.find('.//geom[@name="cad_contact_probe"]') is not None:
        raise ValueError("CAD contact probe name is already in use")
    fixture = ET.SubElement(root.find("worldbody"), "body", name="cad_probe_body", mocap="true")
    ET.SubElement(
        fixture,
        "geom",
        name="cad_contact_probe",
        type="sphere",
        size=str(radius),
        mass="0",
        contype="1",
        conaffinity="1",
        group="4",
    )
    xml = ET.tostring(root, encoding="unicode")
    inputs["xml"] = xml
    augmented = replace(robot, model=compile_xml(xml, assets=inputs["assets"]), **inputs)
    robot = augmented
    model = augmented.model
    probe = model.geom("cad_contact_probe").id
    mocap = model.body_mocapid[model.body("cad_probe_body").id]
    records = bundle["definition"]["links"]
    rows = []
    for case in config["cases"]:
        link = case["link"]
        qualification = records[link].get("cad_geometry", {})
        if (
            qualification.get("source_mesh_manifest_sha256") != fingerprint
            or qualification.get("passed") is not True
        ):
            raise ValueError("Contact fixture CAD source does not match the named collider link")
        pose = qualification["registration"]["T_link_part"]
        reference = Mesh(points_in_frame(pose, source.vertices), source.faces)
        index = SurfaceIndex(reference.triangles)
        point = vector(case["point_link_m"], 3, "CAD contact point")
        normal = vector(case["outward_normal_link"], 3, "CAD contact normal")
        length = np.linalg.norm(normal)
        if length == 0 or not case.get("joint_positions"):
            raise ValueError("Provide a nonzero outward normal and named joint configurations")
        normal /= length
        distance, nearest = index.closest(point[None])
        reference_error = metadata.get("surface_reference_error_m", metadata["linear_deflection_m"])
        if reference_error is None or not np.isfinite(reference_error) or reference_error < 0:
            raise ValueError("Contact source requires an explicit finite reference error bound")
        if distance[0] > reference_error:
            raise ValueError("Declared contact point is not on the supplied CAD mesh")
        triangle = reference.triangles[nearest[0]]
        outward = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if outward @ normal <= 0:
            raise ValueError("Declared probe direction is not outward from the CAD surface")
        ids = {
            model.geom(name).id
            for name, geometry in robot.geometry.items()
            if geometry["kind"] == "collision"
            and geometry["link"] == link
            and (not component_scope or name in robot.manifest["cad_component_geoms"][link])
        }
        if not ids:
            raise ValueError("Named CAD contact link has no collision geometry")
        body = model.body(link).id
        for positions in case["joint_positions"]:
            if not isinstance(positions, dict):
                raise ValueError("Probe configurations require explicit URDF joint names")
            data = augmented.data(positions)
            rotation, translation = data.xmat[body].reshape(3, 3), data.xpos[body]
            for clearance in clearances:
                center = point + normal * (radius + clearance)
                if abs(_winding(reference, center)) > 0.5:
                    raise ValueError("Probe sphere center must remain outside the source solid")
                expected = float(index.closest(center[None])[0][0] - radius)
                arithmetic = 128 * np.finfo(float).eps * max(1.0, np.abs(center).max())
                if abs(expected) <= arithmetic:
                    raise ValueError(
                        "Reference contact sign is unresolved; choose a finite clearance"
                    )
                world_center = rotation @ center + translation
                data.mocap_pos[mocap] = world_center
                data.mocap_quat[mocap] = [1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                if not np.allclose(data.geom_xpos[probe], world_center, atol=1e-12, rtol=0):
                    raise ValueError("Native probe position differs from the declared link frame")
                maximum = radius + abs(clearance) + tolerance * 2
                best, best_witness = maximum, None
                for geom in ids:
                    measured, witness = checked_geom_distance(model, data, geom, probe, maximum)
                    if measured < best:
                        best, best_witness = float(measured), witness.copy()
                if best_witness is None:
                    raise ValueError(
                        "Native collision geometry is outside the contact search interval"
                    )
                contact_count = sum(
                    (c.geom1 == probe and c.geom2 in ids) or (c.geom2 == probe and c.geom1 in ids)
                    for c in data.contact
                )
                actual_normal = world_center - best_witness[:3]
                norm = np.linalg.norm(actual_normal)
                angle = (
                    None
                    if not norm
                    else float(
                        np.degrees(
                            np.arccos(np.clip(actual_normal @ (rotation @ normal) / norm, -1, 1))
                        )
                    )
                )
                rows.append(
                    {
                        "link": link,
                        "joint_positions": positions,
                        "joint_units": {
                            name: "m"
                            if robot.description.joints[name].kind == "prismatic"
                            else "rad"
                            for name in positions
                        },
                        "requested_clearance_m": float(clearance),
                        "reference_surface_clearance_m": expected,
                        "native_collision_clearance_m": best,
                        "distance_error_m": best - expected,
                        "native_contact_count": int(contact_count),
                        "normal_deviation_deg": angle,
                        "passed": abs(best - expected) <= tolerance
                        and (contact_count > 0) == (expected < 0),
                    }
                )
    if (
        file_sha256(configuration) != config_hash
        or file_sha256(directory / "mesh.json") != fingerprint
    ):
        raise ValueError("CAD contact fixture source changed during verification")
    for record in metadata["solids"]:
        _file(directory, record["file"], record["sha256"])
    return {
        "passed": bool(rows) and all(r["passed"] for r in rows),
        "mujoco_version": mujoco.__version__,
        "scope": (
            "specified spheres, surface points and joint configurations; no grasp/force calibration"
        ),
        "coverage": "visual-component" if component_scope else "complete-link",
        "original_collisions_retained": component_scope,
        "configuration_sha256": config_hash,
        "source_mesh_manifest_sha256": fingerprint,
        "collision_bundle_sha256": bundle["manifest_sha256"],
        "sphere_radius_m": radius,
        "tolerance_m": tolerance,
        "maximum_distance_error_m": max(abs(r["distance_error_m"]) for r in rows),
        "rows": rows,
    }
