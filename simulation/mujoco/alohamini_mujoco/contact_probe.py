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
from .cad_collision import _file, _source
from .compilation import compile_xml
from .frames import points_in_frame, vector
from .geometry import SurfaceIndex
from .meshes import Mesh


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


def probe_cad_contacts(robot, mesh_bundle, configuration):
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
    if not robot.manifest.get("collision_bundle"):
        raise ValueError("Contact probes require an explicitly qualified CAD collision bundle")
    root = ET.fromstring(robot.xml)
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
    augmented = replace(robot, model=compile_xml(xml, assets=robot.assets), xml=xml)
    model = augmented.model
    probe = model.geom("cad_contact_probe").id
    mocap = model.body_mocapid[model.body("cad_probe_body").id]
    records = robot.manifest["collision_bundle"]["definition"]["links"]
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
        if distance[0] > metadata["linear_deflection_m"]:
            raise ValueError("Declared contact point is not on the supplied CAD mesh")
        triangle = reference.triangles[nearest[0]]
        outward = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if outward @ normal <= 0:
            raise ValueError("Declared probe direction is not outward from the CAD surface")
        ids = {
            model.geom(name).id
            for name, geometry in robot.geometry.items()
            if geometry["kind"] == "collision" and geometry["link"] == link
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
                    witness = np.zeros(6)
                    measured = mujoco.mj_geomDistance(model, data, geom, probe, maximum, witness)
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
        "configuration_sha256": config_hash,
        "source_mesh_manifest_sha256": fingerprint,
        "collision_bundle_sha256": robot.manifest["collision_bundle"]["manifest_sha256"],
        "sphere_radius_m": radius,
        "tolerance_m": tolerance,
        "maximum_distance_error_m": max(abs(r["distance_error_m"]) for r in rows),
        "rows": rows,
    }
