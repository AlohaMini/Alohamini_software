"""Bind declared CAD axes to named planes without fitting curved surfaces."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from .cad import file_sha256
from .collision import visual_sources
from .frames import points_in_frame, rigid_transform
from .meshes import binary_stl


def planar_faces(shape):
    """Use zero-based TopExp face indices and native, face-local tolerances."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    result, index = {}, 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        surface = BRepAdaptor_Surface(face)
        if surface.GetType() == GeomAbs_Plane:
            plane = surface.Plane()
            normal = np.array(plane.Axis().Direction().Coord())
            point = np.array(plane.Location().Coord()) * 0.001
            tolerances = [BRep_Tool.Tolerance_s(face) * 0.001]
            for kind, cast in ((TopAbs_EDGE, TopoDS.Edge_s), (TopAbs_VERTEX, TopoDS.Vertex_s)):
                boundary = TopExp_Explorer(face, kind)
                while boundary.More():
                    tolerances.append(BRep_Tool.Tolerance_s(cast(boundary.Current())) * 0.001)
                    boundary.Next()
            properties = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, properties)
            result[index] = {
                "normal_part": normal.tolist(),
                "offset_part_m": float(normal @ point),
                "area_m2": float(properties.Mass()) * 1e-6,
                "local_brep_tolerance_m": float(max(tolerances)),
            }
        index += 1
        explorer.Next()
    return result


def fit_translation(planes, points, errors, rotation, datums):
    """Minimax plane residual with explicit rotation; checks never affect fitting.

    Errors describe source encoding/kernel tolerances, not physical accuracy.
    Every selected vertex contributes a constraint, including patch interiors.
    """
    pose = np.eye(4)
    pose[:3, :3] = rotation
    rotation = rigid_transform(pose)[:3, :3]
    points, errors = np.asarray(points, float), np.asarray(errors, float)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or errors.shape != points.shape
        or not np.isfinite(points).all()
        or not np.isfinite(errors).all()
        or (errors < 0).any()
    ):
        raise ValueError("Expected finite target points and nonnegative coordinate error bounds")
    records, constraints, used_names, used_faces, used_patches = [], [], set(), set(), set()
    for datum in datums:
        name, role, face = datum["name"], datum["role"], datum["cad_face"]
        ids = datum["point_indices"]
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in used_names
            or role not in ("fit", "check")
            or type(face) is not int
            or face not in planes
            or face in used_faces
            or not isinstance(ids, list)
            or len(ids) < 3
            or any(type(i) is not int or not 0 <= i < len(points) for i in ids)
            or len(set(ids)) != len(ids)
            or frozenset(ids) in used_patches
        ):
            raise ValueError(
                "Datums need unique names, planar CAD faces and distinct target patches"
            )
        used_names.add(name)
        used_faces.add(face)
        used_patches.add(frozenset(ids))
        plane = planes[face]
        normal = rotation @ np.asarray(plane["normal_part"])
        selected = points[ids]
        error = errors[ids]
        guard = (
            128
            * np.finfo(float).eps
            * max(1.0, np.abs(selected).max(), abs(plane["offset_part_m"]))
        )
        budget = error @ np.abs(normal) + plane["local_brep_tolerance_m"] + guard
        singular = np.linalg.svd(selected - selected.mean(0), compute_uv=False)
        if singular[1] <= np.linalg.norm(error) + np.sqrt(len(ids)) * (
            plane["local_brep_tolerance_m"] + guard
        ):
            raise ValueError("A datum patch must resolve two independent in-plane directions")
        offsets = selected @ normal - plane["offset_part_m"]
        records.append(
            {
                "name": name,
                "role": role,
                "cad_face": face,
                "cad_plane": plane,
                "normal_link": normal.tolist(),
                "target_points": len(ids),
                "translation_interval_m": [
                    float(max(offsets - budget)),
                    float(min(offsets + budget)),
                ],
                "maximum_encoding_budget_m": float(budget.max()),
            }
        )
        constraints.append((normal, offsets, budget, role))
    fits = [c for c in constraints if c[3] == "fit"]
    normals = np.array([c[0] for c in fits])
    if normals.shape[0] < 3 or np.linalg.matrix_rank(normals) != 3:
        raise ValueError("Fitting planes must constrain all three translation axes")
    if not any(c[3] == "check" for c in constraints):
        raise ValueError("At least one independent check datum is required")
    anchor = np.linalg.lstsq(normals, [c[1].mean() for c in fits], rcond=None)[0]
    scale = float(np.median(np.concatenate([c[2] for c in fits])))
    rows, rhs = [], []
    for normal, offsets, budget, _ in fits:
        coefficients = np.tile(normal * scale, (len(offsets), 1)) / budget[:, None]
        centered = (offsets - normal @ anchor) / budget
        rows.extend(
            (
                np.column_stack((coefficients, -np.ones(len(offsets)))),
                np.column_stack((-coefficients, -np.ones(len(offsets)))),
            )
        )
        rhs.extend((centered, -centered))
    solution = linprog(
        [0, 0, 0, 1],
        A_ub=np.vstack(rows),
        b_ub=np.concatenate(rhs),
        bounds=[(None, None)] * 3 + [(0, None)],
        method="highs",
    )
    if not solution.success:
        raise ValueError("Datum translation solver failed")
    pose[:3, 3] = anchor + solution.x[:3] * scale
    for record, (normal, offsets, budget, _) in zip(records, constraints, strict=True):
        residual = np.abs(offsets - normal @ pose[:3, 3])
        record.update(
            maximum_residual_m=float(residual.max()),
            maximum_budget_excess_m=float(max(0.0, (residual - budget).max())),
            passed=bool((residual <= budget).all()),
        )
    return {
        "passed": all(r["passed"] for r in records),
        "fit_passed": all(r["passed"] for r in records if r["role"] == "fit"),
        "checks_passed": all(r["passed"] for r in records if r["role"] == "check"),
        "T_link_part": pose.tolist(),
        "declared_rotation_fitted": False,
        "fit_normal_condition_number": float(np.linalg.cond(normals)),
        "datums": records,
    }


def place_from_datums(description, directory, binding_path, output):
    """Bind an authored plane correspondence to exact source checksums."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape

    directory, binding_path, output = (
        Path(p).expanduser().resolve() for p in (directory, binding_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    fingerprint = file_sha256(directory / "cad.json")
    binding_hash = file_sha256(binding_path)
    cad, binding = (json.loads(p.read_text()) for p in (directory / "cad.json", binding_path))
    if (
        cad.get("schema_version") != 1
        or cad.get("occt_length_unit_mm") != 1
        or binding.get("schema_version") != 1
        or binding.get("cad_manifest_sha256") != fingerprint
        or file_sha256(directory / "source.step") != cad["source_sha256"]
    ):
        raise ValueError("Datum binding CAD identity or unit mismatch")
    record = cad["definitions"][binding["definition"]]
    brep = (directory / record["file"]).resolve()
    if (
        record.get("brep_units") != "mm"
        or not record.get("valid_brep")
        or not brep.is_relative_to(directory)
        or file_sha256(brep) != cad["files"][record["file"]]
    ):
        raise ValueError("Datum binding BRep identity or validity mismatch")
    sources = visual_sources(description, binding["link"])
    visual_index = binding["visual_index"]
    if type(visual_index) is not int or not 0 <= visual_index < len(sources):
        raise ValueError("Select an existing target visual")
    source = sources[visual_index]
    if source["sha256"] != binding["visual_sha256"]:
        raise ValueError("Datum target visual checksum mismatch")
    target_path = description.robot_model.directory / source["path"]
    target = binary_stl(target_path.read_bytes())
    # Bound rounding to float32 at the stored coordinate magnitude. This does
    # not assume an undocumented earlier CAD export/transform was error-free.
    raw = target.vertices.astype(np.float32)
    spacing = (
        np.maximum(
            np.nextafter(raw, np.float32(np.inf)).astype(float) - raw,
            raw - np.nextafter(raw, np.float32(-np.inf)).astype(float),
        )
        / 2
    )
    pose = rigid_transform(source["T_link_visual"])
    scale = np.asarray(source["scale"])
    points = points_in_frame(pose, target.vertices * scale)
    errors = spacing * np.abs(scale) @ np.abs(pose[:3, :3]).T
    datums, selected_points, selected_errors = [], [], []
    used_triangles = set()
    for datum in binding["datums"]:
        ids = datum["visual_triangles"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(type(i) is not int or not 0 <= i < len(target.faces) for i in ids)
            or len(set(ids)) != len(ids)
            or used_triangles.intersection(ids)
        ):
            raise ValueError("Select disjoint, explicit visual triangle indices")
        used_triangles.update(ids)
        used = np.unique(target.faces[ids])
        offset = sum(len(p) for p in selected_points)
        datums.append({**datum, "point_indices": list(range(offset, offset + len(used)))})
        selected_points.append(points[used])
        selected_errors.append(errors[used])
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(brep), BRep_Builder()):
        raise ValueError("Cannot read datum BRep")
    result = fit_translation(
        planar_faces(shape),
        np.vstack(selected_points),
        np.vstack(selected_errors),
        binding["R_link_part"],
        datums,
    )
    report = {
        "schema_version": 1,
        **result,
        "units": "m",
        "coordinates": "URDF link-local metres; T_link_part maps CAD part into this link",
        "scope": (
            "authored planar correspondences only; not whole-surface equivalence, "
            "rigid ownership, collision admission or physical calibration"
        ),
        "error_model": (
            "local CAD face/boundary kernel tolerances plus stored STL float32 rounding "
            "and float64 arithmetic; earlier export errors unknown"
        ),
        "robot_model": description.robot_model.model_id,
        "cad_manifest_sha256": fingerprint,
        "source_cad_sha256": cad["source_sha256"],
        "brep_sha256": cad["files"][record["file"]],
        "binding_sha256": binding_hash,
        "binding": binding,
        "target_visual": source,
    }
    if (
        file_sha256(directory / "cad.json") != fingerprint
        or file_sha256(directory / "source.step") != cad["source_sha256"]
        or file_sha256(brep) != report["brep_sha256"]
        or file_sha256(binding_path) != binding_hash
        or file_sha256(target_path) != source["sha256"]
        or any(p.read_bytes() != value for p, value in description.source_bytes.items())
    ):
        raise ValueError("Datum source changed during placement")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "placement"
        stage.mkdir()
        (stage / "placement.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return report
