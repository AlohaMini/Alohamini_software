"""Whole-link CAD collision candidates with explicit registration and metric checks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .cad import MESH_REFERENCE_FRAMES, file_sha256
from .collision import (
    collision_sources,
    components,
    quantized_convex,
    validate_convex,
    visual_sources,
)
from .frames import points_in_frame, rigid_transform
from .meshes import Mesh, binary_msh, binary_stl
from .solid_geometry import (
    _solid,
    certify_solid_distance,
    merge_verified_convex,
    refine_convex_union,
)


def _file(directory, relative, checksum):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("CAD collision files need bundle-relative paths")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or file_sha256(path) != checksum:
        raise ValueError("CAD collision source checksum or path mismatch")
    return path


def _source(directory):
    fingerprint = file_sha256(directory / "mesh.json")
    metadata = json.loads((directory / "mesh.json").read_text())
    if (
        metadata.get("schema_version") != 1
        or metadata.get("units") != "m"
        or metadata.get("coordinates") not in MESH_REFERENCE_FRAMES
        or len(metadata.get("solids", [])) != 1
    ):
        raise ValueError("CAD collision requires one explicitly framed closed source solid")
    record = metadata["solids"][0]
    path = _file(directory, record["file"], record["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        mesh = Mesh(archive["vertices"], archive["faces"])
    if mesh.vertices.dtype != np.dtype("float64"):
        raise ValueError("CAD collision source coordinates must retain float64 precision")
    _solid(mesh)
    return mesh, metadata, fingerprint


def _parameters(concavity_m, seed):
    if not math.isfinite(concavity_m) or concavity_m <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Use a positive metric concavity and nonnegative integer seed")
    return dict(
        threshold=float(concavity_m),
        real_metric=True,
        max_convex_hull=-1,
        decimate=False,
        extrude=False,
        seed=seed,
        resolution=2000,
        mcts_iterations=150,
        mcts_nodes=20,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        apx_mode="ch",
        preprocess_resolution=100,
        preprocess_mode="off",
        max_ch_vertex=256,
        extrude_margin=0.01,
    )


def _decomposition(source, metadata, fingerprint, output, parameters, reuse):
    output.mkdir()
    if reuse is not None:
        reuse = Path(reuse).expanduser().resolve()
        cache_hash = file_sha256(reuse / "decomposition.json")
        cache = json.loads((reuse / "decomposition.json").read_text())
        if (
            cache.get("schema_version") != 1
            or cache.get("units") != "m"
            or cache.get("coordinates") != metadata["coordinates"]
            or cache.get("source_manifest_sha256") != fingerprint
            or cache.get("source_mesh_sha256") != metadata["solids"][0]["sha256"]
            or cache.get("coacd_version") != version("coacd")
            or cache.get("parameters") != parameters
            or not cache.get("hulls")
        ):
            raise ValueError("CAD decomposition cache identity, frame or parameters mismatch")
        for record in cache["hulls"]:
            path = _file(reuse, record["file"], record["sha256"])
            target = output / record["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if file_sha256(reuse / "decomposition.json") != cache_hash:
            raise ValueError("CAD decomposition cache changed during import")
    else:
        try:
            import coacd
        except ImportError as exc:
            raise ValueError("Install the simulation geometry extra for CAD collisions") from exc
        coacd.set_log_level("error")
        started = time.monotonic()
        generated = coacd.run_coacd(coacd.Mesh(source.vertices, source.faces), **parameters)
        if not generated:
            raise ValueError("CAD convex decomposition returned no pieces")
        cache = {
            "schema_version": 1,
            "units": "m",
            "coordinates": metadata["coordinates"],
            "source_manifest_sha256": fingerprint,
            "source_mesh_sha256": metadata["solids"][0]["sha256"],
            "coacd_version": version("coacd"),
            "parameters": parameters,
            "elapsed_s": time.monotonic() - started,
            "hulls": [],
        }
        for index, (vertices, faces) in enumerate(generated):
            path = output / f"hull_{index:04d}.npz"
            np.savez_compressed(path, vertices=vertices, faces=faces)
            cache["hulls"].append({"file": path.name, "sha256": file_sha256(path)})
    pieces = []
    for record in cache["hulls"]:
        with np.load(
            _file(output, record["file"], record["sha256"]), allow_pickle=False
        ) as archive:
            mesh = Mesh(archive["vertices"], archive["faces"])
        validate_convex(mesh)
        pieces.append(mesh)
    (output / "decomposition.json").write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    return pieces, file_sha256(output / "decomposition.json")


def _target_mesh(description, report, *, component_only):
    sources = visual_sources(description, report["link"])
    target = report["target"]
    visual_index, component_index = target.get("visual_index"), target.get("component_index")
    if (
        type(visual_index) is not int
        or not 0 <= visual_index < len(sources)
        or type(component_index) is not int
        or target.get("surface_partition") is not None
        or target.get("visual") != sources[visual_index]
    ):
        raise ValueError("CAD registration requires an explicit complete visual component")
    visual = sources[visual_index]
    raw = binary_stl((description.robot_model.directory / visual["path"]).read_bytes())
    parts = components(raw)
    if (
        target.get("component_count") != len(parts)
        or not 0 <= component_index < len(parts)
        or (not component_only and (len(sources) != 1 or len(parts) != 1))
    ):
        raise ValueError("A CAD collision replacement needs verified complete-link registration")
    selected = parts[component_index]
    if target.get("triangles") != len(selected.faces):
        raise ValueError("CAD registration does not cover every target visual component triangle")
    return Mesh(
        points_in_frame(visual["T_link_visual"], selected.vertices * visual["scale"]),
        selected.faces,
    )


def _mesh_identity(mesh):
    return hashlib.sha256(
        mesh.vertices.astype("<f8").tobytes() + mesh.faces.astype("<i8").tobytes()
    ).hexdigest()


def _registration(description, path, fingerprint, metadata, source, *, component_only=False):
    directory = Path(path).expanduser().resolve()
    path = directory / "registration.json"
    digest = file_sha256(path)
    report = json.loads(path.read_text())
    link = report["link"]
    if (
        report.get("passed") is not True
        or report.get("robot_model") != description.robot_model.model_id
        or report.get("mesh_manifest_sha256") != fingerprint
        or report.get("cad_definition") != metadata["definition"]
        or report.get("source_cad_sha256") != metadata["source_cad_sha256"]
        or report.get("coordinates") != "URDF link-local metres"
        or set(report.get("directions", {})) != {"cad_to_visual", "visual_to_cad"}
        or any(r.get("status") != "passed" for r in report["directions"].values())
    ):
        raise ValueError("A CAD collision replacement needs verified complete-link registration")
    target = _target_mesh(description, report, component_only=component_only)
    for name, checksum in report["files"].items():
        _file(directory, name, checksum)
    pose = rigid_transform(report["T_link_part"])
    with np.load(directory / "cad_in_link.npz", allow_pickle=False) as mapped:
        if not np.array_equal(
            mapped["vertices"], points_in_frame(pose, source.vertices)
        ) or not np.array_equal(mapped["faces"], source.faces):
            raise ValueError("Registration transform differs from its verified CAD mesh")
    with np.load(directory / "visual_component.npz", allow_pickle=False) as selected:
        if not np.array_equal(selected["vertices"], target.vertices) or not np.array_equal(
            selected["faces"], target.faces
        ):
            raise ValueError("Registration target differs from the complete current visual")
    return link, pose, report, digest


def validate_cad_qualification(record, link, *, component_only=False):
    """Bind a passed certificate to the actual transported convex-piece bytes."""
    qualification = record["cad_geometry"]
    certificate = qualification["certificate"]
    registration = qualification["registration"]
    coverage = "visual-component" if component_only else "complete-link"
    target = registration["target"]
    visual_index = target.get("visual_index")
    numeric = [
        certificate.get(k, np.nan)
        for k in ("tolerance_m", "kernel_maximum_radius_m", "native_boolean_tolerance_m")
    ]
    if (
        qualification.get("passed") is not True
        or qualification.get("coverage", "complete-link") != coverage
        or certificate.get("passed") is not True
        or qualification.get("collision_piece_sha256")
        != [p.get("sha256") for p in record["pieces"]]
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in numeric)
        or not 0 < numeric[1] <= numeric[0]
        or not 0 <= numeric[2] < numeric[0]
        or set(certificate.get("checks", {}))
        != {"source_inside_expanded_collision", "collision_inside_expanded_source"}
        or any(
            c.get("passed") is not True
            or c.get("residual_volume_m3") != 0
            or c.get("residual_bounds_m") is not None
            for c in certificate["checks"].values()
        )
        or registration.get("passed") is not True
        or registration.get("link") != link
        or type(visual_index) is not int
        or not 0 <= visual_index < len(record["source_visuals"])
        or target["visual"] != record["source_visuals"][visual_index]
        or target.get("surface_partition") is not None
        or (
            not component_only
            and (
                len(record["source_visuals"]) != 1
                or target.get("component_count") != 1
                or target.get("component_index") != 0
            )
        )
        or registration["mesh_manifest_sha256"] != qualification["source_mesh_manifest_sha256"]
    ):
        raise ValueError("CAD collision geometry certificate is failed, stale or incomplete")


def build_cad_collision(
    description,
    mesh_bundle,
    registrations,
    output,
    *,
    tolerance_m,
    concavity_m=0.0005,
    seed=0,
    reuse_decomposition=None,
    refine=False,
    merge_verified=False,
    component_only=False,
):
    """Export either complete-link replacements or isolated component assets.

    Component assets never publish collision.json and cannot replace a link.
    """
    directory, output = (
        Path(mesh_bundle).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    if not registrations or not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Supply complete-link registrations and a positive metric error budget")
    source, metadata, fingerprint = _source(directory)
    bindings = [
        _registration(
            description, path, fingerprint, metadata, source, component_only=component_only
        )
        for path in registrations
    ]
    if len({b[0] for b in bindings}) != len(bindings):
        raise ValueError("CAD collision registrations must select distinct links")
    parameters = _parameters(concavity_m, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        (stage / "geometry").mkdir()
        raw_pieces, decomposition_hash = _decomposition(
            source, metadata, fingerprint, stage / "decomposition", parameters, reuse_decomposition
        )
        records, reports, cached = {}, {}, {}
        for link, pose, registration, registration_hash in bindings:
            key = pose.tobytes()
            if key not in cached:
                mapped = Mesh(points_in_frame(pose, source.vertices), source.faces)
                parts = [
                    quantized_convex(Mesh(points_in_frame(pose, p.vertices), p.faces))[0]
                    for p in raw_pieces
                ]
                refinement, simplification = None, None
                if refine:
                    parts, boundary, refinement = refine_convex_union(
                        mapped, parts, tolerance_m=tolerance_m
                    )
                    certificate = refinement["final_certificate"]
                else:
                    boundary, certificate = certify_solid_distance(
                        mapped, parts, tolerance_m=tolerance_m
                    )
                if merge_verified and certificate["passed"]:
                    parts, boundary, simplification = merge_verified_convex(
                        mapped, parts, tolerance_m=tolerance_m
                    )
                    certificate = simplification["final_certificate"]
                cached[key] = parts, boundary, certificate, refinement, simplification
            parts, boundary, certificate, refinement, simplification = cached[key]
            pieces = []
            for part in parts:
                data = part.msh_bytes()
                digest = hashlib.sha256(data).hexdigest()
                relative = f"hulls/{digest}.msh"
                (stage / relative).write_bytes(data)
                pieces.append(
                    {
                        "type": "mesh",
                        "path": relative,
                        "sha256": digest,
                        "triangles": len(part.faces),
                    }
                )
            qualification = {
                "passed": certificate["passed"],
                "coverage": "visual-component" if component_only else "complete-link",
                "target_geometry_sha256": _mesh_identity(
                    _target_mesh(description, registration, component_only=component_only)
                ),
                "source_mesh_manifest_sha256": fingerprint,
                "registration_sha256": registration_hash,
                "registration": registration,
                "certificate": certificate,
                "refinement": refinement,
                "simplification": simplification,
                "collision_piece_sha256": [p["sha256"] for p in pieces],
            }
            reports[link] = qualification
            records[link] = {
                "frame": "URDF link-local",
                "source_visuals": visual_sources(description, link),
                "source_collisions": collision_sources(description, link),
                "pieces": pieces,
                "cad_geometry": qualification,
            }
            if certificate["passed"]:
                validate_cad_qualification(records[link], link, component_only=component_only)
            np.savez_compressed(
                stage / "geometry" / f"{link}.npz", vertices=boundary.vertices, faces=boundary.faces
            )
        summary = {"passed": all(r["passed"] for r in reports.values()), "links": reports}
        if (
            file_sha256(directory / "mesh.json") != fingerprint
            or any(
                _registration(
                    description, path, fingerprint, metadata, source, component_only=component_only
                )[3]
                != binding[3]
                for path, binding in zip(registrations, bindings, strict=True)
            )
            or any(path.read_bytes() != value for path, value in description.source_bytes.items())
        ):
            raise ValueError("CAD or robot registration changed during collision generation")
        for record in metadata["solids"]:
            _file(directory, record["file"], record["sha256"])
        (stage / "qualification.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n"
        )
        if summary["passed"]:
            manifest = {
                "schema_version": 2,
                "units": "m",
                "robot_model": description.robot_model.model_id,
                "algorithm": "registered closed reference convex collision",
                "status": (
                    "reference material-distance verified candidate; "
                    "contact and dynamics qualification required"
                ),
                "source_mesh_manifest_sha256": fingerprint,
                "source_mesh_manifest": metadata,
                "decomposition_sha256": decomposition_hash,
                "parameters": parameters,
                "geometry_options": {"refine": refine, "merge_verified": merge_verified},
                "coverage": "visual-component" if component_only else "complete-link",
                "links": records,
            }
            (stage / ("component.json" if component_only else "collision.json")).write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n"
            )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return summary


def load_cad_components(description, directory):
    """Load isolated components without making a full-link coverage claim."""
    directory = Path(directory).expanduser().resolve()
    fingerprint = file_sha256(directory / "component.json")
    metadata = json.loads((directory / "component.json").read_text())
    if (
        metadata.get("schema_version") != 2
        or metadata.get("units") != "m"
        or metadata.get("robot_model") != description.robot_model.model_id
        or metadata.get("coverage") != "visual-component"
        or not metadata.get("links")
    ):
        raise ValueError("Use an explicitly scoped registered CAD component bundle")
    meshes = {}
    for link, record in metadata["links"].items():
        if (
            record.get("frame") != "URDF link-local"
            or record.get("source_visuals") != visual_sources(description, link)
            or record.get("source_collisions") != collision_sources(description, link)
        ):
            raise ValueError("CAD component does not match current robot sources")
        validate_cad_qualification(record, link, component_only=True)
        selected = _target_mesh(
            description, record["cad_geometry"]["registration"], component_only=True
        )
        if _mesh_identity(selected) != record["cad_geometry"].get("target_geometry_sha256"):
            raise ValueError("CAD component target selection changed since qualification")
        parts = []
        for piece in record["pieces"]:
            if (
                piece.get("type") != "mesh"
                or {"parameters", "scale", "T_link_geometry"} & piece.keys()
            ):
                raise ValueError("CAD component meshes must already use the link frame")
            path = _file(directory, piece["path"], piece["sha256"])
            mesh = binary_msh(path.read_bytes())
            validate_convex(mesh)
            if len(mesh.faces) != piece.get("triangles"):
                raise ValueError("CAD component triangle count mismatch")
            parts.append(mesh)
        if not parts:
            raise ValueError("CAD component needs at least one qualified convex piece")
        meshes[link] = parts
    if file_sha256(directory / "component.json") != fingerprint:
        raise ValueError("CAD component changed during loading")
    return meshes, {"manifest_sha256": fingerprint, "definition": metadata}
