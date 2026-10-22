"""Verify explicit CAD-part placements against selected authoritative visual parts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from .cad import file_sha256
from .collision import components, manifold_parts, signed_volume, visual_sources
from .frames import points_in_frame, rigid_transform
from .geometry import SurfaceIndex, directed_surface_bounds, measure_surface_distance
from .meshes import Mesh, binary_stl


def register_part(
    description,
    mesh_bundle,
    output,
    *,
    link,
    component_index,
    T_link_part,
    tolerance_m,
    visual_index=0,
    surface_index=None,
    max_samples=1_000_000,
    measure_resolution_m=None,
):
    """Check a declared rigid transform; never fit or replace robot geometry.

    The target is one vertex-connected component of one named URDF visual.
    Checksums bind that selection to the precise source asset. Both complete
    triangle surfaces must pass; insufficient subdivision budget is a failure
    to verify, even when all sampled distances happen to be small.
    """
    directory, output = (
        Path(mesh_bundle).expanduser().resolve(),
        Path(output).expanduser().resolve(),
    )
    if output.exists():
        raise FileExistsError(output)
    if measure_resolution_m is not None and (
        not np.isfinite(measure_resolution_m) or measure_resolution_m <= 0
    ):
        raise ValueError("Use a finite positive surface-distance measurement resolution")
    transform = rigid_transform(T_link_part)
    sources = visual_sources(description, link)
    if type(visual_index) is not int or not 0 <= visual_index < len(sources):
        raise ValueError("Select an existing visual index on the named URDF link")
    source = sources[visual_index]
    target_path = description.robot_model.directory / source["path"]
    raw = binary_stl(target_path.read_bytes())
    parts = components(raw)
    if type(component_index) is not int or not 0 <= component_index < len(parts):
        raise ValueError("Select an existing vertex-connected visual component")
    selected = parts[component_index]
    partition = None
    if surface_index is not None:
        surfaces = manifold_parts(selected)
        if type(surface_index) is not int or not 0 <= surface_index < len(surfaces):
            raise ValueError("Select an existing manifold surface within the visual component")
        face_ids = surfaces[surface_index]
        partition = {
            "method": "face adjacency across edges with exactly two incident faces",
            "surface_index": surface_index,
            "surface_count": len(surfaces),
            "component_triangles": len(selected.faces),
            "source_face_indices_sha256": hashlib.sha256(
                face_ids.astype("<i8").tobytes()
            ).hexdigest(),
            "geometry_changed": False,
        }
        used, faces = np.unique(selected.faces[face_ids], return_inverse=True)
        selected = Mesh(selected.vertices[used], faces.reshape(-1, 3))
    target = Mesh(
        points_in_frame(source["T_link_visual"], selected.vertices * np.asarray(source["scale"])),
        selected.faces,
    )
    manifest_path = directory / "mesh.json"
    fingerprint = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("units") != "m":
        raise ValueError("Use a metre-scale CAD tessellation bundle")
    if manifest.get("coordinates") != "CAD part definition frame, float64":
        raise ValueError("CAD mesh coordinates must be the explicit part definition frame")
    records, vertices, faces = [], [], []
    for solid in manifest["solids"]:
        path = (directory / solid["file"]).resolve()
        if not path.is_relative_to(directory) or file_sha256(path) != solid["sha256"]:
            raise ValueError("CAD mesh checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            mesh = Mesh(archive["vertices"], archive["faces"])
        if (
            mesh.vertices.ndim != 2
            or mesh.vertices.shape[1:] != (3,)
            or mesh.faces.ndim != 2
            or mesh.faces.shape[1:] != (3,)
            or not len(mesh.faces)
            or not np.issubdtype(mesh.faces.dtype, np.integer)
            or mesh.faces.min() < 0
            or mesh.faces.max() >= len(mesh.vertices)
        ):
            raise ValueError("Invalid CAD mesh arrays")
        mapped = points_in_frame(transform, mesh.vertices)
        faces.append(mesh.faces + sum(len(v) for v in vertices))
        vertices.append(mapped)
        records.append((path, solid["sha256"]))
    if not vertices:
        raise ValueError("CAD tessellation contains no solids")
    mapped = Mesh(np.vstack(vertices), np.vstack(faces))
    directions, measurements = {}, {}
    for name, first, second in (
        ("cad_to_visual", mapped, target),
        ("visual_to_cad", target, mapped),
    ):
        print(f"Checking continuous CAD registration: {link}, {name}", file=sys.stderr, flush=True)

        def progress(value, direction=name):
            print(json.dumps({"direction": direction, **value}), file=sys.stderr, flush=True)

        directions[name] = directed_surface_bounds(
            first.triangles,
            SurfaceIndex(second.triangles),
            tolerance_m=tolerance_m,
            max_samples=max_samples,
            progress=progress,
        )
        if measure_resolution_m is not None:
            measurements[name] = measure_surface_distance(
                first.triangles,
                SurfaceIndex(second.triangles),
                resolution_m=measure_resolution_m,
                max_samples=max_samples,
                progress=progress,
            )
    passed = all(value["status"] == "passed" for value in directions.values())
    if (
        file_sha256(manifest_path) != fingerprint
        or file_sha256(target_path) != source["sha256"]
        or any(file_sha256(path) != expected for path, expected in records)
        or any(path.read_bytes() != contents for path, contents in description.source_bytes.items())
    ):
        raise ValueError("CAD or robot source changed during registration")
    report = {
        "schema_version": 1,
        "passed": passed,
        "status": "mesh registration verified" if passed else "mesh registration unverified",
        "scope": (
            "selected CAD part versus selected visual component only; "
            "does not validate other link parts, analytic CAD accuracy, or physical calibration"
        ),
        "robot_model": description.robot_model.model_id,
        "link": link,
        "coordinates": "URDF link-local metres",
        "T_link_part": transform.tolist(),
        "source_cad_sha256": manifest["source_cad_sha256"],
        "mesh_manifest_sha256": fingerprint,
        "cad_definition": manifest["definition"],
        "cad_name": manifest["name"],
        "cad_tessellation": {
            key: manifest[key]
            for key in ("linear_deflection_m", "angular_deflection_rad", "brep_sha256")
        },
        "target": {
            "visual": source,
            "visual_index": visual_index,
            "component_index": component_index,
            "component_count": len(parts),
            "surface_partition": partition,
            "triangles": len(target.faces),
            "signed_volume_m3": signed_volume(target),
        },
        "directions": directions,
        "measurements": measurements,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "registration"
        stage.mkdir()
        for name, mesh in (("cad_in_link.npz", mapped), ("visual_component.npz", target)):
            np.savez_compressed(stage / name, vertices=mesh.vertices, faces=mesh.faces)
        report["files"] = {path.name: file_sha256(path) for path in stage.iterdir()}
        (stage / "registration.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return report
