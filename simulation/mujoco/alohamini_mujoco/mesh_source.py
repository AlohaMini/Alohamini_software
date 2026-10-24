"""Closed visual-component references with source-derived float32 seam bounds."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ._cad_conform import split_index_fans as _split_index_fans
from .cad import file_sha256
from .collision import components, topology, visual_sources, weld_precision_seams
from .meshes import Mesh, binary_stl
from .solid_geometry import _solid


def extract_mesh_source(description, output, *, link, component_index, visual_index=0):
    """Keep every selected triangle; reject holes, orientation damage and ambiguity.

    Coordinates use the scaled visual frame, before T_link_visual is applied.
    A seam may move a vertex only inside its own float32 rounding interval.
    No analytic CAD accuracy, missing thickness or material ownership is inferred.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    sources = visual_sources(description, link)
    if type(visual_index) is not int or not 0 <= visual_index < len(sources):
        raise ValueError("Select an existing visual index")
    source = sources[visual_index]
    path = description.robot_model.directory / source["path"]
    parts = components(binary_stl(path.read_bytes()))
    if type(component_index) is not int or not 0 <= component_index < len(parts):
        raise ValueError("Select an existing vertex-connected visual component")
    selected = parts[component_index]
    stored = selected.vertices.astype(np.float32)
    scale = np.asarray(source["scale"], dtype=float)
    points = stored.astype(float) * scale
    lower = (stored.astype(float) - np.nextafter(stored, np.float32(-np.inf))) / 2 * scale
    upper = (np.nextafter(stored, np.float32(np.inf)).astype(float) - stored) / 2 * scale
    guard = 128 * np.finfo(float).eps * max(1.0, np.abs(points).max())
    bound = float(np.linalg.norm(np.maximum(lower, upper), axis=1).max())
    original = Mesh(points, selected.faces)
    repaired, repair = weld_precision_seams(original, bound + guard)
    distance, mapping = cKDTree(repaired.vertices).query(points)
    displacement = repaired.vertices[mapping] - points
    if (
        repair["collapsed_faces"] != 0
        or not np.array_equal(mapping[original.faces], repaired.faces)
        or (displacement < -lower - guard).any()
        or (displacement > upper + guard).any()
    ):
        raise ValueError(
            "Visual component cannot close within its encoding bounds without losing faces"
        )
    connectivity = None
    if any(topology(repaired).values()):
        repaired, connectivity = _split_index_fans(repaired)
    _solid(repaired)
    metadata = {
        "schema_version": 1,
        "status": "closed mesh reference; analytic CAD error and physical calibration unknown",
        "source_format": "urdf-binary-stl",
        # Retain the source-asset field used by STEP and registration bundles;
        # source_format and the null BRep fields distinguish mesh-only sources.
        "source_cad_sha256": source["sha256"],
        "brep_sha256": None,
        "definition": f"STL:{source['sha256']}:component:{component_index}",
        "name": f"{link} visual component {component_index}",
        "units": "m",
        "coordinates": "scaled URDF visual frame, float64",
        "linear_deflection_m": None,
        "angular_deflection_rad": None,
        "surface_reference_error_m": bound + guard,
        "surface_reference_error_scope": "encoding/seam movement relative to stored STL only",
        "source": {
            "robot_model": description.robot_model.model_id,
            "link": link,
            "visual_index": visual_index,
            "visual": source,
            "component_index": component_index,
            "component_count": len(parts),
            "triangles": len(selected.faces),
        },
        "T_link_part_hint": source["T_link_visual"],
        "seam_repair": {
            **repair,
            "float32_position_error_bound_m": bound,
            "float64_rounding_guard_m": guard,
            "maximum_vertex_displacement_m": float(distance.max()),
            "every_source_triangle_preserved": True,
            "per_coordinate_rounding_intervals_checked": True,
            "hole_filling": False,
        },
        "index_fan_repair": connectivity,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "mesh"
        stage.mkdir()
        target = stage / "solid_0000.npz"
        np.savez_compressed(target, vertices=repaired.vertices, faces=repaired.faces)
        metadata["solids"] = [{"file": target.name, "sha256": file_sha256(target)}]
        (stage / "mesh.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        if (
            file_sha256(path) != source["sha256"]
            or visual_sources(description, link) != sources
            or any(p.read_bytes() != value for p, value in description.source_bytes.items())
        ):
            raise ValueError("Robot visual source changed during component extraction")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return metadata
