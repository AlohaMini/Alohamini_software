"""Fit supported native source primitives without guessing their axes or coverage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .collision import collision_sources, visual_sources
from .frames import points_in_frame
from .meshes import binary_stl
from .shapes import Primitive


def fit_primitives(description, links, output):
    """Minimum enclosing box in declared axes, or cylinder about declared shaft.

    The reference is the complete source visual triangle surface. These convex
    enclosures are candidates only; they do not establish physical calibration
    or the acceptability of filling a concavity. No frame is inferred by PCA.
    """
    if not links or len(set(links)) != len(links):
        raise ValueError("Select unique source primitive links")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    records = {}
    for link in links:
        sources = collision_sources(description, link)
        if len(sources) != 1 or sources[0]["type"] not in ("box", "cylinder"):
            raise ValueError(f"Fit requires one existing native box or cylinder: {link}")
        source = sources[0]
        visuals = visual_sources(description, link)
        vertices = []
        for visual in visuals:
            path = description.robot_model.directory / visual["path"]
            raw = binary_stl(path.read_bytes())
            vertices.append(
                points_in_frame(visual["T_link_visual"], raw.vertices * np.asarray(visual["scale"]))
            )
        points = np.vstack(vertices)
        source_pose = np.asarray(source["T_link_geometry"])
        local = points_in_frame(np.linalg.inv(source_pose), points)
        lower, upper = local.min(0), local.max(0)
        pose = source_pose.copy()
        # Round outward for arithmetic only. This is not a physical safety margin.
        arithmetic_bound = 32 * np.finfo(float).eps * max(1.0, np.abs(points).max())
        if source["type"] == "box":
            center = lower + (upper - lower) / 2
            half_size = np.nextafter(np.abs(local - center).max(0) + arithmetic_bound, np.inf)
            pose[:3, 3] += pose[:3, :3] @ center
            parameters = {"size": (2 * half_size).tolist()}
            derivation = "minimum enclosing box in the source collision axes"
            error = float((np.abs(local - center) - half_size).max())
        else:
            center_z = lower[2] + (upper[2] - lower[2]) / 2
            radius = float(
                np.nextafter(np.linalg.norm(local[:, :2], axis=1).max() + arithmetic_bound, np.inf)
            )
            half_length = float(
                np.nextafter(np.abs(local[:, 2] - center_z).max() + arithmetic_bound, np.inf)
            )
            pose[:3, 3] += pose[:3, 2] * center_z
            parameters = {"radius": radius, "length": 2 * half_length}
            derivation = "minimum enclosing cylinder about the source shaft axis; shaft XY retained"
            error = max(
                float(np.linalg.norm(local[:, :2], axis=1).max() - radius),
                float(np.abs(local[:, 2] - center_z).max() - half_length),
            )
        primitive = Primitive(source["type"], parameters, pose)
        records[link] = {
            "frame": "URDF link-local",
            "source_visuals": visuals,
            "source_collisions": sources,
            "pieces": [primitive.record()],
            "fit": {
                "reference": "complete original visual triangle surface; no STEP dimension updates",
                "derivation": derivation,
                "axis_source": "existing collision element pose; no PCA or axis fitting",
                "source_vertices": len(points),
                "source_bounds_in_collision_frame_m": [lower.tolist(), upper.tolist()],
                "arithmetic_outward_bound_m": float(arithmetic_bound),
                "maximum_vertex_constraint_violation_m": error,
                "continuous_coverage": (
                    "all triangle vertices inside a convex primitive imply every point "
                    "of every source triangle is inside"
                ),
                "qualification": (
                    "visual containment only; concavity, physical dimensions "
                    "and contact review pending"
                ),
            },
        }
    for link, record in records.items():
        if (
            visual_sources(description, link) != record["source_visuals"]
            or collision_sources(description, link) != record["source_collisions"]
            or any(p.read_bytes() != data for p, data in description.source_bytes.items())
        ):
            raise ValueError("Robot geometry changed during primitive fitting")
    manifest = {
        "schema_version": 2,
        "units": "m",
        "robot_model": description.robot_model.model_id,
        "algorithm": "analytic enclosure in declared source axes",
        "status": "source visual enclosure candidate; STEP and contact qualification pending",
        "primitive_contract": (
            "full box dimensions and cylinder length; cylinder local Z; SI link frames"
        ),
        "links": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        stage.mkdir()
        (stage / "collision.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return output
