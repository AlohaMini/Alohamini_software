"""Static, complete CAD reference scenes with explicit assembly coordinates."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import mujoco
import numpy as np

from .cad import file_sha256
from .compilation import load_xml
from .frames import points_in_frame, rigid_transform
from .model import pose_attributes, text


def _rounding_bound(values):
    values = np.asarray(values, dtype=np.float32)
    return (
        np.maximum(
            np.nextafter(values, np.float32(np.inf)).astype(float) - values,
            values - np.nextafter(values, np.float32(-np.inf)).astype(float),
        ).max(0)
        / 2
    )


def _verify_native(model, directory, world_pose, occurrences, parts):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    records = []
    for occurrence in occurrences:
        expected = world_pose @ occurrence["T_cad_part"]
        body = model.body(occurrence["body"]).id
        guard = 256 * np.finfo(float).eps * max(1.0, np.abs(expected).max())
        if not np.allclose(data.xpos[body], expected[:3, 3], atol=guard, rtol=0) or not np.allclose(
            data.xmat[body].reshape(3, 3), expected[:3, :3], atol=guard, rtol=0
        ):
            raise ValueError("Native CAD occurrence transform changed")
        part = parts[occurrence["definition"]]
        with np.load(
            directory / "meshes" / part["asset"] / "mesh.npz", allow_pickle=False
        ) as archive:
            vertices = archive["vertices"]
        target = points_in_frame(expected, vertices)
        geom = model.geom(occurrence["geom"]).id
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        stored = model.mesh_vert[first : first + count]
        rotation = data.geom_xmat[geom].reshape(3, 3)
        actual = stored @ rotation.T + data.geom_xpos[geom]
        # MuJoCo parses OBJ coordinates into float32 before centering, then
        # stores the centered mesh in float32 again. Both errors are counted.
        budget = (
            np.abs(expected[:3, :3]) @ _rounding_bound(vertices)
            + np.abs(rotation) @ _rounding_bound(stored)
            + guard * max(1.0, np.abs(vertices).max())
        )
        error = np.abs(np.array([target.min(0), target.max(0)]) - [actual.min(0), actual.max(0)])
        if (error > budget).any():
            raise ValueError("Native CAD visual bounds exceed coordinate rounding bounds")
        records.append(
            {
                "id": occurrence["id"],
                "maximum_bounds_error_m": float(error.max()),
                "per_axis_rounding_budget_m": budget.tolist(),
            }
        )
    return {
        "occurrence_frames_verified": True,
        "visual_bounds_within_encoding_error": True,
        "scope": "static occurrence transforms and visual bounds; no contact geometry check",
        "occurrences": records,
    }


def export_cad_scene(
    directory,
    output,
    *,
    T_world_cad,
    occurrence=None,
    linear_deflection_m=5e-5,
    angular_deflection_rad=0.1,
):
    """Show every source face, including open shells, with collisions disabled.

    Each selected definition is meshed in a fresh process. Original occurrence
    transforms are preserved; assembly hierarchy is not treated as rigid-body
    ownership. Missing faces abort export instead of producing a partial scene.
    """
    directory, output = (Path(p).expanduser().resolve() for p in (directory, output))
    if output.exists():
        raise FileExistsError(output)
    world_pose = rigid_transform(T_world_cad)
    if (
        not math.isfinite(linear_deflection_m)
        or linear_deflection_m <= 0
        or not math.isfinite(angular_deflection_rad)
        or not 0 < angular_deflection_rad < math.pi
    ):
        raise ValueError("Use positive metric deflection and an angle in (0, pi)")
    fingerprint = file_sha256(directory / "cad.json")
    cad = json.loads((directory / "cad.json").read_text())
    if (
        cad.get("schema_version") != 1
        or cad.get("occt_length_unit_mm") != 1
        or file_sha256(directory / "source.step") != cad["source_sha256"]
    ):
        raise ValueError("CAD scene source identity or unit mismatch")
    occurrences = cad["occurrences"]
    if len({o["id"] for o in occurrences}) != len(occurrences):
        raise ValueError("Duplicate CAD occurrences")
    if occurrence is not None:
        if occurrence not in {o["id"] for o in occurrences}:
            raise ValueError("Select an exact CAD occurrence ID")
        occurrences = [
            o for o in occurrences if o["id"] == occurrence or o["id"].startswith(occurrence + "/")
        ]
    leaves = [o for o in occurrences if not o["assembly"]]
    if not leaves:
        raise ValueError("CAD scene selection contains no leaf parts")
    definitions = {o["definition"]: cad["definitions"][o["definition"]] for o in leaves}
    for o in leaves:
        rigid_transform(o["T_cad_part"])
    sources = {}
    for definition, record in definitions.items():
        path = (directory / record["file"]).resolve()
        if (
            record.get("brep_units") != "mm"
            or not path.is_relative_to(directory)
            or file_sha256(path) != cad["files"][record["file"]]
        ):
            raise ValueError(f"CAD scene BRep identity mismatch: {definition}")
        sources[definition] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "scene"
        (stage / "meshes").mkdir(parents=True)
        (stage / "logs").mkdir()
        parts = {}
        for index, (definition, record) in enumerate(definitions.items()):
            name = f"part_{index}"
            target = stage / "meshes" / name
            environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1"}
            print(f"Meshing CAD visual: {definition} {record['name']}", file=sys.stderr, flush=True)
            with (stage / "logs" / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "alohamini_mujoco._cad_visual",
                        str(sources[definition]),
                        str(target),
                        str(linear_deflection_m),
                        str(angular_deflection_rad),
                    ],
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            if result.returncode:
                details = (stage / "logs" / f"{name}.log").read_text()[-4000:]
                raise ValueError(f"CAD visual worker failed for {definition}: {details}")
            metadata = json.loads((target / "mesh.json").read_text())
            if metadata["source_faces"] != record["faces"]:
                raise ValueError(f"CAD visual source face coverage changed: {definition}")
            parts[definition] = {
                "name": record["name"],
                "asset": name,
                "brep_sha256": cad["files"][record["file"]],
                "valid_brep": record["valid_brep"],
                "source_solids": record["solids"],
                **metadata,
            }
        root = ET.Element("mujoco", model="CAD assembly reference")
        ET.SubElement(root, "compiler", angle="radian", meshdir="meshes")
        ET.SubElement(root, "option", gravity="0 0 0")
        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "global", offwidth="1280", offheight="960")
        ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.6 0.6 0.6")
        asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "texture",
            type="skybox",
            builtin="gradient",
            rgb1="0.95 0.97 0.99",
            rgb2="0.88 0.91 0.94",
            width="128",
            height="128",
        )
        world = ET.SubElement(root, "worldbody")
        cad_body = ET.SubElement(world, "body", name="cad_frame", **pose_attributes(world_pose))
        for part in parts.values():
            ET.SubElement(
                asset, "mesh", name=part["asset"], file=f"{part['asset']}/mesh.obj", inertia="shell"
            )
        bounds, mapped = [], []
        palette = [(0.38, 0.60, 0.75), (0.65, 0.72, 0.78), (0.85, 0.66, 0.34), (0.45, 0.72, 0.61)]
        for index, o in enumerate(leaves):
            part = parts[o["definition"]]
            pose = rigid_transform(o["T_cad_part"])
            body = ET.SubElement(
                cad_body, "body", name=f"occurrence_{index}", **pose_attributes(pose)
            )
            ET.SubElement(
                body,
                "geom",
                name=f"visual_{index}",
                type="mesh",
                mesh=part["asset"],
                mass="0",
                contype="0",
                conaffinity="0",
                group="2",
                rgba=text([*palette[list(parts).index(o["definition"]) % len(palette)], 1]),
            )
            with np.load(
                stage / "meshes" / part["asset"] / "mesh.npz", allow_pickle=False
            ) as archive:
                vertices = points_in_frame(world_pose @ pose, archive["vertices"])
            bounds.append(np.array([vertices.min(0), vertices.max(0)]))
            mapped.append({**o, "body": f"occurrence_{index}", "geom": f"visual_{index}"})
        low, high = np.array(bounds)[:, 0].min(0), np.array(bounds)[:, 1].max(0)
        center, extent = (low + high) / 2, float((high - low).max())
        ET.SubElement(root, "statistic", center=text(center), extent=str(extent * 1.5))
        ET.SubElement(
            world,
            "light",
            pos=text(center + [extent, -extent, extent]),
            dir="-1 1 -1",
            diffuse="0.6 0.6 0.6",
        )
        ET.indent(root)
        (stage / "model.xml").write_text(ET.tostring(root, encoding="unicode") + "\n")
        native = load_xml(stage / "model.xml")
        if (
            native.nq
            or native.nv
            or native.nu
            or native.ngeom != len(leaves)
            or native.geom_contype.any()
            or native.geom_conaffinity.any()
        ):
            raise ValueError("Static CAD reference acquired physical collision or motion")
        manifest = {
            "schema_version": 1,
            "status": "static CAD visual reference; no robot dynamics or collision qualification",
            "visual_only": True,
            "units": "m",
            "source_cad_sha256": cad["source_sha256"],
            "cad_manifest_sha256": fingerprint,
            "ocp_version": version("cadquery-ocp-novtk"),
            "T_world_cad": world_pose.tolist(),
            "selection": occurrence,
            "linear_deflection_m": linear_deflection_m,
            "angular_deflection_rad": angular_deflection_rad,
            "parts": parts,
            "occurrences": mapped,
            "world_vertex_bounds_m": [low.tolist(), high.tolist()],
            "all_source_faces_included": True,
            "native_verification": _verify_native(native, stage, world_pose, mapped, parts),
            "files": {
                p.relative_to(stage).as_posix(): file_sha256(p)
                for p in stage.rglob("*")
                if p.is_file()
            },
        }
        if (
            file_sha256(directory / "cad.json") != fingerprint
            or file_sha256(directory / "source.step") != cad["source_sha256"]
            or any(file_sha256(sources[d]) != parts[d]["brep_sha256"] for d in definitions)
        ):
            raise ValueError("CAD source changed during visual export")
        (stage / "scene.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return manifest
