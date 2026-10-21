"""Whole-robot collision inventory and lossless organization of legacy convex shells."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .cad import file_sha256
from .collision import (
    collision_sources,
    ordered_convex_parts,
    quantized_convex,
    visual_sources,
)
from .frames import points_in_frame
from .meshes import Mesh, binary_stl


def description_hashes(description):
    return {
        path.relative_to(description.robot_model.directory).as_posix(): hashlib.sha256(
            data
        ).hexdigest()
        for path, data in description.source_bytes.items()
    }


def validate_inventory(description, inventory):
    if (
        inventory.get("source_description_sha256") != description_hashes(description)
        or not isinstance(inventory.get("links"), dict)
        or inventory["links"].keys() != description.links.keys()
    ):
        raise ValueError("Whole-robot collision inventory does not cover the current robot tree")
    for name, record in inventory["links"].items():
        visuals = (
            visual_sources(description, name) if description.links[name].findall("visual") else []
        )
        if (
            record.get("source_collisions") != collision_sources(description, name)
            or record.get("source_visuals") != visuals
        ):
            raise ValueError(f"Whole-robot collision source changed: {name}")


def organize_collisions(description, output, *, cad_bundle=None):
    """Keep all physical links and primitives; recover verified source convex pieces.

    This does not fit geometry to CAD. The complete inventory records retained
    envelopes, provisional contact pads and the STEP reference independently
    of recoverable legacy meshes. It is a diagnostic candidate, not promotion
    of a geometry or contact model to hardware-qualified status.
    """
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_description = description_hashes(description)
    mesh_cache, inventory, replacements, prepared = {}, {}, {}, {}
    for name, link in description.links.items():
        sources = collision_sources(description, name)
        record = {
            "source_visuals": visual_sources(description, name) if link.findall("visual") else [],
            "source_collisions": sources,
            "source_shape_count": len(sources),
            "inertial_source": "URDF" if name in description.inertials else None,
            "representation": "frame_only" if not sources else "retained_source",
            "qualification": "no physical geometry"
            if not sources
            else "CAD/contact review pending",
        }
        inventory[name] = record
        if not sources:
            continue
        if any(source["type"] != "mesh" for source in sources):
            record["retained_reason"] = (
                "native primitives and mixed contact assemblies retained; "
                "inertias are not re-estimated"
            )
            continue
        parts, details = [], []
        try:
            for source in sources:
                mesh_source = source["mesh"]
                path = description.robot_model.directory / mesh_source["path"]
                if mesh_source["sha256"] not in mesh_cache:
                    data = path.read_bytes()
                    if hashlib.sha256(data).hexdigest() != mesh_source["sha256"]:
                        raise ValueError("Collision source changed during organization")
                    mesh_cache[mesh_source["sha256"]] = ordered_convex_parts(binary_stl(data))
                source_parts, partition = mesh_cache[mesh_source["sha256"]]
                scale = np.asarray(mesh_source["scale"])
                if not np.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError("Source collision mesh needs a positive finite scale")
                details.append({"source_index": source["index"], "partition": partition})
                for part in source_parts:
                    parts.append(
                        Mesh(
                            points_in_frame(source["T_link_geometry"], part.vertices * scale),
                            part.faces.copy(),
                        )
                    )
        except ValueError as exc:
            record["retained_reason"] = str(exc)
            record["requires_authored_geometry"] = True
            continue
        record["source_partition"] = details
        record["recovered_convex_count"] = len(parts)
        if len(parts) == len(sources):
            record["retained_reason"] = (
                "source convex pieces already have independent collision elements"
            )
            continue
        record["representation"] = "recovered_source_convex_shells"
        replacements[name] = {
            "frame": "URDF link-local",
            "source_visuals": record["source_visuals"],
            "source_collisions": sources,
            "source_partition": details,
            "pieces": [],
        }
        prepared[name] = parts
    cad_reference = None
    if cad_bundle is not None:
        directory = Path(cad_bundle).expanduser().resolve()
        manifest = directory / "cad.json"
        cad = json.loads(manifest.read_text())
        if cad.get("schema_version") != 1 or cad.get("occt_length_unit_mm") != 1:
            raise ValueError("Unsupported STEP assembly coordinate contract")
        if file_sha256(directory / "source.step") != cad["source_sha256"]:
            raise ValueError("STEP source checksum mismatch")
        cad_reference = {
            "source_sha256": cad["source_sha256"],
            "cad_manifest_sha256": file_sha256(manifest),
            "coordinate_contract": cad["coordinate_contract"],
            "scope": (
                "geometry reference; legacy shell organization applies no STEP dimension updates"
            ),
            "leaf_definitions": sum(
                not v.get("assembly", False) for v in cad["definitions"].values()
            ),
            "occurrences": len(cad["occurrences"]),
            "invalid_definitions": {
                k: v["name"]
                for k, v in cad["definitions"].items()
                if "valid_brep" in v and not v["valid_brep"]
            },
        }
    if not replacements:
        raise ValueError("No merged convex collision sources could be recovered")
    whole_robot = {
        "source_description_sha256": source_description,
        "links": inventory,
        "cad_reference": cad_reference,
        "physical_links": sum(bool(v["source_collisions"]) for v in inventory.values()),
        "coordinate_links": sum(not v["source_collisions"] for v in inventory.values()),
        "source_collision_shapes": sum(v["source_shape_count"] for v in inventory.values()),
        "organized_collision_shapes": sum(
            len(prepared[name]) if name in prepared else record["source_shape_count"]
            for name, record in inventory.items()
        ),
        "replaced_links": list(replacements),
        "contact_policy": "complete source SRDF; no new exclusions or contact masks",
        "inertial_policy": "explicit source URDF inertias; no inference from collision shapes",
        "status": "whole-tree inventory; STEP mapping, geometry and contact qualification pending",
    }
    validate_inventory(description, whole_robot)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        (stage / "hulls").mkdir(parents=True)
        for name, parts in prepared.items():
            for index, part in enumerate(parts):
                mesh, transport = quantized_convex(part)
                data = mesh.msh_bytes()
                fingerprint = hashlib.sha256(data).hexdigest()
                relative = f"hulls/{fingerprint}.msh"
                (stage / relative).write_bytes(data)
                replacements[name]["pieces"].append(
                    {
                        "path": relative,
                        "sha256": fingerprint,
                        "triangles": len(mesh.faces),
                        "source_piece": index,
                        "transport": transport,
                    }
                )
        for path, original in description.source_bytes.items():
            if path.read_bytes() != original:
                raise ValueError("Robot description changed during collision organization")
        validate_inventory(description, whole_robot)
        if cad_reference is not None and (
            file_sha256(manifest) != cad_reference["cad_manifest_sha256"]
            or file_sha256(directory / "source.step") != cad_reference["source_sha256"]
        ):
            raise ValueError("STEP reference changed during collision organization")
        metadata = {
            "schema_version": 1,
            "robot_model": description.robot_model.model_id,
            "units": "m",
            "algorithm": "verified source convex-shell organization",
            "status": "organized legacy collision candidate; STEP/contact qualification pending",
            "transport": "float32 convex boundaries; positive source scale and explicit URDF poses",
            "assembly_inventory": whole_robot,
            "links": replacements,
        }
        (stage / "collision.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return whole_robot
