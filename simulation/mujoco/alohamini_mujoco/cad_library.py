"""Inventory all CAD leaf definitions with isolated native meshing processes."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from .cad import file_sha256
from .frames import rigid_transform


def _file(directory, relative):
    if not isinstance(relative, str):
        raise ValueError("CAD asset paths must be strings")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("CAD asset path escapes its bundle")
    return path


def _mesh(directory, *, cad, definition, brep_sha256, settings):
    path = directory / "mesh.json"
    manifest = json.loads(path.read_text())
    record = cad["definitions"][definition]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("units") != "m"
        or manifest.get("coordinates") != "CAD part definition frame, float64"
        or manifest.get("definition") != definition
        or manifest.get("source_cad_sha256") != cad["source_sha256"]
        or manifest.get("brep_sha256") != brep_sha256
        or manifest.get("ocp_version") != version("cadquery-ocp-novtk")
        or any(manifest.get(k) != v for k, v in settings.items())
        or not isinstance(manifest.get("solids"), list)
        or len(manifest["solids"]) != record["solids"]
        or not manifest["solids"]
    ):
        raise ValueError(f"CAD mesh identity or settings mismatch: {definition}")
    for solid in manifest["solids"]:
        if (
            file_sha256(_file(directory, solid["file"])) != solid["sha256"]
            or type(solid["triangles"]) is not int
            or solid["triangles"] <= 0
            or not math.isfinite(solid["signed_mesh_volume_m3"])
            or solid["signed_mesh_volume_m3"] <= 0
        ):
            raise ValueError(f"CAD mesh checksum or solid metadata mismatch: {definition}")
    return manifest, file_sha256(path)


def tessellate_assembly(
    directory,
    output,
    *,
    linear_deflection_m=5e-5,
    angular_deflection_rad=0.1,
    reuse_library=None,
):
    """Attempt every leaf once, preserving unmeshable parts and all occurrences.

    Each part runs in a fresh Python process. Checkpointed catalogs are usable
    as a read-only cache for a new output; completed assets are never edited.
    A complete catalog means all definitions were attempted, not that every
    source has valid closed geometry or a known robot rigid-body assignment.
    """
    if (
        not math.isfinite(linear_deflection_m)
        or linear_deflection_m <= 0
        or not math.isfinite(angular_deflection_rad)
        or not 0 < angular_deflection_rad < math.pi
    ):
        raise ValueError("Use positive metric deflection and an angle in (0, pi)")
    directory, output = Path(directory).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = directory / "cad.json"
    fingerprint = file_sha256(manifest_path)
    cad = json.loads(manifest_path.read_text())
    if (
        cad.get("schema_version") != 1
        or cad.get("occt_length_unit_mm") != 1
        or file_sha256(directory / "source.step") != cad["source_sha256"]
    ):
        raise ValueError("Unsupported CAD assembly contract or STEP checksum mismatch")
    leaves = {k: v for k, v in cad["definitions"].items() if not v.get("assembly", False)}
    if not leaves or any(o["definition"] not in cad["definitions"] for o in cad["occurrences"]):
        raise ValueError("Incomplete CAD assembly inventory")
    occurrences = cad["occurrences"]
    if (
        len({o["id"] for o in occurrences}) != len(occurrences)
        or {o["definition"] for o in occurrences if not o["assembly"]} != leaves.keys()
    ):
        raise ValueError("Incomplete or duplicate CAD occurrences")
    for occurrence in occurrences:
        rigid_transform(occurrence["T_cad_part"])
    for definition, record in leaves.items():
        if (
            not re.fullmatch(r"\d+(?::\d+)+", definition)
            or record.get("brep_units") != "mm"
            or file_sha256(_file(directory, record["file"])) != cad["files"][record["file"]]
        ):
            raise ValueError(f"CAD source checksum or unit mismatch: {definition}")
    settings = {
        "linear_deflection_m": linear_deflection_m,
        "angular_deflection_rad": angular_deflection_rad,
    }
    reuse, reused_hash = None, None
    if reuse_library is not None:
        reuse_library = Path(reuse_library).expanduser().resolve()
        reused_hash = file_sha256(reuse_library / "library.json")
        reuse = json.loads((reuse_library / "library.json").read_text())
        if (
            reuse.get("schema_version") != 1
            or reuse.get("source_cad_sha256") != cad["source_sha256"]
        ):
            raise ValueError("CAD cache must refer to the same original STEP source")
    catalog = {
        "schema_version": 1,
        "source_cad_sha256": cad["source_sha256"],
        "cad_manifest_sha256": fingerprint,
        "coordinate_contract": cad["coordinate_contract"],
        "meshing_settings": settings,
        "occurrences": cad["occurrences"],
        "complete": False,
        "fully_meshed": False,
        "status": "source geometry inventory; robot mapping and collision qualification pending",
        "reuse_library_sha256": reused_hash,
        "cache_policy": (
            "exact BRep hash, part frame, original STEP, OCCT version and meshing settings"
        ),
        "parts": {},
    }
    output.mkdir(parents=True)

    def save():
        with tempfile.NamedTemporaryFile(mode="w", dir=output, delete=False) as stream:
            json.dump(catalog, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(stream.name, output / "library.json")

    save()
    for definition, record in leaves.items():
        brep_hash = cad["files"][record["file"]]
        row = {
            "name": record["name"],
            "brep_sha256": brep_hash,
            "valid_brep": record["valid_brep"],
            "solids": record["solids"],
            "shells": record["shells"],
        }
        catalog["parts"][definition] = row
        if not record["valid_brep"] or not record["solids"]:
            row["status"] = "invalid_brep" if not record["valid_brep"] else "surface_only"
            save()
            print(json.dumps({"definition": definition, **row}), file=sys.stderr, flush=True)
            continue
        relative = "parts/" + definition.replace(":", "_")
        target = output / relative
        target.parent.mkdir(exist_ok=True)
        cache = reuse.get("parts", {}).get(definition, {}) if reuse else {}
        reused = False
        if cache.get("status") == "meshed" and cache.get("brep_sha256") == brep_hash:
            source = _file(reuse_library, cache["directory"])
            cached, cached_hash = _mesh(
                source, cad=cad, definition=definition, brep_sha256=brep_hash, settings=settings
            )
            if cached_hash != cache["mesh_manifest_sha256"]:
                raise ValueError("CAD cache manifest checksum mismatch")
            # Copy only the verified manifest and solid files, not arbitrary cache contents.
            target.mkdir()
            for filename in ["mesh.json"] + [s["file"] for s in cached["solids"]]:
                destination = _file(target, filename)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(_file(source, filename), destination)
            reused = True
        else:
            command = [
                sys.executable,
                "-m",
                "alohamini_mujoco",
                "cad-tessellate",
                "--cad-bundle",
                str(directory),
                "--definition",
                definition,
                "--linear-deflection-m",
                str(linear_deflection_m),
                "--angular-deflection-rad",
                str(angular_deflection_rad),
                "--output",
                str(target),
            ]
            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "2")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            process = subprocess.run(command, capture_output=True, text=True, env=env)
            row["exit_code"] = process.returncode
            if process.returncode:
                row["status"] = "meshing_failed" if process.returncode == 1 else "worker_failed"
                row["error"] = (process.stderr or process.stdout)[-16384:].strip()
                save()
                print(json.dumps({"definition": definition, **row}), file=sys.stderr, flush=True)
                continue
        mesh, mesh_hash = _mesh(
            target, cad=cad, definition=definition, brep_sha256=brep_hash, settings=settings
        )
        if reused and mesh_hash != cached_hash:
            raise ValueError("CAD cache changed during copy")
        row.update(
            status="meshed",
            directory=relative,
            mesh_manifest_sha256=mesh_hash,
            triangles=sum(s["triangles"] for s in mesh["solids"]),
            reused=reused,
            origin_cad_manifest_sha256=mesh["cad_manifest_sha256"],
        )
        save()
        print(json.dumps({"definition": definition, **row}), file=sys.stderr, flush=True)
    if (
        file_sha256(manifest_path) != fingerprint
        or file_sha256(directory / "source.step") != cad["source_sha256"]
        or any(
            file_sha256(_file(directory, r["file"])) != cad["files"][r["file"]]
            for r in leaves.values()
        )
        or (reuse and file_sha256(reuse_library / "library.json") != reused_hash)
    ):
        raise ValueError("CAD assembly or cache changed during library generation")
    catalog["complete"] = True
    catalog["fully_meshed"] = all(r["status"] == "meshed" for r in catalog["parts"].values())
    save()
    return catalog
