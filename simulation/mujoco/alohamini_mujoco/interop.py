"""Engine-neutral URDF/OBJ transport for the same immutable convex pieces."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .collision import load_bundle
from .meshes import Mesh
from .shapes import Primitive


def obj_bytes(mesh):
    # Seventeen digits preserve every transported float32 coordinate exactly.
    rows = ["v " + " ".join(format(float(x), ".17g") for x in p) for p in mesh.vertices]
    rows += ["f " + " ".join(str(int(i) + 1) for i in f) for f in mesh.faces]
    return ("\n".join(rows) + "\n").encode("ascii")


def export_collision_urdf(description, bundle, output, *, mesh_unit="m"):
    if mesh_unit not in ("m", "mm"):
        raise ValueError("OBJ coordinate unit must be m or mm; URDF poses always use metres")
    coordinate_scale = 1.0 if mesh_unit == "m" else 1000.0
    physical_scale = 1.0 / coordinate_scale
    bundle = Path(bundle).expanduser().resolve()
    pieces, provenance = load_bundle(description, bundle)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "asset"
        (stage / "assets").mkdir(parents=True)
        root = ET.fromstring(ET.tostring(description.tree))
        source_files = {}
        for mesh in root.findall(".//mesh"):
            path = description.mesh_path(mesh)
            contents = path.read_bytes()
            filename = hashlib.sha256(contents).hexdigest() + path.suffix.lower()
            source_files[path] = hashlib.sha256(contents).hexdigest()
            (stage / "assets" / filename).write_bytes(contents)
            mesh.set("filename", "assets/" + filename)
        records = {}
        for link, meshes in pieces.items():
            body = root.find(f"link[@name='{link}']")
            for node in list(body.findall("collision")):
                body.remove(node)
            records[link] = []
            for index, mesh in enumerate(meshes):
                if isinstance(mesh, Primitive):
                    body.append(mesh.collision_node(f"cad_piece_{index}"))
                    records[link].append(mesh.record())
                    continue
                encoded = Mesh(mesh.vertices.astype(float) * coordinate_scale, mesh.faces)
                contents = obj_bytes(encoded)
                fingerprint = hashlib.sha256(contents).hexdigest()
                filename = f"assets/{fingerprint}.obj"
                (stage / filename).write_bytes(contents)
                collision = ET.SubElement(body, "collision", name=f"cad_piece_{index}")
                ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
                ET.SubElement(
                    ET.SubElement(collision, "geometry"),
                    "mesh",
                    filename=filename,
                    scale=" ".join([format(physical_scale, ".17g")] * 3),
                )
                records[link].append(
                    {
                        "type": "mesh",
                        "file": filename,
                        "sha256": fingerprint,
                        "vertices": len(mesh.vertices),
                        "triangles": len(mesh.faces),
                        "coordinate_unit": mesh_unit,
                        "scale": [physical_scale] * 3,
                    }
                )
        ET.indent(root)
        (stage / "robot.urdf").write_text(ET.tostring(root, encoding="unicode") + "\n")
        semantic = description.robot_model.description_path("semantic")
        (stage / "robot.srdf").write_bytes(description.source_bytes[semantic])
        definition = (bundle / "collision.json").read_bytes()
        if hashlib.sha256(definition).hexdigest() != provenance["manifest_sha256"]:
            raise ValueError("Collision definition changed during export")
        (stage / "collision_definition.json").write_bytes(definition)
        for path, expected in source_files.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Robot mesh changed during asset export")
        for path, contents in description.source_bytes.items():
            if path.read_bytes() != contents:
                raise ValueError("Robot description changed during asset export")
        manifest = {
            "schema_version": 1,
            "robot_model": description.robot_model.model_id,
            "asset_revision": description.robot_model.asset_revision,
            "units": {"length": "m", "angle": "rad", "mass": "kg"},
            "frame": (
                "one convex piece per collision element; link-local coordinates, "
                "explicit mesh scale restores metres"
            ),
            "mesh_coordinate_unit": mesh_unit,
            "mesh_transport": (
                "OBJ numeric coordinates use mesh_coordinate_unit; "
                "explicit URDF mesh scales restore physical metre-scale geometry; "
                "transforms and inertia units are unchanged"
            ),
            "source_urdf_sha256": hashlib.sha256(
                description.source_bytes[description.path]
            ).hexdigest(),
            "collision_bundle_sha256": provenance["manifest_sha256"],
            "pieces": records,
            "initial_positions": description.home,
            "source_mesh_sha256": {
                p.relative_to(description.robot_model.directory).as_posix(): s
                for p, s in source_files.items()
            },
            "sapien_loader": {"fix_root_link": True, "load_multiple_collisions_from_file": False},
            "status": "transport artifact; each backend must validate cooked shapes and dynamics",
            "files": {
                p.relative_to(stage).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in stage.rglob("*")
                if p.is_file()
            },
        }
        (stage / "asset.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return output


def check_sapien_asset(directory, python, output):
    directory, output = Path(directory).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            str(Path(python).expanduser().resolve()),
            "-I",
            str(Path(__file__).with_name("_sapien_worker.py")),
            "--asset",
            str(directory),
            "--output",
            str(output),
        ],
        check=False,
    )
    if process.returncode not in (0, 1) or not output.exists():
        raise RuntimeError(f"SAPIEN geometry worker failed: exit {process.returncode}")
    return json.loads(output.read_text())
