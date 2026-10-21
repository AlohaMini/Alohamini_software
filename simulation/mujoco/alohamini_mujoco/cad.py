"""Snapshot exact STEP assembly definitions before mapping them to robot links."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_step(source, output):
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.BRepTools import BRepTools
        from OCP.GProp import GProp_GProps
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
        from OCP.TColStd import TColStd_SequenceOfAsciiString
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDF import TDF_Tool
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopTools import TopTools_IndexedMapOfShape
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
        from OCP.XCAFPrs import XCAFPrs_DocumentExplorer, XCAFPrs_DocumentExplorerFlags_None
    except ImportError as exc:
        raise ValueError("Install the simulation cad extra to inspect STEP assemblies") from exc

    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    fingerprint = file_sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)

    def label_id(label):
        text = TCollection_AsciiString()
        TDF_Tool.Entry_s(label, text)
        return text.ToCString()

    def name(label):
        attribute = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            return attribute.Get().ToExtString()
        return ""

    def bounds(shape):
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(shape, box, False, False)
        return (np.asarray(box.Get()).reshape(2, 3) * 0.001).tolist()

    def count(shape, kind):
        mapping = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, kind, mapping)
        return mapping.Extent()

    def placement(location):
        transform = location.Transformation()
        matrix = np.eye(4)
        matrix[:3] = [[transform.Value(i, j) for j in range(1, 5)] for i in range(1, 4)]
        matrix[:3, 3] *= 0.001
        return matrix

    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    application.InitDocument(document)
    reader = STEPCAFControl_Reader()
    print(f"Reading STEP assembly: {source.name}", flush=True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise ValueError("OpenCascade could not read the STEP file")
    length, angle, solid_angle = (TColStd_SequenceOfAsciiString() for _ in range(3))
    reader.Reader().FileUnits(length, angle, solid_angle)
    # OCCT's length-unit factor is in millimetres. Set after ReadFile, before
    # transfer; the original file may use another declared length unit.
    reader.ChangeReader().SetSystemLengthUnit(1.0)
    if not reader.Transfer(document):
        raise ValueError("OpenCascade could not transfer the STEP assembly")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    print("STEP transfer complete; checking assembly definitions", flush=True)
    definitions, occurrences = {}, []
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "cad"
        (stage / "parts").mkdir(parents=True)
        explorer = XCAFPrs_DocumentExplorer(document, XCAFPrs_DocumentExplorerFlags_None)
        while explorer.More():
            node = explorer.Current()
            definition = label_id(node.RefLabel)
            transform = placement(node.Location)
            occurrence = {
                "id": node.Id.ToCString(),
                "definition": definition,
                "name": name(node.Label),
                "assembly": bool(node.IsAssembly),
                "T_cad_part": transform.tolist(),
            }
            if not node.IsAssembly:
                shape = tool.GetShape_s(node.RefLabel).Located(TopLoc_Location())
                if shape.IsNull():
                    raise ValueError(f"STEP leaf has no geometry: {occurrence['id']}")
                world_shape = explorer.FindShapeFromPathId_s(document, node.Id)
                # Check accumulated occurrence placement instead of assuming
                # the product's local frame equals the assembly's CAD frame.
                expected = shape.Moved(node.Location)
                error = np.max(
                    np.abs(placement(expected.Location()) - placement(world_shape.Location()))
                )
                if (
                    not expected.IsPartner(world_shape)
                    or expected.Orientation() != world_shape.Orientation()
                    or error > 1e-10
                ):
                    raise ValueError(f"Ambiguous STEP occurrence placement: {occurrence['id']}")
                occurrence["bounds_cad_m"] = bounds(world_shape)
                if definition not in definitions:
                    print(f"Checking CAD definition: {name(node.RefLabel)}", flush=True)
                    relative = f"parts/{definition.replace(':', '_')}.brep"
                    if not BRepTools.Write_s(shape, str(stage / relative)):
                        raise RuntimeError(f"Could not preserve BRep definition {definition}")
                    solids = count(shape, TopAbs_SOLID)
                    valid = bool(BRepCheck_Analyzer(shape).IsValid())
                    properties = GProp_GProps()
                    if valid and solids:
                        BRepGProp.VolumeProperties_s(shape, properties, True, False, False)
                    definitions[definition] = {
                        "name": name(node.RefLabel),
                        "file": relative,
                        "brep_units": "mm",
                        "bounds_part_m": bounds(shape),
                        "valid_brep": valid,
                        "solids": solids,
                        "shells": count(shape, TopAbs_SHELL),
                        "faces": count(shape, TopAbs_FACE),
                        "volume_m3": float(properties.Mass()) * 1e-9 if valid and solids else None,
                    }
            else:
                definitions.setdefault(definition, {"name": name(node.RefLabel), "assembly": True})
            occurrences.append(occurrence)
            explorer.Next()
        shutil.copyfile(source, stage / "source.step")
        if file_sha256(stage / "source.step") != fingerprint or file_sha256(source) != fingerprint:
            raise ValueError("STEP source changed during inspection")
        manifest = {
            "schema_version": 1,
            "status": "unmapped CAD assembly; not a robot collision asset",
            "source_name": source.name,
            "source_sha256": fingerprint,
            "ocp_version": version("cadquery-ocp-novtk"),
            "source_length_units": [
                length.Value(i).ToCString() for i in range(1, length.Length() + 1)
            ],
            "occt_length_unit_mm": 1.0,
            "coordinate_contract": (
                "BRep files in native OCCT millimetres; transforms and reported lengths in metres; "
                "T_cad_part maps the part definition frame into the CAD assembly frame"
            ),
            "robot_mapping": None,
            "definitions": definitions,
            "occurrences": occurrences,
            "files": {
                p.relative_to(stage).as_posix(): file_sha256(p)
                for p in stage.rglob("*")
                if p.is_file()
            },
        }
        (stage / "cad.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return manifest


def tessellate_definition(
    directory, definition, output, *, linear_deflection_m=5e-5, angular_deflection_rad=0.1
):
    """Mesh each original CAD solid, preserving its cavity shells and face orientation."""
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepTools import BRepTools
        from OCP.TopAbs import TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Shape
    except ImportError as exc:
        raise ValueError("Install the simulation cad extra to tessellate CAD solids") from exc

    from ._cad_mesh import TESSELLATION_METHOD, mesh_solid
    from .collision import signed_volume

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
    manifest_hash = file_sha256(manifest_path)
    cad = json.loads(manifest_path.read_text())
    record = cad["definitions"][definition]
    if cad.get("schema_version") != 1 or record.get("brep_units") != "mm":
        raise ValueError("Unsupported CAD coordinate contract")
    if not record.get("valid_brep") or not record.get("solids"):
        raise ValueError(
            "Tessellation requires a valid CAD solid; open surfaces cannot be solidified implicitly"
        )
    path = (directory / record["file"]).resolve()
    if not path.is_relative_to(directory) or file_sha256(path) != cad["files"][record["file"]]:
        raise ValueError("BRep checksum mismatch")
    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise ValueError("Cannot read the preserved BRep")
    output.parent.mkdir(parents=True, exist_ok=True)
    solids = TopExp_Explorer(shape, TopAbs_SOLID)
    records = []
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "mesh"
        stage.mkdir()
        while solids.More():
            solid = TopoDS.Solid_s(solids.Current())
            mesh, binding = mesh_solid(solid, linear_deflection_m, angular_deflection_rad)
            relative = f"solid_{len(records):04d}.npz"
            np.savez_compressed(stage / relative, vertices=mesh.vertices, faces=mesh.faces)
            records.append(
                {
                    "file": relative,
                    "sha256": file_sha256(stage / relative),
                    "vertices": len(mesh.vertices),
                    "triangles": len(mesh.faces),
                    "signed_mesh_volume_m3": signed_volume(mesh),
                    "boundary_binding": binding,
                }
            )
            solids.Next()
        if len(records) != record["solids"]:
            raise ValueError("CAD solid count changed during tessellation")
        if (
            file_sha256(manifest_path) != manifest_hash
            or file_sha256(path) != cad["files"][record["file"]]
        ):
            raise ValueError("CAD source changed during tessellation")
        report = {
            "schema_version": 1,
            "status": "part-frame CAD mesh; not mapped to robot links",
            "source_cad_sha256": cad["source_sha256"],
            "cad_manifest_sha256": manifest_hash,
            "brep_sha256": cad["files"][record["file"]],
            "definition": definition,
            "name": record["name"],
            "ocp_version": version("cadquery-ocp-novtk"),
            "units": "m",
            "coordinates": "CAD part definition frame, float64",
            "linear_deflection_m": linear_deflection_m,
            "angular_deflection_rad": angular_deflection_rad,
            "tessellation_method": TESSELLATION_METHOD,
            "exact_brep_volume_m3": record["volume_m3"],
            "solids": records,
        }
        (stage / "mesh.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return report
