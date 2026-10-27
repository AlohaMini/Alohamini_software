"""Isolated face tessellation for static CAD viewing, never collision generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def tessellate_visual(path, output, linear, angular):
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS, TopoDS_Shape

    shape = TopoDS_Shape()
    if not BRepTools.Read_s(shape, str(path), BRep_Builder()):
        raise ValueError("Cannot read CAD visual source")
    mesher = BRepMesh_IncrementalMesh(shape, linear * 1000, False, angular, False)
    if not mesher.IsDone():
        raise ValueError("CAD visual tessellation did not complete")
    vertices, faces, ranges, offset = [], [], [], 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is None or not mesh.NbTriangles():
            raise ValueError(f"CAD visual omitted source face {len(ranges)}")
        points = (
            np.array(
                [
                    mesh.Node(i).Transformed(location.Transformation()).Coord()
                    for i in range(1, mesh.NbNodes() + 1)
                ]
            )
            * 0.001
        )
        triangles = (
            np.array(
                [mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles() + 1)], dtype=np.int64
            )
            - 1
        )
        if face.Orientation() == TopAbs_REVERSED:
            triangles = triangles[:, ::-1]
        first = sum(len(f) for f in faces)
        ranges.append({"cad_face": len(ranges), "triangles": [first, first + len(triangles)]})
        vertices.append(points)
        faces.append(triangles + offset)
        offset += len(points)
        explorer.Next()
    if not faces:
        raise ValueError("CAD visual has no faces")
    vertices, faces = np.vstack(vertices), np.vstack(faces)
    if not np.isfinite(vertices).all():
        raise ValueError("Nonfinite CAD visual vertices")
    output.mkdir()
    with (output / "mesh.obj").open("w") as stream:
        np.savetxt(stream, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(stream, faces + 1, fmt="f %d %d %d")
    np.savez_compressed(output / "mesh.npz", vertices=vertices, faces=faces)
    metadata = {
        "visual_only": True,
        "method": "unmodified OCCT face triangulations; no boundary welding or solid inference",
        "source_faces": len(ranges),
        "face_triangle_ranges": ranges,
        "vertices": len(vertices),
        "triangles": len(faces),
        "mesher_status_flags": int(mesher.GetStatusFlags()),
    }
    (output / "mesh.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    tessellate_visual(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]))
