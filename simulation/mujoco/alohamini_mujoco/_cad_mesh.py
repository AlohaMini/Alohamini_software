"""Bind face mesh boundaries through original CAD topology, never point proximity."""

from __future__ import annotations

import numpy as np

from ._cad_conform import conform_face_mesh, split_index_fans
from .meshes import Mesh

TESSELLATION_METHOD = "occt-topological-boundaries-v2"


def mesh_solid(solid, linear_deflection_m, angular_deflection_rad):
    from OCP.BRep import BRep_Tool
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_VERTEX
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape

    from .collision import signed_volume, topology

    if not BRepCheck_Analyzer(solid).IsValid():
        raise ValueError("CAD solid failed native BRep validation")
    mesher = BRepMesh_IncrementalMesh(
        solid, linear_deflection_m * 1000, False, angular_deflection_rad, False
    )
    if not mesher.IsDone() or mesher.GetStatusFlags():
        raise ValueError(
            f"OpenCascade meshing failed or reported a geometry warning: {mesher.GetStatusFlags()}"
        )
    edges, vertices = TopTools_IndexedMapOfShape(), TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(solid, TopAbs_EDGE, edges)
    TopExp.MapShapes_s(solid, TopAbs_VERTEX, vertices)
    points, triangles, boundaries, offset = [], [], {}, 0
    triangle_faces, face_tolerances = [], []
    faces = TopExp_Explorer(solid, TopAbs_FACE)
    while faces.More():
        face = TopoDS.Face_s(faces.Current())
        if face.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED):
            raise ValueError("CAD face is not an oriented solid boundary")
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face, location)
        if mesh is None:
            raise ValueError("OpenCascade omitted a CAD face")
        coordinates = (
            np.array(
                [
                    mesh.Node(i).Transformed(location.Transformation()).Coord()
                    for i in range(1, mesh.NbNodes() + 1)
                ]
            )
            * 0.001
        )
        indices = (
            np.array(
                [mesh.Triangle(i).Get() for i in range(1, mesh.NbTriangles() + 1)], dtype=np.int64
            )
            - 1
        )
        if face.Orientation() == TopAbs_REVERSED:
            indices = indices[:, ::-1]
        triangle_faces.extend([len(face_tolerances)] * len(indices))
        face_tolerances.append(BRep_Tool.Tolerance_s(face) * 0.001)
        face_edges = TopExp_Explorer(face, TopAbs_EDGE)
        while face_edges.More():
            edge = TopoDS.Edge_s(face_edges.Current())
            polygon = BRep_Tool.PolygonOnTriangulation_s(edge, mesh, location)
            if polygon is None or not polygon.HasParameters():
                raise ValueError("CAD edge lacks a parameterized face-mesh boundary")
            nodes, parameters = polygon.Nodes(), polygon.Parameters()
            ids = (
                np.array(
                    [nodes.Value(i) for i in range(nodes.Lower(), nodes.Upper() + 1)],
                    dtype=np.int64,
                )
                - 1
                + offset
            )
            values = np.array(
                [parameters.Value(i) for i in range(parameters.Lower(), parameters.Upper() + 1)]
            )
            if len(ids) != len(values) or not np.isfinite(values).all():
                raise ValueError("Invalid CAD boundary parameter correspondence")
            order = np.argsort(values, kind="stable")
            boundaries.setdefault(edges.FindIndex(edge), []).append((ids[order], values[order]))
            face_edges.Next()
        points.append(coordinates)
        triangles.append(indices + offset)
        offset += len(coordinates)
        faces.Next()
    points, triangles = np.vstack(points), np.vstack(triangles)
    guard = 64 * np.finfo(float).eps * max(1.0, np.abs(points).max())
    parent = np.arange(len(points))
    targets, budgets = points.copy(), np.full(len(points), np.inf)
    vertex_owner, boundary_nodes = {}, set()

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return int(index)

    def union(first, second):
        first, second = find(first), find(second)
        if first != second:
            parent[second] = first

    def bind(index, target, budget):
        if not np.isfinite(target).all() or not np.isfinite(budget) or budget < 0:
            raise ValueError("Nonfinite CAD boundary definition")
        if np.linalg.norm(points[index] - target) > budget + guard:
            raise ValueError("CAD boundary node exceeds its original local tolerance")
        if np.isfinite(budgets[index]) and np.linalg.norm(targets[index] - target) > guard:
            raise ValueError("Incident CAD boundaries disagree on the same mesh node")
        targets[index] = target
        budgets[index] = min(budgets[index], budget)
        boundary_nodes.add(int(index))

    for edge_id, polygons in boundaries.items():
        edge = TopoDS.Edge_s(edges.FindKey(edge_id))
        tolerance = BRep_Tool.Tolerance_s(edge) * 0.001
        degenerate = BRep_Tool.Degenerated_s(edge)
        curve = BRep_Tool.Curve_s(edge, 0.0, 0.0)
        endpoints = []
        for vertex in (TopExp.FirstVertex_s(edge), TopExp.LastVertex_s(edge)):
            if not vertex.IsNull():
                endpoints.append(
                    (
                        BRep_Tool.Parameter_s(vertex, edge),
                        vertices.FindIndex(vertex),
                        np.array(BRep_Tool.Pnt_s(vertex).Coord()) * 0.001,
                        BRep_Tool.Tolerance_s(vertex) * 0.001,
                    )
                )
        if degenerate and not endpoints:
            raise ValueError("Degenerate CAD edge lacks its declared vertex")
        for ids, values in polygons:
            for index, parameter in zip(ids, values, strict=True):
                matching = [
                    e
                    for e in endpoints
                    if abs(e[0] - parameter)
                    <= 32 * np.finfo(float).eps * max(1.0, abs(e[0]), abs(parameter))
                ]
                if matching or degenerate:
                    _, vertex_id, target, vertex_tolerance = (matching or endpoints)[0]
                    bind(index, target, max(tolerance, vertex_tolerance))
                    if vertex_id in vertex_owner:
                        union(index, vertex_owner[vertex_id])
                    else:
                        vertex_owner[vertex_id] = int(index)
                else:
                    if curve is None:
                        raise ValueError("CAD boundary edge lacks an original 3D curve")
                    bind(index, np.array(curve.Value(float(parameter)).Coord()) * 0.001, tolerance)
        if not degenerate:
            first, parameters = polygons[0]
            for ids, values in polygons[1:]:
                if len(ids) != len(first) or not np.allclose(
                    parameters, values, rtol=32 * np.finfo(float).eps, atol=32 * np.finfo(float).eps
                ):
                    raise ValueError("Adjacent CAD faces disagree on shared-edge parameters")
                for a, b in zip(first, ids, strict=True):
                    union(a, b)
    roots, inverse = np.unique([find(i) for i in range(len(points))], return_inverse=True)
    canonical = targets[roots]
    displacement = np.linalg.norm(points - canonical[inverse], axis=1)
    if (displacement > budgets + guard).any():
        raise ValueError("Shared CAD boundary exceeds a participating local tolerance")
    mapped = inverse[triangles]
    keep = (
        (mapped[:, 0] != mapped[:, 1])
        & (mapped[:, 1] != mapped[:, 2])
        & (mapped[:, 2] != mapped[:, 0])
    )
    if not keep.any():
        raise ValueError("CAD boundary binding left no triangles")
    result = Mesh(canonical, mapped[keep])
    # A collapsed triangle may disappear only if its entire image is already
    # an edge or vertex of a retained triangle. No hole is capped or patched.
    if (~keep).any():
        retained_edges = {
            tuple(sorted((int(a), int(b))))
            for face in result.faces
            for a, b in zip(face, np.roll(face, -1), strict=True)
        }
        retained_vertices = set(result.faces.ravel())
        for face in mapped[~keep]:
            unique = np.unique(face)
            if (len(unique) == 1 and unique[0] not in retained_vertices) or (
                len(unique) == 2 and tuple(unique) not in retained_edges
            ):
                raise ValueError("Collapsed CAD triangle is not covered by the remaining boundary")
    conformity = None
    if topology(result)["boundary_edges"]:
        protected = np.zeros(len(canonical), dtype=bool)
        np.logical_or.at(protected, inverse, np.isfinite(budgets))
        result, conformity = conform_face_mesh(
            result, np.asarray(triangle_faces)[keep], protected, face_tolerances, guard
        )
    fan_separation = None
    shape_topology = topology(result)
    if (
        shape_topology["nonmanifold_edges"]
        and not shape_topology["boundary_edges"]
        and not shape_topology["zero_area_faces"]
    ):
        result, fan_separation = split_index_fans(result)
        shape_topology = topology(result)
    if any(shape_topology.values()) or signed_volume(result) <= 0:
        raise ValueError(
            "CAD tessellation is not a closed outward surface after topological binding: "
            f"{shape_topology}; no missing faces were added"
        )
    used, indexed = np.unique(result.faces, return_inverse=True)
    result = Mesh(result.vertices[used], indexed.reshape(-1, 3))
    finite = np.isfinite(budgets)
    return result, {
        "algorithm": TESSELLATION_METHOD,
        "source_face_mesh_nodes": len(points),
        "source_triangles": len(triangles),
        "source_edges": edges.Extent(),
        "bound_edges": len(boundaries),
        "bound_face_nodes": len(boundary_nodes),
        "maximum_vertex_displacement_m": float(displacement.max()),
        "maximum_original_local_budget_m": float(budgets[finite].max()) if finite.any() else 0.0,
        "rounding_guard_m": guard,
        "collapsed_triangles": int((~keep).sum()),
        "face_interior_conformity": conformity,
        "vertex_fan_separation": fan_separation,
        "geometry_policy": (
            "original BRep vertices and edge curves; no proximity weld or hole filling"
        ),
    }
