"""Reproducible CAD-derived convex candidates in named URDF link frames.

No task dimensions, jaw angle correction, hull-count cap, vertex decimation,
or extrusion enters this generator. Concavity is specified in metres.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree

from .description import numbers, origin
from .frames import points_in_frame
from .meshes import Mesh, binary_msh, binary_stl


def topology(mesh):
    faces = mesh.faces
    edges = np.sort(np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    directed = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, direction_counts = np.unique(directed, axis=0, return_counts=True)
    triangles = mesh.triangles.astype(float)
    area_twice = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    return {
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "repeated_directed_edges": int((direction_counts > 1).sum()),
        "zero_area_faces": int((area_twice == 0).sum()),
    }


def components(mesh):
    faces = mesh.faces
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    graph = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(mesh.vertices), len(mesh.vertices)),
    )
    count, labels = connected_components(graph, directed=False)
    result = []
    for index in range(count):
        selected = faces[labels[faces[:, 0]] == index]
        indices, inverse = np.unique(selected, return_inverse=True)
        result.append(Mesh(mesh.vertices[indices].copy(), inverse.reshape(-1, 3)))
    return result


def weld_precision_seams(mesh, tolerance_m):
    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Weld tolerance must be finite and positive")
    points = mesh.vertices.astype(float)
    parent = np.arange(len(points))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for a, b in sorted(cKDTree(points).query_pairs(tolerance_m)):
        parent[find(a)] = find(b)
    _, inverse = np.unique([find(i) for i in range(len(points))], return_inverse=True)
    vertices = np.zeros((int(inverse.max()) + 1, 3))
    np.add.at(vertices, inverse, points)
    vertices /= np.bincount(inverse)[:, None]
    displacement = float(np.linalg.norm(points - vertices[inverse], axis=1).max())
    if displacement > tolerance_m:
        raise ValueError("Weld cluster exceeds the declared metric displacement bound")
    faces = inverse[mesh.faces]
    valid = (
        (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    )
    if not valid.any():
        raise ValueError("Welding left no nondegenerate faces")
    used, indexed = np.unique(faces[valid], return_inverse=True)
    return Mesh(vertices[used], indexed.reshape(-1, 3)), {
        "maximum_vertex_displacement_m": displacement,
        "merged_vertices": len(points) - len(vertices),
        "collapsed_faces": int((~valid).sum()),
    }


def visual_sources(description, link):
    if link not in description.links:
        raise ValueError(f"Unknown link: {link}")
    sources = []
    for node in description.links[link].findall("visual"):
        shape = node.find("geometry/mesh")
        if shape is None:
            raise ValueError("CAD decomposition requires explicit visual meshes")
        path = description.mesh_path(shape)
        scale = numbers(shape.get("scale", "1 1 1"), 3, "visual mesh scale")
        if (scale <= 0).any():
            raise ValueError("Visual mesh scales must be positive")
        data = path.read_bytes()
        sources.append(
            {
                "path": path.relative_to(description.robot_model.directory).as_posix(),
                "sha256": hashlib.sha256(data).hexdigest(),
                "T_link_visual": origin(node).tolist(),
                "scale": scale.tolist(),
            }
        )
    if not sources:
        raise ValueError(f"Link has no CAD visual geometry: {link}")
    return sources


def validate_convex(mesh):
    if (
        not np.isfinite(mesh.vertices).all()
        or len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or mesh.faces.min() < 0
        or mesh.faces.max() >= len(mesh.vertices)
    ):
        raise ValueError("Invalid convex mesh coordinates or indices")
    if any(topology(mesh).values()):
        raise ValueError("Convex collision pieces must be closed oriented triangle manifolds")
    if len(np.unique(mesh.faces)) != len(mesh.vertices) or len(components(mesh)) != 1:
        raise ValueError(
            "Convex collision pieces must have one connected surface and no unused vertices"
        )
    hull = ConvexHull(mesh.vertices)
    reference = mesh.vertices.mean(axis=0)
    triangles = mesh.triangles.astype(float) - reference
    volume = abs(
        float(
            np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
            / 6
        )
    )
    if hull.volume <= 0 or not np.isclose(volume, hull.volume, rtol=2e-5, atol=1e-18):
        raise ValueError("Collision piece is concave or has inconsistent surface winding")


def quantized_convex(mesh):
    """Triangulate the exact float32 convex boundary transported to MuJoCo.

    Merely casting vertices can make a previously planar triangle fan concave.
    Reconstructing its hull matches the engine's physical shape without a
    tolerance relaxation, vertex decimation, hull merging or feature removal.
    """
    validate_convex(mesh)
    vertices = mesh.vertices.astype(np.float32)
    error = float(np.linalg.norm(mesh.vertices - vertices.astype(float), axis=1).max())
    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    triangles = vertices[faces].astype(float)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
    faces[reverse] = faces[reverse, ::-1]
    used, indexed = np.unique(faces, return_inverse=True)
    result = Mesh(vertices[used], indexed.reshape(-1, 3))
    validate_convex(result)
    return result, {
        "maximum_vertex_quantization_m": error,
        "native_vertices": len(mesh.vertices),
        "boundary_vertices": len(used),
        "native_triangles": len(mesh.faces),
    }


def generate_bundle(
    description, links, output, *, tolerance_m=0.0005, seed=0, nonmanifold="reject"
):
    try:
        import coacd
    except ImportError as exc:
        raise ValueError(
            "Install the simulation geometry extra to generate collision candidates"
        ) from exc

    if not math.isfinite(tolerance_m) or tolerance_m <= 0:
        raise ValueError("Concavity tolerance must be a positive distance in metres")
    if type(seed) is not int or seed < 0 or not links or len(set(links)) != len(links):
        raise ValueError("Use unique links and a nonnegative integer seed")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output exists: {output}")
    if nonmanifold not in ("reject", "remesh"):
        raise ValueError("Nonmanifold handling must be reject or remesh")
    # Preflight every source before spending time in the native decomposition.
    prepared = {}
    for link in links:
        sources = visual_sources(description, link)
        prepared[link] = []
        for source in sources:
            data = (description.robot_model.directory / source["path"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"CAD source changed during preparation: {source['path']}")
            raw = binary_stl(data)
            vertices = points_in_frame(
                source["T_link_visual"], raw.vertices * np.asarray(source["scale"])
            )
            # Weld before splitting: float32 seams can disconnect a solid.
            welded, weld = weld_precision_seams(Mesh(vertices, raw.faces), 5e-8)
            parts = components(welded)
            for index, part in enumerate(parts):
                shape_topology = topology(part)
                if any(shape_topology.values()) and nonmanifold == "reject":
                    raise ValueError(
                        f"{link} CAD component {index} is not a closed oriented "
                        f"manifold: {shape_topology}. No geometry was generated; "
                        "repair the source or explicitly select remesh candidates."
                    )
            prepared[link].append((source, parts, weld))
    output.parent.mkdir(parents=True, exist_ok=True)
    options = dict(
        threshold=tolerance_m,
        real_metric=True,
        max_convex_hull=-1,
        decimate=False,
        extrude=False,
        seed=seed,
        resolution=2000,
        mcts_iterations=150,
        mcts_nodes=20,
        mcts_max_depth=3,
        pca=False,
        merge=True,
        apx_mode="ch",
        preprocess_resolution=100,
    )
    manifest = {
        "schema_version": 1,
        "status": "candidate; requires geometry and contact validation",
        "robot_model": description.robot_model.model_id,
        "algorithm": "CoACD",
        "algorithm_version": version("coacd"),
        "parameters": options,
        "nonmanifold_handling": nonmanifold,
        "threshold_interpretation": (
            "CoACD collision-aware concavity target; not a certified surface error bound"
        ),
        "weld_displacement_bound_m": 5e-8,
        "transport": "float32 MSH; convex boundary retriangulated after quantization",
        "runtime": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "units": "m",
        "links": {},
    }
    cache = {}
    coacd.set_log_level("error")
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        for link in links:
            sources = [source for source, _, _ in prepared[link]]
            # File paths are not part of geometric identity: mirrored entries
            # may share work only when their bytes, scale and local pose match.
            geometry_key = hashlib.sha256(
                json.dumps(
                    [{k: v for k, v in source.items() if k != "path"} for source in sources],
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            if geometry_key in cache:
                record = json.loads(json.dumps(cache[geometry_key]))
                paths = {
                    old["path"]: new["path"]
                    for old, new in zip(record["source_visuals"], sources, strict=True)
                }
                for component in record["components"]:
                    component["source_path"] = paths[component["source_path"]]
                for piece in record["pieces"]:
                    piece["source_path"] = paths[piece["source_path"]]
                record["source_visuals"] = sources
                manifest["links"][link] = record
                continue
            record = {
                "frame": "URDF link-local",
                "source_visuals": sources,
                "components": [],
                "pieces": [],
            }
            for source, parts, weld in prepared[link]:
                for component_index, component in enumerate(parts):
                    shape_topology = topology(component)
                    # Native remeshing is opt-in and recorded. An edge check is
                    # not a proof against self intersections; candidates still
                    # require geometric and physical contact validation.
                    mode = "on" if any(shape_topology.values()) else "off"
                    pieces = coacd.run_coacd(
                        coacd.Mesh(component.vertices, component.faces),
                        preprocess_mode=mode,
                        **options,
                    )
                    if not pieces:
                        raise RuntimeError(f"CoACD returned no collision pieces for {link}")
                    record["components"].append(
                        {
                            "source_component": component_index,
                            "source_faces": len(component.faces),
                            "source_path": source["path"],
                            "whole_source_weld": weld,
                            "topology_after_weld": shape_topology,
                            "preprocess_mode": mode,
                            "generated_pieces": len(pieces),
                        }
                    )
                    for piece_index, (vertices, faces) in enumerate(pieces):
                        try:
                            hull, transport = quantized_convex(
                                Mesh(
                                    np.asarray(vertices, dtype=np.float64),
                                    np.asarray(faces, dtype=np.int32),
                                )
                            )
                        except ValueError as exc:
                            raise ValueError(
                                f"{link} CAD component {component_index} hull {piece_index}: {exc}"
                            ) from exc
                        data = hull.msh_bytes()
                        digest = hashlib.sha256(data).hexdigest()
                        name = f"hulls/{digest}.msh"
                        (stage / name).write_bytes(data)
                        record["pieces"].append(
                            {
                                "path": name,
                                "sha256": digest,
                                "triangles": len(hull.faces),
                                "source_path": source["path"],
                                "source_component": component_index,
                                "native_piece": piece_index,
                                "transport": transport,
                            }
                        )
            record["geometry_identity"] = geometry_key
            cache[geometry_key] = record
            manifest["links"][link] = record
        for link, record in manifest["links"].items():
            if record["source_visuals"] != visual_sources(description, link):
                raise ValueError(f"CAD source changed during decomposition: {link}")
        (stage / "collision.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(f"Output appeared during generation: {output}")
        os.rename(stage, output)
    return output


def load_bundle(description, directory):
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "collision.json"
    manifest_bytes = manifest_path.read_bytes()
    metadata = json.loads(manifest_bytes)
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or metadata.get("units") != "m"
        or metadata.get("robot_model") != description.robot_model.model_id
        or not isinstance(metadata.get("links"), dict)
        or not metadata["links"]
    ):
        raise ValueError("Collision bundle has an incompatible coordinate contract")
    result = {}
    for link, record in metadata["links"].items():
        if (
            not isinstance(record, dict)
            or record.get("frame") != "URDF link-local"
            or record.get("source_visuals") != visual_sources(description, link)
        ):
            raise ValueError(f"Collision candidate does not match current CAD frames: {link}")
        if not isinstance(record.get("pieces"), list):
            raise ValueError(f"Invalid collision pieces: {link}")
        pieces = []
        for piece in record.get("pieces", []):
            if (
                not isinstance(piece, dict)
                or not isinstance(piece.get("path"), str)
                or not piece["path"]
                or not isinstance(piece.get("sha256"), str)
                or len(piece["sha256"]) != 64
            ):
                raise ValueError(f"Invalid collision piece definition: {link}")
            path = (directory / piece["path"]).resolve()
            if not path.is_relative_to(directory):
                raise ValueError("Collision mesh path escapes its bundle")
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != piece["sha256"]:
                raise ValueError("Collision mesh checksum mismatch")
            mesh = binary_msh(data)
            validate_convex(mesh)
            if piece.get("triangles") != len(mesh.faces):
                raise ValueError("Collision mesh triangle count mismatch")
            pieces.append(mesh)
        if not pieces:
            raise ValueError(f"Empty collision replacement: {link}")
        result[link] = pieces
    return result, {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "definition": metadata,
    }


def merge_bundles(description, directories, output):
    """Combine validated, disjoint link replacements without regenerating any hull."""
    if len(directories) < 2:
        raise ValueError("Use at least two collision bundles")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    sources, links = [], {}
    for directory in directories:
        directory = Path(directory).expanduser().resolve()
        _, provenance = load_bundle(description, directory)
        definition = provenance["definition"]
        if links.keys() & definition["links"].keys():
            raise ValueError("Collision bundles replace overlapping links")
        links.update(json.loads(json.dumps(definition["links"])))
        sources.append((directory, provenance))
    manifest = {
        "schema_version": 1,
        "units": "m",
        "robot_model": description.robot_model.model_id,
        "status": "composed CAD candidate; requires geometry and contact validation",
        "algorithm": "composition of independently validated bundles",
        "source_manifests": [
            {
                "sha256": p["manifest_sha256"],
                "generator": {k: v for k, v in p["definition"].items() if k != "links"},
                "links": list(p["definition"]["links"]),
            }
            for _, p in sources
        ],
        "links": links,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "candidate"
        (stage / "hulls").mkdir(parents=True)
        for directory, provenance in sources:
            for link, record in provenance["definition"]["links"].items():
                for index, piece in enumerate(record["pieces"]):
                    data = (directory / piece["path"]).read_bytes()
                    if hashlib.sha256(data).hexdigest() != piece["sha256"]:
                        raise ValueError("Collision hull changed during composition")
                    relative = f"hulls/{piece['sha256']}.msh"
                    (stage / relative).write_bytes(data)
                    manifest["links"][link]["pieces"][index]["path"] = relative
            if (
                hashlib.sha256((directory / "collision.json").read_bytes()).hexdigest()
                != provenance["manifest_sha256"]
            ):
                raise ValueError("Collision definition changed during composition")
        (stage / "collision.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        if output.exists():
            raise FileExistsError(output)
        os.rename(stage, output)
    return output
