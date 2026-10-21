"""Read actual CPU PhysX collision shapes; no renderer or ManiSkill installation mutation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


def solid_distance(points, vertices):
    """Point-to-convex-solid distance, including projection onto faces and edges."""
    hull = ConvexHull(vertices)
    triangles = vertices[hull.simplices]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b - a, c - a
    normal = np.cross(u, v)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    uu, vv, uv = np.sum(u * u, axis=1), np.sum(v * v, axis=1), np.sum(u * v, axis=1)
    determinant = uu * vv - uv * uv
    distances = []
    for point in points:
        if np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]) <= 1e-12:
            distances.append(0.0)
            continue
        signed = np.sum((point - a) * normal, axis=1)
        foot = point - signed[:, None] * normal - a
        fu, fv = np.sum(foot * u, axis=1), np.sum(foot * v, axis=1)
        alpha, beta = (vv * fu - uv * fv) / determinant, (uu * fv - uv * fu) / determinant
        inside = (alpha >= 0) & (beta >= 0) & (alpha + beta <= 1)
        squared = np.where(inside, signed * signed, np.inf)
        for first, last in [(a, b), (b, c), (c, a)]:
            edge = last - first
            t = np.clip(np.sum((point - first) * edge, axis=1) / np.sum(edge * edge, axis=1), 0, 1)
            squared = np.minimum(squared, np.sum((point - first - t[:, None] * edge) ** 2, axis=1))
        distances.append(float(np.sqrt(squared.min())))
    return max(distances, default=0.0)


def check(directory):
    import sapien

    manifest = json.loads((directory / "asset.json").read_text())
    for relative, expected in manifest["files"].items():
        path = (directory / relative).resolve()
        if (
            not path.is_relative_to(directory)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Asset checksum mismatch: {relative}")
    tree = ET.parse(directory / "robot.urdf")
    for link in tree.findall("link"):
        for node in list(link.findall("visual")):
            link.remove(node)
    for mesh in tree.findall(".//mesh"):
        mesh.set("filename", str(directory / mesh.get("filename")))
    scene = sapien.Scene([sapien.physx.PhysxCpuSystem()])
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.load_multiple_collisions_from_file = False
    reports = []
    with tempfile.TemporaryDirectory(prefix="alohamini_sapien_") as temporary:
        path = Path(temporary) / "physics.urdf"
        tree.write(path)
        articulation = loader.load(str(path), srdf_file=str(directory / "robot.srdf"))
        if articulation is None:
            raise RuntimeError("SAPIEN did not load the collision articulation")
        links = {link.name: link for link in articulation.get_links()}
        for name, expected_shapes in manifest["pieces"].items():
            shapes = links[name].collision_shapes
            errors = []
            maximum = 0.0
            if len(shapes) != len(expected_shapes):
                errors.append(
                    f"Expected {len(expected_shapes)} convex pieces, loaded {len(shapes)}"
                )
            for index, (shape, expected) in enumerate(zip(shapes, expected_shapes, strict=False)):
                if not isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
                    errors.append(f"Piece {index} is {type(shape).__name__}")
                    continue
                vertices = np.array(
                    [
                        [float(x) for x in row.split()[1:]]
                        for row in (directory / expected["file"]).read_text().splitlines()
                        if row.startswith("v ")
                    ]
                )
                scale = np.asarray(expected.get("scale", [1.0, 1.0, 1.0]), dtype=float)
                if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError("Invalid expected mesh scale")
                vertices *= scale
                cooked = np.asarray(shape.vertices) * np.asarray(shape.scale)
                transform = shape.local_pose.to_transformation_matrix().astype(float)
                cooked = cooked @ transform[:3, :3].T + transform[:3, 3]
                error = max(solid_distance(vertices, cooked), solid_distance(cooked, vertices))
                maximum = max(maximum, error)
                if error > 1e-6:
                    errors.append(f"Piece {index} cooking changes convex surface by {error:.9g} m")
            reports.append(
                {
                    "link": name,
                    "expected_pieces": len(expected_shapes),
                    "loaded_pieces": len(shapes),
                    "maximum_convex_hausdorff_m": maximum,
                    "passed": not errors,
                    "errors": errors,
                }
            )
    return {
        "passed": all(r["passed"] for r in reports),
        "sapien_version": sapien.__version__,
        "scope": "piece counts and cooked convex surfaces in link frames; not dynamics equivalence",
        "collision_bundle_sha256": manifest["collision_bundle_sha256"],
        "mesh_coordinate_unit": manifest.get("mesh_coordinate_unit", "m"),
        "links": reports,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = check(arguments.asset.resolve())
    with arguments.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "output": str(arguments.output)}), flush=True)
    raise SystemExit(0 if result["passed"] else 1)
