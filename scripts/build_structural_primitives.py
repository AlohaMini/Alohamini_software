"""Rebuild shared shoulder-base and T-carriage exterior collision primitives."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from alohamini_mujoco.collision import components, manifold_parts
from alohamini_mujoco.functional_geometry import (
    _outside_box,
    certify_cylinder_box_cover,
    cylinder_distance,
    prepare_shared_convex,
)
from alohamini_mujoco.interop import obj_bytes
from alohamini_mujoco.meshes import Mesh, read_mesh, surface_distances
from alohamini_mujoco.shapes import Primitive
from alohamini_mujoco.solid_geometry import convex_boundary
from scipy.spatial import ConvexHull

from alohamini.model import get_robot_model
from alohamini.model.loader import load_robot_model
from alohamini.simulation.description import Description, origin


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_mesh(root, record):
    path = root / record["source"]
    if digest(path) != record["sha256"]:
        raise ValueError("Structural primitive reference checksum mismatch")
    return read_mesh(path.read_bytes(), path)


def box(bounds):
    lower, upper = bounds
    pose = np.eye(4)
    pose[:3, 3] = (lower + upper) / 2
    return Primitive("box", {"size": (upper - lower).tolist()}, pose)


def arm_base(root, record, description):
    source = source_mesh(root, record)
    parts = components(source)
    bearing = parts[record["bearing_component_index"]]
    if len(bearing.faces) != record["bearing_triangles"]:
        raise ValueError("Unexpected shoulder bearing component")
    joint = description.joints[record["shaft_joint"]]
    if not (
        np.allclose(joint.T_parent_joint[:3, :3], np.eye(3), atol=1e-12)
        and np.allclose(joint.T_parent_joint[:2, 3], 0, atol=1e-12)
        and np.allclose(np.abs(joint.axis), [0, 0, 1], atol=1e-12)
    ):
        raise ValueError("Shoulder shaft must be the declared source-Z axis")
    vertices, ring = source.vertices.astype(float), bearing.vertices.astype(float)
    guard = 8 * float(np.linalg.norm(np.spacing(np.abs(vertices).max(0).astype(np.float32))))
    radius = float(np.linalg.norm(ring[:, :2], axis=1).max()) + guard
    plane = record["foot_partition_plane_m"]
    z_range = [plane - guard, float(ring[:, 2].max()) + guard]
    triangles = source.triangles.astype(float)
    first, last = triangles, np.roll(triangles, -1, axis=1)
    crossing = ((first[..., 2] < plane) & (last[..., 2] > plane)) | (
        (last[..., 2] < plane) & (first[..., 2] > plane)
    )
    first, last = first[crossing], last[crossing]
    intersections = (
        first + (last - first) * ((plane - first[:, 2]) / (last[:, 2] - first[:, 2]))[:, None]
    )
    reference_foot = convex_boundary(np.vstack([vertices[vertices[:, 2] <= plane], intersections]))
    foot = prepare_shared_convex(reference_foot)[0]
    if len(foot.vertices) > 128 or len(foot.faces) > 252:
        raise ValueError("A shoulder foot exceeds the shared convex mesh budget")
    foot_error = max(
        float(surface_distances(a.vertices, b.triangles).max())
        for a, b in ((reference_foot, foot), (foot, reference_foot))
    )
    rear = vertices[(vertices[:, 2] > plane) & (cylinder_distance(vertices, radius, z_range) > 0)]
    rear = np.vstack([rear, parts[record["mount_component_index"]].vertices])
    brace = np.array([rear.min(0) - guard, rear.max(0) + guard])
    # Join the rear bracket to the round housing at the chord spanning its width.
    brace[0, 1] = min(brace[0, 1], float(np.sqrt(radius**2 - np.abs(brace[:, 0]).max() ** 2)))
    proof = certify_cylinder_box_cover(
        source, radius, z_range, [brace], hulls=[reference_foot], tolerance_m=1e-6
    )
    error_bound = proof["maximum_source_outside_distance_bound_m"] + foot_error + guard
    if error_bound > record["tolerance_m"]:
        raise ValueError("Prepared shoulder foot exceeds the declared material tolerance")
    pose = np.eye(4)
    pose[2, 3] = sum(z_range) / 2
    shapes = [
        Primitive("cylinder", {"radius": radius, "length": z_range[1] - z_range[0]}, pose),
        foot,
        box(brace),
    ]
    return shapes, {
        "surface_coverage": proof,
        "prepared_surface_error_bound_m": error_bound,
        "tolerance_m": record["tolerance_m"],
        "source_encoding_guard_m": guard,
    }


def lift_body(root, record, description):
    source = source_mesh(root, record)
    parts = components(source)
    housing = parts[record["housing_component_index"]]
    surfaces = manifold_parts(housing)
    if sorted(len(p) for p in surfaces) != record["housing_face_counts"]:
        raise ValueError("Unexpected major T-carriage housing surfaces")
    boxes, shells, errors = [], [], []
    for faces in surfaces:
        vertices = housing.vertices[np.unique(housing.faces[faces])].astype(float)
        if vertices[:, 0].min() < 0 < vertices[:, 0].max():
            boxes.append(np.array([vertices.min(0), vertices.max(0)]))
        else:
            shell = prepare_shared_convex(convex_boundary(vertices))[0]
            if len(shell.vertices) > 128 or len(shell.faces) > 252:
                raise ValueError("A structural shell exceeds the shared convex mesh budget")
            hull = ConvexHull(shell.vertices)
            outside = np.any(vertices @ hull.equations[:, :3].T + hull.equations[:, 3] > 0, axis=1)
            # Distance to a convex solid is convex: its maximum on every
            # source triangle is bounded by the three vertex distances.
            error = (
                float(surface_distances(vertices[outside], shell.triangles).max())
                if outside.any()
                else 0.0
            )
            guard = 8 * float(
                np.linalg.norm(np.spacing(np.abs(vertices).max(0).astype(np.float32)))
            )
            errors.append(error + guard)
            shells.append(shell)
    if len(boxes) != 1 or len(shells) != 2:
        raise ValueError("Expected one central housing and two lateral housings")
    beam = parts[record["beam_component_index"]].vertices.astype(float)
    boxes.append(np.array([beam.min(0), beam.max(0)]))
    # Enclose only accessory surfaces outside the two main boxes.
    remainder = []
    for part in parts[2:]:
        for triangle in part.triangles.astype(float):
            if any(np.all(triangle >= lo) and np.all(triangle <= hi) for lo, hi in boxes):
                continue
            polygons = [triangle]
            for lo, hi in boxes:
                polygons = [p for poly in polygons for p in _outside_box(poly, lo, hi)]
            remainder.extend(polygons)
    if remainder:
        vertices = np.vstack(remainder)
        boxes.append(np.array([vertices.min(0), vertices.max(0)]))
    shapes = [box(bounds) for bounds in boxes] + shells
    pair = record["internal_contact_pair"]
    semantic = ET.parse(description.semantic_path)
    if not any(
        {node.get("link1"), node.get("link2")} == set(pair)
        for node in semantic.findall("disable_collisions")
    ):
        raise ValueError("The lift housing requires its declared internal guide-contact exclusion")
    joint = description.joints["vertical_move"]
    if pair != [joint.parent, joint.child]:
        raise ValueError("The internal guide-contact pair must match the prismatic joint")
    error_bound = max(errors)
    if error_bound > record["tolerance_m"]:
        raise ValueError("Prepared T-carriage hulls exceed the declared material tolerance")
    return shapes, {
        "algorithm": "major-housing convex envelopes and accessory triangle-box subtraction",
        "source_triangles": len(source.faces),
        "surface_coverage_passed": True,
        "surface_coverage_error_bound_m": error_bound,
        "tolerance_m": record["tolerance_m"],
        "convex_shell_vertices": [len(s.vertices) for s in shells],
        "box_count": len(boxes),
        "internal_contact_pair": pair,
        "lift_range_m": [joint.lower, joint.upper],
        "scope": (
            "external T-carriage envelope; internal guide and drive contact use the joint model"
        ),
    }


def replace_collisions(root, urdf, link, shapes, old_meshes, output_meshes):
    for node in list(link.findall("collision")):
        for mesh in node.findall("geometry/mesh"):
            old_meshes.add((urdf.parent / mesh.get("filename")).resolve())
        link.remove(node)
    encoded = []
    for index, shape in enumerate(shapes):
        name = f"{link.get('name')}__collision_{index}"
        if isinstance(shape, Primitive):
            link.append(shape.collision_node(name))
            encoded.append(shape.record())
            continue
        contents = obj_bytes(Mesh(shape.vertices * 1000, shape.faces))
        checksum = hashlib.sha256(contents).hexdigest()
        relative = f"meshes/collision/structural/{checksum}.obj"
        (root / relative).write_bytes(contents)
        output_meshes[relative] = shape
        node = ET.SubElement(link, "collision", name=name)
        ET.SubElement(node, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(
            ET.SubElement(node, "geometry"),
            "mesh",
            filename="../" + relative,
            scale="0.001 0.001 0.001",
        )
        encoded.append(
            {
                "type": "mesh",
                "mesh": relative,
                "sha256": checksum,
                "scale": [0.001] * 3,
                "T_link_geometry": np.eye(4).tolist(),
                "vertices": len(shape.vertices),
                "triangles": len(shape.faces),
            }
        )
    return encoded


def rebuild(root):
    root = root.resolve()
    path = root / "config/structural_collision.json"
    record = json.loads(path.read_text())
    if record["format"] != "alohamini-structural-primitives-v1":
        raise ValueError("Unsupported structural collision recipe")
    description = Description(load_robot_model(root))
    urdf = root / "urdf/alohamini2pro.urdf"
    tree = ET.parse(urdf)
    old_meshes, output_meshes = set(), {}
    for name, recipe in record["arm_bases"].items():
        visual = description.links[name].find("visual")
        mesh = visual.find("geometry/mesh")
        if not (
            np.array_equal(origin(visual), np.eye(4))
            and mesh.get("scale", "1 1 1") == "1 1 1"
            and description.mesh_path(mesh) == (root / recipe["source"]).resolve()
            and description.joints[recipe["shaft_joint"]].parent == name
        ):
            raise ValueError("Structural source visual must use its declared link coordinates")
        shapes, proof = arm_base(root, recipe, description)
        link = tree.find(f"link[@name='{name}']")
        recipe["pieces"] = replace_collisions(root, urdf, link, shapes, old_meshes, output_meshes)
        recipe["qualification"] = proof
    recipe = record["lift_body"]
    visual = description.links[recipe["link"]].find("visual")
    mesh = visual.find("geometry/mesh")
    if not (
        np.array_equal(origin(visual), np.eye(4))
        and mesh.get("scale", "1 1 1") == "1 1 1"
        and description.mesh_path(mesh) == (root / recipe["source"]).resolve()
    ):
        raise ValueError("The lift source visual must use its declared link coordinates")
    shapes, proof = lift_body(root, recipe, description)
    link = tree.find(f"link[@name='{recipe['link']}']")
    recipe["pieces"] = replace_collisions(root, urdf, link, shapes, old_meshes, output_meshes)
    recipe["qualification"] = proof
    ET.indent(tree)
    urdf.write_text(ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    used = {
        (p.parent / mesh.get("filename")).resolve()
        for p in (root / "urdf").rglob("*.urdf")
        for mesh in ET.parse(p).findall(".//mesh")
        if not mesh.get("filename").startswith("package://")
    }
    manifest_path = root / "config/physical_model.json"
    manifest = json.loads(manifest_path.read_text())
    for mesh in old_meshes - used:
        if mesh.is_relative_to(root / "meshes/collision/structural"):
            relative = mesh.relative_to(root).as_posix()
            mesh.unlink()
            manifest["files"].pop(relative, None)
            manifest["mesh_transport"].pop(relative, None)
    for relative in ("config/structural_collision.json", "urdf/alohamini2pro.urdf"):
        manifest["files"][relative] = digest(root / relative)
    for relative, shape in output_meshes.items():
        manifest["files"][relative] = digest(root / relative)
        manifest["mesh_transport"][relative] = {
            "coordinate_unit": "mm",
            "scale": [0.001] * 3,
            "vertices": len(shape.vertices),
            "triangles": len(shape.faces),
            "msh_sha256": hashlib.sha256(shape.msh_bytes()).hexdigest(),
            "geometry_changed": False,
            "qualification": "config/structural_collision.json",
        }
    manifest["files"] = dict(sorted(manifest["files"].items()))
    manifest["structural_primitives"] = "config/structural_collision.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"arm_base_shapes_each": 3, "lift_body_shapes": len(shapes)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    root = get_robot_model("alohamini2pro").directory
    if args.output is not None:
        target = args.output.expanduser()
        shutil.copytree(root, target)
        root = target
    rebuild(root)


if __name__ == "__main__":
    main()
