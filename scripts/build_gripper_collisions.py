"""Rebuild shared gripper collision assets from the registered CAD reference library.

Requires the MuJoCo geometry dependencies and SAPIEN. Use --output to prepare a complete
model directory, or --in-place to replace the repository's generated assets.
"""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from importlib.metadata import version
from pathlib import Path

import numpy as np
from alohamini_mujoco.convex_cover import _prism, certify_rotating_clearance, select_convex_cover
from alohamini_mujoco.frames import rotation_axis
from alohamini_mujoco.functional_geometry import (
    axis_aligned_box,
    certify_material_coverage,
    merge_support_envelopes,
    prepare_shared_convex,
)
from alohamini_mujoco.interop import obj_bytes
from alohamini_mujoco.meshes import Mesh
from alohamini_mujoco.shapes import Primitive
from alohamini_mujoco.solid_geometry import (
    _solid,
    _spatial_union,
)

from alohamini.model import get_robot_model
from alohamini.model.loader import load_robot_model
from alohamini.simulation.description import Description


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reference(root, record):
    path = root / record["mesh"]
    if digest(path) != record["sha256"]:
        raise ValueError("Gripper reference library checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        source = Mesh(data["source_vertices"], data["source_faces"])
        points, triangles = data["candidate_vertices"], data["candidate_faces"]
        vo, fo = data["candidate_vertex_offsets"], data["candidate_face_offsets"]
        if len(vo) != len(fo) or len(vo) < 2 or vo[0] != 0 or fo[0] != 0:
            raise ValueError("Invalid convex reference offsets")
        candidates = [
            Mesh(points[vo[i] : vo[i + 1]], triangles[fo[i] : fo[i + 1]])
            for i in range(len(vo) - 1)
        ]
        contact = {
            "name": record["contact"]["name"],
            "normal": record["contact"]["normal"],
            "tolerance_m": record["contact"]["tolerance_m"],
            "vertices": data["contact_vertices"],
            "faces": data["contact_faces"],
        }
    return source, candidates, contact


def simplify_supports(geometry, sources, contacts, certificates, joint, tolerance_m):
    reference_clearance = certify_rotating_clearance(
        geometry["fixed"],
        geometry["moving"],
        joint.T_parent_joint,
        joint.axis,
        [joint.lower, joint.upper],
    )
    margin = reference_clearance["certified_gap_lower_bound_m"]
    print(json.dumps({"phase": "reference_motion", "clearance_m": margin}), flush=True)
    angles = list(np.linspace(joint.lower, joint.upper, 17))

    def placement(q):
        turn = np.eye(4)
        turn[:3, :3] = rotation_axis(joint.axis, q)
        return joint.T_parent_joint @ turn

    for _ in range(8):
        result, reports = {}, {}
        for kind, other_kind in (("fixed", "moving"), ("moving", "fixed")):
            other = _spatial_union(
                [_solid(p) for p in result.get(other_kind, geometry[other_kind])]
            )
            poses = [placement(q) for q in angles]
            if kind == "moving":
                poses = [np.linalg.inv(pose) for pose in poses]
            obstacles = [other.transform(pose[:3]) for pose in poses]
            contact = contacts[kind]
            guard = certificates[kind]["arithmetic_guard_m"]
            depth = 2 * np.linalg.norm(np.ptp(sources[kind].vertices, axis=0)) + tolerance_m
            reserved = _solid(
                _prism(
                    np.asarray(contact["vertices"]),
                    np.asarray(contact["normal"]),
                    -certificates[kind]["contacts"][0]["structural_setback_m"],
                    depth,
                    inset=2 * guard,
                )
            )
            protected = certificates[kind]["contacts"][0]["piece_indices"]
            pieces, report = merge_support_envelopes(
                geometry[kind],
                protected_indices=protected,
                forbidden_regions=[reserved],
                obstacles=obstacles,
                clearance_m=margin,
                adjacency_m=2 * tolerance_m,
            )
            # Freeze the final file-imported convex shape before motion checks.
            prepared = []
            for index, piece in enumerate(pieces):
                if index < len(protected) or axis_aligned_box(piece) is not None:
                    prepared.append(piece)
                else:
                    prepared.extend(prepare_shared_convex(piece))
            report["material_coverage"] = certify_material_coverage(
                sources[kind], prepared, tolerance_m=tolerance_m
            )
            result[kind], reports[kind] = prepared, report
            print(
                json.dumps({"phase": "support_envelopes", "part": kind, "pieces": len(prepared)}),
                flush=True,
            )
        try:
            proof = certify_rotating_clearance(
                result["fixed"],
                result["moving"],
                joint.T_parent_joint,
                joint.axis,
                [joint.lower, joint.upper],
            )
        except ValueError as exc:
            text = str(exc)
            if "clearance is not positive at " not in text:
                raise
            angles.append(float(text.rsplit("at ", 1)[1].split(" rad")[0]))
            continue
        for report in reports.values():
            report["motion_proposal_angles_rad"] = angles
            report["continuous_motion_certified"] = True
        return result, reports, proof, reference_clearance
    raise ValueError("Support-envelope refinement exceeded its motion-validation budget")


def rebuild(root):
    root = root.resolve()
    reference_path = root / "config/gripper_reference.json"
    reference = json.loads(reference_path.read_text())
    if (
        reference["format"] != "alohamini-gripper-reference-v1"
        or reference["coordinates"] != "URDF link-local metres"
    ):
        raise ValueError("Unsupported gripper reference coordinate contract")
    urdf_path = root / "urdf/alohamini2pro.urdf"
    tree = ET.parse(urdf_path)
    description = Description(load_robot_model(root))
    original_meshes = set()
    certificates, output_meshes, geometry, sources, contacts = {}, {}, {}, {}, {}
    for kind, record in reference["parts"].items():
        source, candidates, contact = load_reference(root, record)
        pieces, certificate = select_convex_cover(
            source,
            candidates,
            tolerance_m=reference["tolerance_m"],
            contacts=[contact],
            prepare_mesh=prepare_shared_convex,
        )
        certificate["mesh_preparation"] = {
            "backend": "sapien",
            "version": version("sapien"),
            "cooked_source_distance_tolerance_m": 0.000002,
        }
        certificates[kind], geometry[kind] = certificate, pieces
        sources[kind], contacts[kind] = source, contact
        print(json.dumps({"part": kind, "reference_pieces": len(pieces)}), flush=True)

    left, right = (description.joints[f"{side}_gripper"] for side in ("left", "right"))
    if not (
        np.array_equal(left.T_parent_joint, right.T_parent_joint)
        and np.array_equal(left.axis, right.axis)
        and (left.lower, left.upper) == (right.lower, right.upper)
    ):
        raise ValueError("Shared jaw envelopes require identical left and right joint definitions")
    geometry, support_reports, proof, reference_clearance = simplify_supports(
        geometry, sources, contacts, certificates, left, reference["tolerance_m"]
    )
    clearance = {side: proof for side in ("left", "right")}
    for kind, record in reference["parts"].items():
        pieces, certificate = geometry[kind], certificates[kind]
        if any(len(piece.vertices) > 128 for piece in pieces):
            # At most 2V-4 triangular faces: this stays within PhysX's 255-face
            # and 255-vertex limits without relying on cooking-time decimation.
            raise ValueError("A selected gripper hull exceeds the shared backend mesh budget")
        encoded = []
        for piece in pieces:
            box = axis_aligned_box(piece)
            if box is not None:
                size, pose = box
                encoded.append(Primitive("box", {"size": size.tolist()}, pose).record())
                continue
            contents = obj_bytes(Mesh(piece.vertices.astype(float) * 1000, piece.faces))
            checksum = hashlib.sha256(contents).hexdigest()
            relative = f"meshes/collision/structural/{checksum}.obj"
            output_meshes[relative] = contents
            encoded.append(
                {
                    "type": "mesh",
                    "mesh": relative,
                    "sha256": checksum,
                    "vertices": len(piece.vertices),
                    "triangles": len(piece.faces),
                    "scale": [0.001] * 3,
                    "msh_sha256": hashlib.sha256(piece.msh_bytes()).hexdigest(),
                }
            )
        for name in record["links"]:
            body = tree.find(f"link[@name='{name}']")
            retained = record["retained_collision_count"]
            for collision in body.findall("collision")[retained:]:
                for mesh in collision.findall("geometry/mesh"):
                    original_meshes.add((urdf_path.parent / mesh.get("filename")).resolve())
                body.remove(collision)
            for index, piece in enumerate(encoded, start=retained):
                if piece["type"] == "box":
                    body.append(
                        Primitive.from_record(piece).collision_node(
                            f"{name}__collision_{index:03d}"
                        )
                    )
                    continue
                collision = ET.SubElement(body, "collision", name=f"{name}__collision_{index:03d}")
                ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
                mesh = ET.SubElement(ET.SubElement(collision, "geometry"), "mesh")
                mesh.set("filename", "../" + piece["mesh"])
                mesh.set("scale", "0.001 0.001 0.001")
        certificates[kind] = {
            "reference_cad_cover": certificate,
            "contacts": certificate["contacts"],
            "mesh_preparation": certificate["mesh_preparation"],
            "support_envelopes": support_reports[kind],
            "output_pieces": len(encoded),
            "pieces": encoded,
        }
        print(json.dumps({"part": kind, "collision_pieces": len(pieces)}), flush=True)

    # Qualify both parts before writing their shared generated files.
    for relative, contents in output_meshes.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
    ET.indent(tree)
    urdf_path.write_text(ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    used = set()
    for path in (root / "urdf").rglob("*.urdf"):
        for mesh in ET.parse(path).findall(".//mesh"):
            filename = mesh.get("filename")
            if not filename.startswith("package://"):
                used.add((path.parent / filename).resolve())
    removed = set()
    for path in original_meshes - used:
        if path.is_relative_to(root / "meshes/collision/structural"):
            removed.add(path.relative_to(root).as_posix())
            path.unlink()
    certificate_path = root / "config/gripper_collision.json"
    certificate_path.write_text(
        json.dumps(
            {
                "format": "alohamini-gripper-collision-v1",
                "coordinates": reference["coordinates"],
                "reference_sha256": digest(reference_path),
                "parts": certificates,
                "articulation_clearance": clearance,
                "reference_articulation_clearance": reference_clearance,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    manifest_path = root / "config/physical_model.json"
    manifest = json.loads(manifest_path.read_text())
    for relative in removed:
        manifest["files"].pop(relative, None)
        manifest["mesh_transport"].pop(relative, None)
    tracked = {
        "urdf/alohamini2pro.urdf",
        "config/gripper_reference.json",
        "config/gripper_collision.json",
        *output_meshes,
        *(r["mesh"] for r in reference["parts"].values()),
    }
    for relative in tracked:
        manifest["files"][relative] = digest(root / relative)
    manifest["files"] = dict(sorted(manifest["files"].items()))
    for part in certificates.values():
        for piece in part["pieces"]:
            if piece["type"] != "mesh":
                continue
            manifest["mesh_transport"][piece["mesh"]] = {
                "coordinate_unit": "mm",
                "scale": [0.001] * 3,
                "vertices": piece["vertices"],
                "triangles": piece["triangles"],
                "msh_sha256": piece["msh_sha256"],
                "geometry_changed": False,
                "qualification": "config/gripper_collision.json",
            }
    manifest["gripper_collision"] = {
        "reference": "config/gripper_reference.json",
        "certificate": "config/gripper_collision.json",
        "source_collision_bundle_sha256": manifest["collision_bundle_sha256"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return certificates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--in-place", action="store_true")
    args = parser.parse_args()
    source = get_robot_model("alohamini2pro").directory
    if args.output is not None:
        shutil.copytree(source, args.output.expanduser())
        source = args.output.expanduser()
    rebuild(source)


if __name__ == "__main__":
    main()
