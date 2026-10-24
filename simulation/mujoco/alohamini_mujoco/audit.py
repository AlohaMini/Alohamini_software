"""Check model equivalence, inertia and actual convex-piece grasp clearances."""

from __future__ import annotations

import hashlib
import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree

from .collision import topology
from .frames import points_in_frame
from .meshes import ray_x_intersections


def contacts(robot, data):
    result = []
    for index, contact in enumerate(data.contact):
        force = np.empty(6)
        mujoco.mj_contactForce(robot.model, data, index, force)
        result.append(
            {
                "geom1": robot.model.geom(contact.geom1).name,
                "geom2": robot.model.geom(contact.geom2).name,
                "body1": robot.model.body(robot.model.geom_bodyid[contact.geom1]).name,
                "body2": robot.model.body(robot.model.geom_bodyid[contact.geom2]).name,
                "distance_m": float(contact.dist),
                "normal_force_n": float(force[0]),
                "position_world_m": contact.pos.tolist(),
            }
        )
    return result


def validate_model(robot, *, samples=16, seed=0):
    if type(samples) is not int or samples < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Use a positive integer sample count and nonnegative integer seed")
    model, description = robot.model, robot.description
    rng = np.random.default_rng(seed)
    active = [
        j
        for j in description.joints.values()
        if j.kind != "fixed" and j.name not in robot.profile.locked_positions
    ]
    max_position = max_rotation = max_com = max_inertia = max_mesh = 0.0
    meshes = []
    initial = robot.data()
    for sample in range(samples):
        values = dict(robot.initial_positions)
        if sample:
            values.update({j.name: rng.uniform(j.lower, j.upper) for j in active})
        data = robot.data(values)
        poses = description.forward(values, robot.profile.T_world_root)
        for name, pose in poses.items():
            body = data.body(name)
            max_position = max(max_position, float(np.max(np.abs(body.xpos - pose[:3, 3]))))
            max_rotation = max(
                max_rotation, float(np.max(np.abs(body.xmat.reshape(3, 3) - pose[:3, :3])))
            )
        for name, (mass, T_link_com, inertia) in description.inertials.items():
            body = model.body(name).id
            T_world_com = poses[name] @ T_link_com
            max_com = max(max_com, float(np.max(np.abs(data.xipos[body] - T_world_com[:3, 3]))))
            rotation = data.ximat[body].reshape(3, 3)
            actual = rotation @ np.diag(model.body_inertia[body]) @ rotation.T
            rotation = T_world_com[:3, :3]
            expected = rotation @ inertia @ rotation.T
            max_inertia = max(max_inertia, float(np.max(np.abs(actual - expected))))
            if not np.isclose(model.body_mass[body], mass, rtol=0, atol=1e-12):
                raise ValueError(f"Imported mass differs from URDF: {name}")
    initial_poses = description.forward(robot.initial_positions, robot.profile.T_world_root)
    for name, geometry in robot.geometry.items():
        if geometry["shape"] != "mesh":
            continue
        mesh = robot.meshes[geometry["path"]]
        geom_id = model.geom(name).id
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        expected = points_in_frame(
            initial_poses[geometry["link"]] @ geometry["T_link_geometry"],
            mesh.vertices * geometry["scale"],
        )
        actual = (
            model.mesh_vert[start : start + count] @ initial.geom_xmat[geom_id].reshape(3, 3).T
            + initial.geom_xpos[geom_id]
        )
        # The compiler recentres/rotates meshes into principal frames. Compare
        # geometry in world coordinates, allowing harmless vertex reordering.
        error = max(
            float(cKDTree(expected).query(actual)[0].max()),
            float(cKDTree(actual).query(expected)[0].max()),
        )
        face_count = int(model.mesh_facenum[mesh_id])
        max_mesh = max(max_mesh, error)
        meshes.append(
            {
                "geom": name,
                "source_triangles": len(mesh.faces),
                "compiled_triangles": face_count,
                "max_vertex_error_m": error,
                "triangles_preserved": face_count == len(mesh.faces),
            }
        )
    robot_bodies = [model.body(n).id for n in description.inertials]
    mass = float(model.body_mass[robot_bodies].sum())
    expected_mass = sum(v[0] for v in description.inertials.values())
    passed = (
        max(max_position, max_rotation, max_com, max_inertia) <= 1e-10
        and max_mesh <= 1e-6
        and all(m["triangles_preserved"] for m in meshes)
        and abs(mass - expected_mass) <= 1e-10
    )
    # Independent rigid-body kinetic energy assembly validates the imported
    # inertias in joint coordinates, beyond just comparing total mass.
    assembled = np.zeros((model.nv, model.nv))
    for name, (body_mass, _, _) in description.inertials.items():
        body_id = model.body(name).id
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBodyCom(model, initial, linear, angular, body_id)
        rotation = initial.ximat[body_id].reshape(3, 3)
        inertia = rotation @ np.diag(model.body_inertia[body_id]) @ rotation.T
        assembled += body_mass * linear.T @ linear + angular.T @ inertia @ angular
    matrix = np.empty((model.nv, model.nv))
    mujoco.mju_sym2dense(matrix, initial.M, model.M_rownnz, model.M_rowadr, model.M_colind)
    robot_dofs = [model.jnt_dofadr[model.joint(j.name).id] for j in active]
    robot_block = np.ix_(robot_dofs, robot_dofs)
    mass_matrix_error = float(np.max(np.abs((matrix - assembled)[robot_block])))
    return {
        "passed": bool(passed and mass_matrix_error <= 1e-10),
        "scope": "URDF frame, mesh transport and inertial equivalence; not contact certification",
        "sample_count": samples,
        "seed": seed,
        "robot_links": len(description.links),
        "active_joints": len(active),
        "scene_joints": model.njnt - len(active),
        "visual_geoms": sum(g["kind"] == "visual" for g in robot.geometry.values()),
        "collision_geoms": sum(g["kind"] == "collision" for g in robot.geometry.values()),
        "source_mass_kg": expected_mass,
        "imported_mass_kg": mass,
        "whole_scene_mass_kg": float(model.body_mass.sum()),
        "max_fk_position_error_m": max_position,
        "max_fk_rotation_matrix_error": max_rotation,
        "max_com_position_error_m": max_com,
        "max_inertia_error_kg_m2": max_inertia,
        "mass_matrix_max_absolute_error": mass_matrix_error,
        "minimum_mass_matrix_eigenvalue": float(np.linalg.eigvalsh(matrix)[0]),
        "initial_contacts": contacts(robot, initial),
        "meshes": meshes,
    }


def geometry_triangles(robot, name, *, physical=False):
    geometry = robot.geometry[name]
    if geometry["shape"] == "mesh":
        mesh = robot.meshes[geometry["path"]]
        vertices = mesh.vertices.astype(float) * geometry["scale"]
        # MuJoCo's rigid mesh collision is convex, irrespective of the visual
        # triangles. Preserve each URDF piece as a separate convex shape.
        faces = ConvexHull(vertices).simplices if physical else mesh.faces
    elif geometry["shape"] == "box":
        vertices = np.asarray(list(itertools.product((-0.5, 0.5), repeat=3))) * geometry["size"]
        faces = ConvexHull(vertices).simplices
    else:
        raise ValueError("Cross-section audit requires mesh or box pieces")
    return points_in_frame(geometry["T_link_geometry"], vertices)[faces]


def convexification_report(robot):
    """Expose where an imported mesh is changed by rigid convex collision.

    Volume ratios apply only to closed consistently oriented input surfaces.
    They do not measure a surface error or certify contact accuracy.
    """
    result, cache = [], {}
    for name, geometry in robot.geometry.items():
        if geometry["kind"] != "collision" or geometry["shape"] != "mesh":
            continue
        key = (geometry["path"], tuple(geometry["scale"]))
        if key not in cache:
            mesh = robot.meshes[geometry["path"]]
            vertices = mesh.vertices.astype(float) * geometry["scale"]
            hull = ConvexHull(vertices)
            triangles = vertices[mesh.faces] - vertices.mean(axis=0)
            volume = abs(
                float(
                    np.einsum(
                        "ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])
                    ).sum()
                    / 6
                )
            )
            shape_topology = topology(mesh)
            closed = not any(shape_topology.values())
            cache[key] = {
                "source_topology": shape_topology,
                "source_volume_m3": volume if closed else None,
                "physical_convex_hull_volume_m3": float(hull.volume),
                "convex_to_source_volume_ratio": float(hull.volume / volume)
                if closed and volume > 0
                else None,
                "source_vertices": len(vertices),
                "convex_hull_vertices": len(hull.vertices),
            }
        result.append(
            {"geom": name, "link": geometry["link"], "source": geometry["path"], **cache[key]}
        )
    return {
        "interpretation": (
            "each mesh geom collides as one convex hull; "
            "volumes are diagnostics, not surface-error bounds"
        ),
        "meshes": result,
    }


def gripper_sections(robot, *, side="right", positions=None, depths_m=None, y_m=0.0):
    if side not in ("left", "right"):
        raise ValueError("Expected left or right gripper")
    joint = robot.description.joints[f"{side}_gripper"]
    if positions is None:
        positions = np.linspace(joint.lower, joint.upper, 9)
    if depths_m is None:
        # These are explicitly reported probe locations, not TCP corrections.
        depths_m = np.linspace(0.05, 0.1, 11)
    if not np.isfinite([*positions, *depths_m, y_m]).all():
        raise ValueError("Cross-section probe coordinates must be finite")
    sections = []
    groups = {}
    for kind in ("visual", "collision"):
        for link in (f"{side}_Fixed_Jaw", f"{side}_Moving_Jaw"):
            pieces = [
                geometry_triangles(robot, name, physical=kind == "collision")
                for name, g in robot.geometry.items()
                if g["link"] == link and g["kind"] == kind
            ]
            groups[kind, link] = np.concatenate(pieces)
    fixed = {kind: groups[kind, f"{side}_Fixed_Jaw"] for kind in ("visual", "collision")}
    for q in positions:
        joint.check_position(float(q))
        T_fixed_moving = joint.T_parent_joint @ joint.motion(q)
        moving = {
            kind: points_in_frame(
                T_fixed_moving, groups[kind, f"{side}_Moving_Jaw"].reshape(-1, 3)
            ).reshape(-1, 3, 3)
            for kind in ("visual", "collision")
        }
        for z in depths_m:
            row = {"q_rad": float(q), "y_fixed_jaw_m": float(y_m), "z_fixed_jaw_m": float(z)}
            for kind in ("visual", "collision"):
                a = ray_x_intersections(fixed[kind], y_m, z)
                b = ray_x_intersections(moving[kind], y_m, z)
                row[f"{kind}_fixed_contact_x_m"] = float(a[-1]) if len(a) else None
                row[f"{kind}_moving_contact_x_m"] = float(b[0]) if len(b) else None
                row[f"{kind}_gap_m"] = float(b[0] - a[-1]) if len(a) and len(b) else None
            for finger in ("fixed", "moving"):
                visual_x = row[f"visual_{finger}_contact_x_m"]
                collision_x = row[f"collision_{finger}_contact_x_m"]
                row[f"{finger}_contact_error_m"] = (
                    collision_x - visual_x
                    if visual_x is not None and collision_x is not None
                    else None
                )
            a, b = row["visual_gap_m"], row["collision_gap_m"]
            row["collision_minus_visual_gap_m"] = b - a if a is not None and b is not None else None
            sections.append(row)
    differences = [
        s["collision_minus_visual_gap_m"]
        for s in sections
        if s["collision_minus_visual_gap_m"] is not None
    ]
    surface_errors = [
        s[f"{finger}_contact_error_m"]
        for s in sections
        for finger in ("fixed", "moving")
        if s[f"{finger}_contact_error_m"] is not None
    ]
    surface_counts = {}
    for finger in ("fixed", "moving"):
        surface_counts[finger] = {
            "paired": sum(
                s[f"visual_{finger}_contact_x_m"] is not None
                and s[f"collision_{finger}_contact_x_m"] is not None
                for s in sections
            ),
            "visual_only": sum(
                s[f"visual_{finger}_contact_x_m"] is not None
                and s[f"collision_{finger}_contact_x_m"] is None
                for s in sections
            ),
            "collision_only": sum(
                s[f"visual_{finger}_contact_x_m"] is None
                and s[f"collision_{finger}_contact_x_m"] is not None
                for s in sections
            ),
        }
    return {
        "side": side,
        "frame": f"{side}_Fixed_Jaw",
        "ray_direction": "+X",
        "interpretation": (
            "collision gap minus visual gap: positive widens the opening; negative narrows it"
        ),
        "comparison": "actual visual triangles versus union of individual physical convex pieces",
        "probe_counts": {
            "total": len(sections),
            "paired_gaps": len(differences),
            "visual_gap_without_collision_gap": sum(
                s["visual_gap_m"] is not None and s["collision_gap_m"] is None for s in sections
            ),
            "collision_gap_without_visual_gap": sum(
                s["collision_gap_m"] is not None and s["visual_gap_m"] is None for s in sections
            ),
            "neither_gap": sum(
                s["visual_gap_m"] is None and s["collision_gap_m"] is None for s in sections
            ),
        },
        "maximum_absolute_gap_difference_m": max(map(abs, differences)) if differences else None,
        "maximum_absolute_contact_surface_difference_m": (
            max(map(abs, surface_errors)) if surface_errors else None
        ),
        "contact_surface_probe_counts": surface_counts,
        "sections": sections,
    }


def compare_maniskill(robot, checkout):
    asset = Path(checkout).expanduser().resolve() / "mani_skill/assets/robots/alohamini2pro"
    path = asset / "urdf/alohamini2pro.urdf"
    reference = ET.parse(path).getroot()
    source = robot.description.tree

    def canonical(node, directory):
        attributes = dict(node.attrib)
        if node.tag == "mesh":
            resource = (directory / attributes["filename"]).resolve()
            expected_root = (
                asset if directory == path.parent else robot.description.robot_model.directory
            )
            if not resource.is_relative_to(expected_root) or not resource.is_file():
                raise ValueError(f"Invalid mesh in comparison: {resource}")
            attributes["filename"] = hashlib.sha256(resource.read_bytes()).hexdigest()
        return (
            node.tag,
            tuple(sorted(attributes.items())),
            tuple(canonical(c, directory) for c in node),
        )

    differences = []
    for kind in ("link", "joint"):
        a = {n.get("name"): n for n in source.findall(kind)}
        b = {n.get("name"): n for n in reference.findall(kind)}
        for name in sorted(a.keys() | b.keys()):
            if (
                name not in a
                or name not in b
                or canonical(a[name], robot.description.path.parent)
                != canonical(b[name], path.parent)
            ):
                differences.append(f"{kind}:{name}")
    return {
        "passed": not differences,
        "reference_urdf": str(path),
        "reference_urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "differences": differences,
        "scope": "URDF kinematics, inertias, primitives, and mesh contents; not a dynamics rollout",
    }
