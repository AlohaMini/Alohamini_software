"""Shared assertions for native collision-import integration tests."""

import json

import numpy as np
from scipy.spatial import ConvexHull


def _convex_distance(first, second):
    import trimesh

    # trimesh's default near-edge tolerances assume unit-sized geometry.
    # Use numeric millimetres so they cannot mask micrometre-scale errors.
    first, second = first * 1000, second * 1000
    hull = ConvexHull(second)
    outside = np.any(first @ hull.equations[:, :3].T + hull.equations[:, 3] > 0, axis=1)
    if not outside.any():
        return 0.0
    surface = trimesh.Trimesh(second, hull.simplices, process=False)
    return float(trimesh.proximity.closest_point_naive(surface, first[outside])[1].max()) / 1000


def assert_cooked_gripper_shapes(root, shapes_by_link):
    reference = json.loads((root / "config/gripper_reference.json").read_text())
    certificate = json.loads((root / "config/gripper_collision.json").read_text())

    for kind, record in reference["parts"].items():
        for name in record["links"]:
            shapes = shapes_by_link[name]
            offset = record["retained_collision_count"]
            for index, piece in enumerate(certificate["parts"][kind]["pieces"], start=offset):
                shape = shapes[index]
                pose = shape.get_local_pose().to_transformation_matrix()
                if piece.get("type") == "box":
                    np.testing.assert_allclose(
                        shape.get_half_size() * 2, piece["parameters"]["size"], atol=1e-8, rtol=0
                    )
                    np.testing.assert_allclose(pose, piece["T_link_geometry"], atol=1e-8, rtol=0)
                    continue
                expected = (
                    np.array(
                        [
                            [float(v) for v in line.split()[1:]]
                            for line in (root / piece["mesh"]).read_text().splitlines()
                            if line.startswith("v ")
                        ]
                    )
                    * piece["scale"]
                )
                cooked = np.asarray(shape.get_vertices()) * shape.get_scale()
                cooked = cooked @ pose[:3, :3].T + pose[:3, 3]
                assert piece["vertices"] <= 128
                assert piece["triangles"] <= 252
                assert (
                    max(_convex_distance(expected, cooked), _convex_distance(cooked, expected))
                    < 0.000002
                ), (
                    name,
                    index,
                )


def assert_structural_primitives(description, shapes_by_link):
    import sapien

    from alohamini.simulation.sapien import rigid_groups

    record = json.loads(
        (description.robot_model.directory / "config/structural_collision.json").read_text()
    )
    frames, _ = rigid_groups(description)
    records = {
        **record["arm_bases"],
        record["lift_body"]["link"]: record["lift_body"],
    }
    for name, recipe in records.items():
        owner, local = frames[name]
        for primitive in recipe["pieces"]:
            expected = local @ np.asarray(primitive["T_link_geometry"])
            kind = primitive["type"]
            if kind == "mesh":
                path = description.robot_model.directory / primitive["mesh"]
                vertices = (
                    np.array(
                        [
                            [float(x) for x in line.split()[1:]]
                            for line in path.read_text().splitlines()
                            if line.startswith("v ")
                        ]
                    )
                    * primitive["scale"]
                )
                vertices = vertices @ expected[:3, :3].T + expected[:3, 3]
                meshes = []
                for shape in shapes_by_link[owner]:
                    if not isinstance(shape, sapien.physx.PhysxCollisionShapeConvexMesh):
                        continue
                    pose = shape.get_local_pose().to_transformation_matrix()
                    points = np.asarray(shape.get_vertices()) * shape.get_scale()
                    points = points @ pose[:3, :3].T + pose[:3, 3]
                    if np.allclose(
                        [points.min(0), points.max(0)],
                        [vertices.min(0), vertices.max(0)],
                        atol=2e-6,
                        rtol=0,
                    ):
                        meshes.append(points)
                assert len(meshes) == 1, (name, primitive["mesh"])
                error = max(
                    _convex_distance(vertices, meshes[0]),
                    _convex_distance(meshes[0], vertices),
                )
                assert error < 2e-6, (name, primitive["mesh"], error)
                continue
            shape_type = (
                sapien.physx.PhysxCollisionShapeBox
                if kind == "box"
                else sapien.physx.PhysxCollisionShapeCylinder
            )
            matches = [
                shape
                for shape in shapes_by_link[owner]
                if isinstance(shape, shape_type)
                and np.allclose(shape.get_local_pose().p, expected[:3, 3], atol=1e-7, rtol=0)
            ]
            assert len(matches) == 1, (name, kind)
            shape = matches[0]
            actual = shape.get_local_pose().to_transformation_matrix()
            if kind == "box":
                np.testing.assert_allclose(
                    shape.get_half_size() * 2, primitive["parameters"]["size"], atol=1e-7, rtol=0
                )
                np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0)
            else:
                np.testing.assert_allclose(
                    [shape.get_radius(), shape.get_half_length() * 2],
                    [primitive["parameters"]["radius"], primitive["parameters"]["length"]],
                    atol=1e-7,
                    rtol=0,
                )
                # SAPIEN cylinders use local X; URDF cylinders use local Z.
                assert abs(actual[:3, 0] @ expected[:3, 2]) > 1 - 1e-6
