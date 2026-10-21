import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco._sapien_worker import primitive_distance_bound
from alohamini_mujoco.collision import load_bundle, merge_bundles, visual_sources
from alohamini_mujoco.description import Description, origin
from alohamini_mujoco.frames import points_in_frame, transform
from alohamini_mujoco.interop import export_collision_urdf
from alohamini_mujoco.meshes import binary_stl
from alohamini_mujoco.model import ModelProfile, build_model
from alohamini_mujoco.primitives import fit_primitives
from alohamini_mujoco.shapes import Primitive

from alohamini.model import get_robot_model

LINKS = ["chest_camera", "front_camera", "back_camera", "wheel1", "wheel2", "wheel3"]


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    description = Description(get_robot_model("alohamini2pro"))
    output = tmp_path_factory.mktemp("primitive") / "fitted"
    fit_primitives(description, LINKS, output)
    return description, output


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("box", {"size": [1, 2, 3]}),
        ("sphere", {"radius": 0.01}),
        ("cylinder", {"radius": 0.04, "length": 0.1}),
    ],
)
@pytest.mark.parametrize("pitch", [-np.pi / 2, -0.4, np.pi / 2])
def test_full_si_dimensions_and_link_pose_roundtrip_without_axis_inference(kind, parameters, pitch):
    pose = transform([0.11, -0.07, 0.31], [0.3, pitch, -0.8])
    primitive = Primitive.from_record(Primitive(kind, parameters, pose).record())
    node = primitive.collision_node("part")
    np.testing.assert_allclose(origin(node), pose, rtol=0, atol=2e-12)
    assert primitive.parameters == parameters
    assert node.find(f"geometry/{kind}") is not None


@pytest.mark.parametrize(
    "kind,parameters",
    [
        ("capsule", {"radius": 1, "length": 1}),
        ("box", {"size": [1, 2]}),
        ("box", {"size": [1, 2, 0]}),
        ("box", {"size": [1, np.inf, 3]}),
        ("box", {"size": [True, 2, 3]}),
        ("sphere", {"radius": True}),
        ("sphere", {"radius": "1"}),
        ("sphere", {"radius": -1}),
        ("cylinder", {"radius": 1, "half_length": 0.1}),
    ],
)
def test_primitive_dimensions_require_explicit_positive_si_contract(kind, parameters):
    with pytest.raises(ValueError):
        Primitive(kind, parameters, np.eye(4))


def test_fits_contain_the_entire_visual_in_declared_axes_and_preserve_wheel_shafts(fitted):
    description, output = fitted
    shapes, provenance = load_bundle(description, output)
    assert provenance["definition"]["schema_version"] == 2
    for name, pieces in shapes.items():
        assert len(pieces) == 1
        primitive = pieces[0]
        points = []
        for source in visual_sources(description, name):
            mesh = binary_stl((description.robot_model.directory / source["path"]).read_bytes())
            points.append(points_in_frame(source["T_link_visual"], mesh.vertices * source["scale"]))
        vertices = points_in_frame(np.linalg.inv(primitive.T_link_geometry), np.vstack(points))
        source_pose = origin(description.links[name].find("collision"))
        np.testing.assert_allclose(
            primitive.T_link_geometry[:3, :3], source_pose[:3, :3], atol=1e-15
        )
        if primitive.kind == "box":
            size = np.asarray(primitive.parameters["size"])
            assert np.all(np.abs(vertices) <= size / 2)
            source_size = np.array(
                [
                    float(x)
                    for x in description.links[name]
                    .find("collision/geometry/box")
                    .get("size")
                    .split()
                ]
            )
            np.testing.assert_allclose(source_size - size, [0.004] * 3, rtol=0, atol=5e-8)
        else:
            assert np.linalg.norm(vertices[:, :2], axis=1).max() <= primitive.parameters["radius"]
            assert np.abs(vertices[:, 2]).max() <= primitive.parameters["length"] / 2
            offset = source_pose[:3, :3].T @ (primitive.T_link_geometry[:3, 3] - source_pose[:3, 3])
            np.testing.assert_allclose(offset[:2], 0, atol=1e-15)


def test_native_mujoco_and_portable_urdf_preserve_analytic_types_frames_and_dynamics(
    fitted, tmp_path
):
    import xml.etree.ElementTree as ET

    description, output = fitted
    shapes, _ = load_bundle(description, output)
    reference = build_model()
    robot = build_model(profile=ModelProfile(collision_bundle=output))
    np.testing.assert_array_equal(robot.model.body_mass, reference.model.body_mass)
    np.testing.assert_array_equal(robot.model.body_inertia, reference.model.body_inertia)
    np.testing.assert_array_equal(robot.model.jnt_range, reference.model.jnt_range)
    assert robot.model.nexclude == reference.model.nexclude == 38
    assert robot.model.ngeom == reference.model.ngeom
    export = export_collision_urdf(description, output, tmp_path / "portable", mesh_unit="mm")
    tree = ET.parse(export / "robot.urdf")
    manifest = json.loads((export / "asset.json").read_text())
    for name, pieces in shapes.items():
        primitive = pieces[0]
        entry = robot.geometry[f"{name}__collision_0"]
        assert entry["shape"] == primitive.kind
        np.testing.assert_allclose(entry["T_link_geometry"], primitive.T_link_geometry, atol=2e-12)
        geom = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}__collision_0")
        expected_type = (
            mujoco.mjtGeom.mjGEOM_BOX if primitive.kind == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        )
        assert robot.model.geom_type[geom] == expected_type
        node = tree.find(f"link[@name='{name}']/collision")
        np.testing.assert_allclose(origin(node), primitive.T_link_geometry, atol=2e-12)
        assert node.find(f"geometry/{primitive.kind}") is not None
        assert manifest["pieces"][name][0] == primitive.record()


def test_mixed_schema_composition_keeps_legacy_hulls_and_native_primitives(fitted, tmp_path):
    from test_collision import fixture_bundle

    description, primitive = fitted
    legacy = tmp_path / "legacy"
    fixture_bundle(description, legacy)
    result = merge_bundles(description, [legacy, primitive], tmp_path / "mixed")
    loaded, metadata = load_bundle(description, result)
    assert metadata["definition"]["schema_version"] == 2
    assert len(loaded) == 7
    assert len(loaded["right_Moving_Jaw"][0].faces) == 4
    assert isinstance(loaded["chest_camera"][0], Primitive)
    with pytest.raises(ValueError, match="one existing native"):
        fit_primitives(description, ["left_Upper_Arm"], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_one_rigid_link_can_combine_native_shapes_with_convex_meshes(tmp_path):
    import xml.etree.ElementTree as ET

    from test_collision import fixture_bundle

    description = Description(get_robot_model("alohamini2pro"))
    directory = tmp_path / "compound"
    metadata, record = fixture_bundle(description, directory)
    metadata["schema_version"] = 2
    record["pieces"][0]["type"] = "mesh"
    record["pieces"].append(
        Primitive("box", {"size": [0.01, 0.02, 0.03]}, transform([0.01, 0.02, 0.03])).record()
    )
    record["pieces"].append(Primitive("sphere", {"radius": 0.005}, np.eye(4)).record())
    (directory / "collision.json").write_text(json.dumps(metadata))
    robot = build_model(profile=ModelProfile(collision_bundle=directory))
    assert [robot.geometry[f"right_Moving_Jaw__collision_{i}"]["shape"] for i in range(3)] == [
        "mesh",
        "box",
        "sphere",
    ]
    exported = export_collision_urdf(description, directory, tmp_path / "portable")
    tree = ET.parse(exported / "robot.urdf")
    assert [
        list(n.find("geometry"))[0].tag
        for n in tree.findall("link[@name='right_Moving_Jaw']/collision")
    ] == ["mesh", "box", "sphere"]


def test_loader_rejects_ambiguous_primitive_or_misplaced_mesh_and_stale_sources(fitted, tmp_path):
    description, output = fitted
    data = json.loads((output / "collision.json").read_text())
    changed = tmp_path / "changed"
    changed.mkdir()
    for mutation in ("scale", "missing_pose", "unknown_type", "schema1", "source_changed"):
        clone = json.loads(json.dumps(data))
        record = clone["links"]["chest_camera"]
        piece = record["pieces"][0]
        if mutation == "scale":
            piece["scale"] = [1, 1, 1]
        elif mutation == "missing_pose":
            del piece["T_link_geometry"]
        elif mutation == "unknown_type":
            piece["type"] = "capsule"
        elif mutation == "schema1":
            clone["schema_version"] = 1
        else:
            record["source_collisions"][0]["parameters"]["size"] = "1 1 1"
        (changed / "collision.json").write_text(json.dumps(clone))
        with pytest.raises(ValueError):
            load_bundle(description, changed)


def test_analytic_sapien_bounds_detect_dimension_frame_and_representation_changes():
    class Pose:
        def __init__(self, matrix):
            self.matrix = matrix

        def to_transformation_matrix(self):
            return self.matrix

    class Box:
        pass

    class Sphere:
        pass

    class Cylinder:
        pass

    physx = SimpleNamespace(
        PhysxCollisionShapeBox=Box,
        PhysxCollisionShapeSphere=Sphere,
        PhysxCollisionShapeCylinder=Cylinder,
    )
    pose = transform([0.01, -0.1, 0.2], [0.3, -0.2, 0.7])
    box = Box()
    box.local_pose, box.half_size = Pose(pose), np.array([0.02, 0.03, 0.04])
    expected = Primitive("box", {"size": [0.04, 0.06, 0.08]}, pose).record()
    assert primitive_distance_bound(box, expected, physx) < 1e-12
    box.half_size[1] += 0.002
    assert primitive_distance_bound(box, expected, physx) == pytest.approx(0.002, abs=1e-12)
    sphere = Sphere()
    sphere.local_pose, sphere.radius = Pose(pose), 0.04
    ball = Primitive("sphere", {"radius": 0.04}, pose).record()
    assert primitive_distance_bound(sphere, ball, physx) == 0
    cylinder = Cylinder()
    # Loader turns local X into the URDF local Z axis.
    cylinder.local_pose = Pose(pose @ transform(rpy=[0, np.pi / 2, 0]))
    cylinder.radius, cylinder.half_length = 0.03, 0.05
    expected = Primitive("cylinder", {"radius": 0.03, "length": 0.1}, pose).record()
    assert primitive_distance_bound(cylinder, expected, physx) < 1e-12
    cylinder.local_pose = Pose(pose)
    assert primitive_distance_bound(cylinder, expected, physx) > 0.05
    with pytest.raises(ValueError, match="Expected native"):
        primitive_distance_bound(box, expected, physx)
