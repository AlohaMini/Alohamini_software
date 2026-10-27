import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from alohamini_mujoco._sapien_worker import solid_distance
from alohamini_mujoco.collision import visual_sources
from alohamini_mujoco.description import Description
from alohamini_mujoco.interop import export_collision_urdf, obj_bytes
from alohamini_mujoco.meshes import Mesh

from alohamini.model import get_robot_model


def test_convex_surface_distance_handles_faces_edges_corners_and_interior():
    import itertools

    cube = np.array(list(itertools.product([-1.0, 1.0], repeat=3)))
    for point, distance in [
        ([2, 0, 0], 1),
        ([2, 2, 0], np.sqrt(2)),
        ([2, 2, 2], np.sqrt(3)),
        ([0, 0, 0], 0),
    ]:
        assert solid_distance(np.array([point]), cube) == pytest.approx(distance, abs=1e-12)


@pytest.mark.parametrize("mesh_unit", ["m", "mm"])
def test_obj_and_urdf_export_preserve_each_convex_piece_and_robot_frames(tmp_path, mesh_unit):
    description = Description(get_robot_model("alohamini2pro"))
    mesh = Mesh(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=np.float32),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
    )
    rows = obj_bytes(mesh).decode().splitlines()
    vertices = np.array(
        [[float(x) for x in row.split()[1:]] for row in rows if row.startswith("v ")]
    )
    faces = np.array(
        [[int(x) - 1 for x in row.split()[1:]] for row in rows if row.startswith("f ")]
    )
    np.testing.assert_array_equal(vertices, mesh.vertices)
    np.testing.assert_array_equal(vertices[faces], mesh.triangles)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "tetra.msh").write_bytes(mesh.msh_bytes())
    contract = {
        "schema_version": 1,
        "units": "m",
        "robot_model": description.robot_model.model_id,
        "links": {
            "left_Moving_Jaw": {
                "frame": "URDF link-local",
                "source_visuals": visual_sources(description, "left_Moving_Jaw"),
                "pieces": [
                    {
                        "path": "tetra.msh",
                        "sha256": hashlib.sha256(mesh.msh_bytes()).hexdigest(),
                        "triangles": 4,
                    }
                ],
            }
        },
    }
    (bundle / "collision.json").write_text(json.dumps(contract))
    output = export_collision_urdf(description, bundle, tmp_path / "asset", mesh_unit=mesh_unit)
    exported = ET.parse(output / "robot.urdf").getroot()
    for joint in description.tree.findall("joint"):
        candidate = exported.find(f"joint[@name='{joint.get('name')}']")
        assert candidate.attrib == joint.attrib
        assert [(c.tag, c.attrib) for c in candidate] == [(c.tag, c.attrib) for c in joint]
    for link in description.tree.findall("link"):
        original = link.find("inertial")
        if original is not None:
            candidate = exported.find(f"link[@name='{link.get('name')}']/inertial")
            assert [(c.tag, c.attrib) for c in candidate] == [(c.tag, c.attrib) for c in original]
    collisions = exported.findall("link[@name='left_Moving_Jaw']/collision")
    assert len(collisions) == 1
    assert collisions[0].find("origin").attrib == {"xyz": "0 0 0", "rpy": "0 0 0"}
    shape = collisions[0].find("geometry/mesh")
    scale = np.array([float(x) for x in shape.get("scale").split()])
    transported = np.array(
        [
            [float(x) for x in row.split()[1:]]
            for row in (output / shape.get("filename")).read_text().splitlines()
            if row.startswith("v ")
        ]
    )
    np.testing.assert_allclose(transported * scale, mesh.vertices, rtol=0, atol=1e-17)
    for node in exported.findall(".//mesh"):
        assert (output / node.get("filename")).is_file()
    manifest = json.loads((output / "asset.json").read_text())
    assert manifest["pieces"]["left_Moving_Jaw"][0]["vertices"] == 4
    assert manifest["mesh_coordinate_unit"] == mesh_unit
    assert not manifest["sapien_loader"]["load_multiple_collisions_from_file"]
