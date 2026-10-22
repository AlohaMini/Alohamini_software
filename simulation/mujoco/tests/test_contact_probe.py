import itertools
import json
from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.compilation import compile_xml
from alohamini_mujoco.contact_probe import probe_cad_contacts
from alohamini_mujoco.meshes import Mesh
from alohamini_mujoco.solid_geometry import convex_boundary

pytest.importorskip("manifold3d")


@dataclass
class ProbeRobot:
    xml: str
    assets: dict
    model: object
    geometry: dict
    manifest: dict
    description: object
    scene: object = None

    def data(self, positions):
        data = mujoco.MjData(self.model)
        data.qpos[0] = positions["pivot"]
        mujoco.mj_forward(self.model, data)
        return data


@pytest.mark.parametrize("collision_offset,expected", [(0.0, True), (0.001, False)])
@pytest.mark.parametrize("component_only", [False, True])
def test_native_probe_follows_link_coordinates_and_detects_a_displaced_collider(
    tmp_path, monkeypatch, collision_offset, expected, component_only
):
    source = convex_boundary(np.array(list(itertools.product((-0.01, 0.01), repeat=3))))
    mesh_bundle = tmp_path / "mesh"
    mesh_bundle.mkdir()
    np.savez(mesh_bundle / "solid.npz", vertices=source.vertices, faces=source.faces)
    (mesh_bundle / "mesh.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "units": "m",
                "coordinates": "CAD part definition frame, float64",
                "linear_deflection_m": 5e-5,
                "solids": [{"file": "solid.npz", "sha256": file_sha256(mesh_bundle / "solid.npz")}],
            }
        )
    )
    fingerprint = file_sha256(mesh_bundle / "mesh.json")
    collider = Mesh(source.vertices + [collision_offset, 0, 0], source.faces)
    old_geometry = Mesh(collider.vertices + [0.003 if component_only else 0, 0, 0], collider.faces)
    assets = {"part.msh": old_geometry.msh_bytes()}
    xml = """<mujoco><asset><mesh name="part_mesh" file="part.msh"/></asset>
      <worldbody><body name="part"><joint name="pivot" axis="0 0 1"/>
        <inertial mass="1" pos="0 0 0" diaginertia="1 1 1"/>
        <geom name="part_collision" type="mesh" mesh="part_mesh" mass="0"/>
      </body></worldbody></mujoco>"""
    robot = ProbeRobot(
        xml,
        assets,
        compile_xml(xml, assets=assets),
        {"part_collision": {"kind": "collision", "link": "part"}},
        {
            "collision_bundle": {
                "manifest_sha256": "fixture",
                "definition": {
                    "links": {
                        "part": {
                            "cad_geometry": {
                                "passed": True,
                                "source_mesh_manifest_sha256": fingerprint,
                                "registration": {"T_link_part": np.eye(4).tolist()},
                            }
                        }
                    }
                },
            }
        },
        SimpleNamespace(joints={"pivot": SimpleNamespace(kind="revolute")}),
    )
    configuration = tmp_path / "probe.json"
    configuration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_mesh_manifest_sha256": fingerprint,
                "sphere_radius_m": 0.002,
                "tolerance_m": 0.0005,
                "clearances_m": [0.001, -0.0002],
                "cases": [
                    {
                        "link": "part",
                        "point_link_m": [0.01, 0, 0],
                        "outward_normal_link": [1, 0, 0],
                        "joint_positions": [{"pivot": 0}, {"pivot": 0.5}],
                    }
                ],
            }
        )
    )
    if component_only:
        from alohamini_mujoco import contact_probe

        monkeypatch.setattr(
            contact_probe,
            "load_cad_components",
            lambda description, directory: (
                {"part": [collider]},
                robot.manifest["collision_bundle"],
            ),
        )
        augmented = contact_probe.attach_component_fixture(robot, tmp_path)
        assert augmented.model.ngeom == robot.model.ngeom + 1
        np.testing.assert_array_equal(augmented.model.body_mass, robot.model.body_mass)
        np.testing.assert_array_equal(augmented.model.body_inertia, robot.model.body_inertia)
        from alohamini_mujoco.control import DynamicFollower

        with pytest.raises(ValueError, match="cannot run dynamics"):
            DynamicFollower(augmented)
        calls = []
        native_compile = contact_probe.compile_xml

        def counted_compile(*args, **kwargs):
            calls.append(True)
            return native_compile(*args, **kwargs)

        monkeypatch.setattr(contact_probe, "compile_xml", counted_compile)
    report = probe_cad_contacts(
        robot, mesh_bundle, configuration, component_bundle=tmp_path if component_only else None
    )
    assert report["passed"] == expected
    assert report["coverage"] == ("visual-component" if component_only else "complete-link")
    assert report["original_collisions_retained"] == component_only
    if component_only:
        assert len(calls) == 1
    assert len(report["rows"]) == 4
    assert all(row["joint_units"] == {"pivot": "rad"} for row in report["rows"])
    if expected:
        assert report["maximum_distance_error_m"] < 1e-6
        assert [row["native_contact_count"] > 0 for row in report["rows"]] == [
            False,
            True,
            False,
            True,
        ]
    else:
        assert report["maximum_distance_error_m"] > 0.0005
