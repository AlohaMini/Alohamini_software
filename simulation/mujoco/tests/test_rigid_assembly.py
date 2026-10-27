import copy
import json
import struct
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.compilation import compile_xml
from alohamini_mujoco.control import DynamicFollower
from alohamini_mujoco.description import Description
from alohamini_mujoco.frames import points_in_frame, transform
from alohamini_mujoco.meshes import Mesh, binary_stl
from alohamini_mujoco.model import ModelProfile, RobotModel, pose_attributes
from alohamini_mujoco.organization import description_hashes
from alohamini_mujoco.rigid_assembly import assembly_inputs, audit_assembly, bridge_joints
from alohamini_mujoco.shapes import Primitive


@pytest.fixture
def assembly(tmp_path):
    first = np.array([[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 30]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    triangles = np.vstack([first[faces], (first + [50, 40, 30])[faces]])
    contents = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        contents += struct.pack("<12fH", *([0, 0, 0] + triangle.ravel().tolist()), 0)
    mesh_path = tmp_path / "part.stl"
    mesh_path.write_bytes(contents)
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("""<robot name="fixture"><link name="root"/>
      <link name="source"><visual><origin xyz=".02 -.03 .04" rpy=".2 -.3 .4"/>
        <geometry><mesh filename="part.stl" scale=".001 .002 .003"/></geometry>
      </visual></link><link name="owner"/>
      <joint name="upstream" type="revolute"><parent link="root"/><child link="source"/>
        <origin xyz=".3 -.2 .1"/><axis xyz="0 0 1"/>
        <limit lower="-1.1" upper="1.2" effort="10" velocity="2"/></joint>
      <joint name="bend" type="revolute"><parent link="source"/><child link="owner"/>
        <origin xyz=".1 .02 .03" rpy="0 .3 -.2"/><axis xyz="1 0 0"/>
        <limit lower="-1.3" upper=".7" effort="10" velocity="2"/></joint></robot>""")
    semantic = tmp_path / "robot.srdf"
    semantic.write_text('<robot name="fixture"/>')
    nominal = SimpleNamespace(
        directory=tmp_path,
        model_id="fixture",
        description_path=lambda key: urdf if key == "collision" else semantic,
    )
    description = Description(nominal)
    mesh = binary_stl(contents)
    root_pose = transform([0.3, 0.4, -0.1], [0.1, 0.2, -0.3])
    root_attributes = " ".join(
        f'{key}="{value}"' for key, value in pose_attributes(root_pose).items()
    )
    visual_pose = transform([0.02, -0.03, 0.04], [0.2, -0.3, 0.4])
    visual_attributes = " ".join(
        f'{key}="{value}"' for key, value in pose_attributes(visual_pose).items()
    )
    owner_pose = transform([0.1, 0.02, 0.03], [0, 0.3, -0.2])
    owner_attributes = " ".join(
        f'{key}="{value}"' for key, value in pose_attributes(owner_pose).items()
    )
    xml = f"""<mujoco><compiler inertiafromgeom="false"/><asset>
      <mesh name="original" file="original.msh" scale=".001 .002 .003"/>
      </asset><worldbody><body name="root" {root_attributes}>
      <body name="source" pos=".3 -.2 .1"><joint name="upstream" axis="0 0 1" range="-63 69"/>
        <inertial mass="2" pos=".01 0 0" diaginertia=".02 .02 .03"/>
        <geom name="source__visual_0" type="mesh" mesh="original" {visual_attributes}
          mass="0" contype="0" conaffinity="0" group="2"/>
        <geom name="legacy" type="box" size=".01 .02 .03" group="3" mass="0"/>
        <body name="owner" {owner_attributes}>
          <joint name="bend" axis="1 0 0" range="-75 41"/>
          <inertial mass="1" pos="0 .02 0" diaginertia=".01 .02 .02"/>
        </body></body></body></worldbody></mujoco>"""
    robot = RobotModel(
        description,
        compile_xml(xml, assets={"original.msh": mesh.msh_bytes()}),
        ModelProfile(locked_positions={}, T_world_root=root_pose),
        xml,
        {"original.msh": mesh.msh_bytes()},
        {"original": mesh},
        {
            "source__visual_0": {"kind": "visual", "link": "source"},
            "legacy": {"kind": "collision", "link": "source"},
        },
        {},
        {"upstream": 0.25, "bend": -0.3},
    )
    from alohamini_mujoco.collision import visual_sources

    definition = {
        "schema_version": 1,
        "robot_model": "fixture",
        "units": "m",
        "source_description_sha256": description_hashes(description),
        "sources": [
            {
                "link": "source",
                "visual_index": 0,
                "source_visual": visual_sources(description, "source")[0],
                "components": [
                    {"component_index": 0, "owner": "source", "reference_positions": {}},
                    {"component_index": 1, "owner": "owner", "reference_positions": {"bend": 0.4}},
                ],
            }
        ],
    }
    path = tmp_path / "assembly.json"
    path.write_text(json.dumps(definition))
    return robot, definition, path


def assembled(robot, path):
    inputs = assembly_inputs(robot, path)
    return replace(robot, **inputs, model=compile_xml(inputs["xml"], assets=inputs["assets"]))


def test_native_ownership_conserves_all_triangles_reference_pose_and_physical_model(assembly):
    robot, definition, path = assembly
    candidate = assembled(robot, path)
    report = audit_assembly(candidate, samples=7)
    assert report["passed"] and not report["dynamics_allowed"]
    stationary, moving = report["components"]
    assert stationary["max_reference_displacement_m"] == 0
    assert moving["max_reference_displacement_m"] > 0.1
    assert moving["max_native_vertex_error_m"] < 1e-7
    assert moving["reference_preservation_error_m"] < 1e-14
    assert [j.name for j in bridge_joints(robot.description, "root", "owner")] == [
        "upstream",
        "bend",
    ]
    assert [j.name for j in bridge_joints(robot.description, "owner", "source")] == ["bend"]
    np.testing.assert_array_equal(candidate.model.body_mass, robot.model.body_mass)
    np.testing.assert_array_equal(candidate.model.body_inertia, robot.model.body_inertia)
    np.testing.assert_array_equal(candidate.model.body_ipos, robot.model.body_ipos)
    for q in [-1.3, 0.4, 0.7]:
        np.testing.assert_array_equal(candidate.data({"bend": q}).M, robot.data({"bend": q}).M)
    assert (
        candidate.model.geom_bodyid[candidate.model.geom("legacy").id]
        == robot.model.body("source").id
    )
    assert sum(row["triangles"] for row in report["components"]) == 8
    assert "original.msh" not in candidate.assets
    original = robot.meshes["original"].triangles
    partition = np.vstack(
        [
            candidate.meshes[candidate.geometry[r["geom"]]["path"]].triangles
            for r in report["components"]
        ]
    )
    np.testing.assert_array_equal(partition, original)
    assert robot.manifest == {} and "source__visual_0" in robot.geometry
    with pytest.raises(ValueError, match="cannot run dynamics"):
        DynamicFollower(candidate)
    with pytest.raises(ValueError, match="unmodified"):
        assembly_inputs(candidate, path)
    # An owner error must be detected even if the rendered mesh still looks plausible.
    candidate.model.geom_bodyid[candidate.model.geom(moving["geom"]).id] = robot.model.body(
        "source"
    ).id
    assert not audit_assembly(candidate, samples=2)["passed"]


@pytest.mark.parametrize(
    "damage",
    [
        "omission",
        "duplicate_component",
        "float_index",
        "duplicate_source",
        "unknown_owner",
        "implicit_reference",
        "extra_reference",
        "outside_limits",
        "wrong_scale",
        "wrong_pose",
        "wrong_mesh",
        "wrong_description",
        "wrong_units",
        "locked_reference",
    ],
)
def test_rejects_incomplete_or_stale_ownership_and_ambiguous_reference(assembly, damage):
    robot, original, path = assembly
    definition = copy.deepcopy(original)
    source = definition["sources"][0]
    moving = source["components"][1]
    if damage == "omission":
        source["components"].pop()
    elif damage == "duplicate_component":
        moving["component_index"] = 0
    elif damage == "float_index":
        moving["component_index"] = 1.0
    elif damage == "duplicate_source":
        definition["sources"].append(copy.deepcopy(source))
    elif damage == "unknown_owner":
        moving["owner"] = "missing"
    elif damage == "implicit_reference":
        moving["reference_positions"] = {}
    elif damage == "extra_reference":
        moving["reference_positions"]["upstream"] = 0
    elif damage == "outside_limits":
        moving["reference_positions"]["bend"] = 4
    elif damage == "wrong_scale":
        source["source_visual"]["scale"][0] *= 2
    elif damage == "wrong_pose":
        source["source_visual"]["T_link_visual"][0][3] += 0.001
    elif damage == "wrong_mesh":
        robot.description.robot_model.directory.joinpath("part.stl").write_bytes(b"changed")
    elif damage == "wrong_description":
        definition["source_description_sha256"] = {}
    elif damage == "wrong_units":
        definition["units"] = "mm"
    elif damage == "locked_reference":
        robot.profile = ModelProfile(locked_positions={"bend": -0.3})
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError):
        assembly_inputs(robot, path)


@pytest.mark.parametrize("envelopes", [False, True])
def test_assembly_viewer_uses_kinematic_sliders_and_reset_without_steps(
    assembly, monkeypatch, envelopes
):
    from contextlib import nullcontext

    from alohamini_mujoco import viewer

    robot, definition, path = assembly
    if envelopes:
        definition["sources"][0]["components"][1]["collision_envelope"] = {
            "primitives": [Primitive("sphere", {"radius": 0.01}, np.eye(4)).record()],
            "limitations": ["Inspection only"],
        }
        path.write_text(json.dumps(definition))
    candidate = assembled(robot, path)
    snapshots = []

    class Window:
        def __init__(self, model, data, key_callback, **kwargs):
            self.data, self.callback = data, key_callback
            self.opt, self.cam = mujoco.MjvOption(), mujoco.MjvCamera()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def lock(self):
            return nullcontext()

        def is_running(self):
            return len(snapshots) < 4

        def set_texts(self, texts):
            assert "mass allocation unqualified" in texts[3]

        def sync(self):
            snapshots.append((self.data.qpos.copy(), self.data.time, self.opt.geomgroup.copy()))
            if len(snapshots) == 1:
                self.data.qpos[1] = 0.5
                self.callback(296)
            elif len(snapshots) == 2:
                self.callback(297)
            elif len(snapshots) == 3:
                self.callback(298)

    def forbidden_step(*args, **kwargs):
        raise AssertionError("Rigid ownership inspection cannot step physics")

    monkeypatch.setattr(viewer, "check_display", lambda: None)
    monkeypatch.setattr(viewer.time, "sleep", lambda _: None)
    monkeypatch.setattr(viewer, "component_window", Window)
    monkeypatch.setattr(mujoco, "mj_step", forbidden_step)
    viewer.run_assembly_viewer(candidate)
    assert [s[0][1] for s in snapshots] == [-0.3, 0.5, 0.5, -0.3]
    assert all(s[1] == 0 for s in snapshots)
    assert [s[2][2] for s in snapshots] == [1, 1, 0, 0]
    assert [s[2][4 if envelopes else 3] for s in snapshots] == [0, 1, 1, 1]


def test_scoped_collision_reparenting_preserves_native_sphere_distance(assembly, monkeypatch):
    from alohamini_mujoco import contact_probe

    robot, _, path = assembly
    source_mesh = robot.meshes["original"]
    source_vertices = points_in_frame(
        transform([0.02, -0.03, 0.04], [0.2, -0.3, 0.4]),
        (source_mesh.vertices[4:] * [0.001, 0.002, 0.003]),
    )
    # Use the second connected tetrahedron in source-link metric coordinates.
    faces = source_mesh.faces[4:] - 4
    hull = Mesh(source_vertices, faces)

    def load(*args):
        return {"source": [hull]}, {
            "definition": {
                "links": {
                    "source": {
                        "cad_geometry": {
                            "registration": {
                                "target": {"visual_index": 0, "component_index": 1},
                            }
                        }
                    }
                }
            },
        }

    monkeypatch.setattr(contact_probe, "load_cad_components", load)
    inputs = contact_probe._component_fixtures_inputs(robot, ["qualified"])
    scoped = replace(robot, **inputs)
    assigned = assembly_inputs(scoped, path)
    root = ET.fromstring(assigned["xml"])
    probe_body = ET.SubElement(root.find("worldbody"), "body", name="probe_body", mocap="true")
    ET.SubElement(probe_body, "geom", name="probe", type="sphere", size=".0005")
    assigned["xml"] = ET.tostring(root, encoding="unicode")
    candidate = replace(
        robot, **assigned, model=compile_xml(assigned["xml"], assets=assigned["assets"])
    )
    report = audit_assembly(candidate, samples=5)
    assert report["passed"] and report["native_configurations"] == 7
    collision = report["components"][-1]
    assert collision["kind"] == "collision" and collision["owner"] == "owner"
    assert collision["max_reference_displacement_m"] > 0.1
    assert candidate.manifest["cad_component_geoms"] == {"owner": [collision["geom"]]}
    local_pose = np.array(collision["T_owner_geometry"])
    face = hull.triangles[0]
    normal = np.cross(face[1] - face[0], face[2] - face[0])
    normal /= np.linalg.norm(normal)
    probe = candidate.model.geom("probe").id
    mocap = candidate.model.body_mocapid[candidate.model.body("probe_body").id]
    geom = candidate.model.geom(collision["geom"]).id
    for q in [-1.3, -0.4, 0.4, 0.7]:
        positions = {**candidate.initial_positions, "bend": q}
        poses = candidate.description.forward(positions, candidate.profile.T_world_root)
        pose = poses["owner"] @ local_pose
        world_face = points_in_frame(pose, [face.mean(0)])[0]
        world_normal = pose[:3, :3] @ normal
        for clearance in [0.0002, -0.0001]:
            data = candidate.data(positions)
            data.mocap_pos[mocap] = world_face + (0.0005 + clearance) * world_normal
            mujoco.mj_forward(candidate.model, data)
            np.testing.assert_allclose(data.geom_xpos[probe], data.mocap_pos[mocap], atol=1e-14)
            distance = mujoco.mj_geomDistance(candidate.model, data, probe, geom, 0.01, None)
            assert abs(distance - clearance) < 1e-7
    with pytest.raises(ValueError, match="before assigning"):
        contact_probe.attach_component_fixtures(candidate, ["qualified"])
    with pytest.raises(ValueError, match="original rigid owners"):
        contact_probe.probe_cad_contacts(candidate, None, None)


def test_repository_gripper_definition_matches_current_sources_and_compiles_once(monkeypatch):
    from alohamini_mujoco import model as builder

    calls, sentinel = [], object()

    def compile_once(xml, **kwargs):
        calls.append(xml)
        root = ET.fromstring(xml)
        for side in ["left", "right"]:
            for component in [11, 12]:
                owner = root.find(f".//body[@name='{side}_Moving_Jaw']")
                assert (
                    owner.find(f"geom[@name='{side}_Fixed_Jaw__visual_0__component_{component}']")
                    is not None
                )
            envelopes = owner.findall("geom[@type='cylinder']")
            assert len(envelopes) == 2
            assert all(node.get("mass") == "0" and node.get("group") == "4" for node in envelopes)
        return sentinel

    monkeypatch.setattr(builder, "compile_xml", compile_once)
    path = Path(__file__).resolve().parents[3] / "examples/simulation/gripper_assembly.json"
    robot = builder.build_model(rigid_assembly=path)
    assert len(calls) == 1 and robot.model is sentinel
    sources = robot.manifest["rigid_assembly_fixture"]["sources"]
    assert [len(source["components"]) for source in sources] == [18, 18]
    assert sum(source["triangles"] for source in sources) == 174608


@pytest.mark.parametrize(
    "kind,parameters,point,size",
    [
        ("box", {"size": [0.04, 0.06, 0.08]}, [0.02, 0, 0], [0.02, 0.03, 0.04]),
        ("cylinder", {"radius": 0.02, "length": 0.06}, [0.02, 0, 0], [0.02, 0.03, 0]),
        ("sphere", {"radius": 0.02}, [0.02, 0, 0], [0.02, 0, 0]),
    ],
)
def test_primitive_envelope_preserves_owner_frame_inertia_and_native_contacts(
    assembly, kind, parameters, point, size
):
    robot, definition, path = assembly
    primitive = Primitive(kind, parameters, transform([0.08, -0.05, 0.03], [0.3, -0.4, 0.5]))
    definition["sources"][0]["components"][1]["collision_envelope"] = {
        "primitives": [primitive.record()],
        "limitations": ["Synthetic envelope; not a material-equivalence certificate"],
    }
    path.write_text(json.dumps(definition))
    inputs = assembly_inputs(robot, path)
    root = ET.fromstring(inputs["xml"])
    probe_body = ET.SubElement(root.find("worldbody"), "body", name="probe_body", mocap="true")
    ET.SubElement(probe_body, "geom", name="probe", type="sphere", size=".0005")
    inputs["xml"] = ET.tostring(root, encoding="unicode")
    candidate = replace(robot, **inputs, model=compile_xml(inputs["xml"], assets=inputs["assets"]))
    report = audit_assembly(candidate, samples=5)
    assert report["passed"] and not report["dynamics_allowed"]
    row = report["components"][-1]
    assert row["shape"] == kind and row["owner"] == "owner"
    assert row["max_reference_displacement_m"] > 0.1
    assert row["max_native_vertex_error_m"] < 1e-14
    assert row["reference_preservation_error_m"] < 1e-14
    geom = candidate.model.geom(row["geom"]).id
    np.testing.assert_array_equal(candidate.model.geom_size[geom], size)
    np.testing.assert_array_equal(candidate.model.body_mass[:4], robot.model.body_mass)
    assert candidate.manifest["assembly_envelope_geoms"] == {"owner": [row["geom"]]}
    assert "cad_component_fixture" not in candidate.manifest
    probe = candidate.model.geom("probe").id
    mocap = candidate.model.body_mocapid[candidate.model.body("probe_body").id]
    for q in [-1.3, 0.4, 0.7]:
        data = candidate.data({"bend": q})
        np.testing.assert_array_equal(data.M, robot.data({"bend": q}).M)
        poses = candidate.description.forward(
            {**candidate.initial_positions, "bend": q}, candidate.profile.T_world_root
        )
        pose = poses["owner"] @ row["T_owner_geometry"]
        for clearance in [0.0002, -0.0001]:
            center = np.array(point) + np.array([0.0005 + clearance, 0, 0])
            data.mocap_pos[mocap] = points_in_frame(pose, [center])[0]
            mujoco.mj_forward(candidate.model, data)
            distance = mujoco.mj_geomDistance(candidate.model, data, probe, geom, 0.01, None)
            assert abs(distance - clearance) < 1e-8
    with pytest.raises(ValueError, match="cannot run dynamics"):
        DynamicFollower(candidate)
    candidate.model.geom_size[geom, 0] *= 1.1
    assert not audit_assembly(candidate, samples=2)["passed"]


@pytest.mark.parametrize(
    "damage", ["empty", "implicit_limits", "extra_claim", "invalid_pose", "duplicate"]
)
def test_rejects_ambiguous_or_duplicate_inspection_envelopes(assembly, damage):
    robot, definition, path = assembly
    primitive = Primitive("sphere", {"radius": 0.01}, np.eye(4)).record()
    envelope = {"primitives": [primitive], "limitations": ["Bore accessibility unverified"]}
    definition["sources"][0]["components"][1]["collision_envelope"] = envelope
    if damage == "empty":
        envelope["primitives"] = []
    elif damage == "implicit_limits":
        envelope["limitations"] = []
    elif damage == "extra_claim":
        envelope["passed"] = True
    elif damage == "invalid_pose":
        primitive["T_link_geometry"][0][0] = 2
    elif damage == "duplicate":
        robot.manifest["cad_component_bindings"] = {
            "existing_hull": {"source_link": "source", "visual_index": 0, "component_index": 1}
        }
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError):
        assembly_inputs(robot, path)
