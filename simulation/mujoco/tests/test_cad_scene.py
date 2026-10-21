import json

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.cad_scene import export_cad_scene
from alohamini_mujoco.frames import points_in_frame, transform

pytest.importorskip("OCP")


def source_bundle(tmp_path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS_Shell

    source = tmp_path / "cad"
    source.mkdir()
    box = BRepPrimAPI_MakeBox(20, 40, 100).Shape()
    builder, shell = BRep_Builder(), TopoDS_Shell()
    builder.MakeShell(shell)
    explorer = TopExp_Explorer(box, TopAbs_FACE)
    explorer.Next()
    while explorer.More():
        builder.Add(shell, explorer.Current())
        explorer.Next()
    BRepTools.Write_s(box, str(source / "box.brep"))
    BRepTools.Write_s(shell, str(source / "open.brep"))
    (source / "source.step").write_bytes(b"fixture identity")
    manifest = {
        "schema_version": 1,
        "occt_length_unit_mm": 1,
        "source_sha256": file_sha256(source / "source.step"),
        "definitions": {
            key: {
                "name": key,
                "file": key + ".brep",
                "brep_units": "mm",
                "valid_brep": True,
                "solids": int(key == "box"),
                "faces": 6 if key == "box" else 5,
            }
            for key in ("box", "open")
        },
        "occurrences": [
            {"id": "assembly", "definition": "root", "assembly": True},
            {
                "id": "assembly/box",
                "definition": "box",
                "assembly": False,
                "T_cad_part": transform([0.1, 0.2, 0.3], [0.2, 0.3, 0.4]).tolist(),
            },
            {
                "id": "assembly/open",
                "definition": "open",
                "assembly": False,
                "T_cad_part": transform([-0.1, 0.3, 0.5], [0.4, -0.5, 0.6]).tolist(),
            },
            {
                "id": "assembly2/box",
                "definition": "box",
                "assembly": False,
                "T_cad_part": transform([10, 0, 0]).tolist(),
            },
        ],
        "files": {key + ".brep": file_sha256(source / (key + ".brep")) for key in ("box", "open")},
    }
    (source / "cad.json").write_text(json.dumps(manifest))
    return source


def test_scene_keeps_all_faces_open_shells_and_exact_occurrence_transforms(tmp_path):
    source, output = source_bundle(tmp_path), tmp_path / "scene"
    world = transform([0.4, -0.7, 0.2], [0.5, 0.6, -0.7])
    result = export_cad_scene(source, output, T_world_cad=world, occurrence="assembly")
    assert len(result["occurrences"]) == 2
    assert result["all_source_faces_included"] and result["visual_only"]
    assert result["native_verification"]["occurrence_frames_verified"]
    assert result["native_verification"]["visual_bounds_within_encoding_error"]
    assert result["parts"]["open"]["source_solids"] == 0
    assert result["parts"]["open"]["source_faces"] == 5
    native = mujoco.MjModel.from_xml_path(str(output / "model.xml"))
    data = mujoco.MjData(native)
    mujoco.mj_forward(native, data)
    assert native.nq == native.nv == native.nu == 0
    assert native.ngeom == 2
    assert not native.geom_contype.any() and not native.geom_conaffinity.any()
    for occurrence in result["occurrences"]:
        body = native.body(occurrence["body"]).id
        expected = world @ occurrence["T_cad_part"]
        np.testing.assert_allclose(data.xpos[body], expected[:3, 3], atol=1e-14)
        np.testing.assert_allclose(data.xmat[body].reshape(3, 3), expected[:3, :3], atol=1e-14)
        part = result["parts"][occurrence["definition"]]
        with np.load(output / "meshes" / part["asset"] / "mesh.npz") as archive:
            target = points_in_frame(expected, archive["vertices"])
        geom = native.geom(occurrence["geom"]).id
        mesh = native.geom_dataid[geom]
        first, count = native.mesh_vertadr[mesh], native.mesh_vertnum[mesh]
        actual = (
            native.mesh_vert[first : first + count] @ data.geom_xmat[geom].reshape(3, 3).T
            + data.geom_xpos[geom]
        )
        # Read-back includes MuJoCo's centering, principal axes and float32 mesh storage.
        np.testing.assert_allclose(
            [actual.min(0), actual.max(0)], [target.min(0), target.max(0)], atol=2e-8, rtol=0
        )
    for path, checksum in result["files"].items():
        assert file_sha256(output / path) == checksum
    with pytest.raises(FileExistsError):
        export_cad_scene(source, output, T_world_cad=world)


def test_scene_refuses_source_tampering_and_inexact_assembly_selection(tmp_path):
    source = source_bundle(tmp_path)
    with pytest.raises(ValueError, match="exact CAD occurrence"):
        export_cad_scene(source, tmp_path / "wrong", T_world_cad=np.eye(4), occurrence="assem")
    (source / "box.brep").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="BRep identity"):
        export_cad_scene(source, tmp_path / "bad", T_world_cad=np.eye(4))
    assert not (tmp_path / "bad").exists()


def test_scene_aborts_if_a_source_face_is_missing_from_the_inventory(tmp_path):
    source = source_bundle(tmp_path)
    path = source / "cad.json"
    manifest = json.loads(path.read_text())
    manifest["definitions"]["box"]["faces"] += 1
    path.write_text(json.dumps(manifest))
    output = tmp_path / "scene"
    with pytest.raises(ValueError, match="face coverage changed"):
        export_cad_scene(source, output, T_world_cad=np.eye(4))
    assert not output.exists()
