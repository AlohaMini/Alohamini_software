import json
import math

import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256, inspect_step, tessellate_definition
from alohamini_mujoco.meshes import ray_x_intersections, surface_distances

pytest.importorskip("OCP")


def test_surface_distance_includes_face_edge_corner_and_zero_area_triangle():
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    points = np.array([[0.2, 0.2, 1], [0.7, 0.7, 0], [-1, -1, 0], [0.2, 0.2, 0]])
    np.testing.assert_allclose(
        surface_distances(points, triangle), [1, np.sqrt(0.08), np.sqrt(2), 0]
    )
    np.testing.assert_array_equal(
        surface_distances(points, triangle, batch_size=1), surface_distances(points, triangle)
    )
    np.testing.assert_allclose(surface_distances([[1, 0, 0]], np.zeros((1, 3, 3))), [1])


def test_step_snapshot_preserves_nested_placement_units_and_a_through_hole(tmp_path):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    document = TDocStd_Document(TCollection_ExtendedString("BinXCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(document)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    box = BRepPrimAPI_MakeBox(20, 30, 40).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(10, 15, 0), gp_Dir(0, 0, 1)), 5, 40).Shape()
    part = tool.AddShape(BRepAlgoAPI_Cut(box, bore).Shape(), False)
    TDataStd_Name.Set_s(part, TCollection_ExtendedString("housing_with_bore"))
    inner, root = tool.NewShape(), tool.NewShape()
    local = gp_Trsf()
    local.SetTranslation(gp_Vec(100, 0, 0))
    tool.AddComponent(inner, part, TopLoc_Location(local))
    outer = gp_Trsf()
    outer.SetRotation(gp_Ax1(gp_Pnt(), gp_Dir(0, 0, 1)), math.pi / 2)
    outer.SetTranslationPart(gp_Vec(10, 20, 30))
    tool.AddComponent(root, inner, TopLoc_Location(outer))
    tool.UpdateAssemblies()
    writer = STEPCAFControl_Writer()
    assert writer.Transfer(document, STEPControl_AsIs)
    source = tmp_path / "assembly.step"
    assert writer.Write(str(source)) == IFSelect_RetDone
    report = inspect_step(source, tmp_path / "before")
    assert report["source_sha256"] == file_sha256(source)
    leaves = [node for node in report["occurrences"] if not node["assembly"]]
    assert len(leaves) == 1
    leaf = leaves[0]
    np.testing.assert_allclose(
        leaf["T_cad_part"],
        [[0, -1, 0, 0.010], [1, 0, 0, 0.120], [0, 0, 1, 0.030], [0, 0, 0, 1]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        leaf["bounds_cad_m"], [[-0.02, 0.12, 0.03], [0.01, 0.14, 0.07]], atol=1e-8
    )
    definition = report["definitions"][leaf["definition"]]
    assert definition["valid_brep"] and definition["solids"] == 1
    assert definition["volume_m3"] == pytest.approx((24000 - math.pi * 25 * 40) * 1e-9)
    assert report["robot_mapping"] is None
    (tmp_path / "before").rename(tmp_path / "after")
    persisted = json.loads((tmp_path / "after/cad.json").read_text())
    for relative, expected in persisted["files"].items():
        assert file_sha256(tmp_path / "after" / relative) == expected
    tessellation = tessellate_definition(tmp_path / "after", leaf["definition"], tmp_path / "mesh")
    assert len(tessellation["solids"]) == 1
    mesh = np.load(tmp_path / "mesh" / tessellation["solids"][0]["file"])
    triangles = mesh["vertices"][mesh["faces"]][:, :, [2, 0, 1]]
    assert len(ray_x_intersections(triangles, 0.01, 0.015)) == 0
    np.testing.assert_allclose(ray_x_intersections(triangles, 0.002, 0.002), [0, 0.04], atol=1e-10)
    assert tessellation["solids"][0]["signed_mesh_volume_m3"] == pytest.approx(
        definition["volume_m3"], rel=0.001
    )
    with pytest.raises(FileExistsError):
        inspect_step(source, tmp_path / "after")
