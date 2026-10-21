import numpy as np
import pytest
from alohamini_mujoco._cad_mesh import mesh_solid
from alohamini_mujoco.collision import topology
from alohamini_mujoco.meshes import ray_x_intersections

pytest.importorskip("OCP")


def test_a_real_narrow_slot_is_not_welded_closed_by_spatial_proximity():
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS

    outer = BRepPrimAPI_MakeBox(10, 20, 30).Shape()
    slit = BRepPrimAPI_MakeBox(gp_Pnt(4.99999, 5, 0), 0.00002, 10, 30).Shape()
    cut = BRepAlgoAPI_Cut(outer, slit).Shape()
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    solid = TopoDS.Solid_s(TopExp_Explorer(cut, TopAbs_SOLID).Current())
    mesh, binding = mesh_solid(solid, 5e-5, 0.1)
    assert not any(topology(mesh).values())
    hits = ray_x_intersections(mesh.triangles, 0.01, 0.015)
    np.testing.assert_allclose(hits, [0, 0.00499999, 0.00500001, 0.01], atol=1e-14)
    assert binding["maximum_vertex_displacement_m"] < 1e-12


@pytest.mark.parametrize("kind", ["sphere", "cylinder"])
def test_periodic_seams_and_degenerate_poles_keep_a_closed_oriented_boundary(kind):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
    from OCP.TopoDS import TopoDS

    shape = (
        BRepPrimAPI_MakeSphere(5).Shape()
        if kind == "sphere"
        else BRepPrimAPI_MakeCylinder(5, 20).Shape()
    )
    mesh, binding = mesh_solid(TopoDS.Solid_s(shape), 5e-5, 0.1)
    assert not any(topology(mesh).values())
    assert binding["bound_edges"] == binding["source_edges"]
    assert binding["maximum_vertex_displacement_m"] < 1e-12
