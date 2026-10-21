import json

import mujoco
import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.cad_prism import export_hollow_prism, hollow_prism

pytest.importorskip("OCP")


def tube(axis=2):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    outer, inner_low, inner_size = (
        np.array([20.0, 40.0, 100.0]),
        np.array([2.0, 2.0, 0.0]),
        np.array([16.0, 36.0, 100.0]),
    )
    order = np.roll(np.arange(3), axis - 2)
    outer, inner_low, inner_size = outer[order], inner_low[order], inner_size[order]
    outside = BRepPrimAPI_MakeBox(*outer).Shape()
    inside = BRepPrimAPI_MakeBox(gp_Pnt(*inner_low), *inner_size).Shape()
    return BRepAlgoAPI_Cut(outside, inside).Shape()


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_four_walls_match_the_complete_source_solid_in_all_explicit_axes(axis):
    primitives, proof = hollow_prism(tube(axis), axis)
    assert len(primitives) == 4
    assert proof["source_minus_boxes_empty"] and proof["boxes_minus_source_empty"]
    assert proof["box_volume_sum_m3"] == pytest.approx((20 * 40 - 16 * 36) * 100 * 1e-9)
    assert proof["source_brep_volume_m3"] == pytest.approx(proof["box_volume_sum_m3"], abs=1e-16)
    assert proof["axis"] == "xyz"[axis]


@pytest.mark.parametrize("change", ["solid_box", "extra_bore", "end_cap", "reversed"])
def test_non_prismatic_features_are_rejected_instead_of_filled_or_removed(change):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

    shape = tube()
    if change == "solid_box":
        shape = BRepPrimAPI_MakeBox(20, 40, 100).Shape()
    elif change == "extra_bore":
        drill = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 20, 50), gp_Dir(1, 0, 0)), 1, 20).Shape()
        shape = BRepAlgoAPI_Cut(shape, drill).Shape()
    elif change == "end_cap":
        shape = BRepAlgoAPI_Fuse(shape, BRepPrimAPI_MakeBox(20, 40, 2).Shape()).Shape()
    else:
        shape = shape.Reversed()
    with pytest.raises(ValueError):
        hollow_prism(shape, 2)


def test_component_export_preserves_part_frame_instance_transforms_and_cavity(tmp_path):
    from OCP.BRepTools import BRepTools

    source, output = tmp_path / "cad", tmp_path / "component"
    source.mkdir()
    (source / "source.step").write_bytes(b"source identity")
    assert BRepTools.Write_s(tube(), str(source / "part.brep"))
    occurrence = {
        "id": "tube1",
        "definition": "0:1:1:1",
        "T_cad_part": [[1, 0, 0, 0.3], [0, 1, 0, -0.2], [0, 0, 1, 0.5], [0, 0, 0, 1]],
    }
    metadata = {
        "schema_version": 1,
        "source_sha256": file_sha256(source / "source.step"),
        "files": {"part.brep": file_sha256(source / "part.brep")},
        "occurrences": [occurrence],
        "definitions": {
            "0:1:1:1": {
                "file": "part.brep",
                "brep_units": "mm",
                "valid_brep": True,
                "solids": 1,
                "name": "tube",
            }
        },
    }
    (source / "cad.json").write_text(json.dumps(metadata))
    result = export_hollow_prism(source, "0:1:1:1", output, axis=2)
    assert result["occurrences"] == [occurrence]
    assert result["proof"]["mujoco_through_hole_and_four_walls"]
    assert result["frame"] == "CAD part definition frame"
    assert not (output / "collision.json").exists()
    model = mujoco.MjModel.from_xml_path(str(output / "component.xml"))
    assert model.ngeom == 4 and model.nv == 0
    assert np.all(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    assert file_sha256(output / "component.xml") == result["files"]["component.xml"]
    with pytest.raises(FileExistsError):
        export_hollow_prism(source, "0:1:1:1", output, axis=2)
    (source / "part.brep").write_bytes(b"stale")
    with pytest.raises(ValueError, match="checksum"):
        export_hollow_prism(source, "0:1:1:1", tmp_path / "bad", axis=2)
    assert not (tmp_path / "bad").exists()
