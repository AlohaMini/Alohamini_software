import json
import struct
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.cad_datums import fit_translation, place_from_datums, planar_faces

pytest.importorskip("OCP")


def fixture(tmp_path):
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pnt

    cad = tmp_path / "cad"
    cad.mkdir()
    shape = BRepPrimAPI_MakeBox(gp_Pnt(1000, 2000, 3000), 31.25, 62.5, 125).Shape()
    BRepTools.Write_s(shape, str(cad / "part.brep"))
    (cad / "source.step").write_bytes(b"identity")
    record = {
        "schema_version": 1,
        "occt_length_unit_mm": 1,
        "source_sha256": file_sha256(cad / "source.step"),
        "files": {"part.brep": file_sha256(cad / "part.brep")},
        "definitions": {"box": {"file": "part.brep", "brep_units": "mm", "valid_brep": True}},
    }
    (cad / "cad.json").write_text(json.dumps(record))
    planes = planar_faces(shape)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    translation = np.array([0.125, 0.25, 0.5]) - rotation @ [1, 2, 3]
    triangles, datums = [], []
    for index, plane in planes.items():
        n = np.array(plane["normal_part"])
        axis = abs(n).argmax()
        fixed = plane["offset_part_m"] / n[axis]
        low, high = np.array([1, 2, 3.0]), np.array([1.03125, 2.0625, 3.125])
        a, b = [i for i in range(3) if i != axis]
        corners = []
        for u, v in [(0, 0), (1, 0), (1, 1), (0, 1)]:
            point = low.copy()
            point[axis], point[a], point[b] = fixed, [low[a], high[a]][u], [low[b], high[b]][v]
            corners.append(rotation @ point + translation)
        first = len(triangles)
        triangles.extend(np.array(corners)[[[0, 1, 2], [0, 2, 3]]])
        datums.append(
            {
                "name": f"plane_{index}",
                "role": "fit" if abs(fixed - low[axis]) < 1e-12 else "check",
                "cad_face": index,
                "visual_triangles": [first, first + 1],
            }
        )
    stl = b"\0" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        stl += struct.pack("<12fH", *([0, 0, 0] + triangle.ravel().tolist()), 0)
    (tmp_path / "part.stl").write_bytes(stl)
    # Initialize the XML parser before OCCT in a fresh process as the CLI does.
    link = ET.fromstring(
        '<link><visual><geometry><mesh filename="part.stl"/></geometry></visual></link>'
    )
    description = SimpleNamespace(
        robot_model=SimpleNamespace(directory=tmp_path, model_id="fixture"),
        links={"part": link},
        mesh_path=lambda _: tmp_path / "part.stl",
        source_bytes={},
    )
    binding = {
        "schema_version": 1,
        "definition": "box",
        "cad_manifest_sha256": file_sha256(cad / "cad.json"),
        "link": "part",
        "visual_index": 0,
        "visual_sha256": file_sha256(tmp_path / "part.stl"),
        "R_link_part": rotation.tolist(),
        "datums": datums,
    }
    path = tmp_path / "binding.json"
    path.write_text(json.dumps(binding))
    return cad, path, description, translation


def test_plane_binding_keeps_declared_axes_and_checks_opposite_faces(tmp_path):
    cad, binding, description, translation = fixture(tmp_path)
    report = place_from_datums(description, cad, binding, tmp_path / "placed")
    assert report["passed"] and report["checks_passed"]
    assert not report["declared_rotation_fitted"]
    np.testing.assert_allclose(np.array(report["T_link_part"])[:3, 3], translation, atol=1e-12)
    assert all(d["maximum_residual_m"] < 1e-12 for d in report["datums"])
    saved = json.loads((tmp_path / "placed/placement.json").read_text())
    assert saved == report
    with pytest.raises(FileExistsError):
        place_from_datums(description, cad, binding, tmp_path / "placed")
    (tmp_path / "part.stl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="visual checksum"):
        place_from_datums(description, cad, binding, tmp_path / "bad")


def synthetic_datums():
    planes, points, datums = {}, [], []
    for axis in range(3):
        for upper in (False, True):
            index = 2 * axis + upper
            normal = np.eye(3)[axis]
            planes[index] = {
                "normal_part": normal,
                "offset_part_m": float(upper),
                "local_brep_tolerance_m": 1e-10,
            }
            a, b = [i for i in range(3) if i != axis]
            patch = np.zeros((3, 3))
            patch[:, axis], patch[1, a], patch[2, b] = upper, 1, 1
            points.extend(patch)
            datums.append(
                {
                    "name": str(index),
                    "role": "check" if upper else "fit",
                    "cad_face": index,
                    "point_indices": list(range(3 * index, 3 * index + 3)),
                }
            )
    return planes, np.array(points), datums


def test_changed_dimensions_and_skew_fail_without_moving_the_reference_frame():
    planes, points, datums = synthetic_datums()
    # Change a held-out wall by 0.4 mm. It cannot bias the fitted translation.
    points[3:6, 0] += 0.0004
    result = fit_translation(planes, points, np.zeros_like(points), np.eye(3), datums)
    assert result["fit_passed"] and not result["checks_passed"]
    assert not result["passed"]
    np.testing.assert_allclose(np.array(result["T_link_part"])[:3, 3], 0, atol=1e-12)
    assert result["datums"][1]["maximum_residual_m"] == pytest.approx(0.0004)
    points[1, 0] += 1e-5
    assert not fit_translation(planes, points, np.zeros_like(points), np.eye(3), datums)[
        "fit_passed"
    ]


def test_adjacent_datums_may_share_boundary_vertices():
    planes, points, datums = synthetic_datums()
    vertices, inverse = np.unique(points, axis=0, return_inverse=True)
    for datum in datums:
        datum["point_indices"] = inverse[datum["point_indices"]].tolist()
    result = fit_translation(planes, vertices, np.zeros_like(vertices), np.eye(3), datums)
    assert result["passed"]


@pytest.mark.parametrize("invalid", ["rank", "collinear", "no_check", "duplicate"])
def test_ambiguous_or_reused_datums_are_rejected(invalid):
    planes, points, datums = synthetic_datums()
    if invalid == "rank":
        planes[4]["normal_part"] = [1, 0, 0]
    elif invalid == "collinear":
        points[1:3] = points[0]
    elif invalid == "no_check":
        datums = [d for d in datums if d["role"] == "fit"]
    else:
        datums[-1]["point_indices"] = datums[0]["point_indices"]
    with pytest.raises(ValueError):
        fit_translation(planes, points, np.zeros_like(points), np.eye(3), datums)
