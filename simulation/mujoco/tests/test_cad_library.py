import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.cad_library import tessellate_assembly
from alohamini_mujoco.frames import transform

pytest.importorskip("OCP")


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools
    from OCP.gp import gp_Pln
    from OCP.TopoDS import TopoDS_Solid

    root = tmp_path_factory.mktemp("cad_library")
    source = root / "cad"
    (source / "parts").mkdir(parents=True)
    (source / "source.step").write_bytes(b"source identity; preserved BRep files supply geometry")
    empty = TopoDS_Solid()
    BRep_Builder().MakeSolid(empty)
    shapes = [
        BRepPrimAPI_MakeBox(10, 20, 30).Shape(),
        BRepBuilderAPI_MakeFace(gp_Pln(), 0, 10, 0, 10).Face(),
        empty,
    ]
    definitions, files, occurrences = {}, {}, []
    for i, shape in enumerate(shapes, 1):
        definition, relative = f"0:1:1:{i}", f"parts/{i}.brep"
        assert BRepTools.Write_s(shape, str(source / relative))
        files[relative] = file_sha256(source / relative)
        definitions[definition] = {
            "file": relative,
            "name": f"part{i}",
            "brep_units": "mm",
            "valid_brep": i != 3,
            "solids": int(i != 2),
            "shells": int(i == 1),
            "volume_m3": 6e-6 if i == 1 else None,
        }
        occurrences.append(
            {
                "id": f"instance{i}",
                "definition": definition,
                "assembly": False,
                "T_cad_part": transform([0.1, i * 0.2, 0.3], [0.2, 0, -0.3]).tolist(),
            }
        )
    occurrences.append(
        {
            **occurrences[0],
            "id": "second_instance",
            "T_cad_part": transform([-0.5, 0, 0], [0, np.pi / 2, 0]).tolist(),
        }
    )
    metadata = {
        "schema_version": 1,
        "source_sha256": file_sha256(source / "source.step"),
        "occt_length_unit_mm": 1,
        "coordinate_contract": "BRep mm; occurrences SI, T_cad_part",
        "definitions": definitions,
        "occurrences": occurrences,
        "files": files,
    }
    (source / "cad.json").write_text(json.dumps(metadata))
    output = root / "library"
    report = tessellate_assembly(source, output)
    return source, output, report


def test_every_leaf_and_instance_is_retained_when_some_sources_cannot_be_meshed(library):
    source, output, report = library
    metadata = json.loads((source / "cad.json").read_text())
    assert report["complete"] and not report["fully_meshed"]
    assert report["occurrences"] == metadata["occurrences"]
    assert [r["status"] for r in report["parts"].values()] == [
        "meshed",
        "surface_only",
        "invalid_brep",
    ]
    first = report["parts"]["0:1:1:1"]
    manifest = json.loads((output / first["directory"] / "mesh.json").read_text())
    assert manifest["solids"][0]["signed_mesh_volume_m3"] == pytest.approx(6e-6, abs=1e-15)
    with np.load(output / first["directory"] / manifest["solids"][0]["file"]) as arrays:
        np.testing.assert_allclose(
            np.ptp(arrays["vertices"], axis=0), [0.01, 0.02, 0.03], atol=1e-15
        )
    with pytest.raises(FileExistsError):
        tessellate_assembly(source, output)


def test_identical_brep_cache_reuses_bytes_across_derived_assembly_manifests(
    library, tmp_path, monkeypatch
):
    source, output, original = library
    derived = tmp_path / "derived"
    shutil.copytree(source, derived)
    cad = json.loads((derived / "cad.json").read_text())
    cad["derivation"] = {"parent_manifest_sha256": file_sha256(source / "cad.json")}
    (derived / "cad.json").write_text(json.dumps(cad))

    def unexpected(*args, **kwargs):
        raise AssertionError("Exact BRep cache should avoid another native process")

    monkeypatch.setattr("alohamini_mujoco.cad_library.subprocess.run", unexpected)
    target = tmp_path / "reused"
    report = tessellate_assembly(derived, target, reuse_library=output)
    first = report["parts"]["0:1:1:1"]
    assert (
        first["reused"] and first["origin_cad_manifest_sha256"] == original["cad_manifest_sha256"]
    )
    assert report["cad_manifest_sha256"] == file_sha256(derived / "cad.json")
    assert report["cad_manifest_sha256"] != first["origin_cad_manifest_sha256"]
    assert (target / first["directory"] / "mesh.json").read_bytes() == (
        output / first["directory"] / "mesh.json"
    ).read_bytes()


@pytest.mark.parametrize("mutation", ["corrupt_solid", "path_escape", "different_settings"])
def test_invalid_or_incompatible_cache_never_becomes_a_complete_library(
    library, tmp_path, mutation
):
    source, output, _ = library
    cache = tmp_path / "cache"
    shutil.copytree(output, cache)
    catalog = json.loads((cache / "library.json").read_text())
    row = catalog["parts"]["0:1:1:1"]
    if mutation == "corrupt_solid":
        (cache / row["directory"] / "solid_0000.npz").write_bytes(b"corrupt")
    elif mutation == "path_escape":
        row["directory"] = "../elsewhere"
        (cache / "library.json").write_text(json.dumps(catalog))
    settings = {"linear_deflection_m": 1e-4} if mutation == "different_settings" else {}
    target = tmp_path / "failed"
    with pytest.raises(ValueError):
        tessellate_assembly(source, target, reuse_library=cache, **settings)
    assert not json.loads((target / "library.json").read_text())["complete"]


def test_native_worker_failure_is_recorded_without_omitting_remaining_parts(
    library, tmp_path, monkeypatch
):
    source, _, _ = library
    monkeypatch.setattr(
        "alohamini_mujoco.cad_library.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=-9, stderr="worker killed", stdout=""),
    )
    report = tessellate_assembly(source, tmp_path / "failure")
    assert report["complete"] and not report["fully_meshed"]
    assert [r["status"] for r in report["parts"].values()] == [
        "worker_failed",
        "surface_only",
        "invalid_brep",
    ]
    assert report["parts"]["0:1:1:1"]["exit_code"] == -9


@pytest.mark.parametrize("mutation", ["source_changed", "missing_instance", "reflection"])
def test_preflight_rejects_changed_sources_or_incomplete_coordinate_contract(
    library, tmp_path, mutation
):
    source, _, _ = library
    broken = tmp_path / "cad"
    shutil.copytree(source, broken)
    cad = json.loads((broken / "cad.json").read_text())
    if mutation == "source_changed":
        (broken / "parts/1.brep").write_bytes(b"changed")
    elif mutation == "missing_instance":
        cad["occurrences"] = [o for o in cad["occurrences"] if o["definition"] != "0:1:1:2"]
    else:
        cad["occurrences"][0]["T_cad_part"] = np.diag([-1, 1, 1, 1]).tolist()
    (broken / "cad.json").write_text(json.dumps(cad))
    target = tmp_path / "result"
    with pytest.raises(ValueError):
        tessellate_assembly(broken, target)
    assert not target.exists()
