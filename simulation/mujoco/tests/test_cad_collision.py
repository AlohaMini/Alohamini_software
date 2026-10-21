import json
import struct
import xml.etree.ElementTree as ET
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np
import pytest
from alohamini_mujoco.cad import file_sha256
from alohamini_mujoco.cad_collision import _parameters, build_cad_collision
from alohamini_mujoco.collision import load_bundle, merge_bundles
from alohamini_mujoco.registration import register_part

pytest.importorskip("manifold3d")
pytest.importorskip("coacd")


@pytest.fixture
def assets(tmp_path):
    vertices = np.array([[0, 0, 0], [0.03125, 0, 0], [0, 0.0625, 0], [0, 0, 0.125]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    data = b"\0" * 80 + struct.pack("<I", len(faces))
    for triangle in vertices[faces]:
        data += struct.pack("<12fH", *([0, 0, 0] + triangle.ravel().tolist()), 0)
    (tmp_path / "part.stl").write_bytes(data)
    link = ET.fromstring(
        '<link><visual><geometry><mesh filename="part.stl"/></geometry></visual>'
        '<collision><geometry><mesh filename="part.stl"/></geometry></collision></link>'
    )
    description = SimpleNamespace(
        robot_model=SimpleNamespace(directory=tmp_path, model_id="fixture"),
        links={"part": link},
        mesh_path=lambda _: tmp_path / "part.stl",
        source_bytes={},
    )
    source = tmp_path / "mesh"
    source.mkdir()
    local = vertices - [0.125, 0.25, 0.5]
    np.savez(source / "solid.npz", vertices=local, faces=faces)
    metadata = {
        "schema_version": 1,
        "source_cad_sha256": "fixture-step",
        "units": "m",
        "coordinates": "CAD part definition frame, float64",
        "definition": "fixture-part",
        "name": "tetrahedron",
        "brep_sha256": "fixture-brep",
        "linear_deflection_m": 5e-5,
        "angular_deflection_rad": 0.1,
        "solids": [{"file": "solid.npz", "sha256": file_sha256(source / "solid.npz")}],
    }
    (source / "mesh.json").write_text(json.dumps(metadata))
    pose = np.eye(4)
    pose[:3, 3] = [0.125, 0.25, 0.5]
    registration = tmp_path / "registration"
    assert register_part(
        description,
        source,
        registration,
        link="part",
        component_index=0,
        T_link_part=pose,
        tolerance_m=1e-7,
    )["passed"]
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "hull.npz", vertices=local, faces=faces)
    manifest = {
        "schema_version": 1,
        "units": "m",
        "coordinates": metadata["coordinates"],
        "source_manifest_sha256": file_sha256(source / "mesh.json"),
        "source_mesh_sha256": metadata["solids"][0]["sha256"],
        "coacd_version": version("coacd"),
        "parameters": _parameters(0.0005, 0),
        "hulls": [{"file": "hull.npz", "sha256": file_sha256(cache / "hull.npz")}],
    }
    (cache / "decomposition.json").write_text(json.dumps(manifest))
    return description, source, registration, cache


def build(assets, output):
    description, source, registration, cache = assets
    return build_cad_collision(
        description, source, [registration], output, tolerance_m=0.0005, reuse_decomposition=cache
    )


def test_complete_cad_registration_exports_transport_bound_certificates_and_merges(
    assets, tmp_path
):
    output = tmp_path / "candidate"
    assert build(assets, output)["passed"]
    description = assets[0]
    loaded, _ = load_bundle(description, output)
    assert len(loaded["part"]) == 1
    meta = json.loads((output / "collision.json").read_text())
    certificate = meta["links"]["part"]["cad_geometry"]
    assert certificate["passed"]
    assert certificate["collision_piece_sha256"] == [meta["links"]["part"]["pieces"][0]["sha256"]]
    description.links["other"] = description.links["part"]
    other = tmp_path / "other"
    other.mkdir()
    record = json.loads(json.dumps(meta["links"]["part"]))
    record.pop("cad_geometry")
    for piece in record["pieces"]:
        target = other / piece["path"]
        target.parent.mkdir(exist_ok=True)
        target.write_bytes((output / piece["path"]).read_bytes())
    (other / "collision.json").write_text(json.dumps({**meta, "links": {"other": record}}))
    merged = merge_bundles(description, [output, other], tmp_path / "merged")
    assert set(load_bundle(description, merged)[0]) == {"part", "other"}


def test_native_decomposition_and_exact_cache_reuse(assets, tmp_path):
    description, source, registration, _ = assets
    generated = tmp_path / "generated"
    assert build_cad_collision(
        description, source, [registration], generated, tolerance_m=0.0005, merge_verified=True
    )["passed"]
    reused = tmp_path / "reused"
    assert build_cad_collision(
        description,
        source,
        [registration],
        reused,
        tolerance_m=0.0005,
        merge_verified=True,
        reuse_decomposition=generated / "decomposition",
    )["passed"]
    first, second = (json.loads((p / "collision.json").read_text()) for p in (generated, reused))
    assert first["links"]["part"]["pieces"] == second["links"]["part"]["pieces"]
    assert first["links"]["part"]["cad_geometry"]["simplification"]["passed"]


def test_wrong_cached_geometry_cannot_publish_a_collision_asset(assets, tmp_path):
    cache = assets[3]
    with np.load(cache / "hull.npz") as archive:
        vertices, faces = archive["vertices"], archive["faces"]
    np.savez(cache / "hull.npz", vertices=vertices + [1, 0, 0], faces=faces)
    path = cache / "decomposition.json"
    manifest = json.loads(path.read_text())
    manifest["hulls"][0]["sha256"] = file_sha256(cache / "hull.npz")
    path.write_text(json.dumps(manifest))
    output = tmp_path / "rejected"
    assert not build(assets, output)["passed"]
    assert (output / "qualification.json").exists()
    assert not (output / "collision.json").exists()


def test_partial_link_registration_and_changed_cache_are_rejected(assets, tmp_path):
    path = assets[2] / "registration.json"
    report = json.loads(path.read_text())
    report["target"]["component_count"] = 2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="complete-link registration"):
        build(assets, tmp_path / "partial")
    report["target"]["component_count"] = 1
    path.write_text(json.dumps(report))
    (assets[3] / "hull.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        build(assets, tmp_path / "changed")


def test_registration_cannot_move_a_link_by_editing_only_the_declared_transform(assets, tmp_path):
    path = assets[2] / "registration.json"
    report = json.loads(path.read_text())
    report["T_link_part"][0][3] += 0.1
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="transform differs"):
        build(assets, tmp_path / "moved")


@pytest.mark.parametrize("tamper", ["failed", "digest", "link"])
def test_loading_rejects_failed_or_detached_cad_certificates(assets, tmp_path, tamper):
    output = tmp_path / "candidate"
    assert build(assets, output)["passed"]
    path = output / "collision.json"
    manifest = json.loads(path.read_text())
    qualification = manifest["links"]["part"]["cad_geometry"]
    if tamper == "failed":
        qualification["certificate"]["checks"]["collision_inside_expanded_source"]["passed"] = False
    elif tamper == "digest":
        qualification["collision_piece_sha256"] = ["0" * 64]
    else:
        qualification["registration"]["link"] = "another_link"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="certificate"):
        load_bundle(assets[0], output)
