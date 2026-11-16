"""Previously integrated bundles must not override the canonical model twice."""

import hashlib
import json

import pytest
from alohamini_mujoco.embedded import embedded_source
from alohamini_mujoco.model import PARKED_JOINTS, ModelProfile, build_model


def test_only_exact_embedded_inertial_input_is_recognized(tmp_path):
    path = tmp_path / "inertia.json"
    path.write_text('{"fixture": "prepared numeric properties"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prepared = {"inertial_bundle_sha256": digest}
    assert embedded_source(prepared, "inertial", path)["embedded"]
    assert embedded_source(None, "inertial", path) is None
    path.write_text('{"fixture": "different numeric properties"}')
    assert embedded_source(prepared, "inertial", path) is None


@pytest.mark.parametrize("damage", [None, "bytes", "escape"])
def test_embedded_collision_input_still_verifies_its_external_meshes(tmp_path, damage):
    mesh = tmp_path / "piece.msh"
    mesh.write_bytes(b"original prepared piece")
    piece = {"path": mesh.name, "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest()}
    if damage == "escape":
        piece["path"] = "../piece.msh"
    path = tmp_path / "collision.json"
    path.write_text(json.dumps({"links": {"fixture": {"pieces": [piece]}}}))
    prepared = {"collision_bundle_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if damage == "bytes":
        mesh.write_bytes(b"damaged piece")
    if damage is None:
        assert embedded_source(prepared, "collision", path)["embedded"]
    else:
        with pytest.raises(ValueError, match="checksum"):
            embedded_source(prepared, "collision", path)


def test_explicit_virtual_base_coordinates_select_a_generated_view():
    parked = {joint: 0.0 for joint in PARKED_JOINTS}
    parked["root_x_axis_joint"] = 0.1
    model = build_model(profile=ModelProfile(locked_positions=parked))
    assert model.description.name == "planning"
    assert model.description.root == "root"
    assert model.data().body("base_link").xpos[0] == pytest.approx(0.1)
    assert len(model.description.tree.findall(".//collision")) == 1115
