import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from alohamini_validation.validate_assets import (
    UrdfKinematics,
    main,
    validate_collision_baseline,
    validate_derived_views,
    validate_fk,
    validate_physical_assets,
)
from ament_index_python.packages import get_package_share_directory

from alohamini.model import get_robot_model


def test_authoritative_offline_assets():
    main()


@pytest.mark.parametrize("name", ["collision", "planning", "kinematic"])
def test_fk_drift_fails_instead_of_refreshing_the_baseline(name):
    description = get_robot_model("alohamini2pro")
    model = UrdfKinematics(description.description_tree(name))
    joint = model.joint_by_child["left_tcp"]
    joint.find("origin").set("xyz", "0 0 0.1")
    validation = Path(get_package_share_directory("alohamini_validation"))
    with pytest.raises(AssertionError, match="FK drift"):
        validate_fk(model, description.directory, validation)


def test_cycle_in_model_is_rejected_without_hanging():
    model = UrdfKinematics(get_robot_model("alohamini2pro").description_path("collision"))
    model.joint_by_child["left_tcp"].find("parent").set("link", "left_tcp")
    with pytest.raises(AssertionError, match="cycle"):
        model.chain("base_link", "left_tcp")


@pytest.mark.parametrize("name", ["collision", "semantic"])
def test_physical_description_checksum_drift_is_rejected(tmp_path, name):
    source = get_robot_model("alohamini2pro")
    files = {}
    for kind in ("collision", "semantic"):
        original = source.description_path(kind)
        relative = original.relative_to(source.directory)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        contents = original.read_bytes()
        files[relative.as_posix()] = hashlib.sha256(contents).hexdigest()
        target.write_bytes(contents + (b"\n" if kind == name else b""))
    # Checksum rejection happens before mesh or semantic interpretation. Keep the
    # fixture small rather than copying all of the prepared collision geometry.
    manifest = {
        "format": "alohamini-physical-model-v1",
        "robot_model": source.model_id,
        "files": files,
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config/physical_model.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Physical model asset checksum mismatch"):
        validate_physical_assets(replace(source, directory=tmp_path))


def test_planning_view_cannot_drop_the_authoritative_collision_geometry():
    source = get_robot_model("alohamini2pro")
    views = dict(source.description_views)
    views["planning"] = replace(views["planning"], geometry=False)
    with pytest.raises(AssertionError, match="differs from physical source"):
        validate_derived_views(replace(source, description_views=views))


def test_collision_drift_cannot_silently_refresh_the_reviewed_baseline(tmp_path):
    source = get_robot_model("alohamini2pro")
    model = UrdfKinematics(source.description_tree("collision"))
    link = model.root.find("link[@name='left_Upper_Arm']")
    link.remove(link.find("collision"))
    for name in ("collision", "semantic"):
        relative = source.description_path(name).relative_to(source.directory)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if name == "collision":
            target.write_text(ET.tostring(model.root, encoding="unicode"))
        else:
            target.write_bytes(source.description_path(name).read_bytes())
    validation = Path(get_package_share_directory("alohamini_validation"))
    with pytest.raises(AssertionError, match="Collision baseline source changed"):
        validate_collision_baseline(replace(source, directory=tmp_path), validation)
