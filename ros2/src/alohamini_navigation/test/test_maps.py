import json

import pytest
import yaml
from alohamini_navigation.maps import MapRevision, validate_map_revision

from alohamini.paths import WorkspacePaths


def artifacts(revision):
    (revision.staging / "map.pgm").write_bytes(b"P5\n1 1\n255\n\xff")
    (revision.staging / "map.yaml").write_text(
        yaml.safe_dump(
            {
                "image": str(revision.staging / "map.pgm"),
                "resolution": 0.05,
                "origin": [0.0, 0.0, 0.0],
                "negate": 0,
                "occupied_thresh": 0.65,
                "free_thresh": 0.25,
            }
        )
    )


def test_incomplete_posegraph_never_becomes_a_loadable_map(tmp_path):
    revision = MapRevision("office", "v1", workspace=WorkspacePaths(tmp_path))
    artifacts(revision)
    with pytest.raises(ValueError, match="artifacts"):
        revision.finish()
    assert not revision.target.exists()
    with pytest.raises(ValueError, match="incomplete"):
        validate_map_revision(revision.staging / "map.yaml", "alohamini2edu")
    # A crash after manifest completion but before the final rename is still incomplete.
    revision.manifest["status"] = "complete"
    revision._write()
    with pytest.raises(ValueError, match="incomplete"):
        validate_map_revision(revision.staging / "map.yaml", "alohamini2edu")


def test_save_relocates_image_validates_checksums_and_refuses_overwrite(tmp_path):
    workspace = WorkspacePaths(tmp_path)
    revision = MapRevision("office", "v1", workspace=workspace)
    artifacts(revision)
    for name in ("slam.data", "slam.posegraph"):
        (revision.staging / name).write_bytes(b"graph")
    target = revision.finish()
    assert not revision.staging.exists()
    assert yaml.safe_load((target / "map.yaml").read_text())["image"] == "map.pgm"
    validate_map_revision(target / "map.yaml", "alohamini2edu")
    with pytest.raises(ValueError, match="another robot"):
        validate_map_revision(target / "map.yaml", "alohamini2pro")
    with pytest.raises(FileExistsError):
        MapRevision("office", "v1", workspace=workspace)
    (target / "map.pgm").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        validate_map_revision(target / "map.yaml", "alohamini2edu")


def test_failed_save_preserves_partial_files(tmp_path):
    revision = MapRevision("office", "v1", workspace=WorkspacePaths(tmp_path))
    artifacts(revision)
    revision.fail("service timeout")
    assert (revision.staging / "map.pgm").is_file()
    assert json.loads((revision.staging / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("name, revision", [("../bad", "v1"), ("ok", "../bad"), ("ok", "")])
def test_names_cannot_escape_workspace(tmp_path, name, revision):
    with pytest.raises(ValueError):
        MapRevision(name, revision, workspace=WorkspacePaths(tmp_path))
