"""Revisioned map artifacts. Incomplete saves never become selectable maps."""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from alohamini.model import get_robot_model
from alohamini.paths import WorkspacePaths


class MapRevision:
    def __init__(self, name, revision=None, *, model="alohamini2edu", workspace=None):
        self.model = get_robot_model(model)
        if revision is None:
            revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.target = (workspace or WorkspacePaths()).map_revision(name, revision)
        self.target.parent.mkdir(parents=True, exist_ok=True)
        if self.target.exists():
            raise FileExistsError(self.target)
        self.staging = self.target.parent / f".incomplete-{revision}-{uuid.uuid4().hex}"
        self.staging.mkdir()
        self.manifest = dict(
            schema_version=1,
            name=name,
            revision=revision,
            status="incomplete",
            robot_model=model,
            model_asset_revision=self.model.asset_revision,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._write()

    def _write(self):
        (self.staging / "manifest.json").write_text(json.dumps(self.manifest, indent=2) + "\n")

    def fail(self, reason):
        self.manifest.update(status="failed", error=str(reason))
        self._write()

    def finish(self, *, pose_graph=True, metadata=None):
        yaml_path = self.staging / "map.yaml"
        data = yaml.safe_load(yaml_path.read_text())
        image = Path(data["image"])
        image = image if image.is_absolute() else self.staging / image
        if image.resolve().parent != self.staging.resolve() or not image.is_file():
            raise ValueError("Saved map image must be in the revision directory")
        # Nav2 may write an absolute staging filename; make the revision relocatable.
        data["image"] = image.name
        yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
        required = [yaml_path, image]
        if pose_graph:
            required += [self.staging / "slam.posegraph", self.staging / "slam.data"]
        if any(not p.is_file() or p.stat().st_size == 0 for p in required):
            raise ValueError("Map save did not produce all requested artifacts on this machine")
        self.manifest.update(
            status="complete",
            grid="map.yaml",
            pose_graph="slam" if pose_graph else None,
            metadata=metadata or {},
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in required},
        )
        self._write()
        # Never replace another revision, including one created concurrently.
        # A per-destination lock serializes saves while preserving failed staging data.
        lock = self.target.parent / f".{self.target.name}.lock"
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            if self.target.exists():
                raise FileExistsError(self.target)
            self.staging.rename(self.target)
        finally:
            os.close(fd)
            lock.unlink()
        return self.target


def validate_map_revision(path, model):
    """Validate a managed map; external Nav2 YAMLs remain usable explicitly."""
    path = Path(path).expanduser().resolve()
    if path.parent.name.startswith(".incomplete-"):
        raise ValueError("Map revision is incomplete")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("robot_model") != model:
        raise ValueError("Map revision is incomplete or belongs to another robot model")
    if path.name != manifest.get("grid"):
        raise ValueError("Map YAML differs from the recorded grid artifact")
    for filename, checksum in manifest["files"].items():
        if Path(filename).name != filename:
            raise ValueError("Invalid map manifest filename")
        if hashlib.sha256((path.parent / filename).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Map artifact checksum mismatch: {filename}")
