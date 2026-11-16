"""Recognize exact prepared inputs already incorporated into canonical assets."""

import hashlib
import json
from pathlib import Path


def embedded_source(prepared, kind, path):
    """Return provenance only for byte-identical embedded inputs, never a fallback."""
    if prepared is None:
        return None
    path = Path(path).expanduser().resolve()
    contents = path.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if digest != prepared.get(f"{kind}_bundle_sha256"):
        return None
    definition = json.loads(contents)
    if kind == "collision":
        for link in definition["links"].values():
            for piece in link["pieces"]:
                if "path" not in piece:
                    continue
                resource = (path.parent / piece["path"]).resolve()
                if not resource.is_relative_to(path.parent) or (
                    hashlib.sha256(resource.read_bytes()).hexdigest() != piece["sha256"]
                ):
                    raise ValueError("Embedded collision source mesh checksum mismatch")
    return {"manifest_sha256": digest, "definition": definition, "embedded": True}
