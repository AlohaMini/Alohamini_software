"""Portable MJCF model snapshots with self-contained meshes and textures."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from .compilation import load_xml


def model_xml(robot):
    xml = ET.fromstring(robot.xml)
    xml.find("compiler").set("meshdir", "assets")
    xml.find("compiler").set("texturedir", "assets")
    ET.indent(xml)
    return ET.tostring(xml, encoding="unicode") + "\n"


def export_robot(robot, report, output):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        assets = stage / "assets"
        assets.mkdir()
        for name, contents in robot.assets.items():
            (assets / name).write_bytes(contents)
        xml = model_xml(robot)
        (stage / "model.xml").write_text(xml)
        manifest = {
            **robot.manifest,
            "model_xml_sha256": hashlib.sha256(xml.encode()).hexdigest(),
            "derived_asset_sha256": {
                n: hashlib.sha256(b).hexdigest() for n, b in robot.assets.items()
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Do not publish a package whose relative mesh references cannot load.
        reloaded = load_xml(stage / "model.xml")
        if (reloaded.nq, reloaded.ngeom) != (robot.model.nq, robot.model.ngeom):
            raise RuntimeError("Export changed model topology")
        if output.exists():
            raise FileExistsError(f"Output appeared during export: {output}")
        os.rename(stage, output)
    return output
