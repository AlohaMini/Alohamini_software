"""Refresh EDU Pro mechanics and its nominal C1 collision box."""

import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from alohamini.simulation.frames import rotation_rpy

ROOT = Path(__file__).resolve().parents[1]
PRO = ROOT / "models/alohamini2pro"
EDU = ROOT / "models/alohamini2edu"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    # These directories contain nominal assets only; retain EDU sensor and
    # navigation configuration and every installed-device calibration.
    for directory in ("urdf", "srdf", "meshes", "config/kinematics"):
        shutil.rmtree(EDU / directory)
    for source in PRO.rglob("*"):
        if not source.is_file() or source.name in ("model.json", "provenance.json"):
            continue
        target = EDU / str(source.relative_to(PRO)).replace("alohamini2pro", "alohamini2edu")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        try:
            content = content.decode("utf-8").replace("alohamini2pro", "alohamini2edu").encode()
        except UnicodeDecodeError:
            pass
        target.write_bytes(content)
    shutil.copyfile(PRO / "provenance.json", EDU / "source_provenance.json")

    profile = yaml.safe_load((EDU / "config/sensors.yaml").read_text())["lidar"]
    housing = json.loads((EDU / "config/lidar_collision.json").read_text())
    tree = ET.parse(EDU / "urdf/alohamini2edu.urdf")
    robot = tree.getroot()
    link = ET.SubElement(robot, "link", name=housing["link"])
    size = housing["size_m"]
    for kind in ("visual", "collision"):
        item = ET.SubElement(link, kind)
        geometry = ET.SubElement(item, "geometry")
        ET.SubElement(geometry, "box", size=" ".join(map(str, size)))
        if kind == "visual":
            material = ET.SubElement(item, "material", name="lidar_black")
            ET.SubElement(material, "color", rgba="0.08 0.08 0.08 1")
    inertial = ET.SubElement(link, "inertial")
    mass = housing["mass_kg"]
    ET.SubElement(inertial, "mass", value=str(mass))
    x, y, z = size
    ET.SubElement(
        inertial,
        "inertia",
        ixx=str(mass * (y * y + z * z) / 12),
        iyy=str(mass * (x * x + z * z) / 12),
        izz=str(mass * (x * x + y * y) / 12),
        ixy="0",
        ixz="0",
        iyz="0",
    )
    joint = ET.SubElement(robot, "joint", name="lidar_housing_joint", type="fixed")
    ET.SubElement(joint, "parent", link=profile["parent_frame"])
    ET.SubElement(joint, "child", link=housing["link"])
    center = np.asarray(profile["xyz"]) + rotation_rpy(profile["rpy"]) @ np.asarray(
        housing["center_in_reference_m"]
    )
    ET.SubElement(
        joint, "origin", xyz=" ".join(map(str, center)), rpy=" ".join(map(str, profile["rpy"]))
    )
    ET.indent(tree, space="  ")
    tree.write(EDU / "urdf/alohamini2edu.urdf", encoding="unicode")

    srdf_path = EDU / "srdf/alohamini2edu.srdf"
    srdf = ET.parse(srdf_path)
    for parent in housing["internal_collision_links"]:
        ET.SubElement(
            srdf.getroot(),
            "disable_collisions",
            link1=parent,
            link2=housing["link"],
            reason="Adjacent",
        )
    ET.indent(srdf, space="  ")
    srdf.write(srdf_path, encoding="unicode")

    data = json.loads((PRO / "model.json").read_text().replace("alohamini2pro", "alohamini2edu"))
    data["asset_revision"] = max(7, data["asset_revision"] + 1)
    data["description_views"].update(
        navigation={"source": "collision"},
        base={"source": "collision", "exclude_joints": ["vertical_move"]},
    )
    save_json(EDU / "model.json", data)
    for path in (EDU / "config/kinematics").glob("*.yaml"):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "source_urdf_sha256" in data:
            data["source_urdf_sha256"] = sha(path.parent / data["source_urdf"])
            path.write_text(yaml.safe_dump(data, sort_keys=False))
    path = EDU / "config/physical_model.json"
    data = json.loads(path.read_text())
    data["files"] = {name: sha(EDU / name) for name in data["files"]}
    save_json(path, data)
    path = EDU / "provenance.json"
    data = json.loads(path.read_text())
    data.update(
        base_revision=json.loads((PRO / "model.json").read_text())["asset_revision"],
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "main"], cwd=ROOT, text=True
        ).strip(),
    )
    data["files"] = [
        {"path": str(file.relative_to(EDU)), "sha256": sha(file)}
        for file in sorted(EDU.rglob("*"))
        if file.is_file() and file.name not in ("provenance.json", "source_provenance.json")
    ]
    save_json(path, data)


if __name__ == "__main__":
    main()
