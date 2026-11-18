"""Derived URDF views contain no independent physical robot definitions."""

import math
from copy import deepcopy


def root_link(tree):
    links = {node.attrib["name"] for node in tree.findall("link")}
    children = {node.find("child").attrib["link"] for node in tree.findall("joint")}
    roots = links - children
    if len(roots) != 1:
        raise ValueError("Robot description must have exactly one root link")
    return roots.pop()


def attach_planar_base(tree, overlay):
    """Attach an empty X/Y/yaw chain without replacing any physical element."""
    links, joints = overlay.findall("link"), overlay.findall("joint")
    names = {link.attrib["name"] for link in links}
    existing = {link.attrib["name"] for link in tree.findall("link")}
    existing_joints = {joint.attrib["name"] for joint in tree.findall("joint")}
    if overlay.tag != "robot" or len(links) != 3 or len(names) != 3 or len(joints) != 3:
        raise ValueError("Planar-base overlay must define three empty links and three joints")
    if names & existing or any(list(link) for link in links):
        raise ValueError("Planar-base links must be empty and separate from physical links")
    if len({j.attrib["name"] for j in joints}) != 3 or any(
        j.attrib["name"] in existing_joints for j in joints
    ):
        raise ValueError("Planar-base joint names must be unique")
    parent = root_link(overlay)
    visited = {parent}
    for index, (joint, kind, axis) in enumerate(
        zip(
            joints,
            ("prismatic", "prismatic", "revolute"),
            ("1 0 0", "0 1 0", "0 0 1"),
            strict=True,
        )
    ):
        if (
            joint.attrib.get("type") != kind
            or joint.find("parent").attrib["link"] != parent
            or joint.find("axis").attrib.get("xyz") != axis
            or joint.find("origin") is not None
            or joint.find("mimic") is not None
        ):
            raise ValueError("Planar-base overlay must be an identity-origin X/Y/yaw chain")
        limit = joint.find("limit")
        values = [float(limit.attrib[k]) for k in ("lower", "upper", "effort", "velocity")]
        if not all(math.isfinite(v) for v in values) or not (
            values[0] <= 0 <= values[1]
            and values[0] < values[1]
            and values[2] > 0
            and values[3] > 0
        ):
            raise ValueError("Planar-base limits must be finite and include zero")
        parent = joint.find("child").attrib["link"]
        if index < 2:
            if parent not in names or parent in visited:
                raise ValueError("Planar-base overlay must be a connected, acyclic chain")
            visited.add(parent)
    if parent != root_link(tree):
        raise ValueError("Planar-base overlay must terminate at the physical root link")
    tree.extend(deepcopy(links + joints))
    root_link(tree)
    return tree


def derive_tree(tree, view, overlay=None):
    if view.exclude_joints:
        joints = {joint.attrib["name"]: joint for joint in tree.findall("joint")}
        missing = set(view.exclude_joints) - joints.keys()
        if missing:
            raise ValueError(f"Unknown excluded joints: {sorted(missing)}")
        removed = set()
        pending = [joints[name].find("child").attrib["link"] for name in view.exclude_joints]
        while pending:
            parent = pending.pop()
            if parent in removed:
                continue
            removed.add(parent)
            pending.extend(
                joint.find("child").attrib["link"]
                for joint in joints.values()
                if joint.find("parent").attrib["link"] == parent
            )
        for link in tree.findall("link"):
            if link.attrib["name"] in removed:
                tree.remove(link)
        for joint in tree.findall("joint"):
            if joint.find("child").attrib["link"] in removed:
                tree.remove(joint)
    if view.root_link is not None:
        links = {link.attrib["name"]: link for link in tree.findall("link")}
        if view.root_link not in links:
            raise ValueError(f"Unknown derived-description root: {view.root_link}")
        retained, pending = set(), [view.root_link]
        while pending:
            parent = pending.pop()
            if parent in retained:
                raise ValueError("Cycle in derived-description subtree")
            retained.add(parent)
            pending.extend(
                j.find("child").attrib["link"]
                for j in tree.findall("joint")
                if j.find("parent").attrib["link"] == parent
            )
        for link in tree.findall("link"):
            if link.attrib["name"] not in retained:
                tree.remove(link)
        for joint in tree.findall("joint"):
            if joint.find("parent").attrib["link"] not in retained:
                tree.remove(joint)
    if overlay is not None:
        attach_planar_base(tree, overlay)
    if not view.geometry:
        for link in tree.findall("link"):
            for node in list(link):
                if node.tag in ("visual", "collision", "inertial"):
                    link.remove(node)
        for material in tree.findall("material"):
            tree.remove(material)
    return tree
