"""Isolated upstream compiler. Run with a LIBERO/robosuite Python, never in the Host."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import random
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source, output, suite_name, indices):
    import yaml

    base = source / "libero/libero"
    configuration = output / "configuration"
    configuration.mkdir()
    (configuration / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(base),
                "bddl_files": str(base / "bddl_files"),
                "init_states": str(base / "init_files"),
                "assets": str(base / "assets"),
                "datasets": str(output / "unused_datasets"),
            }
        )
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(configuration)
    # The official repository uses a namespace outer package. Bind that package
    # explicitly in this subprocess, avoiding an unrelated installed hf_libero.
    package = types.ModuleType("libero")
    package.__path__ = [str(source / "libero")]
    package.__spec__ = importlib.machinery.ModuleSpec("libero", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["libero"] = package
    import torch
    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    for module in (benchmark, sys.modules[ControlEnv.__module__]):
        if not Path(module.__file__).resolve().is_relative_to(source):
            raise RuntimeError("LIBERO import escaped the selected source checkout")
    suite = benchmark.get_benchmark(suite_name)(task_order_index=0)
    assets = output / "assets"
    assets.mkdir()
    tasks = []
    for index in indices:
        task = suite.get_task(index)
        bddl = Path(suite.get_task_bddl_file_path(index))
        bank = base / "init_files" / task.problem_folder / task.init_states_file
        # Historical NumPy arrays from the pinned, clean Git checkout only.
        states = torch.load(bank, map_location="cpu", weights_only=False)
        np.random.seed(0)
        random.seed(0)
        environment = ControlEnv(
            bddl_file_name=str(bddl),
            use_camera_obs=False,
            has_offscreen_renderer=False,
            has_renderer=False,
        )
        try:
            core = environment.env
            core.seed(0)
            core.reset()
            model = core.sim.model._model
            data = core.sim.data._data
            tree = ET.fromstring(core.sim.model.get_xml())
            world = tree.find("worldbody")
            removed = []
            for node in list(world):
                if node.tag == "body" and node.get("name", "").startswith(
                    ("robot", "mount", "gripper")
                ):
                    removed.append(node.get("name"))
                    world.remove(node)
            if not removed:
                raise RuntimeError("No upstream robot identified; refusing a two-robot scene")
            retained_bodies = {node.get("name") for node in world.iter("body")}
            retained_joints = {node.get("name") for node in world.iter("joint")}
            if any(n.startswith(("robot", "gripper")) for n in retained_joints):
                raise RuntimeError("Upstream robot joints survived scene extraction")
            objects = {}
            for name, obj in {**core.objects_dict, **core.fixtures_dict}.items():
                if obj.root_body not in retained_bodies:
                    raise RuntimeError(f"Missing object body: {name}")
                body_id = model.body(obj.root_body).id
                objects[name] = {
                    "root_body": obj.root_body,
                    "contact_geoms": obj.contact_geoms,
                    "joints": obj.joints,
                    "is_fixture": name in core.fixtures_dict,
                    "mass_kg": float(model.body_mass[body_id]),
                }
            # Freeze inferred upstream mass/COM/inertia explicitly. The robot
            # compiler intentionally never infers inertias from collision geoms.
            for body in world.iter("body"):
                bid = model.body(body.get("name")).id
                for inertial in list(body.findall("inertial")):
                    body.remove(inertial)
                if model.body_mass[bid] > 0:
                    ET.SubElement(
                        body,
                        "inertial",
                        mass=str(float(model.body_mass[bid])),
                        pos=" ".join(map(str, model.body_ipos[bid])),
                        quat=" ".join(map(str, model.body_iquat[bid])),
                        diaginertia=" ".join(map(str, model.body_inertia[bid])),
                    )
            for geom in world.iter("geom"):
                gid = model.geom(geom.get("name")).id
                geom.set(
                    "group", "3" if model.geom_contype[gid] or model.geom_conaffinity[gid] else "2"
                )
            joint_bank = []
            addresses = {}
            for name in sorted(retained_joints):
                address = core.sim.model.get_joint_qpos_addr(name)
                addresses[name] = slice(*address) if isinstance(address, tuple) else address
            for state in states:
                core.sim.set_state_from_flattened(np.asarray(state))
                core.sim.forward()
                joint_bank.append(
                    {
                        name: np.atleast_1d(data.qpos[addresses[name]]).astype(float).tolist()
                        for name in sorted(retained_joints)
                    }
                )
            scene = ET.Element("mujoco", model=f"{suite_name}_{index}")
            default = tree.find("default")
            if default is not None:
                default.set("class", "libero_defaults")
                scene.append(default)
                for node in world:
                    if node.tag == "body":
                        node.set("childclass", node.get("childclass", "libero_defaults"))
                    elif node.tag in ("geom", "site", "camera", "light"):
                        node.set("class", node.get("class", "libero_defaults"))
            scene.append(world)
            # Keep the transitive asset closure, preserving every source triangle
            # and texture byte. Unused Panda assets are not part of this scene.
            upstream_assets = {n.get("name"): n for n in tree.find("asset")}
            requested = set()
            for node in scene.iter():
                for key in ("mesh", "material", "texture", "hfield"):
                    if node.get(key):
                        requested.add(node.get(key))
            selected = set()
            while requested - selected:
                name = sorted(requested - selected)[0]
                node = upstream_assets[name]
                selected.add(name)
                for key in ("texture", "material"):
                    if node.get(key):
                        requested.add(node.get(key))
            asset_node = ET.SubElement(scene, "asset")
            sources = {}
            for name in sorted(selected):
                node = upstream_assets[name]
                if node.get("file"):
                    path = Path(node.get("file")).resolve()
                    contents = path.read_bytes()
                    fingerprint = digest(path)
                    filename = fingerprint + path.suffix.lower()
                    (assets / filename).write_bytes(contents)
                    node.set("file", filename)
                    sources[str(path)] = fingerprint
                asset_node.append(node)
            folder = output / f"task_{index:03d}"
            folder.mkdir()
            ET.indent(scene)
            (folder / "scene.xml").write_text(ET.tostring(scene, encoding="unicode") + "\n")
            (folder / "task.bddl").write_bytes(bddl.read_bytes())
            metadata = {
                "schema_version": 2,
                "suite": suite_name,
                "task_index": index,
                "task": task._asdict(),
                "parsed_problem": core.parsed_problem,
                "objects": objects,
                "initial_states": joint_bank,
                "source_init_sha256": digest(bank),
                "source_bddl_sha256": digest(bddl),
                "source_assets": sources,
                "upstream_table_top_m": list(core.table_offset),
                "source_timestep_s": float(model.opt.timestep),
                "source_gravity_m_s2": model.opt.gravity.tolist(),
                "removed_robot_roots": removed,
                "on_predicate": {"maximum_xy_distance_m": 0.03, "requires_contact": True},
                "body_inertials": {
                    name: {
                        "mass": float(model.body(name).mass[0]),
                        "ipos": model.body(name).ipos.tolist(),
                        "iquat": model.body(name).iquat.tolist(),
                        "inertia": model.body(name).inertia.tolist(),
                    }
                    for name in sorted(retained_bodies)
                },
                "contact_parameters": {
                    geom.get("name"): {
                        field: np.atleast_1d(getattr(model.geom(geom.get("name")), field)).tolist()
                        for field in (
                            "contype",
                            "conaffinity",
                            "condim",
                            "friction",
                            "solref",
                            "solimp",
                        )
                    }
                    for geom in world.iter("geom")
                },
                "geom_geometry": {
                    geom.get("name"): {
                        "body": model.body(model.geom(geom.get("name")).bodyid[0]).name,
                        **{
                            field: np.atleast_1d(
                                getattr(model.geom(geom.get("name")), field)
                            ).tolist()
                            for field in ("type", "size", "pos", "quat")
                        },
                    }
                    for geom in world.iter("geom")
                },
            }
            (folder / "task.json").write_text(
                json.dumps(metadata, indent=2, allow_nan=False) + "\n"
            )
            tasks.append(index)
            print(
                f"Exported {suite_name} task {index}: {len(states)} object initial states",
                flush=True,
            )
        finally:
            environment.close()
    # No runtime dependency on this machine's source/configuration directories.
    import shutil
    from importlib.metadata import version

    shutil.rmtree(configuration)
    return {
        "tasks": tasks,
        "upstream_packages": {p: version(p) for p in ("mujoco", "robosuite", "torch")},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    arguments = parser.parse_args()
    result = export(
        arguments.source.resolve(), arguments.output.resolve(), arguments.suite, arguments.tasks
    )
    (arguments.output / "upstream.json").write_text(json.dumps(result, indent=2) + "\n")
