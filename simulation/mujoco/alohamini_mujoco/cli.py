"""Model inspection, interactive viewing and dynamic joint-trajectory execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from alohamini.model import get_robot_model

from .audit import compare_maniskill, convexification_report, gripper_sections, validate_model
from .collision import generate_bundle, merge_bundles
from .control import DynamicFollower, JointTrajectory
from .description import Description
from .model import ModelProfile, build_model
from .viewer import set_camera, set_geometry


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
        xml = ET.fromstring(robot.xml)
        xml.find("compiler").set("meshdir", "assets")
        xml.find("compiler").set("texturedir", "assets")
        ET.indent(xml)
        model_xml = ET.tostring(xml, encoding="unicode") + "\n"
        (stage / "model.xml").write_text(model_xml)
        manifest = {
            **robot.manifest,
            "model_xml_sha256": hashlib.sha256(model_xml.encode()).hexdigest(),
            "derived_asset_sha256": {
                n: hashlib.sha256(b).hexdigest() for n, b in robot.assets.items()
            },
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        # Do not publish a package whose relative mesh references cannot load.
        reloaded = mujoco.MjModel.from_xml_path(str(stage / "model.xml"))
        if (reloaded.nq, reloaded.ngeom) != (robot.model.nq, robot.model.ngeom):
            raise RuntimeError("Export changed model topology")
        if output.exists():
            raise FileExistsError(f"Output appeared during export: {output}")
        os.rename(stage, output)
    return output


def preview(robot, data, path, *, collisions=False):
    from PIL import Image

    options = mujoco.MjvOption()
    set_geometry(options, collisions=collisions)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    set_camera(robot, camera)
    with mujoco.Renderer(
        robot.model, height=720, width=960, max_geom=max(10000, robot.model.ngeom + 1000)
    ) as renderer:
        renderer.update_scene(data, camera=camera, scene_option=options)
        Image.fromarray(renderer.render()).save(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlohaMini MuJoCo asset inspection and following")
    commands = parser.add_subparsers(dest="operation", required=True)
    inspect = commands.add_parser(
        "inspect", help="validate frames, inertia, geometry and gripper sections"
    )
    inspect.add_argument("--samples", type=int, default=16)
    inspect.add_argument("--seed", type=int, default=0)
    inspect.add_argument("--compare-maniskill", type=Path)
    inspect.add_argument(
        "--render", action="store_true", help="save visual and physical-collision previews"
    )
    follow = commands.add_parser(
        "follow", help="execute a named, time-defined trajectory using dynamics"
    )
    follow.add_argument("trajectory", type=Path)
    collect = commands.add_parser("collect", help="collect and replay-validate a task episode")
    collect.add_argument("trajectory", type=Path)
    collect.add_argument("--cameras", nargs="+", default=["agentview", "left_wrist", "right_wrist"])
    collect.add_argument("--image-size", type=int, default=128)
    replay = commands.add_parser(
        "replay", help="verify saved commands against physical state history"
    )
    replay.add_argument("dataset", type=Path)
    view = commands.add_parser("view", help="open an interactive model and dynamics viewer")
    view.add_argument("--trajectory", type=Path, help="play a named, time-defined trajectory")
    view.add_argument("--collisions", action="store_true", help="start with physical convex hulls")
    view.add_argument("--paused", action="store_true", help="start paused; Space plays dynamics")
    view.add_argument("--links", nargs="+", help="inspect only these named robot links")
    decompose = commands.add_parser("decompose", help="generate explicit CAD collision candidates")
    decompose.add_argument("--links", nargs="+", required=True, help="named URDF links")
    decompose.add_argument("--tolerance-m", type=float, default=0.0005)
    decompose.add_argument("--seed", type=int, default=0)
    decompose.add_argument("--nonmanifold", choices=("reject", "remesh"), default="reject")
    decompose.add_argument("--output", required=True, type=Path, help="new candidate directory")
    cad = commands.add_parser("cad-inspect", help="preserve exact STEP parts and assembly frames")
    cad.add_argument("source", type=Path)
    cad.add_argument("--output", type=Path, required=True)
    tessellate = commands.add_parser("cad-tessellate", help="tessellate each original CAD solid")
    tessellate.add_argument("--cad-bundle", type=Path, required=True)
    tessellate.add_argument("--definition", required=True)
    tessellate.add_argument("--linear-deflection-m", type=float, default=5e-5)
    tessellate.add_argument("--angular-deflection-rad", type=float, default=0.1)
    tessellate.add_argument("--output", type=Path, required=True)
    merger = commands.add_parser(
        "collision-merge", help="combine disjoint CAD collision candidates"
    )
    merger.add_argument("--bundles", type=Path, nargs="+", required=True)
    merger.add_argument("--output", type=Path, required=True)
    exporter = commands.add_parser(
        "collision-export", help="export the same convex pieces as URDF/OBJ"
    )
    exporter.add_argument("--collision-bundle", type=Path, required=True)
    exporter.add_argument("--output", type=Path, required=True)
    checker = commands.add_parser(
        "collision-check-sapien", help="check actual cooked SAPIEN convex shapes"
    )
    checker.add_argument("--asset", type=Path, required=True)
    checker.add_argument("--python", type=Path, required=True)
    checker.add_argument("--output", type=Path, required=True)
    importer = commands.add_parser("libero-import", help="snapshot pinned upstream Spatial scenes")
    importer.add_argument("--source", type=Path, required=True)
    importer.add_argument(
        "--python", type=Path, required=True, help="existing LIBERO/robosuite Python"
    )
    importer.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    importer.add_argument("--output", type=Path, required=True)
    for command in (inspect, follow, collect):
        command.add_argument("--output", required=True, type=Path, help="new output directory")
    for command in (inspect, follow, view, collect):
        command.add_argument("--libero-bundle", type=Path)
        command.add_argument(
            "--scene-config", type=Path, help="explicit world/robot transforms and cameras"
        )
        command.add_argument("--task-index", type=int, default=0)
        command.add_argument("--reset-index", type=int, default=0)
        command.add_argument(
            "--initial-positions",
            type=json.loads,
            default={},
            help="JSON mapping of URDF joint names to radians/metres",
        )
        command.add_argument(
            "--collision-profile", choices=("authoritative", "cad-pieces"), default="authoritative"
        )
        command.add_argument(
            "--collision-bundle", type=Path, help="explicit CAD candidate directory"
        )
    args = parser.parse_args(argv)
    try:
        if args.operation == "cad-tessellate":
            from .cad import tessellate_definition

            report = tessellate_definition(
                args.cad_bundle,
                args.definition,
                args.output,
                linear_deflection_m=args.linear_deflection_m,
                angular_deflection_rad=args.angular_deflection_rad,
            )
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "solids": len(report["solids"]),
                        "status": report["status"],
                    }
                )
            )
            return 0
        if args.operation == "cad-inspect":
            from .cad import inspect_step

            report = inspect_step(args.source, args.output)
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "definitions": len(report["definitions"]),
                        "occurrences": len(report["occurrences"]),
                        "status": report["status"],
                    }
                )
            )
            return 0
        if args.operation == "collision-export":
            from .interop import export_collision_urdf

            output = export_collision_urdf(
                Description(get_robot_model("alohamini2pro")), args.collision_bundle, args.output
            )
            print(json.dumps({"output": str(output)}))
            return 0
        if args.operation == "collision-check-sapien":
            from .interop import check_sapien_asset

            report = check_sapien_asset(args.asset, args.python, args.output)
            print(json.dumps({"passed": report["passed"], "output": str(args.output)}))
            return 0 if report["passed"] else 1
        if args.operation == "collision-merge":
            output = merge_bundles(
                Description(get_robot_model("alohamini2pro")), args.bundles, args.output
            )
            print(json.dumps({"status": "candidate", "output": str(output)}))
            return 0
        if args.operation == "replay":
            from .collection import replay_episode

            report = replay_episode(args.dataset)
            print(json.dumps(report))
            return 0 if report["passed"] else 1
        if args.operation == "libero-import":
            from .libero import import_libero

            output = import_libero(args.source, args.python, args.output, tasks=args.tasks)
            print(json.dumps({"output": str(output), "tasks": args.tasks}))
            return 0
        if args.operation == "decompose":
            output = generate_bundle(
                Description(get_robot_model("alohamini2pro")),
                args.links,
                args.output,
                tolerance_m=args.tolerance_m,
                seed=args.seed,
                nonmanifold=args.nonmanifold,
            )
            print(json.dumps({"status": "candidate", "output": str(output)}, ensure_ascii=False))
            return 0
        if not isinstance(args.initial_positions, dict):
            raise ValueError("Initial positions must be an object keyed by URDF joint name")
        if (
            args.operation == "inspect"
            and args.compare_maniskill
            and (args.collision_profile != "authoritative" or args.collision_bundle)
        ):
            raise ValueError(
                "Exact ManiSkill comparison requires the authoritative collision profile"
            )
        scene, configuration = None, {}
        if args.libero_bundle or args.scene_config:
            if not (args.libero_bundle and args.scene_config):
                raise ValueError("Use --libero-bundle together with --scene-config")
            from .libero import LiberoScene

            configuration = json.loads(args.scene_config.expanduser().read_text())
            scene = LiberoScene(
                args.libero_bundle, args.task_index, configuration, args.reset_index
            )
        robot = build_model(
            profile=ModelProfile(
                initial_positions={
                    **configuration.get("initial_positions", {}),
                    **args.initial_positions,
                },
                **({"T_world_root": configuration["T_world_root"]} if scene else {}),
                collision_profile=args.collision_profile,
                collision_bundle=args.collision_bundle,
            ),
            scene=scene,
        )
        trajectory = None
        if args.operation in ("follow", "view", "collect") and args.trajectory is not None:
            specification = json.loads(args.trajectory.expanduser().read_text())
            if not isinstance(specification, dict) or specification.get("schema_version") != 1:
                raise ValueError("Trajectory must declare schema_version 1")
            trajectory = JointTrajectory(
                robot,
                specification["joint_names"],
                specification["times_s"],
                specification["positions"],
            )
        if args.operation == "view":
            from .viewer import run_viewer

            run_viewer(
                robot, trajectory, collisions=args.collisions, paused=args.paused, links=args.links
            )
            return 0
        if args.operation == "collect":
            from .collection import CollectionSettings, collect_episode

            report = collect_episode(
                robot,
                trajectory,
                args.output,
                settings=CollectionSettings(image_size=args.image_size),
                cameras=args.cameras,
            )
            print(json.dumps(report, ensure_ascii=False))
            return 0 if report["passed"] else 1
        if args.operation == "inspect":
            report = {
                "validation": validate_model(robot, samples=args.samples, seed=args.seed),
                "convexification": convexification_report(robot),
                "gripper_cross_sections": [
                    gripper_sections(robot, side=s) for s in ("left", "right")
                ],
            }
            if args.compare_maniskill:
                report["maniskill_comparison"] = compare_maniskill(robot, args.compare_maniskill)
            passed = report["validation"]["passed"] and report.get("maniskill_comparison", {}).get(
                "passed", True
            )
            if scene is not None:
                report["scene"] = scene.validate(robot)
                passed = (
                    passed
                    and report["scene"]["passed"]
                    and report["scene"]["static_robot_scene"]["passed"]
                )
        else:
            follower = DynamicFollower(robot)
            report = {"follow": follower.follow(trajectory)}
            passed = report["follow"]["passed"]
        output = export_robot(robot, report, args.output)
        if args.operation == "inspect" and args.render:
            preview(robot, robot.data(), output / "visual.png")
            preview(robot, robot.data(), output / "collision.png", collisions=True)
        print(json.dumps({"passed": bool(passed), "output": str(output)}, ensure_ascii=False))
        return 0 if passed else 1
    except KeyboardInterrupt:
        return 130
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        mujoco.FatalError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"alohamini-mujoco: {exc}", file=sys.stderr)
        return 1
