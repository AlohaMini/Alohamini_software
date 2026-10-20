"""Model inspection, interactive viewing and dynamic joint-trajectory execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from alohamini.model import get_robot_model

from .audit import compare_maniskill, convexification_report, gripper_sections, validate_model
from .collision import generate_bundle
from .control import DynamicFollower, JointTrajectory
from .description import Description
from .model import ModelProfile, build_model
from .viewer import set_geometry


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
    camera.lookat[:] = robot.model.stat.center
    camera.distance = 1.3 * robot.model.stat.extent
    camera.azimuth = 45
    camera.elevation = -20
    with mujoco.Renderer(robot.model, height=720, width=960) as renderer:
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
    view = commands.add_parser("view", help="open an interactive model and dynamics viewer")
    view.add_argument("--trajectory", type=Path, help="play a named, time-defined trajectory")
    view.add_argument("--collisions", action="store_true", help="start with physical convex hulls")
    view.add_argument("--paused", action="store_true", help="start paused; Space plays dynamics")
    decompose = commands.add_parser("decompose", help="generate explicit CAD collision candidates")
    decompose.add_argument("--links", nargs="+", required=True, help="named URDF links")
    decompose.add_argument("--tolerance-m", type=float, default=0.0005)
    decompose.add_argument("--seed", type=int, default=0)
    decompose.add_argument("--nonmanifold", choices=("reject", "remesh"), default="reject")
    decompose.add_argument("--output", required=True, type=Path, help="new candidate directory")
    for command in (inspect, follow):
        command.add_argument("--output", required=True, type=Path, help="new output directory")
    for command in (inspect, follow, view):
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
        robot = build_model(
            profile=ModelProfile(
                initial_positions=args.initial_positions,
                collision_profile=args.collision_profile,
                collision_bundle=args.collision_bundle,
            )
        )
        trajectory = None
        if args.operation in ("follow", "view") and args.trajectory is not None:
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

            run_viewer(robot, trajectory, collisions=args.collisions, paused=args.paused)
            return 0
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
    except (OSError, ValueError, KeyError, RuntimeError, mujoco.FatalError) as exc:
        print(f"alohamini-mujoco: {exc}", file=sys.stderr)
        return 1
