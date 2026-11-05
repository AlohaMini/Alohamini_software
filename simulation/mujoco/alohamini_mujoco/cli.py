"""Interactive simulation, LIBERO scene import, collection and replay."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco

from .control import FollowerSettings, JointTrajectory
from .model import ModelProfile, build_model


def _path_argument(value: str) -> Path:
    if not value.strip():
        raise argparse.ArgumentTypeError(
            "path cannot be empty; provide a path or set the referenced environment variable"
        )
    return Path(value)


def _egl_window(device: int, argv: list[str]):
    """Use an EGL GPU with the native viewer and lossless local X11 transport."""
    if sys.platform != "linux":
        raise ValueError("--window-egl-device is supported on Linux")
    if device < 0:
        raise ValueError("--window-egl-device must be nonnegative")
    if not os.environ.get("DISPLAY"):
        raise ValueError("--window-egl-device requires an X11 or Xwayland desktop (DISPLAY)")
    profile = {
        "VGL_ISACTIVE": "1",
        "VGL_DISPLAY": f"egl{device}",
        "VGL_COMPRESS": "proxy",
        "VGL_SPOIL": "0",
        "VGL_GAMMA": "0",
        "MUJOCO_GL": "glfw",
        "PYGLFW_LIBRARY_VARIANT": "x11",
    }
    offload = (
        "__NV_PRIME_RENDER_OFFLOAD",
        "__NV_PRIME_RENDER_OFFLOAD_PROVIDER",
        "__GLX_VENDOR_LIBRARY_NAME",
    )
    if all(os.environ.get(k) == v for k, v in profile.items()) and not any(
        os.environ.get(k) for k in offload
    ):
        return
    executable = shutil.which("vglrun")
    if executable is None:
        executable = next(
            (
                str(p)
                for p in (
                    Path("/opt/VirtualGL/bin/vglrun"),
                    Path.home() / ".local/opt/VirtualGL/bin/vglrun",
                )
                if p.is_file()
            ),
            None,
        )
    if executable is None:
        raise FileNotFoundError("Install VirtualGL 3.1.5 to use --window-egl-device")
    root = Path(executable).resolve().parent.parent
    libraries = [
        str(p)
        for p in (root / "lib64", root / "lib", root.parent.parent / "usr/lib")
        if (p / "libvglfaker.so").is_file()
    ]
    # VirtualGL's dlopen interposer needs PyTorch's $ORIGIN libraries explicitly
    # on its search path when loading a policy. It does not import PyTorch here.
    torch = importlib.util.find_spec("torch")
    if torch is not None and torch.origin:
        library = Path(torch.origin).parent / "lib"
        if library.is_dir():
            libraries.append(str(library))
    command = [executable, "-d", f"egl{device}", "-c", "proxy", "-sp", "-g"]
    if libraries:
        command += ["-ld", os.pathsep.join(libraries)]
    command += [sys.executable, "-m", "alohamini_mujoco.cli", *argv]
    environment = os.environ.copy()
    for key in offload:
        environment.pop(key, None)
    environment["MUJOCO_GL"] = "glfw"
    environment["PYGLFW_LIBRARY_VARIANT"] = "x11"
    print(f"EGL window: device egl{device}, lossless RGB (VirtualGL)", flush=True)
    os.execvpe(executable, command, environment)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="AlohaMini MuJoCo simulation and LIBERO collection"
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    view = commands.add_parser("view", help="open an interactive simulation")
    view.add_argument("--trajectory", type=_path_argument, help="play a named joint trajectory")
    view.add_argument("--collisions", action="store_true", help="show collision geometry")
    view.add_argument("--paused", action="store_true", help="start paused")
    collect = commands.add_parser("collect", help="collect and replay-validate a task episode")
    collect.add_argument("trajectory", type=_path_argument, nargs="?")
    collect.add_argument(
        "--reference", type=_path_argument, help="retarget an admitted training episode"
    )
    collect.add_argument("--recipe", type=_path_argument, help="named motion segment boundaries")
    collect.add_argument(
        "--resume", action="store_true", help="resume an unchanged reference batch"
    )
    collect.add_argument("--cameras", nargs="+", default=["agentview", "left_wrist", "right_wrist"])
    collect.add_argument("--image-size", type=int, default=128)
    collect.add_argument("--output", required=True, type=_path_argument)
    replay = commands.add_parser("replay", help="verify or view a collected episode")
    replay.add_argument("dataset", type=_path_argument)
    replay.add_argument("--view", action="store_true")
    replay.add_argument("--paused", action="store_true")
    replay.add_argument("--collisions", action="store_true")
    evaluate = commands.add_parser("evaluate", help="evaluate an SI policy on task initial states")
    evaluate.add_argument("checkpoint", type=_path_argument)
    evaluate.add_argument("--reset-indices", type=int, nargs="+", default=list(range(40, 50)))
    evaluate.add_argument("--horizon-s", type=float, default=60)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--view", action="store_true")
    evaluate.add_argument("--output", required=True, type=_path_argument)
    evaluate.set_defaults(reset_index=0)
    importer = commands.add_parser("libero-import", help="import pinned LIBERO Spatial scenes")
    importer.add_argument("--source", type=_path_argument, required=True)
    importer.add_argument(
        "--python", type=_path_argument, required=True, help="LIBERO environment Python"
    )
    importer.add_argument("--tasks", type=int, nargs="+", default=list(range(10)))
    importer.add_argument("--output", type=_path_argument, required=True)
    for command in (view, replay, evaluate):
        command.add_argument(
            "--window-egl-device", type=int, metavar="INDEX",
            help="render the native window on an EGL GPU using VirtualGL",
        )
    for command in (view, collect, evaluate):
        command.add_argument("--libero-bundle", type=_path_argument)
        command.add_argument("--scene-config", type=_path_argument)
        command.add_argument("--task-index", type=int, default=0)
        if command is collect:
            resets = command.add_mutually_exclusive_group()
            resets.add_argument("--reset-index", type=int, default=0)
            resets.add_argument("--reset-indices", type=int, nargs="+")
        elif command is view:
            command.add_argument("--reset-index", type=int, default=0)
        command.add_argument(
            "--initial-positions", type=json.loads, default={}, help="named URDF positions in rad/m"
        )
        command.add_argument("--collision-bundle", type=_path_argument)
        command.add_argument("--inertial-bundle", type=_path_argument)
    args = parser.parse_args(argv)
    if args.operation == "evaluate" and not (args.libero_bundle and args.scene_config):
        parser.error("evaluate requires --libero-bundle and --scene-config")
    if args.operation == "collect":
        if (args.trajectory is None) == (args.reference is None):
            parser.error("collect requires either a trajectory or --reference")
        if bool(args.reference) != bool(args.recipe):
            parser.error("use --reference together with --recipe")
        if args.reset_indices is not None and args.reference is None:
            parser.error("--reset-indices requires a reference motion")
        if args.resume and args.reset_indices is None:
            parser.error("--resume requires --reset-indices")
    try:
        if getattr(args, "window_egl_device", None) is not None:
            if args.operation != "view" and not args.view:
                raise ValueError("--window-egl-device requires --view")
            _egl_window(args.window_egl_device, list(sys.argv[1:] if argv is None else argv))
        if args.operation == "replay":
            from .collection import replay_episode

            report = replay_episode(
                args.dataset, view=args.view, paused=args.paused, collisions=args.collisions
            )
            print(json.dumps(report, ensure_ascii=False))
            return 0 if report["passed"] else 1
        if args.operation == "libero-import":
            from .libero import import_libero

            output = import_libero(args.source, args.python, args.output, tasks=args.tasks)
            print(json.dumps({"output": str(output), "tasks": args.tasks}))
            return 0
        if not isinstance(args.initial_positions, dict):
            raise ValueError("Initial positions must be an object keyed by URDF joint name")
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
                collision_bundle=args.collision_bundle,
                inertial_bundle=args.inertial_bundle,
            ),
            scene=scene,
        )
        if args.operation == "evaluate":
            from .evaluation import evaluate_policy

            report = evaluate_policy(
                robot,
                args.checkpoint,
                args.reset_indices,
                args.output,
                horizon_s=args.horizon_s,
                device=args.device,
                view=args.view,
            )
            print(json.dumps(report, ensure_ascii=False))
            return 0
        trajectory, follower_settings, generation = None, None, None
        if args.trajectory is not None:
            specification = json.loads(args.trajectory.expanduser().read_text())
            if not isinstance(specification, dict) or specification.get("schema_version") != 1:
                raise ValueError("Trajectory must declare schema_version 1")
            trajectory = JointTrajectory(
                robot,
                specification["joint_names"],
                specification["times_s"],
                specification["positions"],
            )
            controller = specification.get("controller", {})
            if (
                not isinstance(controller, dict)
                or controller.keys() - FollowerSettings.__dataclass_fields__.keys()
            ):
                raise ValueError("Trajectory controller must contain named FollowerSettings fields")
            follower_settings = FollowerSettings(**controller)
            generation = specification.get("generation")
        if args.operation == "view":
            from .viewer import run_viewer

            run_viewer(
                robot,
                trajectory,
                collisions=args.collisions,
                paused=args.paused,
                follower_settings=follower_settings,
            )
            return 0
        from .collection import CollectionSettings, collect_episode, collect_reference_batch

        settings = CollectionSettings(image_size=args.image_size)
        if args.reference is not None:
            from .retarget import MotionRecipe, ReferenceMotion

            recipe = MotionRecipe.read(args.recipe)

            def progress(item):
                print(json.dumps(item), file=sys.stderr, flush=True)

            if args.reset_indices is not None:
                report = collect_reference_batch(
                    robot,
                    args.reference,
                    recipe,
                    args.reset_indices,
                    args.output,
                    settings=settings,
                    cameras=args.cameras,
                    resume=args.resume,
                    progress=progress,
                )
                print(json.dumps(report, ensure_ascii=False))
                return 0 if report["passed"] else 1
            trajectory, follower_settings, generation = ReferenceMotion(
                robot, args.reference, recipe
            ).plan(robot, progress=progress)

        report = collect_episode(
            robot,
            trajectory,
            args.output,
            settings=settings,
            follower_settings=follower_settings,
            cameras=args.cameras,
            generation=generation,
        )
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["passed"] else 1
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


if __name__ == "__main__":
    raise SystemExit(main())
