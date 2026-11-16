"""RoboTwin research entry point; isolate its checkout-relative runtime paths."""

import argparse
import json
import math
import os
import time
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlohaMini RoboTwin research scene")
    parser.add_argument("operation", choices=["view"])
    parser.add_argument("--source", required=True, type=Path, help="RoboTwin checkout")
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    source = args.source.expanduser().resolve()
    config = None if args.scene_config is None else args.scene_config.expanduser().resolve()
    if not (source / "envs/robot/robot.py").is_file():
        parser.error("--source must point to a RoboTwin checkout")
    # RoboTwin imports object metadata relative to the checkout. This CLI owns
    # the process; the reusable model loader never changes a caller's cwd.
    os.chdir(source)
    import sapien

    from alohamini.simulation.sapien_scene import add_objects
    from alohamini.simulation.scene import load_scene

    from .robot import create_robot

    configuration = load_scene(config)
    systems = [sapien.physx.PhysxCpuSystem()]
    if not args.headless:
        systems.append(sapien.render.RenderSystem())
    scene = sapien.Scene(systems)
    scene.set_timestep(configuration["timestep_s"])
    if not args.headless:
        scene.set_ambient_light([0.3, 0.3, 0.3])
        scene.add_directional_light([0.3, 0.5, -1], [1, 1, 1], shadow=True)
    robot = create_robot(
        scene,
        source,
        render=not args.headless,
        initial_positions=configuration["initial_positions"],
        T_world_base=configuration["T_world_base"],
        timestep=scene.timestep,
        base_mode=configuration["base_mode"],
    )
    add_objects(scene, configuration, render=not args.headless)
    viewer = None
    try:
        if not args.headless:
            from sapien.utils import Viewer

            viewer = Viewer()
            viewer.set_scene(scene)
            viewer.set_camera_xyz(1.5, 1.5, 1.3)
            viewer.set_camera_rpy(0, math.atan2(0.7, math.sqrt(4.5)), -3 * math.pi / 4)
        step = 0
        while (viewer is not None and not viewer.closed) or (args.headless and step < args.steps):
            start = time.monotonic()
            for _ in range(1 if args.headless else max(1, round(1 / scene.timestep / 30))):
                robot.apply_passive_force()
                scene.step()
                step += 1
            if viewer is not None:
                scene.update_render()
                viewer.render()
                time.sleep(max(0, 1 / 30 - (time.monotonic() - start)))
        if args.headless:
            print(
                json.dumps(
                    {
                        "backend": "robotwin",
                        "physics_steps": step,
                        "joint_names": robot.shared.names,
                    }
                )
            )
    finally:
        if viewer is not None:
            viewer.close()
        scene.clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
