"""View a research scene or advance it without a renderer."""

import argparse
import json
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description="AlohaMini ManiSkill research scene")
    parser.add_argument("operation", choices=["view"])
    parser.add_argument("--scene-config")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="headless physics steps")
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps must be positive")
    import gymnasium as gym

    import alohamini_maniskill  # noqa: F401

    env = gym.make(
        "AlohaMiniResearch-v1",
        scene_config=args.scene_config,
        obs_mode="state",
        control_mode="joint_position",
        reward_mode="none",
        sim_backend="physx_cpu",
        render_backend="none" if args.headless else "gpu",
        render_mode=None if args.headless else "human",
    )
    try:
        env.reset(seed=env.unwrapped.configuration["seed"])
        action = env.unwrapped.agent.hold_action()
        substeps = round(env.unwrapped.sim_freq / env.unwrapped.control_freq)
        if args.headless and args.steps % substeps:
            parser.error(f"--steps must be a multiple of {substeps} physics steps")
        count = args.steps // substeps
        step = 0
        while not args.headless or step < count:
            start = time.monotonic()
            env.step(action)
            step += 1
            if not args.headless:
                env.render()
                if env.unwrapped.viewer.closed:
                    break
                time.sleep(max(0, 1 / env.unwrapped.control_freq - (time.monotonic() - start)))
        if args.headless:
            print(
                json.dumps(
                    {
                        "backend": "maniskill",
                        "control_steps": step,
                        "joint_names": [j.name for j in env.unwrapped.agent.robot.active_joints],
                    }
                )
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
