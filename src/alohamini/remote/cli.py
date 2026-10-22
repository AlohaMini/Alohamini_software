"""Cloud service and Pi executor command-line entry points."""

import argparse
import sys
import threading

from alohamini.errors import AlohaMiniError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="alohamini inference")
    commands = parser.add_subparsers(dest="operation", required=True)
    cloud = commands.add_parser("serve", help="Run the policy on the cloud GPU")
    cloud.add_argument("--bind", default="tcp://127.0.0.1:5560", help="Loopback or VPN address")
    cloud.add_argument("--policy.path", dest="checkpoint", required=True)
    cloud.add_argument("--device", default="cuda")
    cloud.add_argument("--chunk-size", type=int)
    cloud.add_argument("--task")
    cloud.add_argument("--robot_model")
    cloud.add_argument(
        "--calibration-mode",
        choices=("strict", "normalized", "legacy_normalized"),
        default="strict",
    )
    cloud.add_argument("--fixed-current", nargs="+")
    cloud.add_argument("--request-timeout-s", type=float, default=15.0, help="Idle session lease")
    pi = commands.add_parser("run", help="Execute cloud action chunks beside the Pi Host")
    pi.add_argument("--server", required=True, help="tcp://HOST:PORT on a VPN or SSH tunnel")
    pi.add_argument("--host", default="127.0.0.1")
    pi.add_argument("--robot_model", required=True)
    pi.add_argument("--fps", type=int, default=30)
    pi.add_argument("--episode_time", type=float, default=90.0)
    pi.add_argument("--max-prediction-age-s", type=float, default=1.0)
    pi.add_argument("--rpc-timeout-s", type=float, default=10.0)
    pi.add_argument("--ack-timeout-s", type=float, default=0.5)
    pi.add_argument("--fixed-current", nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.operation == "serve":
            from alohamini.remote.server import NativeChunkPolicy, serve

            overrides = dict(calibration_mode=args.calibration_mode)
            if args.task is not None:
                overrides["task"] = args.task
            if args.robot_model is not None:
                overrides["robot_model"] = args.robot_model
            backend = NativeChunkPolicy(
                args.checkpoint,
                device=args.device,
                chunk_size=args.chunk_size,
                overrides=overrides,
                fixed_current=tuple(args.fixed_current) if args.fixed_current else None,
            )
            serve(args.bind, backend, threading.Event(), request_timeout_s=args.request_timeout_s)
        else:
            from alohamini.remote.agent import run

            stats = run(
                args.server,
                host=args.host,
                robot_model=args.robot_model,
                fps=args.fps,
                duration_s=args.episode_time,
                max_prediction_age_s=args.max_prediction_age_s,
                rpc_timeout_s=args.rpc_timeout_s,
                ack_timeout_s=args.ack_timeout_s,
                fixed_current=args.fixed_current,
            )
            print(
                f"Remote episode finished: {stats['sent']} commands sent; "
                f"last Host acknowledgement={stats['acknowledged_sequence']}",
                flush=True,
            )
        return 0
    except KeyboardInterrupt:
        print("Remote inference stopped.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError, AlohaMiniError) as exc:
        print(f"alohamini: {exc}", file=sys.stderr)
        return 1
