"""Isolate native CAD tessellation failures from assembly asset builds."""

from __future__ import annotations

import argparse
import json
import pyexpat  # noqa: F401 -- load Expat symbols before OpenCASCADE
import sys
from pathlib import Path

from .cad import tessellate_definition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cad-bundle", type=Path, required=True)
    parser.add_argument("--definition", required=True)
    parser.add_argument("--linear-deflection-m", type=float, required=True)
    parser.add_argument("--angular-deflection-rad", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = tessellate_definition(
            args.cad_bundle,
            args.definition,
            args.output,
            linear_deflection_m=args.linear_deflection_m,
            angular_deflection_rad=args.angular_deflection_rad,
        )
        print(json.dumps({"output": str(args.output), "status": report["status"]}))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"CAD tessellation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
