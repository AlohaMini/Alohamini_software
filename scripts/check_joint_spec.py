#!/usr/bin/env python3
"""Validate fixed joint coordinates against model sources and optional MuJoCo output."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from alohamini.calibration.joint_space import StandardJointSpace
from alohamini.model.joint_spec import ModelJointSpecification


def check(robot_model, *, mujoco=False):
    spec = ModelJointSpecification.from_model(robot_model)
    report = spec.validate()
    space = StandardJointSpace.from_specification(robot_model)
    source = spec.model.description_tree("collision")
    conventions = yaml.safe_load(
        spec.model.asset_path("config/kinematics/encoder_conventions.yaml").read_text()
    )
    mirrored = yaml.safe_load(
        spec.model.asset_path("config/kinematics/joint_limits.yaml").read_text()
    )["joints"]
    if conventions["robot_model"] != robot_model:
        raise ValueError("Encoder conventions belong to another model")
    for name, row in spec.joints.items():
        original = source.find(f"joint[@name='{name}']/limit")
        expected = [float(original.get(key)) for key in ("lower", "upper")]
        if not all(
            math.isclose(mirrored[name][key], value, abs_tol=1e-10)
            for key, value in zip(("lower", "upper"), expected, strict=True)
        ):
            raise ValueError(f"{name}: joint_limits.yaml differs from the source model")
        if row.unit == "rad":
            suffix = name.split("_", 1)[1].removesuffix("_joint")
            if conventions["joints"][suffix]["sign"] != row.direction:
                raise ValueError(f"{name}: direction differs from encoder_conventions.yaml")
    report["source_mirrors_matched"] = True
    report["source_sha256"] = {
        name: hashlib.sha256(spec.model.asset_path(name).read_bytes()).hexdigest()
        for name in spec.descriptor["sources"]
    }
    report["ranges"] = {
        name: dict(
            unit=row.unit,
            nominal=[row.lower, row.upper],
            command=list(row.command_limits),
            reference=row.reference,
            direction=row.direction,
        )
        for name, row in spec.joints.items()
    }
    for name, row in spec.joints.items():
        actual = space.joints[name].position_limits
        if not all(
            math.isclose(a, b, abs_tol=1e-10)
            for a, b in zip(actual, (row.lower, row.upper), strict=True)
        ):
            raise ValueError(f"{name}: standard coordinates differ from model specification")
    report["standard_coordinate_sha256"] = space.signature
    if mujoco:
        import mujoco as mj
        from alohamini_mujoco.model import build_model

        robot = build_model(robot_model=spec.model)
        for name, row in spec.joints.items():
            joint = robot.model.joint(name)
            original = source.find(f"joint[@name='{name}']/limit")
            expected = [float(original.get(key)) for key in ("lower", "upper")]
            if not joint.limited[0] or not all(
                math.isclose(a, b, abs_tol=1e-10)
                for a, b in zip(joint.range, expected, strict=True)
            ):
                raise ValueError(f"{name}: MuJoCo limits differ from source URDF")
            if row.lower < joint.range[0] - 1e-10 or row.upper > joint.range[1] + 1e-10:
                raise ValueError(f"{name}: specification exceeds MuJoCo interval")
        report["mujoco"] = dict(
            version=mj.__version__, joints_checked=len(spec.joints), passed=True
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-model", default="alohamini2pro")
    parser.add_argument(
        "--mujoco", action="store_true", help="Also compile the installed MuJoCo model"
    )
    parser.add_argument(
        "--output", type=Path, help="Write a new JSON report; never overwrite a report"
    )
    args = parser.parse_args()
    report = check(args.robot_model, mujoco=args.mujoco)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    print(
        f"{report['robot_model']} joint specification v{report['specification_version']}: "
        f"{len(report['ranges'])} joints validated"
    )
    if "mujoco" in report:
        print(f"MuJoCo {report['mujoco']['version']}: source limits matched")
    print("Physical travel and collision-free workspace qualification: not performed")


if __name__ == "__main__":
    main()
