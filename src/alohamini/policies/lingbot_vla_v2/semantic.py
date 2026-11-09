"""Named AlohaMini fields in the released 55D LingBot layout.

The upstream real_robot.yaml orders waist/head/base before hand. The LeRobot
default configuration at the imported revision instead orders hand first; never
infer checkpoint slot order from dictionary equality or tensor width.
"""

from copy import deepcopy

import numpy as np

JOINTS = (
    ("arm.position", 14),
    ("end.position", 14),
    ("effector.position", 2),
    ("waist.position", 4),
    ("head.position", 2),
    ("base.position", 3),
    ("hand.position", 12),
    ("reserved.slots", 4),
)
SUFFIXES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
NAMES = tuple(
    f"arm_{side}_{name}.pos" for side in ("left", "right") for name in (*SUFFIXES, "gripper")
) + ("x.vel", "y.vel", "theta.vel", "lift_axis.height_mm")


def make_layout(names, units, *, relative=True, lift_mode="hold"):
    if list(names) != list(NAMES) or len(units) != len(names) or not all(units):
        raise ValueError("LingBot AlohaMini preset requires named 18D fields and declared units")
    slots = list(range(6)) + [28] + list(range(6, 12)) + [29, 36, 37, 38]
    if lift_mode == "reserved":
        slots.append(51)
    elif lift_mode == "hold":
        slots.append(None)
    else:
        raise ValueError("lift_mode must be hold or reserved")
    return dict(
        version=1,
        names=list(names),
        units=list(units),
        slots=slots,
        relative=[i for i in (*range(6), *range(7, 13))] if relative else [],
        canonical_joints=[list(x) for x in JOINTS],
        lift_mode=lift_mode,
        extension={
            "lift_axis.height_mm": "reserved[0], platform-specific; no pretrained semantic claim"
        }
        if lift_mode == "reserved"
        else {},
    )


def validate_layout(layout):
    expected = make_layout(
        layout["names"],
        layout["units"],
        relative=bool(layout["relative"]),
        lift_mode=layout["lift_mode"],
    )
    if layout != expected:
        raise ValueError("LingBot semantic layout differs from the declared field protocol")


def relative_actions(state, actions, layout):
    result = np.array(actions, dtype=np.float64, copy=True)
    dims = layout["relative"]
    result[..., dims] -= np.asarray(state)[..., None, dims]
    return result


def configure(options, samples):
    feature = samples.info["features"]["action"]
    if samples.selection is None or samples.selection.feature["names"] != feature["names"]:
        raise ValueError("LingBot requires the complete named state, including fixed coordinates")
    layout = make_layout(
        feature["names"],
        samples.selection.units,
        relative=options.get("use_relative_actions", True),
        lift_mode=options.get("lift_mode", "hold"),
    )
    if options.get("layout") and options["layout"] != layout:
        raise ValueError("LingBot checkpoint semantic layout differs from this dataset")
    options["layout"] = deepcopy(layout)
    options["canonical_joints"] = dict(JOINTS)
    options["dataset_fps"] = samples.info["fps"]
