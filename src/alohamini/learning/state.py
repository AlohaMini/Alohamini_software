"""State selection for calibrated Host recordings and recorded LeRobot vectors."""

import numpy as np

from alohamini.datasets.record import StateSelection


class RecordedStateSelection:
    """Read recorded coordinates without inventing calibration or derived feedback."""

    def __init__(self, info, selection):
        feature = info["features"].get("observation.state")
        if not feature or len(feature.get("shape", [])) != 1:
            raise ValueError("Recorded state requires a one-dimensional observation.state")
        self.width = feature["shape"][0]
        if type(self.width) is not int or self.width < 1:
            raise ValueError("Recorded state width must be a positive integer")
        names = feature.get("names")
        if names is None:
            names = [f"observation.state[{i}]" for i in range(self.width)]
        if (
            not isinstance(names, list)
            or len(names) != self.width
            or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Recorded state names must match the vector width and be unique")
        self.source_names = list(names)
        selected = []
        if selection == "recorded":
            selected = list(names)
        else:
            groups = selection.split(",")
            if len(set(groups)) != len(groups):
                raise ValueError("State groups must be unique")
            for group in groups:
                if group == "joint_position":
                    subset = [n for n in names if n.endswith(".pos")]
                elif group == "base_velocity":
                    subset = ["x.vel", "y.vel", "theta.vel"]
                elif group == "lift_height":
                    subset = ["lift_axis.height_mm"]
                else:
                    raise ValueError(
                        "Unannotated data supports recorded state or named position groups; "
                        "derived velocity/current requires calibration and feedback metadata"
                    )
                if not subset or any(n not in names for n in subset):
                    raise ValueError(f"Recorded state has no complete {group} group")
                selected.extend(subset)
        self.columns = [("observation.state", names.index(n), 1, None) for n in selected]
        self.feature = {"dtype": "float32", "shape": [len(selected)], "names": selected}
        self.units = ["recorded (unit unverified)"] * len(selected)

    def frame(self, row):
        values = np.asarray(row["observation.state"], dtype=np.float32)
        if values.shape != (self.width,) or not np.isfinite(values).all():
            raise ValueError("Recorded state has an invalid shape or non-finite values")
        return values[[index for _, index, _, _ in self.columns]]


def select_state(info, selection, *, exclude_fixed=False):
    if selection == "none":
        return None
    if info.get("training_only"):
        return RecordedStateSelection(info, selection)
    return StateSelection(info, selection, exclude_fixed=exclude_fixed)
