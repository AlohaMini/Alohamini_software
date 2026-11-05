"""Explicit-frame CAD mass properties, in kg, metres and kg*m^2.

Each manual report has its own frame, including reports labelled "default".
Coordinate transport never estimates a frame or repairs a nonphysical tensor.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np

from .frames import rigid_transform, vector


def _shift(displacement):
    return np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement)


@dataclass(frozen=True)
class MassProperties:
    frame: str
    mass_kg: float
    com_m: np.ndarray
    central_inertia_kg_m2: np.ndarray

    def __post_init__(self):
        com = vector(self.com_m, 3, "Center of mass").copy()
        inertia = np.array(self.central_inertia_kg_m2, dtype=float, copy=True)
        if (
            not isinstance(self.frame, str)
            or not self.frame.strip()
            or not math.isfinite(self.mass_kg)
            or self.mass_kg <= 0
            or inertia.shape != (3, 3)
            or not np.isfinite(inertia).all()
        ):
            raise ValueError("Mass properties need a named frame, positive mass and finite tensor")
        arithmetic = 64 * np.finfo(float).eps * np.max(np.abs(inertia))
        if not np.allclose(inertia, inertia.T, rtol=0, atol=arithmetic):
            raise ValueError("Central inertia must be symmetric")
        inertia = (inertia + inertia.T) / 2
        moments = np.linalg.eigvalsh(inertia)
        if moments[0] <= 0 or moments[-1] > moments[0] + moments[1] + arithmetic:
            raise ValueError(f"Nonphysical central inertia in {self.frame}: {moments.tolist()}")
        com.flags.writeable = inertia.flags.writeable = False
        object.__setattr__(self, "com_m", com)
        object.__setattr__(self, "central_inertia_kg_m2", inertia)

    def in_frame(self, T_target_source, target_frame):
        """Transport COM and central tensor; translation adds no central-inertia term."""
        transform = rigid_transform(T_target_source)
        rotation = transform[:3, :3]
        return MassProperties(
            target_frame,
            self.mass_kg,
            rotation @ self.com_m + transform[:3, 3],
            rotation @ self.central_inertia_kg_m2 @ rotation.T,
        )

    def to_dict(self):
        return {
            "frame": self.frame,
            "mass_kg": self.mass_kg,
            "com_m": self.com_m.tolist(),
            "central_inertia_kg_m2": self.central_inertia_kg_m2.tolist(),
        }

    def mjcf_attributes(self):
        """Explicit inertial attributes for a body whose local frame is self.frame."""
        import mujoco

        moments, axes = np.linalg.eigh(self.central_inertia_kg_m2)
        if np.linalg.det(axes) < 0:
            axes[:, -1] *= -1
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, np.ascontiguousarray(axes).ravel())
        return {
            key: " ".join(format(float(v), ".17g") for v in values)
            for key, values in {
                "mass": [self.mass_kg],
                "pos": self.com_m,
                "quat": quaternion,
                "diaginertia": moments,
            }.items()
        }


def combine(properties):
    """Combine disjoint material selections expressed in the same frame and pose."""
    parts = tuple(properties)
    if not parts or len({p.frame for p in parts}) != 1:
        raise ValueError("Combining inertias requires a nonempty selection in one explicit frame")
    mass = sum(p.mass_kg for p in parts)
    anchor = parts[0].com_m
    com = anchor + sum(p.mass_kg * (p.com_m - anchor) for p in parts) / mass
    inertia = sum(p.central_inertia_kg_m2 + p.mass_kg * _shift(p.com_m - com) for p in parts)
    return MassProperties(parts[0].frame, mass, com, inertia)


def subtract(total, properties):
    """Remove disjoint selections; reject an inconsistent remainder without balancing it."""
    parts = tuple(properties)
    if not parts or any(p.frame != total.frame for p in parts):
        raise ValueError("Subtracting inertias requires selections in the total's explicit frame")
    mass = total.mass_kg - sum(p.mass_kg for p in parts)
    if mass <= 0:
        raise ValueError("Inertial remainder must have positive mass")
    delta = sum(p.mass_kg * (total.com_m - p.com_m) for p in parts) / mass
    inertia = total.central_inertia_kg_m2 - sum(
        p.central_inertia_kg_m2 + p.mass_kg * _shift(p.com_m - total.com_m) for p in parts
    )
    return MassProperties(total.frame, mass, total.com_m + delta, inertia - mass * _shift(delta))


@dataclass(frozen=True)
class SolidWorksRecord:
    record_id: str
    name: str
    configuration: str
    coordinate_system: str
    source_line: int
    volume_m3: float
    surface_area_m2: float
    properties: MassProperties
    mass_overridden: bool
    overridden_components: tuple[str, ...]

    def to_dict(self):
        return {
            **{key: value for key, value in vars(self).items() if key != "properties"},
            "properties": self.properties.to_dict(),
        }


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _number(token, scale):
    value = float(token) * scale
    rounding = float(Decimal(10) ** Decimal(token).as_tuple().exponent) * scale / 2
    return value, rounding


def _single(pattern, text, label):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one {label} in a SolidWorks report")
    return matches[0]


def _tensor(text, prefix, sign):
    values = re.findall(rf"\b{prefix}([xyz])([xyz])\s*=\s*({_NUMBER})", text)
    if len(values) != 9 or len({(a, b) for a, b, _ in values}) != 9:
        raise ValueError(f"SolidWorks {prefix} tensor needs all nine entries")
    matrix, rounding = np.empty((3, 3)), np.empty((3, 3))
    for a, b, token in values:
        i, j = "xyz".index(a), "xyz".index(b)
        matrix[i, j], rounding[i, j] = _number(token, 1e-9)
        if i != j:
            matrix[i, j] *= sign
    return matrix, rounding


def parse_solidworks_reports(text, *, source_id=None):
    """Read Chinese g/mm reports, verifying both tensor conventions and printed precision.

    Override-warning lists are retained as warnings, never treated as complete
    assembly membership. Repeated names remain separate records and frames.
    """
    if source_id is None:
        source_id = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    headings = list(re.finditer(r"^(.+?)[ \t]+的质量属性[ \t]*$", text, flags=re.MULTILINE))
    if not headings:
        raise ValueError("No SolidWorks mass-property reports found")
    records = []
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        block = text[heading.end() : end]
        record_id = f"report_{index + 1:02d}"
        notation = re.findall(r"使用([正负])张量", block)
        if len(notation) != 2 or len(set(notation)) != 1:
            raise ValueError(f"{record_id}: both tensor sign conventions must be explicit")
        sign = -1 if notation[0] == "正" else 1
        mass_label = r"质量(?:（用户覆盖）)?"
        mass, mass_error = _number(
            _single(rf"^\s*{mass_label}\s*=\s*({_NUMBER})\s*克\s*$", block, "mass in grams"),
            1e-3,
        )
        centers = [
            _number(_single(rf"^\s*{axis}\s*=\s*({_NUMBER})\s*$", block, axis), 1e-3)
            for axis in "XYZ"
        ]
        com, com_error = np.array(centers).T
        central, central_error = _tensor(block, "L", sign)
        origin, origin_error = _tensor(block, "I", sign)
        properties = MassProperties(f"{source_id}/{record_id}", mass, com, central)
        # Propagate the actual printed half-unit intervals through the parallel-axis relation.
        squares_error = 2 * abs(com) * com_error + com_error**2
        shift_error = np.empty((3, 3))
        for i in range(3):
            for j in range(3):
                shift_error[i, j] = (
                    sum(squares_error[k] for k in range(3) if k != i)
                    if i == j
                    else abs(com[i]) * com_error[j]
                    + abs(com[j]) * com_error[i]
                    + com_error[i] * com_error[j]
                )
        expected = central + mass * _shift(com)
        budget = (
            central_error
            + origin_error
            + mass * shift_error
            + mass_error * (abs(_shift(com)) + shift_error)
            + 64 * np.finfo(float).eps * np.maximum(abs(expected), abs(origin))
        )
        if (abs(origin - expected) > budget).any():
            raise ValueError(
                f"{record_id}: origin and central tensors violate the parallel-axis rule"
            )
        principal = [
            _number(_single(rf"\bP{axis}\s*=\s*({_NUMBER})", block, f"P{axis}"), 1e-9)
            for axis in "xyz"
        ]
        values, errors = np.array(principal).T
        tolerance = np.linalg.norm(central_error, 2) + max(errors)
        if (abs(np.linalg.eigvalsh(central) - np.sort(values)) > tolerance).any():
            raise ValueError(f"{record_id}: principal moments disagree with the central tensor")
        warning = "一个或多个零部件具有覆盖属性："
        components = (
            tuple(line.strip() for line in block.split(warning, 1)[1].splitlines() if line.strip())
            if warning in block
            else ()
        )
        records.append(
            SolidWorksRecord(
                record_id=record_id,
                name=heading.group(1).strip(),
                configuration=_single(r"^\s*配置[：:]\s*(.+)$", block, "configuration").strip(),
                coordinate_system=_single(
                    r"^\s*坐标系[：:]\s*(.+)$", block, "coordinate system"
                ).strip(),
                source_line=text.count("\n", 0, heading.start()) + 1,
                volume_m3=float(
                    _single(rf"^\s*体积\s*=\s*({_NUMBER})\s*立方毫米\s*$", block, "volume")
                )
                * 1e-9,
                surface_area_m2=float(
                    _single(rf"^\s*表面积\s*=\s*({_NUMBER})\s*平方毫米\s*$", block, "surface area")
                )
                * 1e-6,
                properties=properties,
                mass_overridden=bool(re.search(r"^\s*质量（用户覆盖）\s*=", block, re.MULTILINE)),
                overridden_components=components,
            )
        )
    return tuple(records)


def read_solidworks_reports(path):
    data = Path(path).expanduser().read_bytes()
    return parse_solidworks_reports(
        data.decode("utf-8-sig"), source_id=f"sha256:{hashlib.sha256(data).hexdigest()}"
    )
