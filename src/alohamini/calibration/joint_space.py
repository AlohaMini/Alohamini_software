"""Model-fixed policy coordinates and machine-specific encoder bindings.

Standard ranges are independent of measured motor travel. They are a reversible
view of URDF positions, separate from an algorithm's statistical normalization.
This module converts values only; it never sends a command or establishes home.
"""

from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

from alohamini._validation import finite_number
from alohamini.calibration.encoder import (
    EncoderCalibration,
    HostPositionUnits,
    JointPositionDecoder,
)
from alohamini.calibration.lift import LiftCalibration
from alohamini.model import get_robot_model


@dataclass(frozen=True)
class JointCoordinate:
    name: str
    host_field: str
    unit: str
    standard_min: float
    standard_max: float
    position_at_min: float
    position_at_max: float

    def __post_init__(self):
        for key in ("standard_min", "standard_max", "position_at_min", "position_at_max"):
            finite_number(getattr(self, key), key)
        if self.standard_min >= self.standard_max or self.position_at_min == self.position_at_max:
            raise ValueError("Joint coordinate endpoints must define a nonzero range")
        if self.unit not in ("rad", "m") or not self.name or not self.host_field:
            raise ValueError("Provide named joint coordinates with rad/m units")

    @property
    def position_limits(self):
        return min(self.position_at_min, self.position_at_max), max(
            self.position_at_min, self.position_at_max
        )

    def encode(self, position):
        finite_number(position, self.name)
        return self.standard_min + (position - self.position_at_min) * (
            self.standard_max - self.standard_min
        ) / (self.position_at_max - self.position_at_min)

    def decode(self, target):
        finite_number(target, self.name)
        if not self.standard_min <= target <= self.standard_max:
            raise ValueError(f"{self.name}: target exceeds fixed standard range")
        return self.position_at_min + (target - self.standard_min) * (
            self.position_at_max - self.position_at_min
        ) / (self.standard_max - self.standard_min)


class StandardJointSpace:
    """Arm [-1,1], gripper 0=closed/1=open, lift 0=bottom/1=top.

    Arm zero is the midpoint of its fixed model interval, not the URDF Home.
    Feedback remains observable outside the interval; targets are never clipped.
    Always retain names and this descriptor with data/checkpoints.
    """

    def __init__(self, descriptor):
        if descriptor.get("format") != "alohamini-standard-joints-v1":
            raise ValueError("Unsupported standard joint-space descriptor")
        self._descriptor = deepcopy(descriptor)
        rows = [JointCoordinate(**row) for row in descriptor["joints"]]
        if not rows or len({row.name for row in rows}) != len(rows):
            raise ValueError("Standard joint names must be nonempty and unique")
        self.joints = MappingProxyType({row.name: row for row in rows})
        self.robot_model = descriptor["robot_model"]

    @classmethod
    def from_model(cls, robot_model="alohamini2pro"):
        model = get_robot_model(robot_model)
        urdf_path, semantic_path = (
            model.description_path("collision"),
            model.description_path("semantic"),
        )
        urdf, semantic = ET.parse(urdf_path).getroot(), ET.parse(semantic_path).getroot()
        rows = []
        for actuator in model.actuators:
            if not actuator.name.startswith("arm_"):
                continue
            side, suffix = actuator.name.removeprefix("arm_").split("_", 1)
            name = f"{side}_{'wrist_yaw_joint' if suffix == 'wrist_yaw' else suffix}"
            node = urdf.find(f"joint[@name='{name}']")
            if node is None or node.get("type") != "revolute":
                raise ValueError(f"Model lacks bounded revolute joint {name}")
            limit = node.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            standard_min, standard_max, start, end = -1.0, 1.0, lower, upper
            if suffix == "gripper":

                def endpoint(state, side=side, name=name):
                    pose = semantic.find(
                        f"group_state[@group='{side}_gripper'][@name='{state}']/joint[@name='{name}']"
                    )
                    if pose is None:
                        raise ValueError(f"Model lacks {name} {state} pose")
                    return float(pose.get("value"))

                standard_min, standard_max, start, end = (
                    0.0,
                    1.0,
                    endpoint("closed"),
                    endpoint("open"),
                )
                if not lower <= min(start, end) < max(start, end) <= upper:
                    raise ValueError(f"Gripper semantic poses exceed {name} limits")
            rows.append(
                JointCoordinate(
                    name, f"{actuator.name}.pos", "rad", standard_min, standard_max, start, end
                )
            )
        lift = urdf.find("joint[@name='vertical_move'][@type='prismatic']/limit")
        if lift is None:
            raise ValueError("Model lacks bounded vertical_move")
        rows.append(
            JointCoordinate(
                "vertical_move",
                "lift_axis.height_mm",
                "m",
                0.0,
                1.0,
                float(lift.get("lower")),
                float(lift.get("upper")),
            )
        )
        return cls(
            dict(
                format="alohamini-standard-joints-v1",
                robot_model=robot_model,
                asset_revision=model.asset_revision,
                model_sha256={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (urdf_path, semantic_path)
                },
                joints=[asdict(row) for row in rows],
            )
        )

    @classmethod
    def from_specification(cls, robot_model="alohamini2pro"):
        """Select the versioned common model ranges for new data and checkpoints.

        Existing descriptors and from_model() retain their recorded endpoints.
        A command buffer never changes this coordinate scale.
        """
        from alohamini.model.joint_spec import ModelJointSpecification

        spec = ModelJointSpecification.from_model(robot_model)
        descriptor = cls.from_model(spec.geometry_model.model_id).descriptor
        descriptor["robot_model"] = robot_model
        for row in descriptor["joints"]:
            joint = spec.joints[row["name"]]
            if not row["name"].endswith("_gripper"):
                row.update(position_at_min=joint.lower, position_at_max=joint.upper)
        descriptor["joint_specification"] = dict(
            version=spec.descriptor["specification_version"], sha256=spec.signature,
        )
        descriptor["boundary_policy"] = dict(
            max_projection_rad=math.radians(spec.descriptor["boundary_policy"]["max_projection_deg"]),
            command_inset_rad={name: row.command_inset for name, row in spec.joints.items()},
            feedback_tolerance={name: row.feedback_tolerance for name, row in spec.joints.items()},
        )
        return cls(descriptor)

    @property
    def descriptor(self):
        return deepcopy(self._descriptor)

    @property
    def signature(self):
        return hashlib.sha256(
            json.dumps(self._descriptor, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def _convert(self, values, method):
        unknown = set(values) - self.joints.keys()
        if unknown:
            raise ValueError(f"Unknown model joints: {sorted(unknown)}")
        return {name: getattr(self.joints[name], method)(value) for name, value in values.items()}

    def encode_positions(self, positions):
        return self._convert(positions, "encode")

    def decode_targets(self, targets):
        return self._convert(targets, "decode")


_MOTOR_FIELDS = (
    "id",
    "model",
    "normalization",
    "drive_mode",
    "range_min",
    "range_max",
    "homing_offset",
)


@dataclass(frozen=True)
class TargetProjection:
    """Requested and executable physical positions, suitable for recording together."""

    requested: dict[str, float]
    positions: dict[str, float]

    def __post_init__(self):
        if self.requested.keys() != self.positions.keys():
            raise ValueError("Projected targets must retain every requested joint")
        for mapping in (self.requested, self.positions):
            for name, value in mapping.items():
                finite_number(value, name)
        object.__setattr__(self, "requested", MappingProxyType(dict(self.requested)))
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))

    @property
    def adjustments(self):
        return {
            name: dict(requested=value, applied=self.positions[name])
            for name, value in self.requested.items() if value != self.positions[name]
        }

    def to_dict(self):
        return dict(
            requested=dict(self.requested), applied=dict(self.positions),
            adjustments=self.adjustments,
        )


class DeviceJointSpace:
    """Bind one model standard to one machine's post-offset encoders.

    Device travel may narrow the admissible targets, never rescale the standard.
    Incoming metadata must still match the binding. Prefer raw post-offset tick
    feedback when available, since Host normalized feedback may already saturate.
    """

    def __init__(self, standard, encoders, lift, robot_metadata):
        if robot_metadata["robot_model"] != standard.robot_model:
            raise ValueError("Host robot model differs from the standard")
        required = {n for n, row in standard.joints.items() if row.unit == "rad"}
        if set(encoders) != required:
            raise ValueError("Provide independent calibration for every named arm joint")
        lower, upper = standard.joints["vertical_move"].position_limits
        if not lower <= lift.position_min_m < lift.position_max_m <= upper:
            raise ValueError("Lift calibration exceeds fixed model endpoints")
        if not math.isclose(
            lift.physical_max_m - lift.physical_min_m,
            lift.position_max_m - lift.position_min_m,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Lift calibration must preserve metre scale, not stretch travel")
        self.standard, self.lift = standard, lift
        self.encoders = {}
        for name, encoder in encoders.items():
            lower, upper = standard.joints[name].position_limits
            if encoder.position_min_rad is not None:
                lower, upper = (
                    max(lower, encoder.position_min_rad),
                    min(upper, encoder.position_max_rad),
                )
            self.encoders[name] = replace(encoder, position_min_rad=lower, position_max_rad=upper)
        self.encoders = MappingProxyType(self.encoders)
        self.decoder = JointPositionDecoder(self.encoders)
        self._metadata = deepcopy(robot_metadata)
        self.units = {
            name: HostPositionUnits(
                **{
                    k: self._motor(name, robot_metadata)[k]
                    for k in ("normalization", "range_min", "range_max", "drive_mode")
                }
            )
            for name in self.encoders
        }

    def _motor(self, name, metadata):
        motor = self.standard.joints[name].host_field.removesuffix(".pos")
        return metadata["motors"][motor]

    def validate_host(self, metadata):
        if metadata["robot_model"] != self.standard.robot_model:
            self.reset()
            raise ValueError("Host robot model changed")
        for name in self.encoders:
            expected, actual = self._motor(name, self._metadata), self._motor(name, metadata)
            if any(expected.get(key) != actual.get(key) for key in _MOTOR_FIELDS):
                self.reset()
                raise ValueError(f"{name}: Host motor calibration changed; bind the device again")

    @classmethod
    def from_mappings(cls, standard, arm_mappings, lift_mapping, robot_metadata):
        encoders = {}
        for side in ("left", "right"):
            mapping = arm_mappings[side]
            if mapping["robot_model"] != standard.robot_model or mapping["side"] != side:
                raise ValueError("Arm mapping model/side differs from the standard")
            for name, row in standard.joints.items():
                if not name.startswith(f"{side}_"):
                    continue
                suffix = name.removeprefix(f"{side}_")
                joint = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
                entry = mapping["joints"][joint]
                motor = robot_metadata["motors"][row.host_field.removesuffix(".pos")]
                captured = entry.get("motor_calibration", {})
                if any(
                    key not in captured
                    for key in ("homing_offset", "range_min", "range_max", "drive_mode")
                ):
                    raise ValueError(f"{name}: mapping lacks captured motor_calibration")
                if any(
                    captured[key] != motor[key]
                    for key in ("homing_offset", "range_min", "range_max", "drive_mode")
                ) or any(entry[key] != motor[key] for key in ("id", "model")):
                    raise ValueError(f"{name}: mapping belongs to different motor calibration")
                if joint == "gripper" and (entry["urdf_closed_rad"], entry["urdf_open_rad"]) != (
                    row.position_at_min,
                    row.position_at_max,
                ):
                    raise ValueError(f"{name}: gripper endpoints differ from model semantics")
                encoders[name] = EncoderCalibration(
                    mapping["ticks_per_revolution"],
                    entry["reference_tick"],
                    entry["reference_q_rad"],
                    entry["sign"],
                    entry.get("joint_per_encoder_ratio", 1.0),
                    entry.get("safe_q_min_rad"),
                    entry.get("safe_q_max_rad"),
                )
        if lift_mapping["robot_model"] != standard.robot_model:
            raise ValueError("Lift mapping model differs from the standard")
        mechanism, urdf = lift_mapping["mechanism"], lift_mapping["urdf"]
        lift = LiftCalibration(
            mechanism["physical_min_mm"] / 1000,
            mechanism["physical_max_mm"] / 1000,
            urdf["q_at_physical_min_m"],
            urdf["q_at_physical_max_m"],
        )
        return cls(standard, encoders, lift, robot_metadata)

    def reset(self):
        self.decoder.reset()

    def positions_from_observation(self, observation, *, robot_metadata, post_offset_ticks=None):
        self.validate_host(robot_metadata)
        if post_offset_ticks is None:
            ticks = {
                name: self.units[name].to_tick(observation[self.standard.joints[name].host_field])
                for name in self.encoders
            }
        else:
            if set(post_offset_ticks) != set(self.encoders):
                raise ValueError("Raw feedback must contain every named arm joint")
            ticks = post_offset_ticks
        result = self.decoder.decode(ticks)
        result["vertical_move"] = self.lift.height_to_position(
            observation["lift_axis.height_mm"] / 1000
        )
        return result

    def project_positions(self, positions, *, robot_metadata, tolerance_rad=None):
        """Project small endpoint discrepancies inward without changing coordinates.

        The explicit result retains requested/applied values. Feedback, encoder
        zeros, firmware limits and checkpoint statistics are never modified.
        No mechanical endpoint is inferred from stalls or contact currents.
        """
        self.validate_host(robot_metadata)
        policy = self.standard.descriptor.get("boundary_policy", {})
        tolerance = (
            policy.get("max_projection_rad", 0.0) if tolerance_rad is None else tolerance_rad
        )
        finite_number(tolerance, "tolerance_rad")
        if not 0 <= tolerance <= math.radians(2) + 1e-12:
            raise ValueError("Endpoint projection tolerance must lie in [0, 2] degrees")
        unknown = set(positions) - self.standard.joints.keys()
        if not positions or unknown:
            raise ValueError(f"Provide nonempty known model joint targets: {sorted(unknown)}")
        applied = {}
        for name, position in positions.items():
            finite_number(position, name)
            row = self.standard.joints[name]
            lower, upper = row.position_limits
            inset = policy.get("command_inset_rad", {}).get(name, 0.0)
            finite_number(inset, "command inset")
            if not 0 <= inset < (upper - lower) / 2:
                raise ValueError(f"{name}: invalid inward command buffer")
            lower, upper = lower + inset, upper - inset
            if name == "vertical_move":
                lower, upper = (
                    max(lower, self.lift.position_min_m), min(upper, self.lift.position_max_m)
                )
                if not lower <= position <= upper:
                    raise ValueError("Angular tolerance cannot expand the lift interval")
                applied[name] = position
                continue
            encoder, units = self.encoders[name], self.units[name]
            lower = max(lower, encoder.position_min_rad)
            upper = min(upper, encoder.position_max_rad)
            if lower > upper:
                raise ValueError(f"{name}: no executable interval remains")
            if upper - lower > encoder.joint_period_rad + 1e-10:
                raise ValueError(f"{name}: projection requires an unambiguous encoder branch")
            scale = encoder.direction * encoder.joint_period_rad / encoder.ticks_per_revolution
            raw = [
                encoder.reference_tick + (q - encoder.reference_position_rad) / scale
                for q in (lower, upper)
            ]
            period = encoder.ticks_per_revolution
            first = math.ceil((min(raw) - units.range_max) / period - 1e-12)
            last = math.floor((max(raw) - units.range_min) / period + 1e-12)
            candidates = []
            for turn in range(first, last + 1):
                ends = [
                    encoder.reference_position_rad
                    + (tick + turn * period - encoder.reference_tick) * scale
                    for tick in (units.range_min, units.range_max)
                ]
                start, end = max(lower, min(ends)), min(upper, max(ends))
                if start > end:
                    continue
                candidate = max(start, min(end, position))
                tick = encoder.position_to_tick(candidate)
                if units.range_min <= tick <= units.range_max:
                    candidates.append(candidate)
            if not candidates:
                raise ValueError(f"{name}: no target is representable within installed Host limits")
            selected = min(candidates, key=lambda q: abs(q - position))
            if abs(selected - position) > tolerance + 1e-12:
                raise ValueError(f"{name}: target exceeds endpoint projection tolerance")
            applied[name] = selected
        return TargetProjection(dict(positions), applied)

    def host_targets_from_positions(self, positions, *, robot_metadata):
        self.validate_host(robot_metadata)
        # Validate model endpoints independently of the narrower machine bounds.
        self.standard.decode_targets(self.standard.encode_positions(positions))
        targets = {}
        for name, position in positions.items():
            if name == "vertical_move":
                targets["lift_axis.height_mm"] = self.lift.position_to_height(position) * 1000
                continue
            tick = self.encoders[name].position_to_tick(position)
            units = self.units[name]
            if not units.range_min <= tick <= units.range_max:
                raise ValueError(f"{name}: target exceeds installed Host encoder range")
            value = units.from_tick(tick)
            # Host truncates. When floating-point inversion lands just below a
            # tick boundary, select the centre of that same quantization bin.
            if units.to_tick(value) != tick:
                half_step = (100 if units.normalization == "range_m100_100" else 50) / (
                    units.range_max - units.range_min
                )
                if units.normalization == "degrees":
                    half_step = 180 / 4095
                elif units.drive_mode:
                    half_step = -half_step
                value += half_step
            if units.to_tick(value) != tick:
                raise ValueError(f"{name}: Host normalization cannot represent target tick")
            targets[self.standard.joints[name].host_field] = value
        return targets

    def host_targets_from_standard(self, targets, *, robot_metadata):
        return self.host_targets_from_positions(
            self.standard.decode_targets(targets), robot_metadata=robot_metadata
        )
