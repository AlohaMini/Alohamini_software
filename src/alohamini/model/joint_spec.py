"""Versioned model coordinates; device zeros and motion certification stay separate."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType

from alohamini._validation import finite_number
from alohamini.model import RobotModel, get_robot_model
from alohamini.model.loader import _read_json

_ARM_SUFFIXES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_yaw",
    "wrist_roll",
)


@dataclass(frozen=True)
class JointSpecification:
    name: str
    host_field: str
    unit: str
    lower: float
    upper: float
    reference: float
    direction: int | None
    command_inset: float = 0.0
    feedback_tolerance: float = 0.0

    def __post_init__(self):
        for key in ("lower", "upper", "reference", "command_inset", "feedback_tolerance"):
            finite_number(getattr(self, key), key)
        if self.unit not in ("rad", "m") or self.lower >= self.upper:
            raise ValueError("Provide an ordered joint interval in rad or m")
        if not self.lower <= self.reference <= self.upper:
            raise ValueError(f"{self.name}: reference pose lies outside the model interval")
        if not 0 <= self.command_inset < (self.upper - self.lower) / 2:
            raise ValueError("Command inset must leave a nonempty interval")
        if self.feedback_tolerance < 0:
            raise ValueError("Feedback tolerance must be nonnegative")
        if self.unit == "rad" and (
            type(self.direction) is not int or self.direction not in (-1, 1)
        ):
            raise ValueError("Encoder direction must be -1 or 1")
        if self.unit == "m" and self.direction is not None:
            raise ValueError("Lift coordinates do not use the arm encoder direction")

    @property
    def command_limits(self):
        return self.lower + self.command_inset, self.upper - self.command_inset

    def feedback_status(self, position):
        """Classify actual feedback without clipping it or changing any reference."""
        finite_number(position, self.name)
        if self.lower <= position <= self.upper:
            return "inside"
        excess = max(self.lower - position, position - self.upper)
        return "boundary_tolerance" if excess <= self.feedback_tolerance + 1e-12 else "outside"


class ModelJointSpecification:
    """One fixed contract shared by both arms and every device of this model.

    Nominal coordinates, an inward command buffer and feedback tolerance have
    distinct meanings. None expands a servo limit or estimates a new zero.
    """

    def __init__(self, model: RobotModel, descriptor: dict):
        self.model = model
        self._descriptor = deepcopy(descriptor)
        data = self._descriptor
        self.geometry_model = (
            get_robot_model(data["geometry_model"]) if "geometry_model" in data else model
        )
        if (
            type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or type(data.get("specification_version")) is not int
            or data["specification_version"] not in (1, 2)
            or data.get("robot_model") != model.model_id
        ):
            raise ValueError("Unsupported or mismatched model joint specification")
        if (
            type(data.get("ticks_per_revolution")) is not int
            or data["ticks_per_revolution"] != 4096
        ):
            raise ValueError("This specification requires 4096-tick arm encoders")
        capture = data["reference_capture"]
        finite_number(capture["max_span_deg"], "reference pose stability")
        if (
            type(capture["samples"]) is not int
            or not 2 <= capture["samples"] <= 16
            or not 0 < capture["max_span_deg"] <= 2
        ):
            raise ValueError("Invalid reference-pose stability budget")
        policy = data["boundary_policy"]
        for name in ("command_inset_deg", "max_projection_deg", "feedback_tolerance_deg"):
            finite_number(policy[name], name)
            if not 0 <= policy[name] <= 2:
                raise ValueError("Model boundary budgets must lie in [0, 2] degrees")
        if (
            policy.get("automatic_zero_update") is not False
            or policy.get("automatic_limit_expansion") is not False
        ):
            raise ValueError("Boundary handling cannot update zeros or expand limits")
        if policy["command_inset_deg"] > policy["max_projection_deg"]:
            raise ValueError("Projection budget must cover the command inset")
        validation = data["validation"]
        if (
            validation.get("hardware_qualified") is not False
            or validation.get("collision_free_workspace_verified") is not False
        ):
            raise ValueError(
                "Offline validation cannot certify physical motion or collision freedom"
            )
        arms = data["arm_joints"]
        if set(arms) != set(_ARM_SUFFIXES):
            raise ValueError("Provide exactly the six named arm joints")
        joints = {}
        for side in ("left", "right"):
            for suffix, row in arms.items():
                name = f"{side}_{'wrist_yaw_joint' if suffix == 'wrist_yaw' else suffix}"
                for value in (*row["limits_deg"], row["reference_deg"]):
                    finite_number(value, name)
                lower, upper = (math.radians(x) for x in row["limits_deg"])
                joints[name] = JointSpecification(
                    name,
                    f"arm_{side}_{suffix}.pos",
                    "rad",
                    lower,
                    upper,
                    math.radians(row["reference_deg"]),
                    row["direction"],
                    math.radians(policy["command_inset_deg"]),
                    math.radians(policy["feedback_tolerance_deg"]),
                )
            grip = data["gripper"]
            finite_number(grip.get("joint_per_encoder_ratio"), "gripper encoder ratio")
            if grip.get("joint_per_encoder_ratio") != 1.0:
                raise ValueError("Gripper ratio must match the declared model mechanism")
            closed, opened = grip["closed_rad"], grip["open_rad"]
            for key, value in (("closed_rad", closed), ("open_rad", opened)):
                finite_number(value, key)
            name = f"{side}_gripper"
            joints[name] = JointSpecification(
                name,
                f"arm_{side}_gripper.pos",
                "rad",
                min(closed, opened),
                max(closed, opened),
                closed,
                grip["direction"],
            )
        lower, upper = data["lift"]["limits_m"]
        joints["vertical_move"] = JointSpecification(
            "vertical_move",
            "lift_axis.height_mm",
            "m",
            lower,
            upper,
            lower,
            None,
        )
        expected = {motor.name for motor in model.actuators if motor.name.startswith("arm_")}
        if {
            row.host_field.removesuffix(".pos") for row in joints.values() if row.unit == "rad"
        } != expected:
            raise ValueError("Specification joint names differ from the model actuators")
        self.joints = MappingProxyType(joints)
        pose = data["calibration_pose"]
        if (
            pose.get("name") != "folded_endpoints"
            or set(pose.get("arm_joints", {})) != set(_ARM_SUFFIXES)
            or pose.get("gripper") != "closed"
            or pose.get("contact") != "manual_light_link_or_housing_contact"
            or pose.get("angle_basis") != "nominal_model_endpoints_pending_hardware_qualification"
        ):
            raise ValueError("Provide the explicit, unqualified folded calibration pose")
        capture_positions = {}
        for name, row in joints.items():
            if row.unit != "rad":
                continue
            suffix = name.split("_", 1)[1].removesuffix("_joint")
            if suffix == "gripper":
                capture_positions[name] = row.reference
                continue
            endpoint = pose["arm_joints"][suffix]
            if isinstance(endpoint, dict):
                if (
                    data["specification_version"] < 2
                    or set(endpoint) != {"angle_deg", "encoder_rounding"}
                    or endpoint["encoder_rounding"] not in ("nearest", "floor", "ceil")
                ):
                    raise ValueError(f"{name}: invalid explicit calibration pose")
                finite_number(endpoint["angle_deg"], name)
                angle = math.radians(endpoint["angle_deg"])
                if not row.lower <= angle <= row.upper:
                    raise ValueError(f"{name}: calibration pose exceeds joint limits")
                capture_positions[name] = angle
                continue
            if endpoint not in ("reference", "lower", "upper"):
                raise ValueError(f"{name}: invalid calibration-pose endpoint")
            capture_positions[name] = {
                "reference": row.reference,
                "lower": row.lower,
                "upper": row.upper,
            }[endpoint]
        self.capture_positions = MappingProxyType(capture_positions)
        self.validate()

    @classmethod
    def from_model(cls, model="alohamini2pro", *, version=None):
        model = get_robot_model(model) if isinstance(model, str) else model
        if not isinstance(model, RobotModel):
            raise TypeError("Provide a robot model or registered model name")
        if version not in (None, 1):
            raise ValueError("Unsupported historical specification version")
        filename = "joint_spec.json" if version is None else "joint_spec_v1.json"
        path = model.asset_path(f"config/kinematics/{filename}")
        descriptor = _read_json(path)
        shared = descriptor.get("shared_specification")
        if shared is not None:
            if (
                set(descriptor) != {"schema_version", "robot_model", "shared_specification"}
                or descriptor["robot_model"] != model.model_id
                or not isinstance(shared, dict)
                or set(shared) != {"model", "asset"}
            ):
                raise ValueError("Invalid shared joint specification")
            source = get_robot_model(shared["model"])
            if source.model_id == model.model_id:
                raise ValueError("A shared joint specification cannot refer to itself")
            descriptor = _read_json(source.asset_path(shared["asset"]))
            if (
                descriptor.get("robot_model") != source.model_id
                or "shared_specification" in descriptor
            ):
                raise ValueError("Shared joint specification must name its source model")
            descriptor.update(robot_model=model.model_id, geometry_model=source.model_id)
        return cls(model, descriptor)

    @classmethod
    def from_signature(cls, model, signature):
        """Resolve a recorded contract without reinterpreting its encoder ranges."""
        current = cls.from_model(model)
        if current.signature == signature:
            return current
        previous = cls.from_model(model, version=1)
        if previous.signature == signature:
            return previous
        raise ValueError("Reference profile differs from any supported model specification")

    @property
    def descriptor(self):
        return deepcopy(self._descriptor)

    @property
    def signature(self):
        value = json.dumps(self._descriptor, sort_keys=True, allow_nan=False).encode()
        return hashlib.sha256(value).hexdigest()

    def validate(self):
        """Check source bounds, named poses and directions; never claim hardware testing."""
        model = self.geometry_model
        urdf = model.description_tree("collision")
        semantic = model.description_tree("semantic")
        preset = json.loads(
            model.kinematic_preset_path(self._descriptor["reference_pose"]).read_text()
        )
        checks = []
        for name, row in self.joints.items():
            node = urdf.find(f"joint[@name='{name}']")
            kind = "revolute" if row.unit == "rad" else "prismatic"
            if node is None or node.get("type") != kind:
                raise ValueError(f"{name}: missing or mismatched source joint")
            limit = node.find("limit")
            lower, upper = float(limit.get("lower")), float(limit.get("upper"))
            if row.lower < lower - 1e-10 or row.upper > upper + 1e-10:
                raise ValueError(f"{name}: specification exceeds source model limits")
            checks.append(f"{name}: interval contained in source model")
            if name.endswith("_gripper"):
                side = name.removesuffix("_gripper")
                for pose, expected in (
                    ("closed", self._descriptor["gripper"]["closed_rad"]),
                    ("open", self._descriptor["gripper"]["open_rad"]),
                ):
                    joint = semantic.find(
                        f"group_state[@group='{side}_gripper'][@name='{pose}']/joint[@name='{name}']"
                    )
                    if joint is None or not math.isclose(
                        float(joint.get("value")), expected, abs_tol=1e-10
                    ):
                        raise ValueError(f"{name}: {pose} differs from model semantics")
        for arm in preset["arms"]:
            zero = arm["model"]["joint_zero"]
            chain = _read_json(model.kinematic_chain_path(arm["model"]["chain"]))
            names = chain["joint_names"]
            if len(zero) != len(names):
                raise ValueError("Reference pose must name every model joint zero")
            offsets = dict(zip(names, zero, strict=True))
            for coordinate in arm["coordinates"]:
                host = coordinate["field"]
                matches = [row for row in self.joints.values() if row.host_field == host]
                if len(matches) != 1:
                    raise ValueError(f"{host}: reference pose lacks a unique named joint")
                row = matches[0]
                encoder = coordinate["encoder"]
                reference = encoder["reference_position_rad"] + offsets[row.name]
                if (
                    encoder["ticks_per_revolution"] != self._descriptor["ticks_per_revolution"]
                    or encoder["direction"] != row.direction
                    or not math.isclose(reference, row.reference, abs_tol=1e-10)
                ):
                    raise ValueError(f"{row.name}: reference pose or encoder direction differs")
                checks.append(f"{row.name}: explicit reference pose and direction matched")
        return {
            "passed": True,
            "robot_model": self.model.model_id,
            "specification_version": self._descriptor["specification_version"],
            "specification_sha256": self.signature,
            "calibration_pose": self._descriptor["calibration_pose"]["name"],
            "calibration_q_rad": dict(self.capture_positions),
            "checks": checks,
            **self._descriptor["validation"],
        }
