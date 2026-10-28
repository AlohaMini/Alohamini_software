"""Named joint-to-TCP derived fields, independent of algorithms and Torch.

Descriptors freeze source coordinates, geometry, reference frames and gripper
mapping. Relative action encoding is a separate, composable data transform.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.kinematics.arm import ARM_SUFFIXES, SerialArmKinematics
from alohamini.model import get_robot_model, load_robot_model

from .action_transforms import RelativeActions

HOST_UNIT_KEYS = ("normalization", "range_min", "range_max", "drive_mode")
MOTOR_COORDINATE_KEYS = (*HOST_UNIT_KEYS, "homing_offset", "id", "model")


class KinematicFields:
    def __init__(self, descriptor):
        self.descriptor = deepcopy(descriptor)
        descriptor = self.descriptor
        if descriptor.get("version") != 1:
            raise ValueError("Unsupported kinematic-fields descriptor")
        self.names = list(descriptor["source_names"])
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Kinematic source names must be nonempty and unique")
        self.arms, self.arm_offsets, used = [], [], []
        self.feature_names, self.units = [], []
        for arm in descriptor["arms"]:
            coordinates = arm["coordinates"]
            geometry = SerialArmKinematics.from_parameters(arm["geometry"])
            if (
                len(coordinates) != len(geometry.rows)
                or not arm.get("frame")
                or not arm.get("name")
            ):
                raise ValueError("Each arm needs named coordinates for its DH rows and a frame")
            indices = [self.names.index(c["field"]) for c in coordinates]
            for coordinate in coordinates:
                if coordinate["unit"] in ("host", "encoder"):
                    EncoderCalibration(**coordinate["encoder"])
                    if coordinate["unit"] == "host":
                        HostPositionUnits(**coordinate["host_units"])
                elif coordinate["unit"] not in ("rad", "deg"):
                    raise ValueError(
                        "Joint unit must be model rad/deg or "
                        "explicitly calibrated host/encoder units"
                    )
            grip = arm.get("gripper")
            gi = None
            if grip is not None:
                endpoints = np.array(
                    [grip[k] for k in ("source_closed", "source_open", "closed", "open")]
                )
                if (
                    not np.isfinite(endpoints).all()
                    or endpoints[0] == endpoints[1]
                    or endpoints[2] == endpoints[3]
                    or not grip.get("unit")
                ):
                    raise ValueError("Gripper mapping needs finite distinct endpoints and a unit")
                gi = self.names.index(grip["field"])
                used.append(gi)
            used.extend(indices)
            self.arms.append((arm, indices, gi, geometry))
            self.arm_offsets.append(len(self.feature_names))
            self.feature_names += [f"{arm['name']}.{k}" for k in ("x", "y", "z", "rx", "ry", "rz")]
            self.units += ["m"] * 3 + ["rad"] * 3
            if grip is not None:
                self.feature_names.append(f"{arm['name']}.gripper")
                self.units.append(grip["unit"])
        passthrough = list(descriptor.get("passthrough", []))
        self.passthrough = [self.names.index(k) for k in passthrough]
        used += self.passthrough
        if not self.arms or sorted(used) != list(range(len(self.names))):
            raise ValueError(
                "Arm and passthrough fields must cover source coordinates exactly once"
            )
        self.pose_width = len(self.feature_names)
        self.width = self.pose_width + len(self.passthrough)
        self.feature_names += passthrough
        self.units += list(descriptor.get("passthrough_units", []))
        if len(set(self.feature_names)) != self.width:
            raise ValueError("Derived field names must be unique")
        if len(self.units) != self.width:
            raise ValueError("Declare one unit for every passthrough field")

    def _values(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1:] != (len(self.names),) or not np.isfinite(values).all():
            raise ValueError("Expected the declared finite source coordinates")
        return values

    @staticmethod
    def joint(value, coordinate, previous=None):
        if coordinate["unit"] == "rad":
            return float(value)
        if coordinate["unit"] == "deg":
            return float(np.deg2rad(value))
        if coordinate["unit"] == "host":
            tick = HostPositionUnits(**coordinate["host_units"]).to_tick(float(value))
        else:
            if not np.isfinite(value) or float(value) != int(value):
                raise ValueError("Encoder positions must be integer post-offset ticks")
            tick = int(value)
        return EncoderCalibration(**coordinate["encoder"]).position_from_tick(
            tick, previous_position_rad=previous
        )

    @staticmethod
    def source_joint(value, coordinate):
        if coordinate["unit"] == "rad":
            return float(value)
        if coordinate["unit"] == "deg":
            return float(np.rad2deg(value))
        tick = EncoderCalibration(**coordinate["encoder"]).position_to_tick(float(value))
        if coordinate["unit"] == "encoder":
            return tick
        units = HostPositionUnits(**coordinate["host_units"])
        if not units.range_min <= tick <= units.range_max:
            raise ValueError("IK joint target exceeds the installed Host encoder range")
        return units.from_tick(tick)

    @staticmethod
    def gripper(value, grip, inverse=False):
        keys = (
            ("closed", "open", "source_closed", "source_open")
            if inverse
            else ("source_closed", "source_open", "closed", "open")
        )
        low, high, out_low, out_high = (grip[k] for k in keys)
        return (value - low) / (high - low) * (out_high - out_low) + out_low

    def absolute(self, values):
        values = self._values(values)
        result = np.empty((*values.shape[:-1], self.width), dtype=np.float64)
        if not values.size:
            return result
        for offset, (arm, indices, gi, model) in zip(self.arm_offsets, self.arms, strict=True):
            joints = np.empty((*values.shape[:-1], len(indices)), dtype=np.float64)
            for j, (i, coordinate) in enumerate(zip(indices, arm["coordinates"], strict=True)):
                if coordinate["unit"] == "rad":
                    joints[..., j] = values[..., i]
                elif coordinate["unit"] == "deg":
                    joints[..., j] = np.deg2rad(values[..., i])
                else:
                    units = (
                        HostPositionUnits(**coordinate["host_units"])
                        if coordinate["unit"] == "host"
                        else None
                    )
                    encoder = EncoderCalibration(**coordinate["encoder"])
                    source = values[..., i].reshape(-1)
                    decoded = np.fromiter(
                        (
                            encoder.position_from_tick(units.to_tick(float(v)))
                            if units is not None
                            else self.joint(v, coordinate)
                            for v in source
                        ),
                        dtype=np.float64,
                        count=len(source),
                    )
                    joints[..., j] = decoded.reshape(values.shape[:-1])
            poses = model.forward_batch(joints)
            result[..., offset : offset + 3] = poses[..., :3, 3]
            result[..., offset + 3 : offset + 6] = (
                Rotation.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3))
                .as_rotvec()
                .reshape((*values.shape[:-1], 3))
            )
            if gi is not None:
                result[..., offset + 6] = self.gripper(values[..., gi], arm["gripper"])
        if self.passthrough:
            result[..., self.pose_width :] = values[..., self.passthrough]
        return result

    def relative_actions(self):
        """Declare pose groups from the field layout; grippers and passthrough stay absolute."""
        return RelativeActions(
            self.width,
            linear_dims=[i for start in self.arm_offsets for i in range(start, start + 3)],
            rotation_groups=[list(range(start + 3, start + 6)) for start in self.arm_offsets],
        )

    def validate_source(self, info, *, state_names=None):
        """Check source names and recorded encoder coordinates before statistics or inference."""
        if info["features"]["action"]["names"] != self.names or (
            state_names is not None and state_names != self.names
        ):
            raise ValueError(
                "Kinematic source field names/order differ from selected state/actions"
            )
        metadata = info.get("robot_metadata")
        if metadata is None:
            return
        if self.descriptor.get("robot_model", metadata["robot_model"]) != metadata["robot_model"]:
            raise ValueError("Kinematic robot model differs from the recording")
        bindings = self.descriptor.get("source_calibration", {})
        for arm, _, _, _ in self.arms:
            for coordinate in arm["coordinates"]:
                motor = coordinate.get("motor", coordinate["field"].removesuffix(".pos"))
                recorded = metadata["motors"].get(motor)
                if coordinate["unit"] == "host":
                    if recorded is None or any(
                        coordinate["host_units"][key] != recorded[key] for key in HOST_UNIT_KEYS
                    ):
                        raise ValueError(
                            f"Kinematic Host units differ from recorded motor: {motor}"
                        )
                    if motor not in bindings:
                        raise ValueError(
                            f"Host coordinate requires a source_calibration binding: {motor}"
                        )
                elif recorded is not None and coordinate["field"] == motor + ".pos":
                    raise ValueError(
                        f"Host position is not a model rad/deg or raw encoder field: {motor}"
                    )
        for motor, expected in bindings.items():
            recorded = metadata["motors"].get(motor)
            if (
                recorded is None
                or any(
                    key not in expected or expected[key] != recorded.get(key)
                    for key in (*HOST_UNIT_KEYS, "homing_offset")
                )
                or any(
                    expected[key] != recorded.get(key) for key in ("id", "model") if key in expected
                )
            ):
                raise ValueError(f"Kinematic source calibration differs from recording: {motor}")

    def derive(self, row, fields):
        """Add named absolute-TCP fields without changing recorded values or pairing."""
        if len(set(fields.values())) != len(fields) or set(fields.values()) & set(row):
            raise ValueError(
                "Derived field names must be unique and must not overwrite recorded fields"
            )
        result = dict(row)
        for source, destination in fields.items():
            result[destination] = self.absolute(row[source])
        return result

    def feature(self):
        return dict(
            dtype="float64",
            shape=[self.width],
            names=list(self.feature_names),
            units=list(self.units),
            representation="absolute_tcp_rotvec",
            frames={arm["name"]: arm["frame"] for arm, _, _, _ in self.arms},
            derivation=deepcopy(self.descriptor),
        )

    def restore_absolute(self, poses, reference):
        """Solve trajectories with continuous IK seeds, restoring original field order."""
        reference = self._values(reference)
        poses = np.asarray(poses, dtype=np.float64)
        if poses.shape[-1:] != (self.width,) or not np.isfinite(poses).all():
            raise ValueError("Expected finite absolute TCP fields")
        if poses.ndim != reference.ndim + 1 or poses.shape[:-2] != reference.shape[:-1]:
            raise ValueError("TCP trajectories must match their original reference state")
        output = np.repeat(reference[..., None, :], poses.shape[-2], axis=-2)
        for index in np.ndindex(reference.shape[:-1]):
            for offset, (arm, indices, gi, model) in zip(self.arm_offsets, self.arms, strict=True):
                seed = np.array(
                    [
                        self.joint(reference[index][i], c)
                        for i, c in zip(indices, arm["coordinates"], strict=True)
                    ]
                )
                for step, pose in enumerate(poses[index]):
                    target = np.eye(4)
                    target[:3, 3] = pose[offset : offset + 3]
                    target[:3, :3] = Rotation.from_rotvec(pose[offset + 3 : offset + 6]).as_matrix()
                    seed = model.inverse(target, seed, **self.descriptor.get("ik", {}))
                    for i, q, c in zip(indices, seed, arm["coordinates"], strict=True):
                        output[index + (step, i)] = self.source_joint(q, c)
                    if gi is not None:
                        output[index + (step, gi)] = self.gripper(
                            pose[offset + 6], arm["gripper"], inverse=True
                        )
            for n, i in enumerate(self.passthrough):
                output[index + (slice(None), i)] = poses[index + (slice(None), self.pose_width + n)]
        return output


class KinematicActions:
    """Algorithm-independent composition of FK and same-anchor relative pose targets."""

    def __init__(self, descriptor):
        self.geometry = KinematicFields(descriptor)
        self.relative = self.geometry.relative_actions()

    def encode(self, state, actions=None):
        observation = self.geometry.absolute(state)
        targets = (
            None
            if actions is None
            else self.relative.encode(observation, self.geometry.absolute(actions))
        )
        return observation, targets

    def decode(self, state, actions):
        anchor = self.geometry.absolute(state)
        return self.geometry.restore_absolute(self.relative.decode(anchor, actions), state)


def descriptor_from_spec(info, spec, *, directory=None):
    """Freeze explicitly named coordinates and inline or registered model geometry."""
    descriptor = deepcopy(spec)
    descriptor.setdefault("source_names", list(info["features"]["action"]["names"]))
    for arm in descriptor["arms"]:
        reference = arm.pop("model", None)
        if reference is not None:
            if "geometry" in arm or ("id" in reference) == ("directory" in reference):
                raise ValueError("Choose inline geometry or one model ID/directory and chain")
            model = reference.get("id")
            if model is None:
                path = Path(reference["directory"]).expanduser()
                if not path.is_absolute():
                    path = Path(directory or ".") / path
                model = load_robot_model(path)
            geometry = SerialArmKinematics.from_model(model, reference["chain"])
            if arm.get("frame", geometry.frame) != geometry.frame:
                raise ValueError("Changing a model frame requires an explicit geometry transform")
            arm["frame"] = geometry.frame
            provenance = geometry.model_reference
            if "joint_zero" in reference:
                geometry = geometry.with_joint_zero(reference["joint_zero"])
                provenance = {**provenance, "joint_zero": list(reference["joint_zero"])}
            arm["geometry"] = geometry.parameters()
            arm["model_reference"] = provenance
    fields = KinematicFields(descriptor)
    fields.validate_source(info)
    return descriptor


def descriptor_from_reference_pose(info, name, *, model=None):
    """Bind an explicitly selected Host homing protocol to this recording's motor units."""
    metadata = info.get("robot_metadata")
    if not metadata:
        raise ValueError("Reference-pose conversion needs recorded Host motor metadata")
    model = get_robot_model(metadata["robot_model"]) if model is None else model
    if model.model_id != metadata["robot_model"]:
        raise ValueError("Reference-pose model differs from the recording")
    spec = json.loads(model.kinematic_preset_path(name).read_text())
    if spec.get("robot_model") != model.model_id:
        raise ValueError("Reference-pose preset and model IDs disagree")
    bindings = {}
    for arm in spec["arms"]:
        reference = arm.get("model", {})
        if reference.get("id") == model.model_id:
            reference.pop("id")
            reference["directory"] = str(model.directory)
        for coordinate in arm["coordinates"]:
            if coordinate["unit"] != "host":
                raise ValueError("Host reference-pose presets require explicit host coordinates")
            motor = coordinate.get("motor", coordinate["field"].removesuffix(".pos"))
            captured = metadata["motors"][motor]
            if "homing_offset" not in captured:
                raise ValueError(f"Recording lacks the homing offset for {motor}")
            units = HostPositionUnits(**{key: captured[key] for key in HOST_UNIT_KEYS})
            encoder = coordinate["encoder"]
            reference_tick = encoder["reference_tick"]
            if not units.range_min <= reference_tick <= units.range_max:
                raise ValueError(f"Reference-pose tick is outside recorded motor range: {motor}")
            scale = (
                encoder["direction"]
                * 2
                * np.pi
                / encoder["ticks_per_revolution"]
                * encoder.get("joint_per_encoder_ratio", 1.0)
            )
            limits = sorted(
                encoder["reference_position_rad"] + (tick - reference_tick) * scale
                for tick in (units.range_min, units.range_max)
            )
            encoder.update(position_min_rad=limits[0], position_max_rad=limits[1])
            coordinate["host_units"] = asdict(units)
            bindings[motor] = {
                key: captured[key] for key in MOTOR_COORDINATE_KEYS if key in captured
            }
        grip = arm.get("gripper")
        if grip is not None:
            motor = grip.get("motor", grip["field"].removesuffix(".pos"))
            captured = metadata["motors"][motor]
            if captured["normalization"] != grip["source_normalization"]:
                raise ValueError(f"Reference-pose gripper units differ from recording: {motor}")
            bindings[motor] = {
                key: captured[key] for key in MOTOR_COORDINATE_KEYS if key in captured
            }
    spec["source_calibration"] = bindings
    descriptor = descriptor_from_spec(info, spec, directory=model.directory)
    for arm in descriptor["arms"]:
        # Installed ranges belong to this recording, not a prior device's URDF limits.
        arm["geometry"]["joint_limits"] = [
            [c["encoder"]["position_min_rad"], c["encoder"]["position_max_rad"]]
            for c in arm["coordinates"]
        ]
    descriptor["limit_source"] = "recorded_host_encoder_range"
    KinematicFields(descriptor).validate_source(info)
    return descriptor


def descriptor_from_host(info, mappings):
    """Import the existing two-arm Host YAML format, with recorded calibration checks."""
    metadata = info.get("robot_metadata")
    if not metadata:
        raise ValueError(
            "Host kinematics requires recorded motor units and installed joint mappings"
        )
    robot_model = metadata["robot_model"]
    names = info["features"]["action"]["names"]
    state = info["features"].get("observation.state")
    if state is not None and state["names"] != names:
        raise ValueError("Host state/action names must agree")
    arms, consumed, bindings = [], [], {}
    for side in ("left", "right"):
        mapping = mappings[side]
        if (
            mapping.get("robot_model", robot_model) != robot_model
            or mapping.get("side", side) != side
        ):
            raise ValueError("Imported mapping model/side differs from the recording")
        geometry = SerialArmKinematics.from_model(robot_model, side)
        if len(geometry.rows) != len(ARM_SUFFIXES):
            raise ValueError(
                "The two-arm YAML importer requires six revolute joints per arm; use --spec"
            )
        coordinates = []
        for index, suffix in enumerate((*ARM_SUFFIXES, "gripper")):
            joint = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
            field = f"arm_{side}_{joint}.pos"
            entry = mapping["joints"][joint]
            motor = field.removesuffix(".pos")
            recorded = metadata["motors"][motor]
            captured = entry.get("motor_calibration", {})
            required = ("homing_offset", "range_min", "range_max", "drive_mode")
            if any(key not in captured for key in required):
                raise ValueError(
                    f"Imported mapping lacks motor_calibration for {motor}; "
                    "provide the mapping's recorded encoder coordinates or a bound --spec"
                )
            if any(captured[key] != recorded[key] for key in required) or any(
                entry[key] != recorded.get(key) for key in ("id", "model") if key in entry
            ):
                raise ValueError(
                    f"Imported mapping motor_calibration differs from recording: {motor}"
                )
            bindings[motor] = {
                key: recorded[key] for key in MOTOR_COORDINATE_KEYS if key in recorded
            }
            limits = (
                sorted([entry["urdf_closed_rad"], entry["urdf_open_rad"]])
                if joint == "gripper"
                else geometry.joint_limits[index]
            )
            if entry.get("safe_q_min_rad") is not None or entry.get("safe_q_max_rad") is not None:
                limits = (entry.get("safe_q_min_rad"), entry.get("safe_q_max_rad"))
            calibration = EncoderCalibration(
                mapping["ticks_per_revolution"],
                entry["reference_tick"],
                entry["reference_q_rad"],
                entry["sign"],
                entry.get("joint_per_encoder_ratio", 1.0),
                limits[0],
                limits[1],
            )
            units = HostPositionUnits(
                **{
                    k: metadata["motors"][field[:-4]][k]
                    for k in ("normalization", "range_min", "range_max", "drive_mode")
                }
            )
            consumed.append(field)
            if joint == "gripper":
                gripper = dict(
                    field=field,
                    motor=motor,
                    source_closed=units.from_tick(
                        calibration.position_to_tick(entry["urdf_closed_rad"])
                    ),
                    source_open=units.from_tick(
                        calibration.position_to_tick(entry["urdf_open_rad"])
                    ),
                    closed=0.0,
                    open=0.1,
                    unit="canonical_0_0.1",
                )
            else:
                coordinates.append(
                    dict(
                        field=field,
                        motor=motor,
                        unit="host",
                        host_units=asdict(units),
                        encoder=asdict(calibration),
                    )
                )
        arms.append(
            dict(
                name=side,
                frame=geometry.frame,
                coordinates=coordinates,
                gripper=gripper,
                geometry=geometry.parameters(),
                model_reference=geometry.model_reference,
            )
        )
    extra = [n for n in names if n not in consumed]
    known_units = {
        "x.vel": "m/s",
        "y.vel": "m/s",
        "theta.vel": "deg/s",
        "lift_axis.height_mm": "mm",
    }
    descriptor = dict(
        version=1,
        robot_model=robot_model,
        source_names=names,
        arms=arms,
        passthrough=extra,
        passthrough_units=[known_units[n] for n in extra],
        source_calibration=bindings,
    )
    KinematicFields(descriptor).validate_source(info)
    return descriptor


def descriptor_from_dataset(root, mappings=None, *, spec=None, directory=None, reference_pose=None):
    root = Path(root).expanduser()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") == "v3.0":
        sidecar = root / "meta/alohamini.json"
        if not sidecar.is_file():
            if spec is None:
                raise ValueError(
                    "Original v3 requires an explicit coordinate descriptor (--spec); "
                    "Host metadata is unavailable"
                )
        else:
            source = json.loads(sidecar.read_text())["source_info"]
            if info["features"]["action"] != source["features"]["action"]:
                raise ValueError("Exported action fields differ from their Host annotation")
            info["robot_metadata"] = source["robot_metadata"]
    if reference_pose is not None:
        if mappings is not None or spec is not None:
            raise ValueError("Choose a reference pose, generic spec or two-arm YAML importer")
        return descriptor_from_reference_pose(info, reference_pose)
    if spec is not None:
        if mappings is not None:
            raise ValueError("Choose a generic spec or the two-arm YAML importer")
        return descriptor_from_spec(info, spec, directory=directory)
    if mappings is None:
        raise ValueError("Provide --spec or explicit two-arm mappings")
    return descriptor_from_host(info, mappings)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze named joint-to-TCP derived-field geometry")
    parser.add_argument("--dataset", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--spec",
        type=Path,
        help="Named coordinates and inline geometry or model/chain references (JSON)",
    )
    source.add_argument(
        "--left-map", type=Path, help="Import the existing two-arm Host YAML format"
    )
    source.add_argument(
        "--reference-pose", help="Explicit model homing-pose protocol used during recording"
    )
    parser.add_argument("--right-map", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if bool(args.left_map) != bool(args.right_map):
        parser.error("--left-map and --right-map must be supplied together")
    if args.reference_pose:
        descriptor = descriptor_from_dataset(args.dataset, reference_pose=args.reference_pose)
    elif args.spec:
        spec_path = args.spec.expanduser().resolve()
        descriptor = descriptor_from_dataset(
            args.dataset, spec=json.loads(spec_path.read_text()), directory=spec_path.parent
        )
    else:
        import yaml

        mappings = {
            side: yaml.safe_load(getattr(args, side + "_map").expanduser().read_text())
            for side in ("left", "right")
        }
        descriptor = descriptor_from_dataset(args.dataset, mappings)
    with args.output.expanduser().open("x") as stream:
        json.dump(descriptor, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
