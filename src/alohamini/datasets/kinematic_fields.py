"""Named joint-to-TCP derived fields, independent of algorithms and Torch.

Descriptors freeze source coordinates, geometry, reference frames and gripper
mapping. Delta actions and normalization remain policy-processor operations.
"""

import argparse
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from alohamini.calibration.encoder import EncoderCalibration, HostPositionUnits
from alohamini.kinematics.arm import ARM_SUFFIXES, AlohaMiniArmKinematics


class KinematicFields:
    def __init__(self, descriptor):
        self.descriptor = deepcopy(descriptor)
        if descriptor.get("version") != 1:
            raise ValueError("Unsupported kinematic-fields descriptor")
        self.names = list(descriptor["source_names"])
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Kinematic source names must be nonempty and unique")
        self.arms, used = [], []
        self.feature_names, self.units = [], []
        for arm in descriptor["arms"]:
            coordinates = arm["coordinates"]
            if len(coordinates) != 6 or not arm.get("frame"):
                raise ValueError("Each arm needs six named joints and a reference frame")
            indices = [self.names.index(c["field"]) for c in coordinates]
            for coordinate in coordinates:
                if coordinate["unit"] == "host":
                    HostPositionUnits(**coordinate["host_units"])
                    EncoderCalibration(**coordinate["encoder"])
                elif coordinate["unit"] != "rad":
                    raise ValueError("Joint unit must be rad or explicitly calibrated host units")
            grip = arm["gripper"]
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
            used.extend([*indices, gi])
            geometry = AlohaMiniArmKinematics.from_parameters(arm["geometry"])
            self.arms.append((arm, indices, gi, geometry))
            self.feature_names += [
                f"{arm['name']}.{k}" for k in ("x", "y", "z", "rx", "ry", "rz", "gripper")
            ]
            self.units += ["m"] * 3 + ["rad"] * 3 + [grip["unit"]]
        passthrough = list(descriptor.get("passthrough", []))
        self.passthrough = [self.names.index(k) for k in passthrough]
        used += self.passthrough
        if not self.arms or sorted(used) != list(range(len(self.names))):
            raise ValueError(
                "Arm and passthrough fields must cover source coordinates exactly once"
            )
        self.width = 7 * len(self.arms) + len(self.passthrough)
        self.feature_names += passthrough
        self.units += list(descriptor.get("passthrough_units", []))
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
        tick = HostPositionUnits(**coordinate["host_units"]).to_tick(float(value))
        return EncoderCalibration(**coordinate["encoder"]).position_from_tick(
            tick, previous_position_rad=previous
        )

    @staticmethod
    def source_joint(value, coordinate):
        if coordinate["unit"] == "rad":
            return float(value)
        tick = EncoderCalibration(**coordinate["encoder"]).position_to_tick(float(value))
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
        for a, (arm, indices, gi, model) in enumerate(self.arms):
            joints = np.empty((*values.shape[:-1], 6), dtype=np.float64)
            for j, (i, coordinate) in enumerate(zip(indices, arm["coordinates"], strict=True)):
                if coordinate["unit"] == "rad":
                    joints[..., j] = values[..., i]
                else:
                    units = HostPositionUnits(**coordinate["host_units"])
                    encoder = EncoderCalibration(**coordinate["encoder"])
                    source = values[..., i].reshape(-1)
                    decoded = np.fromiter(
                        (encoder.position_from_tick(units.to_tick(float(v))) for v in source),
                        dtype=np.float64,
                        count=len(source),
                    )
                    joints[..., j] = decoded.reshape(values.shape[:-1])
            poses = model.forward_batch(joints)
            result[..., 7 * a : 7 * a + 3] = poses[..., :3, 3]
            result[..., 7 * a + 3 : 7 * a + 6] = (
                Rotation.from_matrix(poses[..., :3, :3].reshape(-1, 3, 3))
                .as_rotvec()
                .reshape((*values.shape[:-1], 3))
            )
            result[..., 7 * a + 6] = self.gripper(values[..., gi], arm["gripper"])
        if self.passthrough:
            result[..., 7 * len(self.arms) :] = values[..., self.passthrough]
        return result

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
            for a, (arm, indices, gi, model) in enumerate(self.arms):
                seed = np.array(
                    [
                        self.joint(reference[index][i], c)
                        for i, c in zip(indices, arm["coordinates"], strict=True)
                    ]
                )
                for step, pose in enumerate(poses[index]):
                    target = np.eye(4)
                    target[:3, 3] = pose[7 * a : 7 * a + 3]
                    target[:3, :3] = Rotation.from_rotvec(pose[7 * a + 3 : 7 * a + 6]).as_matrix()
                    seed = model.inverse(target, seed, **self.descriptor.get("ik", {}))
                    for i, q, c in zip(indices, seed, arm["coordinates"], strict=True):
                        output[index + (step, i)] = self.source_joint(q, c)
                    output[index + (step, gi)] = self.gripper(
                        pose[7 * a + 6], arm["gripper"], inverse=True
                    )
            for n, i in enumerate(self.passthrough):
                output[index + (slice(None), i)] = poses[
                    index + (slice(None), 7 * len(self.arms) + n)
                ]
        return output


def descriptor_from_host(info, mappings):
    """Freeze explicit installed encoder mappings and packaged geometry for a recording."""
    metadata = info.get("robot_metadata")
    if not metadata:
        raise ValueError(
            "Host kinematics requires recorded motor units and installed joint mappings"
        )
    robot_model = metadata["robot_model"]
    if robot_model != "alohamini2pro":
        raise ValueError("No packaged arm geometry for this robot model")
    names = info["features"]["action"]["names"]
    state = info["features"].get("observation.state")
    if state is not None and state["names"] != names:
        raise ValueError("Host state/action names must agree")
    arms, consumed = [], []
    for side in ("left", "right"):
        mapping = mappings[side]
        geometry = AlohaMiniArmKinematics.from_description(side)
        coordinates = []
        for index, suffix in enumerate((*ARM_SUFFIXES, "gripper")):
            joint = "wrist_yaw" if suffix == "wrist_yaw_joint" else suffix
            field = f"arm_{side}_{joint}.pos"
            entry = mapping["joints"][joint]
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
                        unit="host",
                        host_units=asdict(units),
                        encoder=asdict(calibration),
                    )
                )
        arms.append(
            dict(
                name=side,
                frame=f"{side}_Base",
                coordinates=coordinates,
                gripper=gripper,
                geometry=geometry.parameters(),
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
    )
    KinematicFields(descriptor)
    return descriptor


def descriptor_from_dataset(root, mappings):
    root = Path(root).expanduser()
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") == "v3.0":
        sidecar = root / "meta/alohamini.json"
        if not sidecar.is_file():
            raise ValueError(
                "Original v3 requires an explicit coordinate descriptor; "
                "Host metadata is unavailable"
            )
        source = json.loads(sidecar.read_text())["source_info"]
        if info["features"]["action"] != source["features"]["action"]:
            raise ValueError("Exported action fields differ from their Host annotation")
        info["robot_metadata"] = source["robot_metadata"]
    return descriptor_from_host(info, mappings)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze named joint-to-TCP derived-field geometry")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--left-map", required=True, type=Path)
    parser.add_argument("--right-map", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
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
