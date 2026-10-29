"""Local policy checkpoints and the adapter to the shared protected evaluator."""

from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from alohamini.apps.replay import check_calibration
from alohamini.datasets.field_transforms import FieldTransforms
from alohamini.datasets.images import decode_host_image
from alohamini.datasets.record import StateSelection, motor_feedback_frame, state_names
from alohamini.learning.checkpoint import read_checkpoint, write_checkpoint_metadata
from alohamini.learning.data_fields import FieldProcessor, active
from alohamini.learning.precision import apply_trainable_dtype, validate_trainable_dtype
from alohamini.learning.processor import image_tensor, scale_action
from alohamini.policies.registry import algorithm


def make_policy(kind, options, stats=None, *, trainable_dtype="preserve"):
    """Instantiate the configured policy model."""
    validate_trainable_dtype(trainable_dtype)
    return apply_trainable_dtype(algorithm(kind).build(options, stats), trainable_dtype)


def make_processor(model, stats, device, *, data_fields=None):
    processor = algorithm(model.name).processor(model, stats, device)
    if data_fields and active(data_fields):
        return FieldProcessor(processor, FieldTransforms(data_fields), device)
    return processor


def save_checkpoint(path, model, stats, samples, *, training, model_state=None):
    """Commit a new checkpoint directory; never overwrite an existing checkpoint."""
    trainable_dtype = getattr(model, "_trainable_dtype", "preserve")
    if training.get("trainable_dtype", trainable_dtype) != trainable_dtype:
        raise ValueError("Checkpoint trainable_dtype differs from the model's storage mode")
    training = {**training, "trainable_dtype": trainable_dtype}
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".pending")
    if path.exists():
        raise FileExistsError(path)
    staging.mkdir()
    manifest = {
        "format": "alohamini-policy",
        "version": 1,
        "kind": model.name,
        "config": asdict(model.config),
        "stats": stats,
        "source_info": samples.info,
        "state": samples.state,
        "state_feature": samples.selection.feature if samples.selection else None,
        "state_units": samples.selection.units if samples.selection else [],
        "cameras": samples.cameras,
        "image_size": samples.image_size,
        "training": training,
        "review_note": samples.review_note,
        "dataset_check": samples.report,
        "table_sha256": samples.table_sha256,
        "sample_windows": samples.delta_indices,
        "sample_boundary": "episode",
        "sample_filter": samples.sample_filter,
    }
    from alohamini.learning.fixed import action_summary

    manifest["fixed_state_excluded"] = getattr(samples, "fixed_state_excluded", True)
    summary = action_summary(getattr(samples, "source", samples))
    if summary is not None:
        manifest["physical_action_summary"] = summary
    algorithm(model.name).save(staging, model, manifest, model_state)
    transform = getattr(samples, "field_transform", None)
    if transform is not None:
        algorithm_operations = manifest.get("field_transforms", [])
        manifest.update(
            transform.manifest(
                samples.selection.feature if samples.selection else None,
                samples.selection.units if samples.selection else [],
                samples.info["features"]["action"],
            )
        )
        manifest["field_transforms"] += [
            op
            for op in algorithm_operations
            if op["operation"] not in ("forward_kinematics", "relative_actions")
        ]
        manifest["recorded_state_stats"] = samples.recorded_state_statistics()
    write_checkpoint_metadata(staging, manifest)
    staging.rename(path)


class NativePolicy:
    """reset()/select_action(snapshot) interface consumed by apps.evaluation.

    The tensor-level network remains available as ``model`` for custom training.
    This adapter assembles the same ordered fields and RGB preprocessing as training.
    """

    def __init__(
        self,
        checkpoint,
        *,
        device="cpu",
        n_action_steps=None,
        temporal_ensemble_coeff="checkpoint",
        task=None,
        fixed_dimensions=None,
        fixed_dataset=None,
        fixed_current=None,
        calibration_mode="strict",
        robot_model=None,
        offline=False,
    ):
        if calibration_mode not in ("strict", "normalized", "legacy_normalized"):
            raise ValueError("calibration_mode must be strict, normalized or legacy_normalized")
        self.calibration_mode = calibration_mode
        self.offline = offline
        self._robot_bound = False
        path, manifest = read_checkpoint(checkpoint)
        original_v3 = manifest["source_info"].get("training_only")
        legacy_binding = calibration_mode == "legacy_normalized"
        if original_v3 and not legacy_binding and not offline:
            raise ValueError(
                "This checkpoint is for offline training only: its original LeRobot "
                "data has no verified Host calibration/action contract. For confirmed "
                "normalized action units, explicitly select --calibration-mode legacy_normalized"
            )
        if legacy_binding and (
            not isinstance(original_v3, dict)
            or original_v3.get("source") != "original_lerobot_v3"
            or robot_model is None
        ):
            raise ValueError(
                "legacy_normalized requires an original LeRobot v3 checkpoint "
                "and an explicit robot_model"
            )
        self.checkpoint_path = path
        self.checkpoint_manifest_sha256 = sha256((path / "policy.json").read_bytes()).hexdigest()
        self.manifest = manifest
        self.source = manifest["source_info"]
        # An incomplete declaration only identifies the intended model. It cannot
        # pass calibration checks until bind_robot validates a live Host snapshot.
        self.robot_metadata = (
            {"robot_model": robot_model, "binding_pending": True}
            if legacy_binding
            else deepcopy(self.source.get("robot_metadata"))
        )
        if robot_model is not None and (
            self.robot_metadata is None or robot_model != self.robot_metadata["robot_model"]
        ):
            raise ValueError("Declared robot model differs from checkpoint")
        self.fps = self.source["fps"]
        self.names = self.source["features"]["action"]["names"]
        if not offline and self.names != state_names(self.robot_metadata["robot_model"]):
            raise ValueError("Checkpoint action coordinates must retain all Host targets")
        from alohamini.learning.state import select_state

        self.selection = select_state(
            self.source,
            manifest["state"],
            exclude_fixed=manifest.get("fixed_state_excluded", False),
        )
        if self.selection and (
            self.selection.feature != manifest["state_feature"]
            or self.selection.units != manifest["state_units"]
        ):
            raise ValueError("Checkpoint state names/units mismatch")
        self.algorithm = algorithm(manifest["kind"])
        self.field_transform = (
            FieldTransforms(manifest["data_fields"]) if manifest.get("data_fields") else None
        )
        if self.field_transform:
            self.field_transform.validate_source(self.source, manifest["state_feature"])
        self.task = task
        options = self.algorithm.checkpoint_options(
            path, manifest["config"], device, n_action_steps, temporal_ensemble_coeff
        )
        self.model = make_policy(
            manifest["kind"],
            options,
            trainable_dtype=manifest.get("training", {}).get("trainable_dtype", "preserve"),
        )
        self.config = self.model.config
        expected = {
            f"observation.images.{c}": ("VISUAL", (3, *manifest["image_size"]))
            for c in manifest["cameras"]
        }
        if self.selection:
            expected["observation.state"] = ("STATE", tuple(self.selection.feature["shape"]))
        action_names = self.names
        action_width = len(self.names)
        if self.field_transform:
            features = self.field_transform.features(
                manifest["state_feature"],
                manifest["state_units"],
                self.source["features"]["action"],
            )
            if "observation.state" in features:
                expected["observation.state"] = (
                    "STATE",
                    tuple(features["observation.state"]["shape"]),
                )
            action_names = features["action"]["names"]
            action_width = len(action_names)
        actual = {k: (v.type, v.shape) for k, v in self.config.input_features.items()}
        if actual != expected or self.config.action_feature.shape != (action_width,):
            raise ValueError("Checkpoint feature dimensions/order do not match its data contract")
        self.algorithm.validate_statistics(
            manifest["stats"], expected, action_names, manifest=manifest
        )
        self.algorithm.load(path, self.model)
        self.model.to(device).eval()
        self.processor = make_processor(
            self.model, manifest["stats"], device, data_fields=manifest.get("data_fields")
        )
        from alohamini.learning.fixed import configure

        if offline and self.robot_metadata is None:
            if fixed_dimensions or fixed_dataset or fixed_current:
                raise ValueError("Unannotated offline inference does not bind Host fixed targets")
            self.fixed_dimensions = None
            self.fixed_current = []
            self.fixed_state_references = {}
            self.fixed_deployment = {}
        else:
            configure(self, fixed_dimensions, fixed_dataset, current=fixed_current)
        self.fixed_deployment["calibration_mode"] = calibration_mode
        self.reset()

    def bind_robot(self, snapshot):
        """Bind once before execution; later changes always require strict matching."""
        if getattr(self, "offline", False):
            raise ValueError("Offline policy cannot bind to a robot")
        if self.calibration_mode == "strict" or self._robot_bound:
            self._check_calibration(snapshot)
            return None
        from alohamini.learning.calibration import legacy_normalized_binding, normalized_transfer

        if self.calibration_mode == "legacy_normalized":
            if self.selection and self.selection.source_names != state_names(snapshot.robot_model):
                raise ValueError("Legacy state names/order differ from the declared Host model")
            metadata, report = legacy_normalized_binding(
                self.source, self.robot_metadata["robot_model"], snapshot
            )
        else:
            metadata, report = normalized_transfer(self.source["robot_metadata"], snapshot)
        if getattr(self, "field_transform", None) is not None:
            self.field_transform.validate_source(
                {**self.source, "robot_metadata": metadata}, self.manifest["state_feature"]
            )
        validate_binding = getattr(getattr(self, "algorithm", None), "validate_robot_binding", None)
        if validate_binding is not None:
            validate_binding(self.config, self.source, metadata)
        if self.selection and self.calibration_mode != "legacy_normalized":
            live_source = {**self.source, "robot_metadata": metadata}
            self.selection = StateSelection(
                live_source,
                self.manifest["state"],
                exclude_fixed=self.manifest.get("fixed_state_excluded", False),
            )
        self.robot_metadata = metadata
        self._robot_bound = True
        report.update(
            checkpoint=str(self.checkpoint_path),
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            fixed_dimensions=self.fixed_dimensions,
        )
        return report

    def _check_calibration(self, snapshot):
        if getattr(self, "offline", False):
            raise ValueError("Offline policy cannot execute on a robot")
        if (
            getattr(self, "calibration_mode", "strict") == "legacy_normalized"
            and not self._robot_bound
        ):
            raise ValueError(
                "Legacy policy must bind to validated Host calibration before execution"
            )
        # Compare owned copies: in-place changes must invalidate the static check.
        context = (
            snapshot.robot_model,
            snapshot.payload.get("_safety", {}).get("host_session_id"),
            snapshot.payload.get("lift_axis.reference_sequence"),
            snapshot.payload["_robot_metadata"],
            self.robot_metadata,
        )
        if context != getattr(self, "_calibration_context", None):
            check_calibration(self.robot_metadata, snapshot)
            self._calibration_context = deepcopy(context)

    def bind_fixed_current(self, snapshot, client_id):
        from alohamini.learning.fixed import bind_current

        self._check_calibration(snapshot)
        bind_current(self, snapshot, client_id)

    def _require_fixed_current(self):
        if getattr(self, "fixed_current", None) and not self._fixed_current_bound:
            raise RuntimeError("--fixed-current needs live Host preparation before inference")

    def reset(self):
        self.model.reset()
        self._execution_actions = deque()

    def refresh_latent(self):
        """Notify AM-ACT of a task phase boundary before its next observation."""
        if not hasattr(self.model, "refresh_latent"):
            raise ValueError("This policy does not expose latent phase control")
        self.model.refresh_latent()

    def execution_action(self, tensor, *, context=None):
        """Restore physical outputs and apply the checkpoint's deployment scaling."""
        self._require_fixed_current()
        action = scale_action(
            self.processor.action(tensor, context=context),
            getattr(self.config, "inference_action_scale_dims", ()),
            getattr(self.config, "inference_action_scale", 1.0),
        )
        if self.fixed_dimensions:
            action = action.clone()
            for name, value in self.fixed_dimensions["targets"].items():
                action[..., self.names.index(name)] = value
        return action

    def fixed_input(self, sample):
        self._require_fixed_current()
        if self.fixed_state_references and "observation.state" in sample:
            sample = dict(sample)
            if "_fields_encoded" in sample:
                sample["observation.state"] = sample["_recorded_state"]
                if "action" in sample:
                    sample["action"] = sample["_recorded_action"]
                sample.pop("_fields_encoded")
            state = sample["observation.state"].clone()
            for index, value in self.fixed_state_references.items():
                state[..., index] = value
            sample["observation.state"] = state
        return sample

    def inference_context(self):
        precision = self.manifest.get("training", {}).get("mixed_precision")
        if precision in ("bfloat16", "float16"):
            return torch.autocast(
                next(self.model.parameters()).device.type, dtype=getattr(torch, precision)
            )
        return nullcontext()

    @torch.inference_mode()
    def predict(self, sample):
        """Offline physical action chunk; sample has no batch dimension."""
        batch = self.processor(
            {
                k: ([v] if k == "task" else v[None])
                for k, v in self.fixed_input(sample).items()
                if k not in ("action", "action_is_pad")
            }
        )
        with self.inference_context():
            return (
                self.execution_action(
                    self.model.predict_action_chunk(batch), context=batch.get("_action_context")
                )[0]
                .float()
                .cpu()
            )

    def _observation_batch(self, snapshot):
        """Prepare one live Host snapshot with the checkpoint's original input contract."""
        self._check_calibration(snapshot)
        observation = {}
        if getattr(self, "field_transform", None) is not None:
            observation["_action_context"] = torch.tensor(
                [snapshot.payload[n] for n in self.field_transform.geometry.names],
                dtype=torch.float32,
            )
        if self.selection:
            row = {
                "observation.state": [snapshot.payload[n] for n in self.selection.source_names],
                **motor_feedback_frame(
                    self.source["features"], snapshot.payload.get("_motor_feedback", {})
                ),
            }
            observation["observation.state"] = torch.from_numpy(self.selection.frame(row))
            for _, index, _, mask in self.selection.columns:
                if mask:
                    now = snapshot.payload.get("_host_timing", {}).get(
                        "state_sample_finished_monotonic_s"
                    )
                    if (
                        now is None
                        or not 0 <= now - row["motor_feedback.sample_finished_s"][index] <= 0.25
                    ):
                        raise ValueError("Selected motor feedback is stale")
        for camera in self.manifest["cameras"]:
            if camera not in snapshot.images:
                raise ValueError(f"Missing policy camera: {camera}")
            observation[f"observation.images.{camera}"] = image_tensor(
                decode_host_image(snapshot.images[camera]),
                self.manifest["image_size"],
            )
        batch = {k: v[None] for k, v in observation.items()}
        if self.algorithm.include_task:
            batch["task"] = [self.task]
        return self.processor(self.fixed_input(batch))

    def validate_rtc(self, config):
        if self.manifest["kind"] not in ("pi05", "smolvla"):
            raise ValueError("RTC currently supports pi05 and smolvla checkpoints")
        horizon = (
            self.config.network["action_horizon"]
            if self.manifest["kind"] == "pi05" else self.config.chunk_size
        )
        config.validate_horizon(horizon)
        if getattr(self.config, "inference_action_scale", 1.0) <= 0:
            raise ValueError("RTC requires positive inference_action_scale")
        return horizon

    def predict_action_chunk(
        self, snapshot, *, rtc=None, previous_actions=None, inference_delay=0, max_steps=None
    ):
        """Return absolute CPU targets; optionally trim before trajectory decoding."""
        guided = rtc is not None and rtc.enabled
        if guided:
            self.validate_rtc(rtc)
        # Prefix caches must be ordinary tensors for the later denoiser VJP.
        with torch.inference_mode(not guided), torch.no_grad():
            return self._predict_action_chunk(
                snapshot,
                rtc=rtc if guided else None,
                previous_actions=previous_actions,
                inference_delay=inference_delay,
                max_steps=max_steps,
            )

    def _predict_action_chunk(
        self, snapshot, *, rtc, previous_actions, inference_delay, batch=None, max_steps=None
    ):
        if batch is None:
            batch = self._observation_batch(snapshot)
        kwargs = {}
        if rtc is not None:
            from alohamini.policies.rtc import RTCGuidance

            prefix = None
            if previous_actions is not None:
                previous = torch.as_tensor(
                    previous_actions, dtype=torch.float32,
                    device=next(self.model.parameters()).device,
                )
                if (
                    previous.ndim != 2 or previous.shape[1] != len(self.names)
                    or not torch.isfinite(previous).all()
                ):
                    raise ValueError("RTC requires finite absolute targets")
                previous = scale_action(
                    previous,
                    getattr(self.config, "inference_action_scale_dims", ()),
                    1 / getattr(self.config, "inference_action_scale", 1.0),
                )
                prefix = self.processor.action_prefix(
                    previous[None], context=batch.get("_action_context")
                )
            kwargs = dict(
                rtc_processor=RTCGuidance(rtc), prev_chunk_left_over=prefix,
                inference_delay=inference_delay,
            )
        with self.inference_context():
            action = self.model.predict_action_chunk(batch, **kwargs)
            if (
                not isinstance(action, torch.Tensor)
                or action.ndim != 3
                or action.shape[0] != 1
                or action.shape[1] < 1
                or action.shape[2] < 1
            ):
                raise ValueError("Policy returned invalid action chunk dimensions or values")
            if max_steps is not None:
                if type(max_steps) is not int or not 1 <= max_steps <= action.shape[1]:
                    raise ValueError("Execution window must fit the predicted action chunk")
                action = action[:, :max_steps]
            raw_finite = torch.isfinite(action).all()
            values = self.execution_action(action, context=batch.get("_action_context"))
        if not isinstance(values, torch.Tensor) or values.shape != (
            1, action.shape[1], len(self.names)
        ):
            raise ValueError("Policy returned invalid absolute action chunk")
        # Transfer the absolute outputs and raw validity in one operation. PI0.5
        # pads model coordinates beyond the Host targets; validate those too
        # without forcing an extra CUDA synchronization before the CPU copy.
        copied = torch.cat((values.reshape(-1).float(), raw_finite.reshape(1).float())).cpu()
        if copied[-1] != 1 or not torch.isfinite(copied[:-1]).all():
            raise ValueError("Policy returned invalid absolute action chunk values")
        return copied[:-1].reshape(action.shape[1], len(self.names))

    def select_action(self, snapshot):
        batch = self._observation_batch(snapshot)
        with torch.inference_mode(), self.inference_context():
            if getattr(self.processor, "decode_before_queue", False) is True:
                if not self._execution_actions:
                    actions = self._predict_action_chunk(
                        snapshot,
                        rtc=None,
                        previous_actions=None,
                        inference_delay=0,
                        batch=batch,
                        max_steps=self.config.n_action_steps,
                    )
                    self._execution_actions.extend(actions.numpy())
                values = self._execution_actions.popleft()[None]
            else:
                action = self.model.select_action(batch)
                values = (
                    self.execution_action(
                        action,
                        context=getattr(self.model, "action_context", batch.get("_action_context")),
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
        if values.shape != (1, len(self.names)) or not np.isfinite(values).all():
            raise ValueError("Policy returned invalid absolute targets")
        return dict(zip(self.names, map(float, values[0]), strict=True))


def evaluate_robot(
    checkpoint,
    *,
    enable_robot=False,
    host=None,
    device="cuda",
    n_action_steps=None,
    temporal_ensemble_coeff="checkpoint",
    fixed_dimensions=None,
    fixed_dataset=None,
    fixed_current=None,
    calibration_mode="strict",
    robot_model=None,
    **kwargs,
):
    """Explicit real-hardware boundary; importing/loading a model never connects."""
    if enable_robot is not True or not host:
        raise ValueError("Real robot evaluation requires enable_robot=True and an explicit host")
    from alohamini.apps.evaluation import evaluate

    policy = NativePolicy(
        checkpoint,
        device=device,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        fixed_dimensions=fixed_dimensions,
        fixed_dataset=fixed_dataset,
        fixed_current=fixed_current,
        calibration_mode=calibration_mode,
        robot_model=robot_model,
        task=kwargs.get("task"),
    )
    return evaluate(
        host,
        policy.robot_metadata["robot_model"],
        policy_factory=lambda: policy,
        fps=policy.fps,
        **kwargs,
    )
