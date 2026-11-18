"""Local policy checkpoints and the adapter to the shared protected evaluator."""

from collections import deque
from contextlib import contextmanager, nullcontext
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
from alohamini.learning.auxiliary import auxiliary_features, snapshot_auxiliary
from alohamini.learning.checkpoint import read_checkpoint, write_checkpoint_metadata
from alohamini.learning.data_fields import FieldProcessor, active
from alohamini.learning.loading import collate_samples
from alohamini.learning.precision import (
    apply_trainable_dtype,
    tf32_context,
    validate_trainable_dtype,
)
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
        "image_resize_mode": getattr(samples, "image_resize_mode", "stretch"),
        "image_pad_value": getattr(samples, "image_pad_value", 0.0),
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
    if getattr(samples, "annotation_context", False) or getattr(samples, "annotation_fields", ()):
        manifest["annotations"] = {
            "fields": list(samples.annotation_fields),
            "time_basis": "recorded_timestamp",
            "identity": ["episode_index", "frame_index"],
        }
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
        goal_images=None,
        time_scale=None,
        language_replan_interval_s=None,
    ):
        if calibration_mode not in ("strict", "normalized", "legacy_normalized"):
            raise ValueError("calibration_mode must be strict, normalized or legacy_normalized")
        self.calibration_mode = calibration_mode
        self.offline = offline
        self._robot_bound = False
        path, manifest = read_checkpoint(checkpoint)
        training_only = manifest["source_info"].get("training_only")
        legacy_binding = calibration_mode == "legacy_normalized"
        if training_only and not legacy_binding and not offline:
            raise ValueError(
                "This checkpoint is for offline training only: its source data has no verified "
                "Host calibration/action contract. --calibration-mode legacy_normalized only "
                "supports original LeRobot v3 with confirmed normalized action units"
            )
        if legacy_binding and (
            not isinstance(training_only, dict)
            or training_only.get("source") != "original_lerobot_v3"
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
        self._language_controller = None
        if language_replan_interval_s is not None:
            from alohamini.learning.language_control import LanguageController

            if "subtask" not in getattr(self.algorithm, "language_query_kinds", ()):
                raise ValueError("This policy does not declare subtask generation support")
            self._language_controller = LanguageController(language_replan_interval_s)
        self.field_transform = (
            FieldTransforms(manifest["data_fields"]) if manifest.get("data_fields") else None
        )
        if self.field_transform:
            self.field_transform.validate_source(self.source, manifest["state_feature"])
        self.task = task
        options = self.algorithm.checkpoint_options(
            path, manifest["config"], device, n_action_steps, temporal_ensemble_coeff
        )
        if time_scale is not None:
            if "time_scale" not in options:
                raise ValueError("This policy does not declare an execution time scale")
            options["time_scale"] = time_scale
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
        expected.update(
            auxiliary_features(self.source, manifest.get("sample_windows", {}), expected)
        )
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
        if self._language_controller is not None:
            self.fixed_deployment["language_control"] = {
                "replan_interval_s": self._language_controller.interval_s,
            }
        self.reset()

        if goal_images is not None:
            self.set_goal(goal_images)

    def set_goal(self, images):
        """Set an explicit visual goal through the algorithm's input processor."""
        prepare = getattr(self.processor, "prepare_goal", None)
        setter = getattr(self.model, "set_goal", None)
        if prepare is None or setter is None:
            raise ValueError("This policy does not support visual goals")
        prepared = prepare(
            images,
            resize_mode=self.manifest.get("image_resize_mode", "stretch"),
            pad_value=self.manifest.get("image_pad_value", 0.0),
        )
        setter({key: prepared[key] for key in self.config.image_features})
        self.reset()

    def bind_robot(self, snapshot):
        """Bind once before execution; later changes always require strict matching."""
        if getattr(self, "offline", False):
            raise ValueError("Offline policy cannot bind to a robot")
        if self._robot_bound:
            self._check_calibration(snapshot)
            return None
        from alohamini.learning.calibration import (
            legacy_normalized_binding,
            normalized_transfer,
            reference_binding,
        )

        shared_reference = self.calibration_mode == "strict" and (
            self.source.get("robot_metadata", {}).get("reference_calibration") is not None
        )
        if self.calibration_mode == "strict" and not shared_reference:
            self._check_calibration(snapshot)
            self._robot_bound = True
            return None
        if shared_reference:
            metadata, report = reference_binding(self.source["robot_metadata"], snapshot)
        elif self.calibration_mode == "legacy_normalized":
            if self.selection and self.selection.source_names != state_names(snapshot.robot_model):
                raise ValueError("Legacy state names/order differ from the declared Host model")
            metadata, report = legacy_normalized_binding(
                self.source, self.robot_metadata["robot_model"], snapshot
            )
        else:
            metadata, report = normalized_transfer(self.source["robot_metadata"], snapshot)
        validation_metadata = self.source["robot_metadata"] if shared_reference else metadata
        if getattr(self, "field_transform", None) is not None:
            self.field_transform.validate_source(
                {**self.source, "robot_metadata": validation_metadata},
                self.manifest["state_feature"],
            )
        validate_binding = getattr(getattr(self, "algorithm", None), "validate_robot_binding", None)
        if validate_binding is not None:
            validate_binding(self.config, self.source, validation_metadata)
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
        if (
            getattr(self, "calibration_mode", "strict") == "strict"
            and not getattr(self, "_robot_bound", False)
            and getattr(self, "source", {}).get("robot_metadata", {}).get("reference_calibration")
            is not None
        ):
            self.bind_robot(snapshot)
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
            if getattr(self, "_robot_bound", False) and self.robot_metadata.get(
                "reference_calibration"
            ) != snapshot.payload["_robot_metadata"].get("reference_calibration"):
                raise ValueError("Host reference calibration changed after binding")
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
        self._observation_history = None
        self._execution_actions = deque()
        self._synchronous_plan = None
        if getattr(self, "_language_controller", None) is not None:
            self._language_controller.reset()

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

    def predict_prepared_actions(self, batch):
        """Batched physical commands for evaluation at dataset action offsets."""
        decode = getattr(self.algorithm, "predict_action_samples", None)
        if decode is None:
            return self.execution_action(
                self.model.predict_action_chunk(batch), context=batch.get("_action_context")
            )
        action = scale_action(
            decode(self.model, self.processor, batch, fps=self.fps),
            getattr(self.config, "inference_action_scale_dims", ()),
            getattr(self.config, "inference_action_scale", 1.0),
        )
        for name, value in (self.fixed_dimensions or {}).get("targets", {}).items():
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

    @contextmanager
    def inference_context(self):
        training = self.manifest.get("training", {})
        precision = training.get("mixed_precision")
        amp = (
            torch.autocast(
                next(self.model.parameters()).device.type, dtype=getattr(torch, precision)
            )
            if precision in ("bfloat16", "float16") else nullcontext()
        )
        with tf32_context(training["tf32"]) if "tf32" in training else nullcontext(), amp:
            yield

    @torch.no_grad()
    def predict(self, sample):
        """Offline physical action chunk; retain version counters for FSDP2 collectives."""
        batch = self.processor(
            collate_samples([{
                k: v
                for k, v in self.fixed_input(sample).items()
                if k not in ("action", "action_is_pad")
            }])
        )
        if getattr(self.algorithm, "predict_plan", None) is not None:
            plan = self._plan_from_batch(batch)
            start = max(0, plan.start_s)
            times = np.arange(start, plan.end_s-1e-9, 1/self.fps)
            return torch.tensor([plan.action_at(float(t)) for t in times])
        with self.inference_context():
            return (
                self.execution_action(
                    self.model.predict_action_chunk(batch), context=batch.get("_action_context")
                )[0]
                .float()
                .cpu()
            )

    @torch.no_grad()
    def _query_language_batch(self, batch, *, kind, question=None, camera=None):
        query = getattr(self.algorithm, "query_language", None)
        if query is None or kind not in getattr(self.algorithm, "language_query_kinds", ()):
            raise ValueError(f"This policy does not declare language query kind {kind!r}")
        with self.inference_context():
            results = query(
                self.model, self.processor, batch, kind=kind, question=question, camera=camera,
            )
        if (
            not isinstance(results, list) or len(results) != 1
            or not isinstance(results[0], dict)
            or not isinstance(results[0].get("text"), str) or not results[0]["text"].strip()
        ):
            raise ValueError("Language query must return one nonempty text response")
        return results[0]

    def query_language(self, observation, *, kind="subtask", question=None, camera=None):
        """Query language or grounding without advancing action queues or planner state."""
        if isinstance(observation, dict):
            batch = collate_samples([{
                key: value for key, value in self.fixed_input(observation).items()
                if key not in ("action", "action_is_pad")
            }])
            if self.task is not None:
                batch["task"] = [self.task]
        else:
            batch = self._observation_batch(
                observation, history=False, preprocess=False, language_control=False,
            )
        return self._query_language_batch(batch, kind=kind, question=question, camera=camera)

    def _observation_batch(
        self, snapshot, *, history=True, preprocess=True, language_control=True,
    ):
        """Prepare one live Host snapshot with the checkpoint's original input contract."""
        self._check_calibration(snapshot)
        observation = {}
        if getattr(self, "field_transform", None) is not None:
            observation["_action_context"] = torch.tensor(
                [snapshot.payload[n] for n in self.field_transform.context_names],
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
                resize_mode=self.manifest.get("image_resize_mode", "stretch"),
                pad_value=self.manifest.get("image_pad_value", 0.0),
            )
        observation.update(
            snapshot_auxiliary(snapshot.payload, self.config.input_features, self.source)
        )
        batch = {k: v[None] for k, v in observation.items()}
        history_spec = getattr(self.algorithm, "live_history_spec", None) if history else None
        if history_spec is not None:
            from alohamini.learning.observation_history import ObservationHistory

            if self._observation_history is None:
                self._observation_history = ObservationHistory(
                    history_spec(asdict(self.config), cameras=self.manifest["cameras"]), self.fps
                )
            batch = self._observation_history.append(batch, snapshot.received_s)
        if self.algorithm.include_task:
            batch["task"] = [self.task]
        batch = self.fixed_input(batch)
        controller = getattr(self, "_language_controller", None)
        if language_control and controller is not None:
            subtask, changed = controller.condition(
                self.task, snapshot.request_started_s,
                lambda _: self._query_language_batch(batch, kind="subtask")["text"],
            )
            if changed:
                self.model.reset()
                self._execution_actions.clear()
                self._synchronous_plan = None
            batch["task"] = [subtask]
        return self.processor(batch) if preprocess else batch

    @property
    def tcp_reference_actions(self):
        return (
            isinstance(self.processor, FieldProcessor)
            and self.processor.field_transform.action in ("tcp_absolute", "tcp_delta")
        )

    @property
    def plan_capabilities(self):
        from alohamini.trajectory_execution import PlanCapabilities

        if self.tcp_reference_actions:
            return PlanCapabilities(kinds=("tcp",), default_replan_interval_s=0.25)
        validate = getattr(self.algorithm, "validate_plan", None)
        if validate is not None:
            validate(self.config)
        return getattr(self.algorithm, "plan_capabilities", PlanCapabilities())

    def predict_plan(self, snapshot, *, prefix=None, continuity="none"):
        """Return a named physical-time plan; model adapters own prefix guidance."""
        from alohamini.trajectory_execution import ExecutionConfig

        self.plan_capabilities.validate(ExecutionConfig(continuity=continuity))
        guided = continuity in self.plan_capabilities.gradient_methods
        with torch.inference_mode(not guided), torch.no_grad():
            controller = getattr(self, "_language_controller", None)
            generation = controller.generation if controller is not None else None
            batch = self._observation_batch(snapshot)
            if controller is not None and generation != controller.generation:
                prefix = None
            return self._plan_from_batch(
                batch, prefix=prefix, continuity=continuity, tcp_reference=True
            )

    def _plan_from_batch(
        self, batch, *, prefix=None, continuity="none", tcp_reference=False
    ):
        from alohamini.trajectory import DenseActionPlan

        special = getattr(self.algorithm, "predict_plan", None)
        with self.inference_context():
            if tcp_reference and self.tcp_reference_actions:
                from alohamini.tcp_trajectory import TCPActionPlan

                self._require_fixed_current()
                if continuity != "none" or prefix is not None:
                    raise ValueError(
                        "TCP tracking supports sync/async without RTC prefixes"
                    )
                if (getattr(self.config, "inference_action_scale_dims", ())
                        and getattr(self.config, "inference_action_scale", 1.0) != 1.0):
                    raise ValueError(
                        "TCP reference tracking cannot apply Host joint action scaling"
                    )
                actions = self.model.predict_action_chunk(batch)
                if (not isinstance(actions, torch.Tensor) or actions.ndim != 3
                        or actions.shape[0] != 1 or actions.shape[1] < 1
                        or not torch.isfinite(actions).all()):
                    raise ValueError("Policy returned invalid TCP action references")
                references = self.processor.tcp_action(
                    actions, context=batch.get("_action_context")
                )
                offset = getattr(self.algorithm, "action_offset", lambda config: 0)(self.config)
                return TCPActionPlan(
                    tuple(self.names), references[0, offset:].tolist(), 1 / self.fps,
                    self.processor.field_transform.geometry.descriptor,
                    fixed_targets=(self.fixed_dimensions or {}).get("targets", {}),
                )
            if special is not None:
                return special(self, batch, prefix=prefix, continuity=continuity)
            prepare = getattr(self.algorithm, "prediction_arguments", None)
            kwargs = prepare(self, batch, prefix, continuity) if prepare else {}
            if continuity != "none" and prepare is None:
                raise ValueError("Policy has no adapter for the requested continuity method")
            actions = self._physical_actions(batch, **kwargs)
        offset = getattr(self.algorithm, "action_offset", lambda config: 0)(self.config)
        return DenseActionPlan(tuple(self.names), actions[offset:].tolist(), 1/self.fps)

    def _physical_actions(self, batch, *, max_steps=None, **kwargs):
        action = self.model.predict_action_chunk(batch, **kwargs)
        if (not isinstance(action, torch.Tensor) or action.ndim != 3 or action.shape[0] != 1
                or action.shape[1] < 1 or action.shape[2] < 1):
            raise ValueError("Policy returned invalid action chunk dimensions")
        if max_steps is not None:
            if type(max_steps) is not int or not 1 <= max_steps <= action.shape[1]:
                raise ValueError("Execution window must fit the predicted action chunk")
            action = action[:, :max_steps]
        raw_finite = torch.isfinite(action).all()
        values = self.execution_action(action, context=batch.get("_action_context"))
        if values.shape != (1, action.shape[1], len(self.names)):
            raise ValueError("Policy returned invalid absolute action chunk")
        copied = torch.cat((values.reshape(-1).float(), raw_finite.reshape(1).float())).cpu()
        if copied[-1] != 1 or not torch.isfinite(copied[:-1]).all():
            raise ValueError("Policy returned invalid absolute action chunk values")
        return copied[:-1].reshape(action.shape[1], len(self.names))

    def predict_action_chunk(self, snapshot, *, max_steps=None):
        """Sample a physical plan at the checkpoint cadence for offline consumers."""
        if getattr(self.algorithm, "predict_plan", None) is None:
            with torch.inference_mode(), self.inference_context():
                return self._physical_actions(
                    self._observation_batch(snapshot), max_steps=max_steps
                )
        plan = self.predict_plan(snapshot)
        count = max(1, int(np.ceil((plan.end_s-max(0, plan.start_s))*self.fps)))
        if max_steps is not None:
            count = min(count, max_steps)
        start = max(0, plan.start_s)
        return torch.tensor([plan.action_at(start+i/self.fps) for i in range(count)])

    def select_action(self, snapshot):
        if getattr(self.algorithm, "predict_plan", None) is not None:
            import time

            now = time.monotonic()
            current = getattr(self, "_synchronous_plan", None)
            if current is None or now-current[0] >= current[1].end_s:
                origin = getattr(snapshot, "request_started_s", now)
                if self.tcp_reference_actions:
                    with torch.inference_mode():
                        plan = self._plan_from_batch(self._observation_batch(snapshot))
                else:
                    plan = self.predict_plan(snapshot)
                current = (origin, plan)
                self._synchronous_plan = current
            elapsed = time.monotonic()-current[0]
            if elapsed < current[1].start_s:
                return None
            values = current[1].action_at(elapsed)
            return dict(zip(self.names, values, strict=True))
        batch = self._observation_batch(
            snapshot, history=not getattr(self.algorithm, "queues_online_observations", False)
        )
        with torch.inference_mode(), self.inference_context():
            if getattr(self.processor, "decode_before_queue", False) is True:
                if not self._execution_actions:
                    actions = self._physical_actions(
                        batch, max_steps=self.config.n_action_steps,
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
    language_replan_interval_s=None,
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
        language_replan_interval_s=language_replan_interval_s,
        task=kwargs.get("task"),
    )
    return evaluate(
        host,
        policy.robot_metadata["robot_model"],
        policy_factory=lambda: policy,
        fps=policy.fps,
        **kwargs,
    )
