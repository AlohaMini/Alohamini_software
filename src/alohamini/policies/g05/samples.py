# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot and native Alohamini integration in 2026.

# ruff: noqa: E501

"""G0.5 samples from the platform annotation/message batch contract."""

import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor

from alohamini.datasets.recipe import MESSAGES_RENDERED

from .templates import G05_COT_PROMPTS, make_g05_cot_prompt_template, make_g05_prompt_template
from .tokenizer_g05 import G05_RUNTIME_PREDICT_COT

ACTION, OBS_STATE = "action", "observation.state"


def _task_values(batch: Mapping[str, Any], task: str | None, batch_size: int) -> list[str]:
    """Broadcast the task string across the batch."""
    if task is not None:
        return [task] * batch_size
    value = batch.get("task")
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, list | tuple) and len(value) == batch_size:
        return [str(item) for item in value]
    raise ValueError(
        "G0.5 requires the already-selected LeRobot task string; no task augmentation "
        "or model-local sampling is performed."
    )


def _batch_item(value: Any, index: int, batch_size: int) -> Any:
    """Take one sample's value out of a batched field."""
    if isinstance(value, Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index]
    if isinstance(value, list | tuple) and len(value) == batch_size:
        return value[index]
    return value


def _sample_messages(batch: Mapping[str, Any], index: int, batch_size: int) -> Any:
    """One sample's recipe-rendered conversation, or None when no recipe ran."""
    messages = batch.get(MESSAGES_RENDERED)
    if (
        isinstance(messages, list | tuple)
        and len(messages) == batch_size
        and (not messages or isinstance(messages[0], list | tuple))
    ):
        messages = messages[index]
    return [messages] if isinstance(messages, Mapping) else messages


def _rendered_task(batch: Mapping[str, Any], index: int, batch_size: int) -> str | None:
    """The task text of one sample's rendered user turn.

    The recipe's ``${task}`` rotates through the episode's ``task_aug`` rephrasings, so
    using it as the command gives the task-prompt diversity the annotations provide.
    """
    messages = _sample_messages(batch, index, batch_size)
    if not isinstance(messages, list | tuple):
        return None
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            text = message.get("content")
            if text:
                return text
    return None


def _recipe_cot_targets(
    batch: Mapping[str, Any],
    index: int,
    batch_size: int,
) -> tuple[str | None, str | None]:
    """Read the selected recipe's supervised Subtask/BBox messages."""

    target_indices = batch.get("target_message_indices")
    sample_messages = _sample_messages(batch, index, batch_size)
    if sample_messages is None or target_indices is None:
        return None, None

    sample_target_indices = target_indices
    has_batched_target_indices = (
        isinstance(target_indices, Tensor) and target_indices.ndim > 1
    ) or (
        isinstance(target_indices, list | tuple)
        and len(target_indices) == batch_size
        and (not target_indices or isinstance(target_indices[0], list | tuple | Tensor))
    )
    if has_batched_target_indices:
        sample_target_indices = target_indices[index]
    if isinstance(sample_target_indices, Tensor):
        sample_target_indices = sample_target_indices.detach().cpu().tolist()
    if not isinstance(sample_messages, list | tuple) or not isinstance(
        sample_target_indices, list | tuple
    ):
        return None, None

    subtask: str | None = None
    bbox_json: str | None = None
    for target_index in sample_target_indices:
        message = sample_messages[int(target_index)]
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            continue
        if content.startswith("Subtask:"):
            value = content.removeprefix("Subtask:").strip()
            if value:
                subtask = value
        elif content.startswith("BBoxJSON:"):
            value = content.removeprefix("BBoxJSON:").strip()
            if value:
                bbox_json = value
    return subtask, bbox_json


def format_g05_bbox_target(bbox_json: str | None) -> str | None:
    """Convert LeRobot grounded-VQA JSON into G0.5's location-token format.

    Coordinates are [0, 1] fractions of the image, the LeRobot annotation convention and
    the frame the checkpoint was trained in; each becomes one of 1024 location tokens, in
    (y1, x1, y2, x2) order.
    """

    if not bbox_json:
        return None
    try:
        payload = json.loads(bbox_json)
        if isinstance(payload, str):
            payload = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping):
        return None
    if isinstance(payload.get("answer"), Mapping):
        payload = payload["answer"]

    boxes: list[tuple[str, list[float]]] = []
    detections = payload.get("detections")
    if isinstance(detections, list):
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            coords = detection.get("bbox")
            bbox_format = detection.get("bbox_format", "xyxy")
            if (
                not isinstance(coords, list | tuple)
                or len(coords) != 4
                or bbox_format not in {"xyxy", "xywh"}
            ):
                continue
            x1, y1, a, b = (float(value) for value in coords)
            corners = [x1, y1, x1 + a, y1 + b] if bbox_format == "xywh" else [x1, y1, a, b]
            boxes.append((str(detection.get("label") or "object"), corners))
    else:
        for label, coords in payload.items():
            if isinstance(coords, list | tuple) and len(coords) == 4:
                boxes.append((str(label), [float(value) for value in coords]))
    if not boxes:
        return None
    if any(not 0.0 <= value <= 1.0 for _, coords in boxes for value in coords):
        raise ValueError(
            "G0.5 bbox targets need [0, 1] image-fraction coordinates (the lerobot-annotate "
            f"convention), got {bbox_json!r}. Convert pixel boxes by dividing by the frame size."
        )

    def location_token(value: float) -> str:
        """Render a coordinate as a location token."""
        location = max(0, min(1023, round(value * 1024)))
        return f"<loc{location:04d}>"

    formatted = []
    for label, (x1, y1, x2, y2) in boxes:
        locations = "".join(location_token(value) for value in (y1, x1, y2, x2))
        formatted.append(f"{label} {locations}")
    return "BBox: " + "; ".join(formatted)


def _apply_recipe_cot(
    config: Any,
    flow_only: bool,
    sample: dict[str, Any],
    batch: Mapping[str, Any],
    index: int,
    batch_size: int,
) -> bool:
    """Populate one author sample from recipe-rendered CoT targets."""

    subtask, bbox_json = _recipe_cot_targets(batch, index, batch_size)
    bbox = format_g05_bbox_target(bbox_json)

    fields = tuple(field for field, value in (("bbox", bbox), ("subtask", subtask)) if value)
    if not fields:
        return False
    sample["template"] = make_g05_cot_prompt_template(
        config.num_prompt_images,
        fields=fields,
        flow_only=flow_only,
    )
    if bbox is not None:
        sample["bbox"] = bbox
    if subtask is not None:
        sample["atomic_task"] = f"Subtask: {subtask}"
    sample["prompt"] = G05_COT_PROMPTS[fields]
    return True


def prepare_g05_policy_batch(
    config: Any,
    batch: Mapping[str, Any],
    *,
    task: str | None = None,
    predict_cot: bool | None = None,
) -> dict[str, Any]:
    """Build G0.5's author-format sample batch from a processed LeRobot batch.

    ``config`` is a `G05Config`, or the namespace of its fields that `G05TokenizerStep` holds.
    """
    run_predict_cot = config.predict_cot if predict_cot is None else predict_cot

    state = batch.get(OBS_STATE)
    if not isinstance(state, Tensor):
        raise ValueError(f"G0.5 requires tensor {OBS_STATE!r}.")
    if state.ndim == 1:
        state = state.unsqueeze(0)
    batch_size = state.shape[0]
    tasks = _task_values(batch, task, batch_size)

    pixel_values: dict[str, Tensor] = {}
    for key in config.camera_order:
        image = batch.get(key)
        if not isinstance(image, Tensor):
            raise ValueError(f"G0.5 requires camera {key!r}; camera order is checkpoint state.")
        if image.ndim == 4:
            image = image.unsqueeze(1)
        pixel_values[key] = image
    image_count = sum(image.shape[1] for image in pixel_values.values())
    if image_count != config.num_input_images:
        raise ValueError(
            f"G0.5 received {image_count} camera/history frames, but the checkpoint "
            f"template requires {config.num_input_images}."
        )

    samples = []
    flow_only = "<action_action" not in config.prompt_template
    system1_template = make_g05_prompt_template(
        config.num_prompt_images, predict_cot=False, flow_only=flow_only
    )
    inference_template = config.prompt_template if run_predict_cot else system1_template
    for index, raw_task in enumerate(tasks):
        if task is None:
            raw_task = _rendered_task(batch, index, batch_size) or raw_task
        proprio = state[index]
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        sample = {
            "template": inference_template,
            # This is the author InputPreprocessor command slot. Keep it byte-for-byte
            # unchanged; checkpoint-specific chat formatting occurs downstream.
            "command": raw_task,
            "embodiment": config.embodiment,
            "proprio": {"value": proprio},
        }
        frequency = config.processor_metadata.get("frequency")
        if frequency is not None:
            sample["frequency"] = frequency
        if run_predict_cot:
            rendered_recipe = MESSAGES_RENDERED in batch
            applied_recipe_cot = rendered_recipe and _apply_recipe_cot(
                config, flow_only, sample, batch, index, batch_size
            )
            if not applied_recipe_cot:
                # During mixed-recipe training an applicable no-CoT branch is a
                # genuine target format. At inference, where actions are absent,
                # retain the checkpoint's configured System 2 prompt.
                if rendered_recipe and isinstance(batch.get(ACTION), Tensor):
                    sample["template"] = system1_template
                else:
                    sample["prompt"] = G05_COT_PROMPTS[tuple(config.runtime_cot_fields)]
                    atomic_task = batch.get("atomic_task")
                    if atomic_task is not None:
                        atomic_task = str(_batch_item(atomic_task, index, batch_size))
                        sample["atomic_task"] = (
                            atomic_task
                            if atomic_task.startswith("Subtask:")
                            else f"Subtask: {atomic_task}"
                        )
        for image_index in range(config.num_prompt_images):
            camera = config.camera_order[image_index % len(config.camera_order)]
            sample[f"image{image_index}"] = config.camera_sizes[camera]
        action = batch.get(ACTION)
        if "<action_action" in config.prompt_template:
            if not isinstance(action, Tensor):
                action = state.new_zeros(batch_size, config.chunk_size, config.policy_action_dim)
            action_dim_is_pad = batch.get("action_dim_is_pad")
            if action_dim_is_pad is None:
                action_dim_is_pad = torch.zeros(
                    batch_size,
                    config.policy_action_dim,
                    dtype=torch.bool,
                    device=action.device,
                )
            elif action_dim_is_pad.ndim == 1:
                action_dim_is_pad = action_dim_is_pad.unsqueeze(0).expand(batch_size, -1)
            action_payload = {
                "value": action[index],
                "action_dim_is_pad": action_dim_is_pad[index],
            }
            action_op_mask = batch.get("action_op_mask")
            if isinstance(action_op_mask, Tensor):
                action_payload["action_op_mask"] = (
                    action_op_mask[index] if action_op_mask.ndim > 1 else action_op_mask
                )
            else:
                action_payload["action_op_mask"] = ~action_dim_is_pad[index]
            sample["action"] = action_payload
        samples.append(sample)
    prepared = dict(batch)
    prepared["samples"] = samples
    prepared["pixel_values"] = pixel_values
    prepared[G05_RUNTIME_PREDICT_COT] = run_predict_cot
    return prepared
