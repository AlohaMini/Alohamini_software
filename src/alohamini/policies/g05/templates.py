# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
# Modified for LeRobot and native Alohamini integration in 2026.

# ruff: noqa: E501

"""Author G0.5 flow and CoT prompt templates."""

G05_COT_PROMPTS: dict[tuple[str, ...], str] = {
    ("bbox",): "predict bbox",
    ("subtask",): "predict subtask",
    ("bbox", "subtask"): "predict bbox, subtask and action",
}


def _g05_user_turn(num_images: int) -> str:
    """The chat-wrapped user turn shared by the System 1 and CoT templates."""
    images = "".join(f"<image{index}_image_!>" for index in range(num_images))
    return (
        f"<chat_user_prefix>{images}<bos>"
        "Embodiment: <embodiment_text_!>; Task: <command_text_!_200> "
        "State: <proprio_proprio_!>;"
        "<chat_user_suffix><chat_assistant_prefix>"
    )


def make_g05_prompt_template(num_images: int, *, predict_cot: bool, flow_only: bool) -> str:
    """Reproduce the selected author SamplesBuilder template exactly."""

    if predict_cot:
        return make_g05_cot_prompt_template(num_images, fields=("subtask",), flow_only=flow_only)
    if flow_only:
        # BaseActionSamplesBuilderFMOnly intentionally has no chat wrapper.
        images = "".join(f"<image{index}_image_!>" for index in range(num_images))
        return (
            f"{images}<bos>Embodiment: <embodiment_text_!>; "
            "Task: <command_text_!_200> State: <proprio_proprio_!>;\n"
            "Action: <EOV><EOC><eos>"
        )
    return f"{_g05_user_turn(num_images)}Action: <EOV><EOC><action_action>|<eos>"


def make_g05_cot_prompt_template(
    num_images: int,
    *,
    fields: tuple[str, ...],
    flow_only: bool,
) -> str:
    """Build the exact author template for a selected Subtask/BBox CoT format."""

    supported_fields = {"bbox", "subtask"}
    if not fields or not set(fields) <= supported_fields:
        raise ValueError(
            f"G0.5 CoT fields must be a non-empty subset of {sorted(supported_fields)}, got {fields}."
        )
    # The released BBoxSubtaskCoTBuilder emits BBox before Subtask. Preserve
    # checkpoint serialization even though the paper's schematic orders the
    # independently composable labels differently.
    ordered_fields = tuple(field for field in ("bbox", "subtask") if field in fields)
    placeholders = {
        "bbox": "<bbox_text>|",
        "subtask": "<atomic_task_text>|",
    }
    action = "Action: <EOV><eos>" if flow_only else "Action: <EOV><action_action>|<eos>"
    return (
        f"{_g05_user_turn(num_images)}<prompt_text_!>\n<EOC>"
        + "".join(placeholders[field] for field in ordered_fields)
        + action
    )
