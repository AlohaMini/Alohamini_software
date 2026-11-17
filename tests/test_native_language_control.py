"""Native controller drops old-phase actions and shares explicit language queries."""

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from test_dataset import jpeg
from test_fineart_vla import tiny_options
from test_language_data import annotated_v3
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.fineart_vla.adapter import FineARTVLAAlgorithm


def test_native_subtask_updates_invalidate_queues_and_vqa_does_not_advance_control(
    recording, tmp_path, monkeypatch
):
    recipe = FineARTVLAAlgorithm()
    options = tiny_options(tmp_path / "assets", monkeypatch)
    data = AlohaMiniDataset(
        annotated_v3(recording, tmp_path / "data"),
        episodes=[0],
        allow_unannotated=True,
        state="recorded",
        image_size=(32, 32),
        **recipe.sample_spec(options),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    model = recipe.build(options)
    stats = recipe.statistics(data)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    policy = NativePolicy(checkpoint, offline=True, task="pick cup", language_replan_interval_s=1)
    monkeypatch.setattr(policy, "_check_calibration", lambda _: None)
    tokenizer = policy.processor.text.tokenizer
    responses = iter(
        [
            [tokenizer.convert_tokens_to_ids("approach")],
            [tokenizer.convert_tokens_to_ids("approach")],
            [tokenizer.convert_tokens_to_ids("grasp")],
            [
                tokenizer.convert_tokens_to_ids("grasp"),
                tokenizer.convert_tokens_to_ids("<loc0512>"),
                tokenizer.convert_tokens_to_ids("<loc0256>"),
            ],
        ]
    )

    def generate(prepared, **kwargs):
        assert not prepared["predict_actions"].any()
        assert not (prepared["text_labels"] != -100).any()
        return [next(responses)]

    monkeypatch.setattr(policy.model, "generate_text_tokens", generate)
    tasks = []
    query = policy.algorithm.query_language

    def capture(model, processor, batch, **kwargs):
        tasks.extend(batch["task"])
        return query(model, processor, batch, **kwargs)

    monkeypatch.setattr(policy.algorithm, "query_language", capture)
    snapshot = SimpleNamespace(
        payload=dict(
            zip(policy.selection.source_names, data[0]["observation.state"].tolist(), strict=True)
        ),
        images={"forward": jpeg()},
        request_started_s=100.0,
        received_s=100.0,
    )
    first = policy._observation_batch(snapshot)
    prompt = tokenizer.decode(first["observation"].tokenized_prompt[0], skip_special_tokens=True)
    assert "approach" in prompt and policy.task == "pick cup"
    policy.model._actions = deque([torch.full((1, 32), 999.0)])
    policy._execution_actions.append("old")
    snapshot.request_started_s = 100.5
    policy._observation_batch(snapshot)
    assert len(policy.model._actions) == len(policy._execution_actions) == 1
    snapshot.request_started_s = 101.0
    policy._observation_batch(snapshot)  # Same generated subtask retains queued actions.
    assert len(policy.model._actions) == len(policy._execution_actions) == 1
    snapshot.request_started_s = 102.0
    changed = policy._observation_batch(snapshot)
    assert not policy.model._actions and not policy._execution_actions
    assert "grasp" in tokenizer.decode(
        changed["observation"].tokenized_prompt[0], skip_special_tokens=True
    )
    assert tasks == ["pick cup"] * 3
    previous = policy._language_controller.last_query_s
    policy._execution_actions.append("keep")
    answer = policy.query_language(
        data[0], kind="vqa", camera="forward", question="Where is the cup?"
    )
    assert answer["answer"]["point"] == pytest.approx([0.25, 0.5], abs=1e-3)
    assert policy._language_controller.last_query_s == previous
    assert list(policy._execution_actions) == ["keep"]
    with pytest.raises(ValueError, match="not selected"):
        policy.query_language(data[0], kind="vqa", camera="missing", question="Where?")
    policy.reset()
    assert policy._language_controller.subtask is None
    monkeypatch.setattr(policy.model, "generate_text_tokens", lambda *args, **kwargs: [[7]])
    action = policy.select_action(snapshot)
    assert set(action) == set(policy.names)
    assert all(torch.isfinite(torch.tensor(value)) for value in action.values())


def test_cloud_cli_forwards_language_controller_setting(monkeypatch):
    from alohamini.remote import cli, server

    captured = {}

    class Backend:
        def __init__(self, path, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(server, "NativePlanPolicy", Backend)
    monkeypatch.setattr(server, "serve", lambda *args, **kwargs: None)
    assert cli.main(["serve", "--policy.path=/checkpoint", "--language-replan-interval-s=2.5"]) == 0
    assert captured["overrides"]["language_replan_interval_s"] == 2.5
