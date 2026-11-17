"""Group-specific phase starts preserve common LR overrides and resume exactly."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization


class GroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = torch.nn.Parameter(torch.ones(1))
        self.core = torch.nn.Parameter(torch.ones(1))
        self.prompt = torch.nn.Parameter(torch.ones(1))
        self.head = torch.nn.Parameter(torch.ones(1))

    def get_optim_params(self):
        return [
            dict(name="vlm", params=[self.vlm], lr_scale=0.1),
            dict(name="core", params=[self.core]),
            dict(name="prompt", params=[self.prompt], lr_scale=0.2),
            dict(name="head", params=[self.head]),
        ]


def settings(kind, **overrides):
    return resolve_optimization(
        dict(
            steps=6,
            optimizer=dict(type="adamw", lr=0.01),
            scheduler=dict(
                type=kind,
                start_step=2,
                warmup_steps=1,
                decay_steps=6,
                before_start=dict(vlm=0.0, core=0.0),
                **overrides,
            ),
        ),
        SimpleNamespace(optimizer_lr=0.01, optimizer_weight_decay=0.0),
    )


def rates(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def update(model, optimizer, scheduler):
    sum(model.parameters()).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()


@pytest.mark.parametrize("kind", ["staged_cosine", "constant_with_warmup", "warmup_linear"])
def test_named_groups_hold_then_warm_up_and_resume_at_stage_boundary(kind):
    cfg = settings(kind)
    model = GroupedModel()
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    assert rates(optimizer) == pytest.approx([0, 0, 0.002, 0.01])
    update(model, optimizer, scheduler)
    assert rates(optimizer) == pytest.approx([0, 0, 0.002, 0.01])
    assert model.vlm.item() == model.core.item() == 1.0
    assert model.prompt.item() < 1.0 and model.head.item() < 1.0
    saved = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    target = GroupedModel()
    restored_optimizer, restored_scheduler = make_optimizer_and_scheduler(cfg, target)
    target.load_state_dict(saved[0])
    restored_optimizer.load_state_dict(saved[1])
    restored_scheduler.load_state_dict(saved[2])
    for step in range(2, 7):
        update(model, optimizer, scheduler)
        update(target, restored_optimizer, restored_scheduler)
        assert rates(optimizer) == rates(restored_optimizer)
        for expected, actual in zip(model.parameters(), target.parameters(), strict=True):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        if step == 2:
            assert rates(optimizer) == [0.0] * 4
        elif step == 3:
            assert rates(optimizer) == pytest.approx([0.001, 0.01, 0.002, 0.01])
    if kind in ("staged_cosine", "warmup_linear"):
        assert rates(optimizer) == [0.0] * 4
    else:
        assert rates(optimizer) == pytest.approx([0.001, 0.01, 0.002, 0.01])


@pytest.mark.parametrize(
    "factors", [[], {"": 0}, {"vlm": -1}, {"vlm": True}, {"vlm": float("nan")}]
)
def test_invalid_group_factors_are_rejected_before_training(factors):
    cfg = settings("staged_cosine")
    cfg["scheduler"]["before_start"] = factors
    with pytest.raises(ValueError, match="before_start"):
        resolve_optimization(cfg, SimpleNamespace(optimizer_lr=0.01, optimizer_weight_decay=0.0))


def test_group_name_must_exist_in_actual_optimizer():
    cfg = settings("staged_cosine")
    cfg["scheduler"]["before_start"] = {"typo": 0}
    with pytest.raises(ValueError, match="unknown parameter groups"):
        make_optimizer_and_scheduler(cfg, GroupedModel())
