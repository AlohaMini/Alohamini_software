"""Native VQ-BeT stage accounting, accumulation and checkpoint restoration."""

import pytest
import torch
from test_native_learning import recording as recording

from alohamini.learning.data import AlohaMiniDataset
from alohamini.learning.policy import NativePolicy, save_checkpoint
from alohamini.policies.registry import algorithm


def tiny_options():
    return dict(
        n_obs_steps=2,
        action_chunk_size=2,
        n_action_pred_token=2,
        pretrained_backbone_weights=None,
        crop_shape=None,
        n_vqvae_training_steps=2,
        vqvae_n_embed=4,
        vqvae_embedding_dim=8,
        vqvae_enc_hidden_dim=16,
        gpt_block_size=32,
        gpt_input_dim=32,
        gpt_output_dim=32,
        gpt_n_layer=1,
        gpt_n_head=2,
        gpt_hidden_dim=32,
        spatial_softmax_num_keypoints=4,
        dropout=0.0,
    )


def assembled(recording):
    recipe = algorithm("vqbet")
    options = tiny_options()
    data = AlohaMiniDataset(
        recording,
        episodes=[0],
        state="joint_position,base_velocity,lift_height",
        image_size=(32, 32),
        **recipe.sample_spec(options, cameras=["forward"]),
    )
    options.update(input_features=data.input_features, output_features=data.output_features)
    stats = recipe.statistics(data)
    model = recipe.build(options)
    processor = recipe.processor(model, stats, "cpu")
    batch = processor(next(iter(torch.utils.data.DataLoader(data, batch_size=2))))
    return recipe, data, stats, model, processor, batch


def test_stage_counter_changes_only_after_success_and_survives_reload(recording, tmp_path):
    recipe, data, stats, model, processor, batch = assembled(recording)
    quantizer = model.vqbet.action_head.vqvae_model
    optimizer = torch.optim.Adam(model.get_optim_params(), lr=1e-4)
    # Multiple forwards and validation do not count as optimizer updates.
    for _ in range(2):
        model(batch)[0].backward()
    assert quantizer.optimized_steps.item() == 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.update()
    assert quantizer.optimized_steps.item() == 1 and not quantizer.discretized.item()
    model.eval()
    buffers = {k: v.clone() for k, v in quantizer.named_buffers()}
    with torch.no_grad():
        model(batch)
    for key, value in quantizer.named_buffers():
        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, stats, data, training={})
    restored = NativePolicy(checkpoint, offline=True)
    clone = restored.model
    counter = clone.vqbet.action_head.vqvae_model
    assert counter.optimized_steps.item() == 1 and not counter.discretized.item()
    clone.train()
    clone(batch)[0].backward()
    clone.update()
    assert counter.optimized_steps.item() == 2 and counter.discretized.item()
    clone.zero_grad(set_to_none=True)
    loss, diagnostics = clone(batch)
    loss.backward()
    assert torch.isfinite(loss) and "classification_loss" in diagnostics
    assert clone.vqbet.policy.lm_head.weight.grad is not None
    clone.eval()
    action = clone.predict_action_chunk(batch)
    assert action.shape == (1, 2, 18)
    predicted = recipe.predict_action_samples(clone, processor, batch, fps=30)
    assert predicted.shape == batch["action"].shape and torch.isfinite(predicted).all()


def test_quantizer_stage_honors_unequal_microbatch_sample_weights(recording):
    recipe, data, stats, model, processor, batch = assembled(recording)
    model.eval()  # Keeps EMA codebooks fixed during the numeric comparison.
    state = torch.get_rng_state()
    full = model(batch)[0]
    torch.set_rng_state(state)
    scaled = model({**batch, "_loss_weight": 0.25})[0]
    torch.testing.assert_close(scaled, full * 0.25, rtol=0, atol=0)


def test_staged_scheduler_and_resume_state():
    from alohamini.learning.optim import make_optimizer_and_scheduler, resolve_optimization

    class Model(torch.nn.Linear):
        get_optim_params = torch.nn.Linear.parameters

    model = Model(1, 1)
    cfg = resolve_optimization(
        {
            "steps": 6,
            "optimizer": {"type": "adam", "lr": 0.1},
            "scheduler": {
                "type": "staged_cosine",
                "start_step": 2,
                "warmup_steps": 1,
                "decay_steps": 6,
            },
        },
        type("Config", (), {"optimizer_lr": 0.1, "optimizer_weight_decay": 0.0})(),
    )
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, model)
    rates = []
    for _ in range(7):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[:4] == pytest.approx([0.1, 0.1, 0.0, 0.1])
    assert rates[4] > rates[5] > rates[6] == 0
