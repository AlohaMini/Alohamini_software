"""Physical coordinate precision and gradients across CPU and Metal."""

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("kind", ["pi0", "pi05"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("state_tensor", [False, True])
def test_coordinate_transforms_preserve_fp64_offsets_and_device(kind, device, state_tensor):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Requires a Mac with MPS access")
    from alohamini.policies.pi0.processor_pi0 import PI0Processor
    from alohamini.policies.pi05.processor_pi05 import PI05Processor

    low = np.zeros(18, dtype=np.float64)
    low[0] = -1e8
    if kind == "pi0":
        stats = {key: {"mean": low, "std": np.ones(18)} for key in ("state", "actions")}
        processor = PI0Processor(stats, tokenizer=None, delta_dims=(0,))
        scale, center = 1 + 1e-6, low
    else:
        stats = {key: {"q01": low, "q99": low + 2} for key in ("state", "actions")}
        processor = PI05Processor(stats, tokenizer=None, delta_dims=(0,))
        scale, center = (2 + 1e-6) / 2, low + (2 + 1e-6) / 2
    state = np.zeros((1, 18), dtype=np.float64)
    state[0, 0] = 1e8 + 0.001
    offset = state[:, None, :]
    expected = torch.tensor((center + offset).repeat(2, axis=1), dtype=torch.float32)
    state = torch.from_numpy(state) if state_tensor else state
    normalized = torch.zeros((1, 2, 32), device=device, requires_grad=True)
    restored = processor.restore_actions(normalized, state)
    assert restored.device == normalized.device and restored.dtype == torch.float32
    torch.testing.assert_close(restored.cpu(), expected, rtol=0, atol=0)
    restored.sum().backward()
    expected_gradient = torch.zeros_like(normalized, device="cpu")
    expected_gradient[..., :18] = scale
    torch.testing.assert_close(normalized.grad.cpu(), expected_gradient)

    physical = expected.to(device).requires_grad_()
    encoded = processor.encode_actions(physical, state)
    oracle = processor.encode_actions(expected, state)
    assert encoded.device == physical.device and encoded.dtype == torch.float32
    torch.testing.assert_close(encoded.cpu(), oracle, rtol=0, atol=0)
    encoded.sum().backward()
    torch.testing.assert_close(physical.grad.cpu(), torch.full_like(expected, 1 / scale))
