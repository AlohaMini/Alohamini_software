import unittest

import torch

from alohamini.learning.observation_history import ObservationHistory
from alohamini.policies.attention import packed_attention, variable_attention


class ObservationHistoryTests(unittest.TestCase):
    def test_causal_time_sampling_warmup_and_sparse_arrivals(self):
        history = ObservationHistory({"image": [-2, -1, 0]}, fps=2)
        first = history.append({"image": torch.tensor([[10.0]])}, 100)
        self.assertEqual(first["image"].flatten().tolist(), [10, 10, 10])
        self.assertEqual(first["image_is_pad"].tolist(), [[True, True, False]])
        history.append({"image": torch.tensor([[20.0]])}, 100.5)
        result = history.append({"image": torch.tensor([[30.0]])}, 101.1)
        self.assertEqual(result["image"].flatten().tolist(), [10, 20, 30])
        self.assertFalse(result["image_is_pad"].any())
        with self.assertRaisesRegex(ValueError, "clock moved backwards"):
            history.append({"image": torch.tensor([[40.0]])}, 99)

    def test_history_owns_values_and_is_bounded(self):
        history = ObservationHistory({"image": [-2, 0]}, fps=10)
        data = torch.zeros(1, 1)
        history.append({"image": data}, 0)
        data.fill_(5)
        self.assertEqual(history.frames[0][1]["image"].item(), 0)
        for step in range(1, 100):
            history.append({"image": data}, step / 10)
        self.assertLessEqual(len(history.frames), 4)

    def test_future_offsets_rejected(self):
        with self.assertRaises(ValueError):
            ObservationHistory({"image": [0, 1]}, 30)


class PackedAttentionTests(unittest.TestCase):
    def test_variable_sequences_do_not_share_keys_or_gradients(self):
        torch.manual_seed(42)
        q = torch.randn(5, 4, 8, requires_grad=True)
        k = torch.randn(6, 2, 8, requires_grad=True)
        v = torch.randn(6, 2, 8, requires_grad=True)
        cuq, cuk = torch.tensor([0, 2, 5]), torch.tensor([0, 3, 6])
        result = variable_attention(q, k, v, cuq, cuk, 3, 3)
        expected = torch.cat(
            [
                packed_attention(q[:2][None], k[:3][None], v[:3][None])[0],
                packed_attention(q[2:][None], k[3:][None], v[3:][None])[0],
            ]
        )
        torch.testing.assert_close(result, expected)
        result[:2].sum().backward()
        self.assertEqual(v.grad[3:].count_nonzero(), 0)
        self.assertGreater(v.grad[:3].count_nonzero(), 0)

    def test_causal_decode_uses_bottom_right_alignment(self):
        q, k = torch.zeros(1, 1, 1, 4), torch.zeros(1, 3, 1, 4)
        v = torch.arange(3.0).reshape(1, 3, 1, 1).expand_as(k)
        torch.testing.assert_close(packed_attention(q, k, v, causal=True), torch.ones_like(q))


if __name__ == "__main__":
    unittest.main()
