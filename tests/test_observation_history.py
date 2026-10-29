import unittest

import torch

from alohamini.learning.observation_history import ObservationHistory


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


if __name__ == "__main__":
    unittest.main()
