"""Training-only numeric statistics for each declared temporal offset."""

from copy import deepcopy

import numpy as np

from alohamini.datasets.statistics import ExactQuantileStats


def window_statistics(samples, windows, *, base, batch_size=128):
    """Fit each offset separately, excluding virtual episode padding and image reads.

    ``windows`` maps numeric field names to their exact dataset row offsets.
    Coordinates are already in the dataset view's declared representation.
    """
    if not isinstance(windows, dict) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Window statistics require named offsets and a positive batch size")
    for key, offsets in windows.items():
        if (
            key.startswith("observation.images.")
            or key not in base
            or not isinstance(offsets, (list, tuple))
            or not offsets
            or any(type(i) is not int for i in offsets)
            or len(set(offsets)) != len(offsets)
            or list(offsets) != samples.delta_indices.get(key)
        ):
            raise ValueError(f"{key}: statistics offsets must match the selected numeric window")
    result = deepcopy(base)
    if not windows:
        return result
    trackers = {key: [ExactQuantileStats() for _ in offsets] for key, offsets in windows.items()}
    counts = {key: np.zeros(len(offsets), dtype=np.int64) for key, offsets in windows.items()}
    pending = []

    def consume():
        for key, group in trackers.items():
            values = np.stack([row[key].numpy() for row in pending])
            if values.ndim != 3 or values.shape[1] != len(group):
                raise ValueError(f"{key}: window statistics need [samples,time,coordinates]")
            valid = np.stack([~row[key + "_is_pad"].numpy() for row in pending])
            for offset, tracker in enumerate(group):
                selected = values[valid[:, offset], offset]
                if len(selected):
                    tracker.update(selected)
                    counts[key][offset] += len(selected)
        pending.clear()

    for index in range(len(samples)):
        pending.append(samples.numeric_sample(index, windows))
        if len(pending) == batch_size:
            consume()
    if pending:
        consume()
    for key, group in trackers.items():
        empty = [
            offset for offset, count in zip(windows[key], counts[key], strict=True) if count == 0
        ]
        if empty:
            raise ValueError(f"{key}: no real training targets at temporal offsets {empty}")
        statistics = [tracker.get_statistics() for tracker in group]
        result[key] = {
            name: np.stack([row[name] for row in statistics]).tolist() for name in statistics[0]
        }
        result[key]["offsets"] = list(windows[key])
    return result
