"""Sliding-window segmentation."""

from __future__ import annotations

import numpy as np


def sliding_windows(n_samples: int, fs: int, length_ms: float = 200, step_ms: float = 50):
    """Return start indices and window length in samples."""
    win = int(round(length_ms * fs / 1000))
    step = int(round(step_ms * fs / 1000))
    if n_samples < win:
        return np.empty(0, dtype=int), win
    return np.arange(0, n_samples - win + 1, step), win
