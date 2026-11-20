"""Classic time-domain EMG features (Hudgins set and friends).

All functions take a window of shape (win, channels) and return (channels,).
"""

from __future__ import annotations

import numpy as np


def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value: overall activation level."""
    return np.mean(np.abs(x), axis=0)


def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square: like MAV but weights peaks more (~ signal power)."""
    return np.sqrt(np.mean(x**2, axis=0))


def wl(x: np.ndarray) -> np.ndarray:
    """Waveform length: cumulative |dx|, grows with both amplitude and frequency."""
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)


FEATURES = {"mav": mav, "rms": rms, "wl": wl}


def extract_features(window: np.ndarray, names=("rms", "mav", "wl")) -> np.ndarray:
    """Concatenate features: (channels * len(names),), grouped by feature."""
    return np.concatenate([FEATURES[n](window) for n in names])
