"""Signal conditioning for sEMG."""

from __future__ import annotations

import numpy as np
from scipy import signal


def preprocess(
    emg: np.ndarray,
    fs: int,
    band: tuple[float, float] = (20.0, 450.0),
    notch_hz: float | None = 50.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase band-pass (+ optional notch) along time (axis 0).

    Zero-phase (sosfiltfilt) is fine offline but is NOT causal; a real-time
    prosthesis must use a causal filter (sosfilt) and accept the phase lag.
    """
    sos = signal.butter(order, band, btype="bandpass", fs=fs, output="sos")
    out = signal.sosfiltfilt(sos, emg, axis=0)
    if notch_hz:
        b, a = signal.iirnotch(notch_hz, Q=30.0, fs=fs)
        out = signal.filtfilt(b, a, out, axis=0)
    return out.astype(np.float32)
