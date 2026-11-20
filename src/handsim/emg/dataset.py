"""Turn a Recording into (X, y, repetition) feature matrices without leakage."""

from __future__ import annotations

import numpy as np

from handsim.emg.features import extract_features
from handsim.emg.filters import preprocess
from handsim.emg.loader import Recording
from handsim.emg.windows import sliding_windows

# NinaPro's standard protocol (Atzori 2014 / Gijsberts 2014)
TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)


def _rest_repetition(label: np.ndarray, repetition: np.ndarray) -> np.ndarray:
    """Rest samples have repetition 0. Assign each rest segment the repetition of
    the movement that follows it, so rest can be split train/test like the rest."""
    rep = repetition.copy()
    nxt = 0
    for i in range(len(rep) - 1, -1, -1):
        if label[i] != 0:
            nxt = rep[i]
        elif rep[i] == 0:
            rep[i] = nxt
    return rep


def make_dataset(
    rec: Recording,
    classes=None,
    length_ms: float = 200,
    step_ms: float = 50,
    features=("rms", "mav", "wl"),
    filtered: bool = True,
):
    """Window the recording and compute features.

    A window is kept only if every sample in it has the same label and the same
    repetition. That discards transition windows (ambiguous labels) and, more
    importantly, guarantees no window spans two repetitions, which would leak
    data between the train and test split.
    """
    emg = preprocess(rec.emg, rec.fs) if filtered else rec.emg
    rep = _rest_repetition(rec.label, rec.repetition)
    starts, win = sliding_windows(len(emg), rec.fs, length_ms, step_ms)

    X, y, r = [], [], []
    for s in starts:
        lab = rec.label[s : s + win]
        rp = rep[s : s + win]
        if lab[0] != lab[-1] or rp[0] != rp[-1]:
            continue  # fast reject; exact check below
        if (lab != lab[0]).any() or (rp != rp[0]).any() or rp[0] == 0:
            continue
        if classes is not None and lab[0] not in classes:
            continue
        X.append(extract_features(emg[s : s + win], features))
        y.append(lab[0])
        r.append(rp[0])
    return np.asarray(X, dtype=np.float32), np.asarray(y), np.asarray(r)


def split_by_repetition(X, y, rep, train=TRAIN_REPS, test=TEST_REPS):
    tr = np.isin(rep, train)
    te = np.isin(rep, test)
    return X[tr], y[tr], X[te], y[te]
