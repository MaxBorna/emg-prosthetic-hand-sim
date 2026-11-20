"""EMG pipeline tests on synthetic data (CI has no NinaPro files)."""

import numpy as np
import scipy.io as sio

from handsim.emg import extract_features, load_db2, make_dataset, sliding_windows
from handsim.emg.dataset import _rest_repetition
from handsim.emg.features import mav, rms, wl
from handsim.emg.loader import Recording

FS = 2000


def synthetic_recording(seed=0, n_reps=6, classes=(18, 32)) -> Recording:
    """rest 1 s -> movement 2 s, repeated; class k activates channel k strongly."""
    rng = np.random.default_rng(seed)
    emg, lab, rep = [], [], []
    for c_i, c in enumerate(classes):
        for r in range(1, n_reps + 1):
            for label, dur in ((0, FS), (c, 2 * FS)):
                amp = np.full(12, 1e-5)
                if label:
                    amp[c_i] = 2e-4
                emg.append(rng.standard_normal((dur, 12)) * amp)
                lab.append(np.full(dur, label))
                rep.append(np.full(dur, 0 if label == 0 else r))
    return Recording(1, 2, np.concatenate(emg).astype(np.float32),
                     np.concatenate(lab), np.concatenate(rep))


def test_features_known_values():
    x = np.array([[1.0], [-1.0], [1.0], [-1.0]])
    assert mav(x)[0] == 1.0
    assert rms(x)[0] == 1.0
    assert wl(x)[0] == 6.0
    assert extract_features(np.ones((10, 12))).shape == (36,)


def test_window_counts():
    starts, win = sliding_windows(FS, FS, 200, 50)  # 1 s of data
    assert win == 400
    assert len(starts) == (FS - 400) // 100 + 1


def test_rest_gets_next_repetition():
    lab = np.array([0, 0, 5, 5, 0, 0, 5, 5])
    rep = np.array([0, 0, 1, 1, 0, 0, 2, 2])
    assert _rest_repetition(lab, rep).tolist() == [1, 1, 1, 1, 2, 2, 2, 2]


def test_no_window_spans_two_repetitions_or_labels():
    rec = synthetic_recording()
    X, y, r = make_dataset(rec)
    assert set(np.unique(y)) == {0, 18, 32}
    assert set(np.unique(r)) == set(range(1, 7))
    # Every window must lie inside one constant label/repetition run.
    rep_full = _rest_repetition(rec.label, rec.repetition)
    starts, win = sliding_windows(len(rec.emg), FS)
    kept = [s for s in starts
            if (rec.label[s:s + win] == rec.label[s]).all()
            and (rep_full[s:s + win] == rep_full[s]).all()]
    assert len(kept) == len(y)


def test_lda_separates_synthetic_classes():
    from handsim.emg import split_by_repetition
    from handsim.emg.classify import make_classifier

    X, y, r = make_dataset(synthetic_recording())
    Xtr, ytr, Xte, yte = split_by_repetition(X, y, r)
    acc = (make_classifier("lda").fit(Xtr, ytr).predict(Xte) == yte).mean()
    assert acc > 0.95


def test_loader_reads_ninapro_layout(tmp_path):
    rec = synthetic_recording()
    d = tmp_path / "DB2_s7"
    d.mkdir()
    sio.savemat(d / "S7_E2_A1.mat", {
        "emg": rec.emg, "restimulus": rec.label[:, None].astype(np.int8),
        "rerepetition": rec.repetition[:, None].astype(np.int8),
        "glove": np.zeros((len(rec.emg), 22), np.float32),
    })
    out = load_db2(7, 2, root=tmp_path)
    assert out.emg.shape == rec.emg.shape
    assert (out.label == rec.label).all()
