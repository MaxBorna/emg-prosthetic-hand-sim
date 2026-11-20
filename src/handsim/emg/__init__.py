"""EMG pipeline (Phase 1): NinaPro DB2 loading, filtering, windowing,
features (RMS, MAV, WL), and classifiers (LDA baseline, later SVM/CNN/LSTM).

Typical use:
    rec = load_db2(subject=1, exercise=2)
    X, y, rep = make_dataset(rec, classes=[0, 18, 32])
"""

from handsim.emg.dataset import make_dataset, split_by_repetition
from handsim.emg.features import FEATURES, extract_features
from handsim.emg.filters import preprocess
from handsim.emg.loader import FS_HZ, Recording, load_db2
from handsim.emg.windows import sliding_windows

__all__ = [
    "FEATURES",
    "FS_HZ",
    "Recording",
    "extract_features",
    "load_db2",
    "make_dataset",
    "preprocess",
    "sliding_windows",
    "split_by_repetition",
]
