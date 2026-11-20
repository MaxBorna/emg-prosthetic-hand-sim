"""Load NinaPro DB2 recordings.

File layout (as downloaded from ninapro.hevs.ch, DB2_Preproc):
    data/raw/ninapro_db2/DB2_s{S}/S{S}_E{E}_A1.mat
Each file is MATLAB v5 with, among others:
    emg          (N, 12) float32, volts, 2 kHz
    restimulus   (N, 1)  int8, movement label refined a posteriori (0 = rest)
    rerepetition (N, 1)  int8, repetition index 1..6 (0 during rest)
Labels are numbered globally across exercises: E1 = 1..17, E2 = 18..40, E3 = 41..49.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio

from handsim.paths import DATA_DIR

FS_HZ = 2000
DB2_DIR = DATA_DIR / "raw" / "ninapro_db2"

# Exercise C (E2) movement names, from Atzori et al., Scientific Data 2014.
E2_MOVEMENTS = {
    0: "rest",
    18: "large diameter grasp",
    19: "small diameter grasp",
    20: "fixed hook grasp",
    21: "index finger extension grasp",
    22: "medium wrap",
    23: "ring grasp",
    24: "prismatic four fingers grasp",
    25: "stick grasp",
    26: "writing tripod grasp",
    27: "power sphere grasp",
    28: "three finger sphere grasp",
    29: "precision sphere grasp",
    30: "tripod grasp",
    31: "prismatic pinch grasp",
    32: "tip pinch grasp",
    33: "quadpod grasp",
    34: "lateral grasp",
    35: "parallel extension grasp",
    36: "extension type grasp",
    37: "power disk grasp",
    38: "open a bottle with a tripod grasp",
    39: "turn a screw",
    40: "cut something",
}


@dataclass
class Recording:
    subject: int
    exercise: int
    emg: np.ndarray  # (N, 12) float32
    label: np.ndarray  # (N,) int, restimulus
    repetition: np.ndarray  # (N,) int, rerepetition
    fs: int = FS_HZ

    @property
    def duration_s(self) -> float:
        return len(self.emg) / self.fs


def db2_path(subject: int, exercise: int, root: Path = DB2_DIR) -> Path:
    return Path(root) / f"DB2_s{subject}" / f"S{subject}_E{exercise}_A1.mat"


def load_db2(subject: int, exercise: int = 2, root: Path = DB2_DIR) -> Recording:
    """Load one subject/exercise. Only the needed variables are read, which keeps
    memory at ~150 MB per file instead of ~700 MB (glove/acc are skipped)."""
    path = db2_path(subject, exercise, root)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; see data/README.md for download steps")
    m = sio.loadmat(path, variable_names=["emg", "restimulus", "rerepetition"])
    return Recording(
        subject=subject,
        exercise=exercise,
        emg=m["emg"].astype(np.float32, copy=False),
        label=m["restimulus"].ravel().astype(np.int16),
        repetition=m["rerepetition"].ravel().astype(np.int16),
    )
