# EMG-Controlled Prosthetic Hand (Simulation-First)

An underactuated prosthetic hand controlled by surface EMG, developed simulation-first in
[MuJoCo](https://mujoco.org/) and trained/evaluated on the public
[NinaPro DB2](http://ninapro.hevs.ch/) dataset, with an optional path to real hardware
(ESP32 + servos).

> **Status:** Phase 0 done (scripted grasp, CI). Phase 1 in progress: feature + LDA/SVM baseline on NinaPro DB2.

## Goal

Decode a user's grasp intent from forearm EMG and execute it robustly with an
underactuated hand, then quantify how decoding errors and latency affect grasp success.

## Architecture

```
 EMG (NinaPro DB2)        intent decoding              control                 simulation
┌──────────────┐   ┌──────────────────────────┐   ┌───────────────┐   ┌──────────────────────┐
│ raw sEMG     │──▶│ filter → 200 ms windows  │──▶│ Intent →      │──▶│ MuJoCo hand (MJCF)   │
│ 12 ch, 2 kHz │   │ → RMS/MAV/WL → LDA/SVM   │   │ joint/tendon  │   │ tendon-coupled       │
└──────────────┘   │   (later 1D-CNN / LSTM)  │   │ targets (PID/ │   │ fingers + objects    │
                   └──────────────────────────┘   │ impedance)    │   └──────────┬───────────┘
                               Intent(grasp, level)└───────────────┘              │
                                                                     eval: accuracy, LOSO,
                                                                     grasp success, latency
```

The **`Intent`** dataclass (`src/handsim/control/intent.py`) is the fixed contract between
decoding and control. Any decoder produces one, and any controller consumes one, so either
side (or the simulator vs. real hardware) can be swapped independently.

## Repository layout

```
models/            MJCF/URDF models (gripper_v0 now, own Onshape CAD hand later)
src/handsim/
  sim/             model loading, stepping, sensors
  emg/             NinaPro loading, filtering, windowing, features, classifiers
  control/         Intent, scripted grasps, PID/impedance, later RL
  eval/            metrics, leave-one-subject-out evaluation
scripts/           runnable entry points
configs/           experiment configs (YAML)
notebooks/         exploration only
tests/             pytest, runs headless in CI
data/              gitignored; see data/README.md for download instructions
```

## Roadmap

| Phase | Goal | Done when |
|---|---|---|
| **0** | Setup + scripted grasp | Gripper closes and lifts an object; tests pass in CI ✅ |
| **1** | EMG classifier | LDA/SVM on RMS/MAV/WL features, leave-one-subject-out accuracy reported; then 1D-CNN/LSTM compared against it |
| **2** | EMG → sim closed loop | Recorded EMG drives the simulated hand; grasp success vs. classifier accuracy and latency |
| **3** | Own hand design | Onshape CAD exported to MJCF/URDF and replaces `gripper_v0` |
| **4** | Adaptive grasping | Continuous intensity (regression) and/or RL-based grasp adaptation |
| **5** | Sim-to-real (optional) | ESP32 + servos running the same `Intent` interface |

## Setup

```bash
git clone https://github.com/MaxBorna/emg-prosthetic-hand-sim.git
cd emg-prosthetic-hand-sim
conda env create -f environment.yml
conda activate hand
pip install -e ".[dev]"
nbstripout --install        # keeps notebook diffs clean
pytest -q
```

## Phase 0: scripted grasp

```bash
python scripts/phase0_scripted_grasp.py              # headless, prints result
mjpython scripts/phase0_scripted_grasp.py --viewer   # interactive viewer (macOS needs mjpython)
```

`gripper_v0` is a two-finger gripper in which **one** actuator pulls **one** tendon spanning
all four finger joints. Joint springs set how the motion is split between links, so the
fingers adapt to the object's shape without per-joint control. Because the tendon also
couples the two fingers, it can settle asymmetrically (one finger curls more), which is
expected underactuated behaviour and something to evaluate rather than hide.

## Phase 1: EMG baseline

```bash
python scripts/train_emg_baseline.py --subjects 1 2 3 --classes 0 18 32          # LDA
python scripts/train_emg_baseline.py --subjects 1 2 3 --classes 0 18 32 --clf svm
```

Pipeline: band-pass 20–450 Hz + 50 Hz notch → 200 ms windows, 50 ms step → RMS/MAV/WL per
channel (36 features) → standardise → LDA or RBF-SVM. Windows that straddle a label or
repetition boundary are dropped, so train/test (repetitions 1,3,4,6 vs 2,5) never share data.

First results, exercise 2, subjects 1–3, **balanced** accuracy (rest dominates the data, so
plain accuracy is inflated):

| Setup | Classes | Within-subject | Leave-one-subject-out |
|---|---|---|---|
| LDA | rest, large diameter, tip pinch | 72–89 % | 46–69 % |
| SVM | rest, large diameter, tip pinch | 82–96 % | 36–40 % |
| LDA | all 23 grasps + rest | 46–55 % | 7–10 % (chance 4 %) |

Takeaways so far: grasp recognition works well when the model is calibrated on the user;
it largely fails on an unseen subject, and the more flexible SVM overfits the training
subjects worse than LDA. With only 2 training subjects in LOSO these numbers are noisy.

## Data and license

Code: MIT (see `LICENSE`). NinaPro data is **not** included. Download it yourself under its
own terms (see `data/README.md`) and cite Atzori et al., *Scientific Data* 1, 140053 (2014).
