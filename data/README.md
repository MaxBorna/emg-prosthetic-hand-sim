# data/

Nothing in this folder is committed except this file.

## NinaPro DB2

Public download, no login required (http://ninapro.hevs.ch/instructions/DB2.html).
Each subject is a ~450 MB zip; start with a few subjects.

```bash
mkdir -p data/raw/ninapro_db2 && cd data/raw/ninapro_db2
for s in 1 2 3; do curl -L -O https://ninapro.hevs.ch/files/DB2_Preproc/DB2_s$s.zip; done
for f in DB2_s*.zip; do unzip -q "$f"; done
```

Resulting layout (verified):
```
data/raw/ninapro_db2/DB2_s1/S1_E1_A1.mat   exercise B, 17 finger movements   (~145 MB)
data/raw/ninapro_db2/DB2_s1/S1_E2_A1.mat   exercise C, 23 grasps            (~230 MB)
data/raw/ninapro_db2/DB2_s1/S1_E3_A1.mat   exercise D, 9 force patterns     (~70 MB)
```

Each file is MATLAB v5 (`scipy.io.loadmat` works). Variables used here:
`emg` (N x 12, float32, volts, 2 kHz), `restimulus` (refined label, 0 = rest),
`rerepetition` (1..6). Labels are global: E1 = 1-17, E2 = 18-40, E3 = 41-49.
`load_db2` reads only these three, which keeps memory to ~150 MB per file.

If you use the data, cite: Atzori et al., "Electromyography data for non-invasive
naturally-controlled robotic hand prostheses", *Scientific Data* 1, 140053 (2014).
