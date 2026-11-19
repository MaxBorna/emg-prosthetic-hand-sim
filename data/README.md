# data/

Nothing in this folder is committed except this file.

## NinaPro DB2

1. Register and download from the official NinaPro site (http://ninapro.hevs.ch/).
   Use is subject to the dataset's own license terms, which is why it is not
   redistributed here.
2. Unzip so the layout is:
   ```
   data/raw/ninapro_db2/DB2_s1/S1_E1_A1.mat
   data/raw/ninapro_db2/DB2_s1/S1_E2_A1.mat
   ...
   ```
3. Derived features go in `data/processed/` (also ignored).

DB2 is several GB in total. On a machine with 8 GB RAM, process one subject at
a time and cache windowed features to disk rather than loading everything at once.
