#!/bin/bash
# Stage-1 rerun at flat lmax=1400: verdict probe (same-staircase
# tipsv gate), then the h40 production rung + scoring + figures.
set -e
cd /pscratch/sd/y/ytian159/astroseis_qtest/sem_bridge
module load cray-python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYASTROSEIS_FL_LARGEX=1
export PYASTROSEIS_JUMP_UNSCALED=1
export PYASTROSEIS_JUMP_COLNORM=1

python3 probe_1400.py

python3 run_leg.py 40 1400 60
python3 synth_export.py out_ak135/spectral_h040.npz out_ak135/uxyz_h040_c1 1
python3 compare_bridge.py out_ak135/uxyz_h040_c1 h040_c1

module unload cray-python
module load pytorch/2.8.0
python3 compare_bridge.py out_ak135/uxyz_h040_c1 h040_c1 --plots
echo "DRIVER2 DONE"
