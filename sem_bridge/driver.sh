#!/bin/bash
# ak135 hdur80 SEM-bridge stage 1: two ladder rungs + both conj
# variants + comparator metrics, then figures under pytorch module.
set -e
cd /pscratch/sd/y/ytian159/astroseis_qtest/sem_bridge
module load cray-python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYASTROSEIS_FL_LARGEX=1

python3 probe_highk.py

python3 run_leg.py 40 420 60
python3 synth_export.py out_ak135/spectral_h040.npz out_ak135/uxyz_h040_c1 1
python3 synth_export.py out_ak135/spectral_h040.npz out_ak135/uxyz_h040_c0 0
python3 compare_bridge.py out_ak135/uxyz_h040_c1 h040_c1
python3 compare_bridge.py out_ak135/uxyz_h040_c0 h040_c0

python3 run_leg.py 20 420 48
python3 synth_export.py out_ak135/spectral_h020.npz out_ak135/uxyz_h020_c1 1
python3 synth_export.py out_ak135/spectral_h020.npz out_ak135/uxyz_h020_c0 0
python3 compare_bridge.py out_ak135/uxyz_h020_c1 h020_c1
python3 compare_bridge.py out_ak135/uxyz_h020_c0 h020_c0

module unload cray-python
module load pytorch/2.8.0
python3 compare_bridge.py out_ak135/uxyz_h020_c1 h020_c1 --plots
python3 compare_bridge.py out_ak135/uxyz_h020_c0 h020_c0 --plots
echo "DRIVER DONE"
