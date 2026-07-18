#!/bin/bash
# Apples-to-apples arbitration: tipsv on our exact 127-layer
# staircase (background, serial) + our spectral probe on the same
# stack, then the per-k verdict.
set -e
cd /pscratch/sd/y/ytian159/astroseis_qtest/sem_bridge
module load cray-python
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1
export PYASTROSEIS_FL_LARGEX=1

TIPSV=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi/tipsv-mpi/tipsv
cd stair_dsm
$TIPSV < stair127.inf > tipsv_stair.log 2>&1 &
TPID=$!
cd ..
python3 probe_stair.py
echo "waiting for tipsv (pid $TPID)"
wait $TPID
tail -2 stair_dsm/tipsv_stair.log
python3 compare_stair.py
echo "STAIR DRIVER DONE"
