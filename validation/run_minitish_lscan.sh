#!/bin/bash
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"
module load pytorch/2.8.0
for L in 800 1200 1600; do
  echo "== mini-tish 50 km lmax=$L =="
  D=$Q/dsm_arbitration_src50_ref2
  ARB_ROOT=$D python3 dsm_arbitration/run_minitish.py homog_q50 32 $L \
    || { echo MINITISH-FAILED l$L; continue; }
  mv $D/minitish_homog_q50.npz $D/minitish_homog_q50_l$L.npz
done
echo "== LSCAN DONE $(date +%T) =="
