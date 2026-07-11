#!/bin/bash
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"
module load pytorch/2.8.0
echo "== V2: mini-tish at 637 km =="
ARB_ROOT=$Q/dsm_arbitration_ref637 python3 dsm_arbitration/run_minitish.py homog_q50 32 \
  || { echo MINITISH-FAILED ref637; exit 1; }
python3 dsm_arbitration/minitish_compare.py $Q/dsm_arbitration_ref637 minitish_homog_q50.npz homog_q50 mrt \
  || echo MINITISH-COMPARE-FAILED ref637
echo "== 50 km: mini-tish production =="
ARB_ROOT=$Q/dsm_arbitration_src50_ref2 python3 dsm_arbitration/run_minitish.py homog_q50 32 \
  || { echo MINITISH-FAILED src50; exit 1; }
echo "== MINITISH RUNS DONE $(date +%T) =="
