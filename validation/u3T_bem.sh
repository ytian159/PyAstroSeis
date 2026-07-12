#!/bin/bash
# SH-pure meridian-station BEM re-evaluation + toroidal mode-sum
set -uo pipefail
cd /pscratch/sd/y/ytian159/astroseis_qtest
export ARB_ROOT=$PWD/dsm_arbitration_u3T
export ARB_ELIM=1
export OMP_NUM_THREADS=4
module load pytorch/2.8.0
python3 dsm_arbitration/run_bem.py corefluid_q50 24 || echo U3T-BEM-FAILED
python3 dsm_arbitration/run_tormodes.py dsm_arbitration_u3T corefluid_q50 50 -1 || echo U3T-TM-FAILED
echo "== U3T DONE $(date +%T) =="
