#!/bin/bash
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"
module load pytorch/2.8.0
export OMP_NUM_THREADS=4
export ARB_ROOT=$Q/dsm_arbitration_u3 ARB_MODELS=corefluid_q50 ARB_NEAR_TIER=1.5:10:2 ARB_ELIM=1
python3 dsm_arbitration/run_bem.py corefluid_q50 24 || { echo U3-BEM-FAILED; exit 1; }
python3 dsm_arbitration/synthesize_compare.py || echo U3-COMPARE-FAILED
echo "== U3 BEM RERUN DONE $(date +%T) =="
