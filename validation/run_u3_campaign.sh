#!/bin/bash
# uniform3_q50 (== corefluid_q50 constants) 50-km-source arbitration:
# BEM + DSM(tipsv) legs via the standard campaign, then SPECFEM legs.
# Stage 1 runs inside a compute allocation (this script IS the job
# payload); SEM legs are launched separately (see run_u3_sem.sh).
# NOTE: the synthesize_compare verdict here mixes the broken tish SH
# into mrt horizontals — ONLY mrr channels + the sem_compare results
# are meaningful (fast_methods_notes.md section 8).
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"
export ARB_ROOT=$Q/dsm_arbitration_u3
export ARB_MODELS=corefluid_q50
export ARB_SRC_DEPTH_KM=50
export ARB_REFINE_HMIN_KM=20
export ARB_REFINE_GRADE=0.5
export ARB_NEAR_TIER=1.5:10:2
export ARB_ELIM=1
bash dsm_arbitration/run_campaign.sh 24
echo "== U3 STAGE-1 (BEM+DSM) DONE $(date +%T) =="
