#!/bin/bash
# Fluid-fluid G5 + graded-mesh Earth-scale campaigns (run inside a
# compute allocation; ~2 h on 24 procs).
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"

echo "########## G5: fluid-fluid physics vs DSM ##########"
ARB_ROOT=$Q/dsm_arbitration_ff \
ARB_MODELS=homog_q50,corefluid_q50,ocsplit2_q50,ocstair2_q50 \
  bash dsm_arbitration/run_campaign.sh 24 \
  2>&1 | tee "$Q/validation/ff_campaign.txt" \
  || echo "CAMPAIGN-FAILED ff"

echo "########## graded-mesh transparency (637-km source) ##########"
ARB_ROOT=$Q/dsm_arbitration_ref637 ARB_MODELS=homog_q50 \
ARB_REFINE_HMIN_KM=45 ARB_NEAR_TIER=1.5:10:2 \
  bash dsm_arbitration/run_campaign.sh 24 \
  2>&1 | tee "$Q/validation/ref637_campaign.txt" \
  || echo "CAMPAIGN-FAILED ref637"

echo "########## 50-km source, graded mesh + near tier ##########"
ARB_ROOT=$Q/dsm_arbitration_src50_ref ARB_MODELS=homog_q50 \
ARB_SRC_DEPTH_KM=50 ARB_REFINE_HMIN_KM=45 ARB_NEAR_TIER=1.5:10:2 \
  bash dsm_arbitration/run_campaign.sh 24 \
  2>&1 | tee "$Q/validation/src50_ref_campaign.txt" \
  || echo "CAMPAIGN-FAILED src50_ref"

echo "########## ALL CAMPAIGNS DONE $(date +%T) ##########"
