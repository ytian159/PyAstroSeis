#!/bin/bash
set -uo pipefail
A=/pscratch/sd/y/ytian159/astroseis_qtest/validation/dsm_audit/tipsv_ltrunc
R2=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_src50_ref2
run () { # name binary
  mkdir -p $A/run_mrt_$1/spc && cd $A/run_mrt_$1
  cp $R2/dsm/homog_q50/tipsv_mrt.inf .
  $A/$2 < tipsv_mrt.inf > tipsv.log 2>&1
  echo "tipsv mrt $1 done $(date +%T)"
}
run l256 tipsv_l256 &
run l1024 tipsv_l1024 &
run l4096 tipsv_l4096 &
run full tipsv_full &
wait
echo "TIPSV MRT LADDER DONE $(date +%T)"
