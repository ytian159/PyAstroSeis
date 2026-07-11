#!/bin/bash
set -uo pipefail
A=/pscratch/sd/y/ytian159/astroseis_qtest/validation/dsm_audit/tish_ltrunc
R2=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_src50_ref2
run () { # name re binary
  mkdir -p $A/run_$1/spc && cd $A/run_$1
  sed "3s/.*/  $2/" $R2/dsm/homog_q50/tish_mrt.inf > tish_mrt.inf
  $A/$3 < tish_mrt.inf > tish.log 2>&1
  echo "tish $1 done $(date +%T)"
}
run gre4_l1024 1.0e-04 tish_g8m_l1024 &
run gre4_l4096 1.0e-04 tish_g8m_l4096 &
run gre5_l1024 1.0e-05 tish_g8m_l1024 &
run gre5_l4096 1.0e-05 tish_g8m_l4096 &
run gre4_full 1.0e-04 tish_g8m &
run gre5_full 1.0e-05 tish_g8m &
wait
echo "TISH FINE-GRID RUNS DONE $(date +%T)"
