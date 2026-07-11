#!/bin/bash
# DSM shallow-source noise audit: tipsv l-truncation ladder (mrr) +
# tish fine-grid (re) probes (mrt), all single-core runs in parallel.
set -uo pipefail
SP=/pscratch/sd/y/ytian159/astroseis_qtest/validation/dsm_audit
D=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi
R2=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_src50_ref2

# ---- build tipsv variants ----
mkdir -p $SP/tipsv_ltrunc && cd $SP/tipsv_ltrunc
cp $D/tipsv-mpi/{calmat,glu2,rk3,solver,others,operation,tipsv}.f90 .
cp $D/common/{fileio,utils,trialf}.f90 .
gfortran -O2 -fdec -c fileio.f90
gfortran -O2 -c utils.f90 trialf.f90 others.f90 calmat.f90 glu2.f90 rk3.f90 solver.f90 operation.f90
for L in 64 256 1024 4096; do
  sed "s/maxL = 80000/maxL = $L/" tipsv.f90 > tipsv_l$L.f90
  gfortran -O2 -o tipsv_l$L tipsv_l$L.f90 *.o
done
gfortran -O2 -o tipsv_full tipsv.f90 *.o
echo "tipsv builds done"

run_tipsv () {  # name binary
  local n=$1 b=$2
  mkdir -p $SP/tipsv_ltrunc/run_$n/spc && cd $SP/tipsv_ltrunc/run_$n
  cp $R2/dsm/homog_q50/tipsv_mrr.inf .
  ../$b < tipsv_mrr.inf > tipsv.log 2>&1
  echo "tipsv $n done $(date +%T)"
}
run_tish_re () {  # name re binary
  local n=$1 re=$2 b=$3
  mkdir -p $SP/tish_ltrunc/run_$n/spc && cd $SP/tish_ltrunc/run_$n
  sed "3s/.*/  $re/" $R2/dsm/homog_q50/tish_mrt.inf > tish_mrt.inf
  $SP/tish_ltrunc/$b < tish_mrt.inf > tish.log 2>&1
  echo "tish $n done $(date +%T)"
}

run_tipsv full tipsv_full &
run_tipsv l64 tipsv_l64 &
run_tipsv l256 tipsv_l256 &
run_tipsv l1024 tipsv_l1024 &
run_tipsv l4096 tipsv_l4096 &
run_tish_re re4_full 1.0e-04 tish_full &
run_tish_re re4_l1024 1.0e-04 tish_l1024 &
run_tish_re re4_l4096 1.0e-04 tish_l4096 &
run_tish_re re5_full 1.0e-05 tish_full &
run_tish_re re5_l1024 1.0e-05 tish_l1024 &
run_tish_re re5_l4096 1.0e-05 tish_l4096 &
wait
echo "DSM AUDIT RUNS DONE $(date +%T)"
