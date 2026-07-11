#!/bin/bash
# Moment-fitted-RHS falsification campaigns (docs/moment_fitted_rhs.md
# gates 4 + 5), run inside a compute allocation: full battery, then
# BEM-only reruns with ARB_MFIT_LMAX=16 against the EXISTING DSM
# references of dsm_arbitration_ref637 (637-km propagating
# preservation) and dsm_arbitration_src50_ref2 (50-km quasi-static
# acceptance). ~30-40 min on 24 procs.
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
cd "$Q"
module load pytorch/2.8.0
export OMP_NUM_THREADS=4
ulimit -s unlimited

echo "########## battery + momentfit self-tests ##########"
for t in test_oracle test_oracle2 test_kernel_equiv test_meshgen \
         test_adaptive_quad test_rung0 test_rung1 test_rung2 \
         test_rung2b test_rung2c test_self_convergence test_elim \
         test_ff test_near_quad test_momentfit; do
  echo "=== $t ==="
  python "tests/$t.py" || echo "BATTERY-FAILED $t"
done

run_leg () {  # src_dir dst_dir
  local src=$1 dst=$2
  mkdir -p "$dst"
  cp "$src/manifest.json" "$dst/"
  cp "$src"/mesh_*.npz "$dst/"
  rm -rf "$dst/dsm"
  cp -r "$src/dsm" "$dst/"
  ARB_ROOT="$Q/$dst" ARB_MODELS=homog_q50 ARB_NEAR_TIER=1.5:10:2 \
  ARB_MFIT_LMAX=16 python dsm_arbitration/run_bem.py homog_q50 24 \
    || { echo "CAMPAIGN-FAILED $dst bem"; return 1; }
  ARB_ROOT="$Q/$dst" ARB_MODELS=homog_q50 \
    python dsm_arbitration/synthesize_compare.py \
    || echo "CAMPAIGN-FAILED $dst compare"
}

echo "########## gate 4: 637-km propagating preservation ##########"
run_leg dsm_arbitration_ref637 dsm_arbitration_ref637_mfit

echo "########## gate 5: 50-km quasi-static acceptance ##########"
run_leg dsm_arbitration_src50_ref2 dsm_arbitration_src50_mfit

echo "########## MFIT CAMPAIGNS DONE $(date +%T) ##########"
