#!/bin/bash
# Rung-1 DSM arbitration campaign (run inside a compute allocation).
set -euo pipefail
ROOT="${ARB_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DSM=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi
NPROCS="${1:-20}"

module load pytorch/2.8.0
export OMP_NUM_THREADS=4
ulimit -s unlimited

cd "$ROOT"
echo "== stage meshes + DSM inputs $(date +%T) =="
python make_inputs.py

echo "== DSM runs $(date +%T) =="
for model in homog twolayer; do
  cd "$ROOT/dsm/$model"
  for src in mrr mrt; do
    echo "-- $model tipsv $src $(date +%T)"
    "$DSM/tipsv-mpi/tipsv" < "tipsv_$src.inf" > "tipsv_$src.log" 2>&1
    echo "-- $model tish  $src $(date +%T)"
    "$DSM/tish-mpi/tish" < "tish_$src.inf" > "tish_$src.log" 2>&1
  done
done
cd "$ROOT"

echo "== BEM homogeneous leg $(date +%T) =="
python run_bem.py homog "$NPROCS"
echo "== BEM two-layer (welded) leg $(date +%T) =="
python run_bem.py twolayer "$NPROCS"

echo "== synthesis + comparison $(date +%T) =="
python synthesize_compare.py
echo "== CAMPAIGN DONE $(date +%T) =="
