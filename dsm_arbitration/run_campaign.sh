#!/bin/bash
# DSM arbitration campaign (run inside a compute allocation).
# Models via ARB_MODELS (comma list, first must be the homog
# baseline); scripts stay in dsm_arbitration/, outputs go to
# ARB_ROOT.
set -euo pipefail
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"
ROOT="${ARB_ROOT:-$SCRIPTS}"
DSM=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi
NPROCS="${1:-20}"
MODELS="${ARB_MODELS:-homog,twolayer}"

module load pytorch/2.8.0
export OMP_NUM_THREADS=4
ulimit -s unlimited

mkdir -p "$ROOT"
cd "$ROOT"
echo "== stage meshes + DSM inputs $(date +%T) =="
python "$SCRIPTS/make_inputs.py"

echo "== DSM runs $(date +%T) =="
for model in ${MODELS//,/ }; do
  cd "$ROOT/dsm/$model"
  for src in mrr mrt; do
    echo "-- $model tipsv $src $(date +%T)"
    "$DSM/tipsv-mpi/tipsv" < "tipsv_$src.inf" > "tipsv_$src.log" 2>&1
    echo "-- $model tish  $src $(date +%T)"
    "$DSM/tish-mpi/tish" < "tish_$src.inf" > "tish_$src.log" 2>&1
  done
done
cd "$ROOT"

for model in ${MODELS//,/ }; do
  echo "== BEM leg: $model $(date +%T) =="
  python "$SCRIPTS/run_bem.py" "$model" "$NPROCS"
done

echo "== synthesis + comparison $(date +%T) =="
python "$SCRIPTS/synthesize_compare.py"
echo "== CAMPAIGN DONE $(date +%T) =="
