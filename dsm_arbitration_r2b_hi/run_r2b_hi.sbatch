#!/bin/bash
#SBATCH -A m4661
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 06:00:00
#SBATCH -J bem_r2b_hires
#SBATCH -o /pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_r2b_hi/slurm_%j.out
# Rung-2b decisive convergence run: corefluid3 at mesh scale 2
# (reflector h/R 0.045). Prediction: trapped-mode bias ~0.2%
# (order-1.9 in h/R from the 0T2 control), misfit -> scale-2 floor,
# min vector corr > 0.9. Non-convergence = back to bug hunting.
set -euo pipefail
export ARB_ROOT=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_r2b_hi
export ARB_MESH_SCALE=2
export ARB_MODELS=homog,corefluid3
S=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration
DSM=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi
module load pytorch/2.8.0
ulimit -s unlimited
export OMP_NUM_THREADS=4
mkdir -p "$ARB_ROOT"
cd "$ARB_ROOT"
echo "== inputs $(date) =="
python "$S/make_inputs.py"
echo "== DSM $(date) =="
for model in homog corefluid3; do
  cd "$ARB_ROOT/dsm/$model"
  for src in mrr mrt; do
    "$DSM/tipsv-mpi/tipsv" < "tipsv_$src.inf" > "tipsv_$src.log" 2>&1
    "$DSM/tish-mpi/tish"  < "tish_$src.inf"  > "tish_$src.log" 2>&1
  done
done
cd "$ARB_ROOT"
echo "== BEM homog $(date) =="
python "$S/run_bem.py" homog 24
echo "== BEM corefluid3 $(date) =="
python "$S/run_bem.py" corefluid3 12
echo "== score $(date) =="
python "$S/synthesize_compare.py"
echo "== DONE $(date) =="
