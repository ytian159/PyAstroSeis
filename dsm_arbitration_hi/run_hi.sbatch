#!/bin/bash
#SBATCH -A m4661
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 256
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 08:00:00
#SBATCH -J bem_dsm_hires
#SBATCH -o /pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_hi/slurm_%j.out
set -euo pipefail
export ARB_ROOT=/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_hi
export ARB_MESH_SCALE=4
S=$ARB_ROOT/scripts
DSM=/pscratch/sd/y/ytian159/dfdm_3d_perf/external/DSMsynTI-mpi
module load pytorch/2.8.0
ulimit -s unlimited
cd "$ARB_ROOT"
echo "== inputs $(date) =="
export OMP_NUM_THREADS=4
python "$S/make_inputs.py"
for model in homog twolayer; do
  cd "$ARB_ROOT/dsm/$model"
  for src in mrr mrt; do
    "$DSM/tipsv-mpi/tipsv" < "tipsv_$src.inf" > "tipsv_$src.log" 2>&1
    "$DSM/tish-mpi/tish"  < "tish_$src.inf"  > "tish_$src.log" 2>&1
  done
done
cd "$ARB_ROOT"
echo "== BEM homog $(date) =="
export OMP_NUM_THREADS=8
python "$S/run_bem.py" homog 16
echo "== BEM twolayer $(date) =="
python "$S/run_bem.py" twolayer 12
echo "== compare $(date) =="
python "$S/synthesize_compare.py"
echo "== HI-RES CAMPAIGN DONE $(date) =="
