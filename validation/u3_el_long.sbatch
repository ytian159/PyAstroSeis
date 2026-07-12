#!/bin/bash
#SBATCH -A m4661
#SBATCH -q shared
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -n 24
#SBATCH -c 2
#SBATCH --mem=55G
#SBATCH -t 04:30:00
#SBATCH -J u3_el_long
#SBATCH -o /pscratch/sd/y/ytian159/astroseis_qtest/validation/u3_el_long_sbatch.log
# qos-justified: 3.6-h long-record elastic run must survive the session; shared sbatch per house rule
# FALLBACK for the orphaned srun of 2026-07-11 (job 55805698); submit only
# if that run died before writing OUTPUT_FILES seismograms.
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
V=$Q/validation/specfem_u3
CASE=$V/sem32_el_long
rm -rf "$CASE"
mkdir -p "$CASE/DATABASES_MPI" "$CASE/OUTPUT_FILES"
cp -r $V/sem32_elastic/DATA "$CASE/DATA"
sed -i "s|^RECORD_LENGTH_IN_MINUTES  *=.*|RECORD_LENGTH_IN_MINUTES        = 330.0d0|" \
  "$CASE/DATA/Par_file"
cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D "$CASE/"
cd "$CASE"
echo "== el_long mesher $(date +%T) =="
srun -n 24 -c 2 --cpu-bind=cores ./xmeshfem3D || { echo EL-LONG-FAILED mesher; exit 1; }
echo "== el_long solver $(date +%T) =="
srun -n 24 -c 2 --cpu-bind=cores ./xspecfem3D || { echo EL-LONG-FAILED solver; exit 1; }
echo "== EL-LONG DONE $(date +%T) =="
