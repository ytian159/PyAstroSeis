#!/bin/bash
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
CASE=$Q/validation/specfem_u3/sem32_check
cp $Q/external/specfem3d_globe_u3/bin/xmeshfem3D $Q/external/specfem3d_globe_u3/bin/xspecfem3D "$CASE/"
SR="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 01:30:00"
echo "== mesher $(date +%T) ==" && $SR --chdir="$CASE" "$CASE/xmeshfem3D" || { echo CHECK-FAILED mesher; exit 1; }
echo "== solver $(date +%T) ==" && $SR --chdir="$CASE" "$CASE/xspecfem3D" || { echo CHECK-FAILED solver; exit 1; }
grep -i "USER_T0" "$CASE/OUTPUT_FILES/output_solver.txt" | head -2
echo "== DP CHECK DONE $(date +%T) =="
