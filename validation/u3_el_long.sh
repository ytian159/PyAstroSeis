#!/bin/bash
# Long-record elastic NEX32 run for absolute SEM mode-frequency
# fitting (l=2..5 spheroidal fundamentals vs exact_modes table).
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
SR="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 03:55:00"
echo "== el_long mesher $(date +%T) =="
$SR --chdir="$CASE" ./xmeshfem3D || { echo EL-LONG-FAILED mesher; exit 1; }
echo "== el_long solver $(date +%T) =="
$SR --chdir="$CASE" ./xspecfem3D || { echo EL-LONG-FAILED solver; exit 1; }
echo "== EL-LONG DONE $(date +%T) =="
