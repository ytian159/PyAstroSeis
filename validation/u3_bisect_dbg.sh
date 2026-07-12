#!/bin/bash
# A/B bisection arm B: short-record rerun of the attenuated case with
# the FORKDBG one-shot print of the realized scale-factor chain.
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
V=$Q/validation/specfem_u3
CASE=$V/sem32_dbg
rm -rf "$CASE"
mkdir -p "$CASE/DATABASES_MPI" "$CASE/OUTPUT_FILES"
cp -r $V/sem32_check/DATA "$CASE/DATA"
sed -i "s|^RECORD_LENGTH_IN_MINUTES  *=.*|RECORD_LENGTH_IN_MINUTES        = 5.0d0|" \
  "$CASE/DATA/Par_file"
cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D "$CASE/"
SR="srun -A m4661 -q interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=0 -t 01:00:00"
echo "== dbg mesher $(date +%T) =="
$SR --chdir="$CASE" ./xmeshfem3D || { echo DBG-FAILED mesher; exit 1; }
echo "== dbg solver $(date +%T) =="
$SR --chdir="$CASE" ./xspecfem3D || { echo DBG-FAILED solver; exit 1; }
echo "== DBG DONE $(date +%T) =="
