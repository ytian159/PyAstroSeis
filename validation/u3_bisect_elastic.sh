#!/bin/bash
# A/B bisection arm A: sem32_check clone with ATTENUATION=.false.
# (elastic run => in-band velocity must be exactly the input 3.0 km/s;
# scored against DSM this should appear ~5.5% EARLY if everything
# outside the attenuation path is healthy).
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
V=$Q/validation/specfem_u3
CASE=$V/sem32_elastic
rm -rf "$CASE"
mkdir -p "$CASE/DATABASES_MPI" "$CASE/OUTPUT_FILES"
cp -r $V/sem32_check/DATA "$CASE/DATA"
sed -i "s|^ATTENUATION  *=.*|ATTENUATION                     = .false.|" \
  "$CASE/DATA/Par_file"
cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D "$CASE/"
SR="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 02:30:00"
echo "== elastic mesher $(date +%T) =="
$SR --chdir="$CASE" ./xmeshfem3D || { echo ELASTIC-FAILED mesher; exit 1; }
echo "== elastic solver $(date +%T) =="
$SR --chdir="$CASE" ./xspecfem3D || { echo ELASTIC-FAILED solver; exit 1; }
echo "== ELASTIC DONE $(date +%T) =="
