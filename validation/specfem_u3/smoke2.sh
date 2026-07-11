#!/bin/bash
# uniform3_q50 NEX64 smoke, driven from login: case gen (instant),
# then mesher and solver as separate top-level shared_interactive
# sruns (nested job steps hit a port limit on shared nodes).
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
CASE=$Q/validation/specfem_u3/smoke_nex64
TMPL=/pscratch/sd/y/ytian159/dfdm_automesh/gates/runs/semucb_sem/nex64/DATA
SRUN="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 00:40:00"

module load pytorch/2.8.0
python3 $Q/validation/specfem_u3/gen_case.py "$TMPL" "$CASE" \
  "$Q/dsm_arbitration_src50_ref2/manifest.json" mrt 64 2 10 0.0
cd "$CASE"
mkdir -p DATABASES_MPI OUTPUT_FILES
cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D .

echo "== mesher $(date +%T) =="
$SRUN --chdir="$CASE" ./xmeshfem3D || { echo SMOKE-FAILED mesher; exit 1; }
echo "--- attenuation band + model lines from mesher log:"
grep -i "attenuation period\|min_attenuation\|max_attenuation\|minimum period" \
  OUTPUT_FILES/output_mesher.txt | head -6
grep -i "model:" OUTPUT_FILES/output_mesher.txt | head -6

echo "== solver $(date +%T) =="
$SRUN --chdir="$CASE" ./xspecfem3D || { echo SMOKE-FAILED solver; exit 1; }
ls OUTPUT_FILES/*.sem.ascii 2>/dev/null | head -4
echo "== SMOKE DONE $(date +%T) =="
