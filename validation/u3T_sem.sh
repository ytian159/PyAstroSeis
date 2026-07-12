#!/bin/bash
# SEM NEX32 viscous run at the SH-pure meridian stations (cross-check
# of the BEM SH-deficit verdict against the mode-validated SEM toroidal)
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
V=$Q/validation/specfem_u3
CASE=$V/sem32_meridian
rm -rf "$CASE"
mkdir -p "$CASE/DATABASES_MPI" "$CASE/OUTPUT_FILES"
cp -r $V/sem32_check/DATA "$CASE/DATA"
python3 - <<'PY'
import json
man = json.load(open("/pscratch/sd/y/ytian159/astroseis_qtest/dsm_arbitration_u3T/manifest.json"))
with open("/pscratch/sd/y/ytian159/astroseis_qtest/validation/specfem_u3/sem32_meridian/DATA/STATIONS", "w") as f:
    for st in man["stations"]:
        f.write("%-6s DF %12.6f %12.6f 0.0 0.0\n" % (st["name"], st["lat"], st["lon"]))
print("STATIONS written")
PY
cp $Q/external/specfem3d_globe_u3/bin/xmeshfem3D $Q/external/specfem3d_globe_u3/bin/xspecfem3D "$CASE/"
SR="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 02:30:00"
echo "== meridian mesher $(date +%T) =="
$SR --chdir="$CASE" ./xmeshfem3D || { echo MERID-FAILED mesher; exit 1; }
echo "== meridian solver $(date +%T) =="
$SR --chdir="$CASE" ./xspecfem3D || { echo MERID-FAILED solver; exit 1; }
echo "== MERIDIAN DONE $(date +%T) =="
