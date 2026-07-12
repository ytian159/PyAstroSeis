#!/bin/bash
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
V=$Q/validation/specfem_u3
CASE=$V/sem32_check
cd "$F"
./configure --enable-double-precision FC=ftn CC=cc MPIFC=ftn MPICC=cc > configure2.log 2>&1 || { echo BUILD-FAILED configure; exit 1; }
make -j 16 xmeshfem3D xspecfem3D > build2.log 2>&1 || { echo BUILD-FAILED make; tail -20 build2.log; exit 1; }
echo "== rebuild ok $(date +%T) =="
module load pytorch/2.8.0
python3 $V/gen_case.py /pscratch/sd/y/ytian159/dfdm_automesh/gates/runs/semucb_sem/nex64/DATA \
  "$CASE" "$Q/dsm_arbitration_u3/manifest.json" mrt 32 2 100 2000
sed -i "s|^DT  *=.*|DT                              = 0.25d0|" "$CASE/DATA/Par_file"
cd "$CASE" && mkdir -p DATABASES_MPI OUTPUT_FILES
cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D .
SR="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 01:30:00"
echo "== check mesher $(date +%T) ==" && $SR --chdir="$CASE" ./xmeshfem3D || { echo CHECK-FAILED mesher; exit 1; }
echo "== check solver $(date +%T) ==" && $SR --chdir="$CASE" ./xspecfem3D || { echo CHECK-FAILED solver; exit 1; }
grep -i "USER_T0\|start time" OUTPUT_FILES/output_solver.txt | head -4
echo "== CHECK DONE $(date +%T) =="
