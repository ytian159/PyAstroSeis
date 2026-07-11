#!/bin/bash
# SPECFEM uniform3 legs, driven from login (each mesher/solver is a
# top-level srun): NEX32 mrt (shared lane) + NEX64 mrt,mrr (exclusive
# interactive), erf STF hdur 2000 s, 450-min records, then velocity
# comparisons vs the BEM leg + SEM32-vs-SEM64 self-agreement.
set -uo pipefail
Q=/pscratch/sd/y/ytian159/astroseis_qtest
F=$Q/external/specfem3d_globe_u3
V=$Q/validation/specfem_u3
ROOT=$Q/dsm_arbitration_u3
TMPL=/pscratch/sd/y/ytian159/dfdm_automesh/gates/runs/semucb_sem/nex64/DATA
module load pytorch/2.8.0

run_case () {  # name nex nproc src dt SRUNPREFIX
  local name=$1 nex=$2 nproc=$3 src=$4 dt=$5; shift 5
  local CASE=$V/$name
  python3 $V/gen_case.py "$TMPL" "$CASE" "$ROOT/manifest.json" \
    "$src" "$nex" "$nproc" 450 2000
  sed -i "s|^DT  *=.*|DT                              = ${dt}d0|" \
    "$CASE/DATA/Par_file" 2>/dev/null || true
  cd "$CASE"
  mkdir -p DATABASES_MPI OUTPUT_FILES
  cp "$F"/bin/xmeshfem3D "$F"/bin/xspecfem3D .
  echo "== $name mesher $(date +%T) =="
  "$@" --chdir="$CASE" ./xmeshfem3D \
    || { echo U3-SEM-FAILED $name mesher; return 1; }
  echo "== $name solver $(date +%T) =="
  "$@" --chdir="$CASE" ./xspecfem3D \
    || { echo U3-SEM-FAILED $name solver; return 1; }
  echo "== $name done $(date +%T) =="
}

SR_SHARED="srun -A m4661 -q shared_interactive -C cpu -N 1 -n 24 -c 2 --cpu-bind=cores --mem=55G -t 03:30:00"
SR_EXCL="srun -A m4661 -q interactive -C cpu -N 1 -n 96 -c 2 --cpu-bind=cores --mem=0 -t 03:30:00"

run_case sem32_mrt 32 2 mrt 0.25 $SR_SHARED &
run_case sem64_mrt 64 4 mrt 0.14 $SR_EXCL &
wait
run_case sem64_mrr 64 4 mrr 0.14 $SR_EXCL

echo "== velocity comparisons =="
cd "$Q"
for pair in "sem64_mrt mrt" "sem64_mrr mrr" "sem32_mrt mrt"; do
  set -- $pair
  python3 dsm_arbitration/sem_compare.py "$ROOT" "$V/$1" \
    bem_corefluid_q50.npz "$2" 2000 \
    || echo U3-COMPARE-FAILED "$1"
done
echo "== U3 SEM STAGE DONE $(date +%T) =="
