# pyastroseis — Python port of AstroSeis

Standalone NumPy port of the AstroSeis boundary-element solver
(Tian & Zheng, https://arxiv.org/abs/2004.07410). Full feature parity
with the MATLAB code — no MATLAB required at runtime:

- homogeneous body, **single-force** and **moment-tensor** sources
  (`solver.run_case`)
- solid body with a **liquid core**, incl. pressure sources in the
  core (`liquidcore.run_case_lc`)
- **mesh generation**: Riesz-energy quasi-uniform sphere sampling,
  spherical quadrisection, Phobos spherical-harmonic shape (Willner et
  al. 2014, `data/TABLEA1.DAT`) and Mars-like pseudo-random topography
  (`meshgen`)

## Usage

```bash
module load pytorch/2.8.0      # or cray-python with PYTHONNOUSERSITE=1
python run_astroseis_py.py BEM_para_Q570 --nprocs 45          # legacy format
python run_astroseis_py.py examples/config_Q570.yml --nprocs 45 --plot
python run_astroseis_py.py inp_sft_lc --liquidcore --nprocs 44
pip install -e .               # optional: installable package
```

Parameter files: legacy fixed-line AstroSeis format or YAML (see
examples/config_*.yml). Meshes: `.mat` (face struct or V/Tri) or
Wavefront `.obj` shape models (`mesh.load_shape_obj`, e.g. published
asteroid shapes). Frequencies are independent, so `--nprocs`
parallelizes over them without changing any numbers. Output goes to
`pyout/<name from the parameter file>`; `--plot` writes a
record-section PNG (`pyastroseis.plotting`).

## Phase-2 numerics (defaults differ from MATLAB!)

Two switches control MATLAB equivalence:

- `--qp-mode physical|legacy` — see Qp section below.
- `--self-scheme polar|grid` — singular self-integration. "grid" is the
  MATLAB 300x300 punch-out scheme, bit-compatible. "polar" (default)
  integrates the same CPV-excluded annulus in polar coordinates
  (`assembly.polar_self_points`): the 1/r^2 kernel is regular in
  (rho, theta), so ~576 points beat the 45,000-point grid in both
  accuracy and cost (tests/test_self_convergence.py quantifies this
  against an nxi=1501 converged reference).

`--qp-mode legacy --self-scheme grid` reproduces original MATLAB
outputs to ~1e-14.

The traction kernel is evaluated in a canonical radial-function form
(~3x fewer array operations than the Mathematica-generated
transcription; `greens.green_traction_tensor`, equivalence enforced by
tests/test_kernel_equiv.py), and all frequency-independent geometry
(quadrature sub-grids, self-integration points) is built once per run
(`assembly.Geometry`) and shared with the frequency workers.

## Qp convention (differs from MATLAB by default!)

The original code hardcoded Qp = 2.5·Qs in the BEM kernels and
Qp = 2.25·Qs in the single-force source term — mutually inconsistent,
and both wrong for vp/vs ≠ √3. The default here is the community
standard (Aki & Richards; PREM/SPECFEM with Qκ→∞):

    Qp = (3/4) (vp/vs)^2 · Qs        (solid layers)
    Qp = Q                            (fluid core: Q is Qp directly)

Use `--qp-mode legacy` to reproduce original MATLAB outputs exactly
(this is what all validation comparisons use).

## Validation

1. `tests/dump_oracle.m` + `tests/dump_oracle2.m` (matlab -batch, one
   time) dump kernels, quadrature, self integrals, interaction
   matrices, small coupled systems, spherical harmonics, topography,
   and subdivision fixtures.
2. `python tests/test_oracle.py` / `test_oracle2.py` check every ported
   routine against the dumps (machine precision, 1e-12..1e-16).
3. `python compare_py_matlab.py` — full homogeneous runs vs
   `out_Q*.mat` (legacy mode, max rel err ~5e-14) with record-section
   overlay + magnified difference figures in `validation/`.
4. `python compare_lc.py` — full liquid-core run vs
   `examples/shifted_liquid_core/uu_liqcore_bem20km_shift.mat`.
5. `python tests/test_meshgen.py` — geometric/topological checks for
   mesh generation (the particle optimizer is stochastic, so this is
   functional, not bitwise; topography and subdivision ARE bitwise
   against the oracle).

Any change to the numerics must keep 1–5 passing, or consciously
re-baseline with a documented reason.

## Known quirks replicated on purpose

- `mu2 = rho1*vs2^2` in the liquid-core driver (rho1, not rho2;
  harmless for a true fluid where vs2=0).
- `gen_mesh_ph_topo` hardcodes r0 = 10.993 km and the TABLEA1 l=0 term
  adds another mean radius → body radius ≈ 22 km regardless of the R
  parameter (`radius_mode="parameter"` gives the sane scaling).
- `smooth_topo_rand.m` discards its analysis coefficients and uses
  MATLAB `rng(4)` random values; those are shipped verbatim in
  `data/coefrand_rng4.txt`.
- The 300x300 brute-force self-integration grid and per-frequency
  geometry rebuild (`PHASE2` comments mark the optimization targets).
