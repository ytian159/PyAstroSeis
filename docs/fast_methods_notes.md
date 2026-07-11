# Scaling, memory, and fast-method assessment (2026-07-10)

Context for the "50-km source / memory / FMM / higher frequency"
decisions. Measured numbers from the campaign logs cited inline;
survey citations at the end. Companion: docs/fluid_fluid_derivation.md.

## 1. Where memory actually stands (measured)

Elimination already removed the monolithic dense wall. 16 B per
complex128; "peak" = 3-4 largest interface-group blocks held by the
Thomas sweep (elimination.py):

| model     | total N | dense N^2   | largest group m | elim peak (~4 blocks) |
|-----------|---------|-------------|-----------------|------------------------|
| homog_q50 | 4,752   | 0.36 GB     | 4,752           | (dense used, 1 group)  |
| gprem_s8  | 31,872  | 16.3 GB     | 5,232           | ~1.6 GiB               |
| prem_s8   | 37,760  | 22.8 GB     | 6,336 (CMB 4n)  | ~2.4 GiB               |

(dsm_arbitration_gprem/, dsm_arbitration_prem/ manifests + logs.)
Layer scaling is LINEAR: 138-harmonic legs on 24 procs ran 95 s
(1 layer) / 497 s (4) / 592 s (6) / 1506 s (10). The wall is no
longer layer count — it is the size of the largest single interface.

## 2. Frequency scaling (fixed elements/wavelength, n ∝ f²)

Surface interface of gprem_s8 (n0 = 1584 panels, band top
5.2643e-4 Hz), largest block m = 3n, LU ~ (8/3)m³ real flops:

| band | n_surf   | m = 3n  | block memory | block LU flops |
|------|----------|---------|--------------|----------------|
| ×1   | 1,584    | 4.8k    | 0.34 GiB     | 2.9e11         |
| ×2   | 6,336    | 19k     | 5.4 GiB      | 1.8e13         |
| ×5   | 39,600   | 119k    | 210 GiB      | 4.5e15         |
| ×10  | 158,400  | 475k    | 3.3 TiB      | 2.9e17         |

So: ×2 is comfortably dense-feasible per block (with elimination);
×5 is the cliff where per-interface compression becomes mandatory;
×10 is impossible without it. Assembly cost scales as f⁴ per
interface pair; kernels (Numba/CUDA) buy the constant only.

Regime check: even at ×10 the per-region kR is ~10-100 — the LOW-to-
mid frequency regime where rank-structured (H/H²/HBS) methods remain
efficient (Chaillat-Desiderio-Ciarlet 2017 quantified H-LU for
Helmholtz + elastodynamic kernels exactly in this band). Directional
FMM / diagonal-form translation is machinery for a much higher band.

## 3. 50-km (shallow) sources — graded mesh, not FMM

Measured limit: incident/scattered field near a source at depth d
needs panel size h ≲ d at the epicenter (d/h = 0.09-0.18 failed with
an acausal burst + missing wavefield, dsm_arbitration_src50*;
d/h = 1.12 validated). Global h = 50 km ⇒ ~200k surface panels —
hopeless. But GRADED refinement with h(r) ∝ max(r, d) from the
epicenter gives LOGARITHMIC panel count: N_cap ~ 60·ln(R/d) ≈ a few
hundred to ~2k extra panels. Feasibility audit of the assembly
(2026-07-10 audit): piecewise-constant collocation at incenters, all
self-terms and adaptive-quadrature tiers are PER-FACE (inradius,
max-side h, k·h caps) — nonconforming/graded triangle soups are legal
inputs (`faces_from_vertices` accepts any V/Tri).

ONE real gap: near-singular cross-panel quadrature. Off-diagonal
pairs use a fixed degree-10 rule; the closest tier fires at
dist < 4·h_source and there is NO promotion path when a small panel's
incenter sits very close to a LARGE neighbor panel (1/r²-type kernels
go near-singular). A graded-mesh rung therefore needs: (a) a
size-jump-bounded grading (neighbor ratio ≤ ~2, standard), and
(b) either a subdivision fallback for near-singular pairs
(dist < c·h_target) or an extra high-order tier. Small, testable
change; gate = refined-vs-uniform on the validated 637-km source
before trusting any 50-km run.

Mesh generator status: meshgen.py (Riesz-energy + one quadrisection)
is quasi-uniform with NO grading hook — the graded cap mesh needs a
new generator path (e.g. subdivide the cap triangles of an existing
mesh k times toward the epicenter, or an external graded sphere
mesh loaded via faces_from_vertices).

## 4. Method ladder (recommendation)

1. NOW (this band, 50-km goal): graded epicentral mesh + near-
   singular quadrature promotion (§3). No new solver machinery.
2. NOW/CHEAP (memory, pure numpy, keeps the direct sweep): compress
   the INTERFACE-COUPLING blocks — adjacent closed surfaces are
   separated by the layer thickness, so those blocks are globally
   low-rank; ACA or scipy.linalg.interpolative ID over the existing
   kernel entries, Woodbury updates inside the Thomas sweep, cached
   dense LU of unperturbed diagonal blocks. Order-of-magnitude sweep
   memory/flops cut in a few hundred lines.
3. BEFORE any fast method (accuracy-per-dof): the measured
   ~6%/interface error at h/R 0.09-0.2 is GEOMETRY-dominated (flat
   panels on curved interfaces, order 2). Curved (quadratic) panels
   with the same P0 unknowns — same N, same solver — kill the
   dominant error; higher-order Nyström/QBX cuts N by 10-100× at
   fixed error, absorbing a 2-3× band push with NO fast method.
4. END-GAME ×5-10:
   * Near-spherical Earth/planet models: spectral — spherical-
     harmonic (block-)diagonal interface operators + perturbative
     relief (transformed field expansions, Nicholls-Reitich). This is
     the BEM twin of DSM, preserves the block-tridiagonal sweep in
     (l,m) space, makes ensembles nearly free, and beats FMM outright
     for this geometry class. ("Better than FMM": yes, for spheres.)
   * Arbitrary asteroid shapes (the code's raison d'être): H-LU /
     HBS (recursive skeletonization, FMM-LU-class) factorization of
     each diagonal block inside the SAME sweep — factor once, solve
     many; ~100× better than FMM+GMRES for many-RHS ensembles
     (Ho-Greengard 2012). Libraries: HLIBpro (free academic binary),
     H2Lib (open C), or DIY HBS on top of ACA.
   * FMM only as the iterative arm if a case outgrows fast-direct:
     FMM3D (Apache-2.0, Python, wideband Helmholtz, vectorized
     multi-density) for fluid blocks; elastic matvec via the
     Fujiwara/Chaillat 4-Helmholtz-potential decomposition; GMRES
     right-preconditioned by a frozen low-accuracy compressed sweep
     (the cached unperturbed factorization doubles as the ensemble
     preconditioner). Avoid pFFT/AIM (volume grid wastes a dimension
     on closed surfaces) and directional FMM (wrong regime).

Why not jump straight to FMM: an FMM gives a matvec only, forcing
global GMRES — per-RHS cost multiplies by iteration count, which is
structurally wrong for our ensemble/many-source economy, and at our
kR the rank-structured direct routes are both feasible and a better
fit for the existing elimination architecture.

## 5. Key citations

Wideband Helmholtz FMM: Cheng et al., JCP 216 (2006). Elastodynamic
FM-BEM: Fujiwara, GJI 140 (2000); Chaillat-Bonnet-Semblat, CMAME
(2008); COFFEE (ENSTA, non-commercial). H-matrices for oscillatory
kernels: Chaillat-Desiderio-Ciarlet, JCP 351 (2017), arXiv:1706.09384
(+ Adv. Comput. Math. 2021 for complex wavenumber). Fast direct:
Martinsson-Rokhlin JCP 2005; Ho-Greengard SISC 2012; FMM-LU,
Sushnikova et al., arXiv:2201.07325. Spectral relief: Nicholls-
Reitich TFE, JCP 2007; DSM lineage Geller-Ohminato GJI 116 (1994).
Preconditioners: Darbas et al. JCP 2013 (OSRC, Helmholtz);
Chaillat-Darbas-Le Louër JCP 341 (2017) (elastodynamics).
Libraries: FMM3D github.com/flatironinstitute/FMM3D (Apache-2.0);
fmm3dbie github.com/fastalgorithms/fmm3dbie; exafmm-t (BSD-3);
PVFMM (LGPLv3); Bempp-cl (MIT, no elastic kernels); h2tools; H2Lib;
HLIBpro (binary, academic-free); H2Opus (GPU H²).
