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
2. MEMORY (pure numpy, keeps the direct sweep) — CORRECTED after the
   Codex round-2 review (2026-07-11): the original claim that
   interface-coupling blocks are GLOBALLY low-rank is wrong for THIN
   layers — quasi-static transfer between concentric surfaces decays
   as (1-d/R)^l, so l_max ~ (R/d) log(1/eps) and rank ~ l_max²: a
   thin PREM staircase shell has near-full-rank coupling. Compression
   must be hierarchical (admissible, well-separated PATCH pairs, ACA
   per pair; near pairs stay dense — standard H-matrix clustering),
   and diagonal self-blocks need their own treatment. First step
   before writing any code: a RANK AUDIT of the actual campaign
   blocks (SVD spectra of coupling blocks vs layer thickness,
   2-3 person-days). Woodbury-in-Thomas only pays when model CHANGES
   are genuinely low-rank; cached factors already serve many-RHS.
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

## 6. 50-km-source rung: measured closure (2026-07-11)

The graded mesh + near tier RECOVERED THE PROPAGATING FIELD (nearest
station corr 0.99 vs acausal garbage on the uniform mesh; 637-km
transparency leg improved to 11.1% vs the 12.9% uniform baseline).
What remains is NOT a mesh problem: at this ULP band a 50-km-deep
source's reference (DSM) field is dominated by the QUASI-STATIC
response — measured group delay ~0-3 s at ALL distances on mrt's T
component (vs 1600-6200 s propagating moveout for the 637-km source),
amplitude 10-100x the propagating part. The BEM delivers ~2% of it.

Root cause (proven): the quasi-static channel is driven by a
near-perfectly-cancelling projection of the incident trace. The
l=2,m=1 toroidal projection of the SAMPLED trace wanders
2.2e6 -> 2.1e5 -> 1.3e4 -> -2.0e5 (sign flip) as the epicentral cap
refines hmin 45 -> 5 km — always ~1e-4 of the gross integral — and
the solved mrt amplitude ratio FALLS with refinement (0.022 at
hmin 45, 0.0075 at hmin 20; mrr metrics bit-stable). Pointwise
incident-trace collocation cannot deliver delicately-cancelling
quasi-static projections at any practical refinement. The same
mechanism at partial severity explains the mrr misfit pattern
(station-wise error tracks the DSM static-content fraction).
This UNIFIES with the interior-source radiation deficit
(w*R_region/vs <~ 1): both are incident-source-representation
failures of the same class. Campaign dirs: dsm_arbitration_src50_ref
(hmin 45), dsm_arbitration_src50_ref2 (hmin 20, grade 0.5).

## 7. Roadmap after Codex round 2 (2026-07-11)

Priority for full-PREM ULP Earth with shallow sources: C -> A -> B'.
For asteroid relief: C -> B' -> A. For ensembles: reciprocity first.

* C = scattered-field source formulation (1-2 wk prototype): solve
  u_scat with RHS = +G t_inc (weakly-singular INTEGRAL of the
  analytic incident traction; free surface t_scat = -t_inc).
  Codex verdict: the physical cancellation remains but moves inside
  a CONTROLLED weak integral — with cancellation condition ~1e4 and
  panel-integral accuracy ~1e-6 the modal residue lands at ~1%,
  adequate. Conditions: (i) evaluate analytic t_inc at QUADRATURE
  NODES (centroid sampling recreates the disease); (ii) quadrature
  promotion must key on distance to the SOURCE too, not only panel
  separation; (iii) acceptance test = convergence of the modal
  forcing f_z against the reciprocal-work reference
  f_z = M : eps_z(x_src) (local, cancellation-free). Interior
  sources additionally need the full Cauchy-data jumps (u_inc AND
  t_inc) on every source-region boundary.
* A = spectral spherical-harmonic route (8-12 wk MVP, 16-24 wk
  production + 8-16 wk relief): the production Earth destination;
  also cures the quasi-static channel (per-(l,m) source vectors by
  smooth quadrature, as in DSM). Risks: relief couples |l-l'| <=
  p*L_h and shallow sources need l ~ R/d; complex Bessel stability at
  low ka / high l (use scaled/impedance recurrences + static limits);
  fluid-solid in SH basis is signs-and-limits work, not feasibility.
  Anti-circularity anchors (BEM-spectral vs DSM share structure):
  homogeneous/static closed forms, reciprocity, passivity,
  manufactured modal traces, triangular-BEM cross-checks, source
  coefficients via M : eps_lm(x_src).
* B' = hierarchical (patch-pair) compression -- see corrected item 2
  above; start with the rank audit.
* Cheapest falsification experiment first (0.5-2 wk): MOMENT-FITTED
  RHS — compute the exact low-(l,m) source coefficients by
  reciprocal work and minimally correct the discrete RHS so those
  moments are exact. Directly tests the diagnosis and may rescue
  shallow-source ULP runs without the full C rung. For fixed
  receivers x many sources, RECIPROCITY (adjoint receiver fields,
  evaluate M : eps_adj(x_src)) avoids source-trace sampling entirely.
