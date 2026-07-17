# Rung A: spectral spherical-harmonic sweep (design, 2026-07-16)

Decision context: rung C proved the shallow-source quasi-static
deficit is a P0-BEM dynamic-range limitation (fast_methods_notes §9);
policy C (propagating-band-only BEM verdicts, analytic references for
statics) is ADOPTED as the standing policy, and rung A is now the
build target: the per-(l,m) spectral route, which has O(1) dynamic
range per harmonic and therefore recovers shallow-source statics by
construction. This is the production Earth destination of the
roadmap (§7) and the "BEM twin of DSM" (§4, end-game item).

## 1. Scope of the MVP

A production module `pyastroseis/spectral.py`:

* model class: arbitrary concentric stack of UNIFORM layers, each
  solid (rho, vp, vs) or inviscid fluid (rho, vp), innermost layer a
  solid ball (our models; fluid ball refused for now), no gravity
  (matches every campaign leg);
* source: interior moment tensor at r0 in a SOLID layer, Mr*-type
  (m in {-1, 0, +1} — the campaign class; general M deferred);
* per (l, omega): DIRECT forced solve — banded linear system whose
  unknowns are the basis coefficients of every layer (the block-
  tridiagonal sweep in (l,m) space), complex omega on the DSM damped
  axis, causal constant-Q moduli exactly as spheroidal_ref;
* BOTH parities: spheroidal (U, V, R, S) and toroidal (W, T). The
  toroidal part is a direct forced solve too — this REPLACES the
  mode-sum + closed-form-static-completion machinery of
  toroidal_modes.py (a forced solve at complex omega contains the
  static response exactly; the conditional-convergence problem was a
  mode-sum artifact, not physics);
* output: total displacement spectra at surface stations in the
  campaign npz layout (driver dsm_arbitration/run_spectral.py), with
  the PSV/SH split retained for channel-separated verdicts. This
  gives, from ONE code, the composite T reference (SH + spheroidal-T)
  that section "T-CHANNEL PURITY" of toroidal_reference.md required.

Not in the MVP (documented deferrals): l = 0 (parity with mini-tipsv,
which validated against tipsv from l = 1), |m| = 2 sources, gravity,
fluid-layer sources, TFE relief for near-spherical shapes (the
follow-on stage that makes this a BEM-class method for asteroids).

## 2. Bases, scaling, and layer pruning

Per layer, uniform-material analytic solutions (spheroidal_ref
primitives, all previously gated):

* solid: P- and S-type 4-vectors (U, V, R, S) built on j_l and y_l
  (`solid_cols_c`), scaled-series variants for high l (`_fl_scaled`:
  j-family ~ (z/zref)^l S_l, y-family ~ (zref/z)^{l+1} T_l);
* fluid: potential solutions (u_r, s_rr) on j_l, y_l (`fluid_cols_c`,
  extended here with the same zref scaling);
* per-layer zref = k * r_top(layer), so every column is O(1) at its
  layer top and the within-layer span is bounded by the layer's
  radius ratio; the assembled system gets an explicit per-column
  rescale before the solve (solves are column-scale invariant).

Layer pruning (the generalization of mini-tipsv's shell-only branch):
an interface at radius r_i below the source is invisible from
r >= r0 when the round-trip factor (r_i/r0)^{2l} < TOL_PRUNE. For
each l, drop all layers wholly below the first visible interface;
the lowest KEPT layer keeps only its regular (j-type) columns and
imposes NO condition at its bottom — exactly the proven mini-tipsv
shell-only construction. TOL_PRUNE = 1e-12 (mini-tipsv's validated
l=19/20 seam discarded (b/a)^{2l} ~ 3e-11, and its full branch was
still well-conditioned there; a seam-smoothness gate re-verifies
here). Pruning also caps the y-column within-layer span at
~TOL_PRUNE^{-1/2} per layer, so scaled columns never overflow.

## 3. Interface conditions and source jumps

Rows of the banded system (top-down):

* free surface: R = S = 0 (solid) / s_rr = 0 (fluid); toroidal T = 0;
* solid-solid: (U, V, R, S) continuous — 4 rows; toroidal (W, T)
  continuous — 2 rows;
* fluid-solid: U and R continuous, S = 0 on the solid side — 3 rows;
  toroidal: T = 0 on the solid side — 1 row (SH does not enter
  fluid; a fluid layer splits the toroidal domain, and only the
  outermost solid run above the outermost fluid layer is solved);
* source radius r0: the source layer is split into sub-layers below/
  above r0; continuity rows carry the inhomogeneous jump [y].

Spheroidal jumps: reuse `source_jumps` ([y] = F1 + A(r0) F0 with the
F0/F1 load vectors carrying the 1/L S-row factor — FD-verified and
tipsv-anchored in the mini-tipsv campaign).

Toroidal jump (derived here; static special case was already
FD-verified in toroidal_modes.static_factor): the r-h couple source
with amplitude q = M0 D pairs as int W_t f r^2 dr =
q (W_t'(r0) - W_t(r0)/r0). Writing f through delta', the ODE
mu W'' + (2 mu/r) W' + (rho w^2 - mu L/r^2) W = q d'(r-r0)/r^2
+ q d(r-r0)/r^3 and expanding d'(r-r0)/r^2 = d'/r0^2 + 2 d/r0^3
gives, by distributional order matching:

    delta':  mu [W]  = q / r0^2          -> [W]  = q/(mu r0^2)
    delta :  mu [W'] + (2 mu/r0)[W] = 3q/r0^3
                                         -> [W'] = q/(mu r0^3)

hence in (W, T) variables, [T] = mu([W'] - [W]/r0) = 0: the toroidal
source is a pure displacement jump [W] = q/(mu(w) r0^2), [T] = 0,
frequency-independent (the rho w^2 W term is regular across r0) —
consistent with the FD-verified static jumps.

## 4. Gates (anti-circularity per §7)

* G-A1 partition invariance (internal, exact): splitting any uniform
  layer into identical-material sub-layers must not change the
  surface response (tests the interface assembly with no external
  reference). Tol 1e-9 across an (l, k) grid spanning the seams.
* G-A2 vs mini-tipsv (independent assembly, same primitives):
  corefluid_q50 per-(l,k) surface (U, V) and end-to-end spectra
  equal to spheroidal_ref to ~1e-8.
* G-A3 toroidal static limit (closed form, independent): forced
  solve at |w| -> 0 on ball and annulus matches
  toroidal_modes.static_factor (l >= 2). Resonance positions along
  real w match mode_catalog eigenfrequencies.
* G-A4 toroidal physics anchor: full spectra vs toroidal_mode_spectra
  (independent route: eigen-decomposition + static completion) on the
  corefluid annulus at 50 km — expect agreement at the mode-sum's
  own truncation class (~1%).
* G-A5 pruning/seam smoothness: l-scan of the surface response
  across the pruning seams at every k, plus TOL_PRUNE sensitivity.
* G-A6 acceptance (absolute): corefluid_q50, 50-km source, per-k
  complex spectra vs DSM tipsv (ABSOLUTELY VALIDATED reference) on
  Z, R, and PSV-T + composite T — the spectral solver must sit at
  mini-tipsv's level INCLUDING the quasi-static plateaus that the
  P0 BEM cannot reach.
* Later (multi-layer production): deep-source (637 km) staircase leg
  vs the healthy DSM tipsv gprem/prem legs; exact_modes
  eigenfrequency checks per stack.

## 5. Staging

* A0: module + gates G-A1..A-5 (this commit).
* A1: run_spectral.py driver; 50-km corefluid acceptance G-A6;
  composite-T verdict figure.
* A2: multi-layer staircase leg (gprem_s8-class) vs deep DSM leg.
* A3 (future): TFE relief |l-l'| coupling for near-spherical shapes;
  ensemble economy (factor per (l,m) once, many sources ~ free).

## 6. A0 gate results (2026-07-16, tests/test_spectral.py)

  G-A1 partition invariance      1.5e-09  (per-channel scale)
  G-A2 vs mini-tipsv units       2.9e-07  (two measured floors, see
       below); seam arbitration  2.3e-12  vs the full 12x12;
       end-to-end PSV            3.3e-09  (lmax 60)
  G-A3 toroidal static limit     4.1e-10; 0T2 resonance exact
  G-A4 vs toroidal mode sum      converges 0.52x per fmax doubling
       toward the forced solve (the residual is the mode sum's OWN
       l=1 tail: static completion covers l >= 2 only)
  G-A5 pruning tol 1e-12 vs 1e-16  3.4e-10

The two G-A2 broad-grid floors are the REFERENCE's, not ours:
(a) mini-tipsv's shell-only branch discards (B/r0)^(2l) fluid
coupling at its hard l=20 seam — 1.7e-7 at 637-km depth, where the
new solver instead matches mini-tipsv's own full 12x12 branch to
2e-12 (and to 2e-9 even at k=1, campaign model); (b) a ~3e-7
conditioning floor at the extreme (l >= 400, k = 3) corner where the
harmonic weight (r0/a)^l is physically negligible.

## 7. A1: 50-km corefluid production run + acceptance
## (2026-07-16/17) — G-A6 CLOSED

Driver run_spectral.py (k-parallel; lmax-1400 production run 861 s
on a debug node: 812 s pole couplings — the optimization target —
+ 48 s for ALL 138 x 1400 x 3 solves on 48 procs). Output
spectral_corefluid_q50.npz (u = u_psv + u_sh; parts kept separate
for channel-clean verdicts).

Findings, in discovery order:

1. STALE CAMPAIGN NPZ: spectral PSV == CURRENT mini-tipsv code to
   8.2e-9 end-to-end, but the stored minitipsv npz differed by
   6.9e-3 at k=20 (40% at sub-band k=1) — it predated the zref seam
   fix (0daa86e). Regenerated (1188 s single-proc; now matches
   spectral to 5.6e-9 band-wide). Per-(l,k) unit responses of the
   two current codes agree <= 2.1e-8 over ALL l=1..900 (sum-level
   1.2e-10). No prior campaign conclusion changes (the 0.7% in-band
   staleness is invisible in 5-15%-class verdict tables).

2. LMAX-900 STATIC-TAIL TRUNCATION was the dominant error of the
   old "mini-tipsv class" mrt-vs-tipsv tables: at 50-km depth the
   (r0/a)^l tail (e-folding 127) needs lmax ~ 1400 (tail weight
   ~1%); at 900 it is ~10% of peak weight. Found via the SH
   mode-sum depth arbitration (mode sum at lmax 1100 moved AWAY
   from the lmax-900 spectra by 0.12-0.19 at far stations = the
   l=901..1100 tail). At lmax 1400:
     mrt vs tipsv, per-k complex k=20..138: median rel 0.026
     (Z rel 0.020-0.039 ratios 0.986-1.011; R rel 0.003-0.086
     ratios 0.989-1.005) — the whole-band quasi-static plateaus
     agree with tipsv at the percent level. The historical
     "mrt Z 0.85-1.25, R 0.85-1.24" anchor scatter was mini-tipsv's
     lmax-900 truncation, not solver disagreement.

3. SH CHANNEL VALIDATED at 50 km: vs the toroidal mode sum
   regenerated at proven-converged truncation (fmax 20 mHz, lmax
   1400, n_keep 16 — chosen by a depth-escalation arbitration that
   converged ONTO the forced solve, 0.27-0.65 -> 0.013-0.025 at the
   far stations): vector per-k rel median 0.010 over all 12
   stations, dominant-component ratios 0.990-1.005. The forced
   solve SUPERSEDES the mode sum as the campaign SH reference (no
   truncation ladder, exact statics, includes l=1 exactly).

4. l=0 RADIAL BRANCH ADDED (mini-tipsv omits it): the mrr-Z
   high-band disagreement vs tipsv (k=100-138 median 0.22, growing
   toward the radial-mode branch, absent from mrt and from R —
   the exact l=0 fingerprint) collapses to median 0.025 (max
   0.037) with l=0 included. Partition invariance 2e-16; strict
   additivity (l0=False leaves l>=1 content bitwise unchanged).
   FINAL full-band mrr tables (lmax 1400 + l=0): Z rel 0.016-0.037
   at ALL stations, ratios 0.996-1.003; mrr median rel 0.063 with
   the remainder entirely the low-k mrr-R item below.
   Acceptance waveform figures (band-passed velocity, DSM |
   mini-tipsv | spectral on Z/R, composite ref | spectral on T):
   dsm_arbitration_u3/wf_rungA_mrt.png and wf_rungA_mrr.png — the
   spectral trace sits on DSM at every station/channel; the
   lmax-900 mini-tipsv truncation error is visible on mrt-Z.

5. OPEN (tipsv-side): mrr-R at 50 km disagrees O(1) vs tipsv in
   k=20..59 ONLY (median 1.02 there; 0.012 at k=60-99, 0.034 at
   k=100-138; unchanged by lmax 900->1400, amplitude-ratio ~1.0)
   while BOTH in-house analytic routes agree with each other to
   5.6e-9. This sharpens the long-suspected tipsv fluid-core
   near-DC (gravity-free undertone degeneracy) channel issue —
   previously "whole-band plateau noise" (2026-07-11), then
   over-broadly exonerated by the truncation-widened mrt anchor:
   the mrt-R plateau IS physical (now 1.000 +- 0.005 vs tipsv),
   but tipsv mrr-R below k~60 at shallow depth remains unarbitrated
   (needs a third route; SEM fluid-core is broken at ULP).
   UPDATE (A2 ladder, 2026-07-17): the SAME O(1) low-k mrr-R split
   reproduces on the ALL-SOLID gprem staircases (rel 1.15 at
   k=20-59, identical for 3/5/9 layers, deep 1200-km-radius source)
   and VANISHES on the homogeneous ball with the same source
   (0.035) — so the mechanism is NOT the fluid core: it is
   layered-model-specific, m=0-specific (mrt-R is 1-4% there),
   layer-count-independent. Our l=1 m=0 unit responses show no
   near-null amplification on layered stacks (same order as homog),
   so it is a genuine formulation-level split between DSM tipsv and
   the analytic uniform-layer route on that channel. Third-route
   arbitration (regularized-force shooting, or exact_modes-class
   forced construction) remains the settle path.

## 8. A2: N-layer staircase acceptance (2026-07-17) — PASSED

run_spectral.py on the gprem ladder (all-solid graded-PREM
staircases, source r0 = 1200 km INSIDE the core ball — exercises
the deep-source split-innermost-layer path; Q = 50 causal 1-Hz;
the 'c' models, where DSM ran the exact staircase constants; the
manifest "elastic" label is stale — the .inf zones carry Qmu=50).
9-layer gprem_s8c run: 86 s wall (32 procs shared).

* gprem_s8c vs DSM tipsv per-k complex, k=20..138:
  mrt median rel 0.0069 (Z rel 0.001-0.011, ratios 0.990-1.003;
  R ratios 1.00-1.06) — essentially exact; mrr median rel 0.100
  (Z rel 0.001-0.057, ratios 0.965-1.028). For context the BEM
  ladder verdict on this exact leg was 0.593 median (model-class
  error at the ULP band): the spectral solver removes it entirely.
* T composite (tipsv PSV-T + tish SH-T): rel 0.11-0.16, ratios
  1.01-1.15 — sign-consistent with the KNOWN tish multi-zone
  toroidal bias (peaks 4-7 percent low vs exact modes).
* Ladder controls: s2c / s4c / s8c band tables are IDENTICAL to a
  few percent — no per-interface error accumulation in either code.
* Homog control (same deep source): all channels 0.1-4.8 percent at
  k <= 99; band-top (k=100-138) degrades to 7-12 percent on homog
  and 15-23 percent on the staircases (mrr-R ratio 1.10-1.17,
  mrt-R ratio 1.00 = phase-type), layer-independent — a deep-source
  band-top disagreement, unarbitrated (corefluid at 50-km depth
  showed 1-3 percent there; not blocking: the campaign band centre
  is clean).
