# Semi-analytic toroidal reference (mini-tish) for the homogeneous sphere

Status: V2-VALIDATED 2026-07-11; 50-km projection route precision-
limited (see RESULTS at end). Gates T1-T4 machine precision
(tests/test_toroidal_ref.py). V2 anchor: vs healthy tish at 637 km —
median rel 0.75%, min corr 1.0000, amp 1.006 over 36 traces, decisive
conjugation calibration (0.0075 vs 1.106), lmax 200
(dsm_arbitration_ref637/minitish_compare_homog_q50_mrt.json).

50-KM LIMIT (empirical, matches the Codex Q4/Q5 warnings exactly):
the projection route inherits the full-space kernels' small-|kd|
cancellation loss (~10 digits at |kd| ~ 4e-5..0.06 on the epicentral
panels), so the projected W_lm hit a noise floor well below the
physical (r0/R)^l decay: two-tier disagreement O(1-6) at ALL lmax
tried (800/1200/1600/2548), monitors 0.2-0.8, l-tails non-decaying,
worst at low k. NO truncation rescues it. The fix is the RECIPROCITY
ROUTE (also Codex's recommendation): coefficients from
M : eps[j_l(k r) C_lm](x_src) — analytic at the source, no Green
evaluation, no cancelling surface integral; scaled Bessel products
j_l(k r_s) h_l(k R) via the stable ratio recurrences; calibrate the
one l,m-independent constant K(w) against the projection route at
637 km (where both are clean) and verify K-constancy over (l, m).
OPEN WORK ITEM — until then the homogeneous-ball 50-km T channel has
no exact reference; the SPECFEM uniform3 leg provides the T reference
on the corefluid_q50 model instead. Purpose: a trustworthy SH/T-channel
reference for shallow (< 100 km) sources at ULP, where DSM tish is
proven to emit unconverged high-l noise (fast_methods_notes.md §8).
Scope: homogeneous isotropic solid sphere (radius R, complex vs from
Q with the DSM causal-dispersion convention), interior point moment
tensor, complex frequencies w = 2 pi k/tlen + i omegai — i.e. the
homog_q50 configuration. Toroidal part only (compared against SH legs
only).

## 1. Formulation — incident projection + free-surface closure

Write the total field as u = u_inc + u_sc, with u_inc the analytic
full-space MT field (source.u0eM / greens_deri_src — oracle-validated
to machine precision) and u_sc regular in the ball. In the toroidal
basis C_lm (momentfit conventions, orthonormal), each (l,m)
decouples; the radial functions satisfy the spherical Bessel
equation with k = w/vs_complex:

    u_sc:   W_sc(r)  = B j_l(kr)          (regular)
    u_inc:  W_inc(r) = c h_l^(1)(kr)      (outgoing, r > r_source)

Toroidal traction on r = const: T(r) = mu (W' - W/r). The traction-
free surface T_tot(R) = 0 gives B = -T_inc(R)/T_j(R), hence

    W_tot(R) = W_inc(R) * [ 1 - (k hl'/hl - 1/R) / (k jl'/jl - 1/R) ]   (*)

evaluated at kR. EVERYTHING reduces to the two logarithmic
derivatives and the measured incident coefficient W_inc,lm(R):

* W_inc,lm(R) by numerical projection of u0eM onto conj(C_lm) over
  the sphere (source-frame graded product grid; azimuthal content
  |m| <= 2 exactly in the source frame, so 5 m-columns and an
  m-restricted normalized-Legendre recurrence to l_max ~ thousands;
  the l-projections fold into the single upward l-recurrence pass).
* k jl'/jl at kR: continued fraction / downward ratio recurrence
  (minimal solution) — stable at any l.
* k hl'/hl at kR: upward ratio recurrence (h is the dominant
  solution) — stable at any l.

No raw Bessel values, no source jump conditions, no incident-traction
kernel: (*) uses ratios only, so there is no overflow/underflow at
large l, and the same C_lm code evaluates both the projection and the
station reconstruction, so any basis-convention slip cancels
identically (the momentfit property).

High-l limit check: k jl'/jl -> l/R, k hl'/hl -> -(l+1)/R, so the
bracket in (*) -> (2l+1)/(l-1) — the static free-surface
amplification; smooth, finite, correct static limit.

Station field: u_T(x) = sum_{l>=1} sum_{|m|<=2} W_tot,lm(R)
C_lm(x; source frame), rotated back to the earth frame; truncate
when the l-tail contributes < tol (physical decay ~ (r0/R)^l — the
sum converges by l ~ a few thousand at 50 km depth, ~200 at 637 km).

## 2. Why this is trustworthy where tish is not

tish integrates a radial two-point BVP per (l, freq) on a fixed grid
and accumulates roundoff that does NOT decay with l (proven flat
band-increments). Here each l-term is (measured smooth projection) x
(closed-form stable ratio factor); the projection decays physically
like (r0/R)^l because u0eM is evaluated in closed form, so the l-sum
terminates physically. The quadrature is the validated momentfit
machinery (two-tier convergence check retained).

## 3. Validation gates

V1 basis/quadrature: two-tier grid agreement per (l,m) (as in
   momentfit M3); high-l tail of W_inc,lm decays like (r0/R)^l.
V2 CONVENTION + PHYSICS ANCHOR: 637-km source, homog_q50 — tish is
   HEALTHY there (deep branch, converged sum). mini-tish T spectra at
   the 12 stations must match dsm_arbitration_ref637 tish SH spectra
   to the DSM-vs-BEM floor class (<< 10%); one global conjugation
   flag calibrated here and frozen (house pattern).
V3 in-band modal structure: denominator k jl'/jl - 1/R sweeps near
   toroidal eigenfrequencies (0T2 ~ 0.379 mHz in-band for PREM-class;
   here the homog values) — the complex omegai keeps the solve
   finite; spot-check l=2 resonance location against the elastic
   eigenfrequency condition.
V4 50-km application: replaces the void tish reference; the BEM's
   mrt T at 50 km gets its first valid verdict.

## 4. Implementation

pyastroseis/toroidal_ref.py: legendre_m_restricted (P̄_lm, dP̄/dtheta
for m in 0..2, all l <= lmax, vectorized over nodes, projections
accumulated inside the l-recurrence), bessel log-derivative ratios
(complex z), toroidal_reference(model params, src, stations, freqs)
-> T spectra per station. Driver dsm_arbitration/run_minitish.py
writing spectra in the campaign's npz layout for the existing
synthesize/compare pipeline.

## Mode-sum route — IMPLEMENTED (2026-07-11 night)

pyastroseis/toroidal_modes.py replaces the precision-dead projection
route at shallow depths by NORMAL-MODE SUMMATION with everything
analytic: eigenfrequencies from the uniform-layer Bessel-matching
determinant (ball or annulus-over-fluid), eigenfunctions
W = A j_l + B y_l with (A,B) from the CMB traction condition,
N = int rho W^2 r^2 dr, and excitation from the mode strain AT the
source — for any earth-frame Mr* tensor the source frame (e3 = r)
keeps M = M0(z h^T + h z^T), so only eps_{r,horiz} contracts:

    E_lm = M0 (W' - W/r)|_{r0} conj(C_lm(pole)) . h ,

with C_lm(pole) evaluated through toroidal_reconstruct itself
(conventions cancel; only m = +-1 survive; mrr gives E == 0, verified
1e-20). u(w) = sum_j E_j W_j(a) C_lm / (N_j (w_j^2 - w^2)) at the
campaign's complex w; catalog truncated at fmax = 10 mHz (in-band
truncation error < 1%). Attenuation: first-order causal constant-Q
mode perturbation (1-Hz reference), q_sign = -1 frozen.
V-a ANCHOR PASSED: ball config at 637 km vs the V2 mini-tish forced
solution — median rel 4.6%, corr 0.98-1.000, decisive conjugation
calibration (0.046 vs 1.06/2.18); driver dsm_arbitration/
run_tormodes.py, comparator tormodes_check.py.

T-CHANNEL QUASI-STATIC FINDING (kills naive 50-km T verdicts): on
corefluid at 50 km the EXACT SH T-field is flat-in-k across the
whole ULP band at 20-150 deg (quasi-static dominated; e.g. 8.1e-5 m
at ST00, matching the flat tipsv PSV-T 1.37e-4 m) — so total-field
T scoring at generic azimuths compares quasi-static fields for which
no trustworthy spheroidal complement exists (tipsv near-DC
pathological, SEM spheroidal broken). PROTOCOL: SH-pure verdicts
need SOURCE-MERIDIAN stations (spheroidal-T and toroidal-T are in
azimuthal quadrature for Mrt); dsm_arbitration_u3T = 7 meridian
faces (off-meridian <= 1.9 deg), BEM re-evaluated there, reference =
mode-sum alone (tor_verdict.py).

## Appendix: SPECFEM3D_GLOBE as a ULP reference — protocol findings
(2026-07-11, uniform3 campaign debugging)

Two contamination mechanisms make STOCK SPECFEM output unusable as a
ULP (f <= 5.3e-4 Hz) reference, both invisible in normal-period
benchmarks because ULP signals are ~1e-4 of local field scales:

1. STF truncation step: the CMT erf quasi-Heaviside starts at
   t0 = 1.5*hdur where erf(-2.44) = -0.9994, injecting a ~3e-4*M0
   step at simulation start whose broadband ringing (grid periods,
   essentially undamped in a closed sphere: 20-s spectral peak
   dominating every trace) persists for the whole record. FIX:
   USER_T0 >= 5*hdur (constants.h). Verified: ringing peak moved off
   grid scale and in-band correlation with the BEM appeared
   (ST00 Z: -0.26 -> 0.60).
2. Single-precision solver (default CUSTOM_REAL): ~1e-7 relative
   roundoff of the ~0.1-m near-source field is injected continuously
   into barely-damped modes (T ~ 154 s sits below the forced SLS band
   -> high effective Q), producing a ~1e-7 m broadband floor above the
   ~1e-8..1e-6 m ULP signals. FIX: --enable-double-precision
   (2x cost). [check run pending at this writing]

Plus the band fix from the fork itself: the per-NEX auto SLS band
(~70-3900 s at NEX64) must be forced to cover the source band, else
Q and physical dispersion are wrong in-band.

3. (2026-07-11, double-precision NEX32 check) tipsv R-channel
   pathology on the FLUID-CORE model at 50-km source: the tipsv R
   spectra are a flat-in-k plateau spanning the ENTIRE band (1.1e-5 m
   at ST00, 3e-6 m at ST05 — physical-static order at k -> 0 but
   persisting far beyond the physical near-field corner), the same
   flat-band fingerprint as the shallow-tish noise. SEM and BEM agree
   with each other on R (corr 0.983 at ST00) against tipsv's 15x.
   The gravity-free fluid-core near-DC (spheroidal undertone
   degeneracy) is the suspected mechanism. CONSEQUENCE: on
   corefluid_q50 at 50 km, tipsv is a valid reference for Z ONLY;
   R and T verdicts need the SEM leg.
   Verified-clean SEM chain for the record (NEX32 DP check): input
   moduli exact (mu_nd 0.104338 = rho vs^2), scale factors exact and
   applied once (0.87757 x 0.95607 = 0.83902; muv trace 0.104338 ->
   0.087542 -> 0.095594 incl. x1.09199 unrelaxed), realized model
   table exact in all three regions (xwrite_profile CARDS), SLS fit
   tau_e = {18478.86, 2447.16, 329.27} s == independent scipy refit,
   in-band Q 48.9-51.1, amplitudes vs tipsv Z 0.98-1.06 (Q_eff ~ 50
   confirmed).
4. SETTLED (2026-07-11 evening, exact-eigenmode arbitration): the
   ~3.8 s/deg SEM Z lateness vs tipsv+BEM is a STRUCTURAL SPECFEM
   defect on the fluid-core spheroidal branch at ULP, NOT a material
   or attenuation issue on either side. Chain of evidence:
   (a) elastic A/B: SEM viscous-minus-elastic lag = +1.73 s/deg =
       exactly its designed causal dispersion (attenuation module
       exonerated); the residual +2.07 s/deg survives with
       ATTENUATION=.false. -> structural.
   (b) tipsv Q self-test (Q50 vs elastic input, same binary):
       +1.30 s/deg slower + amp 1.15->0.96 with distance = the full
       CAUSAL fingerprint; tipsv's Q machinery correct.
   (c) ABSOLUTE arbitration via exact_modes.py (analytic uniform-
       layer Bessel matching; machinery triple-validated: full-ball
       toroidal det == mini-tish surface-factor poles to 5 digits;
       uniform-ball 0S2 omega*R/vs = 2.650 = Lamb; tipsv homog-Q50
       peaks = ball-exact x causal shift to ~1%): tipsv-ELASTIC
       corefluid Z peaks sit ON the exact 3-layer spheroidal
       eigenfrequencies to <1%, sub-bin (118.26/117.08,
       194.55/195.17, 282.29/283.62, 324.25/326.46, 377.66/376.63,
       438.69/440.28, 469.21/470.40 uHz). TIPSV Z = ABSOLUTELY
       VALIDATED on the fluid-core model.
   (d) BEM == tipsv per-harmonic phase < 0.1 rad across k=30..105
       (independent methods agree); waveform overlays show SEM as the
       same waveform bodily shifted late, growing with distance.
   => SPECFEM3D_GLOBE (NEX32 AND NEX64, elastic and viscous, model
   table verified exact via xwrite_profile, moduli traced exact) runs
   the fluid-core spheroidal branch ~5-8% slow at ULP. Suspected
   locus: fluid-solid coupling / outer-core potential formulation in
   the gravity-free ULP regime (deficit grows toward low frequency).
   MODE-LEVEL CONFIRMATION (330-min elastic record, variable-
   projection fit, start-point independent; sem_mode_fit2.py):
   SEM spheroidal fundamentals 0S2..0S6 = 100.77/171.44/253.07/
   337.27/425.61 uHz vs exact 117.08/195.17/283.62/376.63/470.40 =
   ratios 0.861/0.878/0.892/0.896/0.905 (worse at low l = deeper
   core penetration); SEM TOROIDAL 0T2..0T5 ratios 1.008/0.996/
   0.989/0.992 -> the defect is ISOLATED to the fluid core /
   fluid-solid coupling; SPECFEM's solid-shell physics at ULP is
   healthy (~1%).
   tish multi-zone anomaly CONFIRMED operator-level with a DEEP
   (637 km, converged) elastic source: 0T2..0T5 uniformly -5.1+-0.4%
   (171.66/267.03/358.58/442.50 vs exact) while single-zone tish is
   healthy (0.75% mini-tish anchor). Full draft reports:
   docs/upstream_issues.md.
   T-CHANNEL PURITY (kills naive T scoring): at these mrt azimuths
   the t_hat projection is a ~50/50 SH/spheroidal MIX (|tipsv PSV-T|
   / |tish SH-T| = 0.7-1.06 at every station), so no total-field T
   comparison is SH-clean; the BEM-vs-SEM T attempt is VOID.
   CAMPAIGN CONSEQUENCES: Z reference at 50 km on corefluid =
   tipsv (absolutely validated). BEM VERDICT (deliverable): BEM Z
   timing/phase correct, amplitude 1.2-1.5x HIGH growing with
   distance = real BEM deficit on the fluid-core model at 50 km.
   R channel: no valid reference (tipsv R = flat-plateau noise; SEM
   spheroidal broken). T channel: SEM toroidal validated (~1%) but
   inseparable from its broken spheroidal in any projection ->
   the ANNULUS TOROIDAL MODE-SUM (exact SH synthesis: eigenfrequency
   catalog from toroidal_det + analytic eigenfunctions + M:eps
   excitation at r0, causal-Q perturbation for attenuated legs) +
   validated tipsv PSV-T as the spheroidal complement = the composite
   T reference; next build.
