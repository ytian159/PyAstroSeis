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
