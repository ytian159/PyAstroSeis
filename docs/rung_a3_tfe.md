# Rung A3: first-order interface relief (TFE) in the spectral sweep
# (design + MVP scope, 2026-07-17)

Goal: make the per-(l,m) spectral sweep (rung A, spectral.py) handle
NEAR-SPHERICAL boundaries by first-order boundary perturbation
(transformed-field-expansion / Woodhouse boundary-perturbation
class): relief r = d + h(theta,phi) on one or more interfaces, h
expanded in fully-normalized Y_LM. This is the "BEM twin of DSM"
end-game of fast_methods_notes section 4, and the ensemble economy
follows: the unperturbed per-(l,m) factorizations are reused for
every relief member (RHS-only changes).

## 1. Transfer of boundary conditions (general, both parities)

Perturbed surface S: r = d + eps h; outward normal to O(eps):
n = rhat - (eps/d) grad_1 h. For any condition that holds on S,
Taylor-transfer to r = d:

  q(d + eps h) . n = [q0 . rhat](d)
    + eps [ h d/dr (q0 . rhat) + q1 . rhat
            - (1/d) q0 . grad_1 h ](d) + O(eps^2).

So the first-order field y1 satisfies the UNPERTURBED equations in
each layer with INHOMOGENEOUS boundary/jump data at r = d built from
two ingredients:
  (A) -h * d/dr(zeroth-order condition quantity)   [radial transfer]
  (B) +(1/d) * (zeroth-order stress) . grad_1 h    [normal tilt;
      displacement conditions have no tilt term]
Both are products of known angular functions with h -> in the
harmonic basis they become coupling sums over (l, m) x (L, M) ->
(l', m'), with m' = m + M and |l - L| <= l' <= l + L.

PARITY STRUCTURE (corrected 2026-07-17): the unperturbed operator
is parity-diagonal, so u1^SH solves toroidal equations with the
TOROIDAL PROJECTION of the transferred data, and u1^PSV the
spheroidal projection. But the transferred data are built from the
FULL unperturbed field u0 = u0^PSV + u0^SH, and the projections mix
parities: the toroidal projection of transfer(u0^PSV) is nonzero
for L >= 1 relief (and vice versa). So du^SH = [SH->SH] +
[PSV->SH] and du^PSV = [PSV->PSV] + [SH->PSV]. The section-2 MVP
implements only [SH->SH]; it is complete ONLY for pure-SH
unperturbed fields and for L = 0 relief (no conversion) — the Y00
gate cannot see the omission. A3b adds the [PSV->PSV] block and
BOTH conversion blocks.

## 2. MVP (this rung): toroidal relief on SH-free boundaries

Scope: relief on boundaries where the toroidal condition is T = 0 —
the outer free surface (r = a) and the top of the outermost fluid
(CMB, r = b): both single-sided (no jump algebra, one inhomogeneous
row per boundary). First-order condition for target (l', m'):

  T1_{l'm'}(d) = - sum_{lm} h_LM [ dTdr0_{lm}(d) A(l',L,l;m',M,m)
                  - (1/d) TILT_{lm,LM->l'm'} ]

  dTdr0 = mu W0'' - mu W0'/d + mu W0/d^2 evaluated via the ODE:
        = -(3 mu/d) W0' + (mu (L_l + 1)/d^2 - rho w^2) W0,
    W0' = T0/mu + W0/d      (all from the unperturbed solve)

  A(...) = <C_l'm', Y_LM C_lm> / N_l'
         = [l(l+1) + l'(l'+1) - L(L+1)]/2 * Gaunt(l'm'; LM; lm)
           / N_l'                                    (gradient Gaunt)

  TILT = <C_l'm', sigma0 . grad_1 h> / N_l' with sigma0 the
    tangential-tangential toroidal stress mu (W0/d) * e(C_lm)
    (plus the sigma_r-tangential part already counted in T0 rows).

MVP implementation choice: evaluate the angular coupling integrals
NUMERICALLY (Gauss-Legendre in theta x exact m-selection in phi,
using the same Legendre recurrences as toroidal_reconstruct — the
momentfit convention-cancellation pattern), not via closed-form
Gaunt/Wigner algebra; cross-validate against closed forms where
available (gradient-Gaunt identity above) and against brute sphere
quadrature (sympy/numeric dual derivation, house pattern). Closed
forms become an optimization later.

First-order solve: for each target (l', m') in the coupled band,
re-run toroidal_unit with the inhomogeneous boundary value moved to
the RHS (the T=0 row of the relieved boundary gets rhs =
T1_{l'm'}); the source-jump forcing stays in place for the
unperturbed part only (u1 has no source). du^SH = reconstruct of
the W1 coefficients.

## 3. Gates

* G-TFE-1 (EXACT, tests the radial-transfer term end-to-end):
  Y00 relief == radius change. For h = c Y00 (grad_1 h = 0, tilt
  absent), du from TFE must match the FINITE DIFFERENCE of exact
  unperturbed solves with the boundary at d and d + c/sqrt(4 pi)
  (Richardson in the FD step; first order in c).
* G-TFE-2 linearity: du(2 eps) - 2 du(eps) scales as eps^2 (order
  fit), and the coupled-band truncation is checked by widening.
* G-TFE-3 selection rules: for zonal Y20 relief, only l' = l, l+-2
  (m' = m) receive coupling through the scalar-Gaunt term; measured
  couplings outside the band < 1e-12.
* G-TFE-4 coupling-integral cross-check: numerical quadrature vs
  the gradient-Gaunt closed form for a grid of (l, L, l', m).
* G-TFE-5 (cross-code physics anchor, A3b when spheroidal TFE
  exists): BEM rung-3 Y20 CMB-relief ensemble (rung3_ensemble/,
  |du|/|u| 4.6e-3 class, linearity 2.00) vs spectral TFE on the
  same model/relief — the BEM du is P-SV-dominated, so this anchor
  activates with the spheroidal implementation.

## 3b. MVP gate results (2026-07-17) — ALL PASSED

  G-TFE-1 Y00 == exact radius change: bottom (CMB) 1.06e-9;
          top (free surface) 2.7e-8 AFTER adding the receiver-
          advection term (stations ride the moved surface:
          du += h dW0/dr = h W0/a at T=0; the term is 11 percent of
          du at 50-m relief and the gate fails loudly without it).
  G-TFE-3 selection rules (Y20): parity-off couplings < 5e-16 of
          on-band; triangle rule enforced exactly.
  G-TFE-4 L=0 closed form: I1 = delta/sqrt(4 pi) to 9.6e-15,
          I2 = 0 exactly.
  G-TFE-4b sympy dual derivation (validation/tfe_sympy_check.py,
          pytorch env): grid-quadrature I1 AND the tilt integral I2
          match exact symbolic surface integrals to 3.2e-14 /
          1.7e-14 over l, l' <= 5, L=2 (convention pin: our Ybar ==
          sympy Ynm at machine precision first).

Implementation: pyastroseis/spectral_tfe.py (toroidal_relief_spectra;
_AngCache k-independent coupling cache; toroidal_unit gained
bc_rhs/full kwargs, no-op defaults). Cost: one extra toroidal solve
per (l', m, w) per relief harmonic + cached couplings.

## 4. A3b: spheroidal relief — derivation (2026-07-17)

Field objects per (l, m) at radius r (code conventions:
u = U Y rhat + V grad1 Y + W C, C = rhat x grad1 Y / sqrt(L)
orthonormal, V on the UNNORMALIZED gradient):

  t = sigma.rhat = R Y rhat + S grad1 Y + T C
  d/dr u = U' Y rhat + V' grad1 Y + W' C     (rhat, grad1 Y, C are
  d/dr t = R' Y rhat + S' grad1 Y + T' C      r-independent shapes)

with (U',V',R',S') = A_sph(r) y from spheroidal_ref.system_matrix
per SIDE, and W' = T/mu + W/r, T' = -3T/r + (mu(L-2)/r^2
- rho w^2) W.

Full stress pieces entering the tilt sigma0 . grad1 h:
  (sigma.v)_r = S (grad1 Y . v) + T (C . v)          v tangential
  (sigma.v)_a = sigma_ab v_b,
  sigma_ab = lam (div u) delta_ab Y-part + 2 mu e_ab,
  div u = U' + 2U/r - L V / r,
  e_ab = (1/2r)(D_a u_b + D_b u_a) + (U Y / r) delta_ab evaluated
  on the theta-grid from (V, W) x second derivatives of Y (same
  covariant formulas as section 2, now for BOTH tangential fields
  u_t = V grad1 Y + W C).

WELDED solid-solid at r = d (zeroth-order continuity of u and t):
  displacement rows:  [u1] = -h [d/dr u0]           (no tilt term)
  traction rows:      [t1] = -h [d/dr t0] + (1/d) [sigma0 . grad1 h]
Each right-hand side is a VECTOR field on the sphere; project onto
Y' rhat (U/R rows), grad1 Y'/L' (V/S rows), C' (W/T rows —
the parity-conversion entries). The jump [d/dr y0] is nonzero
because the material contrast makes A_sph differ across d.

FREE SURFACE (solid top): single-sided traction version
  t1(a) = -h d/dr t0(a) + (1/a) sigma0 . grad1 h,
plus the receiver-advection term du_obs += h d/dr u0(a) when
stations ride the surface (section 3b lesson).

NULL GATE (the strongest welded test, all L): an ARTIFICIAL welded
interface between IDENTICAL materials with ANY relief must produce
du == 0 identically — [d/dr y0] = 0 and [sigma0] = 0 across a
material-free interface, so every transfer term vanishes; any
nonzero du is implementation error (tests radial factors, tilt
algebra, and projections together, with no reference needed).

## 4b. A3b stage-1 results (2026-07-17) — welded + free surface
## PASSED (both parities + conversion blocks)

Implementation: spectral_tfe.relief_spectra (general engine:
ReliefCouplings catalog G0/GA/GC/H0/Hiso/HV/HW with exact triangle
masks; _side_coeffs radial bundles; spheroidal_unit/toroidal_unit
gained iface_rhs/bc_rhs/full — interface side values AND radial
derivatives by basis-FD on the solved coefficients).

Gates (tests/test_spectral_tfe.py):
  NULL WELD (identical materials, Y20 AND Y21, mixed mrt+mrr):
    |du|/|u0| = 3.0e-11 / 2.4e-11 — every transfer term cancels.
  Y00 weld contrast (real material jump) vs exact interface-move
    FD: psv 1.5e-07, sh 1.9e-10.
  Y00 free surface vs exact FD: psv 1.9e-06, sh 2.8e-08
    (receiver advection for BOTH parities).
  Conversion parity: same-parity blocks live on even |l'-l|,
    conversion blocks on odd — wrong-parity entries 1.5e-15.
  Sympy dual derivation extended to the new angular matrices
    (G0, GA_v, GA_w, GC_v, Hiso_v/w, HV_v/w).

TWO NUMERICAL FINDINGS worth keeping:
  (1) Y'Y^-1 (system_matrix) loses ~1e-3 relative on derivatives at
      l >= L_SERIES in surface-referenced scaled bases, and the
      error is zref-SENSITIVE. Interface side derivatives now come
      from basis-FD on the solved coefficients (no inversion), and
      source_jumps now references its basis AT r0 (locally optimal
      conditioning; also makes J independent of the outer radius —
      the zref-sensitivity had made the Y00-top FD reference legs
      carry spuriously different source jumps, an O(1) artifact on
      the tiny-but-coupling-amplified l >= 20 harmonics).
  (2) The l = 0 channel moves with relieved interfaces for Mrr
      sources (measured 1.6% of du when the reference includes it);
      l = 0 relief coupling (l=0 <-> l'=L) is an open A3b item —
      FD gates exclude it via l0=False for now.

Limits of the stage-1 gate set (honest accounting): the NULL gate
proves DELTA-cancellation and the Y00 gates prove the radial
chain, but neither quantitatively validates the finite-L TILT
coefficient assembly (ciso/cV/cW radial factors); the sympy checks
pin the ANGULAR integrals. The full finite-L quantitative anchor is
stage 3's BEM rung-3 Y20-CMB ensemble (fluid-solid).

FLUID-SOLID (CMB; slip): conditions on the perturbed surface are
u.n continuous, sigma.n = -p n (tangential traction zero on the
solid side, normal traction = fluid -p). The transfer involves the
tangential SLIP [u0_t] in the tilt of u.n, and the fluid p0, dp0/dr
in the traction rows — A3b stage 3 (unlocks the BEM rung-3 Y20-CMB
cross-anchor). Ensemble economy: cache per-(l,m) LU factors of the
unperturbed systems — member cost = RHS + back-substitution.
