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

PARITY STRUCTURE: relief couples toroidal <-> spheroidal at first
order, BUT the unperturbed operator is parity-diagonal, so the
SH part of u1 is sourced ONLY by the toroidal projection of the
transferred data and the P-SV part ONLY by the spheroidal
projection. A toroidal-only implementation is therefore COMPLETE
for du^SH at O(eps); same for spheroidal/du^PSV. (The cross terms
feed the OTHER parity's du, not each other's.)

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

## 4. A3b (next): spheroidal relief + fluid-solid interfaces

The spheroidal transfer needs the welded 4-vector jump version of
section 1 (displacement rows get only term (A) with [d/dr y0] built
from the two-sided system matrices; traction rows get (A) + tilt
(B) with the full sigma0 tensor) and the fluid-solid slip
conditions (continuity of u.n and traction, tangential slip free —
the tilt rotates which combination is continuous). Ensemble
economy: cache the per-(l,m) LU factors of the unperturbed systems
(they are the SAME operators for every member) — member cost =
RHS assembly + back-substitution only.
