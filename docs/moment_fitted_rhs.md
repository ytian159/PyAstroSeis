# Moment-fitted RHS — falsification test for the quasi-static diagnosis

Status: EXECUTED 2026-07-11. Outcome (see §6 RESULTS): rig validated;
gate 4 (637 km) IMPROVED 11.10% -> 9.09%; gate 5 (50 km) "failure"
triggered the bug hunt that exposed the DSM shallow-source reference
itself as unconverged high-l numerical noise on the mrt SH channel
(14-700x the physical incident-field bound) — the 50-km quasi-static
"missing channel" of 2026-07-11 was an artifact of the REFERENCE, and
the BEM T response is in the physical class. Original design below. Scope: the cheapest test from
docs/fast_methods_notes.md §7 — correct the low-(l,m) moments of the
discrete incident-trace RHS to their exact continuum values and see
whether the 50-km-source quasi-static channel (mrt T, amp ratio
0.0075 at hmin 20) is recovered against DSM. This is a TEST RIG, not
production machinery: it lives in `pyastroseis/momentfit.py` + an
`ARB_MFIT_LMAX` hook in `dsm_arbitration/run_bem.py`, and is a strict
no-op when the env var is unset (the solver core is untouched).

## 1. What is being corrected, and why it is sufficient

For `homog_q50` the entire system RHS is the incident displacement
sampled at face incenters: `b_f = u_inc(x_f)` (u0eM; "u" rows carry no
scaling). The proven failure (fast_methods_notes §6) is that the
low-(l,m) projections of this SAMPLED trace are cancellation noise
(~1e-4 of the gross integral, sign-wandering under refinement), so the
solved low-l response — which carries the dominant quasi-static field
of a shallow source at the ULP band — is garbage, while the
propagating (mid-l) content is fine.

The discrete solve reads the RHS's low-l content through area-weighted
incenter sums (the discrete operator on a quasi-uniform sphere mesh
nearly diagonalizes over (l,m) up to O(h²)). So the minimal fix:
replace b by b + Δb such that the discrete area-weighted moments of
the RHS equal the exact continuum moments of u_inc, with Δb otherwise
as small as possible.

## 2. Construction

Basis: complex vector spherical harmonics on the unit sphere,
ψ ∈ {P_lm = Y_lm r̂ (l≥0), B_lm ∝ ∇_S Y_lm (l≥1), C_lm = r̂×B_lm
(l≥1)}, all l ≤ L, m = −l..l; K = 3(L+1)² − 2 columns. Y_lm from the
fully-normalized associated-Legendre recurrence (stable to l ≫ 100);
∂θ via the normalized ladder identity; the (m/sinθ)Y terms are finite
(P̄_lm ~ sin^m θ) with an epsilon pole guard.

Discrete moment:  m_k[b] = Σ_f area_f conj(ψ_k(x̂_f)) · b_f
Exact moment:     E_k    = R² ∫ conj(ψ_k(ŷ)) · u_inc(Rŷ) dΩ

Deficit δ = E − m[b]. Minimal W-norm correction (W = diag(area) per
dof) subject to exact moments:

    Δb = Ψ (Ψᴴ W Ψ)⁻¹ δ,      Ψ = [ψ_k(x̂_f)] (3n × K)

so m[b+Δb] = E exactly; the correction is spread as a smooth low-l
field. Complex-symmetric positive-definite Gram (Ψᴴ W Ψ ≈ R²·I on a
quasi-uniform mesh), Cholesky-factored once (frequency-independent).

Convention-robustness (the reason this rig is safe to build fast):
m_k[b] and E_k are evaluated with the SAME ψ code — any normalization
/ Condon-Shortley / conjugation slip cancels identically in δ. The
only requirements are: the basis spans the low-l vector fields (gate:
Gram ≈ R²·I on a dense grid) and the exact integral converges (gate:
two-grid agreement).

## 3. Exact-moment quadrature (the cancellation is here now)

E_k inherits the physical cancellation (condition ~1e4, Codex round
2), so the quadrature must resolve u_inc's epicentral peak (width ~
source depth d) to ~1e-6 relative; smooth high-order quadrature gives
far more. Grid: source-frame product grid, azimuth uniform (source-
frame m-content of a MT field ≤ 3; trapezoid is spectrally exact),
colatitude-from-source in geometrically graded Gauss-Legendre panels
with first panel width d/R (edges 0, a, 2a, 4a, … π; ~11 panels × 16
nodes; base grid ~176×128 ≈ 23k points, refined tier 24 nodes ×
halved panels × 256 azimuths for the convergence check). u_inc on the
grid via the same u0eM/greens_deri_src path as the solve. Per
frequency this is a few 10⁴ Green evaluations + one tall matvec —
negligible next to assembly.

## 4. Gates

1. Basis: Gram(dense uniform grid) = R²·I to ~1e-10; ∂θY vs central
   differences; C = r̂×B orthogonality.
2. Machinery: an analytic single-harmonic field returns E = R²·e_k;
   fit reproduces requested moments to machine precision; δ=0 → Δb=0.
3. No-op: ARB_MFIT_LMAX unset touches no code path (structural) +
   momentfit hook with lmax=None returns b unchanged.
4. Physics A/B (propagating preservation): 637-km campaign
   (dsm_arbitration_ref637 DSM reference reused) with fit ON must not
   degrade 11.10% / 0.9728; expected ‖Δb‖/‖b‖ ≪ 1 there because
   sampled moments of a smooth trace are already O(h²)-accurate.
   This doubles as the strongest convention check: exact vs sampled
   moments must agree to O(h²) where sampling is trustworthy.
5. ACCEPTANCE (the falsification test): 50-km campaign
   (dsm_arbitration_src50_ref2 DSM reference reused, hmin-20 mesh,
   ARB_MFIT_LMAX=16) — diagnosis is CONFIRMED if the mrt T quasi-
   static channel amplitude ratio moves 0.0075 → O(1) with high
   correlation; any principled partial recovery (basis truncation at
   L=16 caps the recoverable static content — the mesh supports
   l ≲ 17) is quantified per station against the DSM static fraction.

## 4b. Codex design review (2026-07-11, folded in)

Verdict: algebraically sound (b_fit = (I-P)b + Psi G^-1 E with P the
W-orthogonal projector — the high-l complement is untouched EXACTLY);
strong as positive confirmation, but a NEGATIVE result is ambiguous
without extra gates. Adopted:

* Primary observable = the ON-OFF difference Δu = u_fit − u_raw
  (= A⁻¹δb, exactly linear in the correction). Report per
  station/component the alignment C = |⟨r,Δu⟩|/(‖r‖‖Δu‖) and gain
  g = ⟨r,Δu⟩/⟨r,r⟩ against the DSM residual r = u_DSM − u_raw.
  Even with high-l operator re-injection left in the absolute
  waveform, C→1, |g|~1 confirms the RHS diagnosis.
* L-sweep 8/12/16: recovery must stabilize with L (recovery only at
  L=16 or oscillating with L is suspect).
* Gram audit on the ACTUAL graded mesh: eigenvalue range of G/R²
  (dense-grid Gram=I validates only the basis code); usable if
  κ₂ < 10 and results L-stable.
* l ≤ 16 does NOT exhaust a 50-km source's static field (e-fold
  l ~ R/d ≈ 127; (1−d/R)^16 ≈ 0.88): partial recovery at near
  stations must be judged against a DSM low-pass (l≤16) target —
  a maxlmax-truncated DSM rerun is the clean arbiter if needed —
  not against total DSM amplitude.
* Known residual assumptions (acceptable for a falsification rig,
  listed for honesty): the discrete solve's left low-l action is
  taken as area-weighted incenter sampling (the true functional is
  A⁻ᴴWq); incenter one-point face quadrature is not formally 2nd
  order (centroid is); lh/R ≈ 1.16 at l=16 is outside the asymptotic
  regime. Upgrade path if ambiguous: exact per-face integrals
  M_{k,f} = ∫_{T_f} ψ̄ dS instead of incenter sampling, and the
  b_E/b_H split-solve leakage test χ = ‖O_L A⁻¹b_H‖/‖O_L A⁻¹b_E‖.

## 5. Expected limits (written pre-run)

L is capped by the coarse-mesh resolution (h ~ 460 km => l <~ 17), so
stations very near the source (static field needs l ~ R/d) may stay
deficient. This rig cannot exceed the mesh's low-l representation;
full closure was expected from rung C (scattered-field RHS) / rung A
(spectral). If mrt recovers to O(1) with L=16 the diagnosis stands;
if it does NOT recover despite exact moments, the diagnosis needs
revision (that is the falsification arm — and it fired, see below).

## 6. RESULTS (2026-07-11)

Rig gates (all passed): M1 basis Gram vs identity 1.8e-14; M2 dtheta
FD 4e-10; M3 graded-grid exact moments 1.9e-14 (incl. l=16 column);
M4 fit residual 1.2e-15. Full 15-test battery green. Gram cond on the
ACTUAL meshes: 1.10 (both 1704-face ref45 and 2151-face ref20).
Quadrature two-tier agreement 1e-12..1e-14 of max|E| at every
frequency. mfit legs: validation/mfit_campaign.txt;
dsm_arbitration_ref637_mfit/, dsm_arbitration_src50_mfit/.

GATE 4 (637 km, healthy DSM): moment fit ON -> median rel RMS
9.09e-2 / max 1.95e-1 / min corr 0.9890, vs 1.110e-1 / 3.19e-1 /
0.9728 without. NOT ONLY preserved but IMPROVED — the fit removed
the sampled-moment corruption (deficit 2-5% there) that had been
identified as the mrr static-content misfit pattern. |db|/|b| ~ 2.7%.

GATE 5 (50 km): absolute waveforms unchanged (median 83.9% vs 84.8%
raw); ON-OFF analysis (dsm_arbitration_src50_mfit/mfit_alignment.json)
showed du 20-600x SMALLER than the DSM residual on all quasi-static-
dominated channels -> low-(l<=16) forcing does not carry the missing
field. Mandated bug hunt on the reference followed:

* l-truncated tish rebuilds (maxL patch; untruncated rebuild matches
  the campaign spc to 1e-13): l<=16 holds 2-11% of the full T norm;
  truncations at l = 128..16384 ALL leave O(1) rel err.
* Band-increment audit: physical excitation bound (2l+1)(r0/R)^l
  falls ~50 orders between l=512 and l=16384; measured DSM T band
  increments stay FLAT (~1e-6..4e-5 m) out to l>16384, and at ST00
  the (16384, full] increment EQUALS the full signal norm. The sum
  is a roundoff random walk, not physics.
* Independent scale check (no l-sums): DSM T = 14-700x the analytic
  full-space incident field at the stations; a free surface amplifies
  by O(1). The BEM total T = 0.16-0.48x incident — physical class.

VERDICT: the "dominant quasi-static mrt T channel" of the 2026-07-11
closure was an artifact of DSM tish's shallow-event path (engaged for
depth < 100 km) at ULP — unconverged high-l noise. The BEM was never
missing a 10-100x physical channel at mrt T. What survives of the
old diagnosis: the sampled low-l RHS projections ARE noisy (directly
measured) and the moment fit repairs them (gate 4's improvement);
DSM references with sources DEEPER than 100 km (all prior rungs, 637
km legs) never enter the shallow path and remain valid.

Follow-up audit (validation/dsm_audit/, figure
tish_noise_evidence.png, regenerate with analyze_tish_noise.py):

* tipsv (PSV) CERTIFIED converged for BOTH sources: band increments
  track the physical bound ((1024,4096] ~ 1e-10 = bound; identical
  zero beyond l=4096); rebuilds match campaign spc to 1e-14.
  tish for mrr is identically zero (no toroidal excitation, no noise
  seed). => mrr reference fully valid; mrt Z (SH-free) valid.
* tish re knob is INERT for this configuration (re=1e-4 and 1e-5
  outputs BIT-IDENTICAL to re=1e-3, enlarged-maxNGrid builds): at
  ULP + shallow-scan lmax the whole zone is in the evanescent branch
  (kzAtZone=0 -> minimal grid). No configuration fix exists.
* The zero-group-delay "quasi-static pulse" is exactly the Ricker
  source-time-function replayed by frequency-flat noise (a flat noise
  spectrum synthesizes to the STF at its center time, no moveout).

Valid-channel re-verdict at 50 km (hmin-20 graded mesh):

* mrr VEC (all comps valid): median rel 0.596, hump 0.15 (20 deg) ->
  0.70-0.72 (75-135 deg) -> 0.36 (170 deg); amp 1.10-1.20; corr
  0.78-0.99. Moment fit: no change (0.607) — correct, those l<=16
  moments were already fine on this leg.
* mrt Z (valid): BEM amp 0.21-0.32 of reference UNIFORMLY, corr
  0.81-0.90; 637-km mrt Z amp is 1.03 with the same conversion code,
  so NOT an MT-convention factor — a real shallow-source deficit.
* mrt T/horizontals: NO valid reference exists (tish noise); the BEM
  T is 0.16-0.48x the analytic incident field (physical class), but
  its accuracy is UNMEASURED pending a semi-analytic toroidal
  reference (closed-form per-l radial solutions, stable q^l sum).

These two real deficits (mrr mid-distance hump, mrt-Z ~4x) — both
untouched by moment fitting — are the actual shallow-source targets
for the rung-C scattered-field formulation and/or near-source
resolution work, with the semi-analytic reference as the T-channel
acceptance anchor.
