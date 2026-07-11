# Fluid–fluid interfaces: derivation + implementation plan (rung-4 candidate)

Status: IMPLEMENTED 2026-07-11 (domains.py / elimination.py /
layered.py, new block kind "un"). Gates G0-G4 PASSED
(validation/ff_gate.txt, tests/test_ff.py): full existing battery
unchanged; transparent split 7.0e-4 (gate 8e-2); registration swap
1.5e-16; A_(+) + A_(-) = I to 1.4e-21; dense==eliminated 1.8e-14 -
5.2e-14 on 4/5-layer fluid-fluid stacks incl. the middle-fluid
own-modulus _scale fallback and a rho 9.0-vs-4.0 contrast; cache
bitwise, 12/24 blocks recomputed as expected. G5 (DSM arbitration,
graded-OC staircase) in dsm_arbitration_ff/. Derivation history:
written 2026-07-10, adversarially checked by Codex before
implementation (section 8).
Goal: graded (staircase) outer core and, later, sub-surface oceans —
stacks of homogeneous fluid shells separated by fluid–fluid interfaces.

## 1. Conventions as coded (verified against source)

* Time convention `e^{-iwt}`; acoustic Green function
  `G_p(x,y) = e^{+i k_p r}/(4 pi r)`, `k_p = w/vp`
  (pyastroseis/greens.py:358-374). `e^{+ikr}` + `e^{-iwt}` = outgoing.
  (dsm_arbitration/synthesize_compare.py applies `conj` at synthesis to
  match the DSM convention; irrelevant to internal consistency here.)
* Acoustic operators (pyastroseis/assembly.py):
  * `A = cal_A_st(faces1, faces2, ...)`: `A[j,i] = ∫_{panel i}
    dG_p/dn_y (x_j, y) dS_y`, with n_y = the SOURCE panel normals of the
    ORIENTED column mesh (passed as `n1a,n2a,n3a` from the hoisted
    Geometry). Self entry = `CPV + 1/2`: `int_self_A` returns
    `CPV − 1/2` (assembly.py:295) and `cal_A_st` adds `+1.0`
    (assembly.py:521).
  * `B = cal_B_st`: `B[j,i] = ∫_{panel i} G_p dS_y` (weakly singular
    self entry = numerical part + analytic disk term,
    assembly.py:298-308). No normals enter B.
* Fluid momentum under `e^{-iwt}`: `rho * d2u/dt2 = −grad p` gives
  `grad p = rho w^2 u`, hence `dp/dn = rho w^2 (u . n)`.
* Interior Green representation of a bounded fluid region Omega with
  outward normal n, x on the (smooth) boundary:

      (1/2) p(x) + CPV ∫ p dG/dn dS − ∫ G dp/dn dS = p_inc(x)

  i.e. in operator form, per collocation interface:

      A p − rho_F w^2 B (u . n_out) = p_inc

  This is EXACTLY the assembled p-row (pyastroseis/domains.py:281-302):
  `A p / scale − sc * rho w^2 B (Smat u) / scale`, using
  `u . n_out = sc * (Smat u)` (sc = region's registration sign of the
  column interface; comment at domains.py:298-299). The `+1/2` diagonal
  jump in A and the `−rho w^2 B` sign are mutually consistent with
  `e^{+ikr}` / `e^{-iwt}`. ✓
* Fluid–solid coupling seen from the solid rows: `t = −p n_out(solid)
  = −sc_solid * p n_canonical`, contributing `+sc_solid * G Smat^T p`
  (domains.py:258-265).
* Orientation bookkeeping: one oriented mesh + hoisted Geometry per
  (interface, sign) (domains.py:189-199). A region registering an
  interface with sign −1 collocates and integrates on the FLIPPED
  Faces object; the `faces1 is faces2` self-block fast path still fires
  because both row and column meshes come from the same
  `_oriented[(iface, sign)]` object. So each fluid side gets the
  correct `+1/2` jump with respect to ITS OWN outward normal without
  any extra sign.

## 2. Fluid–fluid interface: physics and unknowns

Interface `Gamma` (n panels, canonical normals `n_c`) between fluid
regions `F_a` (registered first, orientation sign `s_a` on Gamma) and
`F_b` (`s_b = −s_a`; enforced at build time like welded/fluid_solid).

Matching conditions (inviscid): `[p] = 0` and `[u . n] = 0`;
tangential slip is free. There is NO condition on `dp/dn` — it jumps by
the density ratio: `dp/dn|_a / rho_a = w^2 u.n = dp/dn|_b / rho_b`.
This is why the second unknown must be normal DISPLACEMENT, not the
normal derivative of p.

Unknowns on Gamma:
* `p` — n values, the existing `("p", Gamma)` block kind (shared →
  pressure continuity automatic).
* `q := u . n_canonical` — n values, NEW scalar block kind
  `("un", Gamma)` (shared → normal-displacement continuity automatic).

A 3-vector `("u", Gamma)` block would be RANK-DEFICIENT: tangential
displacement never enters any fluid equation, leaving a 2n-dimensional
null space. The unknown must be scalar. No `Smat` is needed for q
(the projection is baked into the definition); no column scaling is
needed (q has the same units/magnitude as the u unknowns, which are
unscaled).

## 3. Row equations

Every fluid region F contributes its representation equation
collocated on each of its bounding interfaces (as today). On
collocation interface `Gamma_r` (orientation `s_r`), the column terms
per bounding interface `Gamma_c` with sign `s_c` are:

* `Gamma_c` fluid_solid: `A p − s_c rho_F w^2 B (Smat u)`  (existing)
* `Gamma_c` fluid_fluid: `A p − s_c rho_F w^2 B q`          (NEW)

using `u . n_out = s_c q` on the fluid–fluid interface. A and B are
evaluated on the region's own oriented meshes, exactly as now. RHS:
`p_inc` traces if the source is in F — zero for now (fluid-region
sources are a separate rung; DSM tipsv refuses them anyway, measured
2026-07-09).

Row-block assignment mirrors the welded convention: the
FIRST-registered fluid's equation fills the `("p", Gamma)` rows, the
SECOND-registered fluid's equation fills the `("un", Gamma)` rows.
(In `nested_shell_model_from_ifaces`, regions register outermost-first,
so outer fluid → "p" rows, inner fluid → "un" rows — the same pattern
as u/t rows on welded interfaces.) Count balance per fluid–fluid
interface: 2n unknowns, 2n equations. ✓

Row scaling (pure row equilibration, no effect on the solution):
`("p", Gamma)` rows divided by `_scale[F_first]`, `("un", Gamma)` rows
by `_scale[F_second]`; `assemble_rhs` must apply the same division to
"un" rows.

Both fluid sides use the SAME shared p column block: each side's A
carries its own orientation inside the kernel (flipped mesh normals
for the −1 side), so no explicit sign appears on the p columns —
identical to how the existing fluid-annulus inner boundary works.
Within the FLUID–FLUID rows the explicit `s_c` appears only on the q
columns (B has no normals). (Scoped claim: elsewhere in the assembler
explicit signs do sit on p columns — the fluid_solid `t = −p n`
substitution in solid rows carries `sc`, domains.py:265.) Useful A/B
identity for the swap gate: for identical wavenumbers,
`A_(+) + A_(−) = I` (the CPV part flips sign, both keep the +1/2).

Conditioning note: every entry of a q column is proportional to
`w^2 B` — unlike u columns, which also enter solid rows through O(1)
jump-carrying T entries at fluid_solid interfaces. This is a uniform
column scaling (harmless to LU with partial pivoting at fp64), and w
always carries the positive imaginary part `omegai` at solve time
(run_bem.py:91; solver.py:59), so |w| never vanishes. If it ever
matters at very low bands, switch the unknown to the density-
normalized flux `h := w^2 q` — a pure column rescaling (second-
opinion suggestion). G3 below records condition estimates.

## 4. Implementation deltas

pyastroseis/domains.py:
1. `FLUID_FLUID = "fluid_fluid_ff"` — new condition constant (name
   TBD; must not collide with `FLUID_SOLID`). `nested_shell_model`:
   lift the adjacent-fluid `NotImplementedError` (domains.py:404-407)
   and emit the new condition when `materials[i].fluid and
   materials[i+1].fluid`.
2. Adjacency checks: fluid_fluid needs exactly two fluid sides with
   opposite signs and no solid side. Drop the blanket `len(flu) > 1`
   NotImplementedError (domains.py:139-142). Store
   `_fluid_b_of[Gamma] = flu[1]`.
3. `p_ifaces`: currently appended once per adjacent fluid region
   (domains.py:126-129) — a fluid–fluid interface would be appended
   TWICE. Dedup (seen-set), keeping first-appearance order.
4. `u_ifaces`: currently EVERY interface gets a 3n u block
   (domains.py:130-132). Exclude fluid–fluid interfaces (no vector
   displacement unknown there). New `un_ifaces` list; block layout
   `[p blocks; u blocks; t blocks; un blocks]` (layout is free — the
   dense scatter uses `_slices` and the elimination groups by
   interface object).
5. `_smat`: only needed for interfaces where the `Smat u` coupling
   exists — fluid_solid. A fluid–fluid interface needs NO Smat. The
   current `self._smat = {i: smat_func(i.faces) for i in p_ifaces}`
   must not require one for pure fluid–fluid interfaces (or may build
   it harmlessly; decide at implementation, but do not index it in the
   new branch).
6. `_scale`: TWO gaps (second confirmed by the Codex review).
   (a) The current lookup takes "the adjacent solid" from the fluid
   region's FIRST interface (`self._solid_of[freg.interfaces[0][0]]`,
   domains.py:213) — for a middle shell of a graded OC every interface
   is fluid–fluid and this KeyErrors.
   (b) The builder iterates `p_ifaces` and only ever scales
   `_fluid_of[iface][0]` (domains.py:205-216) — the SECOND fluid of a
   fluid–fluid pair (e.g. the innermost fluid, owner of the "un"
   rows) may never receive a scale at all → KeyError at row assembly.
   Fix: build a scale for EVERY fluid region; keep the legacy
   solid-based formula whenever the old lookup succeeds (bitwise
   no-op on existing models), else fall back to the fluid's own P
   modulus, `scale = sqrt(rho_f * lamda_f) * w0` (the `rho*c*w0` form
   with c = the fluid's own vp).
7. `_block_entries`: refactor the fluid p-row body into a
   `_fluid_entries(reg, ifr, sr, w, row=None, skip=None)` generator
   (mirror of `_solid_entries`, including the `keep()` guard for
   rung-3 caching), emitting per bounding interface:
   `("p", ifc): A/scale`; then either
   `("u", ifc): −sc rho w^2 (B @ Smat)/scale` (fluid_solid) or
   `("un", ifc): −sc rho w^2 B/scale` (fluid_fluid).
   "p" rows ← `_fluid_of[ifr]` (first fluid), "un" rows ←
   `_fluid_b_of[ifr]` (second fluid).
8. `assemble_rhs`: "un" rows are the second fluid's equation — divide
   by `_scale[_fluid_b_of[iface][0]]`.

pyastroseis/elimination.py:
9. `ShellElimination` groups by interface object (elimination.py:41)
   — kind-agnostic, no change. `_structural_keys` (elimination.py:172)
   grows the mirrored branches: for "p"/"un" rows (fluid region), per
   bounding interface emit `("p", ifc)` plus `("u", ifc)` when
   fluid_solid / `("un", ifc)` when fluid_fluid. Solid rows unchanged
   (a solid region never touches a fluid–fluid interface).
   Tridiagonality is preserved: regions still couple only adjacent
   interfaces. `cached_blocks` is key-based and works unchanged.

pyastroseis/layered.py:
10. `incident_solid_layer_source`: emit `np.zeros(n)` for "un" rows
    (currently the else branch assumes 3n). Same for any other RHS
    builder that enumerates blocks.

## 5. Non-uniqueness / fictitious frequencies (CORRECTED)

The first draft claimed bounded regions are immune to fictitious
resonances. That is WRONG — refuted by the Codex second opinion with
an explicit counterexample: take equal wavenumbers on both sides of
Gamma and a Dirichlet eigenfunction `phi` of the inner subdomain
(`phi|_Gamma = 0`, `h_D := d(phi)/dn != 0`). The single-layer
potential of `h_D` vanishes on Gamma AND identically outside the
inner domain, so `(p = 0, q = h_D / (rho_inner w^2))` is a nontrivial
null vector of the coupled block system at that real eigenfrequency —
a fictitious interface resonance, not a physical mode of the
assembly.

Practical status: every solve uses complex frequency
`w = 2 pi f + i omegai` (run_bem.py:91; solver.py:59, matching DSM)
AND complex attenuated velocities (Q), which move these nulls off the
solve contour; and at the current band kR of the entire OC is ~1.26 at
band max, below the first spherical Dirichlet eigenvalue kR = pi, so
the first null is not even reachable. Treat as a RISK when pushing
frequency, not a blocker now: G1 below gains a frequency scan near an
inner-subdomain Dirichlet eigenfrequency with condition-number
logging. Robust remedies if ever needed: merge identical adjacent
fluids; combined (Burton–Miller/Calderon) traces.

## 6. Gates (ladder order)

* G0 strict no-op: full existing battery (rungs 0-3) bitwise
  unchanged — new code paths only trigger on the new condition.
* G1 transparent fluid–fluid: fluid core split by an interposed
  sphere, identical material both sides → surface solution matches the
  single-core model (rung-1-style tolerance; thin-shell rule
  h <= 2t applies to the sub-shell thicknesses). At higher-band
  settings, extend with a frequency scan across an inner-subdomain
  Dirichlet eigenfrequency (fictitious-resonance probe, section 5),
  logging condition estimates.
* G2 A/B swap: swap the two fluids' registration order → same
  solution to ~1e-10 (permuted system). Also check the analytic
  identity A_(+) + A_(−) = I on the shared interface (section 3).
* G3 elimination: dense vs ShellElimination on a >=4-layer model
  containing a fluid–fluid interface, IDENT 1e-10; record condition
  estimates of the diagonal groups (q-column w^2 scaling note,
  section 3). Include a strong density-contrast variant.
* G4 cache: perturb a boundary adjacent to fluid–fluid blocks →
  bitwise reuse of untouched blocks, correct recompute set.
* G5 physics: graded-OC staircase — corefluid-style model with the OC
  split into 2 and 4 PREM-volume-averaged sub-shells vs DSM tipsv
  running the SAME staircase model; mantle source at the validated
  637.1-km depth, mrr. Measure the per-fluid-interface error
  increment (expect at or below the +6%/welded-interface law, since
  each fluid interface carries 2n unknowns and scalar kernels).

## 7. Cost

Per fluid–fluid interface: 2n unknowns (vs 6n welded); largest new
blocks are n x n scalars (vs 3n x 3n). Splitting the OC into 4
sub-shells at n ~= 600/interface adds ~3.6k unknowns and only scalar
kernel evaluations — negligible against the solid blocks. Fluid
sub-shells are the CHEAP way to grade the OC.

Explicitly out of scope (later rungs): fluid OUTERMOST layer (free
fluid surface / ocean top), sources inside fluid regions, and
gravitational restoring terms (neither the BEM nor the DSM legs used
here include self-gravitation; both sides omit it consistently).

## 8. Second opinion (Codex, 2026-07-10)

Codex (read-only sandbox, reviewing the multilayer branch) AGREED,
via independent derivation, with: the section-1 sign chain
(G = e^{+ikr}, interior trace of the double layer = K − I/2, momentum
grad p = rho w^2 u, hence A p − rho w^2 B (u·n_out) = p_inc with
diag(A) = CPV + 1/2, matching domains.py:279-302); the section-3
jump/orientation analysis; and the p + q unknown choice (raw shared
dp/dn confirmed wrong under density jumps; h = w^2 q noted as a
valid rescaled alternative). CONFIRMED errors, fixed in this
revision: the original section-5 fictitious-frequency claim
(counterexample above) and the second-fluid `_scale` gap
(section 4.6b). Its risk list is folded into the gates: w -> 0
conditioning, identical-fluid transparency + eigenfrequency scan,
strong impedance contrast, A/B swap with the A_(+) + A_(−) = I
identity, dense-vs-elimination + cached-block reuse, and the existing
liquid-core bitwise regressions.
