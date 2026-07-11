# PyAstroSeis multi-layer roadmap

Goal: generalize the BEM solver from its two hard-wired configurations
(homogeneous body with a free surface; solid shell + liquid core) to N
regions separated by arbitrary closed interfaces, so that boundary
perturbations of layered reference models (PREM / AK135-style onions,
or multi-layer asteroid interiors) can be simulated directly.

## Formulation

The existing code already contains two instances of multi-domain BEM:

- homogeneous solver: 1 solid region, 1 interface with a free-surface
  condition; system `T u = u0`.
- liquid-core solver (`liquidcore.liq_core`): 2 regions (solid shell,
  fluid core), 2 interfaces (free outer surface, fluid–solid core
  boundary); coupled system in `x = [p; u_core; u_surf]`.

The generalization makes regions and interfaces first-class
(`pyastroseis/domains.py`):

- **Region**: homogeneous material (lambda, mu, rho, Q, qp_fac; fluid
  or solid) plus its bounding interfaces, each with an orientation
  sign (+1 if the interface mesh's stored normals point out of the
  region). Each solid region contributes one boundary-integral
  equation per bounding interface, built from the existing T/G
  kernels; each fluid region contributes one acoustic equation per
  bounding interface, built from the existing A/B kernels.
- **Interface**: a closed `Faces` mesh plus a condition:
  - `free` — zero traction, one adjacent solid region (t = 0, so no
    G-term; unknown: u).
  - `fluid_solid` — one solid + one fluid side; unknowns u (solid
    displacement) and p (fluid pressure); coupling via the normal
    projection Smat exactly as in `liq_core` (convention: stored
    interface normals point out of the fluid).
  - `welded` (rung 1) — two solid sides; continuity of u and t;
    unknowns u and t (6N per interface), with the second solid's
    equation providing the extra 3N rows.

Row/unknown layout rule: each unknown block ("p" or "u" on an
interface) is paired with the equation of the adjacent fluid (for p)
or solid (for u) region collocated on that interface. This reproduces
the existing solvers' orderings as special cases.

## Rung ladder (each rung gated before the next starts)

**Rung 0 — abstraction, zero new physics.** DONE 2026-07-09.
`domains.MultiDomainModel` assembles the general system. GATE: for
the 1-region and 2-region configurations the assembled matrix, RHS,
and solution are BITWISE identical to `cal_traction`/`liq_core`
(`tests/test_rung0.py`), in both production mode (physical Qp, polar
self-quadrature, adaptive far quadrature) and MATLAB-exact mode
(legacy, grid, full). Passed on synthetic spheres and the real
my_mesh / my_mesh_lc meshes (validation/rung0_gate.txt). All the
refactoring risk lives here and is fully covered by the existing
oracle/regression harness plus this gate.

**Rung 1 — welded solid–solid interfaces.** Implemented 2026-07-09
(branch `multilayer`). Unknowns on a welded interface: shared u plus
canonical traction t = sigma . n_canonical, stored column-scaled by
rho*c*w0; each solid's equation carries -sign * G * scale on the t
columns; first-registered solid's equation fills the ("u", iface)
rows, the second's the ("t", iface) rows. LOCAL GATES
(`tests/test_rung1.py` + rung-0 bitwise regate,
validation/rung1_gate.txt): (a) transparent interface — artificial
internal boundary, same material both sides, must reproduce the
single-region solution AND the error must converge to zero under
interface refinement (it is piecewise-constant-element
discretization error, ~4.5e-3 at 384 faces / 5.7 elem per S
wavelength, ~1.3e-2 at 80 coarse faces); (b) A/B swap — exchanging
which region owns the u-rows vs t-rows must leave all physical
fields unchanged (observed ~1e-15); (c) contrast-core interface
refinement must converge. DSM ARBITRATION (rung-1 closure) PASSED
2026-07-09 (dsm_arbitration/): Earth-scale two-layer solid sphere
(R=6371 km, welded interface at 3185.5 km, shell vp6/vs3/rho3, core
vp8/vs4.5/rho4, elastic) vs DSMsynTI-mpi tipsv+tish using the vetted
DFDM-campaign protocol (identical complex frequencies, identical
spcsac synthesis, Ricker f0=1.75e-4 Hz, 12 face-incenter stations,
mrr + mrt moment sources exercising both spheroidal and toroidal
coupling). Verdict over the 0-24 ks P/S/R1 window: welded two-layer
median rel RMS 7.3% / min corr 0.955 / amp ratio 0.989 — BELOW the
homogeneous-BEM baseline floor of 12.7% (which measures pure mesh
dispersion + station sag, no interface), so the welded machinery adds
no error above the single-region floor. Record sections + magnified
error panels in dsm_arbitration/waveforms_*.png. Conventions found:
AstroSeis spectra are the complex conjugate of DSM's e^{+iwt}
convention; BEM moment amplitude matches DSM exactly (constant-1
ratio) once MT units are handled (1 DSM input unit = 1e18 N*m).
Known BEM traits quantified on the baseline leg: lowest harmonics
(w*R/vs < ~0.3) carry rigid-mode-amplified noise (negligible under a
band-limited source); late elastic coda dephases at the mesh's
eigenfrequency error.

**Rung 2 — N nested shells + config schema.** DONE 2026-07-09
(branch `multilayer`). `domains.nested_shell_model` (innermost-first
layers; consecutive solids welded; fluid allowed innermost only) +
`layered.py` YAML schema (examples/config_threelayer.yml) +
`layered.incident_outer_source`. GATES (tests/test_rung2.py): the
2-solid-layer and fluid-core special cases reproduce
welded_two_layer_model / liquid_core_model BITWISE; 3-shell
transparent (two artificial interfaces) 3.98e-2 on coarse synthetic
meshes (~ the sum of the rung-1 single-interface errors); fluid-core
+ artificial-weld vs 2-region liquid core 3.65e-2; fluid-not-
innermost rejected. DSM ARBITRATION (dsm_arbitration_r2/, PASSED):
Earth-scale three-layer sphere (interfaces 2200 / 4300 km) vs
tipsv+tish — per-station 3-component vector metrics over the 0-24 ks
window: median rel RMS 12.2%, min corr 0.971, amp ratio 0.981, below
the homogeneous baseline floor (14.5%, 0.965) computed on the same
meshes/stations. (Protocol note: gates use VECTOR metrics because
isolated near-nodal components produce meaningless relative errors —
observed equally in the baseline leg.) Unknown count grows by ~6N
per welded interface; dense solves stay practical to roughly 20–30k
unknowns on one node.

**Rung 2b — fluid annuli + auto-mesh + boundary perturbation.**
Implemented 2026-07-10 (branch `multilayer`).

*Fluid annuli (PREM outer core):* the fluid_solid machinery is
orientation-generalized — a fluid may register an interface with
sign=-1 (its inner boundary): the solid-side pressure coupling
becomes sc * G Smat^T (traction from the fluid w.r.t. the region's
outward normal, t = -sc * p n_canonical) and the fluid-side B
coupling carries the fluid's sign (u . n_fluid_out = sc * Smat u);
both reduce bitwise to the original liq_core expressions for the
canonical orientation. nested_shell_model now accepts a fluid layer
anywhere below the surface (still rejected: fluid outermost, two
adjacent fluids). LOCAL GATES (tests/test_rung2b.py +
rung-0/1/2 regates, validation/rung2b_gate.txt): rungs 0-2 stay
bitwise; tiny-core limit — solid(r->0)/fluid/solid collapses onto
the 2-region liquid-core solution at rel err 4.7e-5; annulus
refinement — refining the two internal meshes (surface fixed) moves
the surface field toward the finest level, err 1.26e-1 -> 4.3e-2
(ratio 0.34; campaign rule: a non-converging result means hunt for
bugs, not thresholds).

*Auto-mesh from the 1-D model:* per-layer `nmesh` is now optional —
`layered.auto_nmesh` sizes every boundary mesh from the input 1-D
model via the wavelength rule (face count F = 8*nmesh - 16, mean
face size h = sqrt(4 pi r^2/F), h <= v_min/(fmax*epw) with v_min the
slowest adjacent wavelength speed, vs — or vp for a fluid). With
epw=10 it reproduces the hand-picked Earth-campaign surface mesh
(n=199 vs 200). YAML: top-level `fmax` (+ optional `epw`).

*Boundary perturbation:* per-layer `perturb` spec adds radial relief
to that layer's outer boundary — {type: ylm, l, m, amp} (peak-
normalized real Y_lm) or {type: random, lmax, amp, seed} (band-
limited random relief); meshgen.gen_mesh_relief mirrors the standard
sphere construction path so amp=0 is BITWISE the unperturbed mesh
(strict no-op gate), and the surface-solution change scales linearly
in amp (measured ratio 2.00 for amp doubling). This is the geometry
front end for rung 3.

DSM ARBITRATION (dsm_arbitration_r2b/): PREM-topology Earth — solid
inner core (1221.5 km) / fluid outer core (3480 km) / mantle vs
tipsv+tish (tish model = solid zones above the fluid, standard DSM
CMB convention; source at 637 km depth in the mantle). The scale-1
legs FAILED the min-corr gate, and the ensuing bug hunt (2026-07-10)
established the cause is NOT a coding defect but a measured,
converging discretization mode:

* Signature: direct arrivals, first core reflections and amplitudes
  match (amp ratio 1.00); the error grows with time — progressive
  dephasing of the trapped-mode coda, BEM ~0.8-1% fast at
  reflector-mesh h/R 0.089.
* Mechanism (0T2 control, diag_toroidal.py): even the homogeneous
  rung-0 solver puts the discrete 0T2 toroidal eigenfrequency HIGH
  by +5.87 / +2.93 / +1.59 % at surface h/R = 0.261 / 0.181 / 0.127
  — order 1.9 in h/R, extrapolating to +0.84% at h/R 0.089, exactly
  the observed drift. Every mode-CONFINING boundary contributes:
  free surfaces and fluid-solid interfaces (near-perfect
  reflectors); welded interfaces are transmissive and benign (their
  campaigns passed at interface h/R up to 0.21).
* Consistent evidence: internal-mesh x4 refinement halves the PSV
  excess but leaves SH untouched (SH modes are confined by the
  surface + CMB, and the CMB-refined legs improve only PSV);
  per-harmonic spectral errors cluster around elastic resonance
  peaks (frequency-bias signature), not isolated bins.
* Side finding: the fluid-coupled system has sparse spurious
  pressure-block resonances on the REAL frequency axis (classic
  acoustic-BIE non-uniqueness; residues ~1e9 in the response scan).
  Physical solves at Im w = omegai sit off the axis and are
  unaffected, but resonance-style post-processing must avoid them.
* Consequence baked into the code: auto_nmesh now applies curvature
  floors h/R <= 0.09 (reflector boundaries) / 0.2 (welded) on top
  of the wavelength rule.

Ladder of legs (all in dsm_arbitration_r2b/, 0-24 ks vector
metrics, homog floor 14.5% / corr 0.965 at scale 1): corefluid (ICB
n12 / CMB n55) 34.8% corr 0.772; corefluid_fine (internal x4) 20.5%
corr 0.848 amp 1.0005; corefluid2 (mantle == baseline material, CMB
n61) 43.6%; corefluid3 (reflectors at h/R 0.089) 31.0% corr 0.830.
Scale-2 run (dsm_arbitration_r2b_hi, job 55744328): homog floor
5.90% corr 0.9954 (converging on schedule), corefluid3 19.6% —
ratio+amp gates pass, but min corr PINNED at 0.831 and the spectral
error concentrates in resonance clusters at FIXED physical
frequencies whose BEM peaks sit ~+1.6% high at every tested
resolution (e.g. DSM peak k=123 vs BEM k=125).

RESOLUTION OF THE HUNT (rung 2c, 2026-07-10). An earlier
intermediate conclusion — "the fluid-solid coupling converges at
order ~1, an inherited accuracy limit" — was WRONG and is
retracted: it was an artifact of the measurement instrument, not
of the solver. A random-vector resonance scan in a dense spectrum
locks onto whichever nearby (2l+1)-degenerate multiplet responds
strongest, so successive resolutions tracked DIFFERENT modes
(diag_radial.py first version; the free-sphere control even
appeared to diverge). With a mode-SELECTIVE scan (l=0 symmetric
drive + projected response, diag_radial2.py) every family
converges at order ~2 in h/R:
  free sphere radial:  +2.31 / +1.09 / +0.53 %  (order 2.05)
  LC fluid-coupled l=0: +1.87 / +0.92 / +0.48 % (order ~2.0)
at h/R = .261/.181/.127 — same class as the toroidal control, with
additive per-boundary contributions of either sign. All operator
consistency checks pass (null-space identities ~1e-4; near-pair
quadrature exonerated by a 16-fold-subdivided near tier changing
+5.513% -> +5.511%; coupling normals exonerated).

Earth-scale closure: corefluid4 (ICB raised to the curvature floor,
all reflectors h/R 0.089, 17.4k unknowns) reproduces corefluid3 to
four digits (30.96% / corr 0.830) — the ICB is exonerated; the
residual excess lives on the SURFACE + CMB curvature and follows
the order-2 law across scale 1 -> 2 (excess 27.4% -> 18.7% with
h/R 0.089 -> 0.063; apparent order < 2 because the waveform metric
saturates once components fully dephase). Extrapolation: the
min-corr 0.9 gate on a LOSSLESS 24-ks multi-orbit coda needs
reflector h/R ~ 0.045, i.e. ~25k-face boundaries and ~60k dense
unknowns per frequency at Earth scale — beyond practical dense
solves. This is not a solver defect: an all-elastic (Q=1e8)
trapped-coda comparison against a semi-analytic 1-D reference
demands mode frequencies to ~0.2%, the harshest possible metric
for any discretized method; transmissive (welded) models pass the
same gates comfortably because nothing rings.

**Rung 2c — closed.** No code change required: the coupling is
second-order. Deliverables: (a) the mode-selective l=0 radial
arbiter as a STANDING GATE (tests/test_rung2c.py,
validation/rung2c_gate.txt: x2 offset 0.915% < 1.4%, order 2.07);
(b) the diagnostic lesson (never scan resonances with a random
drive in a dense spectrum — mode-selective drives only); (c) mesh
policy: auto_nmesh curvature floors stand (reflector boundaries
h/R <= 0.09; hard-ringing fluid-core models wanting lossless-coda
correlation gates need ~0.045, which awaits block elimination).

**Rung 2d — attenuated PREM arbitration (proposed).** The PREM
goal itself resolves the elastic-protocol hardness: real PREM has
Qmu ~ 80-300, which damps the late trapped coda that drives the
min-corr failures. Both DSM (Qmu/Qkappa per zone) and the BEM
(material Q) support attenuation natively; an attenuated
PREM-truncated onion vs DSM at the standard meshes is the
practical precision benchmark for the fluid-core machinery, with
the elastic corefluid case documented as resolution-limited by the
quantified (h/R)^2 law. Optional: CHIEF / Burton-Miller mitigation
of the sparse spurious real-axis pressure resonances if
resonance-style post-processing is ever needed.

**Rung 2e — shell-by-shell block elimination + graded-PREM
staircase.** Implemented 2026-07-10 (branch `multilayer`).

*Block elimination:* the nested-shell system is block tridiagonal
when unknowns are grouped by interface (every equation is a region's
representation formula collocated on one of its bounding interfaces,
and a region touches at most two consecutive interfaces).
MultiDomainModel.assemble was refactored into a per-block generator
(_block_entries / assemble_blocks — BITWISE gate: the full rung
0/1/2/2b battery re-passed unchanged, validation/elim_gate.txt), and
elimination.ShellElimination runs the generalized Thomas algorithm:
forward elimination innermost-first with per-group dense LU,
optional back substitution; full=False returns just the surface
group with peak memory of a few interface-sized blocks. GATES
(tests/test_elim.py): eliminated == dense solve at rel err 1e-15 -
3e-14 on welded-3, fluid-core-3, annulus and mixed-5 models, plus
multi-RHS and (bitwise) surface-only-vs-full agreement. Scale proof:
prem_s8 (10 layers, 37,760 unknowns) runs at ~253 s/harmonic on 24
procs — the dense matrix alone would be 43 GB per frequency.

*Thin-shell accuracy (measured, tests/test_elim.py probe):* a
transparent double interface holds the transparent-interface error
class (~5-6e-2 on the coarse probe meshes) from t/h = 1.56 all the
way down to t/h ~ 0.26, degrading only at ~0.13 — the adaptive
quadrature handles near-touching boundaries gracefully. Mesh rule
adopted for staircase shells: h <= 2 * (adjacent shell thickness)
(ARB_KAPPA = 2, a 2x margin on the measured knee).

*Source-placement lesson (first campaign FAILED, then bug-hunted per
the standing rule; evidence in dsm_arbitration_prem_midsrc_failed/):*
with the source at the historical 637-km depth, the mantle staircase
put welds arbitrarily close to it. Two distinct causes: (1) REAL BUG
— incident_outer_source hard-assumes the source is in the OUTERMOST
layer; prem_s8's source actually sat in the second-from-outer shell,
so the incident term entered the wrong region's representation
equation (unphysical system; corr < 0). Fixed by
layered.incident_solid_layer_source — the incident field enters
exactly the row blocks OWNED by the source layer's region, gated
bitwise against the old path for outermost sources and by a
middle-shell-source transparent test (3.95e-2, single-region class).
(2) PROTOCOL FLAW — the incident-field trace on a boundary at
distance d from the source needs panel size h <~ d; prem_s4 had a
weld 86 km below the source with 1119-km panels (d/h = 0.077 ->
median error 67%, corr -0.12; prem_s2's d/h = 0.83 was only
marginal). Campaign fix: source moved into the constant inner core
(r0 = 600 km; d/h = 2.7 at the ICB), so NO ladder interface is ever
near it; SH cannot be driven from below the fluid outer core, so the
staircase arbitration runs mrr/PSV only (toroidal welds were already
DSM-arbitrated in rungs 1-2d).

*Interior-source low-frequency deficit (second campaign FAILED, then
bug-hunted; evidence in dsm_arbitration_prem/ (IC source) and
dsm_arbitration_deepdiag/):* moving the source into the inner core
produced coherent but ~18x-too-small BEM waveforms on every
fluid-core leg (amp ratio 0.025-0.06, corr ~ 0) while the deep-source
HOMOG baseline stayed clean (23% floor = the longer-path version of
the 12.9% one). The per-harmonic k-scan localizes it: the
|BEM|/|DSM| ratio collapses (to 0.007 at the Ricker center) for
harmonics with w R_region / vs <~ 1 of the SOURCE region and
recovers above — homog (R 6371) is degraded only below k ~ 10,
a welded core (R 3185.5, vs 4.5) through k ~ 25-46 (deep-diag leg:
57%, amp 0.89, waveforms tracking), the inner core (R 1221.5,
vs 3.57) through k ~ 120, i.e. the whole band. This is the rung-1
"rigid-mode-amplified noise below w R/vs ~ 0.3" caveat, promoted to
a hard limitation when an INTERIOR source region is acoustically
small: the near-rigid null space of that region's T operator
misapportions the quasi-static incident field. Remedy (future rung):
a scattered-field interior-source formulation (needs
incident-TRACTION kernels), or keep interior-source regions large
in-band. Also settled: DSM tipsv REFUSES fluid-zone sources ("STOP
The source is in a fluid zone"), so a fluid-core explosion cannot
serve as the reference-side workaround.

*Graded-mantle staircase arbitration — final design
(dsm_arbitration_gprem/):* the three constraints (source region
acoustically large in-band; no weld within ~1 panel of the source;
SH needs the source above any fluid) pin the source into a LARGE
CONSTANT SOLID CORE, so the fluid outer core is dropped for this
convergence study (the fluid-core topology is separately validated,
rungs 2b/2d) and the ENTIRE mantle — including the strongly graded
upper mantle, where the staircase signal lives — is the target:
all-solid Earth, constant core 0-3480 km (PREM-IC-like,
12.89/11.12/3.57, Qmu 50; deficit zone ends by k ~ 43), CMB now a
transmissive weld at the h/R 0.2 floor, ns = 2/4/8 volume-averaged
PREM mantle shells (uniform Qmu = 50 pure-shear, disp_ref_hz = 1),
source at r0 = 1200 km (d/h = 3.3 to the nearest boundary), mrr +
mrt (tish accepts the deep source in an all-solid model), Ricker
recentered at ARB_F0 = 3.2e-4 Hz so the band (k ~ 60-138) sits above
the core's deficit zone. DSM reference: the same constant core plus
the TRUE graded PREM polynomials; control legs (gprem_s2c/s4c/s8c:
DSM runs the same staircase constants, BEM spectra reused via
bem_alias) isolate BEM mesh error from model-approximation error.

VERDICT (dsm_arbitration_gprem/, 2026-07-10): the protocol is sound
— baseline 12.65% / min corr 0.990 (best floor of any campaign) —
and every graded leg sits within 1.5x of its own control at every
rung (34.5 vs 30.7 / 45.7 vs 50.3 / 57.6 vs 59.3 % at ns = 2/4/8):
the BEM tracks its own model class throughout. But the ladder gate
FAILS, and the control legs show why, cleanly:

1. The graded-vs-staircase MODEL signal is unmeasurable at this
   band: the control-subtracted misfit is +3.8 / -4.6 / -1.7 % —
   sign-indefinite, below the between-rung mesh-error variance.
   Physically obvious in hindsight: lambda_S/4 ~ 5000 km EXCEEDS the
   whole mantle thickness (2891 km), so the ULP wavefield cannot
   distinguish even a 2-shell staircase from graded PREM. A real
   convergence demonstration needs lambda_S ~ shell thickness, i.e.
   a ~5-10x higher band, with per-interface mesh cost growing as f^2
   — reachable in principle with block elimination + the h/R laws
   (+ faster kernels), out of scope here.
2. What the controls DO measure: per-interface discretization
   accumulation. BEM-vs-same-model-DSM misfit grows 30.7 -> 50.3 ->
   59.3 % at 2/4/8 welds (~ +6%/interface, median amp ratio growing
   1.11 -> 1.20 -> 1.30, ~ x1.04/interface) at interface h/R
   0.09-0.2 in this recentered band — the record sections show pure
   progressively-growing dephasing/amplitude drift of the late
   multi-orbit ringing with the direct arrivals overlaying
   perfectly. Consistent with the standing order-2 h/R
   eigenfrequency-bias law; many-interface precision work must
   budget h/R per interface accordingly (h/R ~ 0.05 per interface
   for ~1-2%/interface).

Rung-2e CLOSURE: the dense-solve wall is REMOVED and gated (that was
the goal); staircase-convergence-vs-DSM is documented as
band-limited, not machinery-limited, with the accumulation law and
the three source-placement constraints quantified for the future
higher-band study.

**Rung 3 — boundary-perturbation ensembles with block caching.
DONE 2026-07-10** (rung3_ensemble/). The science payoff realized:
when only one interface's shape changes between ensemble members,
only the blocks touching that surface are reassembled.
elimination.cached_blocks invalidates by Interface OBJECT IDENTITY:
domains.nested_shell_model_from_ifaces rebuilds a model sharing every
unperturbed Interface object, so any block whose row/col interface is
the replaced object misses the cache automatically. GATES: cached
blocks + solution BITWISE == fresh (exactly the 11/16 blocks touching
the perturbed boundary recomputed, tests/test_elim.py); END-TO-END
amp=0 ensemble member (gen_mesh_relief zero relief) BITWISE == base
across all 138 harmonics.

Demo campaign: corefluid_q50 Earth, CMB radial relief members
Y20 2.5 km / Y20 5 km / random lmax<=4 5 km + amp0, 138 harmonics x
(base + 4 cached members) in 40 min on 24 procs
(rung3_ensemble/ensemble_run.txt). Physics: Y20 5-km CMB topography
changes the 0-24 ks waveforms by |du|/|u| = 4.6e-3 (time domain;
random l<=4 5 km: 2.5e-3), and the response is LINEAR in amplitude:
scaling ratio 2.0014 (spectra) / 1.9997 (time domain) for amp
doubling. Record sections with magnified difference panels:
rung3_ensemble/ens_*.png. Assembly speedup from caching is 1.29x for
a CMB perturbation on the 3-layer model (the CMB self-blocks are
among the largest); the win grows with shell count and for smaller
perturbed boundaries — for many-shell staircase models a single
perturbed internal interface leaves the vast majority of blocks
cached. Remaining rung-3 economy (not yet needed): two-sided
elimination sweeps cached from both ends toward the perturbed
interface would also reuse the LU factors.

## Engineering notes

- Thin shells make cross-surface integrals near-singular. The
  distance-adaptive tier machinery (assembly.Geometry, quad_mode
  "adaptive") is the hook: add a finer-than-degree-10 near tier when
  the surface separation drops below ~1 element size. Required before
  rung 2 runs thin-layer models.
- Scaling beyond dense solves: the nested-shell system is block
  tridiagonal in the layer index, so shell-by-shell block elimination
  (a generalized propagator method) is the natural next step, before
  any ACA / H-matrix machinery.
- The fluid row scaling (`scale_fac` in `liq_core`) and the
  `mu2 = rho1*vs2^2` driver quirk are preserved verbatim where needed
  for exact reproduction and documented at their definitions.

## Rung 4a — fluid-fluid interfaces (DONE 2026-07-11)

Adjacent fluid layers are legal: condition FLUID_FLUID with shared
pressure p plus a shared SCALAR normal displacement q = u.n_canonical
(new block kind "un"; 2n unknowns/interface vs 6n welded). Derivation
+ Codex second opinion + the complete implementation delta list:
docs/fluid_fluid_derivation.md. Gates (tests/test_ff.py,
validation/ff_gate.txt): existing battery bitwise-unchanged;
transparent identical-fluid split 7.0e-4; registration swap 1.5e-16
with the analytic identity A(+) + A(-) = I holding to 1.4e-21;
dense == eliminated to 2e-14..5e-14 on 4/5-layer fluid stacks
including a middle fluid bounded by fluid-fluid on both sides (the
own-modulus _scale fallback) and a rho 9-vs-4 contrast; rung-3 cache
bitwise with exactly the touching blocks recomputed. DSM arbitration
(dsm_arbitration_ff/): outer core split at 2350 km — identical-fluid
split 27.79% / staircase with ~30% impedance jump vs its own-model
DSM 27.66% vs fluid-solid control 27.83% (min corr all >= 0.905):
the per-fluid-fluid-interface error increment is ~ZERO. Graded
(staircase) outer cores are now free; open: fluid outermost (ocean),
sources in fluids.

## Rung 4b — graded epicentral meshes + the shallow-source verdict
(DONE 2026-07-11)

meshgen.refine_toward (graded quadrisection toward a surface point,
h <= max(grade*dist, hmin): log face count — 45-km epicentral panels
cost +120 faces on the 1584-face Earth surface) plus the
near-singular composite quadrature tier (Geometry near_tier,
subdivided deg-10; 2e4x accuracy at dist/h 0.28; strict no-op off;
knobs ARB_REFINE_HMIN_KM / ARB_REFINE_GRADE / ARB_NEAR_TIER). Gates:
tests/test_near_quad.py + 637-km transparency campaign
(dsm_arbitration_ref637: 11.1% vs 12.9% uniform baseline — refined
is BETTER).

50-km source verdict (dsm_arbitration_src50_ref, _ref2): the
PROPAGATING field is recovered (nearest-station corr 0.99; the
uniform-mesh acausal-burst failure is gone), but at this band the
DSM reference at 50-km depth is DOMINATED by the quasi-static
response (group delay ~0 s at all distances on mrt T, 10-100x the
propagating part) and the BEM delivers ~2% of it: the quasi-static
channel is driven by a near-perfectly-cancelling projection of the
incident trace, which pointwise collocation cannot deliver at ANY
refinement (l=2,m=1 projection wanders sign across hmin 45->5 km at
~1e-4 of gross; solved mrt amplitude FALLS with refinement, 0.022 ->
0.0075). Unified with the interior-source radiation deficit. Fixes,
in cost order (Codex round 2, docs/fast_methods_notes.md section 7):
moment-fitted RHS (0.5-2 wk falsification test), scattered-field
formulation C (RHS = G t_inc integral; cancellation moves into a
controlled weak integral, ~1% modal accuracy), spectral SH route A
(production Earth path; also cures this channel).

## Rung 4c — moment-fitted RHS + the DSM shallow-source audit
(DONE 2026-07-11; full detail: docs/moment_fitted_rhs.md,
docs/fast_methods_notes.md section 8)

pyastroseis/momentfit.py (vector-SH basis, graded source-frame exact
moments, minimal-W-norm fit; ARB_MFIT_LMAX hook in run_bem.py, strict
no-op off; tests/test_momentfit.py all at machine precision; Codex-
reviewed design). 637-km leg IMPROVED: median rel RMS 11.10% -> 9.09%
(min corr 0.9728 -> 0.9890) — the sampled-low-moment corruption was
real and is repaired.

The 50-km leg then EXPOSED THE REFERENCE: DSM tish (SH) in its
shallow-event branch (depth < 100 km) at ULP emits unconverged
high-l roundoff noise (band increments flat to l > 16384 vs a
physical bound falling 50 orders; T = 14-700x the analytic incident
field; the zero-group-delay "quasi-static pulse" is the Ricker STF
replayed by frequency-flat noise; re knob bit-inert). tipsv is
CERTIFIED converged for both sources. Rung 4b's "quasi-static
channel" interpretation is VOID; deep-source (>=100 km) references
all stand. Evidence: validation/dsm_audit/.

True remaining shallow-source deficits (valid channels only):
mrr VEC hump 0.15 (20 deg) -> ~0.71 (75-135 deg) -> 0.36 (170 deg),
amp +10-20%; mrt Z amplitude 0.21-0.32 of reference uniformly
(NOT an MT convention: 637-km mrt-Z amp = 1.03). Neither is a low-l
moment problem (fit does not move them). Rung C (scattered field)
re-scoped to target these, with a semi-analytic homogeneous-sphere
toroidal solution as the T-channel acceptance reference (to be
built BEFORE rung C so T is judgeable).
