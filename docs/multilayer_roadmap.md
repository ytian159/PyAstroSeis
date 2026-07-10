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

**Rung 2d — attenuated arbitration. DONE + PASSED 2026-07-10**
(dsm_arbitration_r2d/, r2d_disp_verdict.log). Ladder: Qmu = 50
solids (strong-signal validation: ~45% amplitude decay over the
24-ks window; at this ultra-long-period band PREM-class Qmu ~ 300
would damp only ~10%, so the elastic corefluid hardness persists
for PREM-Q — Q effects scale with cycle count), fluid outer core
elastic, pure-shear attenuation (BEM physical qp_fac == DSM
Qkappa = 1e8).

FINDING (first run failed at baseline 69.7%): DSM implements CAUSAL
constant-Q — computeCoef applies the Kanamori-Anderson physical
dispersion v(f) = v_ref (1 + ln(f/f_ref)/(pi Q)) with f_ref = 1 Hz
(PREM convention) plus i/(2Q); the BEM's historical constant-Q was
non-dispersive, leaving the BEM ~5.2% fast at Q=50 in this band
(measured as lag growing with window energy age). Implemented as
Material.disp_ref_hz (assembly.wave_speeds + source.u0eM), default
0 = bitwise historical behavior (full rung-0..2c battery re-passed,
validation/rung2d_battery.txt); attenuated arbitration materials
use disp_ref_hz = 1.0.

VERDICT (vector metrics, 0-24 ks, all gates PASS): homog_q50
baseline 12.88% / corr 0.960 (dispersion fix took it from 69.7%);
twolayer_q50 (welded + Q) 9.59% / corr 0.984 / amp 0.977;
corefluid_q50 (solid IC / elastic fluid OC / attenuated mantle)
27.8% / min corr 0.905 / amp 1.013 — the fluid-core topology passes
the min-corr gate once physical attenuation damps the trapped coda.
Amplitude ratios 0.98-1.01 validate the Q-magnitude convention
(factor-2 Q error would read ~0.74/1.35 at 45% window decay).
Optional future: CHIEF / Burton-Miller mitigation of the sparse
spurious real-axis pressure resonances if resonance-style
post-processing is ever needed.

Still needed for the full PREM goal: graded radial profiles
approximated by many constant shells, which pushes past dense
solves -> shell-by-shell block elimination (block-tridiagonal
structure).

**Rung 3 — boundary-perturbation workflows.** The science payoff:
(a) when only one interface's shape changes between ensemble members,
only the blocks touching that surface are reassembled (block caching);
(b) BEM meshes never need to conform across interfaces, so only the
perturbed boundary (CMB/Moho topography) is refined.

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
