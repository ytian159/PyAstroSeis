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
refinement must converge. REMAINING for full rung-1 closure: a real
two-layer solid sphere vs an independent spherically-symmetric
reference (DSM / normal modes / layered-sphere Mie series), same
protocol discipline as the DFDM–SEM benchmarks.

**Rung 2 — N nested shells + config schema.** YAML layer lists
(radius / mesh resolution / material per shell; `meshgen.gen_layer`
already builds the meshes). Validate a truncated PREM/AK135 onion at
long period. Unknown count grows by ~6N per welded interface; dense
solves stay practical to roughly 20–30k unknowns on one node.

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
