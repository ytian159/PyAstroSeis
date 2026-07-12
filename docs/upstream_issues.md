# Upstream bug / limitation reports (drafts, ready to file)

Evidence produced 2026-07-11 during the PyAstroSeis uniform3/corefluid
ULP arbitration campaign (branch `multilayer`, commits 9db475d,
0b8d2e9). All frequencies below are for the 3-region benchmark model
at PREM radii — IC (r <= 1221.5 km): rho 6.0 g/cc, vp 7.0, vs 3.5
km/s; fluid OC (<= 3480 km): 5.0, 5.5; shell (<= 6371 km): 3.0, 6.0,
3.0 — gravity-free, and "exact" means analytic uniform-layer Bessel
matching (dsm_arbitration/exact_modes.py; validated against Lamb's
homogeneous-sphere 0S2 omega*R/vs = 2.650, against the independent
mini-tish surface-factor poles to 5 significant digits, and against
single-zone DSM runs to ~1%).

--------------------------------------------------------------------
## Issue 1 — SPECFEM3D_GLOBE: fluid-core spheroidal modes 10-14% low
   at ultra-long periods (T ~ 2000-10000 s), gravity-free
--------------------------------------------------------------------
Repo: geodynamics/specfem3d_globe, v8.1.0 (b6fa4b6).

SUMMARY. On a uniform 3-region model with a fluid outer core
(custom 1D model, GRAVITY/ROTATION/ELLIPTICITY/TOPOGRAPHY/OCEANS all
.false.), the spheroidal free-oscillation frequencies realized by the
solver sit 10-14% BELOW the exact eigenfrequencies, while the
toroidal modes are correct to ~1%. The defect is therefore isolated
to the fluid outer core / fluid-solid coupling path, and it grows
toward low frequency (deeper core penetration of the mode):

    mode      exact [uHz]   SPECFEM fit [uHz]   ratio
    0S2          117.08         100.77          0.861
    0S3          195.17         171.44          0.878
    0S4          283.62         253.07          0.892
    0S5          376.63         337.27          0.896
    0S6          470.40         425.61          0.905
    0T2          180.57         182.04          1.008
    0T3          283.51         282.37          0.996
    0T4          377.29         373.07          0.989
    0T5          466.55         462.88          0.992

METHOD. NEX 32, DT 0.25 s, double precision, elastic
(ATTENUATION = .false.), erf CMT source (hdur 2000 s) at 50 km
depth, 330-min record; frequencies from a variable-projection
multi-sinusoid least-squares fit of the post-ramp free oscillations
at 5 stations (Z for spheroidal, transverse for toroidal); the fit
is start-point independent (identical from 1.00x, 0.94x, 0.88x
starts). Equivalent group delays were confirmed independently in the
traveling-wave regime: lag vs a validated DSM leg grows linearly
with distance (3.8 s/deg viscous / 2.1 s/deg elastic), identical at
NEX 32 and NEX 64, with waveform correlation 0.97-0.999 after lag
removal (pure clock error).

RULED OUT. Input model realized exactly (xwrite_profile CARDS);
attenuation path exonerated (elastic runs show the same defect;
in the viscous run the modulus chain was traced end-to-end and the
viscous-minus-elastic delay equals the intended physical-dispersion
exactly); mesh resolution (NEX 32 == NEX 64); time step; single
precision (double-precision build); STF truncation (USER_T0 = 3*hdur
so erf starts at erf(-4.9)); geocentric conventions
(ASSUME_PERFECT_SPHERE). Toroidal correctness rules out the solid
solver, the source, the mesher geometry, and the measurement method.

REPRODUCTION. Fork patch (custom model uniform3_q50 + forced SLS
band + USER_T0) and drivers in PyAstroSeis multilayer:
validation/specfem_u3/uniform3_fork.patch, validation/u3_*.sh,
dsm_arbitration/exact_modes.py + sem_mode_fit2.py.

CONTEXT FOR MAINTAINERS. Likely untested regime: ULP band
(f < 0.5 mHz) with a fluid core and gravity OFF. Published
long-period validations run with gravity on and rarely below
T ~ 1000 s. Suspect the outer-core potential formulation /
fluid-solid coupling terms in the gravity-free limit.

--------------------------------------------------------------------
## Issue 2 — DSMsynTI-mpi tish: (a) shallow-source SH output is
   unconverged noise; (b) multi-zone toroidal eigenfrequencies 5% low
--------------------------------------------------------------------
Repo: DSMsynTI-mpi (tish).

(a) SHALLOW-SOURCE NOISE (established 2026-07-11 morning, evidence
in validation/dsm_audit/): for sources shallower than ~100 km at
ultra-long periods, the tish l-sum does not converge — band
increments stay FLAT to l > 16384 while the physical bound
(2l+1)(r0/R)^l falls ~50 orders; the output T spectra are 14-700x
the analytic incident-field bound and are frequency-flat noise.
maxL-truncated rebuilds reproduce the campaign output exactly, so
this is the shipped behavior, not a build artifact. tipsv is
unaffected (increments track the physical bound; certified
converged). Deep sources (>= 600 km) are healthy.

(b) MULTI-ZONE OPERATOR BIAS (established 2026-07-11 evening): on
the 3-zone fluid-core model, tish's realized toroidal resonances sit
a uniform ~5% LOW even with a DEEP (637 km), fully-converged source:

    mode      exact [uHz]   tish peak [uHz]   ratio
    0T2          180.57         171.66        0.951
    0T3          283.51         267.03        0.942
    0T4          377.29         358.58        0.950
    0T5          466.55         442.50        0.948

Same binary on a SINGLE-zone model matches an independent exact
solution to 0.75% (mini-tish anchor at 637 km), and SPECFEM3D_GLOBE
matches the same exact annulus table to ~1% — so the bias is
specific to tish's multi-zone (fluid-zone-containing) model
handling. Hypothesis (unverified): the SH domain assembly around the
excluded fluid zone (e.g. an effectively thicker shell / wrong inner
boundary placement) — a uniform -5% frequency shift corresponds to
an effective shell geometry/velocity error of the same order.
Inputs: dsm_arbitration_u3/dsm_inputs/tish_mrt_el*.inf (elastic
solid marker "-1 -1"; deep-source variant r0 = 5734 km).

--------------------------------------------------------------------
Practical consequences adopted in this project:
- NEVER use tish for sources < 100 km deep; on multi-zone models do
  not use tish resonance timing at all until (b) is understood.
- tipsv remains the absolutely-validated spheroidal (Z) reference on
  fluid-core models (its R channel at shallow sources has a separate
  whole-band quasi-static plateau contamination — do not use R).
- SPECFEM3D_GLOBE is usable at ULP on fluid-core models ONLY for the
  toroidal channel (validated ~1%); not for spheroidal.
