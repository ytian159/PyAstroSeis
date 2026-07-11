#!/usr/bin/env python3
"""Rung-1 DSM arbitration — stage meshes, stations and DSM inputs.

Earth-scale two-layer solid sphere (R = 6371 km, welded interface at
3185.5 km) so the vetted DSMsynTI-mpi protocol from the DFDM campaigns
(bench/dsm_reference) applies unchanged. Elastodynamics is scale-free,
so this validates the same welded machinery used at asteroid scale.

Models (both purely elastic, Qmu=Qkappa=-1 in DSM / Q=1e8 in BEM):
  homog    : vp 6, vs 3, rho 3 everywhere       (BEM baseline leg)
  twolayer : core vp 8, vs 4.5, rho 4 below 3185.5 km, shell as above

Sources at lat 6, lon 12, depth 637.1 km (r0 = 5733.9 km), moment
1e20 N*m (DSM input value 100 in 1e25 dyn*cm):
  mrr : Mrr only  (pure P-SV, tests tipsv + spheroidal weld coupling)
  mrt : Mrt only  (excites SH too, tests tish + toroidal weld coupling)

Stations = actual BEM surface-face incenter directions nearest to the
target epicentral distances (kills BEM station-sampling error); DSM
latitudes are pre-distorted with the inverse geodetic transform
exactly as in bench/dsm_reference/scripts/make_dsm_inputs.py.
"""

import json
import math
import os
import re
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ARB_ROOT", SCRIPT_DIR)
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PKG)

from pyastroseis.liquidcore import flip_normals   # noqa: E402
from pyastroseis.meshgen import gen_layer, refine_toward   # noqa: E402

R_KM = 6371.0
RC_KM = 3185.5
FLATTENING = 1.0 / 298.25

TLEN = 262144.0
NP = 256
RE = 1.0e-3
RATC = 1.0e-10
RATL = 1.0e-5
ADAMP = 1.0e-2
IMIN, IMAX = 0, 138            # fmax = 138/262144 = 5.26e-4 Hz
OMEGAI = math.log(1.0 / ADAMP) / TLEN

SOURCE_LAT = 6.0
SOURCE_LON = 12.0
# ARB_SRC_DEPTH_KM: e.g. 5771 puts the source in the inner core
# (r0 = 600 km) so a many-shell mantle staircase never places a weld
# near the source (rung-2e lesson: the incident-field trace on a
# boundary at distance d needs panel size h <~ d)
SOURCE_DEPTH_KM = float(os.environ.get("ARB_SRC_DEPTH_KM", "637.1"))
SOURCE_R0_KM = R_KM - SOURCE_DEPTH_KM

SOURCES_ALL = {"mrr": (100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
               "mrt": (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)}
# ARB_SOURCES: comma list; a source below the outermost fluid cannot
# drive tish (SH), so deep-source campaigns run mrr/PSV only
SOURCES = {k: SOURCES_ALL[k]
           for k in os.environ.get("ARB_SOURCES", "mrr,mrt").split(",")}

TARGET_DIST_DEG = (20, 30, 45, 60, 75, 90, 105, 120, 135, 150, 160, 170)

# rho [g/cc], vp, vs [km/s]; vs = 0 marks a fluid layer. An optional
# 4th element is Qmu (pure-shear attenuation: DSM gets Qkappa = 1e8,
# the BEM the physical qp_fac = 0.75 (vp/vs)^2, i.e. Qkappa = inf);
# without it the material is elastic (Qmu = Qkappa = -1 / Q = 1e8).
MAT_SHELL = (3.0, 6.0, 3.0)
MAT_CORE = (4.0, 8.0, 4.5)
MAT_MID = (3.5, 7.0, 3.9)
MAT_IC = (12.9, 11.2, 3.6)      # PREM-cartoon inner core
MAT_OC = (11.0, 9.0, 0.0)       # PREM-cartoon fluid outer core
MAT_MANTLE = (4.5, 10.0, 5.5)   # PREM-cartoon whole mantle
MAT_IC2 = (6.0, 7.0, 3.5)       # corefluid2 inner core
MAT_OC2 = (5.0, 5.5, 0.0)       # corefluid2 fluid outer core
MAT_SHELL_Q = (3.0, 6.0, 3.0, 50.0)   # rung 2d: attenuated solids
MAT_CORE_Q = (4.0, 8.0, 4.5, 50.0)
MAT_IC2Q = (6.0, 7.0, 3.5, 50.0)
# fluid-fluid gate (G5, docs/fluid_fluid_derivation.md): a two-step
# fluid staircase around MAT_OC2 (~30% impedance jump at the split)
MAT_OC2A = (5.3, 5.9, 0.0)
MAT_OC2B = (4.7, 5.1, 0.0)

# model registry: (r_km, nmesh, material) innermost first; selected
# via ARB_MODELS (comma list, default "homog,twolayer")
LAYER_SPECS = {
    "homog": ((R_KM, 200, MAT_SHELL),),
    "twolayer": ((RC_KM, 50, MAT_CORE), (R_KM, 200, MAT_SHELL)),
    "threelayer": ((2200.0, 24, MAT_CORE), (4300.0, 90, MAT_MID),
                   (R_KM, 200, MAT_SHELL)),
    # rung 2b: solid inner core / fluid outer core / mantle (PREM
    # topology; tish model = solid zones above the fluid, standard
    # DSM CMB convention)
    "corefluid": ((1221.5, 12, MAT_IC), (3480.0, 55, MAT_OC),
                  (R_KM, 200, MAT_MANTLE)),
    # same 1-D model, internal meshes x4 (surface unchanged):
    # convergence discriminator for the corefluid drift (h/R at the
    # CMB, not h/lambda, controls the shell eigenfrequency error)
    "corefluid_fine": ((1221.5, 48, MAT_IC), (3480.0, 220, MAT_OC),
                       (R_KM, 200, MAT_MANTLE)),
    # fair-differential corefluid: mantle == homog baseline material,
    # so the surface-mesh toroidal mode-frequency bias (measured
    # +5.9% at h/R 0.26, ~(h/R)^2) is common mode between legs and
    # the verdict isolates the fluid-annulus machinery
    "corefluid2": ((1221.5, 12, MAT_IC2), (3480.0, 61, MAT_OC2),
                   (R_KM, 200, MAT_SHELL)),
    # corefluid2 with reflector boundaries at h/R ~ 0.09: fluid-solid
    # interfaces confine trapped modes whose frequencies carry an
    # O((h/R)^2) bias (measured order 2.0 on the 0T2 control), so the
    # CMB needs surface-class curvature resolution, not just
    # wavelength resolution
    "corefluid3": ((1221.5, 48, MAT_IC2), (3480.0, 200, MAT_OC2),
                   (R_KM, 200, MAT_SHELL)),
    # ALL reflector boundaries at the auto_nmesh curvature floor
    # h/R <= 0.09 (n=200 -> 1584 faces), including the ICB — the
    # corefluid/corefluid3 specs violated the rule at the ICB
    # (h/R 0.40/0.19), which pins ICB-confined mode families ~1.6%
    # high regardless of CMB/surface refinement
    "corefluid4": ((1221.5, 200, MAT_IC2), (3480.0, 200, MAT_OC2),
                   (R_KM, 200, MAT_SHELL)),
    # rung 2d: attenuated ladder (Qmu = 50 solids, elastic fluid) —
    # validates the Q machinery through free-surface, welded and
    # fluid-solid couplings; ~45% amplitude decay over the 24-ks
    # window at this band
    "homog_q50": ((R_KM, 200, MAT_SHELL_Q),),
    "twolayer_q50": ((RC_KM, 50, MAT_CORE_Q), (R_KM, 200, MAT_SHELL_Q)),
    "corefluid_q50": ((1221.5, 48, MAT_IC2Q), (3480.0, 200, MAT_OC2),
                      (R_KM, 200, MAT_SHELL_Q)),
    # fluid-fluid gate G5: corefluid_q50 with the outer core split at
    # 2350 km. ocsplit2 = IDENTICAL fluid both sides (transparent
    # split; the DSM model is physically corefluid_q50); ocstair2 = a
    # real two-step fluid staircase, DSM runs the SAME staircase
    # zones. Split mesh: n=42 -> 320 faces, h/R 0.198 (transmissive
    # class), sub-shell t/h ~ 2.4 (>= the h <= 2t staircase rule).
    "ocsplit2_q50": ((1221.5, 48, MAT_IC2Q), (2350.0, 42, MAT_OC2),
                     (3480.0, 200, MAT_OC2), (R_KM, 200, MAT_SHELL_Q)),
    "ocstair2_q50": ((1221.5, 48, MAT_IC2Q), (2350.0, 42, MAT_OC2A),
                     (3480.0, 200, MAT_OC2B), (R_KM, 200, MAT_SHELL_Q)),
}

# ------------------------------------------------------- graded PREM (2e)
# Isotropic PREM "onecrust"/no-ocean polynomial zones (x = r/6371),
# identical to the DFDM-campaign DSM inputs
# (bench/dsm_reference/scripts/make_dsm_inputs.py) and SPECFEM
# model_prem.f90 with ONE_CRUST/no ocean.
PREM_MOHO = R_KM - 24.4
PREM_ZONES_DEF = [
    # rmin, rmax, rho coefs, vp coefs, vs coefs
    (0.0, 1221.5,
     (13.0885, 0.0, -8.8381, 0.0),
     (11.2622, 0.0, -6.3640, 0.0),
     (3.6678, 0.0, -4.4475, 0.0)),
    (1221.5, 3480.0,
     (12.5815, -1.2638, -3.6426, -5.5281),
     (11.0487, -4.0362, 4.8023, -13.5732),
     (0.0, 0.0, 0.0, 0.0)),
    (3480.0, 3630.0,
     (7.9565, -6.4761, 5.5283, -3.0807),
     (15.3891, -5.3181, 5.5242, -2.5514),
     (6.9254, 1.4672, -2.0834, 0.9783)),
    (3630.0, 5600.0,
     (7.9565, -6.4761, 5.5283, -3.0807),
     (24.9520, -40.4673, 51.4832, -26.6419),
     (11.1671, -13.7818, 17.4575, -9.2777)),
    (5600.0, 5701.0,
     (7.9565, -6.4761, 5.5283, -3.0807),
     (29.2766, -23.6027, 5.5242, -2.5514),
     (22.3459, -17.2473, -2.0834, 0.9783)),
    (5701.0, 5771.0,
     (5.3197, -1.4836, 0.0, 0.0),
     (19.0957, -9.8672, 0.0, 0.0),
     (9.9839, -4.9324, 0.0, 0.0)),
    (5771.0, 5971.0,
     (11.2494, -8.0298, 0.0, 0.0),
     (39.7027, -32.6166, 0.0, 0.0),
     (22.3512, -18.5856, 0.0, 0.0)),
    (5971.0, 6151.0,
     (7.1089, -3.8045, 0.0, 0.0),
     (20.3926, -12.2569, 0.0, 0.0),
     (8.9496, -4.4597, 0.0, 0.0)),
    (6151.0, 6291.0,
     (2.6910, 0.6924, 0.0, 0.0),
     (4.1875, 3.9382, 0.0, 0.0),
     (2.1519, 2.3481, 0.0, 0.0)),
    (6291.0, PREM_MOHO,
     (2.6910, 0.6924, 0.0, 0.0),
     (4.1875, 3.9382, 0.0, 0.0),
     (2.1519, 2.3481, 0.0, 0.0)),
    (PREM_MOHO, R_KM,
     (2.6, 0.0, 0.0, 0.0),
     (5.8, 0.0, 0.0, 0.0),
     (3.2, 0.0, 0.0, 0.0)),
]

QMU_STAIR = 50.0     # uniform pure-shear Q for the graded-PREM legs:
                     # keeps the trapped fluid-core coda damped so the
                     # metric measures the mantle staircase, not the
                     # known O((h/R)^2) mesh eigenfrequency bias
                     # (rung 2c/2d); true PREM-Q (~300 in the lower
                     # mantle) damps only ~10% in this band
RC_ICB, RC_CMB = 1221.5, 3480.0


def poly_eval(coefs, r_km):
    x = r_km / R_KM
    return coefs[0] + x * (coefs[1] + x * (coefs[2] + x * coefs[3]))


def prem_sample(r_km):
    for rmin, rmax, rho, vp, vs in PREM_ZONES_DEF:
        if r_km <= rmax:
            return tuple(poly_eval(c, r_km) for c in (rho, vp, vs))
    raise ValueError("r beyond surface")


def prem_shell_avg(rlo, rhi, npts=4000):
    """Volume-weighted PREM average over a shell, midpoint rule
    (samples strictly inside, so zone-boundary points never leak the
    neighboring zone into the average; handles internal polynomial
    breaks/discontinuities)."""
    dr = (rhi - rlo) / npts
    r = rlo + (np.arange(npts) + 0.5) * dr
    vals = np.array([prem_sample(x) for x in r])
    w = r * r
    return tuple(float(np.sum(vals[:, k] * w) / np.sum(w))
                 for k in range(3))


MAT_PREM_IC = prem_shell_avg(0.0, RC_ICB) + (QMU_STAIR,)
MAT_PREM_OC = prem_shell_avg(RC_ICB, RC_CMB)   # fluid, elastic


def prem_graded_zones():
    """DSM zones for the graded reference: constant IC/OC exactly as
    the BEM cartoon, true PREM polynomials through the mantle, uniform
    Qmu with Qkappa = 1e8 (pure shear, BEM physical qp_fac)."""
    zones = [const_zone(0.0, RC_ICB, MAT_PREM_IC),
             const_zone(RC_ICB, RC_CMB, MAT_PREM_OC)]
    for rmin, rmax, rho, vp, vs in PREM_ZONES_DEF:
        if rmin < RC_CMB:
            continue
        zones.append({"rmin": rmin, "rmax": rmax,
                      "rho": rho, "vp": vp, "vs": vs,
                      "qmu": fmt(QMU_STAIR), "qkappa": "1e8"})
    return zones


def _mantle_stair(ns, kappa):
    """ns equal-thickness constant mantle shells (volume-averaged
    PREM), innermost first; internal welded meshes obey
    h <= min(kappa * shell thickness, 0.2 r) (thin-shell accuracy
    measured in tests/test_elim.py; welded curvature floor from
    rung 2b). Returns (shells, t)."""
    bounds = np.linspace(RC_CMB, R_KM, ns + 1)
    t = float(bounds[1] - bounds[0])
    shells = []
    for k in range(ns):
        rhi = float(bounds[k + 1])
        mat = prem_shell_avg(float(bounds[k]), rhi) + (QMU_STAIR,)
        if k == ns - 1:
            shells.append((R_KM, 200, mat))
        else:
            hr = min(kappa * t / rhi, 0.2)
            F = 4.0 * math.pi / hr ** 2
            nm = max(12, int(math.ceil((F + 16.0) / 8.0)))
            shells.append((rhi, nm, mat))
    return shells, t


def prem_stair_spec(ns, kappa):
    """PREM-topology staircase: constant IC + constant fluid OC (the
    graded target is the MANTLE; fluid-fluid interfaces are a later
    rung) + ns constant mantle shells."""
    shells, _ = _mantle_stair(ns, kappa)
    return tuple([(RC_ICB, 48, MAT_PREM_IC),
                  (RC_CMB, 200, MAT_PREM_OC)] + shells)


def gprem_stair_spec(ns, kappa):
    """ALL-SOLID graded-mantle family (rung 2e closure): one constant
    solid core 0-3480 km (PREM-IC-like material) + ns constant mantle
    shells. Rationale: an interior source suffers a low-frequency
    deficit for w R_region/vs <~ 1 (measured, k-scan 2026-07-10), so
    the source region must be LARGE (R = 3480 km, vs 3.57 km/s ->
    deficit ends by k ~ 43) and the Ricker recentered above it
    (ARB_F0 = 3.2e-4); DSM cannot place a source in a fluid zone, and
    a mid-mantle source cannot keep d/h >= 1 from dense staircase
    welds, so the fluid core is dropped for THIS convergence study
    (the fluid-core topology is separately validated, rungs 2b/2d).
    The CMB becomes a transmissive weld (curvature floor h/R 0.2)."""
    shells, t = _mantle_stair(ns, kappa)
    hr = min(kappa * t / RC_CMB, 0.2)
    F = 4.0 * math.pi / hr ** 2
    nm = max(12, int(math.ceil((F + 16.0) / 8.0)))
    return tuple([(RC_CMB, nm, MAT_PREM_IC)] + shells)


def gprem_graded_zones():
    zones = [const_zone(0.0, RC_CMB, MAT_PREM_IC)]
    for rmin, rmax, rho, vp, vs in PREM_ZONES_DEF:
        if rmin < RC_CMB:
            continue
        zones.append({"rmin": rmin, "rmax": rmax,
                      "rho": rho, "vp": vp, "vs": vs,
                      "qmu": fmt(QMU_STAIR), "qkappa": "1e8"})
    return zones


def model_spec(name):
    """-> (layer spec, DSM zones or None for default constant zones,
    bem_alias or None). prem_s<N> / gprem_s<N> = N-shell mantle
    staircase vs graded DSM (PREM topology / all-solid); a trailing
    'c' = control leg: same BEM spectra (aliased, not re-run) vs a
    DSM running the SAME staircase constants (isolates BEM mesh error
    from the model-approximation error)."""
    m = re.match(r"^(g?)prem_s(\d+)(c?)$", name)
    if not m:
        return LAYER_SPECS[name], None, None
    # thin-shell accuracy (tests/test_elim.py probe): the surface
    # error stays at the transparent-interface class down to
    # t/h ~ 0.26 and only degrades at ~0.13, so h <= 2t keeps a 2x
    # margin
    allsolid = bool(m.group(1))
    ns, control = int(m.group(2)), bool(m.group(3))
    kappa = float(os.environ.get("ARB_KAPPA", "2.0"))
    if allsolid:
        spec, zones = gprem_stair_spec(ns, kappa), gprem_graded_zones()
    else:
        spec, zones = prem_stair_spec(ns, kappa), prem_graded_zones()
    if control:
        return spec, None, "%sprem_s%d" % ("g" if allsolid else "", ns)
    return spec, zones, None


def undo_geocentric(lat_deg):
    if abs(lat_deg) >= 90.0:
        return lat_deg
    t = math.tan(math.radians(lat_deg)) / (1.0 - FLATTENING) ** 2
    return math.degrees(math.atan(t))


def outward(faces):
    if np.median(np.sum(faces.nvec * faces.ic, axis=1)) < 0:
        return flip_normals(faces)
    return faces


def save_faces(path, faces):
    np.savez(path, A=faces.A, B=faces.B, C=faces.C, nvec=faces.nvec,
             ic=faces.ic, area=faces.area, r=faces.r,
             a=faces.a, b=faces.b, c=faces.c)


def unit_vec(lat_deg, lon_deg):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon),
                     math.cos(lat) * math.sin(lon), math.sin(lat)])


def pick_stations(face1):
    src = unit_vec(SOURCE_LAT, SOURCE_LON)
    xhat = face1.ic / np.linalg.norm(face1.ic, axis=1)[:, None]
    dist = np.degrees(np.arccos(np.clip(xhat @ src, -1.0, 1.0)))
    stations, used = [], set()
    for k, target in enumerate(TARGET_DIST_DEG):
        order = np.argsort(np.abs(dist - target))
        j = next(int(i) for i in order if int(i) not in used)
        used.add(j)
        lat = math.degrees(math.asin(xhat[j, 2]))
        lon = math.degrees(math.atan2(xhat[j, 1], xhat[j, 0]))
        stations.append({"name": "ST%02d" % k, "face": j,
                         "lat": lat, "lon": lon,
                         "dist_deg": float(dist[j]),
                         "r_km": float(np.linalg.norm(face1.ic[j]) / 1e3)})
    return stations


# ------------------------------------------------------------------ DSM .inf
def fmt(v):
    return "%.10g" % v


def const_zone(rmin, rmax, mat):
    rho, vp, vs = mat[:3]
    qmu = mat[3] if len(mat) > 3 else None
    return {"rmin": rmin, "rmax": rmax,
            "rho": (rho, 0.0, 0.0, 0.0),
            "vp": (vp, 0.0, 0.0, 0.0),
            "vs": (vs, 0.0, 0.0, 0.0),
            # pure-shear attenuation: Qkappa effectively infinite
            # (matches the BEM's physical qp_fac); elastic zones keep
            # the -1 -1 convention
            "qmu": fmt(qmu) if qmu else "-1",
            "qkappa": "1e8" if qmu else "-1"}


def zone_lines_psv(z):
    ln = ["  %s %s %s %s %s %s" % (fmt(z["rmin"]), fmt(z["rmax"]),
          fmt(z["rho"][0]), fmt(z["rho"][1]), fmt(z["rho"][2]),
          fmt(z["rho"][3]))]
    for key in ("vp", "vp", "vs", "vs"):
        c = z[key]
        ln.append("    %s %s %s %s" % (fmt(c[0]), fmt(c[1]), fmt(c[2]),
                                       fmt(c[3])))
    ln.append("    1 0 0 0 %s %s" % (z["qmu"], z["qkappa"]))
    return ln


def zone_lines_sh(z):
    c = z["vs"]
    return ["  %s %s %s %s %s %s" % (fmt(z["rmin"]), fmt(z["rmax"]),
            fmt(z["rho"][0]), fmt(z["rho"][1]), fmt(z["rho"][2]),
            fmt(z["rho"][3])),
            "    %s %s %s %s" % (fmt(c[0]), fmt(c[1]), fmt(c[2]), fmt(c[3])),
            "    %s %s %s %s %s" % (fmt(c[0]), fmt(c[1]), fmt(c[2]),
                                    fmt(c[3]), z["qmu"])]


def write_inf(path, zones, is_psv, mt, stations, suffix):
    lines = ["! auto-generated by dsm_arbitration/make_inputs.py"]
    lines.append("  %.1f %d" % (TLEN, NP))
    lines.append("  %.1e" % RE)
    lines.append("  %.1e" % RATC)
    lines.append("  %.1e" % RATL)
    lines.append("  %.1e" % ADAMP)
    lines.append("  %d %d" % (IMIN, IMAX))
    lines.append("  %d" % len(zones))
    for z in zones:
        lines.extend(zone_lines_psv(z) if is_psv else zone_lines_sh(z))
    lines.append("  %s %.12f %.6f" % (fmt(SOURCE_R0_KM),
                                      undo_geocentric(SOURCE_LAT),
                                      SOURCE_LON))
    lines.append("  %s %s %s %s %s %s" % tuple(fmt(v) for v in mt))
    lines.append("  %d" % len(stations))
    for st in stations:
        lines.append("  %.12f %.10f" % (undo_geocentric(st["lat"]),
                                        st["lon"]))
    for st in stations:
        lines.append("spc/%s.%s.spc" % (st["name"], suffix))
    lines.append("end")
    for ln in lines:
        assert len(ln) <= 80, "line >80 chars: %r" % ln
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(ROOT, exist_ok=True)
    scale = int(os.environ.get("ARB_MESH_SCALE", "1"))
    model_names = os.environ.get("ARB_MODELS",
                                 "homog,twolayer").split(",")

    specs = {name: model_spec(name) for name in model_names}

    # one deterministic mesh per unique boundary (rng seeded by radius)
    meshes = {}
    for name in model_names:
        for r_km, nmesh, _ in specs[name][0]:
            key = (r_km, nmesh * scale)
            if key in meshes:
                continue
            rng = np.random.default_rng(int(round(r_km)))
            faces = outward(gen_layer((r_km * 1e3,), (nmesh * scale,),
                                      (0,), rng=rng)[0][0])
            fn = "mesh_r%05d_n%d.npz" % (int(round(r_km)), nmesh * scale)
            # ARB_REFINE_HMIN_KM: graded epicentral refinement of the
            # SURFACE mesh for shallow sources (h <= max(grade*dist,
            # hmin) toward the epicenter; log extra-face count). Needs
            # the near-singular quadrature tier on the solver side
            # (ARB_NEAR_TIER in run_bem).
            hmin_km = os.environ.get("ARB_REFINE_HMIN_KM")
            if hmin_km and abs(r_km - R_KM) < 1e-6:
                grade = float(os.environ.get("ARB_REFINE_GRADE", "1.0"))
                epi = unit_vec(SOURCE_LAT, SOURCE_LON) * R_KM * 1e3
                n0 = faces.n
                faces = refine_toward(faces, epi, float(hmin_km) * 1e3,
                                      grade=grade)
                fn = "mesh_r%05d_n%d_ref%g.npz" % (
                    int(round(r_km)), nmesh * scale, float(hmin_km))
                print("surface refined toward the epicenter: hmin %s km"
                      " grade %g, %d -> %d faces"
                      % (hmin_km, grade, n0, faces.n))
            save_faces(os.path.join(ROOT, fn), faces)
            meshes[key] = fn
            print("mesh r=%7.1f km: n=%d faces, mean r %.1f km -> %s"
                  % (r_km, faces.n,
                     np.mean(np.linalg.norm(faces.ic, axis=1)) / 1e3, fn))
            if key == (R_KM, 200 * scale):
                surface = faces

    stations = pick_stations(surface)
    for st in stations:
        print("  %s face %5d  dist %7.2f deg  lat %8.3f lon %9.3f "
              "r %.1f km" % (st["name"], st["face"], st["dist_deg"],
                             st["lat"], st["lon"], st["r_km"]))

    models_manifest, bem_alias = {}, {}
    for name in model_names:
        spec, zones_override, alias = specs[name]
        if alias:
            bem_alias[name] = alias
        if zones_override is not None:
            zones = zones_override
        else:
            zones, r_prev = [], 0.0
            for r_km, nmesh, mat in spec:
                zones.append(const_zone(r_prev, r_km, mat))
                r_prev = r_km
        models_manifest[name] = [
            {"r_km": r_km, "nmesh": nmesh * scale, "mat": list(mat),
             "mesh": meshes[(r_km, nmesh * scale)]}
            for r_km, nmesh, mat in spec]
        # SH sees only the contiguous solid stack under the surface:
        # zones above the outermost fluid zone (standard DSM PREM
        # convention — tish inputs start at the CMB). A source below
        # that stack cannot drive SH: skip tish (BEM/DSM then compare
        # the PSV wavefield; enforced via ARB_SOURCES=mrr).
        fluid_tops = [z["rmax"] for z in zones if z["vs"][0] == 0.0]
        zones_sh = [z for z in zones
                    if not fluid_tops or z["rmin"] >= max(fluid_tops)]
        sh_ok = not fluid_tops or SOURCE_R0_KM >= max(fluid_tops)
        if not sh_ok:
            assert list(SOURCES) == ["mrr"], \
                "source below the fluid: run ARB_SOURCES=mrr (no SH)"
        mdir = os.path.join(ROOT, "dsm", name)
        os.makedirs(os.path.join(mdir, "spc"), exist_ok=True)
        for srcname, mt in SOURCES.items():
            write_inf(os.path.join(mdir, "tipsv_%s.inf" % srcname),
                      zones, True, mt, stations, "%s.PSV" % srcname)
            if sh_ok:
                write_inf(os.path.join(mdir, "tish_%s.inf" % srcname),
                          zones_sh, False, mt, stations,
                          "%s.SH" % srcname)

    manifest = {
        "tlen": TLEN, "np": NP, "re": RE, "ratc": RATC, "ratl": RATL,
        "adamp": ADAMP, "imin": IMIN, "imax": IMAX,
        "omegai_1_per_s": OMEGAI,
        "source": {"lat": SOURCE_LAT, "lon": SOURCE_LON,
                   "depth_km": SOURCE_DEPTH_KM, "r0_km": SOURCE_R0_KM,
                   "lat_input_geodetic": undo_geocentric(SOURCE_LAT)},
        "moment_tensors_1e25dyncm": SOURCES,
        "moment_scale_Nm": 1.0e20,
        "stations": stations,
        "materials_gcc_kms": {"shell": MAT_SHELL, "core": MAT_CORE},
        "interface_km": RC_KM,
        "elastic": "Qmu=Qkappa=-1 (DSM) / Q=1e8 (BEM)",
        "mesh_scale": int(os.environ.get("ARB_MESH_SCALE", "1")),
        "models": models_manifest,
        # control legs: the BEM spectra of the aliased model are
        # reused (run_bem skips them); only the DSM side differs
        "bem_alias": bem_alias,
    }
    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote DSM inputs + manifest.json")


if __name__ == "__main__":
    main()
