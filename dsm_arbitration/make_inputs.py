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
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ARB_ROOT", SCRIPT_DIR)
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PKG)

from pyastroseis.liquidcore import flip_normals   # noqa: E402
from pyastroseis.meshgen import gen_layer          # noqa: E402

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
SOURCE_DEPTH_KM = 637.1
SOURCE_R0_KM = R_KM - SOURCE_DEPTH_KM

SOURCES = {"mrr": (100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           "mrt": (0.0, 100.0, 0.0, 0.0, 0.0, 0.0)}

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
}


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

    # one deterministic mesh per unique boundary (rng seeded by radius)
    meshes = {}
    for name in model_names:
        for r_km, nmesh, _ in LAYER_SPECS[name]:
            key = (r_km, nmesh * scale)
            if key in meshes:
                continue
            rng = np.random.default_rng(int(round(r_km)))
            faces = outward(gen_layer((r_km * 1e3,), (nmesh * scale,),
                                      (0,), rng=rng)[0][0])
            fn = "mesh_r%05d_n%d.npz" % (int(round(r_km)), nmesh * scale)
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

    models_manifest = {}
    for name in model_names:
        spec = LAYER_SPECS[name]
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
        # convention — tish inputs start at the CMB). The source must
        # sit in that stack.
        fluid_tops = [z["rmax"] for z in zones if z["vs"][0] == 0.0]
        zones_sh = [z for z in zones
                    if not fluid_tops or z["rmin"] >= max(fluid_tops)]
        if fluid_tops:
            assert SOURCE_R0_KM >= max(fluid_tops), \
                "source below the fluid layer: tish model would miss it"
        mdir = os.path.join(ROOT, "dsm", name)
        os.makedirs(os.path.join(mdir, "spc"), exist_ok=True)
        for srcname, mt in SOURCES.items():
            write_inf(os.path.join(mdir, "tipsv_%s.inf" % srcname),
                      zones, True, mt, stations, "%s.PSV" % srcname)
            write_inf(os.path.join(mdir, "tish_%s.inf" % srcname),
                      zones_sh, False, mt, stations, "%s.SH" % srcname)

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
    }
    with open(os.path.join(ROOT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote DSM inputs + manifest.json")


if __name__ == "__main__":
    main()
