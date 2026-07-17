#!/usr/bin/env python3
"""DSM arbitration — BEM spectra at the DSM harmonics.

usage: run_bem.py <model> [nprocs]      (models from manifest.json)

Every model is built through domains.nested_shell_model (bitwise-gated
against the rung-0/1 special cases), innermost layer first; a single
layer is the plain homogeneous solver.

For each harmonic k = 1..imax, w = 2*pi*k/tlen + i*omegai (identical
complex frequency to DSM). One assembly + one LU per frequency, all
moment sources solved together. Output: bem_<model>.npz with
u[src][station, comp(xyz), k] in meters (moment 1e20 N*m).
"""

import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ARB_ROOT", SCRIPT_DIR)
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PKG)

from pyastroseis.domains import Material, nested_shell_model  # noqa: E402
from pyastroseis.layered import scattered_solid_layer_source  # noqa: E402
from pyastroseis.elimination import ShellElimination          # noqa: E402
from pyastroseis.layered import incident_solid_layer_source   # noqa: E402
from pyastroseis.mesh import Faces                            # noqa: E402
from pyastroseis.source import u0eM                           # noqa: E402

Q_ELASTIC = 1.0e8


class _Pts:
    """Minimal stand-in for Faces so u0eM can evaluate the incident
    field at arbitrary points (moment-fit quadrature grids)."""

    def __init__(self, ic):
        self.ic = np.asarray(ic, dtype=float)
        self.n = len(self.ic)


def load_faces(path):
    z = np.load(path)
    return Faces(**{k: z[k] for k in
                    ("A", "B", "C", "nvec", "ic", "area", "r",
                     "a", "b", "c")})


def sph_to_cart_mt(mt6, lat_deg, lon_deg):
    """DSM (Mrr, Mrt, Mrp, Mtt, Mtp, Mpp) -> Cartesian 3x3 at the
    source location (theta = colatitude, phi = longitude)."""
    th = math.radians(90.0 - lat_deg)
    ph = math.radians(lon_deg)
    rh = np.array([math.sin(th) * math.cos(ph),
                   math.sin(th) * math.sin(ph), math.cos(th)])
    th_h = np.array([math.cos(th) * math.cos(ph),
                     math.cos(th) * math.sin(ph), -math.sin(th)])
    ph_h = np.array([-math.sin(ph), math.cos(ph), 0.0])
    mrr, mrt, mrp, mtt, mtp, mpp = mt6
    M_sph = np.array([[mrr, mrt, mrp],
                      [mrt, mtt, mtp],
                      [mrp, mtp, mpp]])
    P = np.column_stack([rh, th_h, ph_h])
    return P @ M_sph @ P.T


def phys_mat(vp, vs, rho, Q):
    if vs == 0.0:
        return Material.acoustic(vp, rho, Q, qp_fac=1.0)
    return Material.solid(vp, vs, rho, Q, qp_fac=0.75 * (vp / vs) ** 2)


def manifest_mat(la):
    """Material from a manifest layer entry: (rho, vp, vs[, Qmu]) in
    g/cc + km/s; Qmu absent -> elastic. Attenuated materials use the
    DSM/PREM causal-dispersion convention (reference 1 Hz)."""
    m = la["mat"]
    if len(m) > 3 and m[2] != 0.0:
        vp, vs, rho, Q = m[1] * 1e3, m[2] * 1e3, m[0] * 1e3, float(m[3])
        return Material.solid(vp, vs, rho, Q,
                              qp_fac=0.75 * (vp / vs) ** 2,
                              disp_ref_hz=1.0)
    return phys_mat(m[1] * 1e3, m[2] * 1e3, m[0] * 1e3, Q_ELASTIC)


_W = {}


def _worker(k):
    t0 = time.time()
    w = 2.0 * np.pi * k * _W["df"] + 1j * _W["omegai"]
    model, ifaces = _W["model"], _W["ifaces"]

    m = _W["mat_src"]
    xs, ys, zs = _W["src_xyz"]
    B = np.empty((model.size, len(_W["mts"])), dtype=complex)
    mfd = None
    if _W.get("scattered"):
        # rung C: scattered-field source formulation
        B[:, :] = scattered_solid_layer_source(
            model, ifaces, _W["src_layer"], m, w, (xs, ys, zs),
            _W["mts"], geom_opts=_W.get("geom_opts") or {},
            refine_spec=_W.get("scat_refine", ((8.0, 2), (30.0, 1))))
    for col, M in enumerate(_W["mts"]):
        if _W.get("scattered"):
            break
        def field(faces):
            return u0eM(faces, w, m.rho, m.mu, m.lamda, xs, ys, zs,
                        m.Q, M, qp_fac=m.qp_fac,
                        disp_ref_hz=m.disp_ref_hz)
        B[:, col] = model.assemble_rhs(incident_solid_layer_source(
            model, ifaces, _W["src_layer"], field))
        if _W["mfit"] is not None:
            # moment-fitted RHS (docs/moment_fitted_rhs.md): pin the
            # low-(l,m) moments of the sampled incident trace to their
            # exact quadrature values; "u" rows are unscaled so the
            # block can be corrected in place
            sl = model.block_slice("u", ifaces[-1])
            bc, dg = _W["mfit"].correct(B[sl, col],
                                        lambda pts: field(_Pts(pts)))
            B[sl, col] = bc
            if mfd is None:
                mfd = np.zeros((len(_W["mts"]), 3))
            mfd[col] = (dg["dbnorm"], dg["deficit"], dg["griddiff"])

    surf = ifaces[-1]
    if _W["elim"] is not None:
        # shell-by-shell block elimination: the last interface group is
        # the free surface, so full=False returns u_surf directly
        US = _W["elim"].solve(w=w, b=B, full=False)
    else:
        A = model.assemble(w)
        US = np.linalg.solve(A, B)[model.block_slice("u", surf)]
        del A

    n1 = surf.faces.n
    st = _W["st_faces"]
    out = np.empty((len(_W["mts"]), len(st), 3), dtype=complex)
    for col in range(len(_W["mts"])):
        us = US[:, col]
        if _W.get("scattered"):
            # total = scattered + incident on the source-region side
            ui = u0eM(surf.faces, w, m.rho, m.mu, m.lamda, xs, ys,
                      zs, m.Q, _W["mts"][col], qp_fac=m.qp_fac,
                      disp_ref_hz=m.disp_ref_hz)
            us = us + ui
        for i, j in enumerate(st):
            out[col, i] = (us[j], us[n1 + j], us[2 * n1 + j])
    return k, out, time.time() - t0, mfd


def main():
    model_name = sys.argv[1]
    nprocs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    alias = man.get("bem_alias", {}).get(model_name)
    if alias:
        print("%s: BEM leg aliased to %s (control model, DSM side "
              "only) — skipping" % (model_name, alias))
        return
    layers = man["models"][model_name]

    faces_list = [load_faces(os.path.join(ROOT, la["mesh"]))
                  for la in layers]
    mats = [manifest_mat(la) for la in layers]

    df = 1.0 / man["tlen"]
    omegai = man["omegai_1_per_s"]
    imax = man["imax"]
    w0 = 2.0 * np.pi * imax * df / 3.0   # reference frequency for scaling

    t_build = time.time()
    # ARB_NEAR_TIER="edge:deg:levels" (e.g. "1.5:10:2"): near-singular
    # composite quadrature tier for graded meshes (assembly.Geometry
    # near_tier; strict no-op when unset)
    opts = {}
    nt = os.environ.get("ARB_NEAR_TIER")
    if nt:
        edge, deg, lev = nt.split(":")
        opts["near_tier"] = (float(edge), (int(deg), int(lev)))
        print("near-singular quadrature tier: dist/h < %s -> "
              "deg %s x 4^%s composite" % (edge, deg, lev))
    scattered = os.environ.get("ARB_SCATTERED") == "1"
    scat_refine = ((8.0, 2), (30.0, 1))
    sr = os.environ.get("ARB_SCAT_REFINE")
    if sr:
        scat_refine = tuple(
            (float(p.split(":")[0]), int(p.split(":")[1]))
            for p in sr.split(","))
    if scattered:
        if os.environ.get("ARB_MFIT_LMAX"):
            raise ValueError("ARB_SCATTERED and ARB_MFIT_LMAX are "
                             "mutually exclusive")
        print("rung C: SCATTERED-FIELD source formulation, refine %s"
              % (scat_refine,), flush=True)
    model, ifaces = nested_shell_model(faces_list, mats, w0, **opts)
    use_elim = os.environ.get("ARB_ELIM", "auto")
    elim = None
    if use_elim == "1" or (use_elim == "auto" and len(layers) >= 4):
        elim = ShellElimination(model, ifaces)
    print("%s: %d layers, system size %d, geometry built in %.1f s, "
          "solver %s" %
          (model_name, len(layers), model.size, time.time() - t_build,
           "block-elimination" if elim is not None else "dense"),
          flush=True)

    src = man["source"]
    # the layer containing the source: its material drives u0eM and
    # its region's equations carry the incident field (the rung-2e
    # lesson: a source is NOT necessarily in the outermost layer)
    src_layer = next(i for i, la in enumerate(layers)
                     if src["r0_km"] <= la["r_km"])
    if mats[src_layer].fluid:
        raise ValueError("source sits in a fluid layer")
    print("source at r0 = %.1f km -> layer %d (outer radius %.1f km)"
          % (src["r0_km"], src_layer, layers[src_layer]["r_km"]),
          flush=True)
    th = math.radians(90.0 - src["lat"])
    ph = math.radians(src["lon"])
    r0 = src["r0_km"] * 1e3
    src_xyz = (r0 * math.sin(th) * math.cos(ph),
               r0 * math.sin(th) * math.sin(ph), r0 * math.cos(th))

    src_names = list(man["moment_tensors_1e25dyncm"].keys())
    # manifest MT entries are the DSM input values (100 = 1e20 N*m);
    # convert directly: 1 unit = 1e25 dyn*cm = 1e18 N*m
    scale = man["moment_scale_Nm"] / 100.0
    mts = [sph_to_cart_mt(man["moment_tensors_1e25dyncm"][s],
                          src["lat"], src["lon"]) * scale
           for s in src_names]

    # ARB_MFIT_LMAX=<L>: moment-fitted RHS falsification rig
    # (docs/moment_fitted_rhs.md); strict no-op when unset
    mfit = None
    ml = os.environ.get("ARB_MFIT_LMAX")
    if ml:
        from pyastroseis.momentfit import MomentFit
        if len(ifaces) != 1:
            raise ValueError("ARB_MFIT_LMAX supports single-layer "
                             "(homog) models only")
        R = layers[-1]["r_km"] * 1e3
        t0m = time.time()
        mfit = MomentFit(int(ml), ifaces[-1].faces.ic,
                         ifaces[-1].faces.area, R, src_xyz, R - r0)
        print("moment-fit RHS: lmax %s, %d moments, grids %d/%d pts, "
              "gram cond %.2e, built in %.1f s"
              % (ml, len(mfit.labels), len(mfit.pts1), len(mfit.pts2),
                 mfit.gram_cond, time.time() - t0m), flush=True)

    st_faces = [st["face"] for st in man["stations"]]
    _W.update(dict(scattered=scattered, scat_refine=scat_refine, geom_opts=opts, model=model, ifaces=ifaces, mat_src=mats[src_layer],
                   src_layer=src_layer, df=df, omegai=omegai, mts=mts,
                   src_xyz=src_xyz, st_faces=st_faces, elim=elim,
                   mfit=mfit))

    ks = list(range(1, imax + 1))
    kset = os.environ.get("ARB_KSET")
    if kset:
        ks = [int(x) for x in kset.split(",")]
        print("ARB_KSET: computing only k in %s" % ks, flush=True)
    u = np.zeros((len(mts), len(st_faces), 3, imax + 1), dtype=complex)
    mf_diag = (np.zeros((len(mts), 3, imax + 1))
               if mfit is not None else None)

    def took(k, out, dt, mfd):
        u[:, :, :, k] = out
        line = "  k=%3d f=%.4e Hz  %.1f s" % (k, k * df, dt)
        if mfd is not None:
            mf_diag[:, :, k] = mfd
            line += ("  mfit |db|/|b|=%.3g deficit=%.2g grid=%.1g"
                     % (mfd[:, 0].max(), mfd[:, 1].max(),
                        mfd[:, 2].max()))
        print(line, flush=True)

    t0 = time.time()
    if nprocs <= 1:
        for k, out, dt, mfd in map(_worker, ks):
            took(k, out, dt, mfd)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nprocs, len(ks))) as pool:
            for k, out, dt, mfd in pool.imap_unordered(_worker, ks):
                took(k, out, dt, mfd)
    wall = time.time() - t0
    print("%s: %d harmonics in %.1f s (%d procs)" %
          (model_name, len(ks), wall, nprocs), flush=True)

    extra = {}
    if mfit is not None:
        extra = dict(mfit_lmax=int(os.environ["ARB_MFIT_LMAX"]),
                     mfit_diag=mf_diag)
    np.savez(os.path.join(ROOT, "bem_%s.npz" % model_name),
             sources=np.array(src_names), u=u, imax=imax, df=df,
             omegai=omegai, wall_seconds=wall, nprocs=nprocs, **extra)
    print("wrote bem_%s.npz" % model_name)


if __name__ == "__main__":
    main()
