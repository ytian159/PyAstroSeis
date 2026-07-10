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
from pyastroseis.layered import incident_outer_source         # noqa: E402
from pyastroseis.mesh import Faces                            # noqa: E402
from pyastroseis.source import u0eM                           # noqa: E402

Q_ELASTIC = 1.0e8


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
    A = model.assemble(w)

    m = _W["mat_outer"]
    xs, ys, zs = _W["src_xyz"]
    B = np.empty((model.size, len(_W["mts"])), dtype=complex)
    for col, M in enumerate(_W["mts"]):
        def field(faces):
            return u0eM(faces, w, m.rho, m.mu, m.lamda, xs, ys, zs,
                        m.Q, M, qp_fac=m.qp_fac,
                        disp_ref_hz=m.disp_ref_hz)
        B[:, col] = model.assemble_rhs(
            incident_outer_source(model, ifaces, field))
    X = np.linalg.solve(A, B)
    del A

    surf = ifaces[-1]
    sl = model.block_slice("u", surf)
    n1 = surf.faces.n
    st = _W["st_faces"]
    out = np.empty((len(_W["mts"]), len(st), 3), dtype=complex)
    for col in range(len(_W["mts"])):
        us = X[sl, col]
        for i, j in enumerate(st):
            out[col, i] = (us[j], us[n1 + j], us[2 * n1 + j])
    return k, out, time.time() - t0


def main():
    model_name = sys.argv[1]
    nprocs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    layers = man["models"][model_name]

    faces_list = [load_faces(os.path.join(ROOT, la["mesh"]))
                  for la in layers]
    mats = [manifest_mat(la) for la in layers]

    df = 1.0 / man["tlen"]
    omegai = man["omegai_1_per_s"]
    imax = man["imax"]
    w0 = 2.0 * np.pi * imax * df / 3.0   # reference frequency for scaling

    t_build = time.time()
    model, ifaces = nested_shell_model(faces_list, mats, w0)
    print("%s: %d layers, system size %d, geometry built in %.1f s" %
          (model_name, len(layers), model.size, time.time() - t_build),
          flush=True)

    src = man["source"]
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

    st_faces = [st["face"] for st in man["stations"]]
    _W.update(dict(model=model, ifaces=ifaces, mat_outer=mats[-1],
                   df=df, omegai=omegai, mts=mts, src_xyz=src_xyz,
                   st_faces=st_faces))

    ks = list(range(1, imax + 1))
    u = np.zeros((len(mts), len(st_faces), 3, imax + 1), dtype=complex)
    t0 = time.time()
    if nprocs <= 1:
        results = map(_worker, ks)
        for k, out, dt in results:
            u[:, :, :, k] = out
            print("  k=%3d f=%.4e Hz  %.1f s" % (k, k * df, dt),
                  flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nprocs, len(ks))) as pool:
            for k, out, dt in pool.imap_unordered(_worker, ks):
                u[:, :, :, k] = out
                print("  k=%3d f=%.4e Hz  %.1f s" % (k, k * df, dt),
                      flush=True)
    wall = time.time() - t0
    print("%s: %d harmonics in %.1f s (%d procs)" %
          (model_name, len(ks), wall, nprocs), flush=True)

    np.savez(os.path.join(ROOT, "bem_%s.npz" % model_name),
             sources=np.array(src_names), u=u, imax=imax, df=df,
             omegai=omegai, wall_seconds=wall, nprocs=nprocs)
    print("wrote bem_%s.npz" % model_name)


if __name__ == "__main__":
    main()
