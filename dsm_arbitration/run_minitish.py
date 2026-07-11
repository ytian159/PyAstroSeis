#!/usr/bin/env python3
"""Semi-analytic toroidal reference spectra at the campaign harmonics
(docs/toroidal_reference.md). Homogeneous (single-layer) models only.

usage: run_minitish.py <model> [nprocs] [lmax]
env: ARB_ROOT (campaign dir with manifest.json)

Output: minitish_<model>.npz — same layout as bem_<model>.npz
(u[src][station, comp(xyz), k] in meters), so the existing
synthesize/compare machinery can treat it as a solver leg.
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

from pyastroseis.toroidal_ref import toroidal_surface_field  # noqa: E402

import run_bem  # noqa: E402  (load_faces, sph_to_cart_mt, manifest_mat)

_W = {}


def _worker(k):
    t0 = time.time()
    w = 2.0 * np.pi * k * _W["df"] + 1j * _W["omegai"]
    out = np.empty((len(_W["mts"]), len(_W["st_dirs"]), 3),
                   dtype=complex)
    diag = np.zeros((len(_W["mts"]), 3))
    for col, M in enumerate(_W["mts"]):
        u_st, dg = toroidal_surface_field(
            _W["mat"], _W["R"], _W["src_xyz"], M, w, _W["lmax"],
            _W["st_dirs"], tier=1)
        out[col] = u_st
        if _W["tier2"]:
            u2, _ = toroidal_surface_field(
                _W["mat"], _W["R"], _W["src_xyz"], M, w, _W["lmax"],
                _W["st_dirs"], tier=2)
            dd = np.linalg.norm(u2 - u_st) / max(np.linalg.norm(u2),
                                                 1e-300)
            out[col] = u2
        else:
            dd = 0.0
        diag[col] = (dg["mon_ratio"], dg["tail_ratio"], dd)
    return k, out, diag, time.time() - t0


def main():
    model = sys.argv[1]
    nprocs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    lmax = int(sys.argv[3]) if len(sys.argv) > 3 else None
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    layers = man["models"][model]
    if len(layers) != 1:
        raise ValueError("mini-tish: homogeneous (1-layer) models only")
    mat = run_bem.manifest_mat(layers[0])
    R = layers[0]["r_km"] * 1e3

    src = man["source"]
    th = math.radians(90.0 - src["lat"])
    ph = math.radians(src["lon"])
    r0 = src["r0_km"] * 1e3
    src_xyz = np.array([r0 * math.sin(th) * math.cos(ph),
                        r0 * math.sin(th) * math.sin(ph),
                        r0 * math.cos(th)])
    if lmax is None:
        # physical tail (r0/R)^l < 1e-8 with sqrt(l) prefactor margin
        lmax = int(min(4000, max(200, 20.0 * R / (R - r0))))
    print("mini-tish %s: lmax %d, source depth %.1f km"
          % (model, lmax, (R - r0) / 1e3), flush=True)

    scale = man["moment_scale_Nm"] / 100.0
    src_names = list(man["moment_tensors_1e25dyncm"].keys())
    mts = [run_bem.sph_to_cart_mt(man["moment_tensors_1e25dyncm"][s],
                                  src["lat"], src["lon"]) * scale
           for s in src_names]

    faces = run_bem.load_faces(os.path.join(ROOT, layers[0]["mesh"]))
    st_idx = [st["face"] for st in man["stations"]]
    st_dirs = faces.ic[st_idx]
    st_dirs = st_dirs / np.linalg.norm(st_dirs, axis=1)[:, None]

    df = 1.0 / man["tlen"]
    imax = man["imax"]
    _W.update(dict(mat=mat, R=R, src_xyz=src_xyz, mts=mts, df=df,
                   omegai=man["omegai_1_per_s"], lmax=lmax,
                   st_dirs=st_dirs, tier2=True))

    ks = list(range(1, imax + 1))
    u = np.zeros((len(mts), len(st_dirs), 3, imax + 1), dtype=complex)
    dg = np.zeros((len(mts), 3, imax + 1))
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(nprocs, len(ks))) as pool:
        for k, out, diag, dt in pool.imap_unordered(_worker, ks):
            u[:, :, :, k] = out
            dg[:, :, k] = diag
            print("  k=%3d  %.1f s  mon=%.1e tail=%.1e tier=%.1e"
                  % (k, dt, diag[:, 0].max(), diag[:, 1].max(),
                     diag[:, 2].max()), flush=True)
    print("mini-tish %s: %d harmonics in %.1f s" %
          (model, len(ks), time.time() - t0), flush=True)
    np.savez(os.path.join(ROOT, "minitish_%s.npz" % model),
             sources=np.array(src_names), u=u, imax=imax, df=df,
             omegai=man["omegai_1_per_s"], lmax=lmax, diag=dg)
    print("wrote minitish_%s.npz" % model)


if __name__ == "__main__":
    main()
