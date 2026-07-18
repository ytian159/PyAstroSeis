#!/usr/bin/env python3
"""Spectral leg on the exact 127-layer staircase at k=5..60 (+
pruning/lmax self-consistency variants at k=10). Saves
stair_probe.npz (full 18-station spectra per k) and prints ratios
vs the GRADED DSM spc for orientation."""
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import (spectral_poles,   # noqa: E402
                                  spectral_spectra)

KS = (5, 10, 20, 40, 60)
LAYERS = bc.build_layers(1.0e9)
POLES = None


def lmax_for(w, lmax_cap=420):
    return int(min(lmax_cap, max(60, 1.15 * abs(w) * bc.A_EARTH
                                 / 3460.0 + 30)))


def solve(args):
    k, tol, lmul = args
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    lm = min(420, lmax_for(w) * lmul)
    out = spectral_spectra(LAYERS, bc.src_xyz(), [bc.mt_cart()],
                           [w], bc.station_dirs(), Q=None, lmax=lm,
                           tol_prune=tol, poles=POLES)
    return args, out["psv"][0][:, :, 0] + out["sh"][0][:, :, 0]


def read_spc(path):
    toks = []
    for line in open(path):
        toks += line.split()
    vals = [float(t) for t in toks]
    np0 = int(vals[1])
    body = vals[17:]
    u = np.zeros((3, np0 + 1), dtype=complex)
    for kk in range(len(body) // 7):
        b = body[7 * kk: 7 * kk + 7]
        i = int(b[0])
        u[0, i] = b[1] + 1j * b[2]
        u[1, i] = b[3] + 1j * b[4]
        u[2, i] = b[5] + 1j * b[6]
    return u


def unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo), math.sin(la)])


if __name__ == "__main__":
    POLES = spectral_poles(420)
    jobs = [(k, 1.0e-12, 1) for k in KS]
    jobs += [(10, 1.0e-16, 1), (10, 1.0e-12, 2)]
    res = {}
    with mp.Pool(len(jobs)) as pool:
        for args, u in pool.imap_unordered(solve, jobs):
            res[args] = u
    np.savez(os.path.join(HERE, "stair_dsm", "stair_probe.npz"),
             ks=np.array(KS),
             u=np.array([res[(k, 1.0e-12, 1)] for k in KS]))
    d = read_spc("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
                 "dsm_reference/ak135/spc/AZ06.mrr.PSV.spc")
    src = unit(6.0, 12.0)
    rec = unit(bc.LATS[5], bc.LONS[5])
    away = -(src - np.dot(src, rec) * rec)
    away /= np.linalg.norm(away)
    print("stair spectral vs GRADED dsm (AZ06), conj/1000:")
    for args in jobs:
        k = args[0]
        u = res[args][5]
        rz = np.conj(rec @ u) / d[0][k] / 1e3
        rr = np.conj(away @ u) / d[1][k] / 1e3
        print(" k=%3d tol=%g lmul=%d: |Z| %.4f ph %7.2f  |R| %.4f"
              " ph %7.2f" % (k, args[1], args[2], abs(rz),
                             np.angle(rz, deg=True), abs(rr),
                             np.angle(rr, deg=True)))
    print("PROBE_STAIR DONE")
