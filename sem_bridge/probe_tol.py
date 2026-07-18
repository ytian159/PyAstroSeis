#!/usr/bin/env python3
"""Pruning-truncation test at the corrupt band: k = 40, 60, 100 at
lmax 1400 with tol_prune 1e-12 (production) vs 0 (keep the whole
stack, no shell-only truncation). 18-station medians vs
tipsv-on-the-same-staircase. If keep-all lands at ~1.00, the prune
criterion ((rb/r0)^(2l), quasi-static exponent) is the root cause —
it overestimates decay in the transition band l ~ z."""
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import spectral_spectra   # noqa: E402

KS = (40, 60, 100)
# (tol, lmax): keep-all is float-representable only for l <= ~200
# (per-layer y ratios ~2^l at the deepest kept layers), and the
# kband scan shows the k=40..100 answers are set by l <= 200.
CASES = ((1.0e-12, 200), (0.0, 200), (1.0e-12, 1400),
         (1.0e-18, 1400))
LAYERS = bc.build_layers(1.0e9)
POLES = None


def solve(args):
    k, (tol, lmax) = args
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    with np.errstate(all="ignore"):
        out = spectral_spectra(LAYERS, bc.src_xyz(),
                               [bc.mt_cart()], [w],
                               bc.station_dirs(), Q=None,
                               lmax=lmax, tol_prune=tol,
                               poles=POLES)
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
    POLES = bc.get_poles(1400)
    res = {}
    with mp.Pool(len(KS) * len(CASES)) as pool:
        for args, u in pool.imap_unordered(
                solve, [(k, c) for k in KS for c in CASES]):
            res[args] = u
    src = unit(6.0, 12.0)
    spc = {s: read_spc(os.path.join(HERE, "stair_dsm", "spc",
                                    "%s.mrr.PSV.spc" % s))
           for s in bc.STATIONS}
    for k in KS:
        for c in CASES:
            rz = []
            for i, s in enumerate(bc.STATIONS):
                rec = unit(bc.LATS[i], bc.LONS[i])
                u = res[(k, c)][i]
                rz.append(np.conj(rec @ u) / spc[s][0][k] / 1e3)
            az = np.abs(rz)
            pz = np.median(np.abs(np.degrees(np.angle(rz))))
            print("k=%3d tol=%-6g lmax=%4d: |Z| med %8.4f"
                  " [%7.3f..%8.3f] |ph| med %6.2f"
                  % (k, c[0], c[1], np.median(az), az.min(),
                     az.max(), pz))
    print("PROBE_TOL DONE")
