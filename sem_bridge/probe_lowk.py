#!/usr/bin/env python3
"""Low-k discriminator: per-(h, k) conj-ratio of our AZ06 Z/R
spectra against DSM, for staircase caps h = 80/40/20 km. If ratios
are h-stable but wrong vs DSM -> model-class/reference question; if
they move with h -> our multi-layer solve. mp-parallel over (h, k).
"""
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
HS = (80.0, 40.0, 20.0)
IST = 5                                             # AZ06


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


POLES = None
LAYERS = {h: bc.build_layers(h) for h in HS}


def solve(args):
    h, k = args
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    lm = int(min(420, max(60, 1.15 * abs(w) * bc.A_EARTH / 3460.0
                          + 30)))
    out = spectral_spectra(LAYERS[h], bc.src_xyz(), [bc.mt_cart()],
                           [w], bc.station_dirs(), Q=None, lmax=lm,
                           poles=POLES)
    return h, k, (out["psv"][0] + out["sh"][0])[IST, :, 0]


if __name__ == "__main__":
    POLES = spectral_poles(180)
    d = read_spc("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
                 "dsm_reference/ak135/spc/AZ06.mrr.PSV.spc")
    src = unit(6.0, 12.0)
    rec = unit(bc.LATS[IST], bc.LONS[IST])
    away = -(src - np.dot(src, rec) * rec)
    away /= np.linalg.norm(away)
    res = {}
    with mp.Pool(len(HS) * len(KS)) as pool:
        for h, k, u in pool.imap_unordered(
                solve, [(h, k) for h in HS for k in KS]):
            res[(h, k)] = u
    print("conj(ours)/dsm/1000 at AZ06:")
    print("  k   h[km]   |Z|      phZ      |R|      phR")
    for k in KS:
        for h in HS:
            u = res[(h, k)]
            rz = np.conj(rec @ u) / d[0][k] / 1e3
            rr = np.conj(away @ u) / d[1][k] / 1e3
            print(" %3d  %5.0f  %7.4f %8.2f  %7.4f %8.2f"
                  % (k, h, abs(rz), np.angle(rz, deg=True),
                     abs(rr), np.angle(rr, deg=True)))
    print("PROBE_LOWK DONE")
