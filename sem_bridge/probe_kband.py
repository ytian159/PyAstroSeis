#!/usr/bin/env python3
"""Localize the moderate-k high-l garbage: cumulative-lmax scan at
k = 10 (control), 40, 60, 100 on the 127-layer staircase, ratios vs
tipsv-on-the-same-staircase at AZ06 and AZ12. The lmax step where
the ratio walks away from 1 brackets the polluting l-band."""
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

KS = (10, 40, 60, 100)
LMS = (100, 200, 300, 420, 600, 900, 1400)
LAYERS = bc.build_layers(1.0e9)
POLES = None


def solve(args):
    k, lm = args
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    out = spectral_spectra(LAYERS, bc.src_xyz(), [bc.mt_cart()],
                           [w], bc.station_dirs(), Q=None,
                           lmax=lm, poles=POLES)
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
    with mp.Pool(min(28, len(KS) * len(LMS))) as pool:
        for args, u in pool.imap_unordered(
                solve, [(k, lm) for k in KS for lm in LMS]):
            res[args] = u
    src = unit(6.0, 12.0)
    for ist in (5, 11):                             # AZ06, AZ12
        sta = bc.STATIONS[ist]
        d = read_spc(os.path.join(HERE, "stair_dsm", "spc",
                                  "%s.mrr.PSV.spc" % sta))
        rec = unit(bc.LATS[ist], bc.LONS[ist])
        away = -(src - np.dot(src, rec) * rec)
        away /= np.linalg.norm(away)
        print("%s: conj(ours)/stair-tipsv/1000 vs lmax:" % sta)
        for k in KS:
            row = []
            for lm in LMS:
                u = res[(k, lm)][ist]
                rz = np.conj(rec @ u) / d[0][k] / 1e3
                row.append("%7.3f@%+6.1f" % (abs(rz),
                                             np.angle(rz,
                                                      deg=True)))
            print(" k=%3d Z: %s" % (k, " ".join(row)))
    print("PROBE_KBAND DONE")
