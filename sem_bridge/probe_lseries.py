#!/usr/bin/env python3
"""Discriminator: rerun k=40/60 (lmax 200) with L_SERIES raised so
l = 20..L-1 go through the UNSCALED basis (and the source jumps
switch basis with them). If the answer moves to ~1.0 vs
tipsv-on-staircase, the scaled-basis machinery in the propagating
band is the defect; if unchanged, both bases agree and the bug is
elsewhere."""
import math
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
import pyastroseis.spectral as sp                   # noqa: E402
import pyastroseis.spheroidal_ref as sr             # noqa: E402

KS = (40, 60)
LSER = (20, 60, 120)
LMAX = 200
LAYERS = bc.build_layers(1.0e9)
POLES = None


def solve(args):
    k, ls = args
    sp.L_SERIES = ls
    sr.L_SERIES = ls
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    with np.errstate(all="ignore"):
        out = sp.spectral_spectra(LAYERS, bc.src_xyz(),
                                  [bc.mt_cart()], [w],
                                  bc.station_dirs(), Q=None,
                                  lmax=LMAX, poles=POLES)
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
    with mp.Pool(len(KS) * len(LSER)) as pool:
        for args, u in pool.imap_unordered(
                solve, [(k, ls) for k in KS for ls in LSER]):
            res[args] = u
    spc = {s: read_spc(os.path.join(HERE, "stair_dsm", "spc",
                                    "%s.mrr.PSV.spc" % s))
           for s in bc.STATIONS}
    for k in KS:
        for ls in LSER:
            rz = []
            for i, s in enumerate(bc.STATIONS):
                rec = unit(bc.LATS[i], bc.LONS[i])
                u = res[(k, ls)][i]
                rz.append(np.conj(rec @ u) / spc[s][0][k] / 1e3)
            az = np.abs(rz)
            pz = np.median(np.abs(np.degrees(np.angle(rz))))
            print("k=%3d L_SERIES=%3d: |Z| med %8.4f"
                  " [%7.3f..%8.3f] |ph| med %6.2f"
                  % (k, ls, np.median(az), az.min(), az.max(),
                     pz))
    print("PROBE_LSERIES DONE")
