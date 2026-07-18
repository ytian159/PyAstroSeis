#!/usr/bin/env python3
"""Verdict probe at FLAT lmax=1400: 127-layer staircase spectral
leg vs (a) tipsv on the SAME staircase (apples-to-apples; spc from
driver_stair.sh) and (b) the graded-model campaign DSM. Gate: the
same-model comparison must sit near 1.0 (median |ratio| in
[0.9, 1.1], median |phase| < 10 deg) at every probed k. Exits
nonzero on failure."""
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

KS = (5, 10, 20, 40, 60, 100)
LMAX = 1400
LAYERS = bc.build_layers(1.0e9)
POLES = None


def solve(k):
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    out = spectral_spectra(LAYERS, bc.src_xyz(), [bc.mt_cart()],
                           [w], bc.station_dirs(), Q=None,
                           lmax=LMAX, poles=POLES)
    return k, out["psv"][0][:, :, 0] + out["sh"][0][:, :, 0]


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
    POLES = bc.get_poles(LMAX)
    res = dict()
    with mp.Pool(len(KS)) as pool:
        for k, u in pool.imap_unordered(solve, list(KS)):
            res[k] = u
    src = unit(6.0, 12.0)
    stair = {s: read_spc(os.path.join(HERE, "stair_dsm", "spc",
                                      "%s.mrr.PSV.spc" % s))
             for s in bc.STATIONS}
    graded_dir = ("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
                  "dsm_reference/ak135/spc")
    graded = {s: read_spc(os.path.join(graded_dir,
                                       "%s.mrr.PSV.spc" % s))
              for s in bc.STATIONS}
    ok = True
    for tag, ref in (("SAME-STAIRCASE tipsv", stair),
                     ("graded DSM", graded)):
        print("ours / %s (conj/1000), medians over 18 stations:"
              % tag)
        for k in KS:
            rz, rr = [], []
            for i, s in enumerate(bc.STATIONS):
                rec = unit(bc.LATS[i], bc.LONS[i])
                away = -(src - np.dot(src, rec) * rec)
                away /= np.linalg.norm(away)
                u = res[k][i]
                rz.append(np.conj(rec @ u) / ref[s][0][k] / 1e3)
                rr.append(np.conj(away @ u) / ref[s][1][k] / 1e3)
            mz, mr = np.median(np.abs(rz)), np.median(np.abs(rr))
            pz = np.median(np.abs(np.degrees(np.angle(rz))))
            pr = np.median(np.abs(np.degrees(np.angle(rr))))
            print(" k=%3d: |Z| med %.4f ph %6.2f | |R| med %.4f"
                  " ph %6.2f" % (k, mz, pz, mr, pr))
            if tag.startswith("SAME"):
                good = (0.9 < mz < 1.1 and 0.9 < mr < 1.1
                        and pz < 10 and pr < 10)
                ok = ok and good
    print("PROBE_1400", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
