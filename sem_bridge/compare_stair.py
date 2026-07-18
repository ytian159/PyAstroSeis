#!/usr/bin/env python3
"""Arbitration verdict: our 127-layer spectral leg vs tipsv run on
the SAME 127-zone staircase, per-k complex ratios at all 18
stations for k=5..60 (Z and R). Identical model, identical complex
frequencies — any disagreement is solver-side."""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402

SPC = os.path.join(HERE, "stair_dsm", "spc")


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


z = np.load(os.path.join(HERE, "stair_dsm", "stair_probe.npz"))
ks, u_all = z["ks"], z["u"]
src = unit(6.0, 12.0)
print("OURS/tipsv-ON-SAME-STAIRCASE, conj/1000 "
      "(median over 18 stations + AZ06 detail):")
for ik, k in enumerate(ks):
    rats_z, rats_r = [], []
    for i, sta in enumerate(bc.STATIONS):
        d = read_spc(os.path.join(SPC, "%s.mrr.PSV.spc" % sta))
        rec = unit(bc.LATS[i], bc.LONS[i])
        away = -(src - np.dot(src, rec) * rec)
        away /= np.linalg.norm(away)
        u = u_all[ik][i]
        if abs(d[0][k]) > 0:
            rats_z.append(np.conj(rec @ u) / d[0][k] / 1e3)
        if abs(d[1][k]) > 0:
            rats_r.append(np.conj(away @ u) / d[1][k] / 1e3)
    az = np.abs(rats_z)
    ar = np.abs(rats_r)
    pz = np.degrees(np.angle(rats_z))
    pr = np.degrees(np.angle(rats_r))
    print(" k=%3d: |Z| med %.4f [%.3f..%.3f] ph med %7.2f | "
          "|R| med %.4f [%.3f..%.3f] ph med %7.2f"
          % (k, np.median(az), az.min(), az.max(), np.median(pz),
             np.median(ar), ar.min(), ar.max(), np.median(pr)))
print("COMPARE_STAIR DONE")
