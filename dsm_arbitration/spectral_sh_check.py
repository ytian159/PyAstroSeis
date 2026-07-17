#!/usr/bin/env python3
"""SH cross-check: rung-A spectral u_sh (direct forced solve) vs the
toroidal mode-sum reference (tormodes npz), per-k complex, vector
over the 3 components. House conj scan, frozen at the better.

usage: spectral_sh_check.py <root> <model> [source] [klo] [khi]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
MODEL = sys.argv[2]
SOURCE = sys.argv[3] if len(sys.argv) > 3 else "mrt"
KLO = int(sys.argv[4]) if len(sys.argv) > 4 else 20
KHI = int(sys.argv[5]) if len(sys.argv) > 5 else 138

man = json.load(open(os.path.join(ROOT, "manifest.json")))
zs = np.load(os.path.join(ROOT, "spectral_%s.npz" % MODEL))
zt = np.load(os.path.join(ROOT, "tormodes_%s.npz" % MODEL))
i_s = [str(s) for s in zs["sources"]].index(SOURCE)
i_t = [str(s) for s in zt["sources"]].index(SOURCE)
ks = np.arange(KLO, KHI + 1)
ref = zt["u"][i_t][:, :, ks]

best = None
for conj in (False, True):
    u = np.conj(zs["u_sh"][i_s]) if conj else zs["u_sh"][i_s]
    u = u[:, :, ks]
    rows = []
    for i, st in enumerate(man["stations"]):
        r = (np.linalg.norm(u[i] - ref[i])
             / np.linalg.norm(ref[i]))
        c = int(np.argmax(np.abs(ref[i]).max(axis=1)))
        rat = np.median(np.abs(u[i, c] / ref[i, c]))
        rows.append((st["name"], st["dist_deg"], r, rat))
    med = float(np.median([r for _, _, r, _ in rows]))
    if best is None or med < best[0]:
        best = (med, conj, rows)
med, conj, rows = best
print("SH check %s %s | frozen conj=%s | vector per-k complex rel "
      "err over k=%d..%d" % (MODEL, SOURCE, conj, KLO, KHI))
print("station dist |  rel   |dom|ratio")
for name, dist, r, rat in rows:
    print("%-6s %5.1f | %6.3f   %6.3f" % (name, dist, r, rat))
print("median rel %.4f" % med)
