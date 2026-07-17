#!/usr/bin/env python3
"""V-anchor: mini-tipsv vs tipsv, per-harmonic complex comparison on
the spheroidal channels (Z up, R away) at the manifest stations.
Scans the leg conjugation, freezes the better, reports per-station
median |ratio| and rel err over k in [klo, khi].

usage: minitipsv_check.py <root> <model> <npz> [source] [klo] [khi]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
MODEL = sys.argv[2]
NPZ = sys.argv[3]
SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"
KLO = int(sys.argv[5]) if len(sys.argv) > 5 else 20
KHI = int(sys.argv[6]) if len(sys.argv) > 6 else 138

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402

man = json.load(open(os.path.join(ROOT, "manifest.json")))
z = np.load(NPZ)
FIELD = os.environ.get("ARB_FIELD", "u")   # e.g. u_psv for the
isrc = [str(s) for s in z["sources"]].index(SOURCE)  # spectral npz
ks = np.arange(KLO, KHI + 1)

best = None
for conj in (False, True):
    u = np.conj(z[FIELD][isrc]) if conj else z[FIELD][isrc]
    tot = []
    rows = []
    for i, st in enumerate(man["stations"]):
        base = os.path.join(ROOT, "dsm", MODEL, "spc", st["name"])
        _, _, u_psv = sc.read_spc("%s.%s.PSV.spc" % (base, SOURCE))
        away, t_hat, north, east = sc.rotation_to_ne(
            man["source"]["lat"], man["source"]["lon"],
            st["lat"], st["lon"])
        xhat = sc.unit_vec(st["lat"], st["lon"])
        mZ = np.array([sum(u[i, c, k] * xhat[c] for c in range(3))
                       for k in ks])
        mR = np.array([sum(u[i, c, k] * away[c] for c in range(3))
                       for k in ks])
        dZ = u_psv[0][ks] * 1000.0
        dR = u_psv[1][ks] * 1000.0
        relZ = np.linalg.norm(mZ - dZ) / np.linalg.norm(dZ)
        relR = np.linalg.norm(mR - dR) / np.linalg.norm(dR)
        rows.append((st["name"], st["dist_deg"], relZ, relR,
                     np.median(np.abs(mZ / dZ)),
                     np.median(np.abs(mR / dR))))
        tot += [relZ, relR]
    med = float(np.median(tot))
    if best is None or med < best[0]:
        best = (med, conj, rows)
med, conj, rows = best
print("frozen conj=%s | per-k complex rel err over k=%d..%d"
      % (conj, KLO, KHI))
print("station dist |  relZ    relR  | |Z|ratio |R|ratio")
for name, dist, relZ, relR, rz, rr in rows:
    print("%-6s %5.1f | %6.3f  %6.3f |  %6.3f   %6.3f"
          % (name, dist, relZ, relR, rz, rr))
print("median rel %.4f" % med)
