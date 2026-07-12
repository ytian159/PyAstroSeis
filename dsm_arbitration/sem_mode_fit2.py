#!/usr/bin/env python3
"""Clean per-branch SEM mode fit: fundamentals only, loose bounds,
multi-start. usage: sem_mode_fit2.py <case> <Z|T>"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("ARB_ROOT", os.path.abspath("dsm_arbitration_u3"))
os.environ["ARB_ROOT"] = ROOT
import synthesize_compare as sc                     # noqa: E402

CASE = sys.argv[1]
CHAN = sys.argv[2]
T_FIT0 = 3500.0
STATIONS = ("ST02", "ST04", "ST06", "ST08", "ST10")

# exact fundamentals (uHz), from exact_modes.py scans
EXACT = dict(
    Z=[117.08, 195.17, 283.62, 376.63, 470.40],      # 0S2..0S6
    T=[180.57, 283.51, 377.29, 466.55],               # 0T2..0T5 annulus
)[CHAN]
exact = np.array(EXACT) * 1e-6

man = json.load(open(os.path.join(ROOT, "manifest.json")))
recs = []
for name in STATIONS:
    st = [s for s in man["stations"] if s["name"] == name][0]
    if CHAN == "Z":
        d = np.loadtxt(os.path.join(CASE, "OUTPUT_FILES",
                                    "DF.%s.MXZ.sem.ascii" % name))
        t, u = d[:, 0], d[:, 1]
    else:
        dn = np.loadtxt(os.path.join(CASE, "OUTPUT_FILES",
                                     "DF.%s.MXN.sem.ascii" % name))
        de = np.loadtxt(os.path.join(CASE, "OUTPUT_FILES",
                                     "DF.%s.MXE.sem.ascii" % name))
        _, t_hat, north, east = sc.rotation_to_ne(
            man["source"]["lat"], man["source"]["lon"],
            st["lat"], st["lon"])
        t = dn[:, 0]
        u = dn[:, 1] * (t_hat @ north) + de[:, 1] * (t_hat @ east)
    m = t >= T_FIT0
    tt, uu = t[m], u[m]
    step = max(1, int(round(20.0 / (tt[1] - tt[0]))))
    recs.append((tt[::step], uu[::step]))


def resid(fvec):
    out = []
    for tt, uu in recs:
        cols = [np.ones_like(tt), tt / tt[-1]]
        for f in fvec:
            w = 2 * np.pi * f
            cols += [np.cos(w * tt), np.sin(w * tt)]
        A = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(A, uu, rcond=None)
        out.append((uu - A @ coef) / np.linalg.norm(uu))
    return np.concatenate(out)


best = None
for s0 in (1.00, 0.94, 0.88):
    res = least_squares(resid, exact * s0, method="trf",
                        bounds=(exact * 0.80, exact * 1.06),
                        x_scale=exact, xtol=1e-14, ftol=1e-14)
    rn = np.linalg.norm(res.fun)
    print("start x%.2f -> residual %.4f, ratios: %s"
          % (s0, rn / np.sqrt(len(recs)),
             " ".join("%.4f" % r for r in res.x / exact)))
    if best is None or rn < best[0]:
        best = (rn, s0, res.x)
print()
print("BEST (start x%.2f): residual %.4f" % (best[1],
                                             best[0] / np.sqrt(len(recs))))
print(" f_exact[uHz]   f_fit[uHz]   ratio")
for fe, ff in zip(exact, best[2]):
    print("   %8.2f     %8.2f    %.4f" % (fe * 1e6, ff * 1e6, ff / fe))
