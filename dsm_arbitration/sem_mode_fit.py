#!/usr/bin/env python3
"""Absolute SEM mode-frequency fit on the long-record ELASTIC run.

The post-ramp elastic record is static + undamped free oscillations:
    u(t) = c0 + c1*(t/T) + sum_j a_j cos(w_j t) + b_j sin(w_j t)
(c1 mops up any secular fluid-drift leakage). Variable projection:
amplitudes solved linearly per station; frequencies fitted jointly
across stations/components with scipy least_squares. Report
f_fit / f_exact per mode.

usage: sem_mode_fit.py <case_dir> <channel Z|T> <f_lo_uHz> <f_hi_uHz>
                       [start_scale=1.0]
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.environ.get("ARB_ROOT", os.path.abspath("dsm_arbitration_u3"))
os.environ["ARB_ROOT"] = ROOT
import synthesize_compare as sc                     # noqa: E402
from exact_modes import spheroidal_det, toroidal_det, find_zeros  # noqa

CASE = sys.argv[1]
CHAN = sys.argv[2]
F_LO = float(sys.argv[3]) * 1e-6
F_HI = float(sys.argv[4]) * 1e-6
SCALE0 = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
T_FIT0 = 3500.0                       # post-ramp start
STATIONS = ("ST02", "ST04", "ST06", "ST08", "ST10")

man = json.load(open(os.path.join(ROOT, "manifest.json")))

# exact mode list in the fit window
fgrid = np.linspace(3e-5, 6.2e-4, 6000)
exact = []
if CHAN == "Z":
    for l in range(1, 13):
        exact += find_zeros(spheroidal_det, l, fgrid)
else:
    for l in range(1, 10):
        exact += find_zeros(toroidal_det, l, fgrid)
exact = np.array(sorted(f for f in exact if F_LO <= f <= F_HI))
print("exact modes in window (uHz):",
      " ".join("%.2f" % (f * 1e6) for f in exact))

# load traces
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
    step = max(1, int(round(20.0 / (tt[1] - tt[0]))))   # 20-s sampling
    recs.append((tt[::step], uu[::step]))
print("fit window: t in [%.0f, %.0f] s, %d samples/station"
      % (recs[0][0][0], recs[0][0][-1], len(recs[0][0])))


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


f0 = exact * SCALE0
r0 = np.linalg.norm(resid(f0))
res = least_squares(resid, f0, method="trf",
                    bounds=(exact * 0.85, exact * 1.12),
                    x_scale=exact, xtol=1e-14, ftol=1e-14)
print("residual: start %.4f -> fit %.4f  (start scale %.3f)"
      % (r0 / np.sqrt(len(recs)),
         np.linalg.norm(res.x is not None and res.fun) /
         np.sqrt(len(recs)), SCALE0))
print(" f_exact[uHz]   f_fit[uHz]   ratio")
for fe, ff in zip(exact, res.x):
    print("   %8.2f     %8.2f    %.4f" % (fe * 1e6, ff * 1e6, ff / fe))
print("median ratio: %.4f" % np.median(res.x / exact))
