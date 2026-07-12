#!/usr/bin/env python3
"""SEM cross-check of the SH-pure meridian verdict: SEM T (validated
toroidal, SH-pure azimuths) vs the exact mode-sum. Same protocol as
tor_verdict.py (erf-STF velocity, band [1e-4, 5.3e-4] Hz).

usage: tor_verdict_sem.py <root> <tormodes_npz> <sem_case> [source]
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
TMNPZ = sys.argv[2]
SEM = sys.argv[3]
SOURCE = sys.argv[4] if len(sys.argv) > 4 else "mrt"
HDUR, T_WIN = 2000.0, 9000.0

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402

man = json.load(open(os.path.join(ROOT, "manifest.json")))
syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = int(T_WIN / syn.dt) + 1
hg = HDUR / sd.SOURCE_DECAY_MIMIC_TRIANGLE
stf = 0.5 * (1.0 + np.vectorize(math.erf)((syn.t - 1.5 * HDUR) / hg))
s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt


def bp(x):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), syn.dt)
    X[(fr < 1e-4) | (fr > 5.3e-4)] = 0.0
    return np.fft.irfft(X, len(x))


ztm = np.load(TMNPZ)
itm = [str(s) for s in ztm["sources"]].index(SOURCE)
u_tm = np.conj(ztm["u"][itm])

print("SEM cross-check vs exact SH mode-sum (meridian stations)")
print("station dist |  rel    corr    amp")
rows = []
for i, st in enumerate(man["stations"]):
    _, t_hat, north, east = sc.rotation_to_ne(
        man["source"]["lat"], man["source"]["lon"],
        st["lat"], st["lon"])
    ref = bp(sd.spec_to_vel(sum(u_tm[i, c] * t_hat[c] for c in range(3)),
                            syn, nout, s_spec))
    sN = sd.sem_read(SEM, st["name"], "N", syn.t, nout)
    sE = sd.sem_read(SEM, st["name"], "E", syn.t, nout)
    if sN is None or np.any(np.isnan(sN)):
        continue
    s = bp(sN * (t_hat @ north) + sE * (t_hat @ east))
    nd = np.linalg.norm(ref)
    rows.append(dict(station=st["name"], dist=st["dist_deg"],
                     rel=float(np.linalg.norm(s - ref) / nd),
                     corr=float(np.dot(s, ref)
                                / (np.linalg.norm(s) * nd + 1e-300)),
                     amp=float(np.linalg.norm(s) / nd)))
    r = rows[-1]
    print("%-6s %5.1f | %6.3f  %6.3f  %6.3f"
          % (r["station"], r["dist"], r["rel"], r["corr"], r["amp"]))
print("median rel %.4f, median corr %.4f, median amp %.4f"
      % tuple(np.median([[r["rel"], r["corr"], r["amp"]] for r in rows],
                        axis=0)))
out = os.path.join(ROOT, "tor_verdict_sem_%s.json" % SOURCE)
json.dump(rows, open(out, "w"), indent=1)
print("wrote", out)
