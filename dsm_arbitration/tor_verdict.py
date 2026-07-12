#!/usr/bin/env python3
"""Composite T-channel verdict at the campaign band: reference T =
exact toroidal mode-sum (SH) + validated tipsv PSV-T (spheroidal
leakage into t_hat), candidate = BEM total-field T. erf-STF velocity
scoring, band-passed [1e-4, 5.3e-4] Hz (same protocol as
semdsm_compare). Also reports the SH-only and PSV-only reference
split per station.

usage: tor_verdict.py <root> <model> <tormodes_npz> <tm_conj:0|1>
                      [source] [hdur] [t_win]
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(sys.argv[1])
MODEL = sys.argv[2]
TMNPZ = sys.argv[3]
TM_CONJ = bool(int(sys.argv[4]))
SOURCE = sys.argv[5] if len(sys.argv) > 5 else "mrt"
HDUR = float(sys.argv[6]) if len(sys.argv) > 6 else 2000.0
T_WIN = float(sys.argv[7]) if len(sys.argv) > 7 else 9000.0

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
u_tm = np.conj(ztm["u"][itm]) if TM_CONJ else ztm["u"][itm]

zb = np.load(os.path.join(ROOT, "bem_%s.npz" % MODEL))
ib = [str(s) for s in zb["sources"]].index(SOURCE)
u_bem = np.conj(zb["u"][ib])                    # frozen BEM conj

print("T verdict (%s, %s): ref = mode-sum SH + tipsv PSV-T"
      % (MODEL, SOURCE))
print("station dist | SH/PSV mix |  rel    corr    amp")
rows = []
for i, st in enumerate(man["stations"]):
    _, t_hat, _, _ = sc.rotation_to_ne(
        man["source"]["lat"], man["source"]["lon"],
        st["lat"], st["lon"])
    # SH part: mode-sum, project to t_hat
    sh = bp(sd.spec_to_vel(sum(u_tm[i, c] * t_hat[c] for c in range(3)),
                           syn, nout, s_spec))
    # PSV part: tipsv T channel (spc ch 2)
    base = os.path.join(ROOT, "dsm", MODEL, "spc", st["name"])
    _, _, u_psv = sc.read_spc("%s.%s.PSV.spc" % (base, SOURCE))
    psv = bp(sd.spec_to_vel(u_psv[2] * 1000.0, syn, nout, s_spec))
    ref = sh + psv
    b = bp(sd.spec_to_vel(sum(u_bem[i][c] * t_hat[c] for c in range(3)),
                          syn, nout, s_spec))
    nd = np.linalg.norm(ref)
    rows.append(dict(
        station=st["name"], dist=st["dist_deg"],
        mix=float(np.linalg.norm(sh) / (np.linalg.norm(psv) + 1e-300)),
        rel=float(np.linalg.norm(b - ref) / nd),
        corr=float(np.dot(b, ref) / (np.linalg.norm(b) * nd + 1e-300)),
        amp=float(np.linalg.norm(b) / nd)))
    r = rows[-1]
    print("%-6s %5.1f |   %5.2f    | %6.3f  %6.3f  %6.3f"
          % (r["station"], r["dist"], r["mix"], r["rel"], r["corr"],
             r["amp"]))
med = np.median([[r["rel"], r["corr"], r["amp"]] for r in rows], axis=0)
print("median rel %.4f, median corr %.4f, median amp %.4f"
      % tuple(med))
out = os.path.join(ROOT, "tor_verdict_%s_%s.json" % (MODEL, SOURCE))
json.dump(rows, open(out, "w"), indent=1)
print("wrote", out)
