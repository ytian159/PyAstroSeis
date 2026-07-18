#!/usr/bin/env python3
"""PREM-staircase record sections (Z, band-passed velocity):
 left  — spectral vs BEM (independent solvers; the tipsv leg for
         this inner-core source is confirmed non-reference class)
 right — spectral with vs without the 3-km PREM ocean, with the
         ocean effect magnified.

usage: prem_recsec.py <root> <model>
writes <root>/recsec_<model>.png
"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(sys.argv[1])
MODEL = sys.argv[2]
os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402

man = json.load(open(os.path.join(ROOT, "manifest.json")))


def leg(path, name="mrr"):
    z = np.load(os.path.join(ROOT, path))
    return z["u"][[str(s) for s in z["sources"]].index(name)]


u_sp = leg("spectral_%s.npz" % MODEL)
u_oc = leg("spectral_%s_ocean.npz" % MODEL)
u_bem = leg("bem_prem_s8.npz")

syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = int(12000.0 / syn.dt) + 1
HDUR = 2000.0
hg = HDUR / sd.SOURCE_DECAY_MIMIC_TRIANGLE
stf = 0.5 * (1 + np.vectorize(math.erf)((syn.t - 1.5 * HDUR) / hg))
s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt


def bp(x):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), syn.dt)
    X[(fr < 1e-4) | (fr > 5.3e-4)] = 0.0
    return np.fft.irfft(X, len(x))


def vel(spec):
    return bp(sd.spec_to_vel(spec, syn, nout, s_spec))


tt = syn.t[:nout]
trc = {}
for i, st in enumerate(man["stations"]):
    th = math.radians(90.0 - st["lat"])
    ph = math.radians(st["lon"])
    d = np.array([math.sin(th) * math.cos(ph),
                  math.sin(th) * math.sin(ph), math.cos(th)])
    trc[i] = tuple(vel(sum(u[i, c] * d[c] for c in range(3)))
                   for u in (u_sp, u_bem, u_oc))

amp = max(np.abs(np.concatenate([trc[i][0] for i in trc])).max(),
          1e-300)
gain = 12.0
fig, axes = plt.subplots(1, 2, figsize=(15.5, 9.5), sharey=True)
for i, st in enumerate(man["stations"]):
    off = st["dist_deg"]
    s, b, o = trc[i]
    axes[0].plot(tt, off + gain * s / amp, "-", color="tab:red",
                 lw=1.2, label="spectral" if i == 0 else None)
    axes[0].plot(tt, off + gain * b / amp, "k-", lw=0.8,
                 alpha=0.75, label="BEM" if i == 0 else None)
    axes[1].plot(tt, off + gain * s / amp, "k-", lw=1.0,
                 label="no ocean" if i == 0 else None)
    axes[1].plot(tt, off + gain * o / amp, "--", color="tab:red",
                 lw=1.0, label="3-km ocean" if i == 0 else None)
    axes[1].plot(tt, off + gain * (o - s) * 10.0 / amp, "-",
                 color="tab:blue", lw=0.7,
                 label="diff x10" if i == 0 else None)
axes[0].set_title("PREM staircase, Z velocity: spectral | BEM",
                  fontsize=14)
axes[1].set_title("ocean effect (Z at ocean top, diff x10)",
                  fontsize=14)
for ax in axes:
    ax.set_xlabel("t [s]", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.tick_params(labelsize=11)
axes[0].set_ylabel("distance [deg]", fontsize=13)
axes[0].set_ylim(10, 180)
fig.tight_layout()
out = os.path.join(ROOT, "recsec_%s.png" % MODEL)
fig.savefig(out, dpi=115)
print("wrote", out)
rel = np.median([np.linalg.norm(trc[i][2] - trc[i][0])
                 / np.linalg.norm(trc[i][0]) for i in trc])
print("median ocean-effect rel RMS on Z: %.3e" % rel)
