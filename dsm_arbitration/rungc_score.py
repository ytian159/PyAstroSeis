#!/usr/bin/env python3
"""Rung-C acceptance scoring: OLD (total-formulation) vs NEW
(scattered-formulation) BEM against the validated references on the
50-km corefluid campaign.

  - per-k complex Z/R vs tipsv (minitipsv_check metric)
  - band-passed velocity R-channel waveform figure:
    DSM | mini-tipsv | BEM-old | BEM-scattered

usage: rungc_score.py <root_old> <root_new> [source]
(roots must share manifest/stations; dsm/ under root_old)"""
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_O = os.path.abspath(sys.argv[1])
ROOT_N = os.path.abspath(sys.argv[2])
SOURCE = sys.argv[3] if len(sys.argv) > 3 else "mrt"
os.environ.setdefault("ARB_ROOT", ROOT_O)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402

man = json.load(open(os.path.join(ROOT_O, "manifest.json")))
ks = np.arange(20, 139)


def leg(root):
    z = np.load(os.path.join(root, "bem_corefluid_q50.npz"))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    return np.conj(z["u"][isrc])


u_old = leg(ROOT_O)
u_new = leg(ROOT_N)

print("per-k complex rel err vs tipsv over k=20..138 (Z | R):")
print("station dist |  old-Z  new-Z |  old-R  new-R")
for i, st in enumerate(man["stations"]):
    _, _, u_psv = sc.read_spc(os.path.join(
        ROOT_O, "dsm/corefluid_q50/spc/%s.%s.PSV.spc"
        % (st["name"], SOURCE)))
    away, t_hat, north, east = sc.rotation_to_ne(
        man["source"]["lat"], man["source"]["lon"],
        st["lat"], st["lon"])
    xh = sc.unit_vec(st["lat"], st["lon"])
    row = []
    for vec, ch in ((xh, 0), (away, 1)):
        ref = u_psv[ch][ks] * 1000.0
        for u in (u_old, u_new):
            m = np.array([sum(u[i, c, k] * vec[c] for c in range(3))
                          for k in ks])
            row.append(np.linalg.norm(m - ref) / np.linalg.norm(ref))
    print("%-6s %5.1f | %6.3f %6.3f | %6.3f %6.3f"
          % (st["name"], st["dist_deg"], row[0], row[1], row[2],
             row[3]))

# waveform figure (R channel, band-passed velocity)
syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = int(9000.0 / syn.dt) + 1
HDUR = 2000.0
hg = HDUR / sd.SOURCE_DECAY_MIMIC_TRIANGLE
stf = 0.5 * (1 + np.vectorize(math.erf)((syn.t - 1.5 * HDUR) / hg))
s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt
zm = np.load(os.path.join(ROOT_O, "minitipsv_corefluid_q50.npz"))
im_ = [str(s) for s in zm["sources"]].index(SOURCE)
u_m = np.conj(zm["u"][im_])


def bp(x):
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), syn.dt)
    X[(fr < 1e-4) | (fr > 5.3e-4)] = 0.0
    return np.fft.irfft(X, len(x))


tt = syn.t[:nout]
sel = ["ST00", "ST03", "ST05", "ST09"]
fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.7 * len(sel)),
                         sharex=True)
for ax, name in zip(axes, sel):
    i = [j for j, s in enumerate(man["stations"])
         if s["name"] == name][0]
    st = man["stations"][i]
    away, t_hat, north, east = sc.rotation_to_ne(
        man["source"]["lat"], man["source"]["lon"], st["lat"],
        st["lon"])
    _, _, u_psv = sc.read_spc(os.path.join(
        ROOT_O, "dsm/corefluid_q50/spc/%s.%s.PSV.spc"
        % (name, SOURCE)))
    d = bp(sd.spec_to_vel(u_psv[1] * 1000.0, syn, nout, s_spec))
    m = bp(sd.spec_to_vel(sum(u_m[i, c] * away[c] for c in range(3)),
                          syn, nout, s_spec))
    bo = bp(sd.spec_to_vel(sum(u_old[i][c] * away[c]
                               for c in range(3)), syn, nout, s_spec))
    bn = bp(sd.spec_to_vel(sum(u_new[i][c] * away[c]
                               for c in range(3)), syn, nout, s_spec))
    ax.plot(tt, d, "k-", lw=2.2, label="DSM")
    ax.plot(tt, m, "-", color="tab:green", lw=1.1, label="mini-tipsv")
    ax.plot(tt, bo, "b--", lw=1.2, label="BEM")
    ax.plot(tt, bn, "-", color="tab:red", lw=1.4,
            label="BEM scattered")
    ax.set_title("%s (%.0f$^\\circ$)  R" % (name, st["dist_deg"]),
                 fontsize=14)
    ax.tick_params(labelsize=12)
    if ax is axes[0]:
        ax.legend(fontsize=11, ncol=4)
axes[-1].set_xlabel("t [s]", fontsize=13)
fig.tight_layout()
out = os.path.join(ROOT_N, "wf_R_rungC_%s.png" % SOURCE)
fig.savefig(out, dpi=110)
print("wrote", out)
