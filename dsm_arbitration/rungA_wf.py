#!/usr/bin/env python3
"""Rung-A acceptance waveforms at 50 km: band-passed velocity,
4 stations x (Z, R, T).

  Z, R : DSM tipsv | mini-tipsv | spectral (PSV part)
  T    : composite reference (tipsv PSV-T + toroidal mode-sum SH)
         | spectral TOTAL

usage: rungA_wf.py <root> <model> [source] [conj(1|0)]
writes <root>/wf_rungA_<source>.png
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
SOURCE = sys.argv[3] if len(sys.argv) > 3 else "mrt"
CONJ = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402

man = json.load(open(os.path.join(ROOT, "manifest.json")))


def leg(path, field="u"):
    z = np.load(os.path.join(ROOT, path))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    u = z[field][isrc]
    return np.conj(u) if CONJ else u


u_psv_sp = leg("spectral_%s.npz" % MODEL, "u_psv")
u_tot_sp = leg("spectral_%s.npz" % MODEL, "u")
u_mini = leg("minitipsv_%s.npz" % MODEL)
u_tor = leg("tormodes_%s.npz" % MODEL)

syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
nout = int(9000.0 / syn.dt) + 1
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
sel = ["ST00", "ST03", "ST05", "ST09"]
fig, axes = plt.subplots(len(sel), 3,
                         figsize=(16.5, 2.6 * len(sel)),
                         sharex=True)
for irow, name in enumerate(sel):
    i = [j for j, s in enumerate(man["stations"])
         if s["name"] == name][0]
    st = man["stations"][i]
    away, t_hat, north, east = sc.rotation_to_ne(
        man["source"]["lat"], man["source"]["lon"], st["lat"],
        st["lon"])
    xh = sc.unit_vec(st["lat"], st["lon"])
    _, _, u_psv = sc.read_spc(os.path.join(
        ROOT, "dsm/%s/spc/%s.%s.PSV.spc" % (MODEL, name, SOURCE)))
    for icol, (ch, vec, dsm_spec) in enumerate((
            ("Z", xh, u_psv[0] * 1000.0),
            ("R", away, u_psv[1] * 1000.0),
            ("T", t_hat, None))):
        ax = axes[irow, icol]
        if ch in ("Z", "R"):
            d = vel(dsm_spec)
            m = vel(sum(u_mini[i, c] * vec[c] for c in range(3)))
            s = vel(sum(u_psv_sp[i, c] * vec[c] for c in range(3)))
            ax.plot(tt, d, "k-", lw=2.2, label="DSM")
            ax.plot(tt, m, "-", color="tab:green", lw=1.1,
                    label="mini-tipsv")
            ax.plot(tt, s, "--", color="tab:red", lw=1.3,
                    label="spectral")
        else:
            nk = u_tor.shape[-1]
            comp = (u_psv[2][:nk] * 1000.0
                    + sum(u_tor[i, c] * vec[c] for c in range(3)))
            s = vel(sum(u_tot_sp[i, c] * vec[c] for c in range(3)))
            ax.plot(tt, vel(comp), "k-", lw=2.2,
                    label="composite ref")
            ax.plot(tt, s, "--", color="tab:red", lw=1.3,
                    label="spectral")
        ax.set_title("%s (%.0f$^\\circ$)  %s"
                     % (name, st["dist_deg"], ch), fontsize=14)
        ax.tick_params(labelsize=11)
        if irow == 0:
            ax.legend(fontsize=10, ncol=3 if icol < 2 else 2)
for ax in axes[-1]:
    ax.set_xlabel("t [s]", fontsize=13)
fig.tight_layout()
out = os.path.join(ROOT, "wf_rungA_%s.png" % SOURCE)
fig.savefig(out, dpi=110)
print("wrote", out)
