#!/usr/bin/env python3
"""Three-way waveform overlays (tipsv / SEM / BEM), band-limited
velocity, for selected station-channel pairs of the uniform3
arbitration.  usage: semdsm_waveforms.py <root> <sem_case> <leg_npz>
<source> <out_png> <ST:ch> [<ST:ch> ...]"""

import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(sys.argv[1])
SEM = os.path.abspath(sys.argv[2])
LEG = sys.argv[3]
SOURCE = sys.argv[4]
OUT = sys.argv[5]
PICKS = [p.split(":") for p in sys.argv[6:]]
HDUR = 2000.0
T_WIN = 9000.0

os.environ.setdefault("ARB_ROOT", ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthesize_compare as sc                     # noqa: E402
import semdsm_compare as sd                         # noqa: E402


def main():
    man = json.load(open(os.path.join(ROOT, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(T_WIN / syn.dt) + 1
    hg = HDUR / sd.SOURCE_DECAY_MIMIC_TRIANGLE
    stf = 0.5 * (1.0 + np.vectorize(math.erf)(
        (syn.t - 1.5 * HDUR) / hg))
    s_spec = np.fft.fft(stf * np.exp(-syn.omegai * syn.t)) * syn.dt
    z = np.load(os.path.join(ROOT, LEG))
    isrc = [str(s) for s in z["sources"]].index(SOURCE)
    u_bem = np.conj(z["u"][isrc])
    model = os.path.basename(LEG)[4:-4]
    stmap = {st["name"]: (i, st) for i, st in
             enumerate(man["stations"])}
    tt = syn.t[:nout]

    fig, axes = plt.subplots(len(PICKS), 1,
                             figsize=(11, 2.6 * len(PICKS)),
                             sharex=True)
    if len(PICKS) == 1:
        axes = [axes]
    for ax, (name, ch) in zip(axes, PICKS):
        i, st = stmap[name]
        base = os.path.join(ROOT, "dsm", model, "spc", name)
        _, _, u_psv = sc.read_spc("%s.%s.PSV.spc" % (base, SOURCE))
        u_dsm = u_psv * 1000.0
        away, t_hat, north, east = sc.rotation_to_ne(
            man["source"]["lat"], man["source"]["lon"],
            st["lat"], st["lon"])
        xhat = sc.unit_vec(st["lat"], st["lon"])
        ich = {"Z": 0, "R": 1}[ch]
        d = sd._lp(sd.spec_to_vel(u_dsm[ich], syn, nout, s_spec),
                   syn.dt)
        vb = [sd.spec_to_vel(u_bem[i][c], syn, nout, s_spec)
              for c in range(3)]
        vec = xhat if ch == "Z" else away
        b = sd._lp(sum(vb[c] * vec[c] for c in range(3)), syn.dt)
        sN = sd.sem_read(SEM, name, "N", syn.t, nout)
        sE = sd.sem_read(SEM, name, "E", syn.t, nout)
        sZ = sd.sem_read(SEM, name, "Z", syn.t, nout)
        s = sd._lp(sZ if ch == "Z" else
                   sN * (away @ north) + sE * (away @ east), syn.dt)
        ax.plot(tt, d, "k-", lw=1.8, label="DSM")
        ax.plot(tt, s, "r-", lw=1.2, label="SEM")
        ax.plot(tt, b, "b--", lw=1.2, label="BEM")
        ax.set_title("%s (%.0f$^\\circ$)  %s" % (name, st["dist_deg"],
                                                 ch), fontsize=14)
        ax.tick_params(labelsize=12)
        if ax is axes[0]:
            ax.legend(fontsize=12, ncol=3)
    axes[-1].set_xlabel("t on leg axis [s]", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
