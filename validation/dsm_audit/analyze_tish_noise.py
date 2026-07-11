#!/usr/bin/env python3
"""Evidence analysis for the DSM tish shallow-source SH artifact
(docs/fast_methods_notes.md section 8; docs/moment_fitted_rhs.md
section 6). Uses the maxL-truncated tish rebuilds in tish_ltrunc/
(see dsm_audit.sh for the build recipe: sed maxL in tish.f90;
untruncated rebuild == campaign spc to 1e-13).

Outputs: printed tables + tish_noise_evidence.png (band-increment
spectrum vs the physical excitation bound; T-trace non-convergence
at ST05 with the BEM trace for scale).
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.dirname(os.path.dirname(HERE))
R2 = os.path.join(Q, "dsm_arbitration_src50_ref2")
sys.path.insert(0, os.path.join(Q, "dsm_arbitration"))
os.environ.setdefault("ARB_ROOT", R2)
import synthesize_compare as sc                    # noqa: E402

LS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
STS = ["ST00", "ST05", "ST11"]
Q_RATIO = 6321.0 / 6371.0


def spec(run, st):
    p = (os.path.join(HERE, "tish_ltrunc", "run_%s" % run, "spc",
                      "%s.mrt.SH.spc" % st) if run else
         os.path.join(R2, "dsm", "homog_q50", "spc",
                      "%s.mrt.SH.spc" % st))
    return sc.read_spc(p)[2]


def main():
    man = json.load(open(os.path.join(R2, "manifest.json")))
    syn = sc.Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    nout = int(sc.METRICS_T / syn.dt) + 1

    print("band-increment norms of the T spectra (all 138 harmonics)"
          " vs physical excitation bound (2l+1)(r0/R)^l:")
    incr = {st: [] for st in STS}
    bound = []
    prev = {st: spec("l%d" % LS[0], st) for st in STS}
    for L in LS[1:]:
        cur = {st: spec("l%d" % L, st) for st in STS}
        for st in STS:
            incr[st].append(np.linalg.norm(cur[st][2] - prev[st][2]))
        bound.append((2.0 * L + 1.0) * Q_RATIO ** L)
        prev = cur
    full = {st: spec(None, st) for st in STS}
    for st in STS:
        incr[st].append(np.linalg.norm(full[st][2] - prev[st][2]))
    bound.append(bound[-1])   # placeholder for the (16384, full] band
    for i, L in enumerate(LS[1:] + ["full"]):
        print("  band <=%6s : bound %8.1e | " % (L, bound[i])
              + "  ".join("%8.2e" % incr[st][i] for st in STS))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))

    ub = np.array(LS[1:], dtype=float)
    b = np.array(bound[:-1])
    ax[0].loglog(ub, b / b[0] * incr["ST00"][0], "k--",
                 label="physical bound (scaled)")
    for st in STS:
        ax[0].loglog(ub, incr[st][:-1], "o-", label=st)
    ax[0].set_xlabel("upper angular order of band")
    ax[0].set_ylabel("T band-increment norm [m]")
    ax[0].set_title("DSM SH partial-sum increments, 50-km source")
    ax[0].legend(fontsize=12)
    ax[0].grid(alpha=0.3)

    st = "ST05"
    t = syn.t[:nout] / 1e3
    for run, lab in [(None, "DSM full"), ("l4096", "DSM l<=4096"),
                     ("l1024", "DSM l<=1024")]:
        ax[1].plot(t, syn.to_time(spec(run, st)[2] * 1e3)[:nout],
                   label=lab)
    z = np.load(os.path.join(R2, "bem_homog_q50.npz"))
    bem = np.conj(z["u"][list(z["sources"]).index("mrt")])
    i5 = [i for i, s in enumerate(man["stations"])
          if s["name"] == st][0]
    stm = man["stations"][i5]
    _, t_hat, _, _ = sc.rotation_to_ne(man["source"]["lat"],
                                       man["source"]["lon"],
                                       stm["lat"], stm["lon"])
    bt = sum(syn.to_time(bem[i5, c]) * t_hat[c] for c in range(3))
    ax[1].plot(t, bt[:nout], "k", lw=2, label="BEM")
    ax[1].set_xlabel("time [1000 s]")
    ax[1].set_ylabel("T displacement [m]")
    ax[1].set_title("mrt T at %s (90 deg)" % st)
    ax[1].legend(fontsize=12)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(HERE, "tish_noise_evidence.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
