#!/usr/bin/env python3
"""Rung 3 — ensemble figures: record sections of the base model vs a
perturbed member (overlay) with magnified difference panels, plus the
Y20 amplitude-scaling check in the time domain (0-24 ks window)."""

import json
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("R3_ROOT", SCRIPT_DIR)
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PKG, "dsm_arbitration"))

from synthesize_compare import Synth, rotation_to_ne, unit_vec  # noqa: E402

T_OUT = 60000.0
T_MET = 24000.0


def load(name):
    z = np.load(os.path.join(ROOT, "ens_%s.npz" % name))
    return z["u"]


def to_comp(u_st, st, src, comp):
    """u_st: (3, nk) cartesian spectra -> requested component."""
    away, t_hat, north, east = rotation_to_ne(
        src["lat"], src["lon"], st["lat"], st["lon"])
    xhat = unit_vec(st["lat"], st["lon"])
    v = {"Z": xhat, "T": t_hat, "R": away}[comp]
    return u_st[0] * v[0] + u_st[1] * v[1] + u_st[2] * v[2]


def record_section(member, source_i, source, comp, man, syn, u0, um,
                   path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stations = sorted(man["stations"], key=lambda s: s["dist_deg"])
    src = man["source"]
    nout = int(T_OUT / syn.dt) + 1
    t = syn.t[:nout] / 1e3

    tr0, trm = {}, {}
    for i, st in enumerate(man["stations"]):
        s0 = syn.to_time(np.conj(to_comp(u0[source_i, i], st, src,
                                         comp)))[:nout]
        sm = syn.to_time(np.conj(to_comp(um[source_i, i], st, src,
                                         comp)))[:nout]
        tr0[st["name"]], trm[st["name"]] = s0, sm

    gmax = max(np.max(np.abs(v)) for v in tr0.values())
    errs = [np.max(np.abs(trm[s["name"]] - tr0[s["name"]]))
            for s in stations]
    mag = max(1, int(round(gmax / max(errs) / 2.0)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    for st in stations:
        d0 = tr0[st["name"]]
        dm = trm[st["name"]]
        off = st["dist_deg"]
        sc = 8.0 / gmax
        axes[0].plot(t, d0 * sc + off, "k-", lw=1.2)
        axes[0].plot(t, dm * sc + off, "r--", lw=1.0)
        axes[1].plot(t, (dm - d0) * sc * mag + off, "b-", lw=0.9)
    axes[0].plot([], [], "k-", lw=1.2, label="base")
    axes[0].plot([], [], "r--", lw=1.0, label=member)
    axes[0].legend(fontsize=14, loc="upper right")
    axes[0].set_title("%s vs base, %s, %s component"
                      % (member, source, comp), fontsize=15)
    axes[1].set_title("difference $\\times$%d" % mag, fontsize=15)
    for ax in axes:
        ax.set_xlabel("time [$10^3$ s]", fontsize=14)
        ax.tick_params(labelsize=13)
    axes[0].set_ylabel("epicentral distance [deg]", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote %s (difference magnification x%d)" % (path, mag))


def main():
    man = json.load(open(os.path.join(ROOT, "ens_manifest.json")))
    syn = Synth(man["tlen"], man["omegai_1_per_s"], man["imax"])
    u0 = load("base")

    members = {m: load(m) for m in man["members"] if m != "amp0"}
    for member, um in members.items():
        record_section(member, 0, "mrr", "Z", man, syn, u0, um,
                       os.path.join(ROOT, "ens_%s_mrr_Z.png" % member))
        record_section(member, 1, "mrt", "T", man, syn, u0, um,
                       os.path.join(ROOT, "ens_%s_mrt_T.png" % member))

    # time-domain Y20 scaling over the metric window
    nmet = int(T_MET / syn.dt) + 1
    src = man["source"]

    def dnorm(um):
        acc0, accd = 0.0, 0.0
        for si in range(u0.shape[0]):
            for i, st in enumerate(man["stations"]):
                for comp in ("Z", "T", "R"):
                    s0 = syn.to_time(np.conj(to_comp(
                        u0[si, i], st, src, comp)))[:nmet]
                    sm = syn.to_time(np.conj(to_comp(
                        um[si, i], st, src, comp)))[:nmet]
                    acc0 += float(np.sum(s0 ** 2))
                    accd += float(np.sum((sm - s0) ** 2))
        return np.sqrt(accd), np.sqrt(acc0)

    d1, n0 = dnorm(members["y20_2500"])
    d2, _ = dnorm(members["y20_5000"])
    dr, _ = dnorm(members["rand_l4_5000"])
    print("time-domain (0-%.0f s): |du|/|u0|  Y20 2.5 km %.3e, "
          "Y20 5 km %.3e, random l<=4 5 km %.3e" %
          (T_MET, d1 / n0, d2 / n0, dr / n0))
    print("Y20 time-domain amplitude scaling: %.4f (linear = 2)"
          % (d2 / d1))


if __name__ == "__main__":
    main()
