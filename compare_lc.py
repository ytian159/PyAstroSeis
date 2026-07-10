#!/usr/bin/env python3
"""Compare the pyastroseis liquid-core run against the MATLAB reference
(examples/shifted_liquid_core/uu_liqcore_bem20km_shift.mat).

Usage: python compare_lc.py pyout/uu_liqcore_bem20km_shift.mat
"""

import os
import sys

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pyastroseis.greens import ricker  # noqa: E402
from pyastroseis.liquidcore import load_layers_mat  # noqa: E402

NT, DT, F0 = 500, 0.1, 0.3
NTR = 72

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 14, "legend.fontsize": 13})


def reconstruct(uu, faces):
    nt, dt, f0 = NT, DT, F0
    df = 1.0 / (nt * dt)
    dw = 2 * np.pi * df
    ts = 1.5 / f0
    w, t = ricker(f0, ts, dt, nt)
    wf = np.fft.ifft(w) * dt * nt
    uu2 = uu.T * wf[:, None]
    uut2 = np.real(np.fft.fft(uu2, axis=0)) * dw / (2 * np.pi)

    radius = np.linalg.norm(faces.ic, axis=1)
    R = radius.mean()
    loc = faces.ic / radius[:, None]
    ngd = faces.n
    nts = round(ts / dt)
    sl = slice(nts, nt)
    ta = t[sl] - ts
    uz = np.empty((nt - nts, NTR))
    for i, phir in enumerate(np.arange(1, NTR + 1) / NTR * 2 * np.pi):
        vecr = np.array([R * np.cos(phir), R * np.sin(phir), 0.0])
        ind = np.argmax(loc @ vecr / R)
        vn = vecr / np.linalg.norm(vecr)
        uz[:, i] = (uut2[sl, ind] * vn[0] + uut2[sl, ind + ngd] * vn[1]
                    + uut2[sl, ind + 2 * ngd] * vn[2])
    return ta, uz


def main():
    py_file = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "examples", "shifted_liquid_core", "pyout",
                     "uu_liqcore_bem20km_shift.mat")
    ref = sio.loadmat(os.path.join(ROOT, "examples", "shifted_liquid_core",
                                   "uu_liqcore_bem20km_shift.mat"))
    py = sio.loadmat(py_file)
    layers = load_layers_mat(os.path.join(
        ROOT, "examples", "shifted_liquid_core", "layers_shift.mat"))
    faces_outer = layers[1]

    # NOTE: the shipped reference has 75 computed frequencies (fmax
    # implies f0=0.5) while inp_sft_lc specifies f0=0.3 (44
    # frequencies) — the reference predates the shipped parameter file.
    # Compare on the commonly computed columns only, and report the
    # column-count mismatch explicitly.
    nzr = np.nonzero(np.linalg.norm(ref["u1"], axis=0))[0]
    nzp = np.nonzero(np.linalg.norm(py["u1"], axis=0))[0]
    common = np.intersect1d(nzr, nzp)
    print(f"computed frequency columns: reference {nzr.size}, "
          f"python {nzp.size}, common {common.size}")
    print(f"{'field':6s} {'max col rel err (common cols)':>30s}")
    for key in ("up", "u2", "u1"):
        r, p = ref[key], py[key]
        rel = max(np.linalg.norm(p[:, c] - r[:, c]) / np.linalg.norm(r[:, c])
                  for c in common)
        print(f"{key:6s} {rel:30.3e}")

    ta, uz_m = reconstruct(ref["u1"], faces_outer)
    _, uz_p = reconstruct(py["u1"], faces_outer)
    duz = np.abs(uz_p - uz_m).max() / np.abs(uz_m).max()
    print(f"u1 surface waveforms (radial, equator ring): "
          f"max rel err = {duz:.3e}")

    sel = np.arange(0, NTR, 9)
    amp = np.abs(uz_m[:, sel]).max()
    gain = np.abs(uz_p[:, sel] - uz_m[:, sel]).max()
    mag = 10 ** np.floor(np.log10(amp / gain)) if gain > 0 else 1e16
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for k, i in enumerate(sel):
        off = k * 2.2
        axes[0].plot(ta, uz_m[:, i] / amp + off, "-", color="#0072B2",
                     lw=1.4, label="MATLAB" if k == 0 else None)
        axes[0].plot(ta, uz_p[:, i] / amp + off, "--", color="#D55E00",
                     lw=1.1, label="Python" if k == 0 else None)
        axes[1].plot(ta, (uz_p[:, i] - uz_m[:, i]) * mag / amp + off,
                     "-", color="#009E73", lw=1.1)
    yt = [k * 2.2 for k in range(len(sel))]
    ylab = [f"{(i + 1) * 5}°" for i in sel]
    for ax in axes:
        ax.set_yticks(yt)
        ax.set_yticklabels(ylab)
        ax.set_xlabel("time (s)")
        ax.set_xlim(ta[0], ta[-1])
    axes[0].set_ylabel("receiver longitude")
    axes[0].set_title("liquid core: surface radial displacement")
    axes[0].legend(loc="upper right", frameon=False)
    axes[1].set_title(f"difference × {mag:.0e}")
    fig.tight_layout()
    outdir = os.path.join(ROOT, "validation")
    os.makedirs(outdir, exist_ok=True)
    fn = os.path.join(outdir, "pyport_liquidcore_waveforms.png")
    fig.savefig(fn, dpi=160, facecolor="white")
    print(f"figure: {fn}")


if __name__ == "__main__":
    main()
