#!/usr/bin/env python3
"""Compare pyastroseis Phase-1 outputs against the MATLAB references.

For each case: raw spectral solution uu (max relative error, worst
frequency column), plus time-domain record-section overlay at equator
receivers with a magnified error panel (reconstruction identical to
AstroSeis_plot.m: Ricker wavelet convolution via FFT).

Usage: python compare_py_matlab.py [Q570 Q50 Q25]
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
from pyastroseis.mesh import load_faces_mat  # noqa: E402

NT, DT, F0 = 500, 0.1, 0.3
NTR = 72

plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                     "axes.labelsize": 14, "legend.fontsize": 13})


def reconstruct(uu, faces):
    """Time-domain traces at the equator receiver ring, port of the
    reconstruction in AstroSeis_plot.m. Returns (ta, uz (ntpp, NTR))."""
    nt, dt, f0 = NT, DT, F0
    df = 1.0 / (nt * dt)
    dw = 2 * np.pi * df
    ts = 1.5 / f0
    w, t = ricker(f0, ts, dt, nt)
    wf = np.fft.ifft(w) * dt * nt

    uu2 = uu.T * wf[:, None]                       # (nt, 3N)
    uut2 = np.real(np.fft.fft(uu2, axis=0)) * dw / (2 * np.pi)

    xs0, ys0, zs0 = faces.ic[:, 0], faces.ic[:, 1], faces.ic[:, 2]
    radius = np.linalg.norm(faces.ic, axis=1)
    R = radius.mean()
    loc = faces.ic / radius[:, None]
    ngd = faces.n

    nts = round(ts / dt)
    sl = slice(nts, nt)
    ta = t[sl] - ts
    uz = np.empty((nt - nts, NTR))
    phiar = np.arange(1, NTR + 1) / NTR * 2 * np.pi
    for i, phir in enumerate(phiar):
        vecr = np.array([R * np.cos(phir), R * np.sin(phir), 0.0])
        ind = np.argmax(loc @ vecr / R)
        ub = [uut2[sl, ind], uut2[sl, ind + ngd], uut2[sl, ind + 2 * ngd]]
        vn = vecr / np.linalg.norm(vecr)
        uz[:, i] = ub[0] * vn[0] + ub[1] * vn[1] + ub[2] * vn[2]
    return ta, uz


def main():
    cases = sys.argv[1:] or ["Q570", "Q50", "Q25"]
    faces = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))
    outdir = os.path.join(ROOT, "validation")
    os.makedirs(outdir, exist_ok=True)

    print(f"{'case':6s} {'max|d_uu|/max|uu|':>18s} {'worst freq col':>15s} "
          f"{'max|d_uz|/max|uz|':>18s}")
    for case in cases:
        mat = sio.loadmat(os.path.join(ROOT, f"out_{case}.mat"))["uu"]
        py = sio.loadmat(os.path.join(ROOT, "pyout", f"out_{case}.mat"))["uu"]
        d = np.abs(py - mat)
        scale = np.abs(mat).max()
        rel = d.max() / scale
        colerr = d.max(axis=0)
        worst = int(np.argmax(colerr))

        ta, uz_m = reconstruct(mat, faces)
        _, uz_p = reconstruct(py, faces)
        duz = np.abs(uz_p - uz_m).max() / np.abs(uz_m).max()
        print(f"{case:6s} {rel:18.3e} {worst:15d} {duz:18.3e}")

        # record section overlay + magnified error panel
        sel = np.arange(0, NTR, 9)               # 8 azimuths, every 45 deg
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
        axes[0].set_title(f"radial displacement, {case.replace('Q', 'Q = ')}")
        axes[0].legend(loc="upper right", frameon=False)
        axes[1].set_title(f"difference × {mag:.0e}")
        fig.tight_layout()
        fn = os.path.join(outdir, f"pyport_{case}_waveforms.png")
        fig.savefig(fn, dpi=160, facecolor="white")
        plt.close(fig)
        print(f"       figure: {fn}")


if __name__ == "__main__":
    main()
