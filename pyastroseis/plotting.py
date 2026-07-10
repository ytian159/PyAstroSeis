"""Post-processing: time-domain reconstruction and record sections.

The spectral solution uu is convolved with a Ricker wavelet in the
frequency domain and transformed back, exactly as AstroSeis_plot.m.
"""

import numpy as np

from .greens import ricker


def time_traces(uu, nt, dt, f0):
    """Convolve the spectral solution with a Ricker(f0) wavelet and
    return real time-domain traces (nt, 3N) plus the time axis shifted
    by the wavelet delay ts=1.5/f0 (AstroSeis_plot.m convention)."""
    df = 1.0 / (nt * dt)
    dw = 2 * np.pi * df
    ts = 1.5 / f0
    w, t = ricker(f0, ts, dt, nt)
    wf = np.fft.ifft(w) * dt * nt
    uu2 = uu.T * wf[:, None]
    uut = np.real(np.fft.fft(uu2, axis=0)) * dw / (2 * np.pi)
    nts = round(ts / dt)
    return t[nts:] - ts, uut[nts:]


def ring_receivers(faces, ntr=72, lat_deg=0.0):
    """Indices of the mesh faces closest to an ntr-point ring of
    receivers at the given latitude (default: equator)."""
    radius = np.linalg.norm(faces.ic, axis=1)
    R = radius.mean()
    loc = faces.ic / radius[:, None]
    th = (90.0 - lat_deg) * np.pi / 180.0
    idx = np.empty(ntr, dtype=int)
    dirs = np.empty((ntr, 3))
    for i, phir in enumerate(np.arange(1, ntr + 1) / ntr * 2 * np.pi):
        vec = np.array([np.sin(th) * np.cos(phir),
                        np.sin(th) * np.sin(phir), np.cos(th)])
        idx[i] = int(np.argmax(loc @ (R * vec) / R))
        dirs[i] = vec
    return idx, dirs


def record_section(uu, faces, nt, dt, f0, ntr=72, component="radial",
                   lat_deg=0.0):
    """Record section at a receiver ring. Returns (ta, traces (nt', ntr)).
    component: 'radial' | 'x' | 'y' | 'z'."""
    ta, uut = time_traces(uu, nt, dt, f0)
    idx, dirs = ring_receivers(faces, ntr, lat_deg)
    n = faces.n
    ux, uy, uz = uut[:, idx], uut[:, idx + n], uut[:, idx + 2 * n]
    if component == "radial":
        tr = ux * dirs[:, 0] + uy * dirs[:, 1] + uz * dirs[:, 2]
    elif component in ("x", "y", "z"):
        tr = {"x": ux, "y": uy, "z": uz}[component]
    else:
        raise ValueError("component must be radial|x|y|z")
    return ta, tr


def plot_record_section(uu, faces, nt, dt, f0, out_png, ntr=72,
                        step=9, component="radial", title=None):
    """Save a wiggle-style record-section figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15,
                         "axes.labelsize": 14})
    ta, tr = record_section(uu, faces, nt, dt, f0, ntr, component)
    sel = np.arange(0, ntr, step)
    amp = np.abs(tr[:, sel]).max()
    fig, ax = plt.subplots(figsize=(9, 7))
    for k, i in enumerate(sel):
        ax.plot(ta, tr[:, i] / amp + k * 2.2, "-", color="#0072B2", lw=1.3)
    ax.set_yticks([k * 2.2 for k in range(len(sel))])
    ax.set_yticklabels([f"{(i + 1) * (360 // ntr)}°" for i in sel])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("receiver longitude")
    ax.set_xlim(ta[0], ta[-1])
    ax.set_title(title or f"{component} displacement")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, facecolor="white")
    plt.close(fig)
    return out_png
