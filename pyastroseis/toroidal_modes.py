"""Exact toroidal (SH) reference by NORMAL-MODE SUMMATION for
uniform-layer models: full ball or shell annulus over a fluid core
(the fluid decouples from SH, so the annulus catalog is exact for
fluid-core models). Replaces the projection-route mini-tish at
shallow source depths (no incident-field evaluation, no small-|kd|
cancellation: excitation is the analytic mode strain at the source).

    u(x, w) = sum_j  [M : eps*_j(x_s)] s_j(x) / (N_j (w_j^2 - w^2))

with s_j = W_j(r) C_lm, N_j = int rho W_j^2 r^2 dr (C_lm orthonormal,
same convention as toroidal_ref.toroidal_reconstruct, so the basis
cancels against the V2-validated pipeline). For a moment tensor that
is M = M0 (r n^T + n r^T) at the source (any earth-frame Mr* source:
after rotating to the source frame e3 = r), only the eps_{r,horiz}
strain contracts:

    M : eps*_j = (W'_j - W_j/r)|_{r0} * conj(C_lm(pole)) . (M e3)_horiz

evaluated with the SAME reconstruction code (theta -> 0 limit; only
m = +-1 survive). Attenuation (uniform shear Q in the solid): first-
order causal constant-Q mode perturbation, reference 1 Hz:

    w_j -> w_j (1 + ln(f_j/1 Hz)/(pi Q) + i q_sign/(2 Q))

q_sign is calibrated once against the V2 mini-tish anchor and frozen
(house pattern; it encodes the e^{+-iwt} convention of the pipeline).
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from .toroidal_ref import source_frame, toroidal_reconstruct


def _f(kind, l, z):
    if kind == "j":
        return spherical_jn(l, z), spherical_jn(l, z, derivative=True)
    return spherical_yn(l, z), spherical_yn(l, z, derivative=True)


def _traction_factor(l, k, r, coef):
    """(W' - W/r) for W = sum coef_i f_i(k r)."""
    out = 0.0
    for (kind, c) in coef:
        f, fp = _f(kind, l, k * r)
        out += c * (k * fp - f / r)
    return out


def _w_val(l, k, r, coef):
    return sum(c * _f(kind, l, k * r)[0] for kind, c in coef)


def mode_catalog(l, vs, a, b=None, fmax=1.0e-2, nscan=1600):
    """Toroidal eigenfrequencies and radial data for one l.
    Returns list of dicts with w, coef, Wa (=W(a)), norm-integrand
    grid data deferred; b=None -> full ball (j only)."""
    # scan window: from below the branch start to fmax
    f_lo = max(2.0e-5, 0.6 * vs * np.sqrt(max(l * (l + 1.0) - 2.0, 1.0))
               / (2.0 * np.pi * a))
    if f_lo >= fmax:
        return []
    fs = np.linspace(f_lo, fmax, nscan)

    def det(f):
        w = 2.0 * np.pi * f
        k = w / vs
        tj_a = _traction_factor(l, k, a, [("j", 1.0)])
        if b is None:
            return tj_a
        ty_a = _traction_factor(l, k, a, [("y", 1.0)])
        tj_b = _traction_factor(l, k, b, [("j", 1.0)])
        ty_b = _traction_factor(l, k, b, [("y", 1.0)])
        d = tj_a * ty_b - ty_a * tj_b
        n = abs(tj_a * ty_b) + abs(ty_a * tj_b) + 1e-300
        return d / n

    vals = np.array([det(f) for f in fs])
    modes = []
    for i in range(len(fs) - 1):
        if not (np.isfinite(vals[i]) and np.isfinite(vals[i + 1])):
            continue
        if vals[i] * vals[i + 1] < 0:
            lo, hi = fs[i], fs[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if det(lo) * det(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            f_j = 0.5 * (lo + hi)
            w = 2.0 * np.pi * f_j
            k = w / vs
            if b is None:
                coef = [("j", 1.0)]
            else:
                # T(b) = 0: (A, B) ~ (Ty(b), -Tj(b))
                A = _traction_factor(l, k, b, [("y", 1.0)])
                B = -_traction_factor(l, k, b, [("j", 1.0)])
                s = max(abs(A), abs(B))
                coef = [("j", A / s), ("y", B / s)]
            modes.append(dict(l=l, w=w, k=k, coef=coef))
    return modes


def finalize_modes(modes, rho, vs, a, b=None, r0=None, nr=800):
    """Adds Wa, source factor (W'-W/r)(r0), and norm N to each mode."""
    b_eff = 0.0 if b is None else b
    rr = np.linspace(b_eff if b else 1.0, a, nr)   # ball: avoid r=0
    for md in modes:
        l, k, coef = md["l"], md["k"], md["coef"]
        Wr = sum(c * spherical_jn(l, k * rr) if kind == "j"
                 else c * spherical_yn(l, k * rr)
                 for kind, c in coef)
        md["Wa"] = _w_val(l, k, a, coef)
        md["src"] = _traction_factor(l, k, r0, coef)
        md["N"] = rho * np.trapz(Wr * Wr * rr * rr, rr)
    return modes


def build_catalog(vs, rho, a, b, r0, fmax=1.0e-2, lmax=None):
    if lmax is None:
        lmax = int(fmax * 2.0 * np.pi * a / vs) + 12
    cat = []
    for l in range(1, lmax + 1):
        mds = mode_catalog(l, vs, a, b, fmax=fmax)
        finalize_modes(mds, rho, vs, a, b, r0=r0)
        cat += mds
    return cat


def _pole_coupling(lmax):
    """conj(C_lm) horizontal Cartesian components at the source-frame
    pole, for m = -1, +1: returns dict m -> (Cx*, Cy*) via the SAME
    reconstruction code evaluated at theta -> 0."""
    eps = 1.0e-7
    dirs = np.array([[np.sin(eps), 0.0, np.cos(eps)]])
    out = {}
    for m in (-1, 1):
        vecs = np.zeros((lmax + 1, 2), dtype=complex)
        for l in range(1, lmax + 1):
            W = np.zeros((lmax + 1, 9), dtype=complex)
            W[l, m + 4] = 1.0
            u = toroidal_reconstruct(W, dirs)[0]
            vecs[l] = np.conj(u[:2])
        out[m] = vecs
    return out


def toroidal_mode_spectra(model, src_xyz, M, w_arr, station_dirs,
                          Q=None, q_sign=-1.0, fmax=1.0e-2):
    """SH spectra (nst, 3, nw) for one moment tensor M (3,3 earth
    frame) at src_xyz, at complex frequencies w_arr, by mode sum.
    model: dict(vs, rho, a, b) with b=None for the full ball.
    Q: shear Q of the solid (None = elastic); causal 1-Hz reference.
    """
    vs, rho, a, b = (model["vs"], model["rho"], model["a"],
                     model.get("b"))
    r0 = float(np.linalg.norm(src_xyz))
    cat = build_catalog(vs, rho, a, b, r0, fmax=fmax)
    lmax = max(md["l"] for md in cat)
    Qrot = source_frame(src_xyz)
    M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
    h = M_sf @ np.array([0.0, 0.0, 1.0])       # (M e3); take horiz
    pole = _pole_coupling(lmax)

    w_arr = np.asarray(w_arr, dtype=complex)
    Wlm = np.zeros((len(w_arr), lmax + 1, 9), dtype=complex)
    for md in cat:
        l = md["l"]
        wj = md["w"]
        if Q:
            lf = np.log(wj / (2.0 * np.pi) / 1.0)
            wj = wj * (1.0 + lf / (np.pi * Q)
                       + 1j * q_sign / (2.0 * Q))
        den = wj * wj - w_arr * w_arr          # (nw,)
        for m in (-1, 1):
            Cst = pole[m][l]                   # conj(C) at pole (x,y)
            E = md["src"] * (Cst[0] * h[0] + Cst[1] * h[1])
            Wlm[:, l, m + 4] += E * md["Wa"] / (md["N"] * den)

    # stations: rotate to source frame, reconstruct, rotate back
    dirs_sf = station_dirs @ Qrot.T
    out = np.zeros((len(station_dirs), 3, len(w_arr)), dtype=complex)
    for i, w in enumerate(w_arr):
        u_sf = toroidal_reconstruct(Wlm[i], dirs_sf)
        out[:, :, i] = u_sf @ Qrot
    return out
