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


def mode_catalog(l, vs, a, b=None, fmax=1.0e-2, nscan=1600,
                 f_window=None):
    """Toroidal eigenfrequencies and radial data for one l.
    Returns list of dicts with w, coef, Wa (=W(a)), norm-integrand
    grid data deferred; b=None -> full ball (j only). f_window
    overrides the scan range (f_lo, f_hi); vectorized scan."""
    if f_window is None:
        f_lo = max(2.0e-5,
                   0.6 * vs * np.sqrt(max(l * (l + 1.0) - 2.0, 1.0))
                   / (2.0 * np.pi * a))
        f_hi = fmax
    else:
        f_lo, f_hi = f_window
    if f_lo >= f_hi:
        return []
    fs = np.linspace(f_lo, f_hi, nscan)

    def det_arr(farr):
        w = 2.0 * np.pi * np.asarray(farr)
        k = w / vs
        za = k * a
        ja, jap = spherical_jn(l, za), spherical_jn(l, za, True)
        tj_a = k * jap - ja / a
        if b is None:
            return tj_a
        ya, yap = spherical_yn(l, za), spherical_yn(l, za, True)
        ty_a = k * yap - ya / a
        zb = k * b
        jb, jbp = spherical_jn(l, zb), spherical_jn(l, zb, True)
        yb, ybp = spherical_yn(l, zb), spherical_yn(l, zb, True)
        tj_b = k * jbp - jb / b
        ty_b = k * ybp - yb / b
        d = tj_a * ty_b - ty_a * tj_b
        n = np.abs(tj_a * ty_b) + np.abs(ty_a * tj_b) + 1e-300
        return d / n

    vals = det_arr(fs)

    def det(f):
        return float(det_arr(np.array([f]))[0])

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
    """Adds Wa, source factor (W'-W/r)(r0), and norm N to each mode.
    Full-range norm grid; the point count scales with l so the
    surface Airy lobe (~a l^{-2/3}) of trapped modes stays resolved."""
    r_lo = b if b is not None else 1.0
    for md in modes:
        l, k, coef = md["l"], md["k"], md["coef"]
        n = max(nr, 6 * l)
        rr = np.linspace(r_lo, a, n)
        Wr = sum(c * spherical_jn(l, k * rr) if kind == "j"
                 else c * spherical_yn(l, k * rr)
                 for kind, c in coef)
        md["Wa"] = _w_val(l, k, a, coef)
        md["src"] = _traction_factor(l, k, r0, coef)
        md["N"] = rho * np.trapz(Wr * Wr * rr * rr, rr)
    return modes


def build_catalog(vs, rho, a, b, r0, fmax=1.0e-2, lmax=None,
                  n_keep=8, l_switch=80):
    """Hybrid catalog: for l <= l_switch, annulus modes up to fmax
    (in-band physics needs every branch); for l > l_switch the modes
    are surface-trapped (|W(b)/W(a)| < 1e-12 by l ~ 80, and y_l(k b)
    overflows at high l), so use the j-only ball basis and keep the
    first n_keep overtones per l — their 1/(w_j^2 - w^2) tails build
    the quasi-static field, which has an (r0/a)^l l-spectrum
    (e-folding l ~ a/(a - r0)); lmax must cover several e-foldings."""
    if lmax is None:
        lmax = int(fmax * 2.0 * np.pi * a / vs) + 12
    cat = []
    for l in range(1, min(lmax, l_switch) + 1):
        mds = mode_catalog(l, vs, a, b, fmax=fmax)
        finalize_modes(mds, rho, vs, a, b, r0=r0)
        cat += mds
    dfo = vs / (2.0 * (a - (b or 0.0)))          # overtone spacing
    for l in range(l_switch + 1, lmax + 1):
        f0 = vs * np.sqrt(max(l * (l + 1.0) - 2.0, 1.0)) \
            / (2.0 * np.pi * a)
        win = (0.75 * f0, f0 + (n_keep + 1.5) * dfo)
        mds = mode_catalog(l, vs, a, None, f_window=win,
                           nscan=max(400, int(60 * n_keep)))
        mds = mds[:n_keep]
        finalize_modes(mds, rho, vs, a, None, r0=r0)
        cat += mds
    return cat


def static_factor(l, a, b, r0):
    """Closed-form STATIC toroidal response factor: W_s(a) for the
    strain-type point source with unit strength M0*D/mu = 1, i.e.
    returns W_s(a)*mu/(M0*D). Solutions r^l and r^{-l-1} with
    tractions T = mu(l-1)r^{l-1} and -mu(l+2)r^{-l-2}; region I
    (below r0) satisfies T(b)=0 (ball: pure r^l), region II T(a)=0;
    jumps at r0 (from matching mu(W'' + 2W'/r - LW/r^2) = -f_M with
    int W f_M r^2 dr = M0 D (W'(r0) - W(r0)/r0)): [W] = 1/r0^2,
    [W'] = +1/r0^3 (units of M0 D / mu; self-verified against the
    Cesaro-accelerated 1-D mode sum). l = 1 is degenerate (rigid
    rotation) - caller must keep the plain mode sum there. All
    algebra in ratios of (r/r0), overflow-safe to arbitrary l."""
    if l < 2:
        raise ValueError("static_factor needs l >= 2")
    # region I: u1 = (r/r0)^l + alpha (r/r0)^{-l-1}
    if b is None or b <= 0.0:
        alpha = 0.0
    else:
        # T(b) = 0: (l-1)(b/r0)^{l-1} + alpha *(-(l+2))(b/r0)^{-l-2}=0
        alpha = (l - 1.0) / (l + 2.0) * (b / r0) ** (2 * l + 1)
    # region II: u2 = (r/r0)^{-l-1} + beta (r/r0)^l with T(a)=0
    beta = (l + 2.0) / (l - 1.0) * (a / r0) ** (-(2 * l + 1))
    # continuity system at r = r0 (x = r/r0 = 1):
    #   c2*u2(1) - c1*u1(1) = [W]  = 1/r0^2
    #   c2*u2'(1) - c1*u1'(1) = [W'] = -2/r0^3   (d/dr = d/dx / r0)
    u1, du1 = 1.0 + alpha, (l - alpha * (l + 1.0)) / r0
    u2, du2 = 1.0 + beta, (-(l + 1.0) + beta * l) / r0
    det = u2 * (-du1) - (-u1) * du2
    JW, JWp = 1.0 / r0 ** 2, 1.0 / r0 ** 3
    c2 = (JW * (-du1) - (-u1) * JWp) / det
    # W_s(a) = c2 * u2(a/r0)
    x = a / r0
    return c2 * (x ** (-(l + 1.0)) + beta * x ** l)


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
                          Q=None, q_sign=-1.0, fmax=1.0e-2,
                          lmax=None, n_keep=8, l_switch=80):
    """SH spectra (nst, 3, nw) for one moment tensor M (3,3 earth
    frame) at src_xyz, at complex frequencies w_arr, by mode sum.
    model: dict(vs, rho, a, b) with b=None for the full ball.
    Q: shear Q of the solid (None = elastic); causal 1-Hz reference.
    lmax: cover several e-foldings of (r0/a)^l for the quasi-static
    field (default: 6 e-foldings, capped at 2000)."""
    vs, rho, a, b = (model["vs"], model["rho"], model["a"],
                     model.get("b"))
    mu0 = rho * vs * vs
    r0 = float(np.linalg.norm(src_xyz))
    if lmax is None:
        lmax = int(min(2000, max(200, 6.0 * a / (a - r0))))
    lmax_dyn = min(lmax, 250)
    cat = build_catalog(vs, rho, a, b, r0, fmax=fmax, lmax=lmax_dyn,
                        n_keep=n_keep, l_switch=l_switch)
    Qrot = source_frame(src_xyz)
    M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
    h = M_sf @ np.array([0.0, 0.0, 1.0])       # (M e3); take horiz
    pole = _pole_coupling(lmax)

    w_arr = np.asarray(w_arr, dtype=complex)
    # causal-Q material factor at the evaluation frequencies
    if Q:
        lf_w = np.log(np.abs(np.real(w_arr)) / (2.0 * np.pi))
        fac_w = (1.0 + lf_w / (np.pi * Q) + 1j * q_sign / (2.0 * Q))
        mu_w = mu0 * fac_w * fac_w
    else:
        mu_w = mu0 * np.ones(len(w_arr), dtype=complex)

    Wlm = np.zeros((len(w_arr), lmax + 1, 9), dtype=complex)
    # STATIC COMPLETION (l >= 2; l = 1 static is the degenerate
    # rigid-rotation problem, its plain mode sum converges fine)
    for l in range(2, lmax + 1):
        sf = static_factor(l, a, b, r0)
        for m in (-1, 1):
            Cst = pole[m][l]
            D = Cst[0] * h[0] + Cst[1] * h[1]
            Wlm[:, l, m + 4] += sf * D / mu_w
    # MODE TERMS: dynamic correction only (subtracted static tail),
    # which converges like 1/w_j^4 -> n_keep/lmax_dyn truncation-safe
    for md in cat:
        l = md["l"]
        wj = md["w"]
        if Q:
            lf = np.log(wj / (2.0 * np.pi) / 1.0)
            wj = wj * (1.0 + lf / (np.pi * Q)
                       + 1j * q_sign / (2.0 * Q))
        den = wj * wj - w_arr * w_arr          # (nw,)
        kern = 1.0 / den if l < 2 else (1.0 / den - 1.0 / (wj * wj))
        for m in (-1, 1):
            Cst = pole[m][l]                   # conj(C) at pole (x,y)
            E = md["src"] * (Cst[0] * h[0] + Cst[1] * h[1])
            Wlm[:, l, m + 4] += E * md["Wa"] / md["N"] * kern

    # stations: rotate to source frame, reconstruct, rotate back
    dirs_sf = station_dirs @ Qrot.T
    out = np.zeros((len(station_dirs), 3, len(w_arr)), dtype=complex)
    for i, w in enumerate(w_arr):
        u_sf = toroidal_reconstruct(Wlm[i], dirs_sf)
        out[:, :, i] = u_sf @ Qrot
    return out
