"""Semi-analytic toroidal (SH) reference for a homogeneous solid
sphere with an interior moment-tensor source at complex frequency
(docs/toroidal_reference.md; Codex-reviewed 2026-07-11).

Route: project the analytic full-space incident field (source.u0eM)
onto the toroidal basis C_lm on the surface (source-frame product
quadrature, |m| <= 2 exactly), then close the traction-free surface
per (l, m) with the ratio-only factor

    W_tot(R) = W_inc(R) * F_l(z),
    F_l(z)   = z (sigma_l - s_l) / (sigma_l (z - (l+2) s_l)),  z = kR,

where s_l = j_l/j_{l-1} (downward/minimal recurrence) and sigma_l =
h^(1)_l/h^(1)_{l-1} (upward/dominant recurrence): no raw Bessel
values, stable to arbitrary l; static limit F_l -> (2l+1)/(l-1).

The same normalized-Legendre code evaluates projection and station
reconstruction, so basis conventions cancel. Known caveat: the
lowest harmonics (|z| << 1) carry amplified near-field roundoff of
the full-space kernels (rung-1 class, suppressed by the band-limited
STF); the l = 1 factor ~ -15/z^2 amplifies this most.
"""

import numpy as np

from .source import u0eM


class _Pts:
    def __init__(self, ic):
        self.ic = np.asarray(ic, dtype=float)
        self.n = len(self.ic)


def bessel_ratios(lmax, z, extra=60):
    """s_l = j_l(z)/j_{l-1}(z) and sigma_l = h1_l(z)/h1_{l-1}(z) for
    l = 1..lmax, vectorized over z (complex array). Downward
    continued-fraction-style recurrence for j (minimal solution),
    upward for h (dominant)."""
    z = np.asarray(z, dtype=complex)
    nl = lmax + int(extra) + int(np.max(np.abs(z))) + 2
    s = np.empty((lmax + 1,) + z.shape, dtype=complex)
    cur = z / (2.0 * nl + 1.0)                # asymptotic seed
    for l in range(nl - 1, 0, -1):
        cur = 1.0 / ((2.0 * l + 1.0) / z - cur)
        if l <= lmax:
            s[l] = cur
    sig = np.empty_like(s)
    sig[1] = 1.0 / z - 1j                     # h1_1/h1_0
    for l in range(1, lmax):
        sig[l + 1] = (2.0 * l + 1.0) / z - 1.0 / sig[l]
    s[0] = np.nan
    sig[0] = np.nan
    return s, sig


def surface_factor(lmax, z):
    """F_l(z) for l = 1..lmax (row l of the returned (lmax+1, nz)
    array; l = 0 unused)."""
    s, sig = bessel_ratios(lmax, z)
    l = np.arange(lmax + 1, dtype=float).reshape((-1,) + (1,) * np.ndim(z))
    return z * (sig - s) / (sig * (z - (l + 2.0) * s))


def source_frame(src_xyz):
    """Orthonormal frame with e3 through the source; returns the
    3x3 rotation matrix Q (rows e1, e2, e3): x_src = Q @ x_earth."""
    e3 = np.asarray(src_xyz, dtype=float)
    e3 = e3 / np.linalg.norm(e3)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(e3 @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return np.vstack([e1, e2, e3])


def theta_grid(lmax, first_panel, n_gl=24, per_wave=12.0):
    """Colatitude panels: doubling growth from first_panel (resolves
    the epicentral field peak) with width capped so an n_gl-point
    panel keeps >= per_wave nodes per minimum harmonic wavelength
    (2 pi / lmax). Returns theta nodes and weights (sin included)."""
    cap = 2.0 * np.pi * n_gl / (per_wave * max(lmax, 1))
    edges = [0.0]
    a = float(first_panel)
    while a < np.pi:
        edges.append(a)
        a += min(a, cap)
    edges.append(np.pi)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    th, wt = [], []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        th.append(t0 + (t1 - t0) * 0.5 * (xg + 1.0))
        wt.append(0.5 * (t1 - t0) * wg)
    th = np.concatenate(th)
    wt = np.concatenate(wt) * np.sin(th)
    return th, wt


def toroidal_project(lmax, th, wth, n_phi, u_thetaphi):
    """Toroidal coefficients W_lm = int conj(C_lm) . u dOmega for
    l <= lmax, m = -2..2, from the (theta, phi) components of the
    field on the product grid: u_thetaphi = (u_theta, u_phi), each
    (n_theta, n_phi) complex. Accumulates inside the l-recurrence of
    the fully-normalized Legendre functions (momentfit conventions).
    Also returns the |m| = 3, 4 leakage monitors (same formula)."""
    nphi = int(n_phi)
    phi_w = 2.0 * np.pi / nphi
    ut, up = u_thetaphi
    # phi transform: Um[m] = sum_j e^{-i m phi_j} u(:, j) * phi_w
    fft_t = np.fft.fft(ut, axis=1) * phi_w      # bin m = e^{-i m phi}
    fft_p = np.fft.fft(up, axis=1) * phi_w
    ct = np.cos(th)
    st = np.sin(th)
    si = np.where(st > 1e-12, st, 1.0)

    mons = 4
    W = np.zeros((lmax + 1, 2 * mons + 1), dtype=complex)
    for am in range(0, mons + 1):
        # normalized Legendre recurrence at fixed |m| = am
        if am == 0:
            P = np.full_like(ct, 1.0 / np.sqrt(4.0 * np.pi))
        else:
            P = np.full_like(ct, 1.0 / np.sqrt(4.0 * np.pi))
            for mm in range(1, am + 1):
                P = -np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * st * P
        Pm1 = np.zeros_like(P)                  # P_{l-1, am}
        for l in range(am, lmax + 1):
            if l == am:
                Pl = P
            elif l == am + 1:
                Pl = np.sqrt(2.0 * am + 3.0) * ct * P
            else:
                a = np.sqrt((4.0 * l * l - 1.0) / (l * l - am * am))
                b = np.sqrt(((2.0 * l + 1.0) * ((l - 1.0) ** 2 - am * am))
                            / ((2.0 * l - 3.0) * (l * l - am * am)))
                Pl = a * ct * Pcur - b * Pm1
            if l > am:
                Pm1 = Pcur
            Pcur = Pl
            if l < max(1, am):
                continue
            # dP/dtheta via the (l-1) identity
            num = l * ct * Pl
            if l > am:
                e = np.sqrt((l * l - am * am) * (2.0 * l + 1.0)
                            / (2.0 * l - 1.0))
                num = num - e * Pm1
            dP = np.where(st > 1e-12, num / si, 0.0)
            Pos = np.where(st > 1e-12, Pl / si, 0.0)
            f = 1.0 / np.sqrt(l * (l + 1.0))
            for m in ((-am, am) if am else (0,)):
                # conj(C_lm).u = f [conj(t1) u_phi - conj(t2) u_theta]
                # t1 = dP_lm e^{im phi}; t2 = i m (P/sin) e^{im phi}
                # (m < 0 via P̄_{l,-|m|} relations folded into sign)
                sgn = 1.0 if m >= 0 else (-1.0) ** am
                dPm = sgn * dP
                Posm = sgn * Pos
                # phi_j = 2 pi (j + 0.5)/N: half-sample phase on the
                # FFT bin so the sum is exactly sum_j e^{-im phi_j} u_j
                ph_off = np.exp(-1j * np.pi * m / nphi)
                U1 = fft_p[:, m % nphi] * ph_off
                U2 = fft_t[:, m % nphi] * ph_off
                W[l, m + mons] = f * np.sum(
                    wth * (dPm * U1 + 1j * m * Posm * U2))
    return W


def toroidal_reconstruct(W, dirs):
    """u(x) = sum_lm W_lm C_lm(x) at unit directions dirs (n, 3) in
    the SAME frame as the projection. W shape (lmax+1, 9) with m =
    -4..4 columns (monitors included but typically ~0)."""
    lmax = W.shape[0] - 1
    mons = (W.shape[1] - 1) // 2
    ct = np.clip(dirs[:, 2], -1.0, 1.0)
    st = np.hypot(dirs[:, 0], dirs[:, 1])
    ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    si = np.where(st > 1e-12, st, 1.0)
    cp, sp = np.cos(ph), np.sin(ph)
    that = np.stack([ct * cp, ct * sp, -st])
    phat = np.stack([-sp, cp, np.zeros_like(sp)])
    out = np.zeros((len(dirs), 3), dtype=complex)
    for am in range(0, mons + 1):
        if am == 0:
            P = np.full_like(ct, 1.0 / np.sqrt(4.0 * np.pi))
        else:
            P = np.full_like(ct, 1.0 / np.sqrt(4.0 * np.pi))
            for mm in range(1, am + 1):
                P = -np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * st * P
        Pm1 = np.zeros_like(P)
        for l in range(am, lmax + 1):
            if l == am:
                Pl = P
            elif l == am + 1:
                Pl = np.sqrt(2.0 * am + 3.0) * ct * P
            else:
                a = np.sqrt((4.0 * l * l - 1.0) / (l * l - am * am))
                b = np.sqrt(((2.0 * l + 1.0) * ((l - 1.0) ** 2 - am * am))
                            / ((2.0 * l - 3.0) * (l * l - am * am)))
                Pl = a * ct * Pcur - b * Pm1
            if l > am:
                Pm1 = Pcur
            Pcur = Pl
            if l < max(1, am):
                continue
            num = l * ct * Pl
            if l > am:
                e = np.sqrt((l * l - am * am) * (2.0 * l + 1.0)
                            / (2.0 * l - 1.0))
                num = num - e * Pm1
            dP = np.where(st > 1e-12, num / si, 0.0)
            Pos = np.where(st > 1e-12, Pl / si, 0.0)
            f = 1.0 / np.sqrt(l * (l + 1.0))
            for m in ((-am, am) if am else (0,)):
                w = W[l, m + mons]
                if w == 0.0:
                    continue
                sgn = 1.0 if m >= 0 else (-1.0) ** am
                eim = np.exp(1j * m * ph)
                t1 = sgn * dP * eim
                t2 = 1j * m * sgn * Pos * eim
                C = f * (t1 * phat - t2 * that)      # (3, n)
                out += w * C.T
    return out


def vs_complex(w, rho, mu, Q, disp_ref_hz):
    """Complex shear speed, identical to source.u0eM's convention."""
    vs0 = np.sqrt(mu / rho)
    if disp_ref_hz:
        lf = np.log(abs(np.real(w)) / (2.0 * np.pi * disp_ref_hz))
        vs0 = vs0 * (1.0 + lf / (np.pi * Q))
    return vs0 / (1.0 + 1j * 0.5 / Q)


def toroidal_surface_field(mat, R, src_xyz, M, w, lmax,
                           station_dirs_earth, n_gl=24, n_phi=16,
                           tier=1):
    """Toroidal total field (earth-frame Cartesian, (nst, 3)) at unit
    station directions for ONE complex frequency w. mat: object with
    rho, mu, lamda, Q, qp_fac, disp_ref_hz (pyastroseis Material).
    tier=2 doubles the angular resolution (convergence check).
    Returns (u_st, diag) with diag = dict(W=W, tail=..., mon=...)."""
    Qrot = source_frame(src_xyz)
    r_s = np.linalg.norm(src_xyz)
    d = max(R - r_s, 1.0)
    th, wth = theta_grid(lmax, (0.5 if tier == 2 else 1.0) * d / R,
                         n_gl=(2 * n_gl if tier == 2 else n_gl))
    nphi = 2 * n_phi if tier == 2 else n_phi
    phi = 2.0 * np.pi * (np.arange(nphi) + 0.5) / nphi

    # surface grid in the SOURCE frame -> earth frame for u0eM
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    dirs_s = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                       np.cos(TH)], axis=-1)
    pts_e = (dirs_s.reshape(-1, 3) @ Qrot) * R
    u = u0eM(_Pts(pts_e), w, mat.rho, mat.mu, mat.lamda,
             src_xyz[0], src_xyz[1], src_xyz[2], mat.Q, M,
             qp_fac=mat.qp_fac, disp_ref_hz=mat.disp_ref_hz)
    n = len(pts_e)
    u_e = np.stack([u[0:n], u[n:2 * n], u[2 * n:3 * n]], axis=-1)
    u_s = (u_e @ Qrot.T).reshape(len(th), nphi, 3)

    # (theta, phi) components on the source-frame grid
    ct, st_ = np.cos(TH), np.sin(TH)
    cp, sp = np.cos(PH), np.sin(PH)
    that = np.stack([ct * cp, ct * sp, -st_], axis=-1)
    phat = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    ut = np.einsum("ijk,ijk->ij", u_s, that)
    up = np.einsum("ijk,ijk->ij", u_s, phat)

    W = toroidal_project(lmax, th, wth, nphi, (ut, up))

    ks = w / vs_complex(w, mat.rho, mat.mu, mat.Q, mat.disp_ref_hz)
    F = surface_factor(lmax, np.array(ks * R))
    Wt = W.copy()
    Wt[1:] = W[1:] * F[1:].reshape(-1, 1)
    Wt[0] = 0.0

    dirs_st_s = station_dirs_earth @ Qrot.T
    u_st_s = toroidal_reconstruct(Wt, dirs_st_s)
    u_st = u_st_s @ Qrot

    mons = (W.shape[1] - 1) // 2
    core = np.abs(W[:, mons - 2:mons + 3]).max()
    mon = np.abs(np.concatenate([W[:, :mons - 2].ravel(),
                                 W[:, mons + 3:].ravel()])).max()
    ltail = np.abs(Wt[max(1, lmax - 50):, :]).max()
    diag = dict(W=W, mon_ratio=float(mon / core) if core else 0.0,
                tail_ratio=float(ltail / (np.abs(Wt).max() or 1.0)))
    return u_st, diag
