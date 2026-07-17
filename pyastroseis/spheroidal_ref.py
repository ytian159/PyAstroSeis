"""Exact spheroidal (P-SV) reference for uniform-layer fluid-core
models by DIRECT analytic forced solve per (l, omega) — "mini-tipsv".

Model: solid inner core | inviscid fluid outer core | solid shell,
uniform layers, interior moment-tensor source in the shell at r0,
complex frequencies w = 2 pi k / tlen + i sigma (DSM damped axis).

Per (l, m) the radial problem is a 12-unknown linear solve built from
analytic uniform-layer solutions:
    IC   : P-, S-type with j_l           (regular)        -> 2
    OC   : fluid potential j_l, y_l                       -> 2
    shell: P-, S-type with j_l AND y_l, split at r0       -> 8
conditions: ICB (u_r, s_rr continuous, s_rt(solid)=0), CMB (same),
free surface (s_rr = s_rt = 0), and the 4-vector source jump at r0.

Displacement convention per (l, m):
    u = U(r) Y_lm rhat + V(r) [dY/dtheta thetahat
                               + (im Y/sin theta) phihat]
(V multiplies the UNNORMALIZED gradient, matching exact_modes'
solid_cols); the 4-vector is y = (U, V, R, S) with R = sigma_rr
coefficient of Y, S = sigma_rtheta coefficient of dY/dtheta.

Source jumps at r0 (earth-frame M rotated to the source frame where
e3 = rhat; only Mr* components survive for our sources; derived from
the excitation functionals and FD-verified):
    r-h couple, M0 (rhat h + h rhat), |m| = 1 with D = the pole
    coupling of h:      [V] = M0 D / (mu(w) r0^2 L)   (slip jump)
                        [S] = -3 M0 D / (r0^3 L)
                        [R] = -M0 D / r0^3            (from f_U)
    Mrr (m = 0):        [U] = M0 DY / (C(w) r0^2)
                        [R] = -2 M0 DY (1 - 2 F/C ... ) -- see code
where L = l(l+1). All jumps are FD-verified in tests before use.

Attenuation: causal constant-Q on mu (and Q_kappa optional) exactly
as the campaign materials: modulus factors mu(w) = mu0 fac^2 with
fac = 1 + ln(f)/(pi Q) + i q_sign/(2 Q).
"""

import numpy as np

from .toroidal_ref import source_frame


def spheroidal_reconstruct(Wu, Wv, dirs):
    """u(x) = sum_lm [Wu_lm Ybar r̂ + Wv_lm grad1(Ybar)] at unit
    directions dirs (n, 3), fully-normalized Ybar (same Legendre
    recurrences/conventions as toroidal_ref.toroidal_reconstruct).
    Wu, Wv: (lmax+1, 9) complex, m = -4..4 columns."""
    lmax = Wu.shape[0] - 1
    mons = (Wu.shape[1] - 1) // 2
    ct = np.clip(dirs[:, 2], -1.0, 1.0)
    st = np.hypot(dirs[:, 0], dirs[:, 1])
    ph = np.arctan2(dirs[:, 1], dirs[:, 0])
    si = np.where(st > 1e-12, st, 1.0)
    cp, sp = np.cos(ph), np.sin(ph)
    rhat = np.stack([st * cp, st * sp, ct])
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
                aa = np.sqrt((4.0 * l * l - 1.0) / (l * l - am * am))
                bb = np.sqrt(((2.0 * l + 1.0)
                              * ((l - 1.0) ** 2 - am * am))
                             / ((2.0 * l - 3.0) * (l * l - am * am)))
                Pl = aa * ct * Pcur - bb * Pm1
            if l > am:
                Pm1 = Pcur
            Pcur = Pl
            if l < am:
                continue
            num = l * ct * Pl
            if l > am:
                e = np.sqrt((l * l - am * am) * (2.0 * l + 1.0)
                            / (2.0 * l - 1.0))
                num = num - e * Pm1
            dP = np.where(st > 1e-12, num / si, 0.0)
            Pos = np.where(st > 1e-12, Pl / si, 0.0)
            for m in ((-am, am) if am else (0,)):
                wu = Wu[l, m + mons]
                wv = Wv[l, m + mons]
                if wu == 0.0 and wv == 0.0:
                    continue
                sgn = 1.0 if m >= 0 else (-1.0) ** am
                eim = np.exp(1j * m * ph)
                Yl = sgn * Pl * eim
                t1 = sgn * dP * eim              # dY/dtheta
                t2 = 1j * m * sgn * Pos * eim    # im Y / sin(theta)
                out += (wu * (Yl * rhat).T
                        + wv * (t1 * that + t2 * phat).T)
    return out


def spheroidal_pole(lmax):
    """Pole couplings for the excitation functionals, evaluated
    through spheroidal_reconstruct itself (conventions cancel):
    returns (DY[l] for m=0: conj(Ybar(pole)); DG[m][l] = conj of the
    horizontal grad1-vector at the pole, m = -1, +1, as (Cx*, Cy*))."""
    eps = 1.0e-7
    dirs = np.array([[np.sin(eps), 0.0, np.cos(eps)]])
    DY = np.zeros(lmax + 1, dtype=complex)
    for l in range(0, lmax + 1):
        Wu = np.zeros((lmax + 1, 9), dtype=complex)
        Wu[l, 0 + 4] = 1.0
        u = spheroidal_reconstruct(Wu, np.zeros_like(Wu), dirs)[0]
        DY[l] = np.conj(u @ dirs[0])              # radial comp = Ybar
    DG = {}
    for m in (-1, 1):
        vecs = np.zeros((lmax + 1, 2), dtype=complex)
        for l in range(1, lmax + 1):
            Wv = np.zeros((lmax + 1, 9), dtype=complex)
            Wv[l, m + 4] = 1.0
            u = spheroidal_reconstruct(np.zeros_like(Wv), Wv, dirs)[0]
            vecs[l] = np.conj(u[:2])
        DG[m] = vecs
    return DY, DG


# ----------------------------------------------------------------
# complex-argument spherical Bessel j_l, y_l and derivatives
# (downward ratio recurrence for j — minimal solution; upward for y)
# ----------------------------------------------------------------
def sph_bessel_jy(lmax, z, extra=40):
    """j[l], y[l], jp[l], yp[l] for l = 0..lmax at complex z (array).
    Stable for |z| up to ~1e3 and lmax up to a few hundred provided
    y_l does not overflow (caller's responsibility)."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    nl = lmax + extra + int(np.max(np.abs(z))) + 2
    # Miller's algorithm: unnormalized downward sweep, then scale by
    # whichever of j_0, j_1 is larger (robust near zeros of either)
    f = np.zeros((lmax + 2,) + z.shape, dtype=complex)
    fm1 = np.zeros_like(z)                     # f_{nl+1}
    fm0 = np.full_like(z, 1e-280)              # f_{nl}
    for l in range(nl, 0, -1):
        fm1, fm0 = fm0, (2.0 * l + 1.0) / z * fm0 - fm1
        big = np.abs(fm0) > 1e250
        if np.any(big):
            fm0 = np.where(big, fm0 * 1e-200, fm0)
            fm1 = np.where(big, fm1 * 1e-200, fm1)
            f[:, big] *= 1e-200
        lm = l - 1                             # fm0 = f_{l-1}
        if lm <= lmax + 1:
            f[lm] = fm0
        if l <= lmax + 1:
            f[l] = fm1
    j0 = np.sin(z) / z
    j1 = np.sin(z) / z ** 2 - np.cos(z) / z
    use1 = np.abs(j1) > np.abs(j0)
    scale = np.where(use1, j1 / np.where(f[1] == 0, 1.0, f[1]),
                     j0 / np.where(f[0] == 0, 1.0, f[0]))
    j = f * scale
    y = np.zeros((lmax + 2,) + z.shape, dtype=complex)
    y[0] = -np.cos(z) / z
    y[1] = (-np.cos(z) / z - np.sin(z)) / z
    for l in range(1, lmax + 1):
        y[l + 1] = (2.0 * l + 1.0) / z * y[l] - y[l - 1]
    jp = np.zeros((lmax + 1,) + z.shape, dtype=complex)
    yp = np.zeros((lmax + 1,) + z.shape, dtype=complex)
    jp[0] = -j[1]
    yp[0] = -y[1]
    for l in range(1, lmax + 1):
        jp[l] = j[l - 1] - (l + 1.0) / z * j[l]
        yp[l] = y[l - 1] - (l + 1.0) / z * y[l]
    return j[:lmax + 1], y[:lmax + 1], jp, yp


def _fl(kind, l, z, tab=None):
    """(f_l, f_l') for complex scalar/array z, kind 'j'|'y'."""
    j, y, jp, yp = sph_bessel_jy(l, z)
    if kind == "j":
        return j[l], jp[l]
    return y[l], yp[l]


def _series_S(l, z, nmax=80):
    """S_l(z) = sum_m (-z^2/2)^m / (m! (2l+3)(2l+5)..(2l+2m+1)),
    the small-z polynomial factor of j_l = z^l/(2l+1)!! S_l."""
    z2 = z * z
    term = np.ones_like(z, dtype=complex)
    out = term.copy()
    for m in range(1, nmax):
        term = term * (-z2 / 2.0) / (m * (2.0 * l + 2.0 * m + 1.0))
        out = out + term
        if np.all(np.abs(term) < 1e-18 * np.abs(out)):
            break
    return out


def _series_T(l, z, nmax=80):
    """T_l(z): y_l = -(2l-1)!!/z^{l+1} T_l,
    T_l = sum_m (-z^2/2)^m / (m! (1-2l)(3-2l)..(2m-1-2l))."""
    z2 = z * z
    term = np.ones_like(z, dtype=complex)
    out = term.copy()
    for m in range(1, nmax):
        term = term * (-z2 / 2.0) / (m * (2.0 * m - 1.0 - 2.0 * l))
        out = out + term
        if np.all(np.abs(term) < 1e-18 * np.abs(out)):
            break
    return out


def _fl_scaled(kind, l, z, zref):
    """Column-scaled (f_l, f_l') via the small-z series:
    j-family scaled by (2l+1)!!/zref^l, y-family by zref^{l+1}/
    (-(2l-1)!!). Valid for z^2/2 < ~l (used for l >= ~46 in-band).
    Ratios (z/zref)^l stay O(1) across the shell."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    if kind == "j":
        pw = (z / zref) ** l
        S = _series_S(l, z)
        Sm = _series_S(l - 1, z)
        f = pw * S
        fp = (2.0 * l + 1.0) / z * pw * Sm - (l + 1.0) / z * pw * S
        return f, fp
    pw = (zref / z) ** (l + 1)
    T = _series_T(l, z)
    Tm = _series_T(l - 1, z)
    f = pw * T
    # yhat_{l-1} = (zref/z)^l * zref/(2l-1) * T_{l-1}
    fm1 = (zref / z) ** l * zref / (2.0 * l - 1.0) * Tm
    fp = fm1 - (l + 1.0) / z * f
    return f, fp


# ----------------------------------------------------------------
# uniform-layer solution 4-vectors (U, V, R, S); complex moduli
# ----------------------------------------------------------------
def solid_cols_c(kind, l, w, r, rho, lam, mu, zref_a=None):
    """(U, V, R, S) for P- and S-type solutions of one Bessel kind
    at complex w and complex moduli. Mirrors exact_modes.solid_cols
    (validated against Lamb/tipsv) with complex support.
    zref_a: surface radius for the scaled-series basis (used for
    high l where raw Bessel under/overflows; the per-column scale is
    consistent across radii, and the linear solves are column-scale
    invariant)."""
    L = l * (l + 1.0)
    vp = np.sqrt((lam + 2.0 * mu) / rho)
    vs = np.sqrt(mu / rho)
    kp, ks = w / vp, w / vs
    zp, zs = kp * r, ks * r
    if zref_a is None:
        f, fp = _fl(kind, l, zp)
        g, gp = _fl(kind, l, zs)
    else:
        f, fp = _fl_scaled(kind, l, zp, kp * zref_a)
        g, gp = _fl_scaled(kind, l, zs, ks * zref_a)
    P = np.array([
        kp * fp,
        f / r,
        kp ** 2 * (-(lam + 2.0 * mu) * f + 2.0 * mu * L * f / zp ** 2
                   - 4.0 * mu * fp / zp),
        2.0 * mu * kp ** 2 * (fp / zp - f / zp ** 2)], dtype=complex)
    S = np.array([
        L * g / r,
        ks * (g / zs + gp),
        2.0 * mu * L * ks ** 2 * (gp / zs - g / zs ** 2),
        mu * ks ** 2 * (2.0 * (L - 1.0) * g / zs ** 2 - 2.0 * gp / zs
                        - g)], dtype=complex)
    return P, S


def fluid_cols_c(kind, l, w, r, rho, vp):
    """(u_r, s_rr) for the fluid potential solution (displacement
    u = grad(phi), p = rho w^2 phi, s_rr = -p)."""
    kf = w / vp
    f, fp = _fl(kind, l, kf * r)
    return np.array([kf * fp, -rho * w * w * f], dtype=complex)


def q_factor(w, Q, q_sign=-1.0):
    """Causal constant-Q velocity factor at complex w (1-Hz ref)."""
    lf = np.log(np.abs(np.real(w)) / (2.0 * np.pi))
    return 1.0 + lf / (np.pi * Q) + 1j * q_sign / (2.0 * Q)


def spheroidal_spectra(model0, src_xyz, M_list, w_arr, station_dirs,
                       Q=None, q_sign=-1.0, lmax=250):
    """P-SV spectra (nsrc, nst, 3, nw) by direct forced solve.
    model0: dict(a, b, c, sh=(rho,vp,vs), oc=(rho,vp), ic=(rho,vp,vs))
    with REAL reference velocities (1-Hz values); Q applies causal
    dispersion to the shear moduli of the solids (Qkappa = inf).
    Sources: earth-frame moment tensors; in the source frame only
    M_zz (m=0) and M_zh (m=+-1) couple (true for Mrr/Mrt-type)."""
    a, b, c = model0["a"], model0["b"], model0["c"]
    rho_s, vp_s, vs_s = model0["sh"]
    rho_i, vp_i, vs_i = model0["ic"]
    rho_f, vp_f = model0["oc"]
    r0 = float(np.linalg.norm(src_xyz))
    Qrot = source_frame(src_xyz)
    DY, DG = spheroidal_pole(lmax)
    srcs = []
    for M in M_list:
        M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
        srcs.append((M_sf[2, 2], M_sf[2, 0] + 0j, M_sf[2, 1] + 0j))
    w_arr = np.asarray(w_arr, dtype=complex)
    nw, nst = len(w_arr), len(station_dirs)
    dirs_sf = station_dirs @ Qrot.T
    out = np.zeros((len(M_list), nst, 3, nw), dtype=complex)

    mu_s0 = rho_s * vs_s ** 2
    ka_s0 = rho_s * vp_s ** 2 - 4.0 / 3.0 * mu_s0
    mu_i0 = rho_i * vs_i ** 2
    ka_i0 = rho_i * vp_i ** 2 - 4.0 / 3.0 * mu_i0

    Wu = np.zeros((len(M_list), nw, lmax + 1, 9), dtype=complex)
    Wv = np.zeros_like(Wu)
    for iw, w in enumerate(w_arr):
        fac2 = q_factor(w, Q, q_sign) ** 2 if Q else 1.0
        mu_s = mu_s0 * fac2
        lam_s = ka_s0 - 2.0 / 3.0 * mu_s
        mu_i = mu_i0 * fac2
        lam_i = ka_i0 - 2.0 / 3.0 * mu_i
        model = dict(a=a, b=b, c=c, sh=(rho_s, lam_s, mu_s),
                     oc=(rho_f, vp_f), ic=(rho_i, lam_i, mu_i))
        for l in range(1, lmax + 1):
            zr = a if l >= L_SERIES else None
            L = l * (l + 1.0)
            # S-row loads carry 1/L: the horizontal force COEFFICIENT
            # in the unnormalized-gradient basis is the projection/L
            F0v = np.array([0, 0, 0, 1.0 / (L * r0 ** 2)],
                           dtype=complex)
            F1v = np.array([0, 0, -1.0 / r0 ** 3,
                            3.0 / (L * r0 ** 3)], dtype=complex)
            J_rh = source_jumps(l, w, r0, rho_s, lam_s, mu_s, F0v,
                                F1v, zref_a=zr)
            U_rh, V_rh = forced_surface(l, w, model, r0, J_rh)
            F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
            F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
            J_y = source_jumps(l, w, r0, rho_s, lam_s, mu_s, F0y,
                               F1y, zref_a=zr)
            U_y, V_y = forced_surface(l, w, model, r0, J_y)
            for isrc, (mzz, mzx, mzy) in enumerate(srcs):
                if mzz != 0.0:
                    qy = mzz * DY[l]
                    Wu[isrc, iw, l, 0 + 4] += qy * U_y
                    Wv[isrc, iw, l, 0 + 4] += qy * V_y
                for m in (-1, 1):
                    D = DG[m][l][0] * mzx + DG[m][l][1] * mzy
                    if D != 0.0:
                        Wu[isrc, iw, l, m + 4] += D * U_rh
                        Wv[isrc, iw, l, m + 4] += D * V_rh
    for isrc in range(len(M_list)):
        for iw in range(nw):
            u_sf = spheroidal_reconstruct(Wu[isrc, iw], Wv[isrc, iw],
                                          dirs_sf)
            out[isrc, :, :, iw] = u_sf @ Qrot
    return out


L_SERIES = 20      # l >= L_SERIES: shell-only + scaled-series basis.
                   # Fluid-core influence on the surface response
                   # scales as (b/a)^(2l) < 3e-11 by l = 20, and the
                   # full 12x12 fluid columns become ill-conditioned
                   # at (high l, low k); the series basis converges
                   # in-band for l >= 20 (z^2/2 <= 24.8 < 2l+3).


def _shell_matrix(l, w, r, rho, lam, mu, zref_a=None):
    """4x4 solution matrix Y(r) with columns (Pj, Sj, Py, Sy)."""
    Pj, Sj = solid_cols_c("j", l, w, r, rho, lam, mu, zref_a)
    Py, Sy = solid_cols_c("y", l, w, r, rho, lam, mu, zref_a)
    return np.stack([Pj[:, 0], Sj[:, 0], Py[:, 0], Sy[:, 0]], axis=1)


def system_matrix(l, w, r0, rho, lam, mu, h=1.0e-3, zref_a=None):
    """4x4 first-order system matrix A(r0) (y' = A y), built
    numerically from the analytic solution matrix: A = Y' Y^{-1}
    (invariant under the per-column scaling of the basis)."""
    Y0 = _shell_matrix(l, w, r0, rho, lam, mu, zref_a)
    Yp = _shell_matrix(l, w, r0 + h, rho, lam, mu, zref_a)
    Ym = _shell_matrix(l, w, r0 - h, rho, lam, mu, zref_a)
    return ((Yp - Ym) / (2.0 * h)) @ np.linalg.inv(Y0)


def source_jumps(l, w, r0, rho, lam, mu, F0, F1, zref_a=None):
    """Regular-part jump 4-vector for forcing entering the system as
    y' = A y + F0 delta'(r-r0) + F1 delta(r-r0):  [y] = F1 + A F0.
    (The field also carries a -F0 delta singularity on the loaded
    components; irrelevant for r != r0.)

    Per-source loads (code-V convention u_h = V grad1 Y; force
    density f with weak pairing int (U_t f_U + V_t f_V) r^2 dr):
      r-h couple M0 (rhat h + h rhat), amplitude q = M0 D (D = pole
      coupling of h): E = q [U/r0 + V' - V/r0]
        F0 = (0, 0, 0,  q/r0^2)
        F1 = (0, 0, -q/r0^3, 3 q/r0^3)
      Mrr, amplitude qy = M0 D_Y: E = qy U'(r0)
        F0 = (0, 0, qy/r0^2, 0)
        F1 = (0, 0, 2 qy/r0^3, 0)
    """
    # Scaled regime: reference the basis at r0 itself — J is basis-
    # independent in exact arithmetic, and the LOCAL reference
    # minimizes cond(Y(r0)). With a surface-referenced basis the
    # Y'Y^-1 construction wobbles ~1e-3 at l >= L_SERIES and, being
    # zref-sensitive, made J spuriously depend on the outer radius
    # (caught by the rung-A3b Y00 free-surface FD gate, 2026-07-17).
    A = system_matrix(l, w, r0, rho, lam, mu,
                      zref_a=(r0 if zref_a is not None else None))
    return np.asarray(F1, dtype=complex) + A @ np.asarray(F0,
                                                          dtype=complex)


def forced_surface(l, w, model, r0, jump, l_switch=None):
    """Solve the forced 3-layer problem for one (l, w) given the
    source jump 4-vector at r0; returns (U(a), V(a)).
    model: dict with keys ic=(rho,lam,mu), oc=(rho,vp), sh=(rho,lam,
    mu), radii a,b,c (surface, CMB, ICB). Complex moduli allowed.
    High l (fluid core evanescently decoupled; y_l at the CMB would
    overflow): shell-only 6x6 with downward-decaying (j) solutions
    below r0."""
    a, b, c = model["a"], model["b"], model["c"]
    rho_s, lam_s, mu_s = model["sh"]
    if l_switch is None:
        shell_only = l >= L_SERIES
    else:
        shell_only = l > l_switch
    # scaled-series basis whenever raw Bessel would under/overflow,
    # regardless of how shell_only was selected
    zref = a if (shell_only and l >= L_SERIES) else None

    Ya = _shell_matrix(l, w, a, rho_s, lam_s, mu_s, zref)
    Y0 = _shell_matrix(l, w, r0, rho_s, lam_s, mu_s, zref)
    if shell_only:
        M = np.zeros((6, 6), dtype=complex)
        rhs = np.zeros(6, dtype=complex)
        # unknowns: below (Pj, Sj), above (Pj, Sj, Py, Sy)
        M[0:4, 0:2] = -Y0[:, 0:2]
        M[0:4, 2:6] = Y0
        rhs[0:4] = jump
        M[4, 2:6] = Ya[2, :]
        M[5, 2:6] = Ya[3, :]
        sc = np.max(np.abs(M), axis=0)
        sc[sc == 0] = 1.0
        x = np.linalg.solve(M / sc, rhs) / sc
        cu = x[2:6]
    else:
        rho_i, lam_i, mu_i = model["ic"]
        rho_f, vp_f = model["oc"]
        Pi, Si = solid_cols_c("j", l, w, c, rho_i, lam_i, mu_i)
        fjc = fluid_cols_c("j", l, w, c, rho_f, vp_f)
        fyc = fluid_cols_c("y", l, w, c, rho_f, vp_f)
        fjb = fluid_cols_c("j", l, w, b, rho_f, vp_f)
        fyb = fluid_cols_c("y", l, w, b, rho_f, vp_f)
        Yb = _shell_matrix(l, w, b, rho_s, lam_s, mu_s)
        M = np.zeros((12, 12), dtype=complex)
        rhs = np.zeros(12, dtype=complex)
        # cols: 0-1 IC (Pj,Sj); 2-3 OC (j,y); 4-7 shell-below;
        #       8-11 shell-above
        M[0, 0:2] = [Pi[0, 0], Si[0, 0]]   # ICB u_r
        M[0, 2:4] = [-fjc[0, 0], -fyc[0, 0]]
        M[1, 0:2] = [Pi[2, 0], Si[2, 0]]   # ICB s_rr
        M[1, 2:4] = [-fjc[1, 0], -fyc[1, 0]]
        M[2, 0:2] = [Pi[3, 0], Si[3, 0]]   # ICB s_rt (IC side)
        M[3, 2:4] = [fjb[0, 0], fyb[0, 0]]  # CMB u_r
        M[3, 4:8] = -Yb[0, :]
        M[4, 2:4] = [fjb[1, 0], fyb[1, 0]]  # CMB s_rr
        M[4, 4:8] = -Yb[2, :]
        M[5, 4:8] = Yb[3, :]                # CMB s_rt (shell side)
        M[6:10, 4:8] = -Y0                  # jumps at r0
        M[6:10, 8:12] = Y0
        rhs[6:10] = jump
        M[10, 8:12] = Ya[2, :]              # surface s_rr
        M[11, 8:12] = Ya[3, :]              # surface s_rt
        sc = np.max(np.abs(M), axis=0)
        sc[sc == 0] = 1.0
        x = np.linalg.solve(M / sc, rhs) / sc
        cu = x[8:12]
    Ua = Ya[0, :] @ cu
    Va = Ya[1, :] @ cu
    return Ua, Va
