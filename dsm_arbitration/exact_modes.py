#!/usr/bin/env python3
"""Exact eigenfrequencies of the uniform3/corefluid model (elastic,
gravity-free) via analytic uniform-layer solutions.

Toroidal (shell annulus, fluid decouples):
    W = A j_l(kr) + B y_l(kr),  T(r) = mu (W' - W/r),
    det [T_j(R) T_y(Rc); T_j(Rc) T_y(Rc)] = 0.

Spheroidal (solid IC | fluid OC | solid shell), 8x8 determinant:
    P-type (potential f_l(k_p r)):  u_r = k_p f',  V = f/r,
        s_rr = k_p^2 [-(lam+2mu) f + 2 mu L f/z^2 - 4 mu f'/z]
        s_rt = 2 mu k_p^2 (f'/z - f/z^2)
    S-type: u_r = L f/r,  V = k_s(f/z + f'),
        s_rr = 2 mu L k_s^2 (f'/z - f/z^2)
        s_rt = mu k_s^2 (2(L-1) f/z^2 - 2 f'/z - f)
    fluid (potential): u_r = k_f f',  s_rr = -rho w^2 f, s_rt = 0.
    ICB: u_r, s_rr continuous, s_rt(solid)=0 ; CMB likewise;
    surface: s_rr = s_rt = 0.

Model: IC (r<=1221.5 km) rho 6, vp 7, vs 3.5; OC (<=3480) rho 5,
vp 5.5; shell (<=6371) rho 3, vp 6, vs 3.  SI units internally.
"""
import numpy as np
from scipy.special import spherical_jn, spherical_yn

R_ICB, R_CMB, R_SRF = 1221.5e3, 3480.0e3, 6371.0e3
IC = dict(rho=6000.0, vp=7000.0, vs=3500.0)
OC = dict(rho=5000.0, vp=5500.0)
SH = dict(rho=3000.0, vp=6000.0, vs=3000.0)


def _f(kind, l, z):
    if kind == "j":
        return spherical_jn(l, z), spherical_jn(l, z, derivative=True)
    return spherical_yn(l, z), spherical_yn(l, z, derivative=True)


def tor_traction(kind, l, w, r, vs, mu):
    k = w / vs
    f, fp = _f(kind, l, k * r)
    return mu * (k * fp - f / r)


def toroidal_det(l, w, annulus=True):
    """Sign-carrying normalized determinant for toroidal modes."""
    mu = SH["rho"] * SH["vs"] ** 2
    tjR = tor_traction("j", l, w, R_SRF, SH["vs"], mu)
    if not annulus:                       # full uniform ball (j only)
        return tjR
    tyR = tor_traction("y", l, w, R_SRF, SH["vs"], mu)
    tjC = tor_traction("j", l, w, R_CMB, SH["vs"], mu)
    tyC = tor_traction("y", l, w, R_CMB, SH["vs"], mu)
    d = tjR * tyC - tyR * tjC
    n = abs(tjR * tyC) + abs(tyR * tjC) + 1e-300
    return d / n


def solid_cols(kind, l, w, r, mat):
    """(u_r, V, s_rr, s_rt) for P and S solutions of one kind."""
    rho, vp, vs = mat["rho"], mat["vp"], mat["vs"]
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    L = l * (l + 1)
    kp, ks = w / vp, w / vs
    zp, zs = kp * r, ks * r
    f, fp = _f(kind, l, zp)
    P = np.array([
        kp * fp,
        f / r,
        kp ** 2 * (-(lam + 2 * mu) * f + 2 * mu * L * f / zp ** 2
                   - 4 * mu * fp / zp),
        2 * mu * kp ** 2 * (fp / zp - f / zp ** 2)])
    g, gp = _f(kind, l, zs)
    S = np.array([
        L * g / r,
        ks * (g / zs + gp),
        2 * mu * L * ks ** 2 * (gp / zs - g / zs ** 2),
        mu * ks ** 2 * (2 * (L - 1) * g / zs ** 2 - 2 * gp / zs - g)])
    return P, S


def fluid_cols(kind, l, w, r):
    rho, vp = OC["rho"], OC["vp"]
    kf = w / vp
    f, fp = _f(kind, l, kf * r)
    return np.array([kf * fp, -rho * w * w * f])   # (u_r, s_rr)


def spheroidal_det(l, w):
    """8x8 boundary determinant, column-normalized, sign-carrying.
    Unknowns: [IC_Pj, IC_Sj, OC_j, OC_y, SH_Pj, SH_Sj, SH_Py, SH_Sy]"""
    M = np.zeros((8, 8))
    icP, icS = solid_cols("j", l, w, R_ICB, IC)
    fjI = fluid_cols("j", l, w, R_ICB)
    fyI = fluid_cols("y", l, w, R_ICB)
    # rows 0-2: ICB  u_r, s_rr continuity ; s_rt(IC)=0
    M[0, 0], M[0, 1], M[0, 2], M[0, 3] = icP[0], icS[0], -fjI[0], -fyI[0]
    M[1, 0], M[1, 1], M[1, 2], M[1, 3] = icP[2], icS[2], -fjI[1], -fyI[1]
    M[2, 0], M[2, 1] = icP[3], icS[3]
    fjC = fluid_cols("j", l, w, R_CMB)
    fyC = fluid_cols("y", l, w, R_CMB)
    shPj, shSj = solid_cols("j", l, w, R_CMB, SH)
    shPy, shSy = solid_cols("y", l, w, R_CMB, SH)
    # rows 3-5: CMB
    M[3, 2], M[3, 3] = fjC[0], fyC[0]
    M[3, 4], M[3, 5], M[3, 6], M[3, 7] = -shPj[0], -shSj[0], -shPy[0], -shSy[0]
    M[4, 2], M[4, 3] = fjC[1], fyC[1]
    M[4, 4], M[4, 5], M[4, 6], M[4, 7] = -shPj[2], -shSj[2], -shPy[2], -shSy[2]
    M[5, 4], M[5, 5], M[5, 6], M[5, 7] = shPj[3], shSj[3], shPy[3], shSy[3]
    # rows 6-7: free surface
    sPj, sSj = solid_cols("j", l, w, R_SRF, SH)
    sPy, sSy = solid_cols("y", l, w, R_SRF, SH)
    M[6, 4], M[6, 5], M[6, 6], M[6, 7] = sPj[2], sSj[2], sPy[2], sSy[2]
    M[7, 4], M[7, 5], M[7, 6], M[7, 7] = sPj[3], sSj[3], sPy[3], sSy[3]
    # column normalization (positive scalars keep the sign meaningful)
    for c in range(8):
        n = np.max(np.abs(M[:, c]))
        if n > 0:
            M[:, c] /= n
    sign, logd = np.linalg.slogdet(M)
    return sign * np.exp(logd)


def find_zeros(fun, l, fgrid):
    ws = 2 * np.pi * fgrid
    vals = np.array([fun(l, w) for w in ws])
    roots = []
    for i in range(len(ws) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) \
                and vals[i] * vals[i + 1] < 0:
            a, b = ws[i], ws[i + 1]
            for _ in range(60):
                m = 0.5 * (a + b)
                if fun(l, a) * fun(l, m) <= 0:
                    b = m
                else:
                    a = m
            roots.append(0.5 * (a + b) / (2 * np.pi))
    return roots


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    fgrid = np.linspace(2e-5, 5.6e-4, 2200)
    if which in ("tor", "both"):
        print("# toroidal annulus eigenfrequencies (uHz)")
        for l in range(1, 25):
            r = find_zeros(toroidal_det, l, fgrid)
            print("l=%2d: %s" % (l, " ".join("%.2f" % (f * 1e6)
                                             for f in r)))
    if which in ("sph", "both"):
        print("# spheroidal 3-layer eigenfrequencies (uHz)")
        for l in range(2, 25):
            r = find_zeros(spheroidal_det, l, fgrid)
            print("l=%2d: %s" % (l, " ".join("%.2f" % (f * 1e6)
                                             for f in r)))
