#!/usr/bin/env python3
"""Audit stage 7: ANALYTIC ANCHOR. Free-space field of the code's
m=0 (Mrr/Mzz) jump family vs the analytic homogeneous full-space
moment solution (frequency-domain Stokes/Green tensor), per-lp
coefficients at radius rF, at several source depths b.

If the code family represents a FIXED physical point source, the
per-lp ratio code/analytic is a b-INDEPENDENT constant (one global
constant if conventions match; per-lp constants would already
indicate weight-structure deviations).
"""
import numpy as np
from common import (A, CS, LAUD, LMAX, NT, Yg, dYg, ct, st, lam,
                    m0, mu, proj, r0, rho, solve_l_free, w, w2,
                    y_of)

rF = 1.03 * A
w2c = w * w
vp_c = np.sqrt((lam + 2.0 * mu) / rho)
vs_c = np.sqrt(mu / rho)
ka = w / vp_c
kb = w / vs_c
M0 = 1.0e18


def Gmat(x):
    """Frequency-domain elastodynamic Green tensor,
    G = (1/4 pi rho w^2) [grad grad (g_b - g_a) + kb^2 I g_b],
    g = e^{ikr}/r (outgoing for e^{-iwt}), complex k."""
    r = np.sqrt(x @ x)
    g = x / r
    gg = np.outer(g, g)
    I = np.eye(3)

    def gfun(k):
        return np.exp(1j * k * r) / r

    def t1(k):
        return gfun(k) * (-k * k - 2j * k / r + 2.0 / r ** 2)

    def t2(k):
        return gfun(k) * (1j * k / r - 1.0 / r ** 2)

    dd = (gg * (t1(kb) - t1(ka))
          + (I - gg) * (t2(kb) - t2(ka)))
    return (dd + kb * kb * I * gfun(kb)) / (4.0 * np.pi * rho
                                            * w2c)


def u_ana_line(b, eps=1.0):
    """Analytic Mzz field on the theta line at rF: (ur, ut)."""
    ur = np.zeros(NT, dtype=complex)
    ut = np.zeros(NT, dtype=complex)
    for i in range(NT):
        x = np.array([rF * st[i], 0.0, rF * ct[i]])
        Gp_ = Gmat(x - np.array([0.0, 0.0, b + eps]))
        Gm_ = Gmat(x - np.array([0.0, 0.0, b - eps]))
        un = M0 * (Gp_[:, 2] - Gm_[:, 2]) / (2.0 * eps)
        rh = np.array([st[i], 0.0, ct[i]])
        th = np.array([ct[i], 0.0, -st[i]])
        ur[i] = un @ rh
        ut[i] = un @ th
    return ur, ut


def code_lp(b):
    sol = {l: solve_l_free(l, b) for l in range(1, LMAX + 1)}
    U = np.zeros(LMAX + 1, dtype=complex)
    V = np.zeros(LMAX + 1, dtype=complex)
    for l in range(1, LMAX + 1):
        yv = y_of(l, sol[l], rF) * CS[l]
        U[l], V[l] = yv[0], yv[1]
    return U, V


print("b [km depth]   lp   U_code/U_ana        V_code/V_ana")
for b in (r0 - 50.0e3, r0, r0 + 50.0e3):
    ur, ut = u_ana_line(b)
    Ua, Va = proj(ur, ut)
    Uc, Vc = code_lp(b)
    for lp in range(1, LAUD + 1):
        rU = Uc[lp] / Ua[lp]
        rV = Vc[lp] / Va[lp]
        print("%8.1f     %2d   %9.6f<%7.2f  %9.6f<%7.2f"
              % ((A - b) / 1e3, lp, abs(rU),
                 np.degrees(np.angle(rU)), abs(rV),
                 np.degrees(np.angle(rV))))
    print()
