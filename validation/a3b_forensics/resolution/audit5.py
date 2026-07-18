#!/usr/bin/env python3
"""Audit stage 5: toroidal machinery control + h-kind swap.

 CTRL-2b toroidal machinery control: legs = pointwise-translated
         FIXED toroidal free-space field; must give W lhs/rhs = 1.
 H2      spheroidal covariance with incoming kind h2 = j - i y
         (same criterion; sanity that the kind choice is not the
         issue).
"""
import numpy as np
from common import (A, CS, LAUD, LMAX, NT, ct, st, delta, m0, mu,
                    proj, r0, solve_l_free, tfe, w, w2, y_of,
                    zgrad_proj, _Ysolid)
from pyastroseis.spectral import _Ytor
from pyastroseis.toroidal_modes import _pole_coupling

zhat = np.array([0.0, 0.0, 1.0])
rF = 1.03 * A
poleT = _pole_coupling(LMAX)
ctl = {l: poleT[1][l][0] for l in range(1, LMAX + 1)}


def _Yth(l, r, sgn=+1.0):
    return (_Ytor(l, w, r, m0, ("j",), None)
            + sgn * 1j * _Ytor(l, w, r, m0, ("y",), None))


def solve_tor_free(l, b, sgn=+1.0):
    Mx = np.zeros((2, 2), dtype=complex)
    Mx[:, 0] = _Yth(l, b, sgn)[:, 0]
    Mx[:, 1] = -_Ytor(l, w, b, m0, ("j",), None)[:, 0]
    J = np.array([1.0 / (mu * b * b), 0.0], dtype=complex)
    sc = np.max(np.abs(Mx), axis=0)
    x = np.linalg.solve(Mx / sc, J) / sc
    return dict(cout=x[0], cin=x[1], b=b, sgn=sgn)


def W_of(l, sol, r):
    if r >= sol["b"]:
        return _Yth(l, r, sol["sgn"])[:, 0] * sol["cout"]
    return _Ytor(l, w, r, m0, ("j",), None)[:, 0] * sol["cin"]


T0 = {l: solve_tor_free(l, r0) for l in range(1, LMAX + 1)}

S1 = tfe._shapes(1, LMAX + 2, ct, st)
Ct1, Cp1 = S1["Ct"], S1["Cp"]
ll = np.arange(LMAX + 3)
f1 = np.zeros(LMAX + 3)
f1[1:] = 1.0 / np.sqrt(ll[1:] * (ll[1:] + 1.0))
Y1, dY1 = S1["Y"], S1["Bt"]
d2Y1 = S1["eB"][0] / 2.0
cot = ct / st
dCt1 = -1j * f1[:, None] * (dY1 / st - Y1 * cot / st)
dCp1 = f1[:, None] * d2Y1


def projW(gt, gp):
    return (np.einsum('ln,n->l', np.conj(Ct1), gt * w2)
            + np.einsum('ln,n->l', np.conj(Cp1), gp * w2))


def tor_line(r, sols, hfd=0.5):
    ut = np.zeros(NT, dtype=complex)
    up = np.zeros(NT, dtype=complex)
    utp_ = np.zeros(NT, dtype=complex)
    upp_ = np.zeros(NT, dtype=complex)
    dtut = np.zeros(NT, dtype=complex)
    dtup = np.zeros(NT, dtype=complex)
    for l in range(1, LMAX + 1):
        Wv = W_of(l, sols[l], r)[0] * ctl[l]
        Wp = (W_of(l, sols[l], r + hfd)[0]
              - W_of(l, sols[l], r - hfd)[0]) / (2 * hfd) * ctl[l]
        ut += Wv * Ct1[l]
        up += Wv * Cp1[l]
        utp_ += Wp * Ct1[l]
        upp_ += Wp * Cp1[l]
        dtut += Wv * dCt1[l]
        dtup += Wv * dCp1[l]
    return ut, up, utp_, upp_, dtut, dtup


ut, up, utp_, upp_, dtut, dtup = tor_line(rF, T0)
gt = ct * utp_ - (st / rF) * dtut
gp = ct * upp_ - (st / rF) * dtup
Wrhs = delta * projW(gt, gp)


def tor_field_shifted(dz):
    """(gt, gp) components on the grid basis of the FIXED b=r0
    toroidal field evaluated at x = rF rhat + dz zhat (phi=0
    plane, m=1 complex convention)."""
    gt_ = np.zeros(NT, dtype=complex)
    gp_ = np.zeros(NT, dtype=complex)
    for i in range(NT):
        x = np.array([rF * st[i], 0.0, rF * ct[i]]) + dz * zhat
        r = np.linalg.norm(x)
        cth = x[2] / r
        sth = np.hypot(x[0], x[1]) / r
        Sh = tfe._shapes(1, LMAX, np.array([cth]), np.array([sth]))
        utL = 0j
        upL = 0j
        for l in range(1, LMAX + 1):
            Wv = W_of(l, T0[l], r)[0] * ctl[l]
            utL += Wv * Sh["Ct"][l][0]
            upL += Wv * Sh["Cp"][l][0]
        th_l = np.array([cth, 0.0, -sth])
        th_g = np.array([ct[i], 0.0, -st[i]])
        gt_[i] = utL * (th_l @ th_g)
        gp_[i] = upL
    return gt_, gp_


print("== CTRL-2b toroidal machinery control (translated fixed "
      "field) ==")
gtp, gpp = tor_field_shifted(+delta)
gtm, gpm = tor_field_shifted(-delta)
Wc = projW(0.5 * (gtp - gtm), 0.5 * (gpp - gpm))
for lp in range(1, LAUD + 1):
    rW = Wc[lp] / Wrhs[lp]
    print("%2d  W %8.5f<%6.1f" % (lp, abs(rW),
                                  np.degrees(np.angle(rW))))

# ---- H2: spheroidal covariance with incoming kind ---------------
import common


def _Yh2(l, r):
    return (_Ysolid(l, w, r, m0, ("j",), None)
            - 1j * _Ysolid(l, w, r, m0, ("y",), None))


def solve_free2(l, b):
    from common import Jvec
    J = Jvec(l, b)
    M = np.zeros((4, 4), dtype=complex)
    M[:, 0:2] = _Yh2(l, b)
    M[:, 2:4] = -_Ysolid(l, w, b, m0, ("j",), None)
    sc = np.max(np.abs(M), axis=0)
    x = np.linalg.solve(M / sc, J) / sc
    return dict(cout=x[0:2], cin=x[2:4], b=b, kind="free2")


def y_of2(l, sol, r):
    if r >= sol["b"]:
        return _Yh2(l, r) @ sol["cout"]
    return _Ysolid(l, w, r, m0, ("j",), None) @ sol["cin"]


G0 = {l: solve_free2(l, r0) for l in range(1, LMAX + 1)}
Gm = {l: solve_free2(l, r0 - delta) for l in range(1, LMAX + 1)}
Gp = {l: solve_free2(l, r0 + delta) for l in range(1, LMAX + 1)}
# RHS via zgrad on the h2 family: reuse common.u_line via monkey
# route: build a sols dict compatible with common.y_of -> simplest:
# local recompute of the line fields
from common import Yg, dYg, d2Yg


def u_line2(r, sols, hfd=0.5):
    ur = np.zeros(NT, dtype=complex)
    ut_ = np.zeros(NT, dtype=complex)
    urp = np.zeros(NT, dtype=complex)
    utp2 = np.zeros(NT, dtype=complex)
    dtur = np.zeros(NT, dtype=complex)
    dtut2 = np.zeros(NT, dtype=complex)
    for l in range(1, LMAX + 1):
        yv = y_of2(l, sols[l], r) * CS[l]
        yp = (y_of2(l, sols[l], r + hfd)
              - y_of2(l, sols[l], r - hfd)) / (2 * hfd) * CS[l]
        ur += yv[0] * Yg[l]
        ut_ += yv[1] * dYg[l]
        urp += yp[0] * Yg[l]
        utp2 += yp[1] * dYg[l]
        dtur += yv[0] * dYg[l]
        dtut2 += yv[1] * d2Yg[l]
    return ur, ut_, urp, utp2, dtur, dtut2


ur, ut_, urp, utp2, dtur, dtut2 = u_line2(rF, G0)
gr = ct * urp - (st / rF) * (dtur - ut_)
gt2 = ct * utp2 - (st / rF) * (ur + dtut2)
U2r, V2r = proj(gr, gt2)
U2r, V2r = delta * U2r, delta * V2r
print("\n== H2 spheroidal covariance with h2 = j - i y ==")
for lp in range(1, LAUD + 1):
    ym = y_of2(lp, Gm[lp], rF) * CS[lp]
    yp_ = y_of2(lp, Gp[lp], rF) * CS[lp]
    Ul = 0.5 * (ym[0] - yp_[0])
    Vl = 0.5 * (ym[1] - yp_[1])
    rU = Ul / U2r[lp]
    rV = Vl / V2r[lp]
    print("%2d  U %8.5f<%6.1f   V %8.5f<%6.1f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV))))
