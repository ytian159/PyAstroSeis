#!/usr/bin/env python3
"""Audit stage 4: controls for the free-space covariance test.

 CTRL-1  machinery control: legs DEFINED by pointwise translation
         of the fixed b=r0 free-space field must give lhs/rhs = 1.
 CTRL-2  toroidal (SH) free-space covariance with the code's
         [W] = q/(mu b^2) loads (the passing SH gate predicts 1).
 MAIN    spheroidal covariance ratios (recap).
"""
import numpy as np
from common import (A, CS, Jvec, LAUD, LMAX, NT, Yg, dYg, ct, st,
                    delta, m0, mu, proj, r0, solve_l_free, tfe, w,
                    w2, y_of, zgrad_proj)
from pyastroseis.spectral import _Ytor
from pyastroseis.toroidal_modes import _pole_coupling

zhat = np.array([0.0, 0.0, 1.0])
rF = 1.03 * A

F0 = {l: solve_l_free(l, r0) for l in range(1, LMAX + 1)}
Fm = {l: solve_l_free(l, r0 - delta) for l in range(1, LMAX + 1)}
Fp = {l: solve_l_free(l, r0 + delta) for l in range(1, LMAX + 1)}

# RHS (shared): delta * d/dz of the b=r0 field at rF
Ug, Vg = zgrad_proj(rF, F0)
Urhs, Vrhs = delta * Ug, delta * Vg


def field_at_shifted(dz):
    """(ur, ut) of the b=r0 free-space field evaluated pointwise at
    x = rF*rhat(theta) + dz*zhat, expressed in the LOCAL (r', th')
    frame, then rotated back to the rF-grid (r, th) frame."""
    ur = np.zeros(NT, dtype=complex)
    ut = np.zeros(NT, dtype=complex)
    for i in range(NT):
        x = np.array([rF * st[i], 0.0, rF * ct[i]]) + dz * zhat
        r = np.linalg.norm(x)
        cth = x[2] / r
        sth = np.hypot(x[0], x[1]) / r
        Sh = tfe._shapes(0, LMAX, np.array([cth]), np.array([sth]))
        urL = 0j
        utL = 0j
        for l in range(1, LMAX + 1):
            yv = y_of(l, F0[l], r) * CS[l]
            urL += yv[0] * Sh["Y"][l][0]
            utL += yv[1] * Sh["Bt"][l][0]
        # local basis vectors -> components on the grid basis
        rh_l = np.array([sth, 0.0, cth])
        th_l = np.array([cth, 0.0, -sth])
        u3 = urL * rh_l + utL * th_l
        rh_g = np.array([st[i], 0.0, ct[i]])
        th_g = np.array([ct[i], 0.0, -st[i]])
        ur[i] = u3 @ rh_g
        ut[i] = u3 @ th_g
    return ur, ut


print("== CTRL-1 machinery control (translated fixed field) ==")
urp_, utp_ = field_at_shifted(+delta)
urm_, utm_ = field_at_shifted(-delta)
Uc, Vc = proj(0.5 * (urp_ - urm_), 0.5 * (utp_ - utm_))
for lp in range(1, LAUD + 1):
    rU = Uc[lp] / Urhs[lp]
    rV = Vc[lp] / Vrhs[lp]
    print("%2d  U %8.5f<%6.1f   V %8.5f<%6.1f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV))))

# ---------------- CTRL-2: toroidal free space --------------------
poleT = _pole_coupling(LMAX)
ctl = {l: poleT[1][l][0] for l in range(1, LMAX + 1)}   # hv=(1,0)


def _Yth(l, r):
    return (_Ytor(l, w, r, m0, ("j",), None)
            + 1j * _Ytor(l, w, r, m0, ("y",), None))


def solve_tor_free(l, b):
    Mx = np.zeros((2, 2), dtype=complex)
    Mx[:, 0] = _Yth(l, b)[:, 0]
    Mx[:, 1] = -_Ytor(l, w, b, m0, ("j",), None)[:, 0]
    J = np.array([1.0 / (mu * b * b), 0.0], dtype=complex)
    sc = np.max(np.abs(Mx), axis=0)
    x = np.linalg.solve(Mx / sc, J) / sc
    return dict(cout=x[0], cin=x[1], b=b)


def W_of(l, sol, r):
    if r >= sol["b"]:
        return (_Yth(l, r)[:, 0] * sol["cout"])
    return (_Ytor(l, w, r, m0, ("j",), None)[:, 0] * sol["cin"])


T0 = {l: solve_tor_free(l, r0) for l in range(1, LMAX + 1)}
Tm = {l: solve_tor_free(l, r0 - delta) for l in range(1, LMAX + 1)}
Tp = {l: solve_tor_free(l, r0 + delta) for l in range(1, LMAX + 1)}

# m=1 shapes on the grid
S1 = tfe._shapes(1, LMAX + 2, ct, st)
Ct1, Cp1 = S1["Ct"], S1["Cp"]
# d/dtheta of (Ct, Cp): from strain2 pieces? use analytic:
# dCt = -i m f (dY/st - Y ct/st^2), dCp = f d2Y  (m=1, f=1/sqrt(L))
ll = np.arange(LMAX + 3)
f1 = np.zeros(LMAX + 3)
f1[1:] = 1.0 / np.sqrt(ll[1:] * (ll[1:] + 1.0))
Y1, dY1 = S1["Y"], S1["Bt"]
d2Y1 = S1["eB"][0] / 2.0
cot = ct / st
dCt1 = -1j * 1 * f1[:LMAX + 3, None] * (dY1 / st - Y1 * cot / st)
dCp1 = f1[:LMAX + 3, None] * d2Y1


def projW(gt, gp):
    """W-coefficient projections onto orthonormal C_l,m=1."""
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


# RHS: delta * (zhat.grad) u_SH projected on C (ur = 0 for SH)
ut, up, utp_, upp_, dtut, dtup = tor_line(rF, T0)
gt = ct * utp_ - (st / rF) * (0.0 + dtut)
gp = ct * upp_ - (st / rF) * dtup
Wrhs = delta * projW(gt, gp)
# LHS: diagonal source-shift FD
Wlhs = np.zeros(LMAX + 3, dtype=complex)
for l in range(1, LMAX + 1):
    Wlhs[l] = 0.5 * (W_of(l, Tm[l], rF)[0]
                     - W_of(l, Tp[l], rF)[0]) * ctl[l]
print("\n== CTRL-2 toroidal free-space covariance "
      "([W]=q/(mu b^2)) ==")
for lp in range(1, LAUD + 1):
    rW = Wlhs[lp] / Wrhs[lp]
    print("%2d  W %8.5f<%6.1f" % (lp, abs(rW),
                                  np.degrees(np.angle(rW))))

# ---------------- MAIN recap: spheroidal -------------------------
Ulhs = np.zeros(LMAX + 3, dtype=complex)
Vlhs = np.zeros(LMAX + 3, dtype=complex)
for l in range(1, LMAX + 1):
    ym = y_of(l, Fm[l], rF) * CS[l]
    yp_ = y_of(l, Fp[l], rF) * CS[l]
    Ulhs[l] = 0.5 * (ym[0] - yp_[0])
    Vlhs[l] = 0.5 * (ym[1] - yp_[1])
print("\n== MAIN spheroidal free-space covariance (recap) ==")
for lp in range(1, LAUD + 1):
    rU = Ulhs[lp] / Urhs[lp]
    rV = Vlhs[lp] / Vrhs[lp]
    print("%2d  U %8.5f<%6.1f   V %8.5f<%6.1f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV))))
