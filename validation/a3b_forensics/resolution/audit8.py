#!/usr/bin/env python3
"""Audit stage 8: character of the source-family defect.

 E1  delta-scaling: measured (w+vsrc) jump at r0 for
     delta = 10, 30, 90 m -> jump/delta constant (systematic,
     first-order family error) vs ~1/delta (absolute noise floor).
 E1b jitter: same delta, r0 -> r0 + 0.5 m: smooth family error
     moves smoothly; FD-roundoff noise decorrelates.
 E2  defect rate at the 5-km scale: J_true'(b) extracted from the
     ANALYTIC field's coefficient-jump family vs J_code'(b), both
     by 5-km FD; compare against the 30-m measured jump/delta.
"""
import numpy as np
from common import (A, CS, Jvec, LAUD, LMAX, NT, ct, st, lam, m0,
                    mu, proj, r0, rho, solve_l_free, w, y_of,
                    zgrad_proj, _Ysolid)

w2c = w * w
vp_c = np.sqrt((lam + 2.0 * mu) / rho)
vs_c = np.sqrt(mu / rho)
ka = w / vp_c
kb = w / vs_c
M0 = 1.0e18


def free_family(b):
    return {l: solve_l_free(l, b) for l in range(1, LMAX + 1)}


def wv_lp(r, S0, Sm, Sp, dl):
    """(U,V) projections of w+vsrc at radius r for FD step dl."""
    Uw, Vw = zgrad_proj(r, S0)
    Uw, Vw = -dl * Uw, -dl * Vw
    for l in range(1, LMAX + 1):
        ym = y_of(l, Sm[l], r) * CS[l]
        yp_ = y_of(l, Sp[l], r) * CS[l]
        Uw[l] += 0.5 * (ym[0] - yp_[0])
        Vw[l] += 0.5 * (ym[1] - yp_[1])
    return Uw, Vw


def _Yh(l, r):
    return (_Ysolid(l, w, r, m0, ("j",), None)
            + 1j * _Ysolid(l, w, r, m0, ("y",), None))


def fit_jump(S0, Sm, Sp, dl, base=None):
    bb_ = r0 if base is None else base
    r_above = np.linspace(bb_ + 500.0, A - 500.0, 12)
    r_below = np.linspace(0.35 * A, bb_ - 500.0, 10)
    sa = [wv_lp(r, S0, Sm, Sp, dl) for r in r_above]
    sb = [wv_lp(r, S0, Sm, Sp, dl) for r in r_below]
    out = {}
    for lp in range(1, LAUD + 1):
        Ca, ba = [], []
        for i, r in enumerate(r_above):
            Y4 = _Yh(lp, r)
            Ca.append(Y4[0]); ba.append(sa[i][0][lp])
            Ca.append(Y4[1]); ba.append(sa[i][1][lp])
        Ca, ba = np.array(Ca), np.array(ba)
        sca = np.max(np.abs(Ca), axis=0)
        cA = np.linalg.lstsq(Ca / sca, ba, rcond=None)[0] / sca
        Cb, bb = [], []
        for i, r in enumerate(r_below):
            Y2 = _Ysolid(lp, w, r, m0, ("j",), None)
            Cb.append(Y2[0]); bb.append(sb[i][0][lp])
            Cb.append(Y2[1]); bb.append(sb[i][1][lp])
        Cb, bb = np.array(Cb), np.array(bb)
        scb = np.max(np.abs(Cb), axis=0)
        cB = np.linalg.lstsq(Cb / scb, bb, rcond=None)[0] / scb
        out[lp] = (_Yh(lp, bb_) @ cA
                   - _Ysolid(lp, w, bb_, m0, ("j",), None) @ cB)
    return out


print("== E1 delta-scaling of jump/delta (R,S components) ==")
S0 = free_family(r0)
res = {}
for dl in (10.0, 30.0, 90.0):
    Sm = free_family(r0 - dl)
    Sp = free_family(r0 + dl)
    res[dl] = fit_jump(S0, Sm, Sp, dl)
for lp in (2, 3, 4):
    print("lp=%d:" % lp)
    for dl in (10.0, 30.0, 90.0):
        j = res[dl][lp] / dl
        print("   d=%3.0f  U/d %9.3e<%5.0f  R/d %9.3e<%5.0f  "
              "S/d %9.3e<%5.0f"
              % (dl, abs(j[0]), np.degrees(np.angle(j[0])),
                 abs(j[2]), np.degrees(np.angle(j[2])),
                 abs(j[3]), np.degrees(np.angle(j[3]))))

print("\n== E1b jitter r0 -> r0+0.5 m (delta=30) ==")
b2 = r0 + 0.5
S0j = free_family(b2)
Smj = free_family(b2 - 30.0)
Spj = free_family(b2 + 30.0)
jj = fit_jump(S0j, Smj, Spj, 30.0, base=b2)
for lp in (2, 3, 4):
    r_ = jj[lp] / res[30.0][lp]
    print("lp=%d  jitter/base ratios: U %6.3f<%4.0f  R %6.3f<%4.0f"
          "  S %6.3f<%4.0f"
          % (lp, abs(r_[0]), np.degrees(np.angle(r_[0])),
             abs(r_[2]), np.degrees(np.angle(r_[2])),
             abs(r_[3]), np.degrees(np.angle(r_[3]))))

# ---------------- E2: 5-km-scale defect rate ---------------------
def Gmat(x):
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

    dd = gg * (t1(kb) - t1(ka)) + (I - gg) * (t2(kb) - t2(ka))
    return (dd + kb * kb * I * gfun(kb)) / (4.0 * np.pi * rho
                                            * w2c)


def ana_jump(b):
    """coefficient-jump family of the ANALYTIC field at r=b:
    project u_ana on spheres just above/below b, fit to the
    free-space bases, evaluate jump at b."""
    r_above = np.linspace(b + 500.0, A - 500.0, 12)
    r_below = np.linspace(0.35 * A, b - 500.0, 10)

    def line(r):
        ur = np.zeros(NT, dtype=complex)
        ut = np.zeros(NT, dtype=complex)
        eps = 1.0
        for i in range(NT):
            x = np.array([r * st[i], 0.0, r * ct[i]])
            Gp_ = Gmat(x - np.array([0.0, 0.0, b + eps]))
            Gm_ = Gmat(x - np.array([0.0, 0.0, b - eps]))
            un = M0 * (Gp_[:, 2] - Gm_[:, 2]) / (2.0 * eps)
            ur[i] = un @ np.array([st[i], 0.0, ct[i]])
            ut[i] = un @ np.array([ct[i], 0.0, -st[i]])
        return proj(ur, ut)

    sa = [line(r) for r in r_above]
    sb = [line(r) for r in r_below]
    out = {}
    for lp in range(1, LAUD + 1):
        Ca, ba = [], []
        for i, r in enumerate(r_above):
            Y4 = _Yh(lp, r)
            Ca.append(Y4[0]); ba.append(sa[i][0][lp])
            Ca.append(Y4[1]); ba.append(sa[i][1][lp])
        Ca, ba = np.array(Ca), np.array(ba)
        sca = np.max(np.abs(Ca), axis=0)
        cA = np.linalg.lstsq(Ca / sca, ba, rcond=None)[0] / sca
        Cb, bb = [], []
        for i, r in enumerate(r_below):
            Y2 = _Ysolid(lp, w, r, m0, ("j",), None)
            Cb.append(Y2[0]); bb.append(sb[i][0][lp])
            Cb.append(Y2[1]); bb.append(sb[i][1][lp])
        Cb, bb = np.array(Cb), np.array(bb)
        scb = np.max(np.abs(Cb), axis=0)
        cB = np.linalg.lstsq(Cb / scb, bb, rcond=None)[0] / scb
        out[lp] = (_Yh(lp, b) @ cA
                   - _Ysolid(lp, w, b, m0, ("j",), None) @ cB)
    return out


DKM = 5.0e3
Jc_p = {lp: Jvec(lp, r0 + DKM) * CS[lp]
        for lp in range(1, LAUD + 1)}
Jc_m = {lp: Jvec(lp, r0 - DKM) * CS[lp]
        for lp in range(1, LAUD + 1)}
Ja_p = ana_jump(r0 + DKM)
Ja_m = ana_jump(r0 - DKM)
print("\n== E2 defect rate d/db(J_code - J_true) at 5-km scale "
      "vs measured jump/delta (30 m) ==")
for lp in (2, 3, 4):
    rate = ((Jc_p[lp] - Ja_p[lp]) - (Jc_m[lp] - Ja_m[lp])) \
        / (2.0 * DKM)
    meas = res[30.0][lp] / 30.0
    print("lp=%d" % lp)
    for i, nm in enumerate("UVRS"):
        print("   %s  5km-rate %9.3e<%5.0f   30m-meas %9.3e<%5.0f"
              % (nm, abs(rate[i]),
                 np.degrees(np.angle(rate[i])), abs(meas[i]),
                 np.degrees(np.angle(meas[i]))))
