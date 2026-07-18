#!/usr/bin/env python3
"""Audit 21: m=1 interior-jump fit of (w+vsrc), delta-linearity
and jitter stability. Both parities."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  _Ysolid, _Ytor)
from pyastroseis.spheroidal_ref import (source_jumps,
                                        spheroidal_pole)
from pyastroseis.toroidal_modes import _pole_coupling
from tests.test_spectral import A, SH, wk

LMAX = 10
LAUD = 8
R0 = A - 637.0e3
w = wk(90.0)
mats = _mats_at(make_stack([dict(r_top=A, **SH)]), w, 50.0, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
M0 = 1.0e18
DY, DG = spheroidal_pole(LMAX)
poleT = _pole_coupling(LMAX)
CSr = {l: M0 * DG[1][l][0] for l in range(1, LMAX + 1)}
CSt = {l: M0 * poleT[1][l][0] for l in range(1, LMAX + 1)}


def psv_solve(l, b):
    ents = _entries(mats, 0, b, 0)
    L = l * (l + 1.0)
    F0 = np.array([0, 0, 0, 1.0 / (L * b * b)], dtype=complex)
    F1 = np.array([0, 0, -1.0 / b ** 3, 3.0 / (L * b ** 3)],
                  dtype=complex)
    J = source_jumps(l, w, b, rho, lam, mu, F0, F1)
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ysolid(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ysolid(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    bv = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        Am[row:row + 4, ofs[i + 1]:ofs[i + 2]] = Yb[i + 1]
        Am[row:row + 4, ofs[i]:ofs[i + 1]] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row:row + 4] = J
        row += 4
    Am[row, ofs[-2]:ofs[-1]] = Yt[-1][2]
    Am[row + 1, ofs[-2]:ofs[-1]] = Yt[-1][3]
    sc = np.max(np.abs(Am), axis=0)
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs, b=b)


def tor_solve(l, b):
    ents = _entries(mats, 0, b, 0)
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ytor(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ytor(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    bv = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        Am[row:row + 2, ofs[i + 1]:ofs[i + 2]] = Yb[i + 1]
        Am[row:row + 2, ofs[i]:ofs[i + 1]] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row] = 1.0 / (mu * b * b)
        row += 2
    Am[row, ofs[-2]:ofs[-1]] = Yt[-1][1]
    sc = np.max(np.abs(Am), axis=0)
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs, b=b)


def y4_of(l, s, r):
    i = 1 if r >= s["b"] else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return (_Ysolid(l, w, r, m0, kinds, None)
            @ s["x"][s["ofs"][i]:s["ofs"][i + 1]])


def wt_of(l, s, r):
    i = 1 if r >= s["b"] else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return (_Ytor(l, w, r, m0, kinds, None)
            @ s["x"][s["ofs"][i]:s["ofs"][i + 1]])


NT = 64
xg, wg = np.polynomial.legendre.leggauss(NT)
ct, st = xg, np.sqrt(1.0 - xg * xg)
Sh = tfe._shapes(1, LMAX + 1, ct, st)
w2 = 2.0 * np.pi * wg
ll = np.arange(LMAX + 2)
Lp = ll * (ll + 1.0)
Lp[0] = 1.0


def proj_disp(fr, ft, fp):
    U = np.einsum('ln,n->l', np.conj(Sh["Y"]), fr * w2)
    V = (np.einsum('ln,n->l', np.conj(Sh["Bt"]), ft * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Bp"]), fp * w2)) / Lp
    W = (np.einsum('ln,n->l', np.conj(Sh["Ct"]), ft * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Cp"]), fp * w2))
    return U, V, W


def u_point(x, PS, TS):
    r = np.linalg.norm(x)
    cth = np.clip(x[2] / r, -1, 1)
    sth = np.hypot(x[0], x[1]) / r
    ph_ = np.arctan2(x[1], x[0])
    Sh_ = tfe._shapes(1, LMAX + 1, np.array([cth]),
                      np.array([sth]))
    e1 = np.exp(1j * ph_)
    ur = ut = up = 0j
    for l in range(1, LMAX + 1):
        y4 = y4_of(l, PS[l], r) * CSr[l]
        wt = wt_of(l, TS[l], r) * CSt[l]
        ur += y4[0] * Sh_["Y"][l][0] * e1
        ut += (y4[1] * Sh_["Bt"][l][0]
               + wt[0] * Sh_["Ct"][l][0]) * e1
        up += (y4[1] * Sh_["Bp"][l][0]
               + wt[0] * Sh_["Cp"][l][0]) * e1
    rh = np.array([sth * np.cos(ph_), sth * np.sin(ph_), cth])
    th = np.array([cth * np.cos(ph_), cth * np.sin(ph_), -sth])
    p2 = np.array([-np.sin(ph_), np.cos(ph_), 0.0])
    return ur * rh + ut * th + up * p2


def run_fit(b0, DL):
    PS0 = {l: psv_solve(l, b0) for l in range(1, LMAX + 1)}
    TS0 = {l: tor_solve(l, b0) for l in range(1, LMAX + 1)}
    PSm = {l: psv_solve(l, b0 - DL) for l in range(1, LMAX + 1)}
    TSm = {l: tor_solve(l, b0 - DL) for l in range(1, LMAX + 1)}
    PSp = {l: psv_solve(l, b0 + DL) for l in range(1, LMAX + 1)}
    TSp = {l: tor_solve(l, b0 + DL) for l in range(1, LMAX + 1)}
    eps = 2.0
    zh = np.array([0.0, 0.0, eps])

    def wv_lp(r):
        # w = -DL * dz u0 pointwise on the theta line at radius r
        fr = np.zeros(NT, dtype=complex)
        ft = np.zeros(NT, dtype=complex)
        fp = np.zeros(NT, dtype=complex)
        rh_g = np.stack([st, 0 * st, ct], axis=1)
        th_g = np.stack([ct, 0 * ct, -st], axis=1)
        ph_g = np.array([0.0, 1.0, 0.0])
        for i in range(NT):
            x = r * rh_g[i]
            dz = (u_point(x + zh, PS0, TS0)
                  - u_point(x - zh, PS0, TS0)) / (2 * eps)
            v = -DL * dz
            fr[i] = v @ rh_g[i]
            ft[i] = v @ th_g[i]
            fp[i] = v @ ph_g
        U, V, W = proj_disp(fr, ft, fp)
        # vsrc: diagonal FD
        for l in range(1, LMAX + 1):
            ym = y4_of(l, PSm[l], r) * CSr[l]
            yp = y4_of(l, PSp[l], r) * CSr[l]
            U[l] += 0.5 * (ym[0] - yp[0])
            V[l] += 0.5 * (ym[1] - yp[1])
            wm = wt_of(l, TSm[l], r) * CSt[l]
            wp = wt_of(l, TSp[l], r) * CSt[l]
            W[l] += 0.5 * (wm[0] - wp[0])
        return U, V, W

    r_above = np.linspace(b0 + 2 * DL + 500.0, A - 500.0, 10)
    r_below = np.linspace(0.35 * A, b0 - 2 * DL - 500.0, 8)
    sa = [wv_lp(r) for r in r_above]
    sb = [wv_lp(r) for r in r_below]
    out = {}
    for lp in range(1, LAUD + 1):
        Ca, ba = [], []
        for i, r in enumerate(r_above):
            Y4 = _Ysolid(lp, w, r, m0, ("j", "y"), None)
            Ca.append(Y4[0]); ba.append(sa[i][0][lp])
            Ca.append(Y4[1]); ba.append(sa[i][1][lp])
        Ca, ba = np.array(Ca), np.array(ba)
        sca = np.max(np.abs(Ca), axis=0)
        cA, = [np.linalg.lstsq(Ca / sca, ba, rcond=None)[0] / sca]
        resA = np.linalg.norm((Ca) @ cA - ba) / np.linalg.norm(ba)
        Cb, bb = [], []
        for i, r in enumerate(r_below):
            Y2 = _Ysolid(lp, w, r, m0, ("j",), None)
            Cb.append(Y2[0]); bb.append(sb[i][0][lp])
            Cb.append(Y2[1]); bb.append(sb[i][1][lp])
        Cb, bb = np.array(Cb), np.array(bb)
        scb = np.max(np.abs(Cb), axis=0)
        cB = np.linalg.lstsq(Cb / scb, bb, rcond=None)[0] / scb
        jP = (_Ysolid(lp, w, b0, m0, ("j", "y"), None) @ cA
              - _Ysolid(lp, w, b0, m0, ("j",), None) @ cB)
        # SH fit
        Ta, ta = [], []
        for i, r in enumerate(r_above):
            Y2 = _Ytor(lp, w, r, m0, ("j", "y"), None)
            Ta.append(Y2[0]); ta.append(sa[i][2][lp])
        Ta, ta = np.array(Ta), np.array(ta)
        sct = np.max(np.abs(Ta), axis=0)
        cT = np.linalg.lstsq(Ta / sct, ta, rcond=None)[0] / sct
        Tb, tb = [], []
        for i, r in enumerate(r_below):
            Y1_ = _Ytor(lp, w, r, m0, ("j",), None)
            Tb.append(Y1_[0]); tb.append(sb[i][2][lp])
        Tb, tb = np.array(Tb), np.array(tb)
        sctb = np.max(np.abs(Tb), axis=0)
        cTb = np.linalg.lstsq(Tb / sctb, tb, rcond=None)[0] / sctb
        jT_ = (_Ytor(lp, w, b0, m0, ("j", "y"), None) @ cT
               - _Ytor(lp, w, b0, m0, ("j",), None) @ cTb)
        out[lp] = (jP, jT_, resA)
    return out


j3 = run_fit(R0, 3000.0)
j1 = run_fit(R0, 1000.0)
jj = run_fit(R0 + 0.7, 3000.0)
print("m=1 (w+vsrc) jump at r0: delta-linearity + jitter")
print("lp  |jU/d|(3km)   (1km)m       jit(3km)   | |jW/d|(3km)"
      "   (1km)       jit")
for lp in (2, 3, 4, 5):
    a = j3[lp][0][0] / 3000.0
    b = j1[lp][0][0] / 1000.0
    c = jj[lp][0][0] / 3000.0
    d3 = j3[lp][1][0] / 3000.0
    d1 = j1[lp][1][0] / 1000.0
    dj = jj[lp][1][0] / 3000.0
    print("%2d  %9.3e  %9.3e  %9.3e  |  %9.3e  %9.3e  %9.3e"
          % (lp, abs(a), abs(b), abs(c), abs(d3), abs(d1),
             abs(dj)))
    print("     fit resid above: %.1e" % j3[lp][2])
