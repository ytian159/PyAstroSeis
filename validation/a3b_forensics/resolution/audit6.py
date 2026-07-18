#!/usr/bin/env python3
"""Audit stage 6:
 (1) PSV free-space (w+vsrc) jump at r0 vs the BALL jump from
     audit2/3 (local algebra -> must be identical).
 (2) SH part of the actual translation gate on the HOMOG BALL
     (Mrt source, engine relief_spectra vs FD) — confirm docs'
     'SH passes exactly' in the minimal config.
"""
import numpy as np
from common import (A, CS, Jvec, LAUD, LMAX, NT, ct, st, delta,
                    hLM, m0, mats, proj, r0, solve_l, solve_l_free,
                    tfe, u_line, w, y_of, zgrad_proj, _Ysolid,
                    lay)
from pyastroseis import spectral as spsp

zhat = np.array([0.0, 0.0, 1.0])

# ---------- (1) free-space jump fit ------------------------------
F0 = {l: solve_l_free(l, r0) for l in range(1, LMAX + 1)}
Fm = {l: solve_l_free(l, r0 - delta) for l in range(1, LMAX + 1)}
Fp = {l: solve_l_free(l, r0 + delta) for l in range(1, LMAX + 1)}
B0 = {l: solve_l(l, r0) for l in range(1, LMAX + 1)}
Bm = {l: solve_l(l, r0 - delta) for l in range(1, LMAX + 1)}
Bp = {l: solve_l(l, r0 + delta) for l in range(1, LMAX + 1)}


def exact_u1_lp(r, S0, Sm, Sp):
    Uw, Vw = zgrad_proj(r, S0)
    Uw, Vw = -delta * Uw, -delta * Vw
    for l in range(1, LMAX + 1):
        ym = y_of(l, Sm[l], r) * CS[l]
        yp_ = y_of(l, Sp[l], r) * CS[l]
        Uw[l] += 0.5 * (ym[0] - yp_[0])
        Vw[l] += 0.5 * (ym[1] - yp_[1])
    return Uw, Vw


def _Yh(l, r):
    return (_Ysolid(l, w, r, m0, ("j",), None)
            + 1j * _Ysolid(l, w, r, m0, ("y",), None))


def fit_jump(S0, Sm, Sp, above_basis):
    r_above = np.linspace(r0 + 500.0, A - 500.0, 12)
    r_below = np.linspace(0.35 * A, r0 - 500.0, 10)
    sa = [exact_u1_lp(r, S0, Sm, Sp) for r in r_above]
    sb = [exact_u1_lp(r, S0, Sm, Sp) for r in r_below]
    out = {}
    for lp in range(1, LAUD + 1):
        Ca, ba = [], []
        for i, r in enumerate(r_above):
            Y4 = above_basis(lp, r)
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
        out[lp] = (above_basis(lp, r0) @ cA
                   - _Ysolid(lp, w, r0, m0, ("j",), None) @ cB)
    return out


jump_free = fit_jump(F0, Fm, Fp, _Yh)
jump_ball = fit_jump(B0, Bm, Bp,
                     lambda l, r: _Ysolid(l, w, r, m0, ("j", "y"),
                                          None))
print("== (1) jump at r0: FREE SPACE vs BALL (must match) ==")
print("lp   |jump_ball| (U,V,R,S)              free/ball ratios")
for lp in range(1, LAUD + 1):
    jb, jf = jump_ball[lp], jump_free[lp]
    rr = jf / jb
    print("%2d  " % lp
          + " ".join("%8.2e" % abs(v) for v in jb) + "   "
          + " ".join("%6.3f<%4.0f" % (abs(v),
                                      np.degrees(np.angle(v)))
                     for v in rr))

# ---------- (2) SH gate on the homog ball ------------------------
Mrt = np.zeros((3, 3))
Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
th = np.linspace(0.35, np.pi - 0.35, 9)
dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
that = np.stack([np.cos(th), 0.0 * th, -np.sin(th)], axis=1)
du_fd = {}
for s in (+1.0, -1.0):
    dp = dirs + s * (delta / A) * np.sin(th)[:, None] * that
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra(lay, src - s * delta * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    for part in ("psv", "sh"):
        du_fd[part] = du_fd.get(part, 0.0) \
            + 0.5 * s * us[part][0, :, :, 0]
du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs,
                        relief=[("top", 1, 0, hLM)], **kw)
print("\n== (2) translation gate, homog ball, Mrt ==")
for part in ("psv", "sh"):
    e = (np.abs(du[part][0, :, :, 0] - du_fd[part]).max()
         / np.abs(du_fd[part]).max())
    print("  %s: engine vs FD rel err %.3e" % (part, e))
