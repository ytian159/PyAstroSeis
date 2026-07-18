#!/usr/bin/env python3
"""Audit 31: per-lp eng/fd for the welded-contrast translation
gate at m=0, delta=3000 (and the artificial-split control)."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

LMAX = 12
LAUD = 9
r0 = A - 637.0e3
w = wk(90.0)
zhat = np.array([0.0, 0.0, 1.0])
Mrr = np.diag([0.0, 0.0, 1.0e18])
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
NT = 96
xg, wg = np.polynomial.legendre.leggauss(NT)
ct, st = xg, np.sqrt(1.0 - xg * xg)
dirs = np.stack([st, 0.0 * st, ct], axis=1)
that = np.stack([ct, 0.0 * ct, -st], axis=1)
S0 = tfe._shapes(0, LMAX + 2, ct, st)
w2 = 2.0 * np.pi * wg
ll = np.arange(LMAX + 3)
Lp = ll * (ll + 1.0)
Lp[0] = 1.0


def proj(u):
    fr = np.einsum('nc,nc->n', dirs, u)
    ft = np.einsum('nc,nc->n', that, u)
    U = np.einsum('ln,n->l', np.conj(S0["Y"]), fr * w2)
    V = np.einsum('ln,n->l', np.conj(S0["Bt"]), ft * w2) / Lp
    return U, V


def gate_lp(layers, relief_locs, dl):
    c10 = dl * np.sqrt(4.0 * np.pi / 3.0)
    du_fd = 0.0
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * st[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(layers, src - s * dl * zhat,
                                   [Mrr], np.array([w]), dp,
                                   l0=False, **kw)
        du_fd = du_fd + 0.5 * s * (us["psv"]
                                   + us["sh"])[0, :, :, 0]
    du = tfe.relief_spectra(layers, src, [Mrr], np.array([w]),
                            dirs,
                            relief=[(loc, 1, 0, c10)
                                    for loc in relief_locs], **kw)
    tot_e = (du["psv"] + du["sh"])[0, :, :, 0]
    Ue, Ve = proj(tot_e)
    Uf, Vf = proj(du_fd)
    return Ue, Ve, Uf, Vf


MID = dict(rho=4600.0, vp=10000.0, vs=5800.0)
L2 = [dict(r_top=4925.5e3, **MID), dict(r_top=A, **SH)]
L2s = [dict(r_top=4925.5e3, **SH), dict(r_top=A, **SH)]
for nm, lay_ in (("weld-contrast", L2), ("null-split", L2s)):
    Ue, Ve, Uf, Vf = gate_lp(lay_, (4925.5e3, "top"), 3000.0)
    print(nm + ":")
    for lp in range(1, LAUD + 1):
        rU = Ue[lp] / Uf[lp]
        rV = Ve[lp] / Vf[lp]
        print(" lp=%2d |dU_fd|=%.2e eng/fd U %7.4f<%5.0f"
              "  V %7.4f<%5.0f"
              % (lp, abs(Uf[lp]), abs(rU),
                 np.degrees(np.angle(rU)), abs(rV),
                 np.degrees(np.angle(rV))))
