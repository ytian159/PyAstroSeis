#!/usr/bin/env python3
"""Audit 32 FINAL TABLE: m=+-1 (Mrt) translation gate, delta=3000,
TOTAL-field label-frame (U,V,W) projections per lp:
homog ball | solid weld | full 3-layer."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, B, C, IC, OC, SH, wk

LMAX = 12
LAUD = 8
r0 = A - 637.0e3
w = wk(90.0)
zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
DL = 3000.0
c10 = DL * np.sqrt(4.0 * np.pi / 3.0)

NT, NPH = 64, 8
xg, wg = np.polynomial.legendre.leggauss(NT)
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(np.arccos(xg), pg, indexing="ij")
dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                 np.cos(TH)], axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that = np.stack([np.cos(thv) * np.cos(phv),
                 np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat = np.stack([-np.sin(phv), np.cos(phv), 0.0 * phv], axis=1)
wq = np.repeat(wg, NPH) * (2.0 * np.pi / NPH)
S1 = tfe._shapes(1, LMAX + 1, np.cos(np.arccos(xg)),
                 np.sin(np.arccos(xg)))


def rep2(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)


Y1, Bt1, Bp1 = rep2(S1["Y"]), rep2(S1["Bt"]), rep2(S1["Bp"])
Ct1, Cp1 = rep2(S1["Ct"]), rep2(S1["Cp"])
eip = np.exp(1j * phv)
ll = np.arange(LMAX + 2)
Lp = ll * (ll + 1.0)
Lp[0] = 1.0


def coeffs(u):
    ur = np.einsum('nc,nc->n', dirs, u)
    ut = np.einsum('nc,nc->n', that, u)
    up = np.einsum('nc,nc->n', phat, u)
    U = np.einsum('ln,n->l', np.conj(Y1 * eip), ur * wq)
    V = (np.einsum('ln,n->l', np.conj(Bt1 * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Bp1 * eip), up * wq)) / Lp
    W = (np.einsum('ln,n->l', np.conj(Ct1 * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1 * eip), up * wq))
    return U, V, W


def run(layers, locs, name):
    du_fd = 0.0
    for s in (+1.0, -1.0):
        dp = dirs + s * (DL / A) * np.sin(thv)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(layers, src - s * DL * zhat,
                                   [Mrt], np.array([w]), dp,
                                   l0=False, **kw)
        du_fd = du_fd + 0.5 * s * (us["psv"]
                                   + us["sh"])[0, :, :, 0]
    du = tfe.relief_spectra(layers, src, [Mrt], np.array([w]),
                            dirs, relief=[(loc, 1, 0, c10)
                                          for loc in locs], **kw)
    tot = (du["psv"] + du["sh"])[0, :, :, 0]
    Ue, Ve, We = coeffs(tot)
    Uf, Vf, Wf = coeffs(du_fd)
    print(name + " (eng/fd, TOTAL-field projections):")
    for lp in range(1, LAUD + 1):
        print(" lp=%2d U %7.4f<%4.0f V %7.4f<%4.0f W %7.4f<%4.0f"
              "  (|Uf| %.1e |Wf| %.1e)"
              % (lp, abs(Ue[lp] / Uf[lp]),
                 np.degrees(np.angle(Ue[lp] / Uf[lp])),
                 abs(Ve[lp] / Vf[lp]),
                 np.degrees(np.angle(Ve[lp] / Vf[lp])),
                 abs(We[lp] / Wf[lp]),
                 np.degrees(np.angle(We[lp] / Wf[lp])),
                 abs(Uf[lp]), abs(Wf[lp])))


MID = dict(rho=4600.0, vp=10000.0, vs=5800.0)
run([dict(r_top=A, **SH)], ("top",), "homog")
run([dict(r_top=4925.5e3, **MID), dict(r_top=A, **SH)],
    (4925.5e3, "top"), "weld")
run([dict(r_top=C, **IC), dict(r_top=B, **OC),
     dict(r_top=A, **SH)], (C, B, "top"), "3-layer")
