#!/usr/bin/env python3
"""Audit 18: m=1 U-channel ratio vs delta (10km, 30km)."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

LMAX = 10
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
NTg, NPH = 48, 8
xg2, wg2 = np.polynomial.legendre.leggauss(NTg)
tg2 = np.arccos(xg2)
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(tg2, pg, indexing="ij")
dirs2 = np.stack([np.sin(TH) * np.cos(PH),
                  np.sin(TH) * np.sin(PH), np.cos(TH)],
                 axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that2 = np.stack([np.cos(thv) * np.cos(phv),
                  np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat2 = np.stack([-np.sin(phv), np.cos(phv), 0.0 * phv], axis=1)
wq = np.repeat(wg2, NPH) * (2.0 * np.pi / NPH)
S1g = tfe._shapes(1, LMAX + 1, np.cos(tg2), np.sin(tg2))
def rep(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)
Y1g, Bt1g, Bp1g = rep(S1g["Y"]), rep(S1g["Bt"]), rep(S1g["Bp"])
Ct1g, Cp1g = rep(S1g["Ct"]), rep(S1g["Cp"])
eip = np.exp(1j * phv)
llg = np.arange(LMAX + 2)
Lpg = llg * (llg + 1.0); Lpg[0] = 1.0
def coeffs2(u):
    ur = np.einsum('nc,nc->n', dirs2, u)
    ut = np.einsum('nc,nc->n', that2, u)
    upp = np.einsum('nc,nc->n', phat2, u)
    U = np.einsum('ln,n->l', np.conj(Y1g * eip), ur * wq)
    V = (np.einsum('ln,n->l', np.conj(Bt1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Bp1g * eip), upp * wq))
    W = (np.einsum('ln,n->l', np.conj(Ct1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1g * eip), upp * wq))
    return U, V / Lpg, W

for DL in (3000.0, 10000.0, 30000.0):
    hL = DL * np.sqrt(4.0 * np.pi / 3.0)
    du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs2,
                            relief=[("top", 1, 0, hL)], **kw)
    Ue, Ve, _ = coeffs2(du["psv"][0, :, :, 0])
    _, _, We = coeffs2(du["sh"][0, :, :, 0])
    du_fd = {}
    for s in (+1.0, -1.0):
        dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(lay, src - s * DL * zhat,
                                   [Mrt], np.array([w]), dp,
                                   l0=False, **kw)
        for part in ("psv", "sh"):
            du_fd[part] = du_fd.get(part, 0.0) \
                + 0.5 * s * us[part][0, :, :, 0]
    Uf, Vf, _ = coeffs2(du_fd["psv"])
    _, _, Wf = coeffs2(du_fd["sh"])
    rats = ["%d:%.3f" % (lp, abs(Ue[lp] / Uf[lp]))
            for lp in range(2, 9)]
    print("delta=%6.0f  U eng/fd: %s" % (DL, " ".join(rats)))
    rats = ["%d:%.3f" % (lp, abs(We[lp] / Wf[lp]))
            for lp in range(2, 9)]
    print("              W eng/fd: %s" % " ".join(rats))
