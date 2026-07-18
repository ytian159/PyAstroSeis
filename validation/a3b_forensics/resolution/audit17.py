#!/usr/bin/env python3
"""Audit 17: m=1 engine replication vs relief_spectra vs FD at
delta=3000 (noise-suppressed), LMAX=10."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit15.py").read().split("# TRUTH")[0])  # setup
from pyastroseis.spectral import spheroidal_unit as sph_unit
from pyastroseis.spectral import toroidal_unit as tor_unit
from pyastroseis import spectral as spsp

DL = 3000.0
hL3 = DL * np.sqrt(4.0 * np.pi / 3.0)
ang = tfe.ReliefCouplings(1, 0, 1, LMAX)
keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp", "S", "T", "V", "W",
        "ciso", "cV", "cW")
dc = {k: np.zeros(LMAX + 1, dtype=complex) for k in keys}
for l in range(1, LMAX + 1):
    y4 = y4_at(l, A)
    hh = 0.5
    yp4 = (y4_at(l, A + hh) - y4_at(l, A - hh)) / (2 * hh)
    sc_ = tfe._side_coeffs(l, w, A, m0, y4, wt_at(l, A), yp4=yp4)
    for k in keys:
        dc[k][l] = sc_[k]
jU, jV, jR, jS, jW, jT = tfe._jump_rows(hL3, A, ang, dc)
advU = hL3 * (ang.G0 @ dc["Up"])
advV = hL3 * (ang.GA_v @ dc["Vp"] + ang.GA_w @ dc["Wp"])
advW = hL3 * (ang.GC_v @ dc["Vp"] + ang.GC_w @ dc["Wp"])
U1r = np.zeros(LMAX + 1, dtype=complex)
V1r = np.zeros(LMAX + 1, dtype=complex)
W1r = np.zeros(LMAX + 1, dtype=complex)
ents = _entries(mats, 0, r0, 0)
from pyastroseis.spectral import _tor_entries
for lp in range(1, LMAX + 1):
    U1, V1 = sph_unit(lp, w, ents, np.zeros(4, dtype=complex),
                      bc_rhs=(jR[lp], jS[lp]))
    U1r[lp] = U1 + advU[lp]
    V1r[lp] = V1 + advV[lp]
    ents_t, botp = _tor_entries(mats, lp, r0, 0, 1e-12)
    W1 = tor_unit(lp, w, ents_t, botp, 0.0, bc_rhs=(0.0, jT[lp]))
    W1r[lp] = W1 + advW[lp]

# actual relief_spectra + FD on the (theta,phi) grid, project m=+1
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

Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
zhat = np.array([0.0, 0.0, 1.0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs2,
                        relief=[("top", 1, 0, hL3)], **kw)
Ue, Ve, _ = coeffs2(du["psv"][0, :, :, 0])
_, _, We = coeffs2(du["sh"][0, :, :, 0])
du_fd = {}
for s in (+1.0, -1.0):
    dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra(lay, src - s * DL * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    for part in ("psv", "sh"):
        du_fd[part] = du_fd.get(part, 0.0) \
            + 0.5 * s * us[part][0, :, :, 0]
Uf, Vf, _ = coeffs2(du_fd["psv"])
_, _, Wf = coeffs2(du_fd["sh"])
print("lp   rep/engine U      rep/engine W    |  rep/fd U"
      "        rep/fd V        rep/fd W")
for lp in range(1, LAUD + 1):
    r1 = U1r[lp] / Ue[lp]
    r2 = W1r[lp] / We[lp]
    r3 = U1r[lp] / Uf[lp]
    r4 = V1r[lp] / Vf[lp]
    r5 = W1r[lp] / Wf[lp]
    print("%2d  %7.4f<%5.0f  %7.4f<%5.0f  |  %7.4f<%5.0f"
          "  %7.4f<%5.0f  %7.4f<%5.0f"
          % (lp, abs(r1), np.degrees(np.angle(r1)), abs(r2),
             np.degrees(np.angle(r2)), abs(r3),
             np.degrees(np.angle(r3)), abs(r4),
             np.degrees(np.angle(r4)), abs(r5),
             np.degrees(np.angle(r5))))
