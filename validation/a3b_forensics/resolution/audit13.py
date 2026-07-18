#!/usr/bin/env python3
"""Audit 13: m'=+1-resolved coefficient comparison of the Mrt gate
on the homog ball at delta=3000 (noise-suppressed), lmax=12.
Separates PSV (U,V) and SH (W) channels per lp.
For L=1 zonal relief: [same-parity] couples lp = l+-1,
[conversion] couples lp = l. m=0 audits showed same-parity blocks
clean; this isolates the conversion blocks."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

LMAX = 12
LAUD = 10
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3))
Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
dl = 3000.0
hLM = dl * np.sqrt(4.0 * np.pi / 3.0)

# (theta, phi) grid
NT, NPH = 64, 8
xg, wg = np.polynomial.legendre.leggauss(NT)
tg = np.arccos(xg)
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(tg, pg, indexing="ij")
dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                 np.cos(TH)], axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that = np.stack([np.cos(thv) * np.cos(phv),
                 np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat = np.stack([-np.sin(phv), np.cos(phv), 0.0 * phv], axis=1)
wq = np.repeat(wg, NPH) * (2.0 * np.pi / NPH)

S1 = tfe._shapes(1, LMAX + 1, np.cos(tg), np.sin(tg))
Y1 = np.repeat(S1["Y"], NPH, axis=1).reshape(LMAX + 2, -1)
Bt1 = np.repeat(S1["Bt"], NPH, axis=1).reshape(LMAX + 2, -1)
Bp1 = np.repeat(S1["Bp"], NPH, axis=1).reshape(LMAX + 2, -1)
Ct1 = np.repeat(S1["Ct"], NPH, axis=1).reshape(LMAX + 2, -1)
Cp1 = np.repeat(S1["Cp"], NPH, axis=1).reshape(LMAX + 2, -1)
eip = np.exp(1j * phv)
ll = np.arange(LMAX + 2)
Lp = ll * (ll + 1.0)
Lp[0] = 1.0


def coeffs(u):
    """(U, V, W) m'=+1 coefficient arrays of a pointwise field."""
    ur = np.einsum('nc,nc->n', dirs, u)
    ut = np.einsum('nc,nc->n', that, u)
    up = np.einsum('nc,nc->n', phat, u)
    U = np.einsum('ln,n->l', np.conj(Y1 * eip), ur * wq)
    V = np.einsum('ln,n->l', np.conj(Bt1 * eip), ut * wq) \
        + np.einsum('ln,n->l', np.conj(Bp1 * eip), up * wq)
    W = np.einsum('ln,n->l', np.conj(Ct1 * eip), ut * wq) \
        + np.einsum('ln,n->l', np.conj(Cp1 * eip), up * wq)
    return U, V / Lp, W


du_fd = {}
for s in (+1.0, -1.0):
    dp = dirs + s * (dl / A) * np.sin(thv)[:, None] * that
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra(lay, src - s * dl * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    for part in ("psv", "sh"):
        du_fd[part] = du_fd.get(part, 0.0) \
            + 0.5 * s * us[part][0, :, :, 0]
du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs,
                        relief=[("top", 1, 0, hLM)], **kw)

Ue, Ve, _ = coeffs(du["psv"][0, :, :, 0])
_, _, We = coeffs(du["sh"][0, :, :, 0])
Uf, Vf, _ = coeffs(du_fd["psv"])
_, _, Wf = coeffs(du_fd["sh"])

print("m'=+1 coefficients, delta=3000, homog ball, Mrt, lmax=%d"
      % LMAX)
print("lp  |U_fd|      eng/fd U        eng/fd V     |"
      "  |W_fd|      eng/fd W")
for lp in range(1, LAUD + 1):
    rU = Ue[lp] / Uf[lp]
    rV = Ve[lp] / Vf[lp]
    rW = We[lp] / Wf[lp]
    print("%2d  %.2e  %6.3f<%5.0f  %6.3f<%5.0f  |  %.2e"
          "  %6.3f<%5.0f"
          % (lp, abs(Uf[lp]), abs(rU), np.degrees(np.angle(rU)),
             abs(rV), np.degrees(np.angle(rV)), abs(Wf[lp]),
             abs(rW), np.degrees(np.angle(rW))))
