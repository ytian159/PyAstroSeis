#!/usr/bin/env python3
"""Audit 22: m=1 three-way at delta=3000:
 (a) exact decomposition X = (w+vsrc)(A) + adv
 (b) gate du_fd (mapped stations)
 (c) engine relief_spectra du
"""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit21.py").read().split("def run_fit")[0])
from pyastroseis import spectral_tfe as tfe2
from pyastroseis import spectral as spsp

DL = 3000.0
PS0 = {l: psv_solve(l, R0) for l in range(1, LMAX + 1)}
TS0 = {l: tor_solve(l, R0) for l in range(1, LMAX + 1)}
PSm = {l: psv_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
TSm = {l: tor_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
PSp = {l: psv_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
TSp = {l: tor_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
eps = 2.0
zh = np.array([0.0, 0.0, eps])
rh_g = np.stack([st, 0 * st, ct], axis=1)
th_g = np.stack([ct, 0 * ct, -st], axis=1)
ph_g = np.array([0.0, 1.0, 0.0])

# (a): w + adv pointwise at r=A, plus vsrc coefficients
fr = np.zeros(NT, dtype=complex)
ft = np.zeros(NT, dtype=complex)
fp = np.zeros(NT, dtype=complex)
for i in range(NT):
    x = A * rh_g[i]
    dz = (u_point(x + zh, PS0, TS0)
          - u_point(x - zh, PS0, TS0)) / (2 * eps)
    dr = (u_point(x * (A + eps) / A, PS0, TS0)
          - u_point(x * (A - eps) / A, PS0, TS0)) / (2 * eps)
    v = -DL * dz + DL * ct[i] * dr        # w + adv
    fr[i] = v @ rh_g[i]
    ft[i] = v @ th_g[i]
    fp[i] = v @ ph_g
Ua, Va, Wa = proj_disp(fr, ft, fp)
for l in range(1, LMAX + 1):
    ym = y4_of(l, PSm[l], A) * CSr[l]
    yp = y4_of(l, PSp[l], A) * CSr[l]
    Ua[l] += 0.5 * (ym[0] - yp[0])
    Va[l] += 0.5 * (ym[1] - yp[1])
    wm = wt_of(l, TSm[l], A) * CSt[l]
    wp = wt_of(l, TSp[l], A) * CSt[l]
    Wa[l] += 0.5 * (wm[0] - wp[0])

# (b) and (c) on a (theta,phi) grid
NPH = 8
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(np.arccos(xg), pg, indexing="ij")
dirs2 = np.stack([np.sin(TH) * np.cos(PH),
                  np.sin(TH) * np.sin(PH), np.cos(TH)],
                 axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that2 = np.stack([np.cos(thv) * np.cos(phv),
                  np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat2 = np.stack([-np.sin(phv), np.cos(phv), 0.0 * phv], axis=1)
wq = np.repeat(wg, NPH) * (2.0 * np.pi / NPH)


def rep2(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)


Y1g, Bt1g, Bp1g = rep2(Sh["Y"]), rep2(Sh["Bt"]), rep2(Sh["Bp"])
Ct1g, Cp1g = rep2(Sh["Ct"]), rep2(Sh["Cp"])
eip = np.exp(1j * phv)


def coeffs2(u):
    ur = np.einsum('nc,nc->n', dirs2, u)
    ut = np.einsum('nc,nc->n', that2, u)
    upp = np.einsum('nc,nc->n', phat2, u)
    U = np.einsum('ln,n->l', np.conj(Y1g * eip), ur * wq)
    V = (np.einsum('ln,n->l', np.conj(Bt1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Bp1g * eip), upp * wq))
    W = (np.einsum('ln,n->l', np.conj(Ct1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1g * eip), upp * wq))
    return U, V / Lp, W


Mrt = np.zeros((3, 3))
Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, R0])
zhat = np.array([0.0, 0.0, 1.0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
hL = DL * np.sqrt(4.0 * np.pi / 3.0)
du = tfe2.relief_spectra([dict(r_top=A, **SH)], src, [Mrt],
                         np.array([w]), dirs2,
                         relief=[("top", 1, 0, hL)], **kw)
Ue, Ve, _ = coeffs2(du["psv"][0, :, :, 0])
_, _, We = coeffs2(du["sh"][0, :, :, 0])
du_fd = {}
for s in (+1.0, -1.0):
    dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra([dict(r_top=A, **SH)],
                               src - s * DL * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    for part in ("psv", "sh"):
        du_fd[part] = du_fd.get(part, 0.0) \
            + 0.5 * s * us[part][0, :, :, 0]
Uf, Vf, _ = coeffs2(du_fd["psv"])
_, _, Wf = coeffs2(du_fd["sh"])

print("lp   X/fd U        X/fd W       |  X/eng U       X/eng W")
for lp in range(2, LAUD + 1):
    r1 = Ua[lp] / Uf[lp]
    r2 = Wa[lp] / Wf[lp]
    r3 = Ua[lp] / Ue[lp]
    r4 = Wa[lp] / We[lp]
    print("%2d  %7.4f<%5.0f %7.4f<%5.0f  |  %7.4f<%5.0f"
          " %7.4f<%5.0f"
          % (lp, abs(r1), np.degrees(np.angle(r1)), abs(r2),
             np.degrees(np.angle(r2)), abs(r3),
             np.degrees(np.angle(r3)), abs(r4),
             np.degrees(np.angle(r4))))
