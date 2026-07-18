#!/usr/bin/env python3
"""Audit 27: FOUR fields, ONE namespace, grid m'=+1 projections:
   X(exact m=+1) | my_fd(m=+1 legs) | code_fd | engine."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit21.py").read().split("def run_fit")[0])
from pyastroseis import spectral as spsp
from pyastroseis import spectral_tfe as tfe2

DL = 3000.0
PS0 = {l: psv_solve(l, R0) for l in range(1, LMAX + 1)}
TS0 = {l: tor_solve(l, R0) for l in range(1, LMAX + 1)}
PSm = {l: psv_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
TSm = {l: tor_solve(l, R0 - DL) for l in range(1, LMAX + 1)}
PSp = {l: psv_solve(l, R0 + DL) for l in range(1, LMAX + 1)}
TSp = {l: tor_solve(l, R0 + DL) for l in range(1, LMAX + 1)}

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
    up = np.einsum('nc,nc->n', phat2, u)
    U = np.einsum('ln,n->l', np.conj(Y1g * eip), ur * wq)
    V = (np.einsum('ln,n->l', np.conj(Bt1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Bp1g * eip), up * wq))
    W = (np.einsum('ln,n->l', np.conj(Ct1g * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1g * eip), up * wq))
    return U, V / Lp, W


# field 1: my_fd (m=+1 legs at mapped dirs)
my_fd = np.zeros((len(dirs2), 3), dtype=complex)
for i in range(len(dirs2)):
    d = dirs2[i]
    for s, PSx, TSx in ((+1.0, PSm, TSm), (-1.0, PSp, TSp)):
        dp = d + s * (DL / A) * np.sin(thv[i]) * that2[i]
        dp = dp / np.linalg.norm(dp)
        my_fd[i] += 0.5 * s * u_point(A * dp, PSx, TSx)
U_my, V_my, W_my = coeffs2(my_fd)

# field 2: code_fd
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, R0])
zhat = np.array([0.0, 0.0, 1.0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
cf_psv = 0.0
cf_sh = 0.0
for s in (+1.0, -1.0):
    dp = dirs2 + s * (DL / A) * np.sin(thv)[:, None] * that2
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra([dict(r_top=A, **SH)],
                               src - s * DL * zhat, [Mrt],
                               np.array([w]), dp, l0=False, **kw)
    cf_psv = cf_psv + 0.5 * s * us["psv"][0, :, :, 0]
    cf_sh = cf_sh + 0.5 * s * us["sh"][0, :, :, 0]
U_cf, V_cf, _ = coeffs2(cf_psv)
_, _, W_cf = coeffs2(cf_sh)
U_cft, V_cft, W_cft = coeffs2(cf_psv + cf_sh)

# field 3: engine
du = tfe2.relief_spectra([dict(r_top=A, **SH)], src, [Mrt],
                         np.array([w]), dirs2,
                         relief=[("top", 1, 0,
                                  DL * np.sqrt(4 * np.pi / 3))],
                         **kw)
U_en, V_en, _ = coeffs2(du["psv"][0, :, :, 0])
_, _, W_en = coeffs2(du["sh"][0, :, :, 0])

# field 4: X (exact m=+1: w+adv pointwise + vsrc), same grid
X = np.zeros((len(dirs2), 3), dtype=complex)
eps = 2.0
zh = np.array([0.0, 0.0, eps])
for i in range(len(dirs2)):
    x = A * dirs2[i]
    dz = (u_point(x + zh, PS0, TS0)
          - u_point(x - zh, PS0, TS0)) / (2 * eps)
    dr = (u_point(x * (A + eps) / A, PS0, TS0)
          - u_point(x * (A - eps) / A, PS0, TS0)) / (2 * eps)
    X[i] = -DL * dz + DL * np.cos(thv[i]) * dr \
        + 0.5 * (u_point(x, PSm, TSm) - u_point(x, PSp, TSp))
U_X, V_X, W_X = coeffs2(X)

print("lp   my/code(U)  X/code(U)  eng/code(U)  "
      "my/code(W)  eng/code(W)")
for lp in (3, 4, 5, 6):
    print("%2d   %8.4f  %8.4f  %8.4f   %8.4f  %8.4f"
          % (lp, abs(U_my[lp] / U_cf[lp]),
             abs(U_X[lp] / U_cf[lp]),
             abs(U_en[lp] / U_cf[lp]),
             abs(W_my[lp] / W_cf[lp]),
             abs(W_en[lp] / W_cf[lp])))
print("\n(cross-check: code psv+sh total vs psv-only U: "
      "%r)" % np.allclose(U_cft[3], U_cf[3]))

print("RAW a27: U_my[4]=%r U_cf[4]=%r" % (U_my[4], U_cf[4]))
np.save("a27_myfd.npy", my_fd)
np.save("a27_codefd.npy", cf_psv + cf_sh)
