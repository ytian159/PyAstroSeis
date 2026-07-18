#!/usr/bin/env python3
"""Audit 14: analytic anchor for the m=+-1 (Mrt) source family.
Free-space code fields (rh-tag PSV jumps + toroidal [W]=q/(mu b^2))
vs the analytic full-space Mzx+Mxz moment field, m'=+1
coefficients (U, V, W) at rF, three depths."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis.spectral import _Ysolid, _Ytor, make_stack, \
    _mats_at
from pyastroseis.spheroidal_ref import source_jumps, \
    spheroidal_pole
from pyastroseis.toroidal_modes import _pole_coupling
from tests.test_spectral import A, SH, wk

LMAX = 10
LAUD = 8
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
mats = _mats_at(make_stack(lay), w, 50.0, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
M0 = 1.0e18
DY, DG = spheroidal_pole(LMAX)
poleT = _pole_coupling(LMAX)
CSr = {l: M0 * DG[1][l][0] for l in range(1, LMAX + 1)}
CSt = {l: M0 * poleT[1][l][0] for l in range(1, LMAX + 1)}
rF = 1.03 * A
w2c = w * w
ka = w / np.sqrt((lam + 2.0 * mu) / rho)
kb = w / np.sqrt(mu / rho)


def _Yh(l, r):
    return (_Ysolid(l, w, r, m0, ("j",), None)
            + 1j * _Ysolid(l, w, r, m0, ("y",), None))


def _Yth(l, r):
    return (_Ytor(l, w, r, m0, ("j",), None)
            + 1j * _Ytor(l, w, r, m0, ("y",), None))


def psv_free(l, b):
    L = l * (l + 1.0)
    F0v = np.array([0, 0, 0, 1.0 / (L * b * b)], dtype=complex)
    F1v = np.array([0, 0, -1.0 / b ** 3, 3.0 / (L * b ** 3)],
                   dtype=complex)
    J = source_jumps(l, w, b, rho, lam, mu, F0v, F1v)
    M = np.zeros((4, 4), dtype=complex)
    M[:, 0:2] = _Yh(l, b)
    M[:, 2:4] = -_Ysolid(l, w, b, m0, ("j",), None)
    sc = np.max(np.abs(M), axis=0)
    x = np.linalg.solve(M / sc, J) / sc
    return _Yh(l, rF) @ x[0:2]           # (U,V,R,S) at rF


def tor_free(l, b):
    Mx = np.zeros((2, 2), dtype=complex)
    Mx[:, 0] = _Yth(l, b)[:, 0]
    Mx[:, 1] = -_Ytor(l, w, b, m0, ("j",), None)[:, 0]
    J = np.array([1.0 / (mu * b * b), 0.0], dtype=complex)
    sc = np.max(np.abs(Mx), axis=0)
    x = np.linalg.solve(Mx / sc, J) / sc
    return _Yth(l, rF)[:, 0] * x[0]      # (W,T) at rF


# ---- analytic Mzx+Mxz field -------------------------------------
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


NT, NPH = 48, 8
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


def rep(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)


Y1, Bt1, Bp1 = rep(S1["Y"]), rep(S1["Bt"]), rep(S1["Bp"])
Ct1, Cp1 = rep(S1["Ct"]), rep(S1["Cp"])
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
         + np.einsum('ln,n->l', np.conj(Bp1 * eip), up * wq))
    W = (np.einsum('ln,n->l', np.conj(Ct1 * eip), ut * wq)
         + np.einsum('ln,n->l', np.conj(Cp1 * eip), up * wq))
    return U, V / Lp, W


def u_ana(b, eps=1.0):
    """analytic field of M = M0 (zx + xz) at (0,0,b), pointwise."""
    u = np.zeros((len(dirs), 3), dtype=complex)
    ex = np.array([eps, 0.0, 0.0])
    ez = np.array([0.0, 0.0, eps])
    s0 = np.array([0.0, 0.0, b])
    for i in range(len(dirs)):
        x = rF * dirs[i]
        dGx = (Gmat(x - s0 - ex) - Gmat(x - s0 + ex)) / (2.0 * eps)
        dGz = (Gmat(x - s0 - ez) - Gmat(x - s0 + ez)) / (2.0 * eps)
        # u_n = M_pq d/dxi_q G_np ; d/dxi = -grad_x applied above
        u[i] = M0 * (dGx[:, 2] + dGz[:, 0])
    return u


print("m'=+1 free-space: code family vs analytic Mzx+Mxz")
print("b-depth[km] lp   U_code/U_ana     V_code/V_ana"
      "     W_code/W_ana")
for b in (r0 - 50.0e3, r0, r0 + 50.0e3):
    Ua, Va, Wa = coeffs(u_ana(b))
    for lp in range(1, LAUD + 1):
        y4 = psv_free(lp, b) * CSr[lp]
        wt = tor_free(lp, b) * CSt[lp]
        rU = y4[0] / Ua[lp]
        rV = y4[1] / Va[lp]
        rW = wt[0] / Wa[lp]
        print("%8.1f  %2d  %9.6f<%7.2f  %9.6f<%7.2f  %9.6f<%7.2f"
              % ((A - b) / 1e3, lp, abs(rU),
                 np.degrees(np.angle(rU)), abs(rV),
                 np.degrees(np.angle(rV)), abs(rW),
                 np.degrees(np.angle(rW))))
    print()
