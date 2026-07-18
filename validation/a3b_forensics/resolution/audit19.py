#!/usr/bin/env python3
"""Audit 19: m=1 FREE-SPACE covariance at delta=3000 (noise-free):
LHS = per-l diagonal source-shift FD of the code families
RHS = pointwise translation of the FIXED b=r0 field (exact)
If LHS==RHS the m=1 legs are covariant -> engine misses a real
m=+-1 term. Controls: same at m=0."""
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
mats = _mats_at(make_stack([dict(r_top=A, **SH)]), w, 50.0, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
M0 = 1.0e18
DY, DG = spheroidal_pole(LMAX)
poleT = _pole_coupling(LMAX)
CSr = {l: M0 * DG[1][l][0] for l in range(1, LMAX + 1)}
CSt = {l: M0 * poleT[1][l][0] for l in range(1, LMAX + 1)}
CSy = {l: M0 * DY[l] for l in range(1, LMAX + 1)}
rF = 1.03 * A
DL = 3000.0

def _Yh(l, r):
    return (_Ysolid(l, w, r, m0, ("j",), None)
            + 1j * _Ysolid(l, w, r, m0, ("y",), None))

def _Yth(l, r):
    return (_Ytor(l, w, r, m0, ("j",), None)
            + 1j * _Ytor(l, w, r, m0, ("y",), None))

def psv_free(l, b, tag):
    L = l * (l + 1.0)
    if tag == "rh":
        F0 = np.array([0, 0, 0, 1.0 / (L * b * b)], dtype=complex)
        F1 = np.array([0, 0, -1.0 / b ** 3, 3.0 / (L * b ** 3)],
                      dtype=complex)
    else:
        F0 = np.array([0, 0, 1.0 / b ** 2, 0], dtype=complex)
        F1 = np.array([0, 0, 2.0 / b ** 3, 0], dtype=complex)
    J = source_jumps(l, w, b, rho, lam, mu, F0, F1)
    M = np.zeros((4, 4), dtype=complex)
    M[:, 0:2] = _Yh(l, b)
    M[:, 2:4] = -_Ysolid(l, w, b, m0, ("j",), None)
    sc = np.max(np.abs(M), axis=0)
    x = np.linalg.solve(M / sc, J) / sc
    return x

def tor_free(l, b):
    Mx = np.zeros((2, 2), dtype=complex)
    Mx[:, 0] = _Yth(l, b)[:, 0]
    Mx[:, 1] = -_Ytor(l, w, b, m0, ("j",), None)[:, 0]
    J = np.array([1.0 / (mu * b * b), 0.0], dtype=complex)
    sc = np.max(np.abs(Mx), axis=0)
    return np.linalg.solve(Mx / sc, J) / sc

MM = 1            # azimuthal order under test
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
S1 = tfe._shapes(MM, LMAX + 1, np.cos(tg), np.sin(tg))
def rep(a):
    return np.repeat(a, NPH, axis=1).reshape(a.shape[0], -1)
Y1, Bt1, Bp1 = rep(S1["Y"]), rep(S1["Bt"]), rep(S1["Bp"])
Ct1, Cp1 = rep(S1["Ct"]), rep(S1["Cp"])
eip = np.exp(1j * MM * phv)
ll = np.arange(LMAX + 2)
Lp = ll * (ll + 1.0); Lp[0] = 1.0

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

def field_at(pts, b):
    """pointwise m=1 code free-space field at arbitrary points."""
    ps = {l: psv_free(l, b, "rh") for l in range(1, LMAX + 1)}
    ts = {l: tor_free(l, b) for l in range(1, LMAX + 1)}
    u = np.zeros((len(pts), 3), dtype=complex)
    for i, x in enumerate(pts):
        r = np.linalg.norm(x)
        cth = np.clip(x[2] / r, -1, 1)
        sth = np.hypot(x[0], x[1]) / r
        ph_ = np.arctan2(x[1], x[0])
        Sh = tfe._shapes(MM, LMAX + 1, np.array([cth]),
                         np.array([sth]))
        e1 = np.exp(1j * MM * ph_)
        ur = ut = up = 0j
        for l in range(1, LMAX + 1):
            if r >= b:
                y4 = (_Yh(l, r) @ ps[l][0:2]) * CSr[l]
                wt = (_Yth(l, r)[:, 0] * ts[l][0]) * CSt[l]
            else:
                y4 = (_Ysolid(l, w, r, m0, ("j",), None)
                      @ ps[l][2:4]) * CSr[l]
                wt = (_Ytor(l, w, r, m0, ("j",), None)[:, 0]
                      * ts[l][1]) * CSt[l]
            ur += y4[0] * Sh["Y"][l][0] * e1
            ut += (y4[1] * Sh["Bt"][l][0] + wt[0]
                   * Sh["Ct"][l][0]) * e1
            up += (y4[1] * Sh["Bp"][l][0] + wt[0]
                   * Sh["Cp"][l][0]) * e1
        rh_ = np.array([sth * np.cos(ph_), sth * np.sin(ph_), cth])
        th_ = np.array([cth * np.cos(ph_), cth * np.sin(ph_),
                        -sth])
        ph2 = np.array([-np.sin(ph_), np.cos(ph_), 0.0])
        u[i] = ur * rh_ + ut * th_ + up * ph2
    return u

zhat = np.array([0.0, 0.0, 1.0])
pts = rF * dirs
# RHS: pointwise translation of the fixed b=r0 field
up_ = field_at(pts + DL * zhat, r0)
um_ = field_at(pts - DL * zhat, r0)
Urhs, Vrhs, Wrhs = coeffs(0.5 * (up_ - um_))
# LHS: source-shift FD, per-l diagonal
Ulhs = np.zeros(LMAX + 2, dtype=complex)
Vlhs = np.zeros(LMAX + 2, dtype=complex)
Wlhs = np.zeros(LMAX + 2, dtype=complex)
for l in range(1, LMAX + 1):
    pm = psv_free(l, r0 - DL, "rh")
    pp = psv_free(l, r0 + DL, "rh")
    tm = tor_free(l, r0 - DL)
    tp = tor_free(l, r0 + DL)
    ym = (_Yh(l, rF) @ pm[0:2]) * CSr[l]
    yp = (_Yh(l, rF) @ pp[0:2]) * CSr[l]
    Ulhs[l] = 0.5 * (ym[0] - yp[0])
    Vlhs[l] = 0.5 * (ym[1] - yp[1])
    Wlhs[l] = 0.5 * (_Yth(l, rF)[0, 0] * (tm[0] - tp[0])
                     * CSt[l])
print("m=1 free-space covariance, delta=%.0f m" % DL)
print("lp   lhs/rhs U        lhs/rhs V        lhs/rhs W")
for lp in range(1, LAUD + 1):
    rU = Ulhs[lp] / Urhs[lp]
    rV = Vlhs[lp] / Vrhs[lp]
    rW = Wlhs[lp] / Wrhs[lp]
    print("%2d  %8.5f<%6.1f  %8.5f<%6.1f  %8.5f<%6.1f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV)), abs(rW),
             np.degrees(np.angle(rW))))
