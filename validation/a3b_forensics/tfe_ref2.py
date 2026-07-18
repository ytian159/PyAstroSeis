#!/usr/bin/env python3
"""High-resolution collocation referee. Basis l<=LB=30, GL nodes,
weighted LSQ. Stage 1: validate on the exact translated sphere at
DEL=5 km (target: pointwise error << du ~ 4e-3 |u|).
Stage 2: FD du from linearized-relief collocation at +-30 m,
project coefficients, compare vs engine and translation-FD."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  spheroidal_unit, _Ysolid)
from pyastroseis.spheroidal_ref import (source_jumps,
                                        spheroidal_pole)
from tests.test_spectral import A, SH, wk

QSH = 50.0
LSRC = 14                 # source-field truncation (matches legs)
LB = 30                   # correction-basis truncation
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
stack = make_stack(lay)
mats = _mats_at(stack, w, QSH, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
src = np.array([0.0, 0.0, r0])
DY, DG = spheroidal_pole(LSRC)
zhat = np.array([0.0, 0.0, 1.0])

sols = {l: np.zeros(4, dtype=complex) for l in range(1, LB + 1)}
for l in range(1, LSRC + 1):
    ents = _entries(mats, 0, r0, 0)
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J = source_jumps(l, w, r0, rho, lam, mu, F0y, F1y)
    Ua, Va, info = spheroidal_unit(l, w, ents, J, full=True)
    Y = _Ysolid(l, w, A, m0, ("j", "y"), None)
    sols[l] = np.linalg.solve(Y, info["surface"]) * 1.0e18 * DY[l]


def y_of(l, r, coef):
    return _Ysolid(l, w, r, m0, ("j", "y"), None) @ coef


def bundle(l, r, coef):
    h = 0.5
    y = y_of(l, r, coef)
    yp = (y_of(l, r + h, coef) - y_of(l, r - h, coef)) / (2 * h)
    U, V, R, Ss = y
    L = l * (l + 1.0)
    div = yp[0] + 2.0 * U / r - L * V / r
    return dict(U=U, V=V, R=R, S=Ss,
                ciso=lam * div + 2.0 * mu * U / r, cV=mu * V / r)


NC = 120
xc, wc = np.polynomial.legendre.leggauss(NC)
thc = np.arccos(xc)
Sc = tfe._shapes(0, LB, xc, np.sqrt(1 - xc * xc))


def tr_disp(d, coefs, exact_sphere, want_disp=False, th=None,
            Sset=None):
    if th is None:
        th, Sset = thc, Sc
    ct, st = np.cos(th), np.sin(th)
    if exact_sphere:
        rr = d * ct + np.sqrt(A * A - d * d * st * st)
        nr = (rr - d * ct) / A
        nt = d * st / A
    else:
        rr = A + d * ct
        nr = np.ones_like(th)
        nt = d * st / rr
    nn = np.sqrt(nr * nr + nt * nt)
    nr, nt = nr / nn, nt / nn
    Tr = np.zeros(len(th), dtype=complex)
    Tt = np.zeros(len(th), dtype=complex)
    Ur = np.zeros(len(th), dtype=complex)
    Ut = np.zeros(len(th), dtype=complex)
    for l in range(1, LB + 1):
        cl = coefs.get(l)
        if cl is None or not np.any(cl):
            continue
        for i in range(len(th)):
            b = bundle(l, rr[i], cl)
            Y = Sset["Y"][l][i]
            Bt = Sset["Bt"][l][i]
            eBtt = Sset["eB"][0][l][i]
            Tr[i] += (b["R"] * Y) * nr[i] + (b["S"] * Bt) * nt[i]
            Tt[i] += (b["S"] * Bt) * nr[i] \
                + (b["ciso"] * Y + b["cV"] * eBtt) * nt[i]
            if want_disp:
                Ur[i] += b["U"] * Y
                Ut[i] += b["V"] * Bt
    return Tr, Tt, Ur, Ut


WROW = np.sqrt(np.concatenate([wc, wc]))


def solve_colloc(d, exact_sphere):
    Tr0, Tt0, _, _ = tr_disp(d, sols, exact_sphere)
    cols, scale = [], []
    for l in range(1, LB + 1):
        for kind in range(2):
            cc = {l: np.zeros(4, dtype=complex)}
            cc[l][kind] = 1.0
            Trb, Ttb, _, _ = tr_disp(d, cc, exact_sphere)
            col = np.concatenate([Trb, Ttb]) * WROW
            sc = np.linalg.norm(col)
            cols.append(col / sc)
            scale.append(sc)
    Amat = np.stack(cols, axis=1)
    bvec = -np.concatenate([Tr0, Tt0]) * WROW
    coef, *_ = np.linalg.lstsq(Amat, bvec, rcond=None)
    resid = np.linalg.norm(Amat @ coef - bvec) \
        / np.linalg.norm(bvec)
    coef = coef / np.array(scale)
    tot = {l: sols[l].copy() for l in range(1, LB + 1)}
    for j, l in enumerate(range(1, LB + 1)):
        tot[l][0] += coef[2 * j]
        tot[l][1] += coef[2 * j + 1]
    return tot, resid


# ---- stage 1: exact sphere validation at 5 km
DEL = 5000.0
tot, resid = solve_colloc(DEL, True)
print("stage1 residual (rel, weighted): %.2e" % resid)
tht = np.linspace(0.3, np.pi - 0.3, 7)
St = tfe._shapes(0, LB, np.cos(tht), np.sin(tht))
_, _, Ur, Ut = tr_disp(DEL, tot, True, want_disp=True, th=tht,
                       Sset=St)
rhat = np.stack([np.sin(tht), 0 * tht, np.cos(tht)], axis=1)
that = np.stack([np.cos(tht), 0 * tht, -np.sin(tht)], axis=1)
u_col = Ur[:, None] * rhat + Ut[:, None] * that
ct = np.cos(tht)
rr = DEL * ct + np.sqrt(A * A - DEL * DEL * (1 - ct * ct))
xs = rr[:, None] * rhat
xm = xs - DEL * zhat
dirs_m = xm / np.linalg.norm(xm, axis=1)[:, None]
Mrr = np.diag([0.0, 0.0, 1.0e18])
kw = dict(Q=QSH, q_sign=-1.0, lmax=LSRC)
u_ref = spsp.spectral_spectra(lay, src - DEL * zhat, [Mrr],
                              np.array([w]), dirs_m, l0=False,
                              **kw)
u_ref = (u_ref["psv"] + u_ref["sh"])[0, :, :, 0]
rel = [np.linalg.norm(u_col[i] - u_ref[i])
       / np.linalg.norm(u_ref[i]) for i in range(len(tht))]
print("stage1 pointwise rel err:",
      " ".join("%.1e" % r for r in rel))

# ---- stage 2: du referee at +-30 m (linearized surface)
delta = 30.0
NE = 96
xe, we = np.polynomial.legendre.leggauss(NE)
the = np.arccos(xe)
Se = tfe._shapes(0, LB, xe, np.sqrt(1 - xe * xe))
du_r = np.zeros(NE, dtype=complex)
du_t = np.zeros(NE, dtype=complex)
for s in (+1.0, -1.0):
    tot, res2 = solve_colloc(s * delta, False)
    _, _, Ur, Ut = tr_disp(s * delta, tot, False, want_disp=True,
                           th=the, Sset=Se)
    du_r += 0.5 * s * Ur
    du_t += 0.5 * s * Ut
print("stage2 leg residual: %.2e" % res2)
w2 = 2.0 * np.pi * we
print("lp   dU_colloc(hi-res)")
out = []
for lp in range(1, 9):
    Lp = lp * (lp + 1.0)
    dU = np.sum(np.conj(Se["Y"][lp]) * du_r * w2)
    dV = np.sum(np.conj(Se["Bt"][lp]) * du_t * w2) / Lp
    out.append([dU, dV])
    print("lp=%d %12.4e%+12.4ej  V %12.4e%+12.4ej"
          % (lp, dU.real, dU.imag, dV.real, dV.imag))
np.save("/tmp/claude-111957/-pscratch-sd-y-ytian159/a12d7ec3-102b-49eb-84c6-6b6d4c1ca8a4/scratchpad/colloc2_res.npy",
        np.array(out))
