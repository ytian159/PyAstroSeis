"""Shared machinery for the L=1 translation paradox audits
(minimal config: homog ball, mrr on-axis, m=0, Y10 top)."""
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe                    # noqa
from pyastroseis.spectral import (_entries, _mats_at,          # noqa
                                  make_stack, spheroidal_unit,
                                  _Ysolid)
from pyastroseis.spheroidal_ref import (source_jumps,          # noqa
                                        spheroidal_pole)
from tests.test_spectral import A, SH, wk                      # noqa

import os
QSH = 50.0
LMAX = int(os.environ.get("PARADOX_LMAX", "8"))
LAUD = LMAX - 2
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
stack = make_stack(lay)
mats = _mats_at(stack, w, QSH, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
delta = 30.0
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)
DY, DG = spheroidal_pole(LMAX)
CS = {l: 1.0e18 * DY[l] for l in range(1, LMAX + 1)}


def Jvec(l, b):
    F0y = np.array([0, 0, 1.0 / b ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / b ** 3, 0], dtype=complex)
    return source_jumps(l, w, b, rho, lam, mu, F0y, F1y)


def solve_l(l, b):
    """Ball solve with the m=0 (y-tag) source jump at radius b."""
    ents = _entries(mats, 0, b, 0)
    J = Jvec(l, b)
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ysolid(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ysolid(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    bv = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        Am[row:row + 4, chi] = Yb[i + 1]
        Am[row:row + 4, clo] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row:row + 4] = J
        row += 4
    ctop = slice(ofs[-2], ofs[-1])
    Am[row, ctop] = Yt[-1][2]
    Am[row + 1, ctop] = Yt[-1][3]
    sc = np.max(np.abs(Am), axis=0)
    sc[sc == 0] = 1.0
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs, b=b, kind="ball")


def _Yh(l, r):
    """outgoing (h1 = j + i y) 2-column (P,S) matrix."""
    return (_Ysolid(l, w, r, m0, ("j",), None)
            + 1j * _Ysolid(l, w, r, m0, ("y",), None))


def solve_l_free(l, b):
    """FREE-SPACE solve: regular j inside r<b, outgoing h outside,
    4x4 jump condition [y] = J at r=b."""
    J = Jvec(l, b)
    M = np.zeros((4, 4), dtype=complex)
    M[:, 0:2] = _Yh(l, b)
    M[:, 2:4] = -_Ysolid(l, w, b, m0, ("j",), None)
    sc = np.max(np.abs(M), axis=0)
    x = np.linalg.solve(M / sc, J) / sc
    return dict(cout=x[0:2], cin=x[2:4], b=b, kind="free", l=l)


def y_of(l, sol, r):
    """(U,V,R,S) at radius r for either solve type."""
    if sol["kind"] == "free":
        if r >= sol["b"]:
            return _Yh(l, r) @ sol["cout"]
        return _Ysolid(l, w, r, m0, ("j",), None) @ sol["cin"]
    i = 1 if r >= sol["b"] else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return _Ysolid(l, w, r, m0, kinds, None) \
        @ sol["x"][sol["ofs"][i]:sol["ofs"][i + 1]]


# projection grid
NT = 96
xg, wg = np.polynomial.legendre.leggauss(NT)
ct, st = xg, np.sqrt(1.0 - xg * xg)
SH_ = tfe._shapes(0, LMAX + 2, ct, st)
Yg, dYg = SH_["Y"], SH_["Bt"]
d2Yg = SH_["eB"][0] / 2.0
w2 = 2.0 * np.pi * wg
llv = np.arange(LMAX + 3)
Lpv = llv * (llv + 1.0)
Lpv[0] = 1.0


def proj(fr, ft):
    U = np.einsum('ln,n->l', np.conj(Yg), fr * w2)
    V = np.einsum('ln,n->l', np.conj(dYg), ft * w2) / Lpv
    return U, V


def u_line(r, sols, hfd=0.5):
    """theta-line fields and derivatives of a CS-weighted solve
    family at radius r."""
    ur = np.zeros(NT, dtype=complex)
    ut = np.zeros(NT, dtype=complex)
    urp = np.zeros(NT, dtype=complex)
    utp = np.zeros(NT, dtype=complex)
    dtur = np.zeros(NT, dtype=complex)
    dtut = np.zeros(NT, dtype=complex)
    for l in range(1, LMAX + 1):
        yv = y_of(l, sols[l], r) * CS[l]
        yp = (y_of(l, sols[l], r + hfd)
              - y_of(l, sols[l], r - hfd)) / (2.0 * hfd) * CS[l]
        ur += yv[0] * Yg[l]
        ut += yv[1] * dYg[l]
        urp += yp[0] * Yg[l]
        utp += yp[1] * dYg[l]
        dtur += yv[0] * dYg[l]
        dtut += yv[1] * d2Yg[l]
    return ur, ut, urp, utp, dtur, dtut


def zgrad_proj(r, sols):
    """(U, V) projections of (zhat.grad)u at radius r."""
    ur, ut, urp, utp, dtur, dtut = u_line(r, sols)
    gr = ct * urp - (st / r) * (dtur - ut)
    gt = ct * utp - (st / r) * (ur + dtut)
    return proj(gr, gt)
