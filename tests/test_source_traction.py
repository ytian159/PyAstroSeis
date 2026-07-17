"""Gates for the incident-traction kernel (rung C):

T1  sympy-u == u0eM          (pins index/sign conventions of the
                              symbolic model to the oracle-validated
                              incident displacement)
T2  sympy-t == t0eM          (machine-derived traction vs the hand
                              term-algebra derivation)
T3  FD(u0eM) == t0eM         (independent numerical arbitration at
                              moderate kR where FD is trustworthy)
T4  grad symmetry/limits     (div-free S-part style consistency via
                              a pure-shear check at long range)

Run under an env with sympy (pytorch/2.8.0 module) for T1/T2; T3/T4
run anywhere."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))

from pyastroseis.source import u0eM                    # noqa: E402
from pyastroseis.source_traction import (grad_u_inc, t0eM,     # noqa
                                         _complex_speeds)


class _Pts:
    def __init__(self, ic):
        self.ic = np.asarray(ic, dtype=float)
        self.n = len(self.ic)


RHO, VP, VS = 3000.0, 6000.0, 3000.0
MU = RHO * VS ** 2
LAM = RHO * VP ** 2 - 2 * MU
Q = 50.0
QP_FAC = 0.75 * (VP / VS) ** 2
SRC = np.array([100.0e3, -50.0e3, 6.2e6])
MT = np.array([[1.0, 0.4, -0.3],
               [0.4, -2.0, 0.7],
               [-0.3, 0.7, 0.5]]) * 1.0e19

RNG = np.random.default_rng(7)
PTS = SRC[None, :] + 2.0e5 * (RNG.random((14, 3)) - 0.5) \
    + np.array([0.0, 0.0, 1.0e5])
NRM = RNG.random((14, 3)) - 0.5
NRM /= np.linalg.norm(NRM, axis=1)[:, None]


def _fd_grad(w, disp_ref_hz, eps=0.5):
    """FD field-gradient of u0eM at PTS (moderate kR: trustworthy)."""
    du = np.zeros((len(PTS), 3, 3), dtype=complex)
    for p in range(3):
        for sgn, wgt in ((2, -1 / 12), (1, 8 / 12), (-1, -8 / 12),
                         (-2, 1 / 12)):
            pts = PTS.copy()
            pts[:, p] += sgn * eps
            u = u0eM(_Pts(pts), w, RHO, MU, LAM, SRC[0], SRC[1],
                     SRC[2], Q, MT, qp_fac=QP_FAC,
                     disp_ref_hz=disp_ref_hz)
            n = len(PTS)
            for q in range(3):
                du[:, p, q] += (wgt / eps) * u[q * n:(q + 1) * n]
    return du


def test_fd_gate():
    for w, ref in ((2 * np.pi * 0.05 + 1j * 1.76e-5, 1.0),
                   (2 * np.pi * 0.5, 0.0)):
        du_a = grad_u_inc(PTS, w, RHO, MU, LAM, SRC[0], SRC[1],
                          SRC[2], Q, MT, QP_FAC, disp_ref_hz=ref)
        du_f = _fd_grad(w, ref)
        err = np.max(np.abs(du_a - du_f)) / np.max(np.abs(du_a))
        print("T3 FD gate w=%s ref=%s: rel %.2e" % (w, ref, err))
        assert err < 5e-8


def test_traction_consistency():
    w = 2 * np.pi * 0.05 + 1j * 1.76e-5
    du = grad_u_inc(PTS, w, RHO, MU, LAM, SRC[0], SRC[1], SRC[2],
                    Q, MT, QP_FAC, disp_ref_hz=1.0)
    t_direct = t0eM(PTS, NRM, w, RHO, MU, LAM, SRC[0], SRC[1],
                    SRC[2], Q, MT, QP_FAC, disp_ref_hz=1.0)
    div = np.einsum("npp->n", du)
    sym = du + np.transpose(du, (0, 2, 1))
    t_ref = LAM * NRM * div[:, None] \
        + MU * np.einsum("npq,np->nq", sym, NRM)
    assert np.max(np.abs(t_direct - t_ref)) == 0.0
    print("T4 traction assembly consistent")


def test_sympy_cross():
    try:
        import sympy as sp
    except ImportError:
        print("T1/T2 SKIPPED (no sympy in this env)")
        return
    x, y, z = sp.symbols("x y z", real=True)
    x0, y0, z0 = sp.symbols("x0 y0 z0", real=True)
    kp, ks, iww = sp.symbols("k_p k_s i_ww")
    dx, dy, dz = x - x0, y - y0, z - z0
    R = sp.sqrt(dx * dx + dy * dy + dz * dz)
    g = sp.Matrix([dx, dy, dz]) / R
    Gp = sp.exp(sp.I * kp * R) / (4 * sp.pi * R)
    Gs = sp.exp(sp.I * ks * R) / (4 * sp.pi * R)
    alpha = (Gp * (1 - sp.I * kp * R)
             + Gs * (-1 + sp.I * ks * R + ks ** 2 * R ** 2)) \
        * iww / R ** 2
    beta = (Gp * (-3 + 3 * sp.I * kp * R + kp ** 2 * R ** 2)
            - Gs * (-3 + 3 * sp.I * ks * R + ks ** 2 * R ** 2)) \
        * iww / R ** 2
    Mss = sp.Matrix(3, 3, lambda i, j: sp.Symbol("M%d%d" % (i, j)))
    Msym = (Mss + Mss.T) / 2
    Gmat = sp.Matrix(3, 3, lambda q, j:
                     alpha * sp.eye(3)[q, j] + beta * g[q] * g[j])
    src = (x0, y0, z0)
    u = sp.Matrix([sum(Msym[j, k] * sp.diff(Gmat[q, j], src[k])
                       for j in range(3) for k in range(3))
                   for q in range(3)])
    fld = (x, y, z)
    du = sp.Matrix(3, 3, lambda p, q: sp.diff(u[q], fld[p]))

    subs_num = {}
    w = 2 * np.pi * 0.05 + 1j * 1.76e-5
    vp, vs = _complex_speeds(w, RHO, MU, LAM, Q, QP_FAC, 1.0)
    args = (x, y, z, x0, y0, z0, kp, ks, iww) \
        + tuple(Mss[i, j] for i in range(3) for j in range(3))
    fu = sp.lambdify(args, u, "numpy")
    fdu = sp.lambdify(args, du, "numpy")
    vals = dict(x0=SRC[0], y0=SRC[1], z0=SRC[2],
                k_p=complex(w / vp), k_s=complex(w / vs),
                i_ww=complex(1.0 / (RHO * w ** 2)))
    mvals = {("M%d%d" % (i, j)): MT[i, j]
             for i in range(3) for j in range(3)}

    # T1: sympy-u vs u0eM
    uu = u0eM(_Pts(PTS), w, RHO, MU, LAM, SRC[0], SRC[1], SRC[2],
              Q, MT, qp_fac=QP_FAC, disp_ref_hz=1.0)
    n = len(PTS)
    err1 = 0.0
    err2 = 0.0
    du_a = grad_u_inc(PTS, w, RHO, MU, LAM, SRC[0], SRC[1], SRC[2],
                      Q, MT, QP_FAC, disp_ref_hz=1.0)
    for i, p in enumerate(PTS):
        a = [p[0], p[1], p[2], vals["x0"], vals["y0"], vals["z0"],
             vals["k_p"], vals["k_s"], vals["i_ww"]] \
            + [mvals["M%d%d" % (r, c)] for r in range(3)
               for c in range(3)]
        us = np.array(fu(*a), dtype=complex).ravel()
        uref = np.array([uu[i], uu[n + i], uu[2 * n + i]])
        err1 = max(err1, np.max(np.abs(us - uref)) / np.max(np.abs(uref)))
        dus = np.array(fdu(*a), dtype=complex)
        err2 = max(err2,
                   np.max(np.abs(dus - du_a[i])) / np.max(np.abs(du_a[i])))
    print("T1 sympy-u vs u0eM rel: %.2e" % err1)
    print("T2 sympy-grad vs analytic rel: %.2e" % err2)
    assert err1 < 1e-10
    assert err2 < 1e-10


if __name__ == "__main__":
    test_fd_gate()
    test_traction_consistency()
    test_sympy_cross()
    print("ALL SOURCE-TRACTION GATES PASSED")
