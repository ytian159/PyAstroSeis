"""Analytic TRACTION of the full-space moment-tensor incident field
(rung C, scattered-field source formulation): t_inc(y) = sigma_inc(y)
. n(y) for u_inc = M : grad_src G, with the same complex-velocity /
causal-Q conventions as source.u0eM.

Derivation (hand, exact term-algebra; cross-validated against an
independent sympy derivation AND finite differences of u0eM in
tests/test_source_traction.py): write the displacement Green tensor
as G_qj = alpha(R) delta_qj + beta(R) g_q g_j with g = (y - x0)/R
(alpha/beta read off source.bmat_func). With P = M g, s = g.M.g,
tau = tr M, f1 = alpha' + beta/R, f2 = beta' s + (beta/R)(tau - 2s):

    u_q(y)   = -[ f1 P_q + f2 g_q ]
    d_p u_q  = -[ f1' g_p P_q + f1 (M_qp - g_p P_q)/R
                  + f2 (delta_qp - g_q g_p)/R
                  + g_q ( beta'' s g_p
                          + (beta'/R - beta/R^2)(tau - 2s) g_p
                          + (2 beta'/R - 4 beta/R^2)(P_p - s g_p) ) ]
    t_l = lamda n_l div(u) + mu n_p (d_p u_l + d_l u_p)

(the minus sign: u_q = M_jk dG_qj/dx0_k = -M_jk d_k^{(y)} G_qj).
The radial functions and their derivatives are exact via a small
term algebra: each is e^{ik R} sum_t c_t (ik)^{e_t} R^{p_t} per wave
speed, with the mechanical derivative rule."""

import numpy as np

PI = np.pi


class _PolyExp:
    """F(R) = e^{i k R} * sum_t c_t (ik)^{e_t} R^{p_t} for a fixed
    wavenumber k supplied at eval time; exact d/dR."""

    def __init__(self, terms):
        self.terms = dict(terms)              # {(e, p): c}

    def deriv(self):
        out = {}
        for (e, p), c in self.terms.items():
            out[(e + 1, p)] = out.get((e + 1, p), 0.0) + c
            if p != 0:
                out[(e, p - 1)] = out.get((e, p - 1), 0.0) + c * p
        return _PolyExp(out)

    def eval(self, ik, R):
        out = 0.0
        for (e, p), c in self.terms.items():
            out = out + c * ik ** e * R ** p
        return np.exp(ik * R) * out


def _alpha_beta():
    """alpha, beta as (P-group, S-group) _PolyExp pairs, excluding the
    common 1/(rho w^2) factor (applied at eval).  From bmat_func with
    Gp = e^{ikp R}/(4 pi R):
      alpha = R^-2 [Gp (1 - ikp R) + Gs (-1 + iks R + ks^2 R^2)]
      beta  = R^-2 [Gp (-3 + 3 ikp R + kp^2 R^2)
                    - Gs (-3 + 3 iks R + ks^2 R^2)]
    and k^2 R^2 = -(ik R)^2 = -(ik)^2 R^2."""
    c = 1.0 / (4.0 * PI)
    aP = _PolyExp({(0, -3): c, (1, -2): -c})
    aS = _PolyExp({(0, -3): -c, (1, -2): c, (2, -1): -c})
    bP = _PolyExp({(0, -3): -3 * c, (1, -2): 3 * c, (2, -1): -c})
    bS = _PolyExp({(0, -3): 3 * c, (1, -2): -3 * c, (2, -1): c})
    return aP, aS, bP, bS


_AP, _AS, _BP, _BS = _alpha_beta()
_AP1, _AS1, _BP1, _BS1 = (t.deriv() for t in (_AP, _AS, _BP, _BS))
_AP2, _AS2, _BP2, _BS2 = (t.deriv() for t in (_AP1, _AS1, _BP1, _BS1))


def _complex_speeds(w, rho, mu, lamda, Q, qp_fac, disp_ref_hz):
    """Complex vp, vs exactly as source.u0eM."""
    vp0 = np.sqrt((lamda + 2 * mu) / rho)
    vs0 = np.sqrt(mu / rho)
    Qp = qp_fac * Q
    if disp_ref_hz:
        lf = np.log(abs(np.real(w)) / (2 * np.pi * disp_ref_hz))
        vp0 = vp0 * (1 + lf / (np.pi * Qp))
        vs0 = vs0 * (1 + lf / (np.pi * Q))
    vp = vp0 / (1 + 1j * 0.5 / Qp)
    vs = vs0 / (1 + 1j * 0.5 / Q)
    return vp, vs


def grad_u_inc(pts, w, rho, mu, lamda, xs, ys, zs, Q, M,
               qp_fac, disp_ref_hz=0.0):
    """(N, 3, 3) FIELD gradient d_p u_q (index order [n, p, q]) of the
    full-space MT incident field at pts (N, 3)."""
    vp, vs = _complex_speeds(w, rho, mu, lamda, Q, qp_fac, disp_ref_hz)
    ikp = 1j * w / vp
    iks = 1j * w / vs
    iww = 1.0 / (rho * w ** 2)

    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    d = pts - np.array([xs, ys, zs], dtype=float)[None, :]
    R = np.linalg.norm(d, axis=1)
    g = d / R[:, None]

    a1 = (_AP1.eval(ikp, R) + _AS1.eval(iks, R)) * iww
    a2 = (_AP2.eval(ikp, R) + _AS2.eval(iks, R)) * iww
    b = (_BP.eval(ikp, R) + _BS.eval(iks, R)) * iww
    b1 = (_BP1.eval(ikp, R) + _BS1.eval(iks, R)) * iww
    b2 = (_BP2.eval(ikp, R) + _BS2.eval(iks, R)) * iww

    M = np.asarray(M, dtype=float)
    P = g @ M.T                                # (Mg)_q, M symmetric
    s = np.einsum("nq,nq->n", P, g)
    tau = np.trace(M)

    f1 = a1 + b / R
    f2 = b1 * s + (b / R) * (tau - 2.0 * s)
    f1p = a2 + b1 / R - b / R ** 2

    brk = ((b2 * s + (b1 / R - b / R ** 2) * (tau - 2.0 * s))[:, None]
           * g
           + (2.0 * b1 / R - 4.0 * b / R ** 2)[:, None]
           * (P - s[:, None] * g))

    eye = np.eye(3)
    du = np.zeros((len(pts), 3, 3), dtype=complex)
    for p in range(3):
        for q in range(3):
            du[:, p, q] = -(
                f1p * g[:, p] * P[:, q]
                + f1 * (M[q, p] - g[:, p] * P[:, q]) / R
                + f2 * (eye[q, p] - g[:, q] * g[:, p]) / R
                + g[:, q] * brk[:, p])
    return du


def t0eM(pts, normals, w, rho, mu, lamda, xs, ys, zs, Q, M,
         qp_fac, disp_ref_hz=0.0):
    """Incident traction (N, 3) at pts with unit normals (N, 3):
    t = lamda n div(u) + mu (grad u + grad u^T) . n. Conventions
    (complex speeds, Q, dispersion) identical to source.u0eM."""
    du = grad_u_inc(pts, w, rho, mu, lamda, xs, ys, zs, Q, M,
                    qp_fac, disp_ref_hz)
    div = np.einsum("npp->n", du)
    sym = du + np.transpose(du, (0, 2, 1))
    normals = np.atleast_2d(np.asarray(normals, dtype=float))
    return (lamda * normals * div[:, None]
            + mu * np.einsum("npq,np->nq", sym, normals))
