"""Incident wavefield (source terms) on the boundary.

Ports of u0e_func.m (single force), u0eM_func.m (moment tensor) and
u0p_func_exp.m (pressure source in the fluid core).

qp_fac defaults are the original MATLAB values (2.25 for the
single-force Green function, 2.5 elsewhere — inconsistent in the
original); the drivers in solver.py/liquidcore.py override them with
the physically consistent factor unless qp_mode="legacy".
"""

import numpy as np

from .greens import greens_deri_src, greens_func_p, greens_functionQ

QP_FAC_U0E_LEGACY = 9.0 / 4.0
QP_FAC_LEGACY = 2.5


def u0e(faces, w, rho, mu, lamda, xs, ys, zs, Q, fsrc,
        qp_fac=QP_FAC_U0E_LEGACY):
    """Boundary values of the incident field for a single point force
    fsrc (3,) at (xs,ys,zs). Returns (3N,) ordered [ux; uy; uz]."""
    n = faces.n
    G = greens_functionQ(w, rho, mu, lamda, np.array([xs, ys, zs]),
                         faces.ic[:, 0], faces.ic[:, 1], faces.ic[:, 2], Q,
                         qp_fac=qp_fac)
    fsrc = np.asarray(fsrc, dtype=float)
    u0 = np.empty(3 * n, dtype=complex)
    for i in range(3):
        u0[i * n:(i + 1) * n] = (G[i, 0] * fsrc[0] + G[i, 1] * fsrc[1]
                                 + G[i, 2] * fsrc[2])
    return u0


def u0eM(faces, w, rho, mu, lamda, xs, ys, zs, Q, M, qp_fac=QP_FAC_LEGACY,
         disp_ref_hz=0.0):
    """Boundary values of the incident field for a moment tensor M (3,3)
    at (xs,ys,zs). Port of u0eM_func.m. Returns (3N,).
    disp_ref_hz: see assembly.wave_speeds (must match the kernels)."""
    vp0 = np.sqrt((lamda + 2 * mu) / rho)
    vs0 = np.sqrt(mu / rho)
    Qp = qp_fac * Q
    if disp_ref_hz:
        lf = np.log(abs(np.real(w)) / (2 * np.pi * disp_ref_hz))
        vp0 = vp0 * (1 + lf / (np.pi * Qp))
        vs0 = vs0 * (1 + lf / (np.pi * Q))
    vp = vp0 / (1 + 1j * 0.5 / Qp)
    vs = vs0 / (1 + 1j * 0.5 / Q)
    gi1, gi2, gi3 = greens_deri_src(vp, vs, rho, w,
                                    faces.ic[:, 0], faces.ic[:, 1],
                                    faces.ic[:, 2], xs, ys, zs)
    M = np.asarray(M, dtype=float)
    n = faces.n
    u0 = np.empty(3 * n, dtype=complex)
    u0[0:n] = np.einsum("ijn,ij->n", gi1, M)
    u0[n:2 * n] = np.einsum("ijn,ij->n", gi2, M)
    u0[2 * n:3 * n] = np.einsum("ijn,ij->n", gi3, M)
    return u0


def u0p_exp(faces, w, vp0, xs, ys, zs, Q, qp_fac=QP_FAC_LEGACY):
    """Incident pressure on the core surface for an explosion source in
    the fluid (u0p_func_exp.m). Returns (N,)."""
    Qp = qp_fac * Q
    vp = vp0 / (1 + 1j * 0.5 / Qp)
    return greens_func_p(vp, w, faces.ic[:, 0], faces.ic[:, 1],
                         faces.ic[:, 2], xs, ys, zs)
