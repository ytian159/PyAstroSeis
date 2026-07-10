#!/usr/bin/env python3
"""Consistency-residual localizer for the fluid-solid coupling bug.

Two tests, three resolutions each:

1. FLUID OPERATOR IDENTITY: for any interior Helmholtz field
   p = j0(kf r), u = grad p/(rho_f w^2), the discrete fluid equation
   A p - rho_f w^2 B (Smat u) must -> 0 under refinement (relative to
   the same rows applied to a random vector). Isolates cal_A_st /
   cal_B_st / Smat / the rho w^2 factor.

2. FULL-MATRIX EIGENMODE RESIDUAL: the exact l=0 radial eigenmode of
   the fluid-core+shell model (analytic root + null amplitudes) is
   sampled at the collocation points and pushed through the
   assembled liquid-core matrix; block-wise residuals (p rows,
   u_core rows, u_surf rows) show which equation is inconsistent.

Run on a compute node (~10 min).
"""

import os
import sys

import numpy as np
from scipy.special import spherical_jn, spherical_yn

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.assembly import Geometry, cal_A_st, cal_B_st, smat_func
from pyastroseis.domains import Material, liquid_core_model
from pyastroseis.meshgen import gen_layer

from diag_radial import (A_R, B_R, LAM_S, MU_S, MAT_F, MAT_S, RHO_F,
                         RHO_S, VP_F, VP_S, W0, analytic_roots)

Q_EL = 1.0e8


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def _z0v(kind, x):
    if kind == "j":
        z0, z1 = spherical_jn(0, x), spherical_jn(1, x)
    else:
        z0, z1 = spherical_yn(0, x), spherical_yn(1, x)
    return z0, -z1, -z0 + 2.0 * z1 / x


def fluid_identity(f_core, f_hz):
    """Residual of A p - rho w^2 B (Smat u) on an exact interior
    Helmholtz field, relative to a random vector of the same norms."""
    w = 2 * np.pi * f_hz + 0.0j
    kf = w.real / VP_F
    geom = Geometry(f_core, self_scheme="polar", quad_mode="adaptive")
    Ab = cal_A_st(f_core, f_core, w, MAT_F.lamda, 0.0, RHO_F, Q_EL,
                  qp_fac=1.0, geom=geom, self_scheme="polar")
    Bb = cal_B_st(f_core, f_core, w, MAT_F.lamda, 0.0, RHO_F, Q_EL,
                  qp_fac=1.0, geom=geom, self_scheme="polar")
    S = smat_func(f_core)

    r = np.linalg.norm(f_core.ic, axis=1)
    rhat = f_core.ic / r[:, None]
    p = spherical_jn(0, kf * r).astype(complex)
    dpdr = kf * (-spherical_jn(1, kf * r))
    u = (dpdr / (RHO_F * w ** 2))[:, None] * rhat
    uvec = np.concatenate([u[:, 0], u[:, 1], u[:, 2]])

    res = Ab @ p - RHO_F * w ** 2 * (Bb @ (S @ uvec))
    rng = np.random.default_rng(3)
    p_r = rng.standard_normal(p.size) * np.linalg.norm(p) / np.sqrt(p.size)
    u_r = rng.standard_normal(uvec.size) * np.linalg.norm(uvec) \
        / np.sqrt(uvec.size)
    res_r = Ab @ p_r - RHO_F * w ** 2 * (Bb @ (S @ u_r))
    return np.linalg.norm(res) / np.linalg.norm(res_r)


def eigen_x(f_root, f_core, f_surf):
    """Null amplitudes (Af, C1, C2) + sampled block vectors."""
    w = 2 * np.pi * f_root
    kf, kp = w / VP_F, w / VP_S
    M = np.zeros((3, 3))
    pj, pjp, _ = _z0v("j", kf * A_R)
    for c, kind in enumerate(("j", "y")):
        za, zap, zapp = _z0v(kind, kp * A_R)
        zb, zbp, zbpp = _z0v(kind, kp * B_R)
        M[0, c + 1] = kp * zap
        M[1, c + 1] = -LAM_S * kp ** 2 * za + 2 * MU_S * kp ** 2 * zapp
        M[2, c + 1] = -LAM_S * kp ** 2 * zb + 2 * MU_S * kp ** 2 * zbpp
    M[0, 0] = -kf * pjp / (RHO_F * w ** 2)
    M[1, 0] = pj
    _, sv, Vh = np.linalg.svd(M)
    Af, C1, C2 = Vh[-1]

    def u_solid(faces):
        r = np.linalg.norm(faces.ic, axis=1)
        rhat = faces.ic / r[:, None]
        up = C1 * kp * (-spherical_jn(1, kp * r)) \
            + C2 * kp * (-spherical_yn(1, kp * r))
        u = up[:, None] * rhat
        return np.concatenate([u[:, 0], u[:, 1], u[:, 2]]).astype(complex)

    rc = np.linalg.norm(f_core.ic, axis=1)
    p = (Af * spherical_jn(0, kf * rc)).astype(complex)
    return sv[-1] / sv[0], p, u_solid(f_core), u_solid(f_surf)


def eigen_residual(f_root, f_core, f_surf):
    model, surf, core = liquid_core_model(f_surf, f_core, MAT_S, MAT_F,
                                          W0)
    w = 2 * np.pi * f_root + 0.0j
    A = model.assemble(w)
    svr, p, u_c, u_s = eigen_x(f_root, f_core, f_surf)
    x = np.zeros(model.size, dtype=complex)
    x[model.block_slice("p", core)] = p
    x[model.block_slice("u", core)] = u_c
    x[model.block_slice("u", surf)] = u_s

    rng = np.random.default_rng(5)
    xr = np.zeros_like(x)
    for kind, ifc in model.blocks:
        sl = model.block_slice(kind, ifc)
        v = rng.standard_normal(sl.stop - sl.start)
        xr[sl] = v * np.linalg.norm(x[sl]) / np.linalg.norm(v)

    res, res_r = A @ x, A @ xr
    out = {}
    for kind, ifc in model.blocks:
        sl = model.block_slice(kind, ifc)
        name = "%s_%s" % (kind, "core" if ifc is core else "surf")
        out[name] = np.linalg.norm(res[sl]) / np.linalg.norm(res_r[sl])
    return svr, out


def main():
    root = analytic_roots(0.02, 0.30)[0]
    print("analytic l=0 root: %.6f Hz" % root)
    for mult, tag in ((1, "x1"), (2, "x2"), (4, "x4")):
        f_core = sphere(A_R, 12 * mult, 1)
        f_surf = sphere(B_R, 25 * mult, 2)
        fid = fluid_identity(f_core, root)
        svr, blocks = eigen_residual(root, f_core, f_surf)
        print("%s (core n=%d surf n=%d): fluid-identity %.3e | "
              "eigen residual: %s  (3x3 null qual %.1e)"
              % (tag, f_core.n, f_surf.n, fid,
                 "  ".join("%s %.3e" % kv for kv in sorted(blocks.items())),
                 svr), flush=True)


if __name__ == "__main__":
    main()
