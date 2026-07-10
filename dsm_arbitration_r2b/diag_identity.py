#!/usr/bin/env python3
"""Discrete null-space identity tests for rung 2c.

At the static limit (w -> 0):
  * SOLID rows: a rigid translation of the whole shell (u = e on
    both boundaries, p = 0, t = 0) is annihilated by the continuous
    operator. The residual of the discrete u-row blocks on rigid
    translations measures the self-block consistency error directly
    (the classical rigid-body identity).
  * FLUID rows: constant pressure with u = 0 gives row sums equal to
    the jump constant c (Gauss solid-angle identity for the static
    double layer). The spread of row sums around their median — and
    the median's deviation from the ideal constant — measure the
    acoustic self-term consistency.

Residuals are normalized by the same rows applied to a random vector
of matching block norms (as in diag_residual.py). Convergence order
across x1/x2/x4 identifies the first-order block.

Also evaluated at the l=0 radial-mode frequency for reference (the
dynamic rigid-body residual is physical inertia, so only the static
limit is a clean identity).
"""

import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import liquid_core_model
from pyastroseis.meshgen import gen_layer

from diag_radial import A_R, B_R, MAT_F, MAT_S, W0, analytic_roots

F_STATIC = 1.0e-5      # quasi-static: (k*b)^2 ~ 1.8e-7, below any
                       # identity residual of interest


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def rigid_residuals(model, surf, core, w):
    A = model.assemble(w)
    n_c, n_s = core.faces.n, surf.faces.n
    sl_p = model.block_slice("p", core)
    sl_uc = model.block_slice("u", core)
    sl_us = model.block_slice("u", surf)

    rng = np.random.default_rng(11)
    out = {}
    # rigid translations e_x, e_y, e_z
    res_u, ref_u = {"u_core": 0.0, "u_surf": 0.0}, {"u_core": 0.0,
                                                    "u_surf": 0.0}
    for d in range(3):
        x = np.zeros(model.size, dtype=complex)
        uc = np.zeros(3 * n_c)
        uc[d * n_c:(d + 1) * n_c] = 1.0
        us = np.zeros(3 * n_s)
        us[d * n_s:(d + 1) * n_s] = 1.0
        x[sl_uc] = uc
        x[sl_us] = us
        xr = np.zeros_like(x)
        v = rng.standard_normal(3 * n_c)
        xr[sl_uc] = v / np.linalg.norm(v) * np.linalg.norm(uc)
        v = rng.standard_normal(3 * n_s)
        xr[sl_us] = v / np.linalg.norm(v) * np.linalg.norm(us)
        r, rr = A @ x, A @ xr
        for name, sl in (("u_core", sl_uc), ("u_surf", sl_us)):
            res_u[name] += np.linalg.norm(r[sl]) ** 2
            ref_u[name] += np.linalg.norm(rr[sl]) ** 2
    for name in res_u:
        out["rigid_" + name] = np.sqrt(res_u[name] / ref_u[name])

    # Gauss: constant p, u = 0. Interior identity: A*1 = c + sum(D)
    # = 1/2 - 1/2 = 0 exactly in the static limit; report the
    # absolute per-row RMS deviation from 0 (A's diagonal is O(1)).
    x = np.zeros(model.size, dtype=complex)
    x[sl_p] = 1.0
    rp = (A @ x)[sl_p]
    freg = model._fluid_of[core][0]
    rp = rp * model._scale[freg]          # undo the row scaling
    out["gauss_median"] = float(np.median(rp.real))
    out["gauss_rms"] = float(np.linalg.norm(rp) / np.sqrt(rp.size))
    return out


def main():
    root = analytic_roots(0.02, 0.30)[0]
    for mult, tag in ((1, "x1"), (2, "x2"), (4, "x4")):
        f_core = sphere(A_R, 12 * mult, 1)
        f_surf = sphere(B_R, 25 * mult, 2)
        model, surf, core = liquid_core_model(f_surf, f_core, MAT_S,
                                              MAT_F, W0)
        st = rigid_residuals(model, surf, core,
                             2 * np.pi * F_STATIC + 0.0j)
        print("%s (core n=%d surf n=%d):" % (tag, f_core.n, f_surf.n))
        print("  static: rigid u_core %.3e  u_surf %.3e | gauss "
              "median %+.2e rms %.3e"
              % (st["rigid_u_core"], st["rigid_u_surf"],
                 st["gauss_median"], st["gauss_rms"]), flush=True)


if __name__ == "__main__":
    main()
