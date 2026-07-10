#!/usr/bin/env python3
"""Toroidal-mode arbiter for the corefluid SH drift.

A uniform solid shell (a < r < b, speed vs) with shear-free
boundaries (fluid below, free surface above) has analytic toroidal
eigenfrequencies: W(r) = A j_l(kr) + B y_l(kr), traction
mu*(W' - W/r) = 0 at r = a, b. The coupled BEM system matrix is
singular at those frequencies (u tangential, p = 0), so the dips of
its smallest singular value locate the BEM's effective mode
frequencies.

Compares, against the analytic root:
  * liquid_core_model  (rung-0 path, MATLAB-validated)
  * nested solid/fluid/solid (rung-2b annulus path)
at two mesh resolutions. If both paths show the same offset, the
drift class predates rung 2b; the resolution pair gives its
convergence order.

Run on a compute node (~10 min).
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import (Material, liquid_core_model,
                                 nested_shell_model)
from pyastroseis.meshgen import gen_layer

VS = 3000.0
A_R, B_R = 8000.0, 20000.0
MAT_S = Material.solid(6000.0, VS, 3000.0, 1e8, qp_fac=3.0)
MAT_F = Material.acoustic(7000.0, 3500.0, 1e8, qp_fac=1.0)
MAT_C = Material.solid(8000.0, 4500.0, 4000.0, 1e8, qp_fac=2.3717)
W0 = 2 * np.pi * 0.3


def traction_fn(l, k, r):
    """mu-normalized toroidal traction rows [j-part, y-part]."""
    kr = k * r
    j, jp = spherical_jn(l, kr), spherical_jn(l, kr, derivative=True)
    y, yp = spherical_yn(l, kr), spherical_yn(l, kr, derivative=True)
    return np.array([k * jp - j / r, k * yp - y / r])


def det_free_free(f, l):
    k = 2 * np.pi * f / VS
    ta = traction_fn(l, k, A_R)
    tb = traction_fn(l, k, B_R)
    return ta[0] * tb[1] - ta[1] * tb[0]


def analytic_root(l, f_lo, f_hi):
    fs = np.linspace(f_lo, f_hi, 400)
    d = [det_free_free(f, l) for f in fs]
    for i in range(len(fs) - 1):
        if d[i] * d[i + 1] < 0:
            return brentq(det_free_free, fs[i], fs[i + 1], args=(l,))
    raise RuntimeError("no root in bracket")


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def dips_classified(model, ifaces_surf_core, f_grid, rng_seed=7):
    """Resonance peaks of ||A(f)^-1 b|| for a fixed random b, with
    solution-vector classification: p_frac (pressure-block energy)
    and n_frac (normal-motion energy on the shell boundaries).
    Toroidal: both ~ 0. Near a resonance the solution aligns with
    the near-null mode, so one LU per frequency suffices."""
    rng = np.random.default_rng(rng_seed)
    b = rng.standard_normal(model.size) \
        + 1j * rng.standard_normal(model.size)
    resp, vecs = [], []
    for f in f_grid:
        w = 2 * np.pi * f + 0.0j
        x = np.linalg.solve(model.assemble(w), b)
        resp.append(np.linalg.norm(x))
        vecs.append(x)
    resp = np.array(resp)
    out = []
    for i in range(1, len(f_grid) - 1):
        if resp[i] > resp[i - 1] and resp[i] > resp[i + 1]:
            v = vecs[i]
            p_e = sum(np.linalg.norm(v[model.block_slice("p", ifc)]) ** 2
                      for k, ifc in model.blocks if k == "p")
            n_e, t_e = 0.0, 0.0
            for iface in ifaces_surf_core:
                u = v[model.block_slice("u", iface)]
                n = iface.faces.n
                uv = u.reshape(3, n).T
                nrm = iface.faces.nvec
                un = np.einsum("ij,ij->i", uv, nrm)
                n_e += np.linalg.norm(un) ** 2
                t_e += np.linalg.norm(uv) ** 2
            # parabolic refinement on log(resp)
            y0, y1, y2 = np.log(resp[i - 1:i + 2])
            fi = f_grid[i]
            denom = (2 * y1 - y0 - y2)
            if denom > 0:
                fi += 0.5 * (y2 - y0) / denom * (f_grid[1] - f_grid[0])
            out.append((fi, resp[i], p_e / np.linalg.norm(v) ** 2,
                        n_e / max(t_e, 1e-300)))
    return out, resp


def main():
    l = 2
    # sanity: full-sphere 0T2 must satisfy k b ~ 2.501
    kb = brentq(lambda f: traction_fn(
        l, 2 * np.pi * f / VS, B_R)[0], 0.01, 0.08) \
        * 2 * np.pi / VS * B_R
    print("full-sphere check: 0T2 k*b = %.4f (expect ~2.501)" % kb)
    f_ref = analytic_root(l, 0.01, 0.2)
    print("analytic 0T%d (a=%g b=%g vs=%g): f = %.6f Hz"
          % (l, A_R, B_R, VS, f_ref))
    dumps = {}

    # control: homogeneous sphere, known 0T2 at k*b = 2.501.
    # Three resolutions -> convergence order of the toroidal mode-
    # frequency bias in h/R (calibrates the auto-mesh curvature rule).
    from pyastroseis.domains import homogeneous_model
    f_sph = kb * VS / (2 * np.pi * B_R)
    for nm in (25, 50, 100):
        f_s = sphere(B_R, nm, 2)
        m_h, surf_h = homogeneous_model(f_s, MAT_S)
        grid_h = np.linspace(0.95 * f_sph, 1.12 * f_sph, 103)
        peaks, resp = dips_classified(m_h, (surf_h,), grid_h)
        dumps["resp_homog_n%d" % nm] = resp
        dumps["fgrid_homog_n%d" % nm] = grid_h
        hr = np.sqrt(4 * np.pi * B_R ** 2 / f_s.n) / B_R
        tor = [p for p in peaks if p[3] < 0.10]
        print("control sphere n=%d (h/R=%.3f), analytic 0T2 "
              "f=%.6f:" % (f_s.n, hr, f_sph), flush=True)
        for fi, rp, pf, nf in (tor[:1] or peaks[:1]):
            print("  0T2 peak f=%.6f  offset %+.3f%%  (n_frac %.3f)"
                  % (fi, 100 * (fi / f_sph - 1), nf), flush=True)

    # wide hunt for the LC shell's actual modes (x1 only)
    f_surf1 = sphere(B_R, 25, 2)
    f_core1 = sphere(A_R, 12, 1)
    m_lc1, surf1, core1 = liquid_core_model(f_surf1, f_core1, MAT_S,
                                            MAT_F, W0)
    grid_w = np.linspace(0.70 * f_ref, 1.60 * f_ref, 121)
    peaks, resp = dips_classified(m_lc1, (surf1, core1), grid_w)
    dumps["resp_wide"] = resp
    dumps["fgrid_wide"] = grid_w
    print("wide LC scan [0.70, 1.60]*f_ref:", flush=True)
    for fi, rp, pf, nf in peaks:
        kind = "TOROIDAL" if pf < 0.02 and nf < 0.02 else "spher-ish"
        print("  LC peak f=%.6f (%+.3f%%) resp=%.2e p=%.3f n=%.3f  %s"
              % (fi, 100 * (fi / f_ref - 1), rp, pf, nf, kind),
              flush=True)

    np.savez(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "diag_toroidal_resp.npz"),
        f_ref=f_ref, **dumps)


if __name__ == "__main__":
    main()
