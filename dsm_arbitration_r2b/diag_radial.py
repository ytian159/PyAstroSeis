#!/usr/bin/env python3
"""l=0 radial-mode arbiter for the fluid-solid coupling.

Uniform fluid core (radius a, speed af, density rf) + uniform solid
shell (a..b, vp/vs/rho), free surface, no gravity. For l=0 the
spheroidal problem is closed-form:

  fluid:  p(r) = j0(kf r),            u_r = p'/(rf w^2)
  solid:  phi(r) = C1 j0(kp r) + C2 y0(kp r),  u_r = phi'
          sig_rr = -lam kp^2 phi + 2 mu phi''
  BCs: u_r continuous and sig_rr = -p at r=a; sig_rr = 0 at r=b.

Roots of the 3x3 determinant = exact radial eigenfrequencies. The
coupled BEM system resonates there; the response-peak offsets vs
resolution measure the fluid-coupling accuracy for the mode family
that lives on BOTH sides of the interface (u_r continuity + normal
stress balance) — the same machinery implicated by the corefluid
spectral clusters.

Run on a compute node (~15 min).
"""

import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import Material, liquid_core_model
from pyastroseis.meshgen import gen_layer

A_R, B_R = 8000.0, 20000.0
VP_S, VS_S, RHO_S = 6000.0, 3000.0, 3000.0
VP_F, RHO_F = 7000.0, 3500.0
LAM_S = RHO_S * VP_S ** 2 - 2 * RHO_S * VS_S ** 2
MU_S = RHO_S * VS_S ** 2
MAT_S = Material.solid(VP_S, VS_S, RHO_S, 1e8, qp_fac=3.0)
MAT_F = Material.acoustic(VP_F, RHO_F, 1e8, qp_fac=1.0)
W0 = 2 * np.pi * 0.3


def _z0(kind, x):
    """(z0, z0', z0'') for spherical bessel j0/y0."""
    if kind == "j":
        z0, z1 = spherical_jn(0, x), spherical_jn(1, x)
    else:
        z0, z1 = spherical_yn(0, x), spherical_yn(1, x)
    return z0, -z1, -z0 + 2.0 * z1 / x


def det_radial(f):
    w = 2 * np.pi * f
    kf, kp = w / VP_F, w / VP_S
    # fluid column: p = j0(kf r)
    pj, pjp, _ = _z0("j", kf * A_R)
    # solid columns: phi = z0(kp r)
    rows = np.zeros((3, 3))
    for c, kind in enumerate(("j", "y")):
        za, zap, zapp = _z0(kind, kp * A_R)
        zb, zbp, zbpp = _z0(kind, kp * B_R)
        rows[0, c + 1] = kp * zap                          # u_r at a
        rows[1, c + 1] = -LAM_S * kp ** 2 * za \
            + 2 * MU_S * kp ** 2 * zapp                    # sig_rr at a
        rows[2, c + 1] = -LAM_S * kp ** 2 * zb \
            + 2 * MU_S * kp ** 2 * zbpp                    # sig_rr at b
    rows[0, 0] = -kf * pjp / (RHO_F * w ** 2)              # -u_r fluid
    rows[1, 0] = pj                                        # sig_rr = -p
    rows[2, 0] = 0.0
    return np.linalg.det(rows)


def analytic_roots(f_lo, f_hi, n=3000):
    fs = np.linspace(f_lo, f_hi, n)
    d = np.array([det_radial(f) for f in fs])
    roots = []
    for i in range(len(fs) - 1):
        if d[i] * d[i + 1] < 0:
            roots.append(brentq(det_radial, fs[i], fs[i + 1]))
    return roots


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def peak_near(model, f_ref, span=0.06, npts=49, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    b = rng.standard_normal(model.size) \
        + 1j * rng.standard_normal(model.size)
    grid = np.linspace((1 - span) * f_ref, (1 + span) * f_ref, npts)
    resp = []
    for f in grid:
        x = np.linalg.solve(model.assemble(2 * np.pi * f + 0.0j), b)
        resp.append(np.linalg.norm(x))
    resp = np.array(resp)
    i = int(np.argmax(resp))
    fi = grid[i]
    if 0 < i < npts - 1:
        y0, y1, y2 = np.log(resp[i - 1:i + 2])
        den = 2 * y1 - y0 - y2
        if den > 0:
            fi += 0.5 * (y2 - y0) / den * (grid[1] - grid[0])
    return fi, resp[i], bool(0 < i < npts - 1)


def main():
    roots = analytic_roots(0.02, 0.30)
    print("analytic l=0 radial modes (a=%g fluid vp %g rho %g | shell "
          "vp %g vs %g rho %g | b=%g):" % (A_R, VP_F, RHO_F, VP_S,
                                           VS_S, RHO_S, B_R))
    for r in roots:
        print("  f = %.6f Hz" % r)

    targets = roots[:3]
    # mixed refinements apportion the offset between the two meshes
    for mc, ms, tag in ((1, 1, "x1"), (2, 2, "x2"), (4, 4, "x4"),
                        (4, 1, "core-only x4"), (1, 4, "surf-only x4"),
                        (8, 8, "x8")):
        f_core = sphere(A_R, 12 * mc, 1)
        f_surf = sphere(B_R, 25 * ms, 2)
        m_lc, _, _ = liquid_core_model(f_surf, f_core, MAT_S, MAT_F, W0)
        hr_c = np.sqrt(4 * np.pi * A_R ** 2 / f_core.n) / A_R
        hr_s = np.sqrt(4 * np.pi * B_R ** 2 / f_surf.n) / B_R
        line = ["%s (core h/R %.3f, surf h/R %.3f):" % (tag, hr_c, hr_s)]
        for f_ref in targets:
            fi, rp, interior = peak_near(m_lc, f_ref, span=0.09,
                                         npts=73)
            line.append("  mode %.4f: peak %.6f (%+.3f%%)%s"
                        % (f_ref, fi, 100 * (fi / f_ref - 1),
                           "" if interior else " [EDGE]"))
        print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
