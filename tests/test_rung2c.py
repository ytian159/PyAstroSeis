#!/usr/bin/env python3
"""Rung-2c standing gate: mode-selective l=0 radial eigenfrequency
arbiter (docs/multilayer_roadmap.md).

The fluid-core + solid-shell testbed has an exact analytic l=0
radial eigenfrequency (spherical Bessel 3x3 determinant, validated
against the free-sphere limit). The coupled BEM system is driven
with a spherically symmetric vector (u = r_hat, p = 1) and the l=0
projected response peak located near the analytic root.

Gates:
  1. the offset at x2 (core h/R 0.267, surf 0.181) is below 1.4%
     (measured +0.92%);
  2. the offset CONVERGES from x1 to x2 with order >= 1.4 in h
     (measured ~2.0) — this is the gate that catches any regression
     to first-order behavior in the fluid-solid coupling.

Protocol lesson encoded here: resonance scans must be mode-
selective; a random drive in a dense spectrum tracks the wrong
modes (see roadmap rung 2c).

Run on a compute node (~4 min).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scipy.optimize import brentq                              # noqa: E402
from scipy.special import spherical_jn, spherical_yn           # noqa: E402

from pyastroseis.domains import Material, liquid_core_model    # noqa: E402
from pyastroseis.meshgen import gen_layer                      # noqa: E402

FAILED = []

A_R, B_R = 8000.0, 20000.0
VP_S, VS_S, RHO_S = 6000.0, 3000.0, 3000.0
VP_F, RHO_F = 7000.0, 3500.0
LAM_S = RHO_S * VP_S ** 2 - 2 * RHO_S * VS_S ** 2
MU_S = RHO_S * VS_S ** 2
MAT_S = Material.solid(VP_S, VS_S, RHO_S, 1e8, qp_fac=3.0)
MAT_F = Material.acoustic(VP_F, RHO_F, 1e8, qp_fac=1.0)
W0 = 2 * np.pi * 0.3

OFFSET_X2 = 1.4e-2      # measured 0.92e-2
ORDER_MIN = 1.4         # measured ~2.0


def check(label, ok, detail=""):
    print(f"    {label}: {'OK' if ok else 'FAIL'}{detail}", flush=True)
    if not ok:
        FAILED.append(label)


def _z0(kind, x):
    if kind == "j":
        z0, z1 = spherical_jn(0, x), spherical_jn(1, x)
    else:
        z0, z1 = spherical_yn(0, x), spherical_yn(1, x)
    return z0, -z1, -z0 + 2.0 * z1 / x


def det_radial(f):
    w = 2 * np.pi * f
    kf, kp = w / VP_F, w / VP_S
    pj, pjp, _ = _z0("j", kf * A_R)
    rows = np.zeros((3, 3))
    for c, kind in enumerate(("j", "y")):
        za, zap, zapp = _z0(kind, kp * A_R)
        zb, zbp, zbpp = _z0(kind, kp * B_R)
        rows[0, c + 1] = kp * zap
        rows[1, c + 1] = -LAM_S * kp ** 2 * za + 2 * MU_S * kp ** 2 * zapp
        rows[2, c + 1] = -LAM_S * kp ** 2 * zb + 2 * MU_S * kp ** 2 * zbpp
    rows[0, 0] = -kf * pjp / (RHO_F * w ** 2)
    rows[1, 0] = pj
    return np.linalg.det(rows)


def analytic_root():
    fs = np.linspace(0.02, 0.30, 3000)
    d = [det_radial(f) for f in fs]
    for i in range(len(fs) - 1):
        if d[i] * d[i + 1] < 0:
            return brentq(det_radial, fs[i], fs[i + 1])
    raise RuntimeError("no analytic root found")


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def peak_sym(model, f_ref, span=0.09, npts=73):
    b = np.zeros(model.size, dtype=complex)
    proj = []
    for kind, ifc in model.blocks:
        sl = model.block_slice(kind, ifc)
        n = ifc.faces.n
        if kind == "u":
            r = np.linalg.norm(ifc.faces.ic, axis=1)
            rhat = ifc.faces.ic / r[:, None]
            b[sl] = np.concatenate([rhat[:, 0], rhat[:, 1], rhat[:, 2]])
            proj.append((sl, b[sl].real.copy() / n))
        elif kind == "p":
            b[sl] = 1.0
            proj.append((sl, np.ones(n) / n))
    grid = np.linspace((1 - span) * f_ref, (1 + span) * f_ref, npts)
    resp = []
    for f in grid:
        x = np.linalg.solve(model.assemble(2 * np.pi * f + 0.0j), b)
        resp.append(sum(abs(np.dot(w_, x[sl])) for sl, w_ in proj))
    resp = np.array(resp)
    i = int(np.argmax(resp))
    fi = grid[i]
    if 0 < i < npts - 1:
        y0, y1, y2 = np.log(resp[i - 1:i + 2])
        den = 2 * y1 - y0 - y2
        if den > 0:
            fi += 0.5 * (y2 - y0) / den * (grid[1] - grid[0])
    return fi


def main():
    f_ref = analytic_root()
    print(f"[l=0 radial arbiter] analytic root {f_ref:.6f} Hz",
          flush=True)
    offs = []
    for mult in (1, 2):
        t0 = time.time()
        f_core = sphere(A_R, 12 * mult, 1)
        f_surf = sphere(B_R, 25 * mult, 2)
        m_lc, _, _ = liquid_core_model(f_surf, f_core, MAT_S, MAT_F, W0)
        fi = peak_sym(m_lc, f_ref)
        off = abs(fi / f_ref - 1)
        offs.append(off)
        print(f"    x{mult}: peak {fi:.6f} ({100 * off:+.3f}%) "
              f"({time.time() - t0:.1f} s)", flush=True)
    check("x2 offset below gate", offs[1] < OFFSET_X2,
          f"  ({100 * offs[1]:.3f}% < {100 * OFFSET_X2:g}%)")
    order = np.log(offs[0] / offs[1]) / np.log(np.sqrt(2.0))
    check("convergence order >= %.1f" % ORDER_MIN, order >= ORDER_MIN,
          f"  (order {order:.2f})")

    if FAILED:
        print(f"rung-2c gate FAILED: {FAILED}")
        sys.exit(1)
    print("rung-2c gate PASSED")


if __name__ == "__main__":
    main()
