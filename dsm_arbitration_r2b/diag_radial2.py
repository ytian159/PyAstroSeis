#!/usr/bin/env python3
"""Mode-SELECTIVE radial (l=0) eigenfrequency measurement.

The random-vector response scan is unreliable in a dense spectrum
(degenerate l-multiplets outshout the l=0 singlet). Here the system
is driven by a spherically symmetric vector (u = r_hat on every
face, p = 1) and the response is the l=0 projection (mean radial
displacement / mean pressure), so only breathing modes peak.

Measures, against the analytic roots:
  1. free homogeneous solid sphere, first two radial modes
     (control: no fluid anywhere);
  2. fluid-core + shell (LC) first radial mode at x1/x2/x4 plus the
     mixed refinements (core-only x4, surf-only x4).
"""

import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import homogeneous_model, liquid_core_model
from pyastroseis.meshgen import gen_layer

from diag_0s0 import det_sphere
from diag_radial import A_R, B_R, MAT_F, MAT_S, W0, analytic_roots
from scipy.optimize import brentq


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def _sym_vectors(model):
    """b = radial unit on u blocks, 1 on p blocks; projector list of
    (slice, weights) returning the l=0 response amplitude."""
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
    return b, proj


def peak_sym(model, f_ref, span=0.09, npts=73):
    b, proj = _sym_vectors(model)
    grid = np.linspace((1 - span) * f_ref, (1 + span) * f_ref, npts)
    resp = []
    for f in grid:
        x = np.linalg.solve(model.assemble(2 * np.pi * f + 0.0j), b)
        resp.append(sum(abs(np.dot(w, x[sl])) for sl, w in proj))
    resp = np.array(resp)
    i = int(np.argmax(resp))
    fi = grid[i]
    if 0 < i < npts - 1:
        y0, y1, y2 = np.log(resp[i - 1:i + 2])
        den = 2 * y1 - y0 - y2
        if den > 0:
            fi += 0.5 * (y2 - y0) / den * (grid[1] - grid[0])
    return fi, bool(0 < i < npts - 1)


def main():
    # analytic references
    fs = np.linspace(0.02, 0.35, 4000)
    d = [det_sphere(f) for f in fs]
    sph_roots = [brentq(det_sphere, fs[i], fs[i + 1])
                 for i in range(len(fs) - 1) if d[i] * d[i + 1] < 0]
    lc_root = analytic_roots(0.02, 0.30)[0]
    print("analytic: free-sphere radial %s | LC radial %.6f"
          % (" ".join("%.5f" % r for r in sph_roots), lc_root))

    print("[free sphere control]", flush=True)
    for mult in (1, 2, 4):
        f_surf = sphere(B_R, 25 * mult, 2)
        hr = np.sqrt(4 * np.pi * B_R ** 2 / f_surf.n) / B_R
        m, _ = homogeneous_model(f_surf, MAT_S)
        parts = []
        for f_ref in sph_roots[:2]:
            fi, interior = peak_sym(m, f_ref)
            parts.append("%.5f -> %.6f (%+.3f%%)%s"
                         % (f_ref, fi, 100 * (fi / f_ref - 1),
                            "" if interior else " [EDGE]"))
        print("  x%d (h/R %.3f): %s" % (mult, hr, " | ".join(parts)),
              flush=True)

    print("[LC radial + apportionment]", flush=True)
    for mc, ms, tag in ((1, 1, "x1"), (2, 2, "x2"), (4, 4, "x4"),
                        (4, 1, "core-only x4"), (1, 4, "surf-only x4")):
        f_core = sphere(A_R, 12 * mc, 1)
        f_surf = sphere(B_R, 25 * ms, 2)
        m_lc, _, _ = liquid_core_model(f_surf, f_core, MAT_S, MAT_F, W0)
        fi, interior = peak_sym(m_lc, lc_root)
        print("  %-13s: peak %.6f (%+.3f%%)%s"
              % (tag, fi, 100 * (fi / lc_root - 1),
                 "" if interior else " [EDGE]"), flush=True)


if __name__ == "__main__":
    main()
