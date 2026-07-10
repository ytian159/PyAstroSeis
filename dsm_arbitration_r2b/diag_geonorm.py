#!/usr/bin/env python3
"""Test: does replacing the flat-panel normals in the fluid-solid
coupling (Smat projection + traction direction) with exact geometric
(radial) normals restore second-order convergence of the l=0 radial
eigenfrequency?

Monkey-patches model._smat on the liquid-core model; the T/G/A/B
kernels keep their panel geometry (their consistency is order-2
already, per diag_residual.py).
"""

import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import liquid_core_model
from pyastroseis.meshgen import gen_layer

from diag_radial import (A_R, B_R, MAT_F, MAT_S, W0, analytic_roots,
                         peak_near)


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def smat_radial(faces):
    n = faces.n
    rhat = faces.ic / np.linalg.norm(faces.ic, axis=1)[:, None]
    S = np.zeros((n, 3 * n))
    idx = np.arange(n)
    S[idx, idx] = rhat[:, 0]
    S[idx, idx + n] = rhat[:, 1]
    S[idx, idx + 2 * n] = rhat[:, 2]
    return S


def main():
    root = analytic_roots(0.02, 0.30)[0]
    print("analytic l=0 root: %.6f Hz" % root)
    for mult, tag in ((1, "x1"), (2, "x2"), (4, "x4")):
        f_core = sphere(A_R, 12 * mult, 1)
        f_surf = sphere(B_R, 25 * mult, 2)
        line = ["%s (core n=%d surf n=%d):" % (tag, f_core.n, f_surf.n)]
        for label, patch in (("panel", False), ("radial", True)):
            m_lc, surf, core = liquid_core_model(f_surf, f_core, MAT_S,
                                                 MAT_F, W0)
            if patch:
                m_lc._smat[core] = smat_radial(f_core)
            fi, rp, interior = peak_near(m_lc, root, span=0.08, npts=65)
            line.append("  %s normals: peak %.6f (%+.3f%%)%s"
                        % (label, fi, 100 * (fi / root - 1),
                           "" if interior else " [EDGE]"))
        print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
