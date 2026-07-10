#!/usr/bin/env python3
"""Control: radial (0S0-type) eigenmode of the HOMOGENEOUS free solid
sphere — no fluid anywhere. If its BEM eigenfrequency shows the same
slow-order offset as the fluid-coupled radial mode, the mechanism is
the solid T operator's spheroidal (normal-motion) accuracy on curved
panels, not the fluid coupling."""

import os
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from pyastroseis.domains import homogeneous_model
from pyastroseis.meshgen import gen_layer

from diag_radial import B_R, LAM_S, MU_S, MAT_S, VP_S, peak_near


def det_sphere(f):
    kp = 2 * np.pi * f / VP_S
    x = kp * B_R
    j0, j1 = spherical_jn(0, x), spherical_jn(1, x)
    j0pp = -j0 + 2 * j1 / x
    return -LAM_S * j0 + 2 * MU_S * j0pp


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def main():
    fs = np.linspace(0.02, 0.35, 4000)
    d = [det_sphere(f) for f in fs]
    roots = [brentq(det_sphere, fs[i], fs[i + 1])
             for i in range(len(fs) - 1) if d[i] * d[i + 1] < 0]
    print("analytic free-sphere radial modes:",
          " ".join("%.5f" % r for r in roots))
    for mult in (1, 2, 4):
        f_surf = sphere(B_R, 25 * mult, 2)
        hr = np.sqrt(4 * np.pi * B_R ** 2 / f_surf.n) / B_R
        m, _ = homogeneous_model(f_surf, MAT_S)
        line = ["x%d (n=%d, h/R %.3f):" % (mult, f_surf.n, hr)]
        for f_ref in roots[:2]:
            fi, rp, interior = peak_near(m, f_ref, span=0.09, npts=73)
            line.append("  mode %.5f: peak %.6f (%+.3f%%)%s"
                        % (f_ref, fi, 100 * (fi / f_ref - 1),
                           "" if interior else " [EDGE]"))
        print("\n".join(line), flush=True)


if __name__ == "__main__":
    main()
