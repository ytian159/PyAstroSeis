#!/usr/bin/env python3
"""Probe: is the non-converging fluid-coupled eigenfrequency offset
caused by NEAR-SINGULAR PAIR QUADRATURE (deg-10 Gauss on panels
adjacent to the 1/r^2 kernel singularity)?

Reruns the l=0 radial arbiter with a deg-50 near tier for pairs
closer than 1.8 (and 3.0) source-element sizes, by wrapping the
Geometry constructor that domains.py uses. If the offset collapses,
near-pair quadrature is the culprit and the production fix is a
near tier / subdivided rule.
"""

import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

import pyastroseis.assembly as asm
import pyastroseis.domains as dom
from pyastroseis.assembly import Geometry as _Geometry
from pyastroseis.meshgen import gen_layer
from pyastroseis.quadrature import simplex_rule as _simplex_rule

from diag_radial import A_R, B_R, MAT_F, MAT_S, W0, analytic_roots, \
    peak_near


def subdivided_rule(k):
    """Deg-10 rule mapped onto the 4^k-fold uniform subdivision of the
    unit simplex (weights keep summing to 1/2)."""
    ref, w = _simplex_rule(10)
    tris = [((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))]
    for _ in range(k):
        nxt = []
        for v0, v1, v2 in tris:
            v0, v1, v2 = map(np.asarray, (v0, v1, v2))
            m01, m12, m02 = (v0 + v1) / 2, (v1 + v2) / 2, (v0 + v2) / 2
            nxt += [(v0, m01, m02), (m01, v1, m12),
                    (m02, m12, v2), (m01, m12, m02)]
        tris = nxt
    pts, ws = [], []
    for v0, v1, v2 in tris:
        v0, v1, v2 = map(np.asarray, (v0, v1, v2))
        x = v0[0] + (v1[0] - v0[0]) * ref[0] + (v2[0] - v0[0]) * ref[1]
        y = v0[1] + (v1[1] - v0[1]) * ref[0] + (v2[1] - v0[1]) * ref[1]
        pts.append(np.vstack([x, y]))
        ws.append(w / 4.0 ** k)
    return np.hstack(pts), np.concatenate(ws)


# sentinel degrees 1001/1002 = deg-10 on 4- / 16-fold subdivision
def simplex_rule_ext(degree):
    if degree in (1001, 1002):
        return subdivided_rule(degree - 1000)
    return _simplex_rule(degree)


asm.simplex_rule = simplex_rule_ext

TIER_SETS = {
    "default": None,
    "sub4@1.8": ((1.8, 1001), (4.0, 10), (10.0, 5), (None, 2)),
    "sub16@1.8": ((1.8, 1002), (4.0, 10), (10.0, 5), (None, 2)),
    "sub16@3.0": ((3.0, 1002), (4.0, 10), (10.0, 5), (None, 2)),
}


def use_tiers(tiers):
    if tiers is None:
        dom.Geometry = _Geometry
        return

    def geometry_near(faces, **kw):
        kw.setdefault("tiers", tiers)
        return _Geometry(faces, **kw)
    dom.Geometry = geometry_near


def sphere(r, nmesh, seed):
    rng = np.random.default_rng(seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def main():
    root = analytic_roots(0.02, 0.30)[0]
    print("analytic l=0 root: %.6f Hz" % root)
    for mult, tag in ((1, "x1"), (2, "x2")):
        f_core = sphere(A_R, 12 * mult, 1)
        f_surf = sphere(B_R, 25 * mult, 2)
        for name, tiers in TIER_SETS.items():
            use_tiers(tiers)
            m_lc, _, _ = dom.liquid_core_model(f_surf, f_core, MAT_S,
                                               MAT_F, W0)
            fi, rp, interior = peak_near(m_lc, root, span=0.09, npts=73)
            print("%s %-12s: peak %.6f (%+.3f%%)%s"
                  % (tag, name, fi, 100 * (fi / root - 1),
                     "" if interior else " [EDGE]"), flush=True)
    use_tiers(None)


if __name__ == "__main__":
    main()
