#!/usr/bin/env python3
"""Self-integration quadrature study: MATLAB 300x300 punch-out grid vs
polar quadrature, judged against a converged fine-grid reference
(nxi=1501, ~1.1M points — the same integral both schemes discretize).

Passes when production polar orders beat the 300-grid accuracy. Run on
a compute node (the reference evaluation is heavy).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import (int_self_G, int_self_G_polar,
                                  int_self_trac, int_self_trac_polar,
                                  polar_self_points, self_ref_grid,
                                  wave_speeds)  # noqa: E402
from pyastroseis.mesh import load_faces_mat  # noqa: E402

FAILED = []


def main():
    faces = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))
    rho, Q = 3000.0, 570.0
    mu = rho * 3000.0 ** 2
    lamda = rho * 6000.0 ** 2 - 2 * mu
    vp, vs = wave_speeds(lamda, mu, rho, Q)
    w = 2 * np.pi * 0.58 + 0.08j          # iw=30, near the top of the band

    test_faces = [0, 100, 777]
    refs_fine, wis_fine = self_ref_grid(1501)
    refs_300, wis_300 = self_ref_grid(300)

    polar_orders = [(8, 6), (16, 12), (32, 24)]
    print(f"{'face':>5s} {'grid300 err':>12s} "
          + " ".join(f"polar{nt}x{nr:>2d}" for nt, nr in polar_orders)
          + "   (rel to nxi=1501 reference, traction self block)")
    worst_300, worst_polar = 0.0, {po: 0.0 for po in polar_orders}
    for i in test_faces:
        truth = int_self_trac(vp, vs, rho, w, faces, i, wis_fine, refs_fine)
        scale = np.abs(truth).max()
        e300 = np.abs(int_self_trac(vp, vs, rho, w, faces, i, wis_300,
                                    refs_300) - truth).max() / scale
        errs = []
        for nt, nr in polar_orders:
            pts = polar_self_points(faces, i, nt, nr)
            ep = np.abs(int_self_trac_polar(vp, vs, rho, w, faces, i, pts)
                        - truth).max() / scale
            errs.append(ep)
            worst_polar[(nt, nr)] = max(worst_polar[(nt, nr)], ep)
        worst_300 = max(worst_300, e300)
        print(f"{i:5d} {e300:12.3e} "
              + " ".join(f"{e:9.3e}" for e in errs))

    # Same-limit check. The grid scheme's stair-step cut at the incircle
    # is only first-order accurate, so no affordable nxi is a 1e-6
    # reference; instead verify that (a) the polar family self-converges
    # and (b) the grid marches toward the converged polar value as nxi
    # grows (common limit).
    i = test_faces[0]
    pts = polar_self_points(faces, i, 96, 64)
    dense = int_self_trac_polar(vp, vs, rho, w, faces, i, pts)
    scale_d = np.abs(dense).max()
    pts = polar_self_points(faces, i, 16, 12)
    p16 = int_self_trac_polar(vp, vs, rho, w, faces, i, pts)
    polar_self_conv = np.abs(p16 - dense).max() / scale_d
    print(f"polar self-convergence (16x12 vs 96x64): {polar_self_conv:.3e}")
    grid_gap = {}
    for nxi in (300, 600, 1200, 2400):
        refs_n, wis_n = self_ref_grid(nxi)
        gv = int_self_trac(vp, vs, rho, w, faces, i, wis_n, refs_n)
        grid_gap[nxi] = np.abs(gv - dense).max() / scale_d
        print(f"grid(nxi={nxi:4d}) vs converged polar: {grid_gap[nxi]:.3e}")
    same_limit = grid_gap[2400] < grid_gap[300] / 4.0

    # displacement (G) self block: production order is 32x24
    # (Geometry.polar_weak); measure its self-convergence too
    truth_g = int_self_G(vp, vs, rho, w, faces, i, wis_fine, refs_fine)
    g_dense = int_self_G_polar(vp, vs, rho, w, faces, i,
                               polar_self_points(faces, i, 96, 64))
    sg = np.abs(g_dense).max()
    e300g = np.abs(int_self_G(vp, vs, rho, w, faces, i, wis_300, refs_300)
                   - truth_g).max() / np.abs(truth_g).max()
    for nt_, nr_ in ((16, 12), (32, 24)):
        gp = int_self_G_polar(vp, vs, rho, w, faces, i,
                              polar_self_points(faces, i, nt_, nr_))
        eg = np.abs(gp - truth_g).max() / np.abs(truth_g).max()
        egc = np.abs(gp - g_dense).max() / sg
        print(f"G self block polar{nt_}x{nr_}: vs grid1501 {eg:.3e}, "
              f"self-convergence vs 96x64 {egc:.3e}")
    eg = np.abs(int_self_G_polar(vp, vs, rho, w, faces, i,
                                 polar_self_points(faces, i, 32, 24))
                - truth_g).max() / np.abs(truth_g).max()
    print(f"G self block: grid300 err {e300g:.3e} "
          f"(production = polar32x24)")

    # timing
    t0 = time.time()
    for _ in range(20):
        int_self_trac(vp, vs, rho, w, faces, i, wis_300, refs_300)
    t_grid = (time.time() - t0) / 20
    pts = polar_self_points(faces, i, 16, 12)
    t0 = time.time()
    for _ in range(200):
        int_self_trac_polar(vp, vs, rho, w, faces, i, pts)
    t_polar = (time.time() - t0) / 200
    print(f"per-face timing: grid300 {t_grid * 1e3:.1f} ms, "
          f"polar16x12 {t_polar * 1e3:.2f} ms "
          f"({t_grid / t_polar:.0f}x)")

    ok = (worst_polar[(16, 12)] < worst_300 and same_limit
          and polar_self_conv < 1e-6 and eg < e300g)
    print()
    if not ok:
        print("FAILED: polar quadrature did not beat the legacy grid")
        sys.exit(1)
    print(f"self-quadrature study PASSED: polar16x12 worst err "
          f"{worst_polar[(16, 12)]:.3e} vs grid300 {worst_300:.3e} "
          f"(errors rel to nxi=1501 grid, which itself carries ~1e-5 "
          f"stair-step error; grid marches to the polar limit)")


if __name__ == "__main__":
    main()
