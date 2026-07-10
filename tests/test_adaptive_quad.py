#!/usr/bin/env python3
"""Gate for distance-adaptive quadrature with the oscillation cap.

Two regimes on the demo mesh (1584 faces, h ~ half an S wavelength at
the band top):
  - resolved band (f = 0.12 Hz, ks*h ~ 0.8): low-order far tiers kick
    in -> require solution error << self-scheme error and a real
    assembly speedup;
  - under-resolved band top (f = 0.88 Hz, ks*h ~ 6): the cap must
    promote every pair back to the full rule -> bitwise identical to
    quad_mode="full".
Run on a compute node.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import Geometry, cal_traction  # noqa: E402
from pyastroseis.mesh import load_faces_mat  # noqa: E402
from pyastroseis.source import u0e  # noqa: E402

FAILED = []


def main():
    faces = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))
    rho, Q = 3000.0, 570.0
    vs0, vp0 = 3000.0, 6000.0
    mu = rho * vs0 ** 2
    lamda = rho * vp0 ** 2 - 2 * mu

    geom_full = Geometry(faces, self_scheme="polar", quad_mode="full")
    geom_adap = Geometry(faces, self_scheme="polar", quad_mode="adaptive")
    R = np.mean(np.linalg.norm(faces.ic, axis=1))

    for f_hz, regime in ((0.12, "resolved"), (0.88, "band-top")):
        w = 2 * np.pi * f_hz + 0.08j
        cap = geom_adap.tier_cap(abs(w / (vs0 / (1 + 0.5j / Q))))
        t0 = time.time()
        A_full = cal_traction(faces, w, lamda, mu, rho, Q, geom=geom_full)
        t_full = time.time() - t0
        t0 = time.time()
        A_adap = cal_traction(faces, w, lamda, mu, rho, Q, geom=geom_adap)
        t_adap = time.time() - t0

        b = u0e(faces, w, rho, mu, lamda, -(R - 2e3), 0.0, 0.0, Q,
                [1.0, 1.0, 1.0])
        x_full = np.linalg.solve(A_full, b)
        x_adap = np.linalg.solve(A_adap, b)
        sol_err = np.linalg.norm(x_adap - x_full) / np.linalg.norm(x_full)
        print(f"f={f_hz:.2f} Hz ({regime}): cap median tier "
              f"{int(np.median(cap))}, assembly full {t_full:.1f} s / "
              f"adaptive {t_adap:.1f} s ({t_full / t_adap:.2f}x), "
              f"solution rel err {sol_err:.3e}")
        if regime == "resolved":
            if not (sol_err < 1e-5 and t_full / t_adap > 1.5):
                FAILED.append("resolved-band gate")
        else:
            if not sol_err == 0.0:
                FAILED.append("band-top bitwise fallback")

    if FAILED:
        print(f"adaptive quadrature gate FAILED: {FAILED}")
        sys.exit(1)
    print("adaptive quadrature gate PASSED")


if __name__ == "__main__":
    main()
