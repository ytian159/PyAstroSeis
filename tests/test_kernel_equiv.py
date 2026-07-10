#!/usr/bin/env python3
"""Verify the canonical (CSE) traction kernel against the verbatim
Mathematica transcription on random configurations. Pure Python — no
oracle files needed; runs in seconds on a login node.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.greens import (green_traction_tensor,
                                green_traction_tensor_reference)

NAMES = ["T11", "T12", "T13", "T21", "T22", "T23", "T31", "T32", "T33"]


def main():
    rng = np.random.default_rng(7)
    worst = 0.0
    for trial in range(6):
        npts = 4000
        x = rng.uniform(-2e4, 2e4, npts)
        y = rng.uniform(-2e4, 2e4, npts)
        z = rng.uniform(-2e4, 2e4, npts)
        xs, ys, zs = rng.uniform(-1e4, 1e4, 3)
        nvec = rng.standard_normal(3)
        nvec /= np.linalg.norm(nvec)
        vp0 = rng.uniform(2000, 9000)
        vs0 = vp0 / rng.uniform(1.5, 2.2)
        rho = rng.uniform(1000, 6000)
        Q = rng.uniform(20, 800)
        vp = vp0 / (1 + 0.5j / (2.5 * Q))
        vs = vs0 / (1 + 0.5j / Q)
        w = 2 * np.pi * rng.uniform(0.01, 2.0) + 0.08j

        Ta = green_traction_tensor(vp, vs, rho, w, x, y, z, xs, ys, zs,
                                   nvec[0], nvec[1], nvec[2])
        Tb = green_traction_tensor_reference(vp, vs, rho, w, x, y, z,
                                             xs, ys, zs,
                                             nvec[0], nvec[1], nvec[2])
        for name, a, b in zip(NAMES, Ta, Tb):
            scale = np.abs(b).max()
            err = np.abs(a - b).max() / scale
            worst = max(worst, err)
            if err > 1e-12:
                print(f"FAIL trial {trial} {name}: rel err {err:.3e}")
                sys.exit(1)
    print(f"canonical kernel == reference on 6x4000 random points, "
          f"worst rel err = {worst:.3e}")
    print("kernel equivalence PASSED")


if __name__ == "__main__":
    main()
