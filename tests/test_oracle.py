#!/usr/bin/env python3
"""Unit-test the Python port against the MATLAB oracle dump.

Run tests/dump_oracle.m first (matlab -batch dump_oracle), then:
    python tests/test_oracle.py
"""

import os
import sys

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import cal_traction, int_self_trac, self_ref_grid, subgrid_all  # noqa: E402
from pyastroseis.greens import green_traction_tensor  # noqa: E402
from pyastroseis.mesh import Faces, load_faces_mat  # noqa: E402
from pyastroseis.source import u0e  # noqa: E402

FAILED = []


def check(name, got, want, tol):
    got = np.asarray(got).ravel()
    want = np.asarray(want).ravel()
    denom = np.max(np.abs(want))
    err = np.max(np.abs(got - want)) / (denom if denom > 0 else 1.0)
    status = "ok " if err < tol else "FAIL"
    print(f"  [{status}] {name:24s} max rel err = {err:.3e} (tol {tol:.0e})")
    if err >= tol:
        FAILED.append(name)


def subset(faces, n):
    return Faces(**{k: getattr(faces, k)[:n] for k in
                    ("A", "B", "C", "nvec", "ic", "area", "r", "a", "b", "c")})


def main():
    o = sio.loadmat(os.path.join(ROOT, "tests", "oracle.mat"), squeeze_me=True)
    faces = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))

    vpr, vsr, rho, Q = 6000.0, 3000.0, 3000.0, 570.0
    mu0 = rho * vsr * vsr
    lamda0 = rho * vpr * vpr - 2 * mu0
    Qp = 2.5 * Q
    vp = vpr / (1 + 0.5j / Qp)
    vs = vsr / (1 + 0.5j / Q)
    w = 2 * np.pi * 0.02 + 0.08j

    print("kernel green_traction_tensor:")
    idx = np.arange(100, 200)
    x1, x2, x3 = faces.ic[idx].T
    n1, n2, n3 = faces.nvec[idx].T
    ps7 = faces.ic[6]
    T = green_traction_tensor(vp, vs, rho, w, x1, x2, x3,
                              ps7[0], ps7[1], ps7[2], n1, n2, n3)
    names = ["T11", "T12", "T13", "T21", "T22", "T23", "T31", "T32", "T33"]
    for name, t in zip(names, T):
        check(name, t, o[name], 1e-13)

    print("sub-grid quadrature (face 3):")
    xia, yia, zia, wi = subgrid_all(faces)
    check("subgrid x", xia[2], o["sgx"], 1e-13)
    check("subgrid y", yia[2], o["sgy"], 1e-13)
    check("subgrid z", zia[2], o["sgz"], 1e-13)
    check("subgrid w", wi, o["sgw"], 1e-13)

    print("self integral (face 5):")
    refs, wis = self_ref_grid()
    ST5 = int_self_trac(vp, vs, rho, w, faces, 4, wis, refs)
    check("ST5", ST5, o["ST5"], 1e-12)

    print("60-face system:")
    fsub = subset(faces, 60)
    tracs = cal_traction(fsub, w, lamda0, mu0, rho, Q)
    check("TRAC(60)", tracs, o["TRACs"], 1e-12)

    u0s = u0e(fsub, w, rho, mu0, lamda0,
              o["xs"], o["ys"], o["zs"], Q, [1.0, 1.0, 1.0])
    check("u0", u0s, o["u0s"], 1e-13)

    xsol = np.linalg.solve(tracs, u0s)
    check("solve", xsol, o["xsol"], 1e-9)

    # cross-check mesh rebuild from V/Tri against stored face struct
    m = sio.loadmat(os.path.join(ROOT, "my_mesh.mat"), squeeze_me=True)
    from pyastroseis.mesh import faces_from_vertices
    fr = faces_from_vertices(m["V"], m["Tri"])
    print("mesh rebuild from V/Tri:")
    for k in ("ic", "nvec", "area", "r"):
        check(f"faces.{k}", getattr(fr, k), getattr(faces, k), 1e-12)

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} checks: {FAILED}")
        sys.exit(1)
    print("all oracle checks passed")


if __name__ == "__main__":
    main()
