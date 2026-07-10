#!/usr/bin/env python3
"""Unit-test the moment-source, liquid-core, and mesh-generation ports
against the second MATLAB oracle dump.

Run tests/dump_oracle2.m first (matlab -batch dump_oracle2), then:
    python tests/test_oracle2.py
"""

import os
import sys

import numpy as np
import scipy.io as sio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import (cal_A_st, cal_B_st, cal_G_st, cal_T_st,
                                  smat_func)  # noqa: E402
from pyastroseis.greens import (bmat_func, g_singular_value,
                                greens_deri_src, greens_func_deri_p,
                                greens_func_p)  # noqa: E402
from pyastroseis.liquidcore import flip_normals, liq_core, load_layers_mat  # noqa: E402
from pyastroseis.mesh import Faces  # noqa: E402
from pyastroseis.meshgen import (get_phobos_topo, smooth_topo_rand,
                                 subdivide_spherical_mesh, ylm1)  # noqa: E402
from pyastroseis.source import u0eM, u0p_exp  # noqa: E402

FAILED = []


def check(name, got, want, tol):
    got = np.asarray(got).ravel()
    want = np.asarray(want).ravel()
    denom = np.max(np.abs(want))
    err = np.max(np.abs(got - want)) / (denom if denom > 0 else 1.0)
    status = "ok " if err < tol else "FAIL"
    print(f"  [{status}] {name:14s} max rel err = {err:.3e} (tol {tol:.0e})")
    if err >= tol:
        FAILED.append(name)


def subset(faces, n):
    return Faces(**{k: getattr(faces, k)[:n] for k in
                    ("A", "B", "C", "nvec", "ic", "area", "r", "a", "b", "c")})


def main():
    o = sio.loadmat(os.path.join(ROOT, "tests", "oracle2.mat"),
                    squeeze_me=True)
    layers = load_layers_mat(os.path.join(
        ROOT, "examples", "shifted_liquid_core", "layers_shift.mat"))
    f2full, f1full = layers[0], layers[1]
    f2 = subset(f2full, 40)
    f1 = subset(f1full, 60)

    vp1, vs1, rho1, Q = 6000.0, 3000.0, 3000.0, 200.0
    vp2, vs2, rho2 = 8000.0, 0.0, 4000.0
    mu1 = rho1 * vs1 * vs1
    lamda1 = rho1 * vp1 * vp1 - 2 * mu1
    mu2 = rho1 * vs2 * vs2
    lamda2 = rho2 * vp2 * vp2 - 2 * mu2
    w = 2 * np.pi * 0.04 + 0.08j
    w0 = 2 * np.pi * 0.3
    Qp = 2.5 * Q
    vpc1 = vp1 / (1 + 0.5j / Qp)
    vsc1 = vs1 / (1 + 0.5j / Q)
    vpc2 = vp2 / (1 + 0.5j / Qp)

    print("kernels:")
    x1, x2, x3 = f1.ic[:50].T
    n1, n2, n3 = f1.nvec[:50].T
    psrc = f2.ic[2]
    B = bmat_func(vpc1, vsc1, rho1, w, x1, x2, x3, psrc[0], psrc[1], psrc[2])
    for name, t in zip(["B11", "B12", "B13", "B21", "B22", "B23",
                        "B31", "B32", "B33"], B):
        check(name, t, o[name], 1e-13)
    check("greens_func_p", greens_func_p(vpc2, w, x1, x2, x3, *psrc),
          o["gp"], 1e-13)
    check("greens_deri_p", greens_func_deri_p(vpc2, w, x1, x2, x3, *psrc,
                                              n1, n2, n3), o["gdp"], 1e-13)
    check("G_singular", g_singular_value(vpc1, vsc1, rho1, w, f1.r[4],
                                         *f1.nvec[4]), o["GSV"], 1e-13)
    gi1, gi2, gi3 = greens_deri_src(vpc1, vsc1, rho1, w, x1[:20], x2[:20],
                                    x3[:20], psrc[0], psrc[1], psrc[2])
    check("gd1", gi1, np.moveaxis(o["gd1"], -1, -1), 1e-13)
    check("gd2", gi2, o["gd2"], 1e-13)
    check("gd3", gi3, o["gd3"], 1e-13)

    print("liquid-core matrices (subsets):")
    f2w = flip_normals(f2)
    check("T11", cal_T_st(f1, f1, w, lamda1, mu1, rho1, Q), o["T11s"], 1e-12)
    check("T22", cal_T_st(f2w, f2w, w, lamda1, mu1, rho1, Q), o["T22s"], 1e-12)
    check("T12", cal_T_st(f1, f2w, w, lamda1, mu1, rho1, Q), o["T12s"], 1e-12)
    check("T21", cal_T_st(f2w, f1, w, lamda1, mu1, rho1, Q), o["T21s"], 1e-12)
    check("G12", cal_G_st(f1, f2w, w, lamda1, mu1, rho1, Q), o["G12s"], 1e-12)
    check("G22", cal_G_st(f2w, f2w, w, lamda1, mu1, rho1, Q), o["G22s"], 1e-12)
    check("A22", cal_A_st(f2, f2, w, lamda2, mu2, rho2, Q), o["A22s"], 1e-12)
    check("B22", cal_B_st(f2, f2, w, lamda2, mu2, rho2, Q), o["B22s"], 1e-12)

    print("sources + coupled system:")
    xs, ys, zs = float(o["xs"]), float(o["ys"]), float(o["zs"])
    M = np.asarray(o["M"], dtype=float)
    u01m = u0eM(f1, w, rho1, mu1, lamda1, xs, ys, zs, Q, M)
    u02m = u0eM(f2, w, rho1, mu1, lamda1, xs, ys, zs, Q, M)
    check("u0eM(f1)", u01m, o["u01m"], 1e-13)
    check("u0eM(f2)", u02m, o["u02m"], 1e-13)
    check("u0p_exp", u0p_exp(f2, w, vp2, xs, ys, zs, Q), o["u0p"], 1e-13)
    P0 = np.zeros(f2.n, dtype=complex)
    Smat = smat_func(f2)
    A, b = liq_core(f1, f2, w, lamda1, mu1, rho1, lamda2, mu2, rho2, Q,
                    P0, u01m, u02m, Smat, w0)
    check("A_lc", A, o["Alc"], 1e-12)
    check("b_lc", b, o["blc"], 1e-13)
    check("solve_lc", np.linalg.solve(A, b), o["xlc"], 1e-8)

    print("spherical harmonics + topography:")
    tt = np.linspace(0.05, np.pi - 0.05, 25)
    pp = np.linspace(-3, 3, 25)
    check("ylm(45,17)", ylm1(45, 17, tt, pp), o["Y45"], 1e-11)
    check("ylm(5,3)", ylm1(5, 3, tt, pp), o["Y5"], 1e-13)
    check("ylm(7,-4)", ylm1(7, -4, tt, pp), o["Y7m"], 1e-13)
    check("phobos_topo", get_phobos_topo(tt, pp), o["topo25"], 1e-10)
    m = sio.loadmat(os.path.join(ROOT, "pyastroseis", "data",
                                 "marstopo.mat"))
    topos = smooth_topo_rand(m["llon"], 90.0 - m["llat"], m["aa"], 8)
    check("mars_topo", topos, o["topos_mars"], 1e-12)

    print("spherical subdivision (fixed icosahedron):")
    Fs, Vs = subdivide_spherical_mesh(np.asarray(o["FI"], dtype=int),
                                      np.asarray(o["VI"], dtype=float), 2)
    check("subdiv V", Vs, o["Vsub"], 1e-15)
    same_f = np.array_equal(Fs, np.asarray(o["Fsub"], dtype=int))
    print(f"  [{'ok ' if same_f else 'FAIL'}] subdiv F      exact face match: {same_f}")
    if not same_f:
        FAILED.append("subdiv F")

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} checks: {FAILED}")
        sys.exit(1)
    print("all oracle2 checks passed")


if __name__ == "__main__":
    main()
