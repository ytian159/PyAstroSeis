#!/usr/bin/env python3
"""Rung-0 gate (docs/multilayer_roadmap.md): MultiDomainModel must
reproduce the two existing solver configurations BITWISE — matrix,
RHS, and solution —
  * one solid region + free surface   vs  assembly.cal_traction
  * solid shell + fluid core          vs  liquidcore.liq_core
in both production mode (physical Qp, polar self-quadrature, adaptive
far quadrature) and MATLAB-exact mode (legacy Qp, grid, full).

Default: fast synthetic spheres (seconds). --full adds the real
my_mesh.mat / my_mesh_lc.mat cases at one frequency each (compute
node, ~25 min).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import (Geometry, QP_FAC_LEGACY, cal_traction,
                                  smat_func)                    # noqa: E402
from pyastroseis.domains import (FLUID_SOLID, FREE, Interface, Material,
                                 MultiDomainModel, Region,
                                 homogeneous_model,
                                 liquid_core_model)             # noqa: E402
from pyastroseis.liquidcore import (flip_normals, liq_core,
                                    load_layers_mat)            # noqa: E402
from pyastroseis.mesh import load_faces_mat                     # noqa: E402
from pyastroseis.meshgen import gen_layer                       # noqa: E402
from pyastroseis.solver import qp_factors                       # noqa: E402
from pyastroseis.source import u0e                              # noqa: E402

FAILED = []
MODES = (("production", "polar", "adaptive", "physical"),
         ("matlab-exact", "grid", "full", "legacy"))


def check(label, ok, kind="OK (bitwise)"):
    print(f"    {label}: {kind if ok else 'FAIL'}", flush=True)
    if not ok:
        FAILED.append(label)


def outward(faces):
    """Ensure stored normals point away from the body center (the
    convention of the shipped meshes)."""
    if np.median(np.sum(faces.nvec * faces.ic, axis=1)) < 0:
        return flip_normals(faces)
    return faces


def gate_homogeneous(faces, f_hz, tag):
    rho, Q, vs0, vp0 = 3000.0, 570.0, 3000.0, 6000.0
    mu = rho * vs0 * vs0
    lamda = rho * vp0 * vp0 - 2 * mu
    R = np.mean(np.sqrt((faces.ic ** 2).sum(axis=1)))
    w = 2 * np.pi * f_hz + 0.08j
    for label, scheme, qmode, qpm in MODES:
        qk, qs = qp_factors(qpm, vp0, vs0)
        u01 = u0e(faces, w, rho, mu, lamda, -(R - 2e3), 0.0, 0.0, Q,
                  [1.0, 1.0, 1.0], qp_fac=qs["u0e"])
        t0 = time.time()
        # reference path: exactly what run_case does per frequency
        geom = Geometry(faces, self_scheme=scheme, quad_mode=qmode)
        T = cal_traction(faces, w, lamda, mu, rho, Q, qp_fac=qk, geom=geom)
        x_ref = np.linalg.solve(T, u01)
        # multi-domain path
        model, surf = homogeneous_model(
            faces, Material.solid(vp0, vs0, rho, Q, qp_fac=qk),
            self_scheme=scheme, quad_mode=qmode)
        A = model.assemble(w)
        b = model.assemble_rhs({("u", surf): u01})
        x = np.linalg.solve(A, b)
        print(f"[homog/{tag} {label}] n={faces.n} f={f_hz:g} Hz "
              f"({time.time() - t0:.1f} s)", flush=True)
        check("matrix", np.array_equal(A, T))
        check("rhs", np.array_equal(b, u01))
        check("solution", np.array_equal(x, x_ref))


def gate_lc(face1, face2, f_hz, tag):
    # material line-up of BEM_para_lc
    vp1, vs1, rho1, Q = 6000.0, 3000.0, 3000.0, 200.0
    vp2, vs2, rho2 = 8000.0, 0.0, 4000.0
    mu1 = rho1 * vs1 * vs1
    lamda1 = rho1 * vp1 * vp1 - 2 * mu1
    mu2 = rho1 * vs2 * vs2          # driver quirk, kept verbatim
    lamda2 = rho2 * vp2 * vp2 - 2 * mu2
    w0 = 2 * np.pi * 0.3
    n1, n2 = face1.n, face2.n
    R = np.mean(np.sqrt((face1.ic ** 2).sum(axis=1)))
    xs, ys, zs = -(R - 4e3), 0.0, 0.0
    w = 2 * np.pi * f_hz + 0.08j
    Smat = smat_func(face2)
    for label, scheme, qmode, qpm in MODES:
        qk, qs = qp_factors(qpm, vp1, vs1)
        qf = 1.0 if qpm == "physical" else QP_FAC_LEGACY
        u01 = u0e(face1, w, rho1, mu1, lamda1, xs, ys, zs, Q,
                  [1.0, 1.0, 1.0], qp_fac=qs["u0e"])
        u02 = u0e(face2, w, rho1, mu1, lamda1, xs, ys, zs, Q,
                  [1.0, 1.0, 1.0], qp_fac=qs["u0e"])
        P0 = np.zeros(n2, dtype=complex)
        t0 = time.time()
        # reference path: exactly what run_case_lc does per frequency
        face2w1 = flip_normals(face2)
        geoms = (Geometry(face1, self_scheme=scheme, quad_mode=qmode),
                 Geometry(face2w1, self_scheme=scheme, quad_mode=qmode),
                 Geometry(face2, self_scheme=scheme, quad_mode=qmode))
        A_ref, b_ref = liq_core(face1, face2, w, lamda1, mu1, rho1,
                                lamda2, mu2, rho2, Q, P0, u01, u02,
                                Smat, w0, qp_fac=qk, qp_fac_fluid=qf,
                                face2w1=face2w1, geoms=geoms,
                                self_scheme=scheme)
        x_ref = np.linalg.solve(A_ref, b_ref)
        # multi-domain path
        mat1 = Material.solid(vp1, vs1, rho1, Q, qp_fac=qk)
        mat2 = Material(lamda2, mu2, rho2, Q, qp_fac=qf, fluid=True)
        model, surf, core = liquid_core_model(
            face1, face2, mat1, mat2, w0,
            self_scheme=scheme, quad_mode=qmode)
        A = model.assemble(w)
        b = model.assemble_rhs({("p", core): P0, ("u", core): u02,
                                ("u", surf): u01})
        x = np.linalg.solve(A, b)
        print(f"[lc/{tag} {label}] n1={n1} n2={n2} f={f_hz:g} Hz "
              f"({time.time() - t0:.1f} s)", flush=True)
        check("block layout",
              model.block_slice("p", core) == slice(0, n2)
              and model.block_slice("u", core) == slice(n2, 4 * n2)
              and model.block_slice("u", surf) == slice(4 * n2,
                                                        4 * n2 + 3 * n1))
        check("matrix", np.array_equal(A, A_ref))
        check("rhs", np.array_equal(b, b_ref))
        check("solution", np.array_equal(x, x_ref))


def gate_validation(face1, face2):
    """The model must reject configurations rung 0 does not support."""
    solid = Material.solid(6000.0, 3000.0, 3000.0, 200.0)
    fluid = Material.acoustic(8000.0, 4000.0, 200.0)
    iface = Interface(face2, FLUID_SOLID, "core")

    def expects(exc, fn, label):
        try:
            fn()
        except exc:
            check(label, True, kind="OK")
        else:
            check(label, False)

    print("[validation]", flush=True)
    expects(NotImplementedError, lambda: MultiDomainModel(
        (Region(solid, ((Interface(face2, "welded", "w"), +1),)),
         Region(solid, ((Interface(face2, "welded", "w"), -1),)))),
        "welded condition rejected (rung 1)")
    expects(ValueError, lambda: MultiDomainModel(
        (Region(fluid, ((iface, -1),)),
         Region(solid, ((iface, +1), (Interface(face1, FREE, "s"), +1))),),
        w0=1.0),
        "fluid-side sign convention enforced")
    expects(ValueError, lambda: MultiDomainModel(
        (Region(fluid, ((iface, +1),)),
         Region(solid, ((iface, -1), (Interface(face1, FREE, "s"), +1))),)),
        "missing w0 rejected for fluid regions")


def main():
    full = "--full" in sys.argv

    rng = np.random.default_rng(1)
    layers = gen_layer((10000.0, 20000.0), (12, 25), (0, 0), rng=rng)
    face2s = outward(layers[0][0])
    face1s = outward(layers[1][0])

    gate_homogeneous(face1s, 0.05, "synthetic")
    gate_lc(face1s, face2s, 0.05, "synthetic")
    gate_validation(face1s, face2s)

    if full:
        faces = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))
        gate_homogeneous(faces, 0.12, "my_mesh")
        lay = load_layers_mat(os.path.join(ROOT, "my_mesh_lc.mat"))
        face2, face1 = lay[0], lay[1]   # core, free surface
        gate_lc(face1, face2, 0.3, "my_mesh_lc")

    if FAILED:
        print(f"rung-0 gate FAILED: {FAILED}")
        sys.exit(1)
    print("rung-0 gate PASSED" + ("" if full else " (synthetic only; "
          "run with --full on a compute node for the real meshes)"))


if __name__ == "__main__":
    main()
