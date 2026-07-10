#!/usr/bin/env python3
"""Rung-2 gate (docs/multilayer_roadmap.md): N nested shells + YAML.

Gates:
  1. special cases are BITWISE: nested_shell_model with 2 solid layers
     == welded_two_layer_model; with a fluid innermost layer
     == liquid_core_model (matrix, RHS via incident_outer_source,
     solution);
  2. 3-shell transparent — two artificial welded interfaces inside a
     homogeneous sphere (built through the YAML path) must reproduce
     the single-region solution;
  3. LC-transparent — fluid core + two same-material solid shells must
     reproduce the 2-region liquid-core solution (the middle welded
     interface is artificial);
  4. build-time validation (fluid layer not innermost, material count).

Run on a compute node (small synthetic meshes, ~2 min).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import Geometry, cal_traction      # noqa: E402
from pyastroseis.domains import (Material, liquid_core_model,
                                 nested_shell_model,
                                 welded_two_layer_model)     # noqa: E402
from pyastroseis.layered import (build_layered_model, incident_outer_source,
                                 layer_materials, layer_meshes)  # noqa: E402
from pyastroseis.source import u0e                           # noqa: E402

FAILED = []
FSRC = [1.0, 1.0, 1.0]
W0 = 2 * np.pi * 0.3
F_HZ = 0.05

TRANSPARENT3 = 5e-2       # 2 coarse artificial interfaces (80+128 faces)
LC_TRANSPARENT = 8e-2     # LC scheme class + 1 artificial weld

SOLID = {"vp": 6000.0, "vs": 3000.0, "rho": 3000.0, "Q": 200.0}
CORE = {"vp": 8000.0, "vs": 4500.0, "rho": 4000.0, "Q": 500.0}
FLUID = {"vp": 8000.0, "vs": 0.0, "rho": 4000.0, "Q": 200.0}


def layer(r, nmesh, mat):
    return dict(r=r, nmesh=nmesh, **mat)


def check(label, ok, detail=""):
    print(f"    {label}: {'OK' if ok else 'FAIL'}{detail}", flush=True)
    if not ok:
        FAILED.append(label)


def source_field(faces_ref, w, mat):
    R = np.mean(np.sqrt((faces_ref.ic ** 2).sum(axis=1)))
    xs = -(R - 2e3)

    def field(faces):
        return u0e(faces, w, mat.rho, mat.mu, mat.lamda, xs, 0.0, 0.0,
                   mat.Q, FSRC, qp_fac=mat.qp_fac)
    return field


def gate_bitwise(f_core, f_surf):
    w = 2 * np.pi * F_HZ + 0.08j
    mat_shell = layer_materials([layer(0, 0, SOLID)])[0]
    mat_core = layer_materials([layer(0, 0, CORE)])[0]
    mat_fluid = layer_materials([layer(0, 0, FLUID)])[0]
    field = source_field(f_surf, w, mat_shell)

    print("[bitwise special cases]", flush=True)
    for label, mat2, builder in (
            ("welded_two_layer", mat_core,
             lambda: welded_two_layer_model(f_surf, f_core, mat_shell,
                                            mat_core, W0)),
            ("liquid_core", mat_fluid,
             lambda: liquid_core_model(f_surf, f_core, mat_shell,
                                       mat_fluid, W0))):
        m_ref, surf, core = builder()
        m_nest, ifaces = nested_shell_model([f_core, f_surf],
                                            [mat2, mat_shell], W0)
        A_ref = m_ref.assemble(w)
        A = m_nest.assemble(w)
        check("%s matrix" % label, np.array_equal(A, A_ref))
        inc = incident_outer_source(m_nest, ifaces, field)
        b = m_nest.assemble_rhs(inc)
        if label == "welded_two_layer":
            z = np.zeros(3 * f_core.n, dtype=complex)
            b_ref = m_ref.assemble_rhs({("u", surf): field(f_surf),
                                        ("u", core): field(f_core),
                                        ("t", core): z})
        else:
            b_ref = m_ref.assemble_rhs(
                {("p", core): np.zeros(f_core.n, dtype=complex),
                 ("u", core): field(f_core), ("u", surf): field(f_surf)})
        check("%s rhs" % label, np.array_equal(b, b_ref))
        check("%s solution" % label,
              np.array_equal(np.linalg.solve(A, b),
                             np.linalg.solve(A_ref, b_ref)))


def gate_transparent3(faces3):
    w = 2 * np.pi * F_HZ + 0.08j
    layers = [layer(8000.0, 12, SOLID), layer(14000.0, 18, SOLID),
              layer(20000.0, 25, SOLID)]
    cfg = {"layers": layers, "f0": 0.3, "seed": 1}
    t0 = time.time()
    model, ifaces, fl = build_layered_model(cfg, faces_list=faces3)
    mat = layer_materials(layers)[-1]
    field = source_field(fl[-1], w, mat)
    x = model.solve(w, incident_outer_source(model, ifaces, field))
    u1 = x[model.block_slice("u", ifaces[-1])]

    geom = Geometry(fl[-1], self_scheme="polar", quad_mode="adaptive")
    T = cal_traction(fl[-1], w, mat.lamda, mat.mu, mat.rho, mat.Q,
                     qp_fac=mat.qp_fac, geom=geom)
    u_ref = np.linalg.solve(T, field(fl[-1]))
    err = np.linalg.norm(u1 - u_ref) / np.linalg.norm(u_ref)
    print(f"[transparent-3shell] n={[f.n for f in fl]} f={F_HZ:g} Hz "
          f"({time.time() - t0:.1f} s)", flush=True)
    check("surface field vs single-region", err < TRANSPARENT3,
          f"  (rel err {err:.3e}, gate {TRANSPARENT3:g})")


def gate_lc_transparent(faces3):
    w = 2 * np.pi * F_HZ + 0.08j
    mat_shell = layer_materials([layer(0, 0, SOLID)])[0]
    mat_fluid = layer_materials([layer(0, 0, FLUID)])[0]
    field = source_field(faces3[-1], w, mat_shell)
    t0 = time.time()
    m3, if3 = nested_shell_model(faces3, [mat_fluid, mat_shell,
                                          mat_shell], W0)
    x3 = m3.solve(w, incident_outer_source(m3, if3, field))
    u1_3 = x3[m3.block_slice("u", if3[-1])]

    m2, surf, core = liquid_core_model(faces3[-1], faces3[0], mat_shell,
                                       mat_fluid, W0)
    b2 = m2.assemble_rhs(
        {("p", core): np.zeros(faces3[0].n, dtype=complex),
         ("u", core): field(faces3[0]), ("u", surf): field(faces3[-1])})
    x2 = np.linalg.solve(m2.assemble(w), b2)
    u1_2 = x2[m2.block_slice("u", surf)]
    err = np.linalg.norm(u1_3 - u1_2) / np.linalg.norm(u1_2)
    print(f"[lc-transparent] n={[f.n for f in faces3]} f={F_HZ:g} Hz "
          f"({time.time() - t0:.1f} s)", flush=True)
    check("fluid-core + artificial weld vs 2-region LC",
          err < LC_TRANSPARENT,
          f"  (rel err {err:.3e}, gate {LC_TRANSPARENT:g})")


def gate_validation(faces3):
    mat_s = layer_materials([layer(0, 0, SOLID)])[0]
    mat_f = layer_materials([layer(0, 0, FLUID)])[0]

    def expects(exc, fn, label):
        try:
            fn()
        except exc:
            check(label, True)
        else:
            check(label, False)

    print("[validation]", flush=True)
    # rung 2b legalized internal fluid layers (annuli); still
    # unsupported: a fluid OUTERMOST layer (free fluid surface)
    expects(NotImplementedError, lambda: nested_shell_model(
        faces3, [mat_s, mat_s, mat_f], W0),
        "fluid outermost rejected")
    expects(ValueError, lambda: nested_shell_model(
        faces3, [mat_s, mat_s], W0),
        "material/mesh count mismatch rejected")


def main():
    layers = [layer(8000.0, 12, SOLID), layer(14000.0, 18, SOLID),
              layer(20000.0, 25, SOLID)]
    faces3 = layer_meshes(layers, seed=1)

    gate_bitwise(faces3[0], faces3[2])
    gate_transparent3(faces3)
    gate_lc_transparent(faces3)
    gate_validation(faces3)

    if FAILED:
        print(f"rung-2 gate FAILED: {FAILED}")
        sys.exit(1)
    print("rung-2 gate PASSED")


if __name__ == "__main__":
    main()
