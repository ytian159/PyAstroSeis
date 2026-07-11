#!/usr/bin/env python3
"""Fluid-fluid interface gate (docs/fluid_fluid_derivation.md).

Gates:
  G1 transparent split: fluid core divided by an interposed sphere of
     IDENTICAL fluid -> surface solution matches the unsplit
     liquid-core model (transparent-interface class);
  G2 A/B swap: registration order of the two fluids swapped -> same
     physics (permuted system), plus the analytic self-operator
     identity A_(+) + A_(-) = I on the shared interface;
  G3 dense vs eliminated on fluid-fluid stacks (incl. a middle fluid
     bounded by fluid-fluid on both sides -> own-modulus _scale
     fallback, and a strong density-contrast variant), multi-RHS,
     surface-only bitwise; condition numbers printed;
  G4 rung-3 cache with the fluid-fluid boundary perturbed: bitwise
     block reuse, expected recompute count, bitwise solution.

The strict no-op gate (G0) is the existing battery: no code path here
changes unless an interface has two fluid sides.
Run on a compute node (small synthetic meshes, ~ a few minutes).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.domains import (Material, MultiDomainModel, Region,
                                 Interface, nested_shell_model,
                                 nested_shell_model_from_ifaces,
                                 FLUID_FLUID)                   # noqa: E402
from pyastroseis.elimination import (ShellElimination,
                                     cached_blocks)             # noqa: E402
from pyastroseis.layered import (incident_outer_source,
                                 incident_solid_layer_source,
                                 layer_materials,
                                 layer_meshes, _outward)        # noqa: E402
from pyastroseis.meshgen import gen_mesh_relief, relief_ylm     # noqa: E402
from pyastroseis.source import u0e                              # noqa: E402
from pyastroseis.assembly import Geometry, cal_A_st             # noqa: E402
from pyastroseis.liquidcore import flip_normals                 # noqa: E402

FAILED = []
FSRC = [1.0, 1.0, 1.0]
W0 = 2 * np.pi * 0.3
F_HZ = 0.05
IDENT = 1e-10
TRANSP_GATE = 8e-2     # transparent-interface class (rung-1/2e gates)
SWAP_GATE = 1e-8

SOLID = {"vp": 6000.0, "vs": 3000.0, "rho": 3000.0, "Q": 200.0}
CORE = {"vp": 8000.0, "vs": 4500.0, "rho": 4000.0, "Q": 500.0}
MID = {"vp": 7000.0, "vs": 3900.0, "rho": 3500.0, "Q": 300.0}
FLUID = {"vp": 8000.0, "vs": 0.0, "rho": 4000.0, "Q": 200.0}
FLUID2 = {"vp": 6500.0, "vs": 0.0, "rho": 9000.0, "Q": 300.0}
FLUID3 = {"vp": 5500.0, "vs": 0.0, "rho": 2000.0, "Q": 150.0}


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


def build(layers, seed=1):
    faces = layer_meshes(layers, seed=seed)
    mats = layer_materials(layers)
    model, ifaces = nested_shell_model(faces, mats, W0)
    return model, ifaces, faces, mats


def surface_u(model, ifaces, w, field, elim=True):
    b = model.assemble_rhs(incident_outer_source(model, ifaces, field))
    if elim:
        return ShellElimination(model, ifaces).solve(w=w, b=b, full=False)
    x = np.linalg.solve(model.assemble(w), b)
    return x[model.block_slice("u", ifaces[-1])]


def gate_transparent():
    """G1: identical fluid both sides of the split -> the fluid-fluid
    interface must be transparent at the surface."""
    w = 2 * np.pi * F_HZ + 0.08j
    print("[G1 transparent fluid-fluid split]", flush=True)
    ref_layers = [layer(8000.0, 12, FLUID), layer(20000.0, 25, SOLID)]
    faces_ref = layer_meshes(ref_layers, seed=1)
    mats_ref = layer_materials(ref_layers)
    model_r, ifaces_r = nested_shell_model(faces_ref, mats_ref, W0)
    field = source_field(faces_ref[-1], w, mats_ref[-1])
    u_ref = surface_u(model_r, ifaces_r, w, field, elim=False)

    split = layer_meshes([layer(5000.0, 12, FLUID)], seed=42)[0]
    faces_s = [split, faces_ref[0], faces_ref[1]]
    mats_s = layer_materials([layer(5000.0, 12, FLUID),
                              layer(8000.0, 12, FLUID),
                              layer(20000.0, 25, SOLID)])
    model_s, ifaces_s = nested_shell_model(faces_s, mats_s, W0)
    assert ifaces_s[0].condition == FLUID_FLUID
    u_s = surface_u(model_s, ifaces_s, w, field, elim=False)
    err = np.linalg.norm(u_s - u_ref) / np.linalg.norm(u_ref)
    check("split fluid core vs unsplit (surface)", err < TRANSP_GATE,
          f"  (rel err {err:.3e}, gate {TRANSP_GATE:g}, "
          f"size {model_s.size} vs {model_r.size})")


def gate_swap():
    """G2: swap the two fluids' registration order (who owns "p" vs
    "un" rows) -> same solution; plus A_(+) + A_(-) = I."""
    w = 2 * np.pi * F_HZ + 0.08j
    print("[G2 A/B swap + self-operator identity]", flush=True)
    layers = [layer(5000.0, 12, FLUID2), layer(8000.0, 12, FLUID),
              layer(20000.0, 25, SOLID)]
    faces = layer_meshes(layers, seed=1)
    mats = layer_materials(layers)
    model, ifaces = nested_shell_model(faces, mats, W0)
    split, cmb, surf = ifaces
    field = source_field(faces[-1], w, mats[-1])

    reg_a = Region(mats[0], ((split, +1),), "layer0")
    reg_b = Region(mats[1], ((split, -1), (cmb, +1)), "layer1")
    reg_s = Region(mats[2], ((cmb, -1), (surf, +1)), "layer2")
    m1 = MultiDomainModel((reg_a, reg_b, reg_s), w0=W0)
    m2 = MultiDomainModel((reg_b, reg_a, reg_s), w0=W0)
    assert m1._fluid_of[split][0] is reg_a
    assert m2._fluid_of[split][0] is reg_b
    u1 = surface_u(m1, ifaces, w, field, elim=False)
    u2 = surface_u(m2, ifaces, w, field, elim=False)
    err = np.linalg.norm(u2 - u1) / np.linalg.norm(u1)
    check("registration swap (surface)", err < SWAP_GATE,
          f"  (rel err {err:.3e}, gate {SWAP_GATE:g})")

    m = mats[1]
    geo_p = Geometry(split.faces, self_scheme="polar",
                     quad_mode="adaptive")
    Ap = cal_A_st(split.faces, split.faces, w, m.lamda, m.mu, m.rho,
                  m.Q, qp_fac=m.qp_fac, geom=geo_p)
    ff = flip_normals(split.faces)
    geo_m = Geometry(ff, self_scheme="polar", quad_mode="adaptive")
    Am = cal_A_st(ff, ff, w, m.lamda, m.mu, m.rho, m.Q,
                  qp_fac=m.qp_fac, geom=geo_m)
    err = np.abs(Ap + Am - np.eye(split.faces.n)).max()
    check("A_(+) + A_(-) == I", err < 1e-12, f"  (max abs {err:.3e})")


def gate_elim():
    """G3: dense == eliminated on fluid-fluid stacks."""
    w = 2 * np.pi * F_HZ + 0.08j
    cases = {
        "ff_annulus4": [layer(6000.0, 12, CORE), layer(9500.0, 14, FLUID),
                        layer(13000.0, 16, FLUID),
                        layer(20000.0, 25, SOLID)],
        "ff_core4": [layer(5000.0, 12, FLUID), layer(9000.0, 14, FLUID),
                     layer(14000.0, 18, MID), layer(20000.0, 25, SOLID)],
        "ff_contrast4": [layer(5000.0, 12, FLUID2),
                         layer(9000.0, 14, FLUID),
                         layer(14000.0, 18, MID),
                         layer(20000.0, 25, SOLID)],
        "ff_stack5": [layer(4000.0, 12, FLUID3), layer(7000.0, 12, FLUID),
                      layer(10000.0, 14, FLUID2),
                      layer(15000.0, 18, MID), layer(20000.0, 25, SOLID)],
    }
    print("[G3 dense vs eliminated, fluid-fluid]", flush=True)
    for name, layers in cases.items():
        model, ifaces, faces, mats = build(layers)
        nff = sum(1 for i in ifaces if i.condition == FLUID_FLUID)
        field = source_field(faces[-1], w, mats[-1])
        b = model.assemble_rhs(incident_outer_source(model, ifaces, field))
        A = model.assemble(w)
        t0 = time.time()
        x_ref = np.linalg.solve(A, b)
        t_dense = time.time() - t0
        elim = ShellElimination(model, ifaces)
        t0 = time.time()
        x = elim.solve(w=w, b=b, full=True)
        t_elim = time.time() - t0
        err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        cond = np.linalg.cond(A)
        check(f"{name} full solution", err < IDENT,
              f"  (rel err {err:.3e}; {nff} ff ifaces; cond {cond:.2e}; "
              f"dense {t_dense:.1f} s, elim {t_elim:.1f} s, "
              f"size {model.size})")

        blocks = model.assemble_blocks(w)
        B = np.column_stack([b, 2j * b, np.roll(b, 7)])
        X = elim.solve(blocks=blocks, b=B, full=True)
        X_ref = np.linalg.solve(A, B)
        err = np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)
        check(f"{name} multi-RHS", err < IDENT, f"  (rel err {err:.3e})")

        surf = elim.solve(blocks=blocks, b=b, full=False)
        x_full = elim.solve(blocks=blocks, b=b, full=True)
        sl = model.block_slice("u", ifaces[-1])
        check(f"{name} surface-only == full surface block",
              np.array_equal(surf, x_full[sl]))

    # incident helpers: outermost-layer source through the general
    # path must stay bitwise (now with "un" blocks present)
    model, ifaces, faces, mats = build(cases["ff_core4"])
    f_out = source_field(faces[-1], w, mats[-1])
    inc_a = incident_outer_source(model, ifaces, f_out)
    inc_b = incident_solid_layer_source(model, ifaces, len(ifaces) - 1,
                                        f_out)
    same = set(inc_a) == set(inc_b) and all(
        np.array_equal(inc_a[k], inc_b[k]) for k in inc_a)
    check("incident helpers agree bitwise (with un blocks)", same)


def gate_cache():
    """G4: rung-3 block cache with the fluid-fluid boundary perturbed."""
    w = 2 * np.pi * F_HZ + 0.08j
    layers = [layer(6000.0, 12, CORE), layer(9500.0, 14, FLUID),
              layer(13000.0, 16, FLUID), layer(20000.0, 25, SOLID)]
    model, ifaces, faces, mats = build(layers)
    cache = model.assemble_blocks(w)

    rng = np.random.default_rng(101)
    pert_faces, _ = gen_mesh_relief(9500.0, 14, relief_ylm(2, 0, 300.0),
                                    rng=rng)
    pert_faces = _outward(pert_faces)
    if2 = list(ifaces)
    if2[1] = Interface(pert_faces, ifaces[1].condition, ifaces[1].name)
    assert if2[1].condition == FLUID_FLUID
    m2 = nested_shell_model_from_ifaces(if2, mats, W0)

    fresh = m2.assemble_blocks(w)
    cached, nnew = cached_blocks(m2, w, cache)

    print("[G4 block cache across a fluid-fluid boundary]", flush=True)
    same_keys = set(cached) == set(fresh)
    bitwise = same_keys and all(np.array_equal(cached[k], fresh[k])
                                for k in fresh)
    ntouch = sum(1 for (rk, ck) in fresh
                 if rk[1] is if2[1] or ck[1] is if2[1])
    check("cached blocks bitwise == fresh", bitwise,
          f"  (recomputed {nnew}/{len(fresh)} blocks, expected {ntouch})")
    check("only perturbed-boundary blocks recomputed", nnew == ntouch)

    field = source_field(faces[-1], w, mats[-1])
    b = m2.assemble_rhs(incident_outer_source(m2, if2, field))
    elim = ShellElimination(m2, if2)
    x_f = elim.solve(blocks=fresh, b=b, full=True)
    x_c = elim.solve(blocks=cached, b=b, full=True)
    check("cached solution bitwise == fresh", np.array_equal(x_c, x_f))


def main():
    gate_transparent()
    gate_swap()
    gate_elim()
    gate_cache()
    if FAILED:
        print(f"fluid-fluid gate FAILED: {FAILED}")
        sys.exit(1)
    print("fluid-fluid gate PASSED")


if __name__ == "__main__":
    main()
