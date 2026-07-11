#!/usr/bin/env python3
"""Block-elimination gate (docs/multilayer_roadmap.md, rung 2e).

Gates:
  1. eliminated FULL solution == dense np.linalg.solve on every
     interface-condition mix (welded 3-shell; fluid-innermost 3-shell;
     fluid annulus; 5-shell mixed stack), rel err < 1e-10 — plus
     multi-RHS agreement and full=False bitwise-matching the surface
     block of full=True;
  2. cached-block identity (rung-3 hook): rebuilding the mixed model
     with one boundary's Interface replaced and reusing every block
     not touching it (elimination.cached_blocks) gives BITWISE the
     fresh assembly + solution, and recomputes only the expected
     blocks;
  3. thin-shell probe (DIAGNOSTIC, printed): transparent double
     interface at shrinking separation t vs the single-region
     reference — records the accuracy limit t/h that sizes the
     graded-PREM staircase; loose gate: t/h >= 1 stays within 3x the
     rung-1 transparent-interface error class.

Run on a compute node (small synthetic meshes, ~ a few minutes).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.domains import (Material, nested_shell_model,
                                 nested_shell_model_from_ifaces,
                                 Interface)                    # noqa: E402
from pyastroseis.elimination import (ShellElimination,
                                     cached_blocks)            # noqa: E402
from pyastroseis.layered import (incident_outer_source,
                                 incident_solid_layer_source,
                                 layer_materials,
                                 layer_meshes, _outward)       # noqa: E402
from pyastroseis.meshgen import gen_mesh_relief, relief_ylm    # noqa: E402
from pyastroseis.source import u0e                             # noqa: E402
from pyastroseis.assembly import Geometry, cal_traction        # noqa: E402

FAILED = []
FSRC = [1.0, 1.0, 1.0]
W0 = 2 * np.pi * 0.3
F_HZ = 0.05
IDENT = 1e-10
THIN_GATE = 1.5e-1     # 3x the rung-1 transparent class at t/h >= 1

SOLID = {"vp": 6000.0, "vs": 3000.0, "rho": 3000.0, "Q": 200.0}
CORE = {"vp": 8000.0, "vs": 4500.0, "rho": 4000.0, "Q": 500.0}
MID = {"vp": 7000.0, "vs": 3900.0, "rho": 3500.0, "Q": 300.0}
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


def build(layers, seed=1):
    faces = layer_meshes(layers, seed=seed)
    mats = layer_materials(layers)
    model, ifaces = nested_shell_model(faces, mats, W0)
    return model, ifaces, faces, mats


def gate_identity():
    w = 2 * np.pi * F_HZ + 0.08j
    cases = {
        "welded3": [layer(8000.0, 12, CORE), layer(14000.0, 18, MID),
                    layer(20000.0, 25, SOLID)],
        "fluid_core3": [layer(8000.0, 12, FLUID), layer(14000.0, 18, MID),
                        layer(20000.0, 25, SOLID)],
        "annulus": [layer(8000.0, 12, CORE), layer(14000.0, 18, FLUID),
                    layer(20000.0, 25, SOLID)],
        "mixed5": [layer(6000.0, 12, CORE), layer(9500.0, 14, FLUID),
                   layer(13000.0, 16, MID), layer(16500.0, 20, MID),
                   layer(20000.0, 25, SOLID)],
    }
    print("[dense vs eliminated]", flush=True)
    for name, layers in cases.items():
        model, ifaces, faces, mats = build(layers)
        field = source_field(faces[-1], w, mats[-1])
        b = model.assemble_rhs(incident_outer_source(model, ifaces, field))
        t0 = time.time()
        x_ref = np.linalg.solve(model.assemble(w), b)
        t_dense = time.time() - t0
        elim = ShellElimination(model, ifaces)
        t0 = time.time()
        x = elim.solve(w=w, b=b, full=True)
        t_elim = time.time() - t0
        err = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
        check(f"{name} full solution", err < IDENT,
              f"  (rel err {err:.3e}; dense {t_dense:.1f} s, "
              f"elim {t_elim:.1f} s, size {model.size})")

        blocks = model.assemble_blocks(w)
        B = np.column_stack([b, 2j * b, np.roll(b, 7)])
        X = elim.solve(blocks=blocks, b=B, full=True)
        X_ref = np.linalg.solve(model.assemble(w), B)
        err = np.linalg.norm(X - X_ref) / np.linalg.norm(X_ref)
        check(f"{name} multi-RHS", err < IDENT,
              f"  (rel err {err:.3e})")

        surf = elim.solve(blocks=blocks, b=b, full=False)
        x_full = elim.solve(blocks=blocks, b=b, full=True)
        sl = model.block_slice("u", ifaces[-1])
        check(f"{name} surface-only == full surface block",
              np.array_equal(surf, x_full[sl]))


def gate_cache():
    w = 2 * np.pi * F_HZ + 0.08j
    layers = [layer(8000.0, 12, CORE), layer(14000.0, 18, FLUID),
              layer(20000.0, 25, SOLID)]
    model, ifaces, faces, mats = build(layers)
    cache = model.assemble_blocks(w)

    # perturb the middle boundary (fluid annulus top = index 1)
    rng = np.random.default_rng(101)
    pert_faces, _ = gen_mesh_relief(14000.0, 18, relief_ylm(2, 0, 300.0),
                                    rng=rng)
    pert_faces = _outward(pert_faces)
    if2 = list(ifaces)
    if2[1] = Interface(pert_faces, ifaces[1].condition, ifaces[1].name)
    m2 = nested_shell_model_from_ifaces(if2, mats, W0)

    t0 = time.time()
    fresh = m2.assemble_blocks(w)
    t_fresh = time.time() - t0
    t0 = time.time()
    cached, nnew = cached_blocks(m2, w, cache)
    t_cached = time.time() - t0

    print("[rung-3 block cache]", flush=True)
    same_keys = set(cached) == set(fresh)
    bitwise = same_keys and all(np.array_equal(cached[k], fresh[k])
                                for k in fresh)
    ntouch = sum(1 for (rk, ck) in fresh
                 if rk[1] is if2[1] or ck[1] is if2[1])
    check("cached blocks bitwise == fresh", bitwise,
          f"  (recomputed {nnew}/{len(fresh)} blocks, expected "
          f"{ntouch}; fresh {t_fresh:.1f} s, cached {t_cached:.1f} s)")
    check("only perturbed-boundary blocks recomputed", nnew == ntouch)

    field = source_field(faces[-1], w, mats[-1])
    b = m2.assemble_rhs(incident_outer_source(m2, if2, field))
    elim = ShellElimination(m2, if2)
    x_f = elim.solve(blocks=fresh, b=b, full=True)
    x_c = elim.solve(blocks=cached, b=b, full=True)
    check("cached solution bitwise == fresh", np.array_equal(x_c, x_f))


def gate_thin_shell():
    """Transparent double interface: solid sphere R=20000 with two
    same-material internal boundaries at r1 and r1+t; surface field
    must match the single-region solve. h is the mean face size of the
    r1 mesh; sweep t/h down from ~1.2 to ~0.15."""
    w = 2 * np.pi * F_HZ + 0.08j
    mat = layer_materials([layer(0, 0, SOLID)])[0]
    rc, nm = 13000.0, 18   # thin shell CENTERED at rc: both bounding
    # radii move symmetrically so the gaps to the surface (7000 - t/2)
    # and to the center stay thick throughout the sweep

    u_ref, field, h = None, None, None
    print("[thin-shell probe]", flush=True)
    results = []
    for t in (5000.0, 4000.0, 2000.0, 1000.0, 500.0):
        layers = [layer(rc - t / 2, nm, SOLID),
                  layer(rc + t / 2, nm, SOLID),
                  layer(20000.0, 25, SOLID)]
        model, ifaces, faces, mats = build(layers)
        if u_ref is None:
            # the surface mesh is t-independent (same rng seed/params)
            field = source_field(faces[-1], w, mat)
            geom = Geometry(faces[-1], self_scheme="polar",
                            quad_mode="adaptive")
            T = cal_traction(faces[-1], w, mat.lamda, mat.mu, mat.rho,
                             mat.Q, qp_fac=mat.qp_fac, geom=geom)
            u_ref = np.linalg.solve(T, field(faces[-1]))
        h = float(np.sqrt(np.mean(faces[0].area)))
        b = model.assemble_rhs(
            incident_outer_source(model, ifaces, field))
        x = ShellElimination(model, ifaces).solve(w=w, b=b, full=False)
        err = np.linalg.norm(x - u_ref) / np.linalg.norm(u_ref)
        results.append((t / h, err))
        print(f"    t = {t:6.0f} m  t/h = {t / h:5.2f}  "
              f"surface rel err {err:.3e}", flush=True)
    thick = [e for r, e in results if r >= 1.0]
    check("t/h >= 1 stays in the transparent-interface class",
          max(thick) < THIN_GATE,
          f"  (max {max(thick):.3e}, gate {THIN_GATE:g})")


def gate_source_layer():
    """Incident placement for a source in an arbitrary solid layer.
    (a) for the outermost layer the general function reproduces
    incident_outer_source BITWISE; (b) transparent 3-shell with the
    source INSIDE THE MIDDLE shell must reproduce the single-region
    solution (rung-2e bug class: the old path put the incident field
    in the wrong region's representation equation)."""
    w = 2 * np.pi * F_HZ + 0.08j
    mat = layer_materials([layer(0, 0, SOLID)])[0]
    layers = [layer(7000.0, 12, SOLID), layer(15000.0, 24, SOLID),
              layer(20000.0, 25, SOLID)]
    model, ifaces, faces, mats = build(layers)

    print("[source-layer placement]", flush=True)
    f_out = source_field(faces[-1], w, mat)
    inc_a = incident_outer_source(model, ifaces, f_out)
    inc_b = incident_solid_layer_source(model, ifaces, len(ifaces) - 1,
                                        f_out)
    same = set(inc_a) == set(inc_b) and all(
        np.array_equal(inc_a[k], inc_b[k]) for k in inc_a)
    check("outermost layer == incident_outer_source (bitwise)", same)

    xs = -11000.0    # inside the middle shell (7000 < r < 15000)

    def f_mid(faces):
        return u0e(faces, w, mat.rho, mat.mu, mat.lamda, xs, 0.0, 0.0,
                   mat.Q, FSRC, qp_fac=mat.qp_fac)

    b = model.assemble_rhs(
        incident_solid_layer_source(model, ifaces, 1, f_mid))
    x = ShellElimination(model, ifaces).solve(w=w, b=b, full=False)
    geom = Geometry(faces[-1], self_scheme="polar", quad_mode="adaptive")
    T = cal_traction(faces[-1], w, mat.lamda, mat.mu, mat.rho, mat.Q,
                     qp_fac=mat.qp_fac, geom=geom)
    u_ref = np.linalg.solve(T, f_mid(faces[-1]))
    err = np.linalg.norm(x - u_ref) / np.linalg.norm(u_ref)
    check("middle-shell source vs single-region", err < 8e-2,
          f"  (rel err {err:.3e}, gate 8e-2)")


def main():
    gate_identity()
    gate_cache()
    gate_thin_shell()
    gate_source_layer()
    if FAILED:
        print(f"elimination gate FAILED: {FAILED}")
        sys.exit(1)
    print("elimination gate PASSED")


if __name__ == "__main__":
    main()
