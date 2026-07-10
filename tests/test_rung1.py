#!/usr/bin/env python3
"""Rung-1 gate (docs/multilayer_roadmap.md): welded solid-solid
interfaces.

Gates:
  1. transparent interface — a homogeneous sphere split by an
     artificial internal welded boundary (same material both sides)
     must reproduce the single-region solution on the outer surface;
     this catches any sign/orientation/jump-term error in the welded
     blocks (a relative sign error between the two sides breaks it);
  2. A/B swap — exchanging which solid is registered first (u-rows)
     and second (t-rows) permutes the equations and changes the
     traction column scaling; the physical fields must agree to
     solver roundoff;
  3. interface refinement — with a real material contrast the surface
     solution must converge as the interface mesh is refined;
  4. build-time validation of welded configuration errors.

Run together with tests/test_rung0.py (which must stay bitwise).
Default: synthetic spheres. --full adds the real my_mesh_lc meshes.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.assembly import Geometry, cal_traction      # noqa: E402
from pyastroseis.domains import (FREE, WELDED, Interface, Material,
                                 MultiDomainModel, Region)   # noqa: E402
from pyastroseis.liquidcore import (flip_normals,
                                    load_layers_mat)         # noqa: E402
from pyastroseis.meshgen import gen_layer                    # noqa: E402
from pyastroseis.source import u0e                           # noqa: E402

FAILED = []
FSRC = [1.0, 1.0, 1.0]
W0 = 2 * np.pi * 0.3

# gate thresholds (set from observed values, ~2x margin). The
# transparent error is interface-discretization error (piecewise-
# constant elements), NOT solver error: it must CONVERGE to zero with
# interface refinement (gated below); the absolute values just catch
# regressions at the reference resolutions.
TRANSPARENT_SYN = 2.5e-2      # n2=80,  f=0.05 (observed 1.27e-2)
TRANSPARENT_FULL = 1e-2       # n2=384, f=0.3  (observed 4.47e-3)
SWAP_TOL = 1e-9


def phys_solid(vp, vs, rho, Q):
    return Material.solid(vp, vs, rho, Q, qp_fac=0.75 * (vp / vs) ** 2)


MAT_SHELL = phys_solid(6000.0, 3000.0, 3000.0, 200.0)
MAT_CORE_CONTRAST = phys_solid(8000.0, 4500.0, 4000.0, 500.0)


def check(label, ok, detail=""):
    print(f"    {label}: {'OK' if ok else 'FAIL'}{detail}", flush=True)
    if not ok:
        FAILED.append(label)


def outward(faces):
    if np.median(np.sum(faces.nvec * faces.ic, axis=1)) < 0:
        return flip_normals(faces)
    return faces


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def source_in_shell(face1):
    R = np.mean(np.sqrt((face1.ic ** 2).sum(axis=1)))
    return -(R - 2e3), 0.0, 0.0


def welded_solve(face1, face2, mat_shell, mat_core, w, src, order="ab"):
    """Two-layer welded solve; order chooses which solid is registered
    first (owns the ("u", core) rows). Source is in the shell, so the
    incident field lands on the shell's equation rows."""
    surf = Interface(face1, FREE, "surface")
    core = Interface(face2, WELDED, "core")
    shell = Region(mat_shell, ((core, -1), (surf, +1)), "shell")
    inner = Region(mat_core, ((core, +1),), "core")
    regions = (shell, inner) if order == "ab" else (inner, shell)
    model = MultiDomainModel(regions, w0=W0, self_scheme="polar",
                             quad_mode="adaptive")
    m = mat_shell
    xs, ys, zs = src
    u01 = u0e(face1, w, m.rho, m.mu, m.lamda, xs, ys, zs, m.Q, FSRC,
              qp_fac=m.qp_fac)
    u02 = u0e(face2, w, m.rho, m.mu, m.lamda, xs, ys, zs, m.Q, FSRC,
              qp_fac=m.qp_fac)
    z = np.zeros(3 * face2.n, dtype=complex)
    inc = {("u", surf): u01, ("u", core): u02 if order == "ab" else z,
           ("t", core): z if order == "ab" else u02}
    x = model.solve(w, inc)
    return model, surf, core, x


def single_region_ref(face1, f_hz):
    m = MAT_SHELL
    src = source_in_shell(face1)
    w = 2 * np.pi * f_hz + 0.08j
    geom = Geometry(face1, self_scheme="polar", quad_mode="adaptive")
    T = cal_traction(face1, w, m.lamda, m.mu, m.rho, m.Q,
                     qp_fac=m.qp_fac, geom=geom)
    u01 = u0e(face1, w, m.rho, m.mu, m.lamda, *src, m.Q, FSRC,
              qp_fac=m.qp_fac)
    return np.linalg.solve(T, u01)


def transparent_err(face1, face2, f_hz, u_ref):
    m = MAT_SHELL
    src = source_in_shell(face1)
    w = 2 * np.pi * f_hz + 0.08j
    model, surf, core, x = welded_solve(face1, face2, m, m, w, src)
    return rel(x[model.block_slice("u", surf)], u_ref)


def gate_transparent(face1, face2, f_hz, tag, thresh):
    t0 = time.time()
    u_ref = single_region_ref(face1, f_hz)
    err = transparent_err(face1, face2, f_hz, u_ref)
    print(f"[transparent/{tag}] n1={face1.n} n2={face2.n} f={f_hz:g} Hz "
          f"({time.time() - t0:.1f} s)", flush=True)
    check("surface field vs single-region", err < thresh,
          f"  (rel err {err:.3e}, gate {thresh:g})")


def gate_transparent_refine(face1, cores, f_hz):
    """The decisive correctness statement: the transparent-interface
    error must converge to zero as the artificial interface refines
    (a sign/orientation/jump bug would leave an O(1) floor)."""
    t0 = time.time()
    u_ref = single_region_ref(face1, f_hz)
    errs = [transparent_err(face1, fc, f_hz, u_ref) for fc in cores]
    print(f"[transparent-refine/synthetic] cores n={[f.n for f in cores]} "
          f"f={f_hz:g} Hz ({time.time() - t0:.1f} s)", flush=True)
    print("    transparent err: "
          + "  ".join(f"{e:.3e}" for e in errs), flush=True)
    check("transparent error converges",
          all(b < a for a, b in zip(errs, errs[1:])))


def gate_swap(face1, face2, f_hz, tag):
    src = source_in_shell(face1)
    w = 2 * np.pi * f_hz + 0.08j
    t0 = time.time()
    ma, sa, ca, xa = welded_solve(face1, face2, MAT_SHELL,
                                  MAT_CORE_CONTRAST, w, src, order="ab")
    mb, sb, cb, xb = welded_solve(face1, face2, MAT_SHELL,
                                  MAT_CORE_CONTRAST, w, src, order="ba")
    print(f"[swap/{tag}] n1={face1.n} n2={face2.n} f={f_hz:g} Hz "
          f"({time.time() - t0:.1f} s)", flush=True)
    for lbl, ia, ib in (("u_surf", sa, sb), ("u_core", ca, cb)):
        e = rel(xa[ma.block_slice("u", ia)], xb[mb.block_slice("u", ib)])
        check(f"{lbl} A/B-swap", e < SWAP_TOL, f"  (rel {e:.3e})")
    ta = ma.traction_scale(ca) * xa[ma.block_slice("t", ca)]
    tb = mb.traction_scale(cb) * xb[mb.block_slice("t", cb)]
    e = rel(ta, tb)
    check("t_core A/B-swap (physical)", e < SWAP_TOL, f"  (rel {e:.3e})")


def gate_refine(face1, cores, f_hz):
    src = source_in_shell(face1)
    w = 2 * np.pi * f_hz + 0.08j
    t0 = time.time()
    sols = []
    for fc in cores:
        model, surf, core, x = welded_solve(face1, fc, MAT_SHELL,
                                            MAT_CORE_CONTRAST, w, src)
        sols.append(x[model.block_slice("u", surf)])
    errs = [rel(s, sols[-1]) for s in sols[:-1]]
    print(f"[refine/synthetic] cores n={[f.n for f in cores]} "
          f"f={f_hz:g} Hz ({time.time() - t0:.1f} s)", flush=True)
    print(f"    surface-field err vs finest core: "
          + "  ".join(f"{e:.3e}" for e in errs), flush=True)
    check("interface refinement converges", errs[1] < errs[0])


def gate_validation(face1, face2):
    solid = MAT_SHELL

    def expects(exc, fn, label):
        try:
            fn()
        except exc:
            check(label, True)
        else:
            check(label, False)

    print("[validation]", flush=True)
    core = Interface(face2, WELDED, "core")
    surf = Interface(face1, FREE, "surface")
    expects(ValueError, lambda: MultiDomainModel(
        (Region(solid, ((core, -1), (surf, +1))),), w0=1.0),
        "one-sided welded rejected")
    expects(ValueError, lambda: MultiDomainModel(
        (Region(solid, ((core, +1), (surf, +1))),
         Region(solid, ((core, +1),))), w0=1.0),
        "same-sign welded rejected")
    expects(ValueError, lambda: MultiDomainModel(
        (Region(solid, ((core, -1), (surf, +1))),
         Region(solid, ((core, +1),)))),
        "missing w0 rejected for welded")


def main():
    full = "--full" in sys.argv

    rng = np.random.default_rng(1)
    layers = gen_layer((10000.0, 20000.0), (12, 25), (0, 0), rng=rng)
    core12 = outward(layers[0][0])
    face1s = outward(layers[1][0])
    core25 = outward(gen_layer((10000.0,), (25,), (0,),
                               rng=np.random.default_rng(2))[0][0])
    core40 = outward(gen_layer((10000.0,), (40,), (0,),
                               rng=np.random.default_rng(3))[0][0])

    gate_transparent(face1s, core12, 0.05, "synthetic", TRANSPARENT_SYN)
    gate_transparent_refine(face1s, (core12, core25, core40), 0.05)
    gate_swap(face1s, core12, 0.05, "synthetic")
    gate_refine(face1s, (core12, core25, core40), 0.05)
    gate_validation(face1s, core12)

    if full:
        lay = load_layers_mat(os.path.join(ROOT, "my_mesh_lc.mat"))
        face2, face1 = lay[0], lay[1]   # core, free surface
        gate_transparent(face1, face2, 0.3, "my_mesh_lc",
                         TRANSPARENT_FULL)
        gate_swap(face1, face2, 0.3, "my_mesh_lc")

    if FAILED:
        print(f"rung-1 gate FAILED: {FAILED}")
        sys.exit(1)
    print("rung-1 gate PASSED" + ("" if full else " (synthetic only; "
          "run with --full on a compute node for the real meshes)"))


if __name__ == "__main__":
    main()
