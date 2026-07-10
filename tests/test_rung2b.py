#!/usr/bin/env python3
"""Rung-2b gate (docs/multilayer_roadmap.md): fluid annuli + auto-mesh
+ boundary perturbation.

Gates:
  1. validation — solid/fluid/solid builds (two p blocks, no t
     blocks); fluid outermost rejected; adjacent fluids rejected;
  2. tiny-core limit — solid(r->small)/fluid/solid must approach the
     2-region liquid-core solution (the tiny inner solid scatters
     negligibly, so the difference is discretization only);
  3. annulus refinement — genuine solid/fluid/solid: refining the two
     INTERNAL meshes (surface fixed) must move the surface field
     toward the finest level (Cauchy convergence). Per the campaign
     rule: if this does not converge, hunt for bugs before any DSM
     run.
  4. auto_nmesh — wavelength rule reproduces the Earth-campaign
     surface mesh (~200) and respects overrides / fluid vp;
  5. perturbation — amp=0 relief is BITWISE the unperturbed sphere
     (strict no-op gate); solution change scales ~linearly with amp.

Bitwise anchor: rung-0/1/2 gates must be rerun alongside this file
(battery: test_rung0.py && test_rung1.py && test_rung2.py &&
test_rung2b.py) — test_rung0 compares against the original
cal_traction/liq_core solvers and catches any bit drift from the
sign-generalized assembly expressions.

Run on a compute node (small synthetic meshes, ~5 min).
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.domains import (Material, nested_shell_model)   # noqa: E402
from pyastroseis.layered import (auto_nmesh, incident_outer_source,
                                 layer_materials, layer_meshes)  # noqa: E402
from pyastroseis.meshgen import gen_layer, gen_mesh_relief, \
    relief_ylm                                                   # noqa: E402
from pyastroseis.source import u0e                               # noqa: E402

FAILED = []
FSRC = [1.0, 1.0, 1.0]
W0 = 2 * np.pi * 0.3
F_HZ = 0.05

TINY_CORE = 8e-2          # LC scheme class + tiny extra boundary
REFINE_RATIO = 0.75       # err(fine) / err(coarse) must be below this
PERTURB_LIN = (1.4, 2.7)  # err(2a)/err(a) window around linear (2.0)

SOLID = {"vp": 6000.0, "vs": 3000.0, "rho": 3000.0, "Q": 200.0}
CORE = {"vp": 8000.0, "vs": 4500.0, "rho": 4000.0, "Q": 500.0}
FLUID = {"vp": 7000.0, "vs": 0.0, "rho": 3500.0, "Q": 200.0}

MAT_S = layer_materials([dict(r=0, nmesh=0, **SOLID)])[0]
MAT_C = layer_materials([dict(r=0, nmesh=0, **CORE)])[0]
MAT_F = layer_materials([dict(r=0, nmesh=0, **FLUID)])[0]


def check(label, ok, detail=""):
    print(f"    {label}: {'OK' if ok else 'FAIL'}{detail}", flush=True)
    if not ok:
        FAILED.append(label)


def sphere(r, nmesh, rng_seed):
    rng = np.random.default_rng(rng_seed)
    return gen_layer((r,), (nmesh,), (0,), rng=rng)[0][0]


def source_field(faces_ref, w, mat):
    R = np.mean(np.sqrt((faces_ref.ic ** 2).sum(axis=1)))
    xs = -(R - 2e3)

    def field(faces):
        return u0e(faces, w, mat.rho, mat.mu, mat.lamda, xs, 0.0, 0.0,
                   mat.Q, FSRC, qp_fac=mat.qp_fac)
    return field


def surface_u(faces_list, mats, field):
    w = 2 * np.pi * F_HZ + 0.08j
    model, ifaces = nested_shell_model(faces_list, mats, W0)
    x = model.solve(w, incident_outer_source(model, ifaces, field))
    return x[model.block_slice("u", ifaces[-1])], model


def gate_validation(f_core, f_mid, f_surf):
    print("[validation]", flush=True)
    model, _ = nested_shell_model([f_core, f_mid, f_surf],
                                  [MAT_C, MAT_F, MAT_S], W0)
    kinds = [k for k, _ in model.blocks]
    check("solid/fluid/solid builds: 2 p blocks, no t blocks",
          kinds.count("p") == 2 and kinds.count("t") == 0
          and kinds.count("u") == 3)

    def expects(exc, fn, label):
        try:
            fn()
        except exc:
            check(label, True)
        else:
            check(label, False)

    expects(NotImplementedError, lambda: nested_shell_model(
        [f_core, f_surf], [MAT_S, MAT_F], W0),
        "fluid outermost rejected")
    expects(NotImplementedError, lambda: nested_shell_model(
        [f_core, f_mid, f_surf], [MAT_F, MAT_F, MAT_S], W0),
        "adjacent fluid layers rejected")


def gate_tiny_core(f_core, f_surf):
    w = 2 * np.pi * F_HZ + 0.08j
    field = source_field(f_surf, w, MAT_S)
    t0 = time.time()
    f_tiny = sphere(1500.0, 6, 11)
    u_ann, _ = surface_u([f_tiny, f_core, f_surf],
                         [MAT_S, MAT_F, MAT_S], field)
    u_lc, _ = surface_u([f_core, f_surf], [MAT_F, MAT_S], field)
    err = np.linalg.norm(u_ann - u_lc) / np.linalg.norm(u_lc)
    print(f"[tiny-core limit] inner n={f_tiny.n} ({time.time() - t0:.1f} s)",
          flush=True)
    check("solid(tiny)/fluid/solid vs liquid core", err < TINY_CORE,
          f"  (rel err {err:.3e}, gate {TINY_CORE:g})")


def gate_annulus_refine(f_surf):
    w = 2 * np.pi * F_HZ + 0.08j
    field = source_field(f_surf, w, MAT_S)
    print("[annulus refinement] internal meshes x1/x2/x4, surface fixed",
          flush=True)
    levels = []
    for mult in (1, 2, 4):
        t0 = time.time()
        f_in = sphere(8000.0, 12 * mult, 21)
        f_out = sphere(14000.0, 18 * mult, 22)
        u, model = surface_u([f_in, f_out, f_surf],
                             [MAT_C, MAT_F, MAT_S], field)
        levels.append(u)
        print(f"    x{mult}: internal n=({f_in.n},{f_out.n}) "
              f"size={model.size} ({time.time() - t0:.1f} s)", flush=True)
    ref = levels[-1]
    e1 = np.linalg.norm(levels[0] - ref) / np.linalg.norm(ref)
    e2 = np.linalg.norm(levels[1] - ref) / np.linalg.norm(ref)
    check("surface field converges under internal refinement",
          e2 < REFINE_RATIO * e1,
          f"  (err x1 {e1:.3e} -> x2 {e2:.3e}, need ratio < "
          f"{REFINE_RATIO:g})")


def gate_auto_nmesh():
    print("[auto_nmesh]", flush=True)
    # Earth-scale campaign check: shell vs 3 km/s, band top 5.26e-4 Hz,
    # 10 elements/wavelength gave the n=200 (1584-face) surface mesh
    la = [dict(SOLID, r=6371e3, vs=3000.0)]
    n = auto_nmesh(la, fmax=5.26e-4, epw=10.0)[0]
    check("Earth campaign surface ~200", 185 <= n <= 215, f"  (n={n})")
    # fluid-solid boundary = reflector: curvature floor h/R <= 0.09
    # dominates when the wavelength rule is loose
    lay2 = [dict(r=3480e3, vp=8000.0, vs=0.0, rho=10000.0, Q=100.0),
            dict(r=6371e3, vp=10000.0, vs=5500.0, rho=4500.0, Q=300.0)]
    n_cmb, n_srf = auto_nmesh(lay2, fmax=5.26e-4, epw=10.0)
    F_wave = 4.0 * np.pi * 3480e3 ** 2 \
        / (5500.0 / 5.26e-4 / 10.0) ** 2   # min(vp_fl, vs_sh) = 5500
    F_curv = 4.0 * np.pi / 0.09 ** 2
    check("CMB reflector: max(wavelength, curvature h/R<=0.09)",
          n_cmb == max(12, int(np.ceil((max(F_wave, F_curv) + 16.0)
                                       / 8.0))),
          f"  (n_cmb={n_cmb})")
    # welded internal boundary: transmissive tier h/R <= 0.2
    lay3 = [dict(r=1221.5e3, vp=7000.0, vs=3500.0, rho=6000.0, Q=500.0),
            dict(r=6371e3, vp=6000.0, vs=3000.0, rho=3000.0, Q=200.0)]
    n_w = auto_nmesh(lay3, fmax=5.26e-4, epw=10.0)[0]
    F_wcurv = 4.0 * np.pi / 0.2 ** 2
    F_wwave = 4.0 * np.pi * 1221.5e3 ** 2 \
        / (3000.0 / 5.26e-4 / 10.0) ** 2
    check("welded boundary uses h/R<=0.2 tier",
          n_w == max(12, int(np.ceil((max(F_wwave, F_wcurv) + 16.0)
                                     / 8.0))),
          f"  (n_welded={n_w})")
    n_ovr = auto_nmesh([dict(lay2[0], nmesh=99), lay2[1]],
                       fmax=5.26e-4)[0]
    check("explicit nmesh override kept", n_ovr == 99)


def gate_perturb(f_surf_ref):
    print("[perturbation]", flush=True)
    rng_seed = 2
    f0, _ = gen_mesh_relief(20000.0, 25, relief_ylm(2, 0, 0.0),
                            rng=np.random.default_rng(rng_seed))
    fields = ("A", "B", "C", "nvec", "ic", "area", "r", "a", "b", "c")
    same = all(np.array_equal(getattr(f0, k), getattr(f_surf_ref, k))
               for k in fields)
    check("amp=0 relief BITWISE == unperturbed sphere", same)

    w = 2 * np.pi * F_HZ + 0.08j
    field = source_field(f_surf_ref, w, MAT_S)
    errs = []
    for amp in (100.0, 200.0):
        fp, _ = gen_mesh_relief(20000.0, 25, relief_ylm(2, 0, amp),
                                rng=np.random.default_rng(rng_seed))
        check(f"perturbed mesh outward-closed (amp={amp:g})",
              np.median(np.sum(fp.nvec * fp.ic, axis=1)) > 0)
        u_p, _ = surface_u([fp], [MAT_S], field)
        u_0, _ = surface_u([f_surf_ref], [MAT_S], field)
        errs.append(np.linalg.norm(u_p - u_0) / np.linalg.norm(u_0))
    ratio = errs[1] / errs[0]
    check("solution change ~linear in amp",
          PERTURB_LIN[0] < ratio < PERTURB_LIN[1],
          f"  (err {errs[0]:.3e} -> {errs[1]:.3e}, ratio {ratio:.2f})")


def main():
    f_core = sphere(8000.0, 12, 1)
    f_mid = sphere(14000.0, 18, 22)
    f_surf = sphere(20000.0, 25, 2)

    gate_validation(f_core, f_mid, f_surf)
    gate_auto_nmesh()
    gate_perturb(f_surf)
    gate_tiny_core(f_core, f_surf)
    gate_annulus_refine(f_surf)

    if FAILED:
        print(f"rung-2b gate FAILED: {FAILED}")
        sys.exit(1)
    print("rung-2b gate PASSED")


if __name__ == "__main__":
    main()
