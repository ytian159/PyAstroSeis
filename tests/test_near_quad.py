#!/usr/bin/env python3
"""Near-singular quadrature tier + graded epicentral refinement gate
(docs/fast_methods_notes.md section 3; the 50-km-source rung).

Gates:
  N1 single-panel near-field integrals: for a collocation point at
     dist/h 0.28-1.0 above a flat panel, the composite near rule
     (10, 2) must hit the converged reference (composite (10, 5)) to
     <= 1e-3 relative on both acoustic kernels, and beat the default
     deg-10 rule by >= 10x at the closest point;
  N2 refine_toward sanity: log-count growth, hmin floor respected,
     vertices stay on the sphere, cap faces reach hmin size;
  N3 end-to-end transparency: a deep source on a uniform mesh vs the
     SAME model on a mesh refined toward an (unnecessary) surface cap
     with the near tier active — matched far-side faces must agree in
     the transparent-interface class; plus near-tier-only on the
     uniform mesh must be a small quadrature correction.

Run on a compute node (~ a couple of minutes).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.quadrature import (simplex_rule,
                                    simplex_rule_composite)      # noqa: E402
from pyastroseis.greens import greens_func_p, greens_func_deri_p  # noqa: E402
from pyastroseis.assembly import wave_speeds, Geometry, cal_traction  # noqa: E402
from pyastroseis.mesh import faces_from_vertices                 # noqa: E402
from pyastroseis.meshgen import refine_toward                    # noqa: E402
from pyastroseis.layered import layer_materials, layer_meshes    # noqa: E402
from pyastroseis.source import u0e                               # noqa: E402

FAILED = []
F_HZ = 0.05
SOLID = {"vp": 6000.0, "vs": 3000.0, "rho": 3000.0, "Q": 200.0}


def layer(r, nmesh, mat):
    return dict(r=r, nmesh=nmesh, **mat)


def check(label, ok, detail=""):
    print(f"    {label}: {'OK' if ok else 'FAIL'}{detail}", flush=True)
    if not ok:
        FAILED.append(label)


def _panel_integral(rule, tri, xs, ys, zs, vp, w, kernel):
    ref, wi = rule
    A, B, C = tri
    lam0 = 1.0 - ref[0] - ref[1]
    xi = A[0] * lam0 + B[0] * ref[0] + C[0] * ref[1]
    yi = A[1] * lam0 + B[1] * ref[0] + C[1] * ref[1]
    zi = A[2] * lam0 + B[2] * ref[0] + C[2] * ref[1]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A))
    if kernel == "B":
        v = greens_func_p(vp, w, xi, yi, zi, xs, ys, zs)
    else:
        v = greens_func_deri_p(vp, w, xi, yi, zi, xs, ys, zs,
                               0.0, 0.0, 1.0)
    return (v * wi).sum() * area * 2.0


def gate_panel():
    w = 2 * np.pi * F_HZ + 0.08j
    mat = layer_materials([layer(0, 0, SOLID)])[0]
    vp, _ = wave_speeds(mat.lamda, mat.mu, mat.rho, mat.Q,
                        qp_fac=mat.qp_fac)
    h = 5000.0
    tri = (np.array([0.0, 0.0, 0.0]), np.array([h, 0.0, 0.0]),
           np.array([0.0, h, 0.0]))
    r10 = simplex_rule(10)
    rnear = simplex_rule_composite(10, 2)
    rref = simplex_rule_composite(10, 5)

    print("[N1 near-singular panel integrals]", flush=True)
    worst_gain = np.inf
    for frac in (0.28, 0.5, 1.0):
        d = frac * h
        xs, ys, zs = 1200.0, 900.0, d
        for kern in ("B", "A"):
            I10 = _panel_integral(r10, tri, xs, ys, zs, vp, w, kern)
            In = _panel_integral(rnear, tri, xs, ys, zs, vp, w, kern)
            Ir = _panel_integral(rref, tri, xs, ys, zs, vp, w, kern)
            e10 = abs(I10 - Ir) / abs(Ir)
            en = abs(In - Ir) / abs(Ir)
            print(f"    d/h {frac:4.2f} kern {kern}: deg10 {e10:.3e}  "
                  f"near(10,2) {en:.3e}", flush=True)
            check(f"near rule converged (d/h {frac}, {kern})", en < 1e-3,
                  f"  ({en:.3e})")
            if frac == 0.28:
                worst_gain = min(worst_gain,
                                 e10 / max(en, np.finfo(float).tiny))
    check("near rule >= 10x better than deg10 at d/h 0.28",
          worst_gain >= 10.0, f"  (gain {worst_gain:.1f}x)")


def gate_refine():
    print("[N2 refine_toward sanity]", flush=True)
    R = 20000.0
    faces = layer_meshes([layer(R, 25, SOLID)], seed=1)[0]
    tgt = np.array([0.0, 0.0, R])
    hmin = 700.0
    ref = refine_toward(faces, tgt, hmin, grade=1.0)
    hmax = np.maximum.reduce([
        np.linalg.norm(ref.A - ref.B, axis=1),
        np.linalg.norm(ref.B - ref.C, axis=1),
        np.linalg.norm(ref.C - ref.A, axis=1)])
    d = np.linalg.norm((ref.A + ref.B + ref.C) / 3.0 - tgt, axis=1)
    ok_rule = np.all(hmax <= np.maximum(1.0 * d, hmin) + 1e-9)
    check("h <= max(grade*dist, hmin) everywhere", bool(ok_rule))
    radii = np.linalg.norm(
        np.vstack([ref.A, ref.B, ref.C]), axis=1)
    check("vertices stay on the sphere",
          float(np.abs(radii - R).max()) < 1e-6 * R,
          f"  (max dev {np.abs(radii - R).max():.2e} m)")
    grow = ref.n - faces.n
    check("log-count growth (< 4x base faces added)",
          grow < 4 * faces.n, f"  ({faces.n} -> {ref.n} faces)")
    cap = d < 3 * hmin
    check("cap faces reach hmin size",
          bool(cap.any() and hmax[cap].max() <= max(1.0 * 3 * hmin, hmin)
               and hmax[cap].min() < hmin),
          f"  (cap n={int(cap.sum())}, h {hmax[cap].min():.0f}"
          f"-{hmax[cap].max():.0f} m)")


def _solve_surface(faces, w, mat, xs, near=None):
    geom = Geometry(faces, self_scheme="polar", quad_mode="adaptive",
                    near_tier=near)
    T = cal_traction(faces, w, mat.lamda, mat.mu, mat.rho, mat.Q,
                     qp_fac=mat.qp_fac, geom=geom)
    u0 = u0e(faces, w, mat.rho, mat.mu, mat.lamda, xs, 0.0, 0.0,
             mat.Q, [1.0, 1.0, 1.0], qp_fac=mat.qp_fac)
    return np.linalg.solve(T, u0)


def gate_transparency():
    """Refining where refinement is NOT needed (deep source) must not
    change the far-side answer; the near tier alone must be a small
    quadrature correction."""
    w = 2 * np.pi * F_HZ + 0.08j
    mat = layer_materials([layer(0, 0, SOLID)])[0]
    R = 20000.0
    faces = layer_meshes([layer(R, 25, SOLID)], seed=1)[0]
    xs = -10000.0        # deep source under (-R, 0, 0)
    near = (1.5, (10, 2))

    print("[N3 end-to-end transparency]", flush=True)
    u_base = _solve_surface(faces, w, mat, xs)
    u_tier = _solve_surface(faces, w, mat, xs, near=near)
    dq = np.linalg.norm(u_tier - u_base) / np.linalg.norm(u_base)
    check("near tier alone is a small quadrature correction",
          dq < 1e-2, f"  (rel change {dq:.3e})")

    tgt = np.array([-R, 0.0, 0.0])   # cap over the source longitude
    reff = refine_toward(faces, tgt, 700.0, grade=1.0)
    u_ref = _solve_surface(reff, w, mat, xs, near=near)
    # kept faces are geometrically identical: match by incenter
    n = faces.n
    keep = []
    for j in range(n):
        dist = np.linalg.norm(reff.ic - faces.ic[j], axis=1)
        i = int(np.argmin(dist))
        if dist[i] < 1.0:            # identical (kept) panel
            keep.append((j, i))
    jj = np.array([j for j, _ in keep])
    ii = np.array([i for _, i in keep])
    far = np.linalg.norm(faces.ic[jj] - tgt, axis=1) > 0.7 * R
    jj, ii = jj[far], ii[far]
    ub = np.stack([u_base[jj], u_base[jj + n], u_base[jj + 2 * n]])
    ur = np.stack([u_ref[ii], u_ref[ii + reff.n],
                   u_ref[ii + 2 * reff.n]])
    err = np.linalg.norm(ur - ub) / np.linalg.norm(ub)
    check("refined-vs-uniform far-side agreement", err < 8e-2,
          f"  (rel diff {err:.3e} on {jj.size} matched far faces; "
          f"{faces.n} -> {reff.n} faces)")


def main():
    gate_panel()
    gate_refine()
    gate_transparency()
    if FAILED:
        print(f"near-quadrature gate FAILED: {FAILED}")
        sys.exit(1)
    print("near-quadrature gate PASSED")


if __name__ == "__main__":
    main()
