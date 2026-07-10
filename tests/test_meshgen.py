#!/usr/bin/env python3
"""Functional tests for the mesh-generation port.

The particle optimizer is stochastic, so (unlike the kernel oracles)
these are geometric/consistency checks, not bitwise comparisons:
closed-surface topology, positive areas, outward normals, sphere
exactness at nfold=0, the Phobos-radius behavior of the original code,
and within-Python determinism under a fixed seed.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pyastroseis.mesh import load_faces_mat  # noqa: E402
from pyastroseis.meshgen import (gen_mesh_ph_topo, gen_mesh_topo_layers,
                                 particle_sample_sphere,
                                 subdivide_spherical_mesh)  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    print(f"  [{'ok ' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILED.append(name)


def euler(V, Tri):
    edges = set()
    for f in Tri:
        f0 = f - 1
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((f0[a], f0[b]))))
    return len(V) - len(edges) + len(Tri)


def main():
    print("particle_sample_sphere (N=50, seed 0):")
    V, Tri, Ue_i, Ue = particle_sample_sphere(N=50, rng=0)
    check("unit sphere", np.allclose(np.linalg.norm(V, axis=1), 1, atol=1e-12))
    check("energy decreased", Ue[-1] < Ue[0],
          f"(E0={Ue[0]:.1f} -> {Ue[-1]:.1f})")
    check("closed surface", euler(V, Tri) == 2, f"(chi={euler(V, Tri)})")

    V2, Tri2, _, _ = particle_sample_sphere(N=50, rng=0)
    check("deterministic w/ seed", np.array_equal(V, V2)
          and np.array_equal(Tri, Tri2))

    Fs, Vs = subdivide_spherical_mesh(Tri, V, 1)
    check("subdivision x4 faces", Fs.shape[0] == 4 * Tri.shape[0])
    check("subdivided closed", euler(Vs, Fs) == 2)

    print("gen_mesh_ph_topo (Phobos, n=50, nfold=1, matlab radius mode):")
    faces, extras = gen_mesh_ph_topo(20000, 50, 1, rng=0)
    rmean = np.linalg.norm(faces.ic, axis=1).mean()
    check("areas positive", (faces.area > 0).all())
    out = np.einsum("ij,ij->i", faces.nvec, faces.ic)
    check("normals outward", (out > 0).all())
    check("Phobos double-radius quirk", 20e3 < rmean < 24e3,
          f"(mean r = {rmean:.0f} m; my_mesh.mat has 21918)")
    ref = load_faces_mat(os.path.join(ROOT, "my_mesh.mat"))
    q_new = np.quantile(faces.a / faces.a.mean(), [0.1, 0.9])
    q_ref = np.quantile(ref.a / ref.a.mean(), [0.1, 0.9])
    check("element-size spread ~ MATLAB mesh",
          np.all(np.abs(q_new - q_ref) < 0.35),
          f"(norm-size q10/q90: {q_new.round(2)} vs {q_ref.round(2)})")

    print("gen_mesh_topo_layers (sphere, nfold=0):")
    faces0, extras0 = gen_mesh_topo_layers(10000, 50, 0, rng=0)
    rv = np.linalg.norm(extras0["V"], axis=1)
    check("exact sphere at nfold=0", np.allclose(rv, 10000, rtol=1e-12),
          f"(max dev {np.abs(rv - 10000).max():.2e} m)")

    print("gen_mesh_topo_layers (Mars-like topo):")
    # NOTE: the original topo scaling (topos*20*1e3 m) is huge at
    # nfold=1 — the shipped examples all use nfold=0. Test the plumbing
    # at a small nfold and verify linear scaling in nfold.
    faces1, extras1 = gen_mesh_topo_layers(10000, 50, 1e-3, rng=0)
    rv1 = np.linalg.norm(extras1["V"], axis=1) - 10000
    faces2b, extras2b = gen_mesh_topo_layers(10000, 50, 2e-3, rng=0)
    rv2 = np.linalg.norm(extras2b["V"], axis=1) - 10000
    check("topo present", np.abs(rv1).max() > 10,
          f"(max |h| = {np.abs(rv1).max():.0f} m at nfold=1e-3)")
    check("topo linear in nfold", np.allclose(rv2, 2 * rv1, atol=1e-6))

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)}: {FAILED}")
        sys.exit(1)
    print("all meshgen checks passed")


if __name__ == "__main__":
    main()
