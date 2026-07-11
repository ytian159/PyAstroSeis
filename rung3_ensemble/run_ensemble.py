#!/usr/bin/env python3
"""Rung 3 — CMB-topography perturbation ensemble with block caching.

Model: corefluid_q50 (solid inner core / fluid outer core / attenuated
mantle, the rung-2d DSM-arbitrated configuration). Ensemble members
perturb ONLY the CMB mesh (radial relief via meshgen.gen_mesh_relief,
same construction path as the base sphere, so amp=0 is bitwise the
base mesh). Per frequency the base model's blocks are assembled once;
each member reuses every block not touching the CMB Interface object
(elimination.cached_blocks — invalidation by object identity) and
re-solves through ShellElimination.

usage: run_ensemble.py [nprocs]        (output to R3_ROOT or here)

Gates (evaluated at the end, spectra domain):
  * amp=0 member BITWISE == base solution (whole cached pipeline is a
    strict no-op at zero relief);
  * Y20 response linear in amplitude: ||u(5 km) - u0|| == 2x
    ||u(2.5 km) - u0|| within [1.9, 2.1];
  * assembly speedup of cached vs fresh member reported (measured at
    the highest harmonic).
"""

import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("R3_ROOT", SCRIPT_DIR)
PKG = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PKG)
sys.path.insert(0, os.path.join(PKG, "dsm_arbitration"))

import make_inputs as mi                                     # noqa: E402
import run_bem as rb                                         # noqa: E402
from pyastroseis.domains import (Interface, nested_shell_model,
                                 nested_shell_model_from_ifaces)  # noqa: E402
from pyastroseis.elimination import (ShellElimination,
                                     cached_blocks)          # noqa: E402
from pyastroseis.layered import incident_outer_source        # noqa: E402
from pyastroseis.meshgen import (gen_layer, gen_mesh_relief,
                                 relief_random, relief_ylm)  # noqa: E402
from pyastroseis.source import u0eM                          # noqa: E402

SPEC = mi.LAYER_SPECS["corefluid_q50"]
PERT_IDX = 1                       # the CMB (layer 1's outer boundary)

MEMBERS = [
    ("amp0", relief_ylm(2, 0, 0.0)),
    ("y20_2500", relief_ylm(2, 0, 2500.0)),
    ("y20_5000", relief_ylm(2, 0, 5000.0)),
    ("rand_l4_5000", relief_random(4, 5000.0, 7)),
]

_W = {}


def base_mesh(r_km, nmesh):
    rng = np.random.default_rng(int(round(r_km)))
    return mi.outward(gen_layer((r_km * 1e3,), (nmesh,), (0,),
                                rng=rng)[0][0])


def pert_mesh(r_km, nmesh, relief):
    # same rng seeding as the base mesh: amp=0 relief is BITWISE the
    # base sphere (rung-2b no-op gate)
    rng = np.random.default_rng(int(round(r_km)))
    faces, _ = gen_mesh_relief(r_km * 1e3, nmesh, relief, rng=rng)
    return mi.outward(faces)


def rhs_matrix(model, ifaces, w):
    m = _W["mat_outer"]
    xs, ys, zs = _W["src_xyz"]
    B = np.empty((model.size, len(_W["mts"])), dtype=complex)
    for col, M in enumerate(_W["mts"]):
        def field(faces):
            return u0eM(faces, w, m.rho, m.mu, m.lamda, xs, ys, zs,
                        m.Q, M, qp_fac=m.qp_fac,
                        disp_ref_hz=m.disp_ref_hz)
        B[:, col] = model.assemble_rhs(
            incident_outer_source(model, ifaces, field))
    return B


def extract(US, n1):
    st = _W["st_faces"]
    out = np.empty((len(_W["mts"]), len(st), 3), dtype=complex)
    for col in range(len(_W["mts"])):
        us = US[:, col]
        for i, j in enumerate(st):
            out[col, i] = (us[j], us[n1 + j], us[2 * n1 + j])
    return out


def _worker(k):
    t0 = time.time()
    w = 2.0 * np.pi * k * _W["df"] + 1j * _W["omegai"]
    model0, if0, elim0 = _W["base"]
    n1 = if0[-1].faces.n

    blocks0 = model0.assemble_blocks(w)
    U0 = elim0.solve(blocks=blocks0,
                     b=rhs_matrix(model0, if0, w), full=False)
    outs = {"base": extract(U0, n1)}
    for name, model_m, if_m, elim_m in _W["members"]:
        blocks_m, _ = cached_blocks(model_m, w, blocks0)
        Um = elim_m.solve(blocks=blocks_m,
                          b=rhs_matrix(model_m, if_m, w), full=False)
        outs[name] = extract(Um, n1)
    return k, outs, time.time() - t0


def main():
    nprocs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    os.makedirs(ROOT, exist_ok=True)

    df = 1.0 / mi.TLEN
    imax = mi.IMAX
    omegai = mi.OMEGAI
    w0 = 2.0 * np.pi * imax * df / 3.0

    faces_list = [base_mesh(r, nm) for r, nm, _ in SPEC]
    mats = [rb.manifest_mat({"mat": list(m)}) for _, _, m in SPEC]

    t0 = time.time()
    model0, if0 = nested_shell_model(faces_list, mats, w0)
    elim0 = ShellElimination(model0, if0)
    members = []
    r_p, nm_p, _ = SPEC[PERT_IDX]
    for name, relief in MEMBERS:
        fm = pert_mesh(r_p, nm_p, relief)
        if_m = list(if0)
        if_m[PERT_IDX] = Interface(fm, if0[PERT_IDX].condition,
                                   if0[PERT_IDX].name)
        model_m = nested_shell_model_from_ifaces(if_m, mats, w0)
        members.append((name, model_m, if_m,
                        ShellElimination(model_m, if_m)))
    print("base + %d members built in %.1f s (system size %d)"
          % (len(members), time.time() - t0, model0.size), flush=True)

    stations = mi.pick_stations(faces_list[-1])
    src = {"lat": mi.SOURCE_LAT, "lon": mi.SOURCE_LON,
           "r0_km": mi.SOURCE_R0_KM}
    th = math.radians(90.0 - src["lat"])
    ph = math.radians(src["lon"])
    r0 = src["r0_km"] * 1e3
    src_xyz = (r0 * math.sin(th) * math.cos(ph),
               r0 * math.sin(th) * math.sin(ph), r0 * math.cos(th))
    scale = 1.0e20 / 100.0
    src_names = list(mi.SOURCES.keys())
    mts = [rb.sph_to_cart_mt(mi.SOURCES[s], src["lat"], src["lon"])
           * scale for s in src_names]

    _W.update(dict(base=(model0, if0, elim0), members=members,
                   df=df, omegai=omegai, mts=mts, src_xyz=src_xyz,
                   st_faces=[st["face"] for st in stations],
                   mat_outer=mats[-1]))

    # assembly speedup, measured at the top harmonic
    wtop = 2.0 * np.pi * imax * df + 1j * omegai
    blocks0 = model0.assemble_blocks(wtop)
    name, model_m, if_m, elim_m = members[1]
    t0 = time.time()
    model_m.assemble_blocks(wtop)
    t_fresh = time.time() - t0
    t0 = time.time()
    _, nnew = cached_blocks(model_m, wtop, blocks0)
    t_cached = time.time() - t0
    print("member assembly at k=%d: fresh %.1f s, cached %.1f s "
          "(%.2fx, %d blocks recomputed)" %
          (imax, t_fresh, t_cached, t_fresh / t_cached, nnew),
          flush=True)
    del blocks0

    names = ["base"] + [m[0] for m in members]
    u = {n: np.zeros((len(mts), len(stations), 3, imax + 1),
                     dtype=complex) for n in names}
    ks = list(range(1, imax + 1))
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=min(nprocs, len(ks))) as pool:
        for k, outs, dt in pool.imap_unordered(_worker, ks):
            for n in names:
                u[n][:, :, :, k] = outs[n]
            print("  k=%3d  %.1f s (base + %d cached members)"
                  % (k, dt, len(members)), flush=True)
    wall = time.time() - t0
    print("%d harmonics x %d solves in %.1f s (%d procs)"
          % (len(ks), len(names), wall, nprocs), flush=True)

    for n in names:
        np.savez(os.path.join(ROOT, "ens_%s.npz" % n),
                 sources=np.array(src_names), u=u[n], imax=imax, df=df,
                 omegai=omegai)
    with open(os.path.join(ROOT, "ens_manifest.json"), "w") as f:
        json.dump({"tlen": mi.TLEN, "imax": imax,
                   "omegai_1_per_s": omegai, "source": src,
                   "stations": stations, "members": [m[0] for m in
                                                     MEMBERS],
                   "perturbed_boundary_km": r_p,
                   "speedup_assembly": t_fresh / t_cached,
                   "t_fresh_s": t_fresh, "t_cached_s": t_cached},
                  f, indent=2)

    # gates
    ok = True
    bitwise = np.array_equal(u["amp0"], u["base"])
    print("GATE amp=0 member bitwise == base: %s" % bitwise)
    ok &= bitwise
    d1 = np.linalg.norm(u["y20_2500"] - u["base"])
    d2 = np.linalg.norm(u["y20_5000"] - u["base"])
    ratio = d2 / d1
    lin = 1.9 < ratio < 2.1
    print("GATE Y20 linearity ||u(5km)-u0||/||u(2.5km)-u0|| = %.4f "
          "in [1.9, 2.1]: %s" % (ratio, lin))
    ok &= lin
    dr = np.linalg.norm(u["rand_l4_5000"] - u["base"])
    print("random-relief response ||du||/||u0|| = %.3e (Y20 5 km: "
          "%.3e)" % (dr / np.linalg.norm(u["base"]),
                     d2 / np.linalg.norm(u["base"])))
    print("RUNG-3 ENSEMBLE %s" % ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
