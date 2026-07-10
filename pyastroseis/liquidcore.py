"""Solid body with a liquid core: coupled elastic/acoustic BEM solver.

Ports of AstroSeis_liquidcore.m (driver) and liq_core.m (coupled block
system). Unknowns per frequency: x = [p (N2); u_core (3*N2); u_surf (3*N1)]
where N2 = core-interface faces, N1 = free-surface faces.
"""

import dataclasses
import json
import multiprocessing as mp
import os
import time

import numpy as np
import scipy.io as sio

from .assembly import (QP_FAC_LEGACY, cal_A_st, cal_B_st, cal_G_st,
                       cal_T_st, smat_func)
from .mesh import faces_from_struct_array
from .params import read_params_lc
from .solver import qp_factors
from .source import u0e, u0eM, u0p_exp


def flip_normals(faces):
    """Copy of faces with reversed surface normals (used for the core
    interface seen from the solid side)."""
    return dataclasses.replace(faces, nvec=-faces.nvec)


def load_layers_mat(path):
    """Load the MATLAB `layer` struct array (layer(1)=core,
    layer(2)=surface). Returns a list of Faces."""
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    layers = np.atleast_1d(m["layer"])
    return [faces_from_struct_array(np.atleast_1d(lay.face))
            for lay in layers]


def liq_core(face1, face2, w, lamda1, mu1, rho1, lamda2, mu2, rho2, Q,
             P0, u01, u02, Smat, w0, nint=10, nxi=300,
             qp_fac=QP_FAC_LEGACY, qp_fac_fluid=QP_FAC_LEGACY,
             face2w1=None, geoms=None, self_scheme="grid"):
    """Assemble the coupled system (liq_core.m). face1 = free surface,
    face2 = core interface (original outward normals). Returns (A, b).

    geoms: optional (geom_face1, geom_face2w1, geom_face2) prebuilt
    Geometry objects (hoisted across frequencies by run_case_lc)."""
    if face2w1 is None:
        face2w1 = flip_normals(face2)
    g1 = g2w = g2 = None
    if geoms is not None:
        g1, g2w, g2 = geoms

    kw = dict(nint=nint, nxi=nxi, qp_fac=qp_fac, self_scheme=self_scheme)
    T11 = cal_T_st(face1, face1, w, lamda1, mu1, rho1, Q, geom=g1, **kw)
    T22 = cal_T_st(face2w1, face2w1, w, lamda1, mu1, rho1, Q, geom=g2w, **kw)
    T12 = cal_T_st(face1, face2w1, w, lamda1, mu1, rho1, Q, geom=g2w, **kw)
    T21 = cal_T_st(face2w1, face1, w, lamda1, mu1, rho1, Q, geom=g1, **kw)

    G12 = cal_G_st(face1, face2w1, w, lamda1, mu1, rho1, Q, geom=g2w, **kw)
    G22 = cal_G_st(face2w1, face2w1, w, lamda1, mu1, rho1, Q, geom=g2w, **kw)

    kwf = dict(nint=nint, nxi=nxi, qp_fac=qp_fac_fluid,
               self_scheme=self_scheme)
    A22 = cal_A_st(face2, face2, w, lamda2, mu2, rho2, Q, geom=g2, **kwf)
    B22 = cal_B_st(face2, face2, w, lamda2, mu2, rho2, Q, geom=g2, **kwf)

    # scale factor s = rho*c*w0 (as written in liq_core.m)
    scale_fac = rho2 * np.sqrt(lamda1 + 2 * mu1) / np.sqrt(rho2) * w0

    n1, n2 = face1.n, face2.n
    A = np.zeros((n2 + 3 * n2 + 3 * n1,) * 2, dtype=complex)
    b = np.empty(n2 + 3 * n2 + 3 * n1, dtype=complex)

    A[:n2, :n2] = A22 / scale_fac
    A[:n2, n2:4 * n2] = -rho2 * w ** 2 * (B22 @ Smat) / scale_fac
    # A[:n2, 4*n2:] stays zero
    A[n2:4 * n2, :n2] = -G22 @ Smat.T
    A[n2:4 * n2, n2:4 * n2] = T22
    A[n2:4 * n2, 4 * n2:] = T21
    A[4 * n2:, :n2] = -G12 @ Smat.T
    A[4 * n2:, n2:4 * n2] = T12
    A[4 * n2:, 4 * n2:] = T11

    b[:n2] = np.asarray(P0, dtype=complex).ravel() / scale_fac
    b[n2:4 * n2] = u02
    b[4 * n2:] = u01
    return A, b


# worker globals (fork start method)
_W = {}


def _freq_worker_lc(iw):
    p = _W["p"]
    t0 = time.time()
    freq = (iw - 1) * _W["df"]
    w = 2 * np.pi * freq + _W["wi"] * 1j
    face1, face2 = _W["face1"], _W["face2"]
    n1, n2 = face1.n, face2.n
    qk, qs = _W["qp_kernel"], _W["qp_src"]
    qf = _W["qp_fluid"]

    if p.source_type == "single":
        if _W["rs"] < _W["Rc"]:
            raise ValueError("single-force source is inside the liquid core; "
                             "change the source location")
        u02 = u0e(face2, w, p.rho1, _W["mu1"], _W["lamda1"],
                  _W["xs"], _W["ys"], _W["zs"], p.Q, p.fsrc, qp_fac=qs["u0e"])
        u01 = u0e(face1, w, p.rho1, _W["mu1"], _W["lamda1"],
                  _W["xs"], _W["ys"], _W["zs"], p.Q, p.fsrc, qp_fac=qs["u0e"])
        P0 = np.zeros(n2, dtype=complex)
    else:  # moment
        if _W["rs"] < _W["Rc"]:
            u02 = np.zeros(3 * n2, dtype=complex)
            u01 = np.zeros(3 * n1, dtype=complex)
            P0 = u0p_exp(face2, w, p.vp2, _W["xs"], _W["ys"], _W["zs"],
                         p.Q, qp_fac=qs["u0m"])
        else:
            u02 = u0eM(face2, w, p.rho1, _W["mu1"], _W["lamda1"],
                       _W["xs"], _W["ys"], _W["zs"], p.Q, p.M, qp_fac=qs["u0m"])
            u01 = u0eM(face1, w, p.rho1, _W["mu1"], _W["lamda1"],
                       _W["xs"], _W["ys"], _W["zs"], p.Q, p.M, qp_fac=qs["u0m"])
            P0 = np.zeros(n2, dtype=complex)

    t1 = time.time()
    A, b = liq_core(face1, face2, w, _W["lamda1"], _W["mu1"], p.rho1,
                    _W["lamda2"], _W["mu2"], p.rho2, p.Q, P0, u01, u02,
                    _W["Smat"], _W["w0"], qp_fac=qk, qp_fac_fluid=qf,
                    face2w1=_W["face2w1"], geoms=_W["geoms"],
                    self_scheme=_W["self_scheme"])
    t2 = time.time()
    x = np.linalg.solve(A, b)
    t3 = time.time()
    del A, b
    return iw, x, (t1 - t0, t2 - t1, t3 - t2)


def run_case_lc(param_file, nprocs=1, out=None, qp_mode="physical",
                self_scheme="polar", quad_mode="adaptive", verbose=True):
    """Run one liquid-core case. Returns (up, u2, u1, meta).

    self_scheme: "polar" (default) or "grid" (exact MATLAB scheme)."""
    from .params import ParamsLC, resolve_path
    from .solver import load_params_any
    p = param_file if isinstance(param_file, ParamsLC) \
        else load_params_any(param_file, lc=True)
    layers = load_layers_mat(resolve_path(p.mesh_file, param_file))
    face2, face1 = layers[0], layers[1]   # core, free surface

    R = np.mean(np.sqrt((face1.ic ** 2).sum(axis=1)))
    Rc = np.mean(np.sqrt((face2.ic ** 2).sum(axis=1)))
    rs = R - p.h
    thetas = (90.0 - p.lat) * np.pi / 180.0
    phis = p.lon * np.pi / 180.0
    xs = rs * np.sin(thetas) * np.cos(phis)
    ys = rs * np.sin(thetas) * np.sin(phis)
    zs = rs * np.cos(thetas)

    nt, dt, f0 = p.nt, p.dt, p.f0
    T = nt * dt
    fmax = 3 * f0
    wi = 4 / T
    df = 1 / (nt * dt)
    mu1 = p.rho1 * p.vs1 * p.vs1
    lamda1 = p.rho1 * p.vp1 * p.vp1 - 2 * mu1
    # NOTE: mu2 uses rho1, replicating AstroSeis_liquidcore.m line 128
    # verbatim; harmless for a true fluid core (vs2 = 0 -> mu2 = 0).
    mu2 = p.rho1 * p.vs2 * p.vs2
    lamda2 = p.rho2 * p.vp2 * p.vp2 - 2 * mu2

    qk, qs = qp_factors(qp_mode, p.vp1, p.vs1)
    qf = 1.0 if qp_mode == "physical" else QP_FAC_LEGACY

    Smat = smat_func(face2)
    # PHASE2 hoist: one Geometry per surface, shared with fork workers
    from .assembly import Geometry
    face2w1 = flip_normals(face2)
    geoms = (Geometry(face1, self_scheme=self_scheme, quad_mode=quad_mode),
             Geometry(face2w1, self_scheme=self_scheme, quad_mode=quad_mode),
             Geometry(face2, self_scheme=self_scheme, quad_mode=quad_mode))

    iws = []
    for iw in range(2, nt + 1):
        freq = (iw - 1) * df
        if freq > fmax:
            break   # MATLAB uses `continue`; the computed set is identical
        iws.append(iw)

    _W.update(dict(p=p, face1=face1, face2=face2, df=df, wi=wi,
                   mu1=mu1, lamda1=lamda1, mu2=mu2, lamda2=lamda2,
                   xs=xs, ys=ys, zs=zs, rs=rs, Rc=Rc, Smat=Smat,
                   w0=2 * np.pi * f0, qp_kernel=qk, qp_src=qs, qp_fluid=qf,
                   face2w1=face2w1, geoms=geoms, self_scheme=self_scheme))

    n1, n2 = face1.n, face2.n
    up = np.zeros((n2, nt), dtype=complex)
    u2 = np.zeros((3 * n2, nt), dtype=complex)
    u1 = np.zeros((3 * n1, nt), dtype=complex)
    t_start = time.time()

    def store(iw, x):
        up[:, iw - 1] = x[:n2]
        u2[:, iw - 1] = x[n2:4 * n2]
        u1[:, iw - 1] = x[4 * n2:]
        up[:, nt + 1 - iw] = np.conj(up[:, iw - 1])
        u2[:, nt + 1 - iw] = np.conj(u2[:, iw - 1])
        u1[:, nt + 1 - iw] = np.conj(u1[:, iw - 1])

    if nprocs <= 1:
        for iw in iws:
            iw_, x, tms = _freq_worker_lc(iw)
            store(iw_, x)
            if verbose:
                print(f"[iw={iw_:3d}] f={(iw_ - 1) * df:.4f} Hz  "
                      f"src {tms[0]:.1f}s  assembly {tms[1]:.1f}s  "
                      f"solve {tms[2]:.1f}s", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nprocs, len(iws))) as pool:
            for iw_, x, tms in pool.imap_unordered(_freq_worker_lc, iws):
                store(iw_, x)
                if verbose:
                    print(f"[iw={iw_:3d}] f={(iw_ - 1) * df:.4f} Hz  "
                          f"src {tms[0]:.1f}s  assembly {tms[1]:.1f}s  "
                          f"solve {tms[2]:.1f}s", flush=True)

    wall = time.time() - t_start
    meta = dict(param_file=str(param_file), nfreq=len(iws),
                nt=nt, dt=dt, f0=f0, Q=p.Q, nprocs=nprocs, qp_mode=qp_mode,
                self_scheme=self_scheme, quad_mode=quad_mode,
                wall_seconds=wall,
                n_surface=n1, n_core=n2,
                source=[xs, ys, zs], R_mean=R, Rc_mean=Rc)
    if verbose:
        print(f"done: {len(iws)} frequencies in {wall:.1f} s "
              f"({nprocs} procs)", flush=True)

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        sio.savemat(out, {"up": up, "u2": u2, "u1": u1,
                          "nt": float(nt), "T": T})
        with open(out + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"saved {out}")
    return up, u2, u1, meta
