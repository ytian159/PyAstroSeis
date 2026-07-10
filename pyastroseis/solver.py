"""Frequency-domain driver, port of the compute section of AstroSeis.m.

Frequencies are independent (one matrix build + solve each), so they
can be computed in parallel processes without changing any numbers;
--serial reproduces the exact MATLAB loop order.
"""

import json
import multiprocessing as mp
import os
import time

import numpy as np
import scipy.io as sio

from .assembly import Geometry, cal_traction
from .mesh import load_faces_mat
from .params import Params, read_config, read_params
from .source import QP_FAC_LEGACY, QP_FAC_U0E_LEGACY, u0e, u0eM


def load_params_any(param_file, lc=False):
    """Read a parameter file: YAML (.yml/.yaml) or the legacy
    fixed-line-order AstroSeis format."""
    if str(param_file).endswith((".yml", ".yaml")):
        return read_config(param_file)
    from .params import read_params_lc
    return read_params_lc(param_file) if lc else read_params(param_file)


def qp_factors(qp_mode, vp0, vs0):
    """Qp = fac*Qs factors for kernels and sources.

    "physical": pure-shear-attenuation relation Qp = (3/4)(vp/vs)^2 Qs
    computed from the actual velocities — the community-standard default
    (Aki & Richards; PREM/SPECFEM convention with Qkappa -> inf).
    "legacy": the original MATLAB constants (2.5 in the BEM kernels,
    2.25 in the single-force source Green function) — use to reproduce
    original AstroSeis results exactly.
    """
    if qp_mode == "physical":
        fac = 0.75 * (vp0 / vs0) ** 2
        return fac, {"u0e": fac, "u0m": fac}
    if qp_mode == "legacy":
        return QP_FAC_LEGACY, {"u0e": QP_FAC_U0E_LEGACY, "u0m": QP_FAC_LEGACY}
    raise ValueError(f"unknown qp_mode: {qp_mode!r} (physical|legacy)")


# worker globals (fork start method)
_W = {}


def _freq_worker(iw):
    p = _W["p"]
    faces = _W["faces"]
    t0 = time.time()
    df = _W["df"]
    freq = (iw - 1) * df
    w = 2 * np.pi * freq + _W["wi"] * 1j
    if p.source_type == "single":
        u01 = u0e(faces, w, p.rho, _W["mu"], _W["lamda"],
                  _W["xs"], _W["ys"], _W["zs"], p.Q, p.fsrc,
                  qp_fac=_W["qp_src"]["u0e"])
    else:  # moment
        u01 = u0eM(faces, w, p.rho, _W["mu"], _W["lamda"],
                   _W["xs"], _W["ys"], _W["zs"], p.Q, p.M,
                   qp_fac=_W["qp_src"]["u0m"])
    t1 = time.time()
    trac = cal_traction(faces, w, _W["lamda"], _W["mu"], p.rho, p.Q,
                        qp_fac=_W["qp_kernel"], geom=_W["geom"])
    t2 = time.time()
    x = np.linalg.solve(trac, u01)
    t3 = time.time()
    return iw, x, (t1 - t0, t2 - t1, t3 - t2)


def run_case(param_file, nprocs=1, out=None, qp_mode="physical",
             self_scheme="polar", quad_mode="adaptive", verbose=True):
    """Run one AstroSeis case (homogeneous body). Returns (uu, meta).

    self_scheme: "polar" (default; accurate + fast singular quadrature)
    or "grid" (MATLAB 300x300 punch-out scheme, for exact reproduction
    of original results together with qp_mode="legacy").
    """
    p = param_file if isinstance(param_file, Params) \
        else load_params_any(param_file)
    from .params import resolve_path
    faces = load_faces_mat(resolve_path(p.mesh_file, param_file))
    geom = Geometry(faces, self_scheme=self_scheme, quad_mode=quad_mode)

    # source position (AstroSeis.m, corrected source-radius version:
    # R = mean of ||incenter|| over faces)
    R = np.mean(np.sqrt((faces.ic ** 2).sum(axis=1)))
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
    mu = p.rho * p.vs * p.vs
    lamda = p.rho * p.vp * p.vp - 2 * mu
    qp_kernel, qp_src = qp_factors(qp_mode, p.vp, p.vs)

    # frequency list, exactly as the MATLAB loop breaks
    iws = []
    for iw in range(2, nt + 1):
        freq = (iw - 1) * df
        if freq > fmax:
            break
        iws.append(iw)

    _W.update(dict(p=p, faces=faces, df=df, wi=wi, mu=mu, lamda=lamda,
                   xs=xs, ys=ys, zs=zs, qp_kernel=qp_kernel, qp_src=qp_src,
                   geom=geom))

    n3 = 3 * faces.n
    uu = np.zeros((n3, nt), dtype=complex)
    t_start = time.time()

    if nprocs <= 1:
        results = map(_freq_worker, iws)
        for iw, x, tms in results:
            uu[:, iw - 1] = x
            uu[:, nt + 1 - iw] = np.conj(x)
            if verbose:
                print(f"[iw={iw:3d}] f={(iw - 1) * df:.4f} Hz  "
                      f"src {tms[0]:.1f}s  assembly {tms[1]:.1f}s  "
                      f"solve {tms[2]:.1f}s", flush=True)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(nprocs, len(iws))) as pool:
            for iw, x, tms in pool.imap_unordered(_freq_worker, iws):
                uu[:, iw - 1] = x
                uu[:, nt + 1 - iw] = np.conj(x)
                if verbose:
                    print(f"[iw={iw:3d}] f={(iw - 1) * df:.4f} Hz  "
                          f"src {tms[0]:.1f}s  assembly {tms[1]:.1f}s  "
                          f"solve {tms[2]:.1f}s", flush=True)

    wall = time.time() - t_start
    meta = dict(param_file=str(param_file), nfreq=len(iws),
                nt=nt, dt=dt, f0=f0, Q=p.Q, nprocs=nprocs, qp_mode=qp_mode,
                self_scheme=self_scheme, quad_mode=quad_mode,
                wall_seconds=wall, numface=faces.n,
                source=[xs, ys, zs], R_mean=R)
    if verbose:
        print(f"done: {len(iws)} frequencies in {wall:.1f} s "
              f"({nprocs} procs)", flush=True)

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        sio.savemat(out, {"uu": uu, "nt": float(nt), "T": T})
        with open(out + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if verbose:
            print(f"saved {out}")
    return uu, meta
