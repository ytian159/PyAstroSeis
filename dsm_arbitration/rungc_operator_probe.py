#!/usr/bin/env python3
"""Rung-C operator probe: is the missing quasi-static channel an RHS
problem or a discrete-OPERATOR problem?

Homogeneous ball (shell material) on the campaign's graded surface
mesh, 50-km mrt source. Build the EXACT total-field trace u_ex at the
face incenters from the validated analytic machinery (spheroidal
direct solve, ball branch + toroidal mode-sum, both in BEM-native
convention = raw output). Then:

  consistency:   r = A u_ex - u_inc      (small => operator OK)
  amplification: x = A^{-1} u_inc ;  e = x - u_ex
                 compare |e| vs |A^{-1} r| and the sigma-spectrum.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("ARB_ROOT", os.path.abspath("dsm_arbitration_u3"))
import run_bem                                      # noqa: E402
from pyastroseis.domains import nested_shell_model  # noqa: E402
from pyastroseis.source import u0eM                 # noqa: E402
from pyastroseis.spheroidal_ref import (source_jumps, forced_surface,
                                        spheroidal_pole,
                                        spheroidal_reconstruct,
                                        q_factor, L_SERIES)  # noqa
from pyastroseis.toroidal_modes import toroidal_mode_spectra  # noqa
from pyastroseis.toroidal_ref import source_frame   # noqa: E402

man = json.load(open("dsm_arbitration_u3/manifest.json"))
sh = man["models"]["corefluid_q50"][-1]
faces = run_bem.load_faces("dsm_arbitration_u3/" + sh["mesh"])
mat = run_bem.manifest_mat(sh)
a = sh["r_km"] * 1e3
rho, vp, vs = sh["mat"][0] * 1e3, sh["mat"][1] * 1e3, sh["mat"][2] * 1e3
w0 = 2 * np.pi * 138 / man["tlen"] / 3
model, ifaces = nested_shell_model([faces], [mat], w0)

src = man["source"]
th = math.radians(90 - src["lat"])
ph = math.radians(src["lon"])
r0 = src["r0_km"] * 1e3
sx = np.array([r0 * math.sin(th) * math.cos(ph),
               r0 * math.sin(th) * math.sin(ph), r0 * math.cos(th)])
M = run_bem.sph_to_cart_mt(man["moment_tensors_1e25dyncm"]["mrt"],
                           src["lat"], src["lon"]) \
    * man["moment_scale_Nm"] / 100.0

dirs = faces.ic / np.linalg.norm(faces.ic, axis=1)[:, None]
Q = 50.0
Qrot = source_frame(sx)
M_sf = Qrot @ M @ Qrot.T
mzx, mzy = M_sf[2, 0], M_sf[2, 1]
LMAX = 900
DY, DG = spheroidal_pole(LMAX)
dirs_sf = dirs @ Qrot.T
mu_s0 = rho * vs ** 2
ka_s0 = rho * vp ** 2 - 4.0 / 3.0 * mu_s0
ball = dict(a=a, b=None, c=None)


def exact_trace(w):
    """(3n,) BEM-native exact total field at face incenters:
    spheroidal ball solve + toroidal ball mode-sum."""
    fac2 = q_factor(w, Q) ** 2
    mu_s = mu_s0 * fac2
    lam_s = ka_s0 - 2.0 / 3.0 * mu_s
    model_s = dict(a=a, b=3.48e6, c=1.2e6, sh=(rho, lam_s, mu_s),
                   oc=(5000.0, 5500.0),
                   ic=(6000.0, 4.9e11, 7.35e10))
    Wu = np.zeros((LMAX + 1, 9), dtype=complex)
    Wv = np.zeros((LMAX + 1, 9), dtype=complex)
    for l in range(1, LMAX + 1):
        L = l * (l + 1.0)
        zr = a if l >= L_SERIES else None
        F0 = np.array([0, 0, 0, 1.0 / (L * r0 ** 2)], dtype=complex)
        F1 = np.array([0, 0, -1.0 / r0 ** 3, 3.0 / (L * r0 ** 3)],
                      dtype=complex)
        J = source_jumps(l, w, r0, rho, lam_s, mu_s, F0, F1, zref_a=zr)
        U, V = forced_surface(l, w, model_s, r0, J, l_switch=0)
        for m in (-1, 1):
            D = DG[m][l][0] * mzx + DG[m][l][1] * mzy
            Wu[l, m + 4] = D * U
            Wv[l, m + 4] = D * V
    u_sph = spheroidal_reconstruct(Wu, Wv, dirs_sf) @ Qrot
    u_tor = toroidal_mode_spectra(dict(vs=vs, rho=rho, a=a, b=None),
                                  sx, M, np.array([w]), dirs,
                                  Q=Q, q_sign=-1.0)[:, :, 0]
    u = u_sph + u_tor
    n = faces.n
    out = np.empty(3 * n, dtype=complex)
    out[:n], out[n:2 * n], out[2 * n:] = u[:, 0], u[:, 1], u[:, 2]
    return out


for kk in (5, 20, 60, 138):
    w = 2 * np.pi * kk / man["tlen"] + 1j * man["omegai_1_per_s"]
    A = model.assemble(w)
    u_inc = u0eM(faces, w, mat.rho, mat.mu, mat.lamda, *sx, mat.Q, M,
                 qp_fac=mat.qp_fac, disp_ref_hz=mat.disp_ref_hz)
    u_ex = exact_trace(w)
    r = A @ u_ex - u_inc
    x = np.linalg.solve(A, u_inc)
    e = x - u_ex
    Ainv_r = np.linalg.solve(A, r)
    nrm = np.linalg.norm
    print("k=%3d: |u_ex| %.3e  consistency |r|/|u_inc| %.3f ;  "
          "solve err |e|/|u_ex| %.3f ;  |A^-1 r|/|u_ex| %.3f"
          % (kk, nrm(u_ex), nrm(r) / nrm(u_inc), nrm(e) / nrm(u_ex),
             nrm(Ainv_r) / nrm(u_ex)), flush=True)
    if kk == 20:
        sv = np.linalg.svd(A, compute_uv=False)
        print("  sigma: max %.3e min %.3e cond %.2e  smallest5: %s"
              % (sv[0], sv[-1], sv[0] / sv[-1],
                 " ".join("%.2e" % s for s in sv[-5:])))

# ---- per-station far-field error breakdown (append-mode run) ----
if os.environ.get("PROBE_STATIONS") == "1":
    st_faces = [st["face"] for st in man["stations"]]
    for kk in (20, 60):
        w = 2 * np.pi * kk / man["tlen"] + 1j * man["omegai_1_per_s"]
        A = model.assemble(w)
        u_inc = u0eM(faces, w, mat.rho, mat.mu, mat.lamda, *sx,
                     mat.Q, M, qp_fac=mat.qp_fac,
                     disp_ref_hz=mat.disp_ref_hz)
        u_ex = exact_trace(w)
        x = np.linalg.solve(A, u_inc)
        n = faces.n
        print("k=%d per-station R-channel: |u_ex_R|  |err_R|/|u_ex_R|"
              % kk)
        for st in man["stations"]:
            j = st["face"]
            d = faces.ic[j] / np.linalg.norm(faces.ic[j])
            import synthesize_compare as sc2
            away, t_hat, north, east = sc2.rotation_to_ne(
                src["lat"], src["lon"], st["lat"], st["lon"])
            uex = np.array([u_ex[j], u_ex[n + j], u_ex[2 * n + j]])
            ux = np.array([x[j], x[n + j], x[2 * n + j]])
            eR = abs((ux - uex) @ away)
            gR = abs(uex @ away)
            print("  %-6s %5.1f: %.3e  %6.2f"
                  % (st["name"], st["dist_deg"], gR, eR / gR))
        # absolute error scale vs epicentral field scale
        err = np.abs(x - u_ex)
        print("  global: max|u_ex| %.3e  median|err| %.3e  "
              "max|err| %.3e"
              % (np.max(np.abs(u_ex)), np.median(err), np.max(err)))
