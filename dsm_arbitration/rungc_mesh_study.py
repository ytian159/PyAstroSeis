#!/usr/bin/env python3
"""Rung-C cure test: does epicentral-cap refinement collapse the
absolute-error floor that drowns the far-field quasi-statics?
Homog ball, 50-km mrt source, exact-trace probe at k=20 on meshes
hmin = 20 (campaign), 10, 5 km."""
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
from pyastroseis.meshgen import refine_toward       # noqa: E402
from pyastroseis.source import u0eM                 # noqa: E402
from pyastroseis.spheroidal_ref import (source_jumps, forced_surface,
                                        spheroidal_pole,
                                        spheroidal_reconstruct,
                                        q_factor, L_SERIES)  # noqa
from pyastroseis.toroidal_modes import toroidal_mode_spectra  # noqa
from pyastroseis.toroidal_ref import source_frame   # noqa: E402
import synthesize_compare as sc                     # noqa: E402

man = json.load(open("dsm_arbitration_u3/manifest.json"))
sh = man["models"]["corefluid_q50"][-1]
base = run_bem.load_faces("dsm_arbitration_u3/" + sh["mesh"])
mat = run_bem.manifest_mat(sh)
a = sh["r_km"] * 1e3
rho, vp, vs = sh["mat"][0] * 1e3, sh["mat"][1] * 1e3, sh["mat"][2] * 1e3
src = man["source"]
th = math.radians(90 - src["lat"])
ph = math.radians(src["lon"])
r0 = src["r0_km"] * 1e3
sx = np.array([r0 * math.sin(th) * math.cos(ph),
               r0 * math.sin(th) * math.sin(ph), r0 * math.cos(th)])
epi = sx / np.linalg.norm(sx) * a
M = run_bem.sph_to_cart_mt(man["moment_tensors_1e25dyncm"]["mrt"],
                           src["lat"], src["lon"]) \
    * man["moment_scale_Nm"] / 100.0
Q = 50.0
Qrot = source_frame(sx)
M_sf = Qrot @ M @ Qrot.T
mzx, mzy = M_sf[2, 0], M_sf[2, 1]
LMAX = 900
DY, DG = spheroidal_pole(LMAX)
mu_s0 = rho * vs ** 2
ka_s0 = rho * vp ** 2 - 4.0 / 3.0 * mu_s0
kk = 20
w = 2 * np.pi * kk / man["tlen"] + 1j * man["omegai_1_per_s"]

# exact per-(l,m) surface coefficients once (mesh-independent)
fac2 = q_factor(w, Q) ** 2
mu_s = mu_s0 * fac2
lam_s = ka_s0 - 2.0 / 3.0 * mu_s
model_s = dict(a=a, b=3.48e6, c=1.2e6, sh=(rho, lam_s, mu_s),
               oc=(5000.0, 5500.0), ic=(6000.0, 4.9e11, 7.35e10))
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

st_dirs = {}
for st in man["stations"]:
    st_dirs[st["name"]] = (sc.unit_vec(st["lat"], st["lon"]),
                           st["dist_deg"], st["lat"], st["lon"])

def uniform_x4(f):
    # radially-projected uniform quadrisection (geometry-improving)
    return refine_toward(f, epi, hmin=1.0, grade=0.0, max_levels=1)


for tag, mk in (("base(h)", lambda: base),
                ("uniform x4 (h/2, on-sphere)",
                 lambda: uniform_x4(base))):
    faces = mk()
    model, ifaces = nested_shell_model(
        [faces], [mat], 2 * np.pi * 138 / man["tlen"] / 3,
        near_tier=(1.5, (10, 2)))
    dirs = faces.ic / np.linalg.norm(faces.ic, axis=1)[:, None]
    u_sph = spheroidal_reconstruct(Wu, Wv, dirs @ Qrot.T) @ Qrot
    u_tor = toroidal_mode_spectra(dict(vs=vs, rho=rho, a=a, b=None),
                                  sx, M, np.array([w]), dirs,
                                  Q=Q, q_sign=-1.0)[:, :, 0]
    u = u_sph + u_tor
    n = faces.n
    u_ex = np.empty(3 * n, dtype=complex)
    u_ex[:n], u_ex[n:2 * n], u_ex[2 * n:] = u[:, 0], u[:, 1], u[:, 2]
    A = model.assemble(w)
    u_inc = u0eM(faces, w, mat.rho, mat.mu, mat.lamda, *sx, mat.Q, M,
                 qp_fac=mat.qp_fac, disp_ref_hz=mat.disp_ref_hz)
    x = np.linalg.solve(A, u_inc)
    err = np.abs(x - u_ex)
    print("%s: n=%d  global rel %.4f  median|err| %.3e  max|err| %.3e"
          % (tag, n, np.linalg.norm(x - u_ex) / np.linalg.norm(u_ex),
             np.median(err), np.max(err)), flush=True)
    rows = []
    for name, (xh, dd, la, lo) in st_dirs.items():
        j = int(np.argmax(dirs @ xh))
        away, t_hat, north, east = sc.rotation_to_ne(
            src["lat"], src["lon"], la, lo)
        uex = np.array([u_ex[j], u_ex[n + j], u_ex[2 * n + j]])
        ux = np.array([x[j], x[n + j], x[2 * n + j]])
        rows.append(abs((ux - uex) @ away) / abs(uex @ away))
    print("  station R rel errs: median %.3f  " % np.median(rows),
          " ".join("%.2f" % r for r in rows), flush=True)
