#!/usr/bin/env python3
"""Toroidal mode-sum SH reference spectra for the campaign, in the
bem-npz layout. Ball (single-layer models) or annulus (fluid-core
models; SH lives in the outer solid shell only).

usage: run_tormodes.py <root> <model> [Q or 'el'] [q_sign] [fmax_mHz]
                       [lmax] [n_keep]
Writes <root>/tormodes_<model>[_el].npz with u (nsrc, nst, 3, imax+1).
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(sys.argv[1])
os.environ.setdefault("ARB_ROOT", ROOT)
import run_bem                                      # noqa: E402
from pyastroseis.toroidal_modes import toroidal_mode_spectra  # noqa

MODEL = sys.argv[2]
QARG = sys.argv[3] if len(sys.argv) > 3 else "50"
Q = None if QARG == "el" else float(QARG)
Q_SIGN = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
FMAX = (float(sys.argv[5]) if len(sys.argv) > 5 else 10.0) * 1e-3
LMAX = int(sys.argv[6]) if len(sys.argv) > 6 else None
N_KEEP = int(sys.argv[7]) if len(sys.argv) > 7 else 8

man = json.load(open(os.path.join(ROOT, "manifest.json")))
layers = man["models"][MODEL]
solid = layers[-1]                    # outer solid shell
vs = solid["mat"][2] * 1e3
rho = solid["mat"][0] * 1e3
a = solid["r_km"] * 1e3
b = layers[-2]["r_km"] * 1e3 if len(layers) > 1 else None
model = dict(vs=vs, rho=rho, a=a, b=b)

src = man["source"]
th = math.radians(90.0 - src["lat"])
ph = math.radians(src["lon"])
r0 = src["r0_km"] * 1e3
src_xyz = np.array([r0 * math.sin(th) * math.cos(ph),
                    r0 * math.sin(th) * math.sin(ph),
                    r0 * math.cos(th)])

scale = man["moment_scale_Nm"] / 100.0
src_names = list(man["moment_tensors_1e25dyncm"].keys())
mts = [run_bem.sph_to_cart_mt(man["moment_tensors_1e25dyncm"][s],
                              src["lat"], src["lon"]) * scale
       for s in src_names]

faces = run_bem.load_faces(os.path.join(ROOT, solid["mesh"]))
st_idx = [st["face"] for st in man["stations"]]
st_dirs = faces.ic[st_idx]
st_dirs = st_dirs / np.linalg.norm(st_dirs, axis=1)[:, None]

df = 1.0 / man["tlen"]
imax = man["imax"]
ks = np.arange(1, imax + 1)
w_arr = 2.0 * np.pi * ks * df + 1j * man["omegai_1_per_s"]

print("tormodes %s: vs %.0f rho %.0f a %.0f b %s, r0 depth %.0f km, "
      "Q %s sign %+.0f fmax %.1f mHz"
      % (MODEL, vs, rho, a, b and "%.0f" % b, (a - r0) / 1e3,
         QARG, Q_SIGN, FMAX * 1e3), flush=True)

u = np.zeros((len(mts), len(st_dirs), 3, imax + 1), dtype=complex)
for col, M in enumerate(mts):
    spec = toroidal_mode_spectra(model, src_xyz, M, w_arr, st_dirs,
                                 Q=Q, q_sign=Q_SIGN, fmax=FMAX,
                                 lmax=LMAX, n_keep=N_KEEP)
    u[col, :, :, 1:] = spec
    print("  %s: max |u| = %.3e m" % (src_names[col],
                                      np.abs(spec).max()), flush=True)

tag = "_el" if Q is None else ""
out = os.path.join(ROOT, "tormodes_%s%s.npz" % (MODEL, tag))
np.savez(out, sources=np.array(src_names), u=u, imax=imax, df=df,
         omegai=man["omegai_1_per_s"], q_sign=Q_SIGN, fmax=FMAX)
print("wrote", out)
