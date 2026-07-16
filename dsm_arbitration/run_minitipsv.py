#!/usr/bin/env python3
"""mini-tipsv: exact spheroidal (P-SV) spectra for the campaign by
direct analytic forced solve (pyastroseis/spheroidal_ref.py), in the
bem-npz layout.

usage: run_minitipsv.py <root> <model> [Q or 'el'] [q_sign] [lmax]
Writes <root>/minitipsv_<model>[_el].npz  (u: nsrc, nst, 3, imax+1).
"""
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.abspath(sys.argv[1])
os.environ.setdefault("ARB_ROOT", ROOT)
import run_bem                                      # noqa: E402
from pyastroseis.spheroidal_ref import spheroidal_spectra  # noqa

MODEL = sys.argv[2]
QARG = sys.argv[3] if len(sys.argv) > 3 else "50"
Q = None if QARG == "el" else float(QARG)
Q_SIGN = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
LMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 250

man = json.load(open(os.path.join(ROOT, "manifest.json")))
layers = man["models"][MODEL]
assert len(layers) == 3, "mini-tipsv expects IC|OC|shell"
ic, oc, sh = layers
model0 = dict(a=sh["r_km"] * 1e3, b=oc["r_km"] * 1e3,
              c=ic["r_km"] * 1e3,
              sh=(sh["mat"][0] * 1e3, sh["mat"][1] * 1e3,
                  sh["mat"][2] * 1e3),
              oc=(oc["mat"][0] * 1e3, oc["mat"][1] * 1e3),
              ic=(ic["mat"][0] * 1e3, ic["mat"][1] * 1e3,
                  ic["mat"][2] * 1e3))

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

faces = run_bem.load_faces(os.path.join(ROOT, sh["mesh"]))
st_idx = [st["face"] for st in man["stations"]]
st_dirs = faces.ic[st_idx]
st_dirs = st_dirs / np.linalg.norm(st_dirs, axis=1)[:, None]

imax = man["imax"]
ks = np.arange(1, imax + 1)
w_arr = 2.0 * np.pi * ks / man["tlen"] + 1j * man["omegai_1_per_s"]

print("mini-tipsv %s: r0 depth %.0f km, Q %s sign %+.0f lmax %d, "
      "%d stations x %d harmonics"
      % (MODEL, (model0["a"] - r0) / 1e3, QARG, Q_SIGN, LMAX,
         len(st_dirs), imax), flush=True)
t0 = time.time()
spec = spheroidal_spectra(model0, src_xyz, mts, w_arr, st_dirs,
                          Q=Q, q_sign=Q_SIGN, lmax=LMAX)
u = np.zeros((len(mts), len(st_dirs), 3, imax + 1), dtype=complex)
u[:, :, :, 1:] = spec
for i, s in enumerate(src_names):
    print("  %s: max |u| = %.3e m" % (s, np.abs(u[i]).max()))
tag = "_el" if Q is None else ""
out = os.path.join(ROOT, "minitipsv_%s%s.npz" % (MODEL, tag))
np.savez(out, sources=np.array(src_names), u=u, imax=imax,
         df=1.0 / man["tlen"], omegai=man["omegai_1_per_s"],
         q_sign=Q_SIGN, lmax=LMAX)
print("wrote %s  (%.1f s)" % (out, time.time() - t0))
