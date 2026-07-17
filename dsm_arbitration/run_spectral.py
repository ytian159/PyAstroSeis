#!/usr/bin/env python3
"""Rung-A spectral-sweep spectra for the campaign (BOTH parities:
spheroidal P-SV and toroidal SH by direct forced solve,
pyastroseis/spectral.py), in the bem-npz layout.

usage: run_spectral.py <root> <model> [Q or 'el'] [q_sign] [lmax]
                       [nproc]
Writes <root>/spectral_<model>[_el].npz with u (= u_psv + u_sh),
u_psv, u_sh  (each nsrc, nst, 3, imax+1).
"""
import json
import math
import multiprocessing as mp
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
from pyastroseis.spectral import (spectral_poles,   # noqa: E402
                                  spectral_spectra)

MODEL = sys.argv[2]
QARG = sys.argv[3] if len(sys.argv) > 3 else "50"
Q = None if QARG == "el" else float(QARG)
Q_SIGN = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
LMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 900
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 16

man = json.load(open(os.path.join(ROOT, "manifest.json")))
layers_man = man["models"][MODEL]
LAYERS = [dict(r_top=ly["r_km"] * 1e3, rho=ly["mat"][0] * 1e3,
               vp=ly["mat"][1] * 1e3,
               vs=(ly["mat"][2] * 1e3 if len(ly["mat"]) > 2
                   else 0.0))
          for ly in layers_man]

src = man["source"]
th = math.radians(90.0 - src["lat"])
ph = math.radians(src["lon"])
r0 = src["r0_km"] * 1e3
SRC_XYZ = np.array([r0 * math.sin(th) * math.cos(ph),
                    r0 * math.sin(th) * math.sin(ph),
                    r0 * math.cos(th)])

scale = man["moment_scale_Nm"] / 100.0
src_names = list(man["moment_tensors_1e25dyncm"].keys())
MTS = [run_bem.sph_to_cart_mt(man["moment_tensors_1e25dyncm"][s],
                              src["lat"], src["lon"]) * scale
       for s in src_names]

sh_mesh = layers_man[-1]["mesh"]
faces = run_bem.load_faces(os.path.join(ROOT, sh_mesh))
st_idx = [st["face"] for st in man["stations"]]
ST_DIRS = faces.ic[st_idx]
ST_DIRS = ST_DIRS / np.linalg.norm(ST_DIRS, axis=1)[:, None]

imax = man["imax"]
ks = np.arange(1, imax + 1)
W_ARR = 2.0 * np.pi * ks / man["tlen"] + 1j * man["omegai_1_per_s"]

POLES = None


def _worker(iw_chunk):
    w_sub = W_ARR[iw_chunk]
    t0 = time.time()
    out = spectral_spectra(LAYERS, SRC_XYZ, MTS, w_sub, ST_DIRS,
                           Q=Q, q_sign=Q_SIGN, lmax=LMAX,
                           poles=POLES)
    return iw_chunk, out["psv"], out["sh"], time.time() - t0


if __name__ == "__main__":
    print("spectral %s: depth %.0f km, Q %s sign %+.0f lmax %d, "
          "%d stations x %d harmonics, %d procs"
          % (MODEL, (LAYERS[-1]["r_top"] - r0) / 1e3, QARG, Q_SIGN,
             LMAX, len(ST_DIRS), imax, NPROC), flush=True)
    t0 = time.time()
    POLES = spectral_poles(LMAX)
    print("pole couplings: %.1f s" % (time.time() - t0), flush=True)
    chunks = [list(range(i, imax, NPROC)) for i in range(NPROC)]
    chunks = [c for c in chunks if c]
    u_psv = np.zeros((len(MTS), len(ST_DIRS), 3, imax + 1),
                     dtype=complex)
    u_sh = np.zeros_like(u_psv)
    t1 = time.time()
    with mp.Pool(NPROC) as pool:
        for iw_chunk, psv, sh, dt in pool.imap_unordered(
                _worker, chunks):
            for j, iw in enumerate(iw_chunk):
                u_psv[:, :, :, iw + 1] = psv[:, :, :, j]
                u_sh[:, :, :, iw + 1] = sh[:, :, :, j]
            print("  chunk of %d harmonics done (%.1f s)"
                  % (len(iw_chunk), dt), flush=True)
    print("solves: %.1f s" % (time.time() - t1), flush=True)
    for i, s in enumerate(src_names):
        print("  %s: max |u_psv| %.3e  max |u_sh| %.3e m"
              % (s, np.abs(u_psv[i]).max(), np.abs(u_sh[i]).max()))
    tag = "_el" if Q is None else ""
    out = os.path.join(ROOT, "spectral_%s%s.npz" % (MODEL, tag))
    np.savez(out, sources=np.array(src_names), u=u_psv + u_sh,
             u_psv=u_psv, u_sh=u_sh, imax=imax,
             df=1.0 / man["tlen"], omegai=man["omegai_1_per_s"],
             q_sign=Q_SIGN, lmax=LMAX)
    print("wrote %s  (total %.1f s)" % (out, time.time() - t0))
