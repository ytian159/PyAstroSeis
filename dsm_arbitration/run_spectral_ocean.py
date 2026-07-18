#!/usr/bin/env python3
"""PREM-ocean variant of run_spectral.py: same manifest model with
the top solid shell truncated at 6368 km and the PREM ocean
(rho 1.020, vp 1.450, 3 km) on top — exercises the G-A7 fluid-top
support on the real PREM topology. No DSM reference yet (tipsv
ocean leg queued); output is spectral_<model>_ocean.npz.

usage: run_spectral_ocean.py <root> <model> [Q] [q_sign] [lmax]
                             [nproc]
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
Q = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
Q_SIGN = float(sys.argv[4]) if len(sys.argv) > 4 else -1.0
LMAX = int(sys.argv[5]) if len(sys.argv) > 5 else 250
NPROC = int(sys.argv[6]) if len(sys.argv) > 6 else 48
R_OCEAN_BOT = 6368.0e3

man = json.load(open(os.path.join(ROOT, "manifest.json")))
layers_man = man["models"][MODEL]
LAYERS = [dict(r_top=ly["r_km"] * 1e3, rho=ly["mat"][0] * 1e3,
               vp=ly["mat"][1] * 1e3,
               vs=(ly["mat"][2] * 1e3 if len(ly["mat"]) > 2
                   else 0.0))
          for ly in layers_man]
assert abs(LAYERS[-1]["r_top"] - 6371.0e3) < 1.0
LAYERS[-1] = dict(LAYERS[-1], r_top=R_OCEAN_BOT)
LAYERS.append(dict(r_top=6371.0e3, rho=1020.0, vp=1450.0))

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
    print("spectral %s + PREM ocean (3 km): lmax %d, %d st x %d k,"
          " %d procs" % (MODEL, LMAX, len(ST_DIRS), imax, NPROC),
          flush=True)
    POLES = spectral_poles(LMAX)
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
    print("solves: %.1f s" % (time.time() - t1), flush=True)
    out = os.path.join(ROOT, "spectral_%s_ocean.npz" % MODEL)
    np.savez(out, sources=np.array(src_names), u=u_psv + u_sh,
             u_psv=u_psv, u_sh=u_sh, imax=imax,
             ocean_m=6371.0e3 - R_OCEAN_BOT)
    print("wrote", out)
