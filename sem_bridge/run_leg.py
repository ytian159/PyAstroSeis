#!/usr/bin/env python3
"""Spectral ak135-staircase hdur80 sweep (one ladder rung).

usage: run_leg.py <h_cap_km> <lmax> <nproc> [out_npz]

k = 1..IMAX on the DSM damped axis (TLEN 8192, omegai ln(100)/TLEN),
elastic (Q=None) to match SEM ATTENUATION=.false. and the elastic
DSM zones. Per-frequency lmax cap: high k needs large l but prunes
deep layers; low k needs few l — cost stays bounded at both ends.
"""
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import (spectral_poles,   # noqa: E402
                                  spectral_spectra)

H_KM = float(sys.argv[1])
LMAX = int(sys.argv[2])
NPROC = int(sys.argv[3])
OUT = (sys.argv[4] if len(sys.argv) > 4 else
       os.path.join(bc.HERE, "out_ak135",
                    "spectral_h%03d.npz" % int(H_KM)))

LAYERS = bc.build_layers(H_KM)
SRC = bc.src_xyz()
MTS = [bc.mt_cart()]
DIRS = bc.station_dirs()
ks = np.arange(1, bc.IMAX + 1)
W_ARR = 2.0 * np.pi * ks / bc.TLEN + 1j * bc.OMEGAI
POLES = None

# FLAT lmax at every frequency: the 50-km source's quasi-static
# l-series peaks at l ~ a/(a-r0)*1.5 ~ 190 with e-folding 127 at
# ALL k (rung-A lesson: truncating it mid-peak leaves O(1) junk;
# lmax ~ 1400 validated at 50 km). High l is cheap — pruning keeps
# only the near-surface layers.


def _worker(iw_chunk):
    t0 = time.time()
    psv = np.zeros((len(MTS), len(DIRS), 3, len(iw_chunk)),
                   dtype=complex)
    sh = np.zeros_like(psv)
    for j, iw in enumerate(iw_chunk):
        w = W_ARR[iw]
        out = spectral_spectra(LAYERS, SRC, MTS, [w], DIRS,
                               Q=None, lmax=LMAX,
                               poles=POLES)
        psv[:, :, :, j] = out["psv"][:, :, :, 0]
        sh[:, :, :, j] = out["sh"][:, :, :, 0]
    return iw_chunk, psv, sh, time.time() - t0


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("ak135 staircase h<=%g km: %d layers, lmax %d (flat),"
          " %d freqs, %d procs"
          % (H_KM, len(LAYERS), LMAX, len(W_ARR), NPROC),
          flush=True)
    POLES = bc.get_poles(LMAX)
    chunks = [list(range(i, bc.IMAX, NPROC)) for i in range(NPROC)]
    chunks = [c for c in chunks if c]
    u_psv = np.zeros((len(MTS), len(DIRS), 3, bc.IMAX + 1),
                     dtype=complex)
    u_sh = np.zeros_like(u_psv)
    t1 = time.time()
    with mp.Pool(NPROC) as pool:
        for iw_chunk, psv, sh, dtw in pool.imap_unordered(
                _worker, chunks):
            for j, iw in enumerate(iw_chunk):
                u_psv[:, :, :, iw + 1] = psv[:, :, :, j]
                u_sh[:, :, :, iw + 1] = sh[:, :, :, j]
    print("solves: %.1f s wall" % (time.time() - t1), flush=True)
    np.savez(OUT, u=u_psv + u_sh, u_psv=u_psv, u_sh=u_sh,
             h_km=H_KM, lmax=LMAX, nlayers=len(LAYERS),
             imax=bc.IMAX, tlen=bc.TLEN, omegai=bc.OMEGAI)
    print("wrote", OUT)
