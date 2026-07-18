#!/usr/bin/env python3
"""Cumulative station anatomy at one frequency: complex per-l AZ06
Z contributions (m=0 PSV incl. the l=0 branch), cumulative
conj-ratio vs tipsv-on-the-same-staircase as a function of the
l-cut. Shows exactly which l-band walks the answer away from 1.
usage: probe_perl2.py [K=40]
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import (_entries, _icut,  # noqa: E402
                                  _mats_at, _sph0_jump, make_stack,
                                  spheroidal_unit,
                                  spheroidal_unit_l0)
from pyastroseis.spheroidal_ref import (L_SERIES,   # noqa: E402
                                        source_jumps,
                                        spheroidal_reconstruct)
from pyastroseis.toroidal_ref import source_frame   # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LMAX = 1400
IST = 5                                             # AZ06

w = 2 * np.pi * K / bc.TLEN + 1j * bc.OMEGAI
LAYERS = bc.build_layers(1.0e9)
stack = make_stack(LAYERS)
r0 = bc.R0
isrc = [i for i, e in enumerate(stack)
        if e["r_bot"] < r0 < e["r_top"]][0]
mats = _mats_at(stack, w, None, -1.0)
msrc = mats[isrc]
SRC = bc.src_xyz()
Qrot = source_frame(SRC)
M_sf = Qrot @ bc.mt_cart() @ Qrot.T
mzz = M_sf[2, 2]
DIRS = bc.station_dirs()
dirs_sf = (DIRS @ Qrot.T)[IST:IST + 1]
DY = bc.get_poles(LMAX)[0]


def station_z(l, coefU, coefV):
    Wu = np.zeros((LMAX + 1, 9), dtype=complex)
    Wv = np.zeros_like(Wu)
    Wu[l, 4] = coefU
    Wv[l, 4] = coefV
    u_sf = spheroidal_reconstruct(Wu, Wv, dirs_sf)
    u = u_sf @ Qrot                                  # (1, 3)
    rec = DIRS[IST]
    return complex(u[0] @ rec)


def read_spc(path):
    toks = []
    for line in open(path):
        toks += line.split()
    vals = [float(t) for t in toks]
    np0 = int(vals[1])
    body = vals[17:]
    u = np.zeros((3, np0 + 1), dtype=complex)
    for kk in range(len(body) // 7):
        b = body[7 * kk: 7 * kk + 7]
        i = int(b[0])
        u[0, i] = b[1] + 1j * b[2]
        u[1, i] = b[3] + 1j * b[4]
        u[2, i] = b[5] + 1j * b[6]
    return u


dsm = read_spc(os.path.join(HERE, "stair_dsm", "spc",
                            "%s.mrr.PSV.spc" % bc.STATIONS[IST]))
ref = dsm[0][K] * 1e3                                # -> metres

contrib = np.zeros(LMAX + 1, dtype=complex)
ents0 = _entries(mats, 0, r0, isrc)
U0 = spheroidal_unit_l0(w, ents0, _sph0_jump(w, r0, msrc))
contrib[0] = station_z(0, mzz * DY[0] * U0, 0.0)
for l in range(1, LMAX + 1):
    icut = min(_icut(stack, l, r0, 1.0e-12), isrc)
    ents = _entries(mats, icut, r0, isrc)
    zr_src = msrc["r_top"] if l >= L_SERIES else None
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J_y = source_jumps(l, w, r0, msrc["rho"], msrc["lam"],
                       msrc["mu"], F0y, F1y, zref_a=zr_src)
    U_y, V_y = spheroidal_unit(l, w, ents, J_y)
    qy = mzz * DY[l]
    contrib[l] = station_z(l, qy * U_y, qy * V_y)

cum = np.cumsum(contrib)
print("k=%d AZ06 Z: cumulative conj(ours)/tipsv-stair vs l-cut:"
      % K)
for cut in (0, 10, 20, 30, 40, 47, 55, 60, 80, 100, 150, 200,
            300, 420, 600, 900, 1100, 1400):
    r = np.conj(cum[cut]) / ref
    print("  l<=%4d: |r| %9.4f  ph %+7.2f" %
          (cut, abs(r), math.degrees(np.angle(r))))
print("largest per-l station contributions:")
order = np.argsort(np.abs(contrib))[::-1][:10]
for l in order:
    print("  l=%4d |c| %.3e ph %+7.2f" %
          (l, abs(contrib[l]),
           math.degrees(np.angle(contrib[l]))))
print("PROBE_PERL2 DONE")
