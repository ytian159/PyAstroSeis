#!/usr/bin/env python3
"""Per-l anatomy of one frequency: mirror spectral_spectra's m=0
PSV path at k=K on the 127-layer staircase and print each l's
summed contribution |qy*U_y| (+ the running cumulative surface
value), l = 1..1400. Shape of the anomaly: isolated spikes =
per-(l,w) solve blowups; broad plateau = systematic term.
usage: probe_perl.py [K=40]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import (_entries, _icut,  # noqa: E402
                                  _mats_at, make_stack,
                                  spheroidal_unit)
from pyastroseis.spheroidal_ref import (L_SERIES,   # noqa: E402
                                        source_jumps)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 40
LMAX = 1400
LAYERS = bc.build_layers(1.0e9)

w = 2 * np.pi * K / bc.TLEN + 1j * bc.OMEGAI
stack = make_stack(LAYERS)
r0 = bc.R0
isrc = [i for i, e in enumerate(stack)
        if e["r_bot"] < r0 < e["r_top"]][0]
mats = _mats_at(stack, w, None, -1.0)
msrc = mats[isrc]
DY = bc.get_poles(LMAX)[0]

rows = []
for l in range(1, LMAX + 1):
    L = l * (l + 1.0)
    icut = min(_icut(stack, l, r0, 1.0e-12), isrc)
    ents = _entries(mats, icut, r0, isrc)
    zr_src = msrc["r_top"] if l >= L_SERIES else None
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J_y = source_jumps(l, w, r0, msrc["rho"], msrc["lam"],
                       msrc["mu"], F0y, F1y, zref_a=zr_src)
    U_y, V_y = spheroidal_unit(l, w, ents, J_y)
    rows.append((l, len(ents), abs(DY[l] * U_y),
                 abs(DY[l] * V_y)))

arr = np.array([(r[2], r[3]) for r in rows])
print("k=%d: per-l |DY*U_y| summary" % K)
peak = arr[:, 0].max()
for lo, hi in ((1, 20), (20, 60), (60, 120), (120, 200),
               (200, 300), (300, 420), (420, 600), (600, 900),
               (900, 1400)):
    seg = arr[lo - 1:hi, 0]
    print(" l %4d-%4d: max %.3e med %.3e  (max/l %d)"
          % (lo, hi, seg.max(), np.median(seg),
             lo + int(seg.argmax())))
print("global peak %.3e" % peak)
print("top 12 l by |DY*U_y|:")
order = np.argsort(arr[:, 0])[::-1][:12]
for i in order:
    l, ne, au, av = rows[i]
    print("  l=%4d ents %2d |DY U| %.3e |DY V| %.3e"
          % (l, ne, au, av))
print("PROBE_PERL DONE")
