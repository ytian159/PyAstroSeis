#!/usr/bin/env python3
"""Localize the LinAlgError on the h40 stack: per-l loop at given
k values, catch the singular solve, report (k, l), entry count and
any non-finite basis entries.
usage: probe_singular.py k [k ...]
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
import pyastroseis.spectral as sp                   # noqa: E402

LMAX = 1400
LAYERS = bc.build_layers(40)
stack = make_stack(LAYERS)
r0 = bc.R0
isrc = [i for i, e in enumerate(stack)
        if e["r_bot"] < r0 < e["r_top"]][0]

for karg in sys.argv[1:]:
    k = int(karg)
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    mats = _mats_at(stack, w, None, -1.0)
    msrc = mats[isrc]
    vp_src = np.sqrt((msrc["lam"] + 2.0 * msrc["mu"])
                     / msrc["rho"])
    zp0 = abs(w / vp_src) * r0
    bad = []
    with np.errstate(all="ignore"):
        for l in range(1, LMAX + 1):
            icut = min(_icut(stack, l, r0, 1.0e-12), isrc)
            ents = _entries(mats, icut, r0, isrc)
            zr_src = msrc["r_top"] if l >= L_SERIES else None
            if (sp.JUMP_UNSCALED and zr_src is not None
                    and sp._lndf(2 * l + 1) - (l + 1.0)
                    * np.log(zp0) < sp._LN_OVERFLOW):
                zr_src = None
            F0y = np.array([0, 0, 1.0 / r0 ** 2, 0],
                           dtype=complex)
            F1y = np.array([0, 0, 2.0 / r0 ** 3, 0],
                           dtype=complex)
            try:
                J = source_jumps(l, w, r0, msrc["rho"],
                                 msrc["lam"], msrc["mu"], F0y,
                                 F1y, zref_a=zr_src)
                if not np.all(np.isfinite(J)):
                    bad.append((l, len(ents), "J nonfinite"))
                    continue
                U, V = spheroidal_unit(l, w, ents, J)
                if not (np.isfinite(U) and np.isfinite(V)):
                    bad.append((l, len(ents), "UV nonfinite"))
            except np.linalg.LinAlgError:
                bad.append((l, len(ents), "singular"))
    if bad:
        print("k=%d: %d bad l's; first/last:" % (k, len(bad)),
              bad[:3], bad[-2:])
    else:
        print("k=%d: clean" % k)
print("PROBE_SINGULAR DONE")
