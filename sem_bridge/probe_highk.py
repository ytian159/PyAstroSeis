#!/usr/bin/env python3
"""Pre-sweep gate: solve a few high-k frequencies on the h40 stack
and check the spectrum magnitude against the DSM spc (DSM is in km,
we are in m, so a healthy leg sits within ~[1e2, 1e4] of DSM given
component-basis mixing). Exits nonzero on failure so the driver
aborts before burning the full sweep."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import bridge_common as bc                          # noqa: E402
from pyastroseis.spectral import (spectral_poles,   # noqa: E402
                                  spectral_spectra)


def read_spc(path):
    toks = []
    for line in open(path):
        toks += line.split()
    vals = [float(t) for t in toks]
    np0 = int(vals[1])
    body = vals[17:]
    u = np.zeros((3, np0 + 1), dtype=complex)
    for k in range(len(body) // 7):
        b = body[7 * k: 7 * k + 7]
        i = int(b[0])
        u[0, i] = b[1] + 1j * b[2]
        u[1, i] = b[3] + 1j * b[4]
        u[2, i] = b[5] + 1j * b[6]
    return u


L = bc.build_layers(40)
poles = spectral_poles(420)
dsm = read_spc("/pscratch/sd/y/ytian159/dfdm_3d_ppw_opt/bench/"
               "dsm_reference/ak135/spc/AZ06.mrr.PSV.spc")
ok = True
for k in (40, 80, 160, 240):
    w = 2 * np.pi * k / bc.TLEN + 1j * bc.OMEGAI
    ln = abs(w) * bc.A_EARTH / 3460.0
    lm = int(min(420, max(60, 1.15 * ln + 30)))
    out = spectral_spectra(L, bc.src_xyz(), [bc.mt_cart()],
                           [w], bc.station_dirs(), Q=None,
                           lmax=lm, poles=poles)
    au = np.abs(out["psv"][0][5] + out["sh"][0][5]).max()
    ad = np.abs(dsm[:, k]).max()
    ratio = au / ad
    good = 1e2 < ratio < 1e4
    ok = ok and good
    print("probe k=%3d lmax %3d: ours %.3e dsm %.3e ratio %.2e %s"
          % (k, lm, au, ad, ratio, "OK" if good else "FAIL"),
          flush=True)
print("PROBE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
