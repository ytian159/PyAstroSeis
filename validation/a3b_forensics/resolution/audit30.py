#!/usr/bin/env python3
"""Audit 30: solid-solid weld translation gate at m=0 (Mrr),
delta=3000, total field. Plus homog-with-artificial-split control
(must be identical to homog => pure delta-cancellation)."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

LMAX = 12
r0 = A - 637.0e3
w = wk(90.0)
zhat = np.array([0.0, 0.0, 1.0])
Mrr = np.diag([0.0, 0.0, 1.0e18])
src = np.array([0.0, 0.0, r0])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
th = np.linspace(0.35, np.pi - 0.35, 9)
dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
that = np.stack([np.cos(th), 0.0 * th, -np.sin(th)], axis=1)


def gate(layers, relief_locs, dl, M):
    c10 = dl * np.sqrt(4.0 * np.pi / 3.0)
    du_fd = 0.0
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * np.sin(th)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(layers, src - s * dl * zhat,
                                   [M], np.array([w]), dp,
                                   l0=False, **kw)
        du_fd = du_fd + 0.5 * s * (us["psv"]
                                   + us["sh"])[0, :, :, 0]
    du = tfe.relief_spectra(layers, src, [M], np.array([w]),
                            dirs,
                            relief=[(loc, 1, 0, c10)
                                    for loc in relief_locs], **kw)
    tot_e = (du["psv"] + du["sh"])[0, :, :, 0]
    return (np.abs(tot_e - du_fd).max() / np.abs(du_fd).max())


MID = dict(rho=4600.0, vp=10000.0, vs=5800.0)
L2 = [dict(r_top=4925.5e3, **MID), dict(r_top=A, **SH)]
print("solid weld, Mrr (m=0), delta=3000: %.3e"
      % gate(L2, (4925.5e3, "top"), 3000.0, Mrr))
L2s = [dict(r_top=4925.5e3, **SH), dict(r_top=A, **SH)]
print("artificial split (identical mats), Mrr, delta=3000: %.3e"
      % gate(L2s, (4925.5e3, "top"), 3000.0, Mrr))
