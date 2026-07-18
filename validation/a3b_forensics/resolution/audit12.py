#!/usr/bin/env python3
"""Audit 12: lmax dependence of the Mrt gate at delta=3000."""
import os, sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
zhat = np.array([0.0, 0.0, 1.0])
Mrt = np.zeros((3, 3)); Mrt[0, 2] = Mrt[2, 0] = 1.0e18
src = np.array([0.0, 0.0, r0])
th = np.linspace(0.35, np.pi - 0.35, 9)
dirs = np.stack([np.sin(th), 0.0 * th, np.cos(th)], axis=1)
that = np.stack([np.cos(th), 0.0 * th, -np.sin(th)], axis=1)
dl = 3000.0
hLM = dl * np.sqrt(4.0 * np.pi / 3.0)
for lmax in (12, 16, 19):
    kw = dict(Q=50.0, q_sign=-1.0, lmax=lmax)
    du_fd = {}
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * np.sin(th)[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(lay, src - s * dl * zhat, [Mrt],
                                   np.array([w]), dp, l0=False, **kw)
        for part in ("psv", "sh"):
            du_fd[part] = du_fd.get(part, 0.0) + 0.5 * s * us[part][0, :, :, 0]
    du = tfe.relief_spectra(lay, src, [Mrt], np.array([w]), dirs,
                            relief=[("top", 1, 0, hLM)], **kw)
    out = []
    for part in ("psv", "sh"):
        e = (np.abs(du[part][0, :, :, 0] - du_fd[part]).max()
             / np.abs(du_fd[part]).max())
        out.append("%s %.3e" % (part, e))
    print("lmax=%2d  " % lmax + "   ".join(out))
