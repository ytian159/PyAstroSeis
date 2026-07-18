#!/usr/bin/env python3
"""Audit stage 10: lp=1 residual at delta=3000 is noise on a tiny
channel: jitter r0 by 0.5 m -> lp=1 ratio scatters, lp>=2 stable.
Also report |dU_fd| per lp to show the lp=1 amplitude gap."""
import numpy as np
from common import (A, LAUD, LMAX, ct, st, proj, r0, tfe, w, lay)
from pyastroseis import spectral as spsp

zhat = np.array([0.0, 0.0, 1.0])
Mrr = np.diag([0.0, 0.0, 1.0e18])
kw = dict(Q=50.0, q_sign=-1.0, lmax=LMAX)
dirs = np.stack([st, 0.0 * st, ct], axis=1)
that = np.stack([ct, 0.0 * ct, -st], axis=1)
dl = 3000.0
hLM = dl * np.sqrt(4.0 * np.pi / 3.0)

for jit in (0.0, 0.5, 1.1):
    r0j = r0 + jit
    src = np.array([0.0, 0.0, r0j])
    du = tfe.relief_spectra(lay, src, [Mrr], np.array([w]), dirs,
                            relief=[("top", 1, 0, hLM)], **kw)
    du_e = (du["psv"] + du["sh"])[0, :, :, 0]
    du_f = 0.0
    for s in (+1.0, -1.0):
        dp = dirs + s * (dl / A) * st[:, None] * that
        dp /= np.linalg.norm(dp, axis=1)[:, None]
        us = spsp.spectral_spectra(lay, src - s * dl * zhat, [Mrr],
                                   np.array([w]), dp, l0=False,
                                   **kw)
        du_f = du_f + 0.5 * s * (us["psv"] + us["sh"])[0, :, :, 0]
    Ue, Ve = proj(np.einsum('nc,nc->n', dirs, du_e),
                  np.einsum('nc,nc->n', that, du_e))
    Uf, Vf = proj(np.einsum('nc,nc->n', dirs, du_f),
                  np.einsum('nc,nc->n', that, du_f))
    print("jit=%.1f m:" % jit)
    for lp in range(1, LAUD + 1):
        rU = Ue[lp] / Uf[lp]
        print("  lp=%d |dU_fd|=%.2e  eng/fd U %6.3f<%5.0f"
              % (lp, abs(Uf[lp]), abs(rU),
                 np.degrees(np.angle(rU))))
