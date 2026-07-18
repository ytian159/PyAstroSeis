#!/usr/bin/env python3
"""Three-way: collocation vs TFE engine vs translation-FD,
mrr on-axis, m=0, Y10 top relief, homog ball, lmax=14."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from tests.test_spectral import A, SH, wk

QSH = 50.0
LMAX = 14
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
src = np.array([0.0, 0.0, r0])
Mrr = np.diag([0.0, 0.0, 1.0e18])
delta = 30.0
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)
kw = dict(Q=QSH, q_sign=-1.0, lmax=LMAX)
zhat = np.array([0.0, 0.0, 1.0])

NT, NPH = 64, 8
xg, wg = np.polynomial.legendre.leggauss(NT)
tg = np.arccos(xg)
pg = 2.0 * np.pi * np.arange(NPH) / NPH
TH, PH = np.meshgrid(tg, pg, indexing="ij")
dirs = np.stack([np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH),
                 np.cos(TH)], axis=-1).reshape(-1, 3)
thv, phv = TH.ravel(), PH.ravel()
that = np.stack([np.cos(thv) * np.cos(phv),
                 np.cos(thv) * np.sin(phv), -np.sin(thv)], axis=1)
phat = np.stack([-np.sin(phv), np.cos(phv), 0 * phv], axis=1)
wq = np.repeat(wg, NPH) * (2.0 * np.pi / NPH)


def field(src_xyz, dd):
    u = spsp.spectral_spectra(lay, src_xyz, [Mrr], np.array([w]),
                              dd, l0=False, **kw)
    return (u["psv"] + u["sh"])[0, :, :, 0]


def vecproj(du, lp):
    Tg = tfe._shapes(0, LMAX, np.cos(tg), np.sin(tg))
    Yp = np.repeat(Tg["Y"][lp], NPH)
    Btp = np.repeat(Tg["Bt"][lp], NPH)
    Lp = lp * (lp + 1.0)
    dur = np.einsum('nc,nc->n', dirs, du)
    dut = np.einsum('nc,nc->n', that, du)
    return (np.sum(np.conj(Yp) * dur * wq),
            np.sum(np.conj(Btp) * dut * wq) / Lp)


du_fd = 0.0
for s in (+1.0, -1.0):
    dp = dirs + s * (delta / A) * np.sin(thv)[:, None] * that
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    du_fd = du_fd + 0.5 * s * field(src - s * delta * zhat, dp)

du_e = tfe.relief_spectra(lay, src, [Mrr], np.array([w]), dirs,
                          relief=[("top", 1, 0, hLM)], **kw)
du_e = (du_e["psv"] + du_e["sh"])[0, :, :, 0]

col = np.load("/tmp/claude-111957/-pscratch-sd-y-ytian159/a12d7ec3-102b-49eb-84c6-6b6d4c1ca8a4/scratchpad/colloc_res.npy")
print("lp |  U: colloc      engine       transFD   | eng/col "
      "fd/col")
for lp in range(1, 9):
    cU, cV = col[lp - 1]
    fU, fV = vecproj(du_fd, lp)
    eU, eV = vecproj(du_e, lp)
    print("%d U %12.4e %12.4e %12.4e  %6.3f<%4.0f %6.3f<%4.0f"
          % (lp, cU.real, eU.real, fU.real, abs(eU / cU),
             np.degrees(np.angle(eU / cU)), abs(fU / cU),
             np.degrees(np.angle(fU / cU))))
    print("  V %12.4e %12.4e %12.4e  %6.3f<%4.0f %6.3f<%4.0f"
          % (cV.real, eV.real, fV.real, abs(eV / cV),
             np.degrees(np.angle(eV / cV)), abs(fV / cV),
             np.degrees(np.angle(fV / cV))))
