#!/usr/bin/env python3
"""Audit 16: m=1 receiver-advection projections vs pointwise
truth."""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
exec(open("audit15.py").read().split("# TRUTH")[0])  # setup only

ang = tfe.ReliefCouplings(1, 0, 1, LMAX)
keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp", "S", "T", "V", "W",
        "ciso", "cV", "cW")
dc = {k: np.zeros(LMAX + 1, dtype=complex) for k in keys}
for l in range(1, LMAX + 1):
    y4 = y4_at(l, A)
    hh = 0.5
    yp4 = (y4_at(l, A + hh) - y4_at(l, A - hh)) / (2 * hh)
    wt2 = wt_at(l, A)
    sc_ = tfe._side_coeffs(l, w, A, m0, y4, wt2, yp4=yp4)
    for k in keys:
        dc[k][l] = sc_[k]

hh = 0.5
ur = np.zeros(NT, dtype=complex); ut = np.zeros(NT, dtype=complex)
up = np.zeros(NT, dtype=complex)
for l in range(1, LMAX + 1):
    yp4 = (y4_at(l, A + hh) - y4_at(l, A - hh)) / (2 * hh)
    wtp = (wt_at(l, A + hh) - wt_at(l, A - hh)) / (2 * hh)
    ur += yp4[0] * Sh["Y"][l]
    ut += yp4[1] * Sh["Bt"][l] + wtp[0] * Sh["Ct"][l]
    up += yp4[1] * Sh["Bp"][l] + wtp[0] * Sh["Cp"][l]
advr_t = delta * ct * ur
advt_t = delta * ct * ut
advp_t = delta * ct * up

def proj_disp(fr, ft, fp):
    U = np.einsum('ln,n->l', np.conj(Sh["Y"]), fr * w2)
    V = (np.einsum('ln,n->l', np.conj(Sh["Bt"]), ft * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Bp"]), fp * w2)) / Lp
    W = (np.einsum('ln,n->l', np.conj(Sh["Ct"]), ft * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Cp"]), fp * w2))
    return U, V, W

Ut, Vt, Wt_ = proj_disp(advr_t, advt_t, advp_t)
advU = hLM * (ang.G0 @ dc["Up"])
advV = hLM * (ang.GA_v @ dc["Vp"] + ang.GA_w @ dc["Wp"])
advW = hLM * (ang.GC_v @ dc["Vp"] + ang.GC_w @ dc["Wp"])
print("m=1 receiver advection: engine vs pointwise truth")
for lp in range(1, LAUD + 1):
    rU = advU[lp] / Ut[lp]; rV = advV[lp] / Vt[lp]
    rW = advW[lp] / Wt_[lp]
    print("%2d  U %7.4f<%5.0f  V %7.4f<%5.0f  W %7.4f<%5.0f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV)), abs(rW),
             np.degrees(np.angle(rW))))
