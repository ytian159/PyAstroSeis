#!/usr/bin/env python3
"""Audit stage 3: is the r0-jump SOURCE-SHAPED, and which route is
wrong?

 B  proportionality: measured jump[lp] vs CS_lp * J_lp(r0)
    (a single complex constant across lp AND components would mean
    the two routes differ by a small extra point source at src)
 C  surface discrepancy (Uf - Ue) vs CS_lp * u0_surf[lp]:
    same constant?
 D  FREE-SPACE translation covariance of the source-load family
    J(b): u^{(b-d)}(x) == u^{(b)}(x + d zhat) exactly, no gate, no
    relief, no boundaries. Per-lp projections; residual compared
    against the source-shaped direction.
"""
import numpy as np
from common import (A, CS, DY, Jvec, LAUD, LMAX, Lpv, NT, Yg, dYg,
                    ct, st, delta, hLM, lam, m0, mats, mu, proj,
                    r0, rho, solve_l, solve_l_free, tfe, u_line,
                    w, w2, y_of, zgrad_proj, _entries, _Ysolid,
                    spheroidal_unit, source_jumps)

zhat = np.array([0.0, 0.0, 1.0])
SOL0 = {l: solve_l(l, r0) for l in range(1, LMAX + 1)}
SOLm = {l: solve_l(l, r0 - delta) for l in range(1, LMAX + 1)}
SOLp = {l: solve_l(l, r0 + delta) for l in range(1, LMAX + 1)}


def exact_u1_lp(r):
    Uw, Vw = zgrad_proj(r, SOL0)
    Uw, Vw = -delta * Uw, -delta * Vw
    Uv = np.zeros(LMAX + 3, dtype=complex)
    Vv = np.zeros(LMAX + 3, dtype=complex)
    for l in range(1, LMAX + 1):
        ym = y_of(l, SOLm[l], r) * CS[l]
        yp_ = y_of(l, SOLp[l], r) * CS[l]
        Uv[l] = 0.5 * (ym[0] - yp_[0])
        Vv[l] = 0.5 * (ym[1] - yp_[1])
    return Uw + Uv, Vw + Vv


# ---- refit the jump at r0 (as audit2) ---------------------------
r_above = np.linspace(r0 + 500.0, A - 500.0, 12)
r_below = np.linspace(0.35 * A, r0 - 500.0, 10)
samp_a = [exact_u1_lp(r) for r in r_above]
samp_b = [exact_u1_lp(r) for r in r_below]
jumps = {}
for lp in range(1, LAUD + 1):
    Ca, ba = [], []
    for i, r in enumerate(r_above):
        Y4 = _Ysolid(lp, w, r, m0, ("j", "y"), None)
        Ca.append(Y4[0]); ba.append(samp_a[i][0][lp])
        Ca.append(Y4[1]); ba.append(samp_a[i][1][lp])
    Ca, ba = np.array(Ca), np.array(ba)
    sca = np.max(np.abs(Ca), axis=0)
    cA = np.linalg.lstsq(Ca / sca, ba, rcond=None)[0] / sca
    Cb, bb = [], []
    for i, r in enumerate(r_below):
        Y2 = _Ysolid(lp, w, r, m0, ("j",), None)
        Cb.append(Y2[0]); bb.append(samp_b[i][0][lp])
        Cb.append(Y2[1]); bb.append(samp_b[i][1][lp])
    Cb, bb = np.array(Cb), np.array(bb)
    scb = np.max(np.abs(Cb), axis=0)
    cB = np.linalg.lstsq(Cb / scb, bb, rcond=None)[0] / scb
    jumps[lp] = (_Ysolid(lp, w, r0, m0, ("j", "y"), None) @ cA
                 - _Ysolid(lp, w, r0, m0, ("j",), None) @ cB)

print("== B: jump[lp] / (CS_lp J_lp(r0))  (component-wise) ==")
print("lp     U-ratio            V-ratio            R-ratio"
      "            S-ratio")
for lp in range(1, LAUD + 1):
    Jl = Jvec(lp, r0) * CS[lp]
    rr = jumps[lp] / Jl
    print("%2d " % lp + "  ".join(
        "%9.3e<%4.0f" % (abs(v), np.degrees(np.angle(v)))
        for v in rr))

# ---- C: surface discrepancy direction ---------------------------
# engine du coefficients (replicated) and FD du coefficients
ang = tfe.ReliefCouplings(1, 0, 0, LMAX)
keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp", "S", "T", "V", "W",
        "ciso", "cV", "cW")
dc = {k: np.zeros(LMAX + 1, dtype=complex) for k in keys}
for l in range(1, LMAX + 1):
    ents = _entries(mats, 0, r0, 0)
    Ua, Va, info = spheroidal_unit(l, w, ents, Jvec(l, r0),
                                   full=True)
    sc_ = tfe._side_coeffs(l, w, A, m0, info["surface"] * CS[l],
                           None, yp4=info["surface_p"] * CS[l])
    for k in keys:
        dc[k][l] = sc_[k]
jU, jV, jR, jS, jW, jT = tfe._jump_rows(hLM, A, ang, dc)
advU = hLM * (ang.G0 @ dc["Up"])
advV = hLM * (ang.GA_v @ dc["Vp"])
Ue = np.zeros(LMAX + 1, dtype=complex)
Ve = np.zeros(LMAX + 1, dtype=complex)
for lp in range(1, LMAX + 1):
    entsp = _entries(mats, 0, r0, 0)
    U1, V1 = spheroidal_unit(lp, w, entsp,
                             np.zeros(4, dtype=complex),
                             bc_rhs=(jR[lp], jS[lp]))
    Ue[lp] = U1 + advU[lp]
    Ve[lp] = V1 + advV[lp]
# FD reference coefficients = exact_u1 at A + advection projections
UA, VA = exact_u1_lp(A)
ur, ut, urp, utp, _, _ = u_line(A, SOL0)
Uad, Vad = proj(delta * ct * urp, delta * ct * utp)
Uf = UA[:LMAX + 1] + Uad[:LMAX + 1]
Vf = VA[:LMAX + 1] + Vad[:LMAX + 1]

print("\n== C: (Uf-Ue)/(CS u0_surf), (Vf-Ve)/(CS v0_surf) ==")
print("lp     U-direction        V-direction")
for lp in range(1, LAUD + 1):
    y0s = y_of(lp, SOL0[lp], A) * CS[lp]
    cu = (Uf[lp] - Ue[lp]) / y0s[0]
    cv = (Vf[lp] - Ve[lp]) / y0s[1]
    print("%2d  %9.3e<%4.0f   %9.3e<%4.0f"
          % (lp, abs(cu), np.degrees(np.angle(cu)), abs(cv),
             np.degrees(np.angle(cv))))

# ---- D: free-space translation covariance -----------------------
F0 = {l: solve_l_free(l, r0) for l in range(1, LMAX + 1)}
Fm = {l: solve_l_free(l, r0 - delta) for l in range(1, LMAX + 1)}
Fp = {l: solve_l_free(l, r0 + delta) for l in range(1, LMAX + 1)}
rF = 1.03 * A
# LHS: source-shift FD (per-l diagonal)
Ulhs = np.zeros(LMAX + 3, dtype=complex)
Vlhs = np.zeros(LMAX + 3, dtype=complex)
for l in range(1, LMAX + 1):
    ym = y_of(l, Fm[l], rF) * CS[l]
    yp_ = y_of(l, Fp[l], rF) * CS[l]
    Ulhs[l] = 0.5 * (ym[0] - yp_[0])
    Vlhs[l] = 0.5 * (ym[1] - yp_[1])
# RHS: +delta * d/dz of the b=r0 field, projected at rF
Ug, Vg = zgrad_proj(rF, F0)
Urhs, Vrhs = delta * Ug, delta * Vg
print("\n== D: FREE SPACE  u^(b-d)(x) == u^(b)(x+d zhat) ==")
print("lp   |U_lhs|      lhs/rhs(U)      lhs/rhs(V)"
      "     resid_U/(CS u0)   resid_V/(CS v0)")
for lp in range(1, LAUD + 1):
    y0f = y_of(lp, F0[lp], rF) * CS[lp]
    rU = Ulhs[lp] / Urhs[lp]
    rV = Vlhs[lp] / Vrhs[lp]
    cu = (Ulhs[lp] - Urhs[lp]) / y0f[0]
    cv = (Vlhs[lp] - Vrhs[lp]) / y0f[1]
    print("%2d  %.3e  %6.3f<%5.0f  %6.3f<%5.0f  %9.3e<%4.0f"
          "  %9.3e<%4.0f"
          % (lp, abs(Ulhs[lp]), abs(rU),
             np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV)),
             abs(cu), np.degrees(np.angle(cu)),
             abs(cv), np.degrees(np.angle(cv))))
print("\n(delta/r0 = %.4e ; a pure moment-drift exponent q would"
      " give ratios q*delta/r0)" % (delta / r0))
