#!/usr/bin/env python3
"""Audit 15: m=1 (Mrt) traction-row truth test WITH SH content —
the conversion-content check the original forensics (all m=0)
never ran. Engine jR/jS/jT vs projections of the pointwise
translation truth t1 = -delta (zhat.grad sigma0).rhat at the free
surface, plus per-PIECE isolation:
   advection piece: -h d/dr t0      vs -h(GA_*@[Sp,Tp]) rows
   tilt piece: (1/A) sigma0.grad1 h vs (h/A)(Hiso/HV/HW) rows
"""
import sys
import numpy as np
sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  spheroidal_unit, _Ysolid, _Ytor)
from pyastroseis.spheroidal_ref import (source_jumps,
                                        spheroidal_pole)
from pyastroseis.toroidal_modes import _pole_coupling
from tests.test_spectral import A, SH, wk

LMAX = 10
LAUD = 8
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
mats = _mats_at(make_stack(lay), w, 50.0, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
delta = 30.0
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)
M0 = 1.0e18
DY, DG = spheroidal_pole(LMAX)
poleT = _pole_coupling(LMAX)
CSr = {l: M0 * DG[1][l][0] for l in range(1, LMAX + 1)}
CSt = {l: M0 * poleT[1][l][0] for l in range(1, LMAX + 1)}


# ---- ball solves keeping coefficients (PSV 6x6, SH 3x3) ---------
def psv_solve(l):
    ents = _entries(mats, 0, r0, 0)
    L = l * (l + 1.0)
    F0v = np.array([0, 0, 0, 1.0 / (L * r0 * r0)], dtype=complex)
    F1v = np.array([0, 0, -1.0 / r0 ** 3, 3.0 / (L * r0 ** 3)],
                   dtype=complex)
    J = source_jumps(l, w, r0, rho, lam, mu, F0v, F1v)
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ysolid(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ysolid(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    bv = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        Am[row:row + 4, ofs[i + 1]:ofs[i + 2]] = Yb[i + 1]
        Am[row:row + 4, ofs[i]:ofs[i + 1]] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row:row + 4] = J
        row += 4
    Am[row, ofs[-2]:ofs[-1]] = Yt[-1][2]
    Am[row + 1, ofs[-2]:ofs[-1]] = Yt[-1][3]
    sc = np.max(np.abs(Am), axis=0)
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs)


def tor_solve(l):
    ents = _entries(mats, 0, r0, 0)
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        Yb.append(_Ytor(l, w, e["r_bot"], e["mat"], kinds, None)
                  if i > 0 else None)
        Yt.append(_Ytor(l, w, e["r_top"], e["mat"], kinds, None))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    Am = np.zeros((n, n), dtype=complex)
    bv = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        Am[row:row + 2, ofs[i + 1]:ofs[i + 2]] = Yb[i + 1]
        Am[row:row + 2, ofs[i]:ofs[i + 1]] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row] = 1.0 / (mu * r0 * r0)
        row += 2
    Am[row, ofs[-2]:ofs[-1]] = Yt[-1][1]
    sc = np.max(np.abs(Am), axis=0)
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs)


PS = {l: psv_solve(l) for l in range(1, LMAX + 1)}
TS = {l: tor_solve(l) for l in range(1, LMAX + 1)}


def y4_at(l, r):
    s = PS[l]
    i = 1 if r >= r0 else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return (_Ysolid(l, w, r, m0, kinds, None)
            @ s["x"][s["ofs"][i]:s["ofs"][i + 1]]) * CSr[l]


def wt_at(l, r):
    s = TS[l]
    i = 1 if r >= r0 else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return (_Ytor(l, w, r, m0, kinds, None)
            @ s["x"][s["ofs"][i]:s["ofs"][i + 1]]) * CSt[l]


# ---- pointwise m=1 sigma builder --------------------------------
def sigma_sph(r, Sh, i, hfd=0.5):
    """3x3 sigma in (r,t,p) basis at (r, theta_i), m=1 complex."""
    sig = np.zeros((3, 3), dtype=complex)
    for l in range(1, LMAX + 1):
        U, V, R, Ssf = y4_at(l, r)
        yp = (y4_at(l, r + hfd) - y4_at(l, r - hfd)) / (2 * hfd)
        Wt, Tt = wt_at(l, r)
        L = l * (l + 1.0)
        div = yp[0] + 2.0 * U / r - L * V / r
        ciso = lam * div + 2.0 * mu * U / r
        cV = mu * V / r
        cW = mu * Wt / r
        Y = Sh["Y"][l][i]
        Bt, Bp = Sh["Bt"][l][i], Sh["Bp"][l][i]
        Ct, Cp = Sh["Ct"][l][i], Sh["Cp"][l][i]
        eBtt, eBpp, eBtp = (Sh["eB"][0][l][i], Sh["eB"][1][l][i],
                            Sh["eB"][2][l][i])
        eCtt, eCpp, eCtp = (Sh["eC"][0][l][i], Sh["eC"][1][l][i],
                            Sh["eC"][2][l][i])
        sig[0, 0] += R * Y
        srt = Ssf * Bt + Tt * Ct
        srp = Ssf * Bp + Tt * Cp
        sig[0, 1] += srt
        sig[1, 0] += srt
        sig[0, 2] += srp
        sig[2, 0] += srp
        sig[1, 1] += ciso * Y + cV * eBtt + cW * eCtt
        sig[2, 2] += ciso * Y + cV * eBpp + cW * eCpp
        stp = cV * eBtp + cW * eCtp
        sig[1, 2] += stp
        sig[2, 1] += stp
    return sig


NT = 64
xg, wg = np.polynomial.legendre.leggauss(NT)
tg = np.arccos(xg)
ct, st = xg, np.sqrt(1.0 - xg * xg)
Sh = tfe._shapes(1, LMAX + 1, ct, st)
w2 = 2.0 * np.pi * wg
ll = np.arange(LMAX + 2)
Lp = ll * (ll + 1.0)
Lp[0] = 1.0


def proj_rows(tr, tt, tp):
    R = np.einsum('ln,n->l', np.conj(Sh["Y"]), tr * w2)
    S = (np.einsum('ln,n->l', np.conj(Sh["Bt"]), tt * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Bp"]), tp * w2)) / Lp
    T = (np.einsum('ln,n->l', np.conj(Sh["Ct"]), tt * w2)
         + np.einsum('ln,n->l', np.conj(Sh["Cp"]), tp * w2))
    return R, S, T


def sig_to_cart(sig, i):
    rh = np.array([st[i], 0.0, ct[i]])
    th = np.array([ct[i], 0.0, -st[i]])
    ph = np.array([0.0, 1.0, 0.0])
    Q = np.stack([rh, th, ph])
    return Q.T @ sig @ Q


# TRUTH: t1 = -delta dz(sigma).rhat, and its two pieces
eps = 2.0
t1r = np.zeros(NT, dtype=complex)
t1t = np.zeros(NT, dtype=complex)
t1p = np.zeros(NT, dtype=complex)
advr = np.zeros(NT, dtype=complex)
advt = np.zeros(NT, dtype=complex)
advp = np.zeros(NT, dtype=complex)
tilr = np.zeros(NT, dtype=complex)
tilt_ = np.zeros(NT, dtype=complex)
tilp = np.zeros(NT, dtype=complex)
for i in range(NT):
    # z-FD of cartesian sigma along the vertical through the
    # surface point (evaluate sigma at radius |x +- eps zhat|,
    # angle theta' of the displaced point)
    x = A * np.array([st[i], 0.0, ct[i]])
    sig_c = {}
    for s_, dz in (("p", +eps), ("m", -eps)):
        xx = x + np.array([0.0, 0.0, dz])
        rr_ = np.linalg.norm(xx)
        cth = xx[2] / rr_
        sth = np.hypot(xx[0], xx[1]) / rr_
        Sh2 = tfe._shapes(1, LMAX + 1, np.array([cth]),
                          np.array([sth]))
        sig = sigma_sph(rr_, Sh2, 0)
        rh2 = np.array([sth, 0.0, cth])
        th2 = np.array([cth, 0.0, -sth])
        ph2 = np.array([0.0, 1.0, 0.0])
        Q2 = np.stack([rh2, th2, ph2])
        sig_c[s_] = Q2.T @ sig @ Q2
    dsig = (sig_c["p"] - sig_c["m"]) / (2.0 * eps)
    rh = np.array([st[i], 0.0, ct[i]])
    th = np.array([ct[i], 0.0, -st[i]])
    ph = np.array([0.0, 1.0, 0.0])
    t1v = -delta * (dsig @ rh)
    t1r[i] = t1v @ rh
    t1t[i] = t1v @ th
    t1p[i] = t1v @ ph
    # pieces at the surface point
    sigA = sigma_sph(A, Sh, i)
    sig_p = sigma_sph(A + eps, Sh, i)
    sig_m = sigma_sph(A - eps, Sh, i)
    drt0 = (sig_p[:, 0] - sig_m[:, 0]) / (2.0 * eps)  # d/dr t0
    hv = delta * ct[i]
    advr[i] = -hv * drt0[0]
    advt[i] = -hv * drt0[1]
    advp[i] = -hv * drt0[2]
    fac = -delta * st[i] / A          # (1/A)(grad1 h)_theta
    tilr[i] = fac * sigA[1, 0]
    tilt_[i] = fac * sigA[1, 1]
    tilp[i] = fac * sigA[1, 2]

R_true, S_true, T_true = proj_rows(t1r, t1t, t1p)
R_adv, S_adv, T_adv = proj_rows(advr, advt, advp)
R_til, S_til, T_til = proj_rows(tilr, tilt_, tilp)

# ---- ENGINE rows (replicated per relief_spectra m=1 path) -------
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
jU, jV, jR, jS, jW, jT = tfe._jump_rows(hLM, A, ang, dc)
# engine piece split
jS_adv = -hLM * (ang.GA_v @ dc["Sp"] + ang.GA_w @ dc["Tp"])
jS_til = (hLM / A) * (ang.Hiso_v @ dc["ciso"]
                      + ang.HV_v @ dc["cV"] + ang.HW_v @ dc["cW"])
jT_adv = -hLM * (ang.GC_v @ dc["Sp"] + ang.GC_w @ dc["Tp"])
jT_til = (hLM / A) * (ang.Hiso_w @ dc["ciso"]
                      + ang.HV_w @ dc["cV"] + ang.HW_w @ dc["cW"])
jR_adv = -hLM * (ang.G0 @ dc["Rp"])

print("m=1 traction-row truth test (free surface, Mrt, homog "
      "ball)")
print("lp   jR/R_true      jS/S_true      jT/T_true")
for lp in range(1, LAUD + 1):
    rR = jR[lp] / R_true[lp]
    rS = jS[lp] / S_true[lp]
    rT = jT[lp] / T_true[lp]
    print("%2d  %7.4f<%5.0f  %7.4f<%5.0f  %7.4f<%5.0f"
          % (lp, abs(rR), np.degrees(np.angle(rR)), abs(rS),
             np.degrees(np.angle(rS)), abs(rT),
             np.degrees(np.angle(rT))))
print("\npieces: S-row  adv(eng/true)   tilt(eng/true)   |  T-row"
      "  adv          tilt")
for lp in range(1, LAUD + 1):
    a1 = jS_adv[lp] / S_adv[lp]
    a2 = jS_til[lp] / S_til[lp]
    a3 = jT_adv[lp] / T_adv[lp]
    a4 = jT_til[lp] / T_til[lp]
    print("%2d   %7.4f<%5.0f  %7.4f<%5.0f  |  %7.4f<%5.0f"
          "  %7.4f<%5.0f"
          % (lp, abs(a1), np.degrees(np.angle(a1)), abs(a2),
             np.degrees(np.angle(a2)), abs(a3),
             np.degrees(np.angle(a3)), abs(a4),
             np.degrees(np.angle(a4))))
