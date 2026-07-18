#!/usr/bin/env python3
"""L=1 translation paradox hand-audit, minimal config:
homog ball (SH mat), mrr on-axis, m=0, Y10 relief on outer surface,
lmax=8, one frequency. Tests each link of the uniqueness chain:

 C0  reproduce the failure with the engine relief_spectra
 C1  du_fd == w + vsrc + adv       (exact decomposition, per lp)
     w    = -delta (zhat.grad) u0  (semi-analytic, at r=A)
     vsrc = -delta d(u0)/d(r0)     (per-lp coefficient FD)
     adv  = h dr(u0)               (receiver advection)
 C2  engine advection projections == exact adv projections
 C3  traction of exact u1 = w+vsrc at A  ==  engine rows (jR, jS)
     (t[vsrc](A) = 0 exactly by the legs' BCs)
 C6  engine BVP solve: U1bc vs (Uw+Uvsrc); also re-solve with the
     MEASURED exact traction as rhs.
"""
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis import spectral as spsp
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  spheroidal_unit, _Ysolid)
from pyastroseis.spheroidal_ref import (source_jumps,
                                        spheroidal_pole,
                                        spheroidal_reconstruct)
from tests.test_spectral import A, SH, wk

QSH = 50.0
LMAX = 8
LAUD = LMAX - 2                    # audited lp range 1..LAUD
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
stack = make_stack(lay)
mats = _mats_at(stack, w, QSH, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
delta = 30.0
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)   # cos(th) = hLM*Ybar10
DY, _DG = spheroidal_pole(LMAX)
CS = {l: 1.0e18 * DY[l] for l in range(1, LMAX + 1)}
Mrr = np.diag([0.0, 0.0, 1.0e18])
src = np.array([0.0, 0.0, r0])
zhat = np.array([0.0, 0.0, 1.0])


# ---------- compact homog-ball solver keeping ALL coefficients ----
def solve_l(l, b):
    """u0-type solve with source jump at radius b; returns dict with
    coefficient vector x, offsets, entries."""
    ents = _entries(mats, 0, b, 0)
    F0y = np.array([0, 0, 1.0 / b ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / b ** 3, 0], dtype=complex)
    J = source_jumps(l, w, b, rho, lam, mu, F0y, F1y)
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
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        Am[row:row + 4, chi] = Yb[i + 1]
        Am[row:row + 4, clo] = -Yt[i]
        if ents[i]["src_top"]:
            bv[row:row + 4] = J
        row += 4
    ctop = slice(ofs[-2], ofs[-1])
    Am[row, ctop] = Yt[-1][2]
    Am[row + 1, ctop] = Yt[-1][3]
    sc = np.max(np.abs(Am), axis=0)
    sc[sc == 0] = 1.0
    x = np.linalg.solve(Am / sc, bv) / sc
    return dict(x=x, ofs=ofs, b=b)


def y_of(l, sol, r):
    """(U,V,R,S) of the solve at radius r (entry picked by r)."""
    i = 1 if r >= sol["b"] else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return _Ysolid(l, w, r, m0, kinds, None) \
        @ sol["x"][sol["ofs"][i]:sol["ofs"][i + 1]]


SOL0 = {l: solve_l(l, r0) for l in range(1, LMAX + 1)}
SOLm = {l: solve_l(l, r0 - delta) for l in range(1, LMAX + 1)}
SOLp = {l: solve_l(l, r0 + delta) for l in range(1, LMAX + 1)}

# ---------- projection grid --------------------------------------
NT = 96
xg, wg = np.polynomial.legendre.leggauss(NT)
tg = np.arccos(xg)
ct, st = xg, np.sqrt(1.0 - xg * xg)
SH_ = tfe._shapes(0, LMAX + 2, ct, st)
Yg, dYg = SH_["Y"], SH_["Bt"]
d2Yg = SH_["eB"][0] / 2.0
w2 = 2.0 * np.pi * wg


def proj(fr, ft):
    """(U_lp, V_lp) coefficient projections of a (r,theta) field."""
    U = np.einsum('ln,n->l', np.conj(Yg), fr * w2)
    V = np.einsum('ln,n->l', np.conj(dYg), ft * w2)
    Lp = np.arange(LMAX + 3) * (np.arange(LMAX + 3) + 1.0)
    Lp[0] = 1.0
    return U, V / Lp


# ---------- fields of u0 on a constant-r theta line ---------------
def u0_line(r, sols=SOL0, hfd=0.5):
    """ur, ut, and radial derivatives urp, utp; plus analytic theta
    derivatives d_th(ur), d_th(ut) at radius r (all CS-weighted)."""
    ur = np.zeros(NT, dtype=complex)
    ut = np.zeros(NT, dtype=complex)
    urp = np.zeros(NT, dtype=complex)
    utp = np.zeros(NT, dtype=complex)
    dtur = np.zeros(NT, dtype=complex)
    dtut = np.zeros(NT, dtype=complex)
    for l in range(1, LMAX + 1):
        yv = y_of(l, sols[l], r) * CS[l]
        yp = (y_of(l, sols[l], r + hfd)
              - y_of(l, sols[l], r - hfd)) / (2.0 * hfd) * CS[l]
        ur += yv[0] * Yg[l]
        ut += yv[1] * dYg[l]
        urp += yp[0] * Yg[l]
        utp += yp[1] * dYg[l]
        dtur += yv[0] * dYg[l]
        dtut += yv[1] * d2Yg[l]
    return ur, ut, urp, utp, dtur, dtut


def w_field(r):
    """w = -delta (zhat.grad) u0 components on the theta line at
    radius r: w_r, w_t."""
    ur, ut, urp, utp, dtur, dtut = u0_line(r)
    wr = -delta * (ct * urp - (st / r) * (dtur - ut))
    wt = -delta * (ct * utp - (st / r) * (ur + dtut))
    return wr, wt


# ================= C0: engine repro on this config ================
dirs = np.stack([st, 0.0 * st, ct], axis=1)
kw = dict(Q=QSH, q_sign=-1.0, lmax=LMAX)
du_eng_full = tfe.relief_spectra(lay, src, [Mrr], np.array([w]),
                                 dirs, relief=[("top", 1, 0, hLM)],
                                 **kw)
du_eng_grid = (du_eng_full["psv"] + du_eng_full["sh"])[0, :, :, 0]

that = np.stack([ct, 0.0 * ct, -st], axis=1)
du_fd_grid = 0.0
for s in (+1.0, -1.0):
    dp = dirs + s * (delta / A) * st[:, None] * that
    dp /= np.linalg.norm(dp, axis=1)[:, None]
    us = spsp.spectral_spectra(lay, src - s * delta * zhat, [Mrr],
                               np.array([w]), dp, l0=False, **kw)
    du_fd_grid = du_fd_grid + 0.5 * s * (us["psv"]
                                         + us["sh"])[0, :, :, 0]

dur_e = np.einsum('nc,nc->n', dirs, du_eng_grid)
dut_e = np.einsum('nc,nc->n', that, du_eng_grid)
dur_f = np.einsum('nc,nc->n', dirs, du_fd_grid)
dut_f = np.einsum('nc,nc->n', that, du_fd_grid)
Ue, Ve = proj(dur_e, dut_e)
Uf, Vf = proj(dur_f, dut_f)
print("== C0 engine vs translation-FD (minimal config, lmax=%d) =="
      % LMAX)
print("lp  |dU_eng|      |dU_fd|      eng/fd(U)      eng/fd(V)")
for lp in range(1, LAUD + 1):
    rU = Ue[lp] / Uf[lp]
    rV = Ve[lp] / Vf[lp]
    print("%2d  %.4e  %.4e  %6.3f<%5.0f  %6.3f<%5.0f"
          % (lp, abs(Ue[lp]), abs(Uf[lp]), abs(rU),
             np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV))))

# ================= engine internals (replicated) ==================
ang = tfe.ReliefCouplings(1, 0, 0, LMAX)
keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp", "S", "T", "V", "W",
        "ciso", "cV", "cW")
dc = {k: np.zeros(LMAX + 1, dtype=complex) for k in keys}
for l in range(1, LMAX + 1):
    ents = _entries(mats, 0, r0, 0)
    F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
    F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
    J = source_jumps(l, w, r0, rho, lam, mu, F0y, F1y)
    Ua, Va, info = spheroidal_unit(l, w, ents, J, full=True)
    sc_ = tfe._side_coeffs(l, w, A, m0, info["surface"] * CS[l],
                           None, yp4=info["surface_p"] * CS[l])
    for k in keys:
        dc[k][l] = sc_[k]
jU, jV, jR, jS, jW, jT = tfe._jump_rows(hLM, A, ang, dc)

advU = hLM * (ang.G0 @ dc["Up"])
advV = hLM * (ang.GA_v @ dc["Vp"])

U1bc = np.zeros(LMAX + 1, dtype=complex)
V1bc = np.zeros(LMAX + 1, dtype=complex)
for lp in range(1, LMAX + 1):
    entsp = _entries(mats, 0, r0, 0)
    U1, V1 = spheroidal_unit(lp, w, entsp,
                             np.zeros(4, dtype=complex),
                             bc_rhs=(jR[lp], jS[lp]))
    U1bc[lp] = U1
    V1bc[lp] = V1

# replication sanity: engine coefficients == my U1bc + advU
rep = max(abs(Ue[1:LAUD + 1] - (U1bc + advU)[1:LAUD + 1]).max()
          / abs(Ue[1:LAUD + 1]).max(),
          abs(Ve[1:LAUD + 1] - (V1bc + advV)[1:LAUD + 1]).max()
          / abs(Ve[1:LAUD + 1]).max())
print("\nreplication sanity (engine full route vs replicated "
      "U1bc+adv): rel %.2e" % rep)

# ================= exact-side decomposition =======================
# w projections at A
wr, wt = w_field(A)
Uw, Vw = proj(wr, wt)
# vsrc per-lp (coefficient FD in source radius)
Uvs = np.zeros(LMAX + 1, dtype=complex)
Vvs = np.zeros(LMAX + 1, dtype=complex)
for l in range(1, LMAX + 1):
    ym = y_of(l, SOLm[l], A) * CS[l]
    yp_ = y_of(l, SOLp[l], A) * CS[l]
    Uvs[l] = 0.5 * (ym[0] - yp_[0])
    Vvs[l] = 0.5 * (ym[1] - yp_[1])
    # legs' surface traction (must be 0 exactly):
    if max(abs(ym[2]), abs(ym[3]), abs(yp_[2]), abs(yp_[3])) > \
       1e-8 * max(abs(ym[2]), 1.0):
        print("WARN: leg surface traction nonzero at l=%d" % l)
# adv exact (pointwise projection of h dr u0)
ur, ut, urp, utp, _, _ = u0_line(A)
Uad, Vad = proj(delta * ct * urp, delta * ct * utp)

print("\n== C1 decomposition: du_fd vs (w + vsrc + adv) ==")
print("lp  |dU_fd|     rel-err(U)   rel-err(V)")
for lp in range(1, LAUD + 1):
    eU = abs(Uf[lp] - (Uw[lp] + Uvs[lp] + Uad[lp])) / abs(Uf[lp])
    eV = abs(Vf[lp] - (Vw[lp] + Vvs[lp] + Vad[lp])) / abs(Vf[lp])
    print("%2d  %.4e  %.3e  %.3e" % (lp, abs(Uf[lp]), eU, eV))

print("\n== C2 advection: engine proj vs exact proj ==")
for lp in range(1, LAUD + 1):
    eU = abs(advU[lp] - Uad[lp]) / max(abs(Uad[lp]), 1e-300)
    eV = abs(advV[lp] - Vad[lp]) / max(abs(Vad[lp]), 1e-300)
    print("%2d  rel-err U %.3e  V %.3e" % (lp, eU, eV))

# ================= C3 traction of exact u1 at A ===================
h1 = 0.5
UwA = {}
for rr_ in (A - h1, A, A + h1):
    wrr, wtt = w_field(rr_)
    UwA[rr_] = proj(wrr, wtt)
Uw0, Vw0 = UwA[A]
Uwp = (UwA[A + h1][0] - UwA[A - h1][0]) / (2.0 * h1)
Vwp = (UwA[A + h1][1] - UwA[A - h1][1]) / (2.0 * h1)
ll = np.arange(LMAX + 3)
Lp = ll * (ll + 1.0)
div_w = Uwp + 2.0 * Uw0 / A - Lp * Vw0 / A
R_w = lam * div_w + 2.0 * mu * Uwp
S_w = mu * (Vwp - Vw0 / A + Uw0 / A)

print("\n== C3 traction rows: engine (jR,jS) vs exact t[w+vsrc] ==")
print("lp   jR              R_w             ratio    |   jS"
      "              S_w             ratio")
for lp in range(1, LAUD + 1):
    rR = jR[lp] / R_w[lp] if R_w[lp] != 0 else np.nan
    rS = jS[lp] / S_w[lp] if S_w[lp] != 0 else np.nan
    print("%2d  %11.3e%+9.1ej %11.3e%+9.1ej %6.3f<%4.0f | "
          "%11.3e%+9.1ej %11.3e%+9.1ej %6.3f<%4.0f"
          % (lp, jR[lp].real, jR[lp].imag, R_w[lp].real,
             R_w[lp].imag, abs(rR), np.degrees(np.angle(rR)),
             jS[lp].real, jS[lp].imag, S_w[lp].real, S_w[lp].imag,
             abs(rS), np.degrees(np.angle(rS))))

# ================= C6 solve identity ==============================
print("\n== C6 engine BVP solve vs exact interior solution ==")
print("lp  U1bc          Uw+Uvs        ratio     |  V1bc"
      "          Vw+Vvs        ratio")
for lp in range(1, LAUD + 1):
    Ux = Uw[lp] + Uvs[lp]
    Vx = Vw[lp] + Vvs[lp]
    rU = U1bc[lp] / Ux
    rV = V1bc[lp] / Vx
    print("%2d %12.4e %12.4e %6.3f<%5.0f | %12.4e %12.4e "
          "%6.3f<%5.0f"
          % (lp, abs(U1bc[lp]), abs(Ux), abs(rU),
             np.degrees(np.angle(rU)), abs(V1bc[lp]), abs(Vx),
             abs(rV), np.degrees(np.angle(rV))))

# re-solve with MEASURED exact traction as rhs
print("\nre-solve with rhs = (R_w, S_w): vs exact (Uw+Uvs)")
for lp in range(1, LAUD + 1):
    entsp = _entries(mats, 0, r0, 0)
    U1x, V1x = spheroidal_unit(lp, w, entsp,
                               np.zeros(4, dtype=complex),
                               bc_rhs=(R_w[lp], S_w[lp]))
    Ux = Uw[lp] + Uvs[lp]
    Vx = Vw[lp] + Vvs[lp]
    rU = U1x / Ux
    rV = V1x / Vx
    print("%2d  ratio U %6.3f<%5.0f   V %6.3f<%5.0f"
          % (lp, abs(rU), np.degrees(np.angle(rU)), abs(rV),
             np.degrees(np.angle(rV))))
