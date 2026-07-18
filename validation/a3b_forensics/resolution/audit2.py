#!/usr/bin/env python3
"""Audit stage 2: WHERE does the exact first-order solution
(w + vsrc) leave the engine's solution space (regular + continuous
homogeneous per-lp)?

 - fit (w+vsrc)_lp(r) above r0 to the 4-col (j,y) basis and below
   r0 to the 2-col regular basis; report fit residuals (ODE test)
   and the implied 4-vector JUMP at r0.
 - if a jump exists: add it as iface_rhs to the engine solve and
   see if that reproduces the exact surface values.
"""
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/y/ytian159/astroseis_qtest")
from pyastroseis import spectral_tfe as tfe
from pyastroseis.spectral import (_entries, _mats_at, make_stack,
                                  spheroidal_unit, _Ysolid)
from pyastroseis.spheroidal_ref import source_jumps, spheroidal_pole
from tests.test_spectral import A, SH, wk

QSH = 50.0
LMAX = 8
LAUD = LMAX - 2
r0 = A - 637.0e3
w = wk(90.0)
lay = [dict(r_top=A, **SH)]
stack = make_stack(lay)
mats = _mats_at(stack, w, QSH, -1.0)
m0 = mats[0]
mu, lam, rho = m0["mu"], m0["lam"], m0["rho"]
delta = 30.0
hLM = delta * np.sqrt(4.0 * np.pi / 3.0)
DY, _DG = spheroidal_pole(LMAX)
CS = {l: 1.0e18 * DY[l] for l in range(1, LMAX + 1)}


def solve_l(l, b):
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
    i = 1 if r >= sol["b"] else 0
    kinds = ("j",) if i == 0 else ("j", "y")
    return _Ysolid(l, w, r, m0, kinds, None) \
        @ sol["x"][sol["ofs"][i]:sol["ofs"][i + 1]]


SOL0 = {l: solve_l(l, r0) for l in range(1, LMAX + 1)}
SOLm = {l: solve_l(l, r0 - delta) for l in range(1, LMAX + 1)}
SOLp = {l: solve_l(l, r0 + delta) for l in range(1, LMAX + 1)}

NT = 96
xg, wg = np.polynomial.legendre.leggauss(NT)
ct, st = xg, np.sqrt(1.0 - xg * xg)
SH_ = tfe._shapes(0, LMAX + 2, ct, st)
Yg, dYg = SH_["Y"], SH_["Bt"]
d2Yg = SH_["eB"][0] / 2.0
w2 = 2.0 * np.pi * wg
llv = np.arange(LMAX + 3)
Lpv = llv * (llv + 1.0)
Lpv[0] = 1.0


def proj(fr, ft):
    U = np.einsum('ln,n->l', np.conj(Yg), fr * w2)
    V = np.einsum('ln,n->l', np.conj(dYg), ft * w2) / Lpv
    return U, V


def u0_line(r, sols=SOL0, hfd=0.5):
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


def exact_u1_lp(r):
    """(U, V) projections of w + vsrc at radius r (r must be at
    least 300 m away from r0)."""
    ur, ut, urp, utp, dtur, dtut = u0_line(r)
    wr = -delta * (ct * urp - (st / r) * (dtur - ut))
    wt = -delta * (ct * utp - (st / r) * (ur + dtut))
    Uw, Vw = proj(wr, wt)
    Uv = np.zeros(LMAX + 3, dtype=complex)
    Vv = np.zeros(LMAX + 3, dtype=complex)
    for l in range(1, LMAX + 1):
        ym = y_of(l, SOLm[l], r) * CS[l]
        yp_ = y_of(l, SOLp[l], r) * CS[l]
        Uv[l] = 0.5 * (ym[0] - yp_[0])
        Vv[l] = 0.5 * (ym[1] - yp_[1])
    return Uw + Uv, Vw + Vv


# sample radii
r_above = np.linspace(r0 + 500.0, A - 500.0, 12)
r_below = np.linspace(0.35 * A, r0 - 500.0, 10)
samp_a = [exact_u1_lp(r) for r in r_above]
samp_b = [exact_u1_lp(r) for r in r_below]

print("== fit (w+vsrc)_lp to homogeneous bases; residuals + jump "
      "at r0 ==")
print("lp  res_above   res_below   |jumpU|/|U(r0)| |jumpV|/|V| "
      " |jumpR|/scale |jumpS|/scale")
jumps = {}
for lp in range(1, LAUD + 1):
    # above: 4 columns (j, y)
    Ca, ba = [], []
    for i, r in enumerate(r_above):
        Y4 = _Ysolid(lp, w, r, m0, ("j", "y"), None)
        Ca.append(Y4[0])
        Ca.append(Y4[1])
        ba.append(samp_a[i][0][lp])
        ba.append(samp_a[i][1][lp])
    Ca = np.array(Ca)
    ba = np.array(ba)
    sca = np.max(np.abs(Ca), axis=0)
    cA, res_a, *_ = np.linalg.lstsq(Ca / sca, ba, rcond=None)
    resid_a = np.linalg.norm(Ca / sca @ cA - ba) / np.linalg.norm(ba)
    cA = cA / sca
    # below: 2 regular columns
    Cb, bb = [], []
    for i, r in enumerate(r_below):
        Y2 = _Ysolid(lp, w, r, m0, ("j",), None)
        Cb.append(Y2[0])
        Cb.append(Y2[1])
        bb.append(samp_b[i][0][lp])
        bb.append(samp_b[i][1][lp])
    Cb = np.array(Cb)
    bb = np.array(bb)
    scb = np.max(np.abs(Cb), axis=0)
    cB, *_ = np.linalg.lstsq(Cb / scb, bb, rcond=None)
    resid_b = np.linalg.norm(Cb / scb @ cB - bb) / np.linalg.norm(bb)
    cB = cB / scb
    yA = _Ysolid(lp, w, r0, m0, ("j", "y"), None) @ cA
    yB = _Ysolid(lp, w, r0, m0, ("j",), None) @ cB
    jmp = yA - yB
    jumps[lp] = jmp
    # scales
    sU = max(abs(yA[0]), abs(yB[0]))
    sV = max(abs(yA[1]), abs(yB[1]))
    sR = max(abs(yA[2]), abs(yB[2]))
    sS = max(abs(yA[3]), abs(yB[3]))
    print("%2d  %.3e  %.3e   %.3e      %.3e     %.3e     %.3e"
          % (lp, resid_a, resid_b, abs(jmp[0]) / sU,
             abs(jmp[1]) / sV, abs(jmp[2]) / sR, abs(jmp[3]) / sS))

# ---- engine bc solve + measured jump as iface_rhs ---------------
# engine dc/jump rows (same as audit1)
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

print("\n== engine solve WITH measured interface jump at r0 ==")
print("lp  ratio-U(before)  ratio-U(with jump)  ratio-V(before)"
      "  ratio-V(with jump)")
UA, VA = exact_u1_lp(A)          # exact surface values
for lp in range(1, LAUD + 1):
    entsp = _entries(mats, 0, r0, 0)
    U1, V1 = spheroidal_unit(lp, w, entsp,
                             np.zeros(4, dtype=complex),
                             bc_rhs=(jR[lp], jS[lp]))
    U2, V2 = spheroidal_unit(lp, w, entsp,
                             np.zeros(4, dtype=complex),
                             bc_rhs=(jR[lp], jS[lp]),
                             iface_rhs={0: jumps[lp]})
    print("%2d  %6.3f<%5.0f     %6.3f<%5.0f      %6.3f<%5.0f"
          "     %6.3f<%5.0f"
          % (lp, abs(U1 / UA[lp]),
             np.degrees(np.angle(U1 / UA[lp])),
             abs(U2 / UA[lp]),
             np.degrees(np.angle(U2 / UA[lp])),
             abs(V1 / VA[lp]),
             np.degrees(np.angle(V1 / VA[lp])),
             abs(V2 / VA[lp]),
             np.degrees(np.angle(V2 / VA[lp]))))
