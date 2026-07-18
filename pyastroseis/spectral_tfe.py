"""Rung A3/A3b: first-order boundary relief (TFE class) for the
spectral sweep (docs/rung_a3_tfe.md).

A3 MVP (validated 2026-07-17): toroidal [SH->SH] relief on SH-free
boundaries. A3b (this stage): full welded-interface and free-surface
relief for BOTH parities including the parity-CONVERSION blocks —
du^PSV = [PSV->PSV] + [SH->PSV], du^SH = [SH->SH] + [PSV->SH]
(the transferred data are built from the full unperturbed field, and
their basis projections mix parities for L >= 1 relief).

Transfer conditions (docs section 4): at a welded interface r = d
with relief h = h_LM Y_LM,
    [u1] = -h [d/dr u0]
    [t1] = -h [d/dr t0] + (1/d) [sigma0 . grad_1 h]
(the radial part of the tilt vanishes on welds and free surfaces:
(sigma.v)_r involves only the CONTINUOUS shear tractions S, T);
at the free surface the single-sided versions apply, plus the
receiver-advection term du_obs += h d/dr u0(a).

All angular couplings are evaluated numerically on a Gauss-Legendre
theta grid in the exact code conventions (fully-normalized Ybar,
V/S on the UNNORMALIZED gradient, W/T on the ORTHONORMAL C =
rhat x grad1 Y / sqrt(L)), with the exact |l'-l| <= L triangle mask;
they are k-independent and cached. Cross-validation: closed forms at
L=0, selection rules, and sympy exact surface integrals
(validation/tfe_sympy_check.py — dual-derivation house pattern).
"""

import numpy as np

from .spectral import (TOL_PRUNE, _entries, _icut, _mats_at,
                       _tor_entries, make_stack, spheroidal_unit,
                       toroidal_unit)
from .spheroidal_ref import (L_SERIES, source_jumps, spheroidal_pole,
                             spheroidal_reconstruct, system_matrix)
from .toroidal_modes import _pole_coupling
from .toroidal_ref import source_frame, toroidal_reconstruct


# ----------------------------------------------------------------
# normalized associated Legendre grids (reconstruct conventions)
# ----------------------------------------------------------------
def _plm_grid(m, lmax, ct, st):
    """P̄_l^|m|, dP̄/dtheta, d2P̄/dtheta2 arrays (lmax+1, n) at
    ct = cos(theta) (same recurrences as toroidal_reconstruct; the
    second derivative from the Legendre ODE)."""
    am = abs(m)
    n = len(ct)
    P = np.zeros((lmax + 1, n))
    dP = np.zeros_like(P)
    p = np.full(n, 1.0 / np.sqrt(4.0 * np.pi))
    for mm in range(1, am + 1):
        p = -np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * st * p
    pm1 = np.zeros(n)
    pcur = None
    for l in range(am, lmax + 1):
        if l == am:
            pl = p
        elif l == am + 1:
            pl = np.sqrt(2.0 * am + 3.0) * ct * p
        else:
            aa = np.sqrt((4.0 * l * l - 1.0) / (l * l - am * am))
            bb = np.sqrt(((2.0 * l + 1.0) * ((l - 1.0) ** 2
                                             - am * am))
                         / ((2.0 * l - 3.0) * (l * l - am * am)))
            pl = aa * ct * pcur - bb * pm1
        if l > am:
            pm1 = pcur
        pcur = pl
        P[l] = pl
        num = l * ct * pl
        if l > am:
            e = np.sqrt((l * l - am * am) * (2.0 * l + 1.0)
                        / (2.0 * l - 1.0))
            num = num - e * pm1
        dP[l] = num / st
    ll = np.arange(lmax + 1)[:, None]
    d2P = -ct / st * dP + (m * m / (st * st) - ll * (ll + 1.0)) * P
    return P, dP, d2P


def _shapes(m, lmax, ct, st):
    """Angular shape functions per l at the grid, order m: scalar Y;
    unnormalized-gradient B = grad1 Y; orthonormal toroidal C; and
    the unit-sphere symmetric tangential strains 2e(B), 2e(C)
    (components tt, pp, tp). All theta-parts (phi factors handled
    analytically by m-selection)."""
    P, dP, d2P = _plm_grid(m, lmax, ct, st)
    sgn = 1.0 if m >= 0 else (-1.0) ** abs(m)
    Y = sgn * P
    dY = sgn * dP
    d2Y = sgn * d2P
    ll = np.arange(lmax + 1)
    f = np.zeros(lmax + 1)
    f[1:] = 1.0 / np.sqrt(ll[1:] * (ll[1:] + 1.0))
    Bt, Bp = dY, 1j * m * Y / st
    Ct = -1j * m * f[:, None] * Y / st
    Cp = f[:, None] * dY
    cot = ct / st

    def strain2(vt, vp, dvt, dvp):
        e_tt = 2.0 * dvt
        e_pp = 2.0 * (1j * m / st * vp + cot * vt)
        e_tp = dvp - cot * vp + 1j * m / st * vt
        return e_tt, e_pp, e_tp

    dBt = d2Y
    dBp = 1j * m * (dY / st - Y * cot / st)
    dCt = -1j * m * f[:, None] * (dY / st - Y * cot / st)
    dCp = f[:, None] * d2Y
    eB = strain2(Bt, Bp, dBt, dBp)
    eC = strain2(Ct, Cp, dCt, dCp)
    return dict(Y=Y, Bt=Bt, Bp=Bp, Ct=Ct, Cp=Cp, eB=eB, eC=eC,
                L=ll * (ll + 1.0))


class ReliefCouplings:
    """All k-independent angular coupling matrices for one relief
    harmonic (L, M) and one source order m (target m' = m + M),
    (lmax+1, lmax+1), exact triangle mask. Naming: G* = value
    transfer, H* = tilt; suffix _u/_v/_w = target row family
    (Y' rhat / grad1 Y'/L' / C')."""

    def __init__(self, L, M, m, lmax, ngl=None):
        self.L, self.M, self.m, self.lmax = L, M, m, lmax
        mp = m + M
        if ngl is None:
            ngl = 2 * lmax + 2 * L + 60
        x, wgt = np.polynomial.legendre.leggauss(ngl)
        ct, st = x, np.sqrt(1.0 - x * x)
        S = _shapes(m, lmax, ct, st)
        Tg = _shapes(mp, lmax, ct, st)
        R = _shapes(M, L, ct, st)
        YL, GtL, GpL = R["Y"][L], R["Bt"][L], R["Bp"][L]
        Lp = Tg["L"].copy()
        Lp[0] = 1.0
        w2 = 2.0 * np.pi * wgt

        def dotV(at, ap, bt, bp, wf):
            return (np.einsum('an,bn,n->ab', np.conj(at), bt, wf)
                    + np.einsum('an,bn,n->ab', np.conj(ap), bp, wf))

        def dotS(a, b, wf):
            return np.einsum('an,bn,n->ab', np.conj(a), b, wf)

        # value-transfer couplings
        self.G0 = dotS(Tg["Y"], S["Y"], YL * w2)
        self.GA_v = dotV(Tg["Bt"], Tg["Bp"], S["Bt"], S["Bp"],
                         YL * w2) / Lp[:, None]
        self.GA_w = dotV(Tg["Bt"], Tg["Bp"], S["Ct"], S["Cp"],
                         YL * w2) / Lp[:, None]
        self.GC_v = dotV(Tg["Ct"], Tg["Cp"], S["Bt"], S["Bp"],
                         YL * w2)
        self.GC_w = dotV(Tg["Ct"], Tg["Cp"], S["Ct"], S["Cp"],
                         YL * w2)
        # tilt couplings: v = grad1 Y_L = (GtL, GpL)
        BdotG = S["Bt"] * GtL + S["Bp"] * GpL     # (l, n)
        CdotG = S["Ct"] * GtL + S["Cp"] * GpL
        self.H0_s = dotS(Tg["Y"], BdotG, w2)
        self.H0_t = dotS(Tg["Y"], CdotG, w2)
        self.Hiso_v = dotV(Tg["Bt"], Tg["Bp"],
                           S["Y"] * GtL, S["Y"] * GpL,
                           w2) / Lp[:, None]
        self.Hiso_w = dotV(Tg["Ct"], Tg["Cp"],
                           S["Y"] * GtL, S["Y"] * GpL, w2)

        def tiltvec(e):
            e_tt, e_pp, e_tp = e
            return (e_tt * GtL + e_tp * GpL,
                    e_tp * GtL + e_pp * GpL)

        vBt, vBp = tiltvec(S["eB"])
        vCt, vCp = tiltvec(S["eC"])
        self.HV_v = dotV(Tg["Bt"], Tg["Bp"], vBt, vBp,
                         w2) / Lp[:, None]
        self.HV_w = dotV(Tg["Ct"], Tg["Cp"], vBt, vBp, w2)
        self.HW_v = dotV(Tg["Bt"], Tg["Bp"], vCt, vCp,
                         w2) / Lp[:, None]
        self.HW_w = dotV(Tg["Ct"], Tg["Cp"], vCt, vCp, w2)
        lp_g, l_g = np.meshgrid(np.arange(lmax + 1),
                                np.arange(lmax + 1), indexing='ij')
        tri = np.abs(lp_g - l_g) <= L
        for name in ("G0", "GA_v", "GA_w", "GC_v", "GC_w", "H0_s",
                     "H0_t", "Hiso_v", "Hiso_w", "HV_v", "HV_w",
                     "HW_v", "HW_w"):
            getattr(self, name)[~tri] = 0.0
        # MVP aliases (toroidal_relief_spectra)
        self.I1 = self.GC_w
        self.I2 = 0.5 * self.HW_w


_AngCache = ReliefCouplings          # backward-compat alias


def _ang_cache(store, L, M, m, lmax):
    key = (L, M, m, lmax)
    if key not in store:
        store[key] = ReliefCouplings(L, M, m, lmax)
    return store[key]


# ----------------------------------------------------------------
# per-side radial coefficient bundles at the relieved radius
# ----------------------------------------------------------------
def _side_coeffs(l, w, d, mat, y4, wt2, yp4=None):
    """Radial coefficient bundle for one SIDE at radius d:
    y4 = (U, V, R, S) or None; wt2 = (W, T) or None. yp4 = the
    radial-derivative 4-vector from the solver's basis-FD (preferred
    — the Y'Y^-1 fallback loses ~1e-3 relative at l >= L_SERIES).
    Returns dict with U', V', R', S', W', T', ciso, cV, cW (zeros
    where the parity is absent).

    FLUID side (mat["solid"] False, A3b stage 3): y4/yp4 are the
    2-vectors (u_r, s_rr) and their basis-FD radial derivative.
    The bundle carries the potential-slaved tangential displacement
    V = phi/d with phi = -s_rr/(rho w^2) (u = grad phi — the SLIP
    against the solid side enters the tilted u.n condition), the
    isotropic tangential stress ciso = s_rr (sigma_f = s_rr I), and
    a purely radial traction derivative (Sp = Tp = 0, Rp = ds_rr/dr
    = -rho w^2 u_r via the basis-FD)."""
    out = dict(U=0j, V=0j, R=0j, S=0j, W=0j, T=0j, Up=0j, Vp=0j,
               Rp=0j, Sp=0j, Wp=0j, Tp=0j, ciso=0j, cV=0j, cW=0j)
    L = l * (l + 1.0)
    if not mat["solid"]:
        assert wt2 is None, "no toroidal field in a fluid"
        if y4 is not None:
            ur, srr = y4
            phi = -srr / (mat["rho"] * w * w)
            out.update(U=ur, R=srr, V=phi / d, Up=yp4[0], Rp=yp4[1],
                       Vp=ur / d - phi / d ** 2, ciso=srr)
        return out
    mu, lam = mat["mu"], mat["lam"]
    if y4 is not None:
        if yp4 is not None:
            yp = np.asarray(yp4, dtype=complex)
        else:
            A = system_matrix(l, w, d, mat["rho"], lam, mu,
                              zref_a=(mat["r_top"]
                                      if l >= L_SERIES else None))
            yp = A @ np.asarray(y4, dtype=complex)
        U, V, R, S = y4
        out.update(U=U, V=V, R=R, S=S, Up=yp[0], Vp=yp[1],
                   Rp=yp[2], Sp=yp[3])
        div = yp[0] + 2.0 * U / d - L * V / d
        out["ciso"] = lam * div + 2.0 * mu * U / d
        out["cV"] = mu * V / d
    if wt2 is not None:
        W, T = wt2
        out.update(W=W, T=T)
        out["Wp"] = T / mu + W / d
        out["Tp"] = (-3.0 * T / d
                     + (mu * (L - 2.0) / d ** 2
                        - mat["rho"] * w * w) * W)
        out["cW"] = mu * W / d
    return out


def _jump_rows(h, d, ang, dc):
    """First-order transferred data for one relieved location.
    dc = dict of DELTA coefficients (above - below; single-sided:
    the side itself), each an (lmax+1,) array over source l.
    Returns per-target-l' arrays (jU, jV, jR, jS, jW, jT).

    jU is the u.n row: at fluid-adjacent interfaces the tangential
    SLIP [u_t] enters through the tilted normal, (1/d)[u_t.grad1 h]
    (H0 couplings on dc["V"], dc["W"]); at welded interfaces those
    deltas cancel to solver precision and jU reduces to the vector
    displacement transfer."""
    jU = -h * (ang.G0 @ dc["Up"]) + (h / d) * (
        ang.H0_s @ dc["V"] + ang.H0_t @ dc["W"])
    jV = -h * (ang.GA_v @ dc["Vp"] + ang.GA_w @ dc["Wp"])
    jW = -h * (ang.GC_v @ dc["Vp"] + ang.GC_w @ dc["Wp"])
    jR = -h * (ang.G0 @ dc["Rp"]) + (h / d) * (
        ang.H0_s @ dc["S"] + ang.H0_t @ dc["T"])
    jS = -h * (ang.GA_v @ dc["Sp"] + ang.GA_w @ dc["Tp"]) \
        + (h / d) * (ang.Hiso_v @ dc["ciso"]
                     + ang.HV_v @ dc["cV"] + ang.HW_v @ dc["cW"])
    jT = -h * (ang.GC_v @ dc["Sp"] + ang.GC_w @ dc["Tp"]) \
        + (h / d) * (ang.Hiso_w @ dc["ciso"]
                     + ang.HV_w @ dc["cV"] + ang.HW_w @ dc["cW"])
    return jU, jV, jR, jS, jW, jT


# ----------------------------------------------------------------
# general first-order relief response (both parities + conversion)
# ----------------------------------------------------------------
def relief_spectra(layers, src_xyz, M_list, w_arr, station_dirs,
                   relief, Q=None, q_sign=-1.0, lmax=250,
                   tol_prune=TOL_PRUNE):
    """FULL first-order du spectra dict(psv=..., sh=...) each
    (nsrc, nst, 3, nw), including parity conversion.

    relief: list of (where, L, M, h_LM); where = 'top' (free
    surface) or the interface RADIUS in metres — welded solid-solid,
    fluid-solid (either orientation: the u.n condition carries the
    tangential slip in the tilt, the traction rows the fluid
    pressure ciso = s_rr), or fluid-fluid. Mr*-type sources
    (|m| <= 1)."""
    stack = make_stack(layers)
    r0 = float(np.linalg.norm(src_xyz))
    isrc = [i for i, e in enumerate(stack)
            if e["r_bot"] < r0 < e["r_top"]][0]
    msrc0 = stack[isrc]
    Qrot = source_frame(src_xyz)
    DY, DG = spheroidal_pole(lmax)
    poleT = _pole_coupling(lmax)
    srcs, hvecs = [], []
    for M in M_list:
        M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
        srcs.append((M_sf[2, 2], M_sf[2, 0] + 0j, M_sf[2, 1] + 0j))
        hvecs.append(M_sf @ np.array([0.0, 0.0, 1.0]))
    w_arr = np.asarray(w_arr, dtype=complex)
    nsrc, nst, nw = len(M_list), len(station_dirs), len(w_arr)
    dirs_sf = station_dirs @ Qrot.T
    store = {}
    a_top = stack[-1]["r_top"]
    Wu1 = np.zeros((nsrc, nw, lmax + 1, 9), dtype=complex)
    Wv1 = np.zeros_like(Wu1)
    Wt1 = np.zeros_like(Wu1)
    for iw, w in enumerate(w_arr):
        mats = _mats_at(stack, w, Q, q_sign)
        msrc = mats[isrc]
        # ---- unperturbed solves per l, both parities, with side
        # values at every welded interface and the surface
        sph, tor = {}, {}
        for l in range(1, lmax + 1):
            zr = msrc["r_top"] if l >= L_SERIES else None
            Ll = l * (l + 1.0)
            ents = _entries(mats, _icut(stack, l, r0, tol_prune),
                            r0, isrc)
            F0v = np.array([0, 0, 0, 1.0 / (Ll * r0 ** 2)],
                           dtype=complex)
            F1v = np.array([0, 0, -1.0 / r0 ** 3,
                            3.0 / (Ll * r0 ** 3)], dtype=complex)
            J_rh = source_jumps(l, w, r0, msrc["rho"], msrc["lam"],
                                msrc["mu"], F0v, F1v, zref_a=zr)
            F0y = np.array([0, 0, 1.0 / r0 ** 2, 0], dtype=complex)
            F1y = np.array([0, 0, 2.0 / r0 ** 3, 0], dtype=complex)
            J_y = source_jumps(l, w, r0, msrc["rho"], msrc["lam"],
                               msrc["mu"], F0y, F1y, zref_a=zr)
            sol = {}
            for tag, J in (("rh", J_rh), ("y", J_y)):
                Ua, Va, info = spheroidal_unit(l, w, ents, J,
                                               full=True)
                sol[tag] = (Ua, Va, info)
            ents_t, bot = _tor_entries(mats, l, r0, isrc, tol_prune)
            wt = None
            if ents_t is not None:
                wa, wb, tinfo = toroidal_unit(
                    l, w, ents_t, bot, 1.0 / (msrc["mu"] * r0 ** 2),
                    full=True)
                wt = (wa, wb, tinfo)
            sph[l] = (ents, sol)
            tor[l] = (ents_t, bot, wt)
        # ---- per relief location build coupled first-order solves
        for where, L, M, hLM in relief:
            for m_src in (-1, 0, 1):
                mp = m_src + M
                if abs(mp) > 4:
                    continue
                ang = _ang_cache(store, L, M, m_src, lmax)
                for js in range(nsrc):
                    mzz, mzx, mzy = srcs[js]
                    hv = hvecs[js]
                    if m_src == 0:
                        if mzz == 0.0:
                            continue
                    # source couplings per l for this (m, source)
                    cs = np.zeros(lmax + 1, dtype=complex)
                    ct_ = np.zeros(lmax + 1, dtype=complex)
                    for l in range(1, lmax + 1):
                        if m_src == 0:
                            cs[l] = mzz * DY[l]
                        else:
                            cs[l] = (DG[m_src][l][0] * mzx
                                     + DG[m_src][l][1] * mzy)
                            ct_[l] = (poleT[m_src][l][0] * hv[0]
                                      + poleT[m_src][l][1] * hv[1])
                    tag = "y" if m_src == 0 else "rh"
                    if not (np.any(cs) or np.any(ct_)):
                        continue        # source order not excited
                    # DELTA (or single-side) coefficient arrays
                    keys = ("Up", "Vp", "Rp", "Sp", "Wp", "Tp",
                            "S", "T", "V", "W", "ciso", "cV", "cW")
                    dc = {k: np.zeros(lmax + 1, dtype=complex)
                          for k in keys}
                    if where == "top":
                        d_rel = a_top
                        adv = {k: np.zeros(lmax + 1, dtype=complex)
                               for k in ("Up", "Vp", "Wp")}
                        for l in range(1, lmax + 1):
                            ents, sol = sph[l]
                            Ua, Va, info = sol[tag]
                            y4 = info["surface"] * cs[l]
                            yp4 = info["surface_p"] * cs[l]
                            wt2 = None
                            if tor[l][2] is not None and m_src != 0:
                                wt2 = (tor[l][2][2]["surface"]
                                       * ct_[l])
                            sc_ = _side_coeffs(
                                l, w, a_top, mats[-1], y4, wt2,
                                yp4=yp4)
                            for k in keys:
                                dc[k][l] = sc_[k]
                            for k in ("Up", "Vp", "Wp"):
                                adv[k][l] = sc_[k]
                    else:
                        d_rel = float(where)
                        jstack = [j for j in range(len(stack) - 1)
                                  if abs(stack[j]["r_top"] - d_rel)
                                  < 1.0][0]
                        sol_lo = stack[jstack]["solid"]
                        sol_hi = stack[jstack + 1]["solid"]
                        for l in range(1, lmax + 1):
                            ents, sol = sph[l]
                            ii = [i for i in range(len(ents) - 1)
                                  if abs(ents[i]["r_top"] - d_rel)
                                  < 1.0]
                            if not ii:
                                continue          # pruned away
                            i_rel = ii[0]
                            Ua, Va, info = sol[tag]
                            ylo, yhi, yplo, yphi = info[i_rel]
                            wtlo = wthi = None
                            ents_t, bot, wt = tor[l]
                            if wt is not None and m_src != 0:
                                if sol_lo and sol_hi:
                                    jj = [i for i in
                                          range(len(ents_t) - 1)
                                          if abs(ents_t[i]["r_top"]
                                                 - d_rel) < 1.0]
                                    if jj:
                                        wtlo, wthi = wt[2][jj[0]]
                                elif sol_hi and bot is not None \
                                        and abs(bot - d_rel) < 1.0 \
                                        and wt[1] is not None:
                                    # fluid below: solid-side run
                                    # bottom, W = Wb, T = 0
                                    # (enforced bottom condition)
                                    wthi = np.array([wt[1], 0.0],
                                                    dtype=complex)
                            lo = _side_coeffs(
                                l, w, d_rel, ents[i_rel]["mat"],
                                ylo * cs[l],
                                None if wtlo is None
                                else wtlo * ct_[l],
                                yp4=yplo * cs[l])
                            hi = _side_coeffs(
                                l, w, d_rel,
                                ents[i_rel + 1]["mat"],
                                yhi * cs[l],
                                None if wthi is None
                                else wthi * ct_[l],
                                yp4=yphi * cs[l])
                            for k in keys:
                                dc[k][l] = hi[k] - lo[k]
                    jU, jV, jR, jS, jW, jT = _jump_rows(
                        hLM, d_rel, ang, dc)
                    # ---- first-order solves per target l'
                    for lp in range(max(1, abs(mp)), lmax + 1):
                        entsp = _entries(
                            mats, _icut(stack, lp, r0, tol_prune),
                            r0, isrc)
                        if where == "top":
                            U1, V1 = spheroidal_unit(
                                lp, w, entsp,
                                np.zeros(4, dtype=complex),
                                bc_rhs=(jR[lp], jS[lp]))
                            # receiver advection
                            U1 += hLM * (ang.G0 @ adv["Up"])[lp]
                            V1 += hLM * (ang.GA_v @ adv["Vp"]
                                         + ang.GA_w @ adv["Wp"])[lp]
                        else:
                            ii = [i for i in range(len(entsp) - 1)
                                  if abs(entsp[i]["r_top"] - d_rel)
                                  < 1.0]
                            if not ii:
                                continue
                            # row inhomogeneity per interface type
                            # (delta convention above - below; the
                            # solid-below/fluid-above S row is
                            # +S_solid(lo) = -[t1]_B, hence -jS)
                            if sol_lo and sol_hi:
                                rhs = np.array(
                                    [jU[lp], jV[lp], jR[lp],
                                     jS[lp]], dtype=complex)
                            elif sol_hi:        # fluid below (CMB)
                                rhs = np.array(
                                    [jU[lp], jR[lp], jS[lp]],
                                    dtype=complex)
                            elif sol_lo:        # fluid above (ICB)
                                rhs = np.array(
                                    [jU[lp], jR[lp], -jS[lp]],
                                    dtype=complex)
                            else:               # fluid-fluid
                                rhs = np.array(
                                    [jU[lp], jR[lp]],
                                    dtype=complex)
                            U1, V1 = spheroidal_unit(
                                lp, w, entsp,
                                np.zeros(4, dtype=complex),
                                iface_rhs={ii[0]: rhs})
                        Wu1[js, iw, lp, mp + 4] += U1
                        Wv1[js, iw, lp, mp + 4] += V1
                        entsp_t, botp = _tor_entries(
                            mats, lp, r0, isrc, tol_prune)
                        if entsp_t is None:
                            continue
                        if where == "top":
                            W1 = toroidal_unit(
                                lp, w, entsp_t, botp, 0.0,
                                bc_rhs=(0.0, jT[lp]))
                            W1 += hLM * (ang.GC_v @ adv["Vp"]
                                         + ang.GC_w @ adv["Wp"])[lp]
                        elif sol_lo and sol_hi:
                            jj = [i for i in
                                  range(len(entsp_t) - 1)
                                  if abs(entsp_t[i]["r_top"]
                                         - d_rel) < 1.0]
                            if not jj:
                                continue
                            W1 = toroidal_unit(
                                lp, w, entsp_t, botp, 0.0,
                                iface_rhs={jj[0]: (jW[lp],
                                                   jT[lp])})
                        elif sol_hi and botp is not None \
                                and abs(botp - d_rel) < 1.0:
                            # fluid-solid run bottom: [t1]_C on the
                            # solid side (incl. the fluid-pressure
                            # conversion term through dc["ciso"])
                            W1 = toroidal_unit(
                                lp, w, entsp_t, botp, 0.0,
                                bc_rhs=(jT[lp], 0.0))
                        else:
                            continue   # interface below/inside the
                            # fluid: no SH there
                        Wt1[js, iw, lp, mp + 4] += W1
    out = {}
    u = np.zeros((nsrc, nst, 3, nw), dtype=complex)
    for js in range(nsrc):
        for iw in range(nw):
            u_sf = spheroidal_reconstruct(Wu1[js, iw], Wv1[js, iw],
                                          dirs_sf)
            u[js, :, :, iw] = u_sf @ Qrot
    out["psv"] = u
    u = np.zeros((nsrc, nst, 3, nw), dtype=complex)
    for js in range(nsrc):
        for iw in range(nw):
            u_sf = toroidal_reconstruct(Wt1[js, iw], dirs_sf)
            u[js, :, :, iw] = u_sf @ Qrot
    out["sh"] = u
    return out


# ----------------------------------------------------------------
# A3 MVP driver (toroidal-only [SH->SH]; kept, gates depend on it)
# ----------------------------------------------------------------
def toroidal_relief_spectra(layers, src_xyz, M_list, w_arr,
                            station_dirs, relief, Q=None,
                            q_sign=-1.0, lmax=250,
                            tol_prune=TOL_PRUNE):
    """du^SH spectra (nsrc, nst, 3, nw), FIRST ORDER in the relief,
    [SH->SH] block only (complete for pure-SH unperturbed fields
    and L=0; see docs section 1 for the corrected parity structure).

    relief: list of (where, L, M, h_LM) with where in
    {'top', 'bottom'} (outer surface / bottom of the outermost solid
    run) and h_LM the coefficient in meters of fully-normalized
    Y_LM. Mr*-type sources (|m| <= 1)."""
    stack = make_stack(layers)
    r0 = float(np.linalg.norm(src_xyz))
    isrc = [i for i, e in enumerate(stack)
            if e["r_bot"] < r0 < e["r_top"]][0]
    Qrot = source_frame(src_xyz)
    poleT = _pole_coupling(lmax)
    hs = []
    for M in M_list:
        M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
        hs.append(M_sf @ np.array([0.0, 0.0, 1.0]))
    w_arr = np.asarray(w_arr, dtype=complex)
    nsrc, nst, nw = len(M_list), len(station_dirs), len(w_arr)
    dirs_sf = station_dirs @ Qrot.T
    store = {}
    Wt1 = np.zeros((nsrc, nw, lmax + 1, 9), dtype=complex)
    for iw, w in enumerate(w_arr):
        mats = _mats_at(stack, w, Q, q_sign)
        msrc = mats[isrc]
        mu = msrc["mu"]
        rho = msrc["rho"]
        Wa0 = np.zeros(lmax + 1, dtype=complex)
        Wb0 = np.zeros(lmax + 1, dtype=complex)
        entsl, botl = {}, {}
        for l in range(1, lmax + 1):
            ents, bot = _tor_entries(mats, l, r0, isrc, tol_prune)
            entsl[l], botl[l] = ents, bot
            wa, wb, _ = toroidal_unit(l, w, ents, bot,
                                      1.0 / (mu * r0 ** 2),
                                      full=True)
            Wa0[l] = wa
            Wb0[l] = 0.0 if wb is None else wb
        a_top = stack[-1]["r_top"]
        for js in range(nsrc):
            hvec = hs[js]
            for where, L, M, hLM in relief:
                for m in (-1, 1):
                    Dm = (poleT[m][:, 0] * hvec[0]
                          + poleT[m][:, 1] * hvec[1])
                    mp = m + M
                    if abs(mp) > 4:
                        continue
                    ang = _ang_cache(store, L, M, m, lmax)
                    d = a_top if where == "top" else None
                    if where == "bottom":
                        if botl[1] is None:
                            raise ValueError(
                                "bottom relief needs a bottom "
                                "boundary (annulus run)")
                        d = botl[1]
                    W0 = (Wa0 if where == "top" else Wb0) * Dm
                    ll = np.arange(lmax + 1)
                    dTdr0 = W0 * (mu * (ll * (ll + 1.0) - 2.0)
                                  / d ** 2 - rho * w * w)
                    src_amp = 2.0 * mu * W0 / d ** 2
                    bc_l = -hLM * (ang.I1 @ dTdr0
                                   - ang.I2 @ src_amp)
                    adv = (hLM * (ang.I1 @ (W0 / a_top))
                           if where == "top" else None)
                    for lp in range(max(1, abs(mp)), lmax + 1):
                        if bc_l[lp] == 0.0:
                            continue
                        bc = ((bc_l[lp], 0.0) if where == "bottom"
                              else (0.0, bc_l[lp]))
                        ents, bot = entsl.get(lp), botl.get(lp)
                        if ents is None:
                            continue
                        if where == "bottom" and bot is None:
                            continue
                        W1 = toroidal_unit(lp, w, ents, bot, 0.0,
                                           bc_rhs=bc)
                        if adv is not None:
                            W1 = W1 + adv[lp]
                        Wt1[js, iw, lp, mp + 4] += W1
    out = np.zeros((nsrc, nst, 3, nw), dtype=complex)
    for js in range(nsrc):
        for iw in range(nw):
            u_sf = toroidal_reconstruct(Wt1[js, iw], dirs_sf)
            out[js, :, :, iw] = u_sf @ Qrot
    return out
