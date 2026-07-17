"""Rung A: production N-layer per-(l,m) spectral sweep.

Direct forced solves for BOTH parities (spheroidal P-SV and toroidal
SH) on an arbitrary concentric stack of UNIFORM solid/fluid layers,
per (l, omega) at complex omega on the DSM damped axis — the
"block-tridiagonal sweep in (l,m) space" of the roadmap (design:
docs/rung_a_spectral.md). Generalizes spheroidal_ref (mini-tipsv,
fixed IC|OC|shell topology, spheroidal only); the toroidal part is a
direct forced solve too, which contains the quasi-static response
exactly and so REPLACES the mode-sum + static-completion machinery
of toroidal_modes for reference synthesis.

Conventions are inherited unchanged from the validated modules:
  * displacement basis, (U, V, R, S)/(W, T) vectors, Legendre
    recurrences: spheroidal_ref / toroidal_ref (momentfit family);
  * causal constant-Q moduli, q_sign, complex frequencies: q_factor;
  * spheroidal source jumps: spheroidal_ref.source_jumps with the
    1/L S-row loads (FD-verified, tipsv-anchored);
  * toroidal source jump (docs/rung_a_spectral.md section 3):
    [W] = q/(mu(w) r0^2), [T] = 0 — frequency-independent,
    consistent with the FD-verified static jumps.

Layer pruning: an interface at radius r below the source is dropped,
together with everything beneath it, when the round-trip factor
(r/r0)^(2l) < tol_prune; the lowest kept layer then keeps only its
regular (j-type) columns with no bottom condition (the proven
mini-tipsv shell-only construction, applied per l). Scaled-series
bases (spheroidal_ref._fl_scaled) are used for l >= L_SERIES with a
PER-LAYER reference radius zref = r_top(layer), so every column is
O(1) at its layer top; the assembled system is column-rescaled
before the solve (solves are column-scale invariant).

MVP limits (documented in the design doc): solid outermost layer,
source in a solid layer, Mr*-type moment tensors (|m| <= 1), l >= 1,
no gravity.
"""

import numpy as np

from .toroidal_ref import source_frame, toroidal_reconstruct
from .toroidal_modes import _pole_coupling
from .spheroidal_ref import (L_SERIES, _fl, _fl_scaled, q_factor,
                             solid_cols_c, source_jumps,
                             spheroidal_pole, spheroidal_reconstruct)

TOL_PRUNE = 1.0e-12


# ----------------------------------------------------------------
# model stack
# ----------------------------------------------------------------
def make_stack(layers):
    """Normalize a model description into a stack (centre outward).
    layers: list of dicts with r_top [m], rho, vp, and vs (absent or
    0 -> inviscid fluid), SI units. Returns list with r_bot added."""
    st = []
    r_bot = 0.0
    for ly in layers:
        vs = float(ly.get("vs", 0.0) or 0.0)
        e = dict(r_bot=r_bot, r_top=float(ly["r_top"]),
                 rho=float(ly["rho"]), vp=float(ly["vp"]), vs=vs,
                 solid=vs > 0.0)
        if e["r_top"] <= r_bot:
            raise ValueError("layer radii must increase")
        st.append(e)
        r_bot = e["r_top"]
    if not st[-1]["solid"]:
        raise ValueError("MVP requires a solid outermost layer")
    return st


def _mats_at(stack, w, Q, q_sign):
    """Per-layer complex moduli at complex w (causal constant-Q on
    the shear moduli of all solid layers, Qkappa = inf — the
    campaign material convention)."""
    fac2 = q_factor(w, Q, q_sign) ** 2 if Q else 1.0
    out = []
    for e in stack:
        m = dict(e)
        if e["solid"]:
            mu0 = e["rho"] * e["vs"] ** 2
            ka0 = e["rho"] * e["vp"] ** 2 - 4.0 / 3.0 * mu0
            m["mu"] = mu0 * fac2
            m["lam"] = ka0 - 2.0 / 3.0 * m["mu"]
        out.append(m)
    return out


def _icut(stack, l, r0, tol_prune):
    """Deepest kept layer index: layers below an interface with
    round-trip factor (r/r0)^(2l) < tol_prune are invisible."""
    icut = 0
    for i in range(1, len(stack)):
        rb = stack[i]["r_bot"]
        if rb < r0 and (rb / r0) ** (2 * l) < tol_prune:
            icut = i
    return icut


def _entries(mats, icut, r0, isrc):
    """Kept-layer entry list bottom->top with the source layer split
    at r0; the below-half carries src_top=True (its top interface
    takes the source jump)."""
    ents = []
    for i in range(icut, len(mats)):
        m = mats[i]
        if i == isrc:
            ents.append(dict(mat=m, r_bot=m["r_bot"], r_top=r0,
                             src_top=True))
            ents.append(dict(mat=m, r_bot=r0, r_top=m["r_top"],
                             src_top=False))
        else:
            ents.append(dict(mat=m, r_bot=m["r_bot"],
                             r_top=m["r_top"], src_top=False))
    return ents


# ----------------------------------------------------------------
# per-layer column matrices
# ----------------------------------------------------------------
def _Ysolid(l, w, r, m, kinds, zref):
    """(4, 2*len(kinds)) matrix of (U, V, R, S) columns at radius r
    for the uniform solid m; column order per kind: (P, S)."""
    cols = []
    for kind in kinds:
        P, S = solid_cols_c(kind, l, w, r, m["rho"], m["lam"],
                            m["mu"], zref_a=zref)
        cols += [P[:, 0], S[:, 0]]
    return np.stack(cols, axis=1)


def _Yfluid(l, w, r, m, kinds, zref):
    """(2, len(kinds)) matrix of (u_r, s_rr) fluid-potential columns
    (s_rr = -p), with the same scaled-series option as the solids."""
    kf = w / m["vp"]
    cols = []
    for kind in kinds:
        if zref is None:
            f, fp = _fl(kind, l, kf * r)
        else:
            f, fp = _fl_scaled(kind, l, kf * r, kf * zref)
        cols.append(np.array([kf * fp, -m["rho"] * w * w * f],
                             dtype=complex)[:, 0])
    return np.stack(cols, axis=1)


def _Ytor(l, w, r, m, kinds, zref):
    """(2, len(kinds)) matrix of toroidal (W, T) columns,
    T = mu (W' - W/r)."""
    vs = np.sqrt(m["mu"] / m["rho"])
    ks = w / vs
    cols = []
    for kind in kinds:
        if zref is None:
            f, fp = _fl(kind, l, ks * r)
        else:
            f, fp = _fl_scaled(kind, l, ks * r, ks * zref)
        cols.append(np.array([f, m["mu"] * (ks * fp - f / r)],
                             dtype=complex)[:, 0])
    return np.stack(cols, axis=1)


def _solve_scaled(A, b):
    sc = np.max(np.abs(A), axis=0)
    sc[sc == 0] = 1.0
    return np.linalg.solve(A / sc, b) / sc


# ----------------------------------------------------------------
# spheroidal forced solve for one (l, w)
# ----------------------------------------------------------------
def spheroidal_unit(l, w, ents, jump, iface_rhs=None, bc_rhs=None,
                    full=False):
    """Solve the kept stack with source jump 4-vector at the split
    interface; returns (U, V) at the outer surface. Entry 0 always
    uses only its regular (j) columns with no bottom condition
    (centre ball regularity, or the pruned invisible bottom).

    Rung-A3b extensions (no-ops by default): iface_rhs = {i: vec}
    adds an inhomogeneity to the interface rows between ents[i] and
    ents[i+1] (welded 4-vector; fluid-adjacent interfaces not yet
    supported); bc_rhs = (R_val, S_val) inhomogeneous free-surface
    rows; full=True additionally returns a dict with the two-sided
    interface y-vectors and the surface y-vector."""
    scaled = l >= L_SERIES
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        zr = e["r_top"] if scaled else None
        fn = _Ysolid if e["mat"]["solid"] else _Yfluid
        Yb.append(fn(l, w, e["r_bot"], e["mat"], kinds, zr)
                  if i > 0 else None)
        Yt.append(fn(l, w, e["r_top"], e["mat"], kinds, zr))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        lo, hi = ents[i], ents[i + 1]
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        Ylo, Yhi = Yt[i], Yb[i + 1]
        ls, hs = lo["mat"]["solid"], hi["mat"]["solid"]
        if ls and hs:
            A[row:row + 4, chi] = Yhi
            A[row:row + 4, clo] = -Ylo
            if lo["src_top"]:
                b[row:row + 4] = jump
            if iface_rhs and i in iface_rhs:
                b[row:row + 4] = b[row:row + 4] + iface_rhs[i]
            row += 4
        elif ls and not hs:                 # solid below, fluid above
            assert not (iface_rhs and i in iface_rhs), \
                "iface_rhs on fluid-adjacent interfaces: A3b stage 3"
            A[row, chi] = Yhi[0]
            A[row, clo] = -Ylo[0]           # u_r
            A[row + 1, chi] = Yhi[1]
            A[row + 1, clo] = -Ylo[2]       # s_rr
            A[row + 2, clo] = Ylo[3]        # solid-side s_rt = 0
            row += 3
        elif hs:                            # fluid below, solid above
            A[row, chi] = Yhi[0]
            A[row, clo] = -Ylo[0]
            A[row + 1, chi] = Yhi[2]
            A[row + 1, clo] = -Ylo[1]
            A[row + 2, chi] = Yhi[3]
            row += 3
        else:                               # fluid-fluid
            A[row:row + 2, chi] = Yhi
            A[row:row + 2, clo] = -Ylo
            row += 2
    ctop = slice(ofs[-2], ofs[-1])
    A[row, ctop] = Yt[-1][2]                # surface s_rr = 0
    A[row + 1, ctop] = Yt[-1][3]            # surface s_rt = 0
    if bc_rhs is not None:
        b[row] = b[row] + bc_rhs[0]
        b[row + 1] = b[row + 1] + bc_rhs[1]
    assert row + 2 == n, "row/column count mismatch"
    x = _solve_scaled(A, b)
    cu = x[ofs[-2]:ofs[-1]]
    Ua, Va = Yt[-1][0] @ cu, Yt[-1][1] @ cu
    if not full:
        return Ua, Va

    def _yv(e, i0, r):
        kinds = ("j",) if i0 == 0 else ("j", "y")
        zr = e["r_top"] if scaled else None
        fn = _Ysolid if e["mat"]["solid"] else _Yfluid
        return fn(l, w, r, e["mat"], kinds, zr) \
            @ x[ofs[i0]:ofs[i0 + 1]]

    def _ypv(e, i0, r, h=1.0):
        return (_yv(e, i0, r + h) - _yv(e, i0, r - h)) / (2.0 * h)

    # radial derivatives via basis FD on the SOLVED coefficients —
    # no Y'Y^-1 inversion (its conditioning costs ~1e-3 relative on
    # derivatives at l >= L_SERIES; measured 2026-07-17)
    info = {"surface": Yt[-1] @ cu,
            "surface_p": _ypv(ents[-1], len(ents) - 1,
                              ents[-1]["r_top"])}
    for i in range(len(ents) - 1):
        ri = ents[i]["r_top"]
        info[i] = (Yt[i] @ x[ofs[i]:ofs[i + 1]],
                   Yb[i + 1] @ x[ofs[i + 1]:ofs[i + 2]],
                   _ypv(ents[i], i, ri),
                   _ypv(ents[i + 1], i + 1, ri))
    return Ua, Va, info


# ----------------------------------------------------------------
# l = 0 (radial) spheroidal forced solve — the branch mini-tipsv
# omits; it feeds ONLY the m = 0 (Mrr-type) source and ONLY the
# radial displacement, with weight growing toward the high-k end of
# the ULP band (radial-mode branch). Reduced 2-vector y0 = (U, R);
# the S-type solution and the V/S rows do not exist at l = 0.
# Source load (pure Mrr in the source frame, M = M0 e3 e3):
# E = qy U'(r0) -> F0 = (0, qy/r0^2), F1 = (0, 2 qy/r0^3), the l = 0
# reduction of the FD-verified m = 0 loads. (A general M would add a
# horizontal-trace load M_hh U(r0)/r0 — zero for the campaign's
# Mr*-type sources; documented deferral.)
# ----------------------------------------------------------------
def _Ysolid0(w, r, m, kinds):
    """(U, R) columns of the l = 0 radial P-type solutions."""
    cols = []
    for kind in kinds:
        P, _ = solid_cols_c(kind, 0, w, r, m["rho"], m["lam"],
                            m["mu"])
        cols.append(P[[0, 2], 0])
    return np.stack(cols, axis=1)


def _sph0_jump(w, r0, m):
    """[y0] = F1 + A(r0) F0 for the unit (qy = 1) Mrr load."""
    h = 1.0e-3
    Y0 = _Ysolid0(w, r0, m, ("j", "y"))
    Yp = _Ysolid0(w, r0 + h, m, ("j", "y"))
    Ym = _Ysolid0(w, r0 - h, m, ("j", "y"))
    A = ((Yp - Ym) / (2.0 * h)) @ np.linalg.inv(Y0)
    F0 = np.array([0.0, 1.0 / r0 ** 2], dtype=complex)
    F1 = np.array([0.0, 2.0 / r0 ** 3], dtype=complex)
    return F1 + A @ F0


def spheroidal_unit_l0(w, ents, jump):
    """l = 0 forced solve; every interface carries (U, R) continuity
    (solid s_rt vanishes identically at l = 0), free surface R = 0.
    Returns U at the outer surface. No pruning (visibility factor is
    1 at l = 0) and no scaled bases (raw Bessel is safe at l = 0)."""
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j",) if i == 0 else ("j", "y")
        fn = _Ysolid0 if e["mat"]["solid"] else \
            (lambda w_, r_, m_, k_: _Yfluid(0, w_, r_, m_, k_, None))
        Yb.append(fn(w, e["r_bot"], e["mat"], kinds)
                  if i > 0 else None)
        Yt.append(fn(w, e["r_top"], e["mat"], kinds))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    row = 0
    for i in range(len(ents) - 1):
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        A[row:row + 2, chi] = Yb[i + 1]
        A[row:row + 2, clo] = -Yt[i]
        if ents[i]["src_top"]:
            b[row:row + 2] = jump
        row += 2
    ctop = slice(ofs[-2], ofs[-1])
    A[row, ctop] = Yt[-1][1]                # surface R = 0
    assert row + 1 == n, "row/column count mismatch"
    x = _solve_scaled(A, b)
    return Yt[-1][0] @ x[ofs[-2]:ofs[-1]]


# ----------------------------------------------------------------
# toroidal forced solve for one (l, w)
# ----------------------------------------------------------------
def _tor_entries(mats, l, r0, isrc, tol_prune):
    """Entries within the outermost solid run (SH does not enter
    fluid). Returns (ents, bottom_r) with bottom_r the radius for the
    T=0 bottom condition (None: regular centre or pruned bottom), or
    (None, None) if the source is not in the run."""
    jrun = len(mats) - 1
    while jrun > 0 and mats[jrun - 1]["solid"]:
        jrun -= 1
    if isrc < jrun:
        return None, None
    icut = jrun
    for i in range(jrun + 1, len(mats)):
        rb = mats[i]["r_bot"]
        if rb < r0 and (rb / r0) ** (2 * l) < tol_prune:
            icut = i
    ents = _entries(mats, icut, r0, isrc)
    rb = mats[icut]["r_bot"]
    if icut > jrun or rb <= 0.0 \
            or (rb / r0) ** (2 * l) < tol_prune:
        bottom_r = None
    else:
        bottom_r = rb
    return ents, bottom_r


def toroidal_unit(l, w, ents, bottom_r, jumpW, bc_rhs=(0.0, 0.0),
                  full=False, iface_rhs=None):
    """Toroidal forced solve; jump [W] = jumpW, [T] = 0 at the split
    interface; returns W at the outer surface.

    bc_rhs: inhomogeneous values for the (bottom T-row, surface
    T-row) — used by the first-order TFE relief solves (rung A3);
    zeros = unchanged behaviour. iface_rhs = {i: (dW, dT)} adds an
    inhomogeneity to the welded interface rows between ents[i] and
    ents[i+1]. full=True returns (W_surface, W_bottom, info) with
    W_bottom the displacement at the run-bottom radius (None when
    there is no bottom row) and info the two-sided interface
    (W, T) vectors plus the surface (W, T)."""
    scaled = l >= L_SERIES
    Yb, Yt, ncol = [], [], []
    for i, e in enumerate(ents):
        kinds = ("j", "y") if (i > 0 or bottom_r is not None) \
            else ("j",)
        zr = e["r_top"] if scaled else None
        Yb.append(_Ytor(l, w, e["r_bot"], e["mat"], kinds, zr)
                  if (i > 0 or bottom_r is not None) else None)
        Yt.append(_Ytor(l, w, e["r_top"], e["mat"], kinds, zr))
        ncol.append(Yt[-1].shape[1])
    n = int(np.sum(ncol))
    ofs = np.concatenate([[0], np.cumsum(ncol)]).astype(int)
    A = np.zeros((n, n), dtype=complex)
    b = np.zeros(n, dtype=complex)
    row = 0
    if bottom_r is not None:
        A[row, 0:ofs[1]] = Yb[0][1]         # T = 0 at the run bottom
        b[row] = bc_rhs[0]
        row += 1
    for i in range(len(ents) - 1):
        clo = slice(ofs[i], ofs[i + 1])
        chi = slice(ofs[i + 1], ofs[i + 2])
        A[row:row + 2, chi] = Yb[i + 1]
        A[row:row + 2, clo] = -Yt[i]
        if ents[i]["src_top"]:
            b[row] = jumpW
        if iface_rhs and i in iface_rhs:
            b[row:row + 2] = b[row:row + 2] + np.asarray(
                iface_rhs[i], dtype=complex)
        row += 2
    ctop = slice(ofs[-2], ofs[-1])
    A[row, ctop] = Yt[-1][1]                # surface T = 0
    b[row] = b[row] + bc_rhs[1]
    assert row + 1 == n, "row/column count mismatch"
    x = _solve_scaled(A, b)
    Wa = Yt[-1][0] @ x[ofs[-2]:ofs[-1]]
    if not full:
        return Wa
    Wb = (Yb[0][0] @ x[0:ofs[1]]) if bottom_r is not None else None
    info = {"surface": Yt[-1] @ x[ofs[-2]:ofs[-1]]}
    for i in range(len(ents) - 1):
        info[i] = (Yt[i] @ x[ofs[i]:ofs[i + 1]],
                   Yb[i + 1] @ x[ofs[i + 1]:ofs[i + 2]])
    return Wa, Wb, info


# ----------------------------------------------------------------
# spectra driver
# ----------------------------------------------------------------
def spectral_poles(lmax, parts=("psv", "sh")):
    """Precompute the pole couplings (DY, DG, poleT) once — pass as
    poles= to spectral_spectra to avoid recomputation per worker."""
    DY, DG = spheroidal_pole(lmax) if "psv" in parts else (None,
                                                           None)
    poleT = _pole_coupling(lmax) if "sh" in parts else None
    return DY, DG, poleT


def spectral_spectra(layers, src_xyz, M_list, w_arr, station_dirs,
                     Q=None, q_sign=-1.0, lmax=250,
                     tol_prune=TOL_PRUNE, parts=("psv", "sh"),
                     poles=None, l0=True):
    """Total-field spectra by the per-(l,m) spectral sweep.

    layers: model description for make_stack (centre outward);
    M_list: earth-frame moment tensors (Mr*-type); w_arr: complex
    frequencies; station_dirs: unit vectors (nst, 3).
    Returns dict with keys from parts, each (nsrc, nst, 3, nw)."""
    stack = make_stack(layers)
    r0 = float(np.linalg.norm(src_xyz))
    isrc = None
    for i, e in enumerate(stack):
        if e["r_bot"] < r0 < e["r_top"]:
            isrc = i
    if isrc is None or not stack[isrc]["solid"]:
        raise ValueError("source must lie strictly inside a solid "
                         "layer")
    Qrot = source_frame(src_xyz)
    do_psv, do_sh = "psv" in parts, "sh" in parts
    if poles is None:
        poles = spectral_poles(lmax, parts)
    DY, DG, poleT = poles
    srcs = []
    for M in M_list:
        M_sf = Qrot @ np.asarray(M, dtype=float) @ Qrot.T
        srcs.append((M_sf[2, 2], M_sf[2, 0] + 0j, M_sf[2, 1] + 0j))
    w_arr = np.asarray(w_arr, dtype=complex)
    nw, nst, nsrc = len(w_arr), len(station_dirs), len(M_list)
    dirs_sf = station_dirs @ Qrot.T

    Wu = np.zeros((nsrc, nw, lmax + 1, 9), dtype=complex)
    Wv = np.zeros_like(Wu)
    Wt = np.zeros_like(Wu)
    for iw, w in enumerate(w_arr):
        mats = _mats_at(stack, w, Q, q_sign)
        msrc = mats[isrc]
        if do_psv and l0:
            ents0 = _entries(mats, 0, r0, isrc)
            U0 = spheroidal_unit_l0(w, ents0, _sph0_jump(w, r0,
                                                         msrc))
            for js, (mzz, _, _) in enumerate(srcs):
                if mzz != 0.0:
                    Wu[js, iw, 0, 0 + 4] += mzz * DY[0] * U0
        for l in range(1, lmax + 1):
            zr_src = msrc["r_top"] if l >= L_SERIES else None
            L = l * (l + 1.0)
            if do_psv:
                icut = _icut(stack, l, r0, tol_prune)
                if icut > isrc:
                    icut = isrc
                ents = _entries(mats, icut, r0, isrc)
                F0v = np.array([0, 0, 0, 1.0 / (L * r0 ** 2)],
                               dtype=complex)
                F1v = np.array([0, 0, -1.0 / r0 ** 3,
                                3.0 / (L * r0 ** 3)], dtype=complex)
                J_rh = source_jumps(l, w, r0, msrc["rho"],
                                    msrc["lam"], msrc["mu"], F0v,
                                    F1v, zref_a=zr_src)
                U_rh, V_rh = spheroidal_unit(l, w, ents, J_rh)
                F0y = np.array([0, 0, 1.0 / r0 ** 2, 0],
                               dtype=complex)
                F1y = np.array([0, 0, 2.0 / r0 ** 3, 0],
                               dtype=complex)
                J_y = source_jumps(l, w, r0, msrc["rho"],
                                   msrc["lam"], msrc["mu"], F0y,
                                   F1y, zref_a=zr_src)
                U_y, V_y = spheroidal_unit(l, w, ents, J_y)
            if do_sh:
                ents_t, bot_r = _tor_entries(mats, l, r0, isrc,
                                             tol_prune)
                Wt_unit = 0.0
                if ents_t is not None:
                    Wt_unit = toroidal_unit(
                        l, w, ents_t, bot_r,
                        1.0 / (msrc["mu"] * r0 ** 2))
            for js, (mzz, mzx, mzy) in enumerate(srcs):
                if do_psv and mzz != 0.0:
                    qy = mzz * DY[l]
                    Wu[js, iw, l, 0 + 4] += qy * U_y
                    Wv[js, iw, l, 0 + 4] += qy * V_y
                for m in (-1, 1):
                    if do_psv:
                        D = DG[m][l][0] * mzx + DG[m][l][1] * mzy
                        if D != 0.0:
                            Wu[js, iw, l, m + 4] += D * U_rh
                            Wv[js, iw, l, m + 4] += D * V_rh
                    if do_sh:
                        Dt = (poleT[m][l][0] * mzx
                              + poleT[m][l][1] * mzy)
                        if Dt != 0.0 and np.any(Wt_unit != 0.0):
                            Wt[js, iw, l, m + 4] += Dt * Wt_unit
    out = {}
    if do_psv:
        u = np.zeros((nsrc, nst, 3, nw), dtype=complex)
        for js in range(nsrc):
            for iw in range(nw):
                u_sf = spheroidal_reconstruct(Wu[js, iw],
                                              Wv[js, iw], dirs_sf)
                u[js, :, :, iw] = u_sf @ Qrot
        out["psv"] = u
    if do_sh:
        u = np.zeros((nsrc, nst, 3, nw), dtype=complex)
        for js in range(nsrc):
            for iw in range(nw):
                u_sf = toroidal_reconstruct(Wt[js, iw], dirs_sf)
                u[js, :, :, iw] = u_sf @ Qrot
        out["sh"] = u
    return out
