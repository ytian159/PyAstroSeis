"""BEM interaction-matrix assembly.

Faithful ports of the MATLAB assembly stack:
  - subgrid_gen.m                          (per-face quadrature sub-grid)
  - int_self_trac/G/A/B_tri_vec.m          (singular self integrals)
  - trac/G/A/B_tri_int_vec(.m, _diflay.m)  (one collocation row)
  - cal_traction_tri_vec.m / cal_T_st.m    (elastic traction matrix)
  - cal_G_st.m                             (elastic displacement matrix)
  - cal_A_st.m / cal_B_st.m                (acoustic matrices, fluid layer)

Same-surface calls (faces1 is faces2) insert singular self blocks;
cross-surface calls use the "_diflay" regular path. NOTE: MATLAB decides
this by comparing face COUNTS (ngd1==ngd2); the port uses object
identity, which is equivalent for all call sites in this code base and
correct when two distinct surfaces happen to have equal face counts.

Phase 1 keeps the exact MATLAB algorithm, including the 300x300
self-integration grid and per-frequency rebuild of geometry. Known
Phase 2 targets are marked with PHASE2 comments.

qp_fac: Qp = qp_fac * Q for the complex velocities. Default 2.5 is the
original MATLAB value; pass 0.75*(vp0/vs0)**2 for the physically
consistent pure-shear-attenuation relation (see solver.qp_factor).
"""

import numpy as np

from .greens import (bmat_func, g_singular_value, green_traction_tensor,
                     greens_func_deri_p, greens_func_p)
from .quadrature import simplex_rule, simplex_rule_composite, triasymq

NINT = 10          # quadrature degree (MATLAB: "can only be 10, 20, 40, 50")
NXI_SELF = 300     # self-integration grid resolution (int_self, nxi=300)
RIDFAC = 1.0
QP_FAC_LEGACY = 2.5

# distance-adaptive quadrature tiers: (max distance / source-element
# size, rule degree). Pairs closer than 4 element sizes get the full
# degree-10 rule (as MATLAB uses everywhere); beyond 10 sizes the
# kernel is smooth enough for the 3-point rule — PROVIDED the kernel
# oscillation across the element is resolved. The oscillation cap
# OSC_LIMITS enforces that: low-order tiers are only allowed when
# |k|*h is below the listed limits (deg-2 below 0.9, deg-5 below 1.8),
# otherwise the pair is promoted back toward the full rule. On meshes
# that are under-resolved at the band top (like the 1584-face demo
# mesh at 0.88 Hz, ~2 elements per S wavelength) adaptive therefore
# safely degrades to full quadrature; the payoff comes on properly
# resolved meshes (6-10 elements/wavelength) and at lower frequencies.
# Gated by tests/test_adaptive_quad.py.
TIERS_DEFAULT = ((4.0, 10), (10.0, 5), (None, 2))
OSC_LIMITS_DEFAULT = (0.9, 1.8)


def wave_speeds(lamda, mu, rho, Q, qp_fac=QP_FAC_LEGACY, w=None,
                disp_ref_hz=0.0):
    """Complex attenuated vp, vs (header of every cal_*_st routine).

    disp_ref_hz > 0 adds causal constant-Q physical dispersion
    (Kanamori-Anderson), matching DSMsynTI's computeCoef convention:
    the real velocity is scaled by 1 + ln(f/f_ref)/(pi Q) at the
    evaluation frequency f = Re(w)/2pi (f_ref = 1 Hz for PREM-type
    models). Default 0 keeps the historical non-dispersive behavior
    (bitwise)."""
    vp0 = np.sqrt((lamda + 2 * mu) / rho)
    vs0 = np.sqrt(mu / rho)
    Qp = qp_fac * Q
    if disp_ref_hz and w is not None:
        lf = np.log(abs(np.real(w)) / (2 * np.pi * disp_ref_hz))
        vp0 = vp0 * (1 + lf / (np.pi * Qp))
        vs0 = vs0 * (1 + lf / (np.pi * Q))
    vp = vp0 / (1 + 1j * 0.5 / Qp)
    vs = vs0 / (1 + 1j * 0.5 / Q)
    return vp, vs


def subgrid_from_rule(faces, ref, wi):
    """Map a unit-simplex rule onto every face. Returns xia, yia, zia of
    shape (N, npts) and the weights (npts,) (they sum to 1/2; the
    physical area enters via `area` and the factor 2 in assembly)."""
    xi, eta = ref[0][None, :], ref[1][None, :]
    lam0 = 1.0 - xi - eta
    xia = faces.A[:, 0:1] * lam0 + faces.B[:, 0:1] * xi + faces.C[:, 0:1] * eta
    yia = faces.A[:, 1:2] * lam0 + faces.B[:, 1:2] * xi + faces.C[:, 1:2] * eta
    zia = faces.A[:, 2:3] * lam0 + faces.B[:, 2:3] * xi + faces.C[:, 2:3] * eta
    return xia, yia, zia, wi


def subgrid_all(faces, nint=NINT):
    """Quadrature sub-grids for all faces (subgrid_gen.m, vectorized)."""
    ref, wi = triasymq(nint, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    return subgrid_from_rule(faces, ref, wi)


def self_ref_grid(nxi=NXI_SELF):
    """Reference-triangle grid for the self integrals
    (cal_*_st.m self-element sections)."""
    xi = np.linspace(0.0, 1.0, nxi)
    dx = xi[1] - xi[0]
    etai = np.linspace(0.0, 1.0, nxi)
    xx, nn = np.meshgrid(xi, etai)          # matches MATLAB meshgrid
    xi1 = xx.flatten(order="F")
    eta1 = nn.flatten(order="F")
    keep = xi1 + eta1 <= 1.0
    refs = np.vstack([xi1[keep], eta1[keep]])
    wis = dx * dx
    return refs, wis


def _self_points(faces, i, refs, ridfac):
    """Physical grid points on face i with the singular disk punched out
    (shared preamble of every int_self_*_tri_vec.m)."""
    A, B, C, ic = faces.A[i], faces.B[i], faces.C[i], faces.ic[i]
    lam0 = 1.0 - refs[0] - refs[1]
    xi = A[0] * lam0 + B[0] * refs[0] + C[0] * refs[1]
    yi = A[1] * lam0 + B[1] * refs[0] + C[1] * refs[1]
    zi = A[2] * lam0 + B[2] * refs[0] + C[2] * refs[1]
    rr = np.sqrt((xi - ic[0]) ** 2 + (yi - ic[1]) ** 2 + (zi - ic[2]) ** 2)
    keep = rr >= faces.r[i] * ridfac
    return xi[keep], yi[keep], zi[keep]


def polar_self_points(faces, i, ntheta=16, nrho=12, ridfac=RIDFAC):
    """Polar quadrature on face i for the annulus between the punched-out
    incircle (radius r*ridfac, same exclusion as the grid scheme, so the
    CPV structure and analytic jump terms are unchanged) and the triangle
    boundary.

    In polar coordinates the 1/r^2 kernel times the rho drho Jacobian is
    O(1), so ~3*ntheta*nrho points replace the ~45,000-point brute-force
    grid at higher accuracy (see tests/test_self_convergence.py).

    Returns (x, y, z, w) with PHYSICAL-measure weights (integral =
    sum(f(x)*w), no extra area factor).
    """
    ic = faces.ic[i]
    nv = faces.nvec[i]
    e1 = faces.A[i] - ic
    e1 = e1 - (e1 @ nv) * nv
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(nv, e1)

    v2 = np.array([[(P - ic) @ e1, (P - ic) @ e2]
                   for P in (faces.A[i], faces.B[i], faces.C[i])])
    # edges as (outward normal m, distance d) with the incenter inside
    ms, ds = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        t = v2[b] - v2[a]
        m = np.array([t[1], -t[0]])
        m = m / np.linalg.norm(m)
        d = m @ v2[a]
        if d < 0:
            m, d = -m, -d
        ms.append(m)
        ds.append(d)
    ms = np.array(ms)
    ds = np.array(ds)

    phiv = np.sort(np.arctan2(v2[:, 1], v2[:, 0]))
    arcs = [(phiv[0], phiv[1]), (phiv[1], phiv[2]),
            (phiv[2], phiv[0] + 2 * np.pi)]
    tq, wq = np.polynomial.legendre.leggauss(ntheta)
    rq, rw = np.polynomial.legendre.leggauss(nrho)
    r0 = faces.r[i] * ridfac

    xs, ys, zs, ws = [], [], [], []
    for a, b in arcs:
        th = 0.5 * (b - a) * tq + 0.5 * (a + b)
        wth = 0.5 * (b - a) * wq
        u = np.column_stack([np.cos(th), np.sin(th)])       # (nt,2)
        den = u @ ms.T                                       # (nt,3)
        with np.errstate(divide="ignore"):
            cand = np.where(den > 1e-12, ds[None, :] / den, np.inf)
        rho_max = cand.min(axis=1)                           # (nt,)
        half = 0.5 * np.clip(rho_max - r0, 0.0, None)        # (nt,)
        rho = r0 + half[:, None] * (rq[None, :] + 1.0)       # (nt,nr)
        wgt = (wth[:, None] * half[:, None] * rw[None, :]) * rho
        px = rho * u[:, 0:1]
        py = rho * u[:, 1:2]
        P = (ic[None, None, :] + px[..., None] * e1[None, None, :]
             + py[..., None] * e2[None, None, :])
        xs.append(P[..., 0].ravel())
        ys.append(P[..., 1].ravel())
        zs.append(P[..., 2].ravel())
        ws.append(wgt.ravel())
    return (np.concatenate(xs), np.concatenate(ys),
            np.concatenate(zs), np.concatenate(ws))


class Geometry:
    """Frequency-independent assembly geometry for one surface.
    Build once per run (PHASE2 hoist); reused across all frequencies.

    self_scheme: "grid" reproduces the MATLAB 300x300 punch-out scheme
    bit-for-bit; "polar" uses polar_self_points (more accurate, ~100x
    fewer self-integration points).
    """

    def __init__(self, faces, nint=NINT, nxi=NXI_SELF, self_scheme="grid",
                 ntheta=16, nrho=12, ntheta_weak=32, nrho_weak=24,
                 ridfac=RIDFAC, quad_mode="full", tiers=TIERS_DEFAULT,
                 osc_limits=OSC_LIMITS_DEFAULT, near_tier=None):
        self.faces = faces
        self.nint = nint
        self.ridfac = ridfac
        self.self_scheme = self_scheme
        self.quad_mode = quad_mode
        self.xia, self.yia, self.zia, self.wi = subgrid_all(faces, nint)
        self.n1a = faces.nvec[:, 0:1]
        self.n2a = faces.nvec[:, 1:2]
        self.n3a = faces.nvec[:, 2:3]
        self.area = faces.area

        # near-singular promotion tier for graded meshes (STRICT NO-OP
        # when near_tier is None): (edge_ratio, (degree, levels)),
        # e.g. (1.5, (10, 2)), prepended to the tier ladder so
        # cross-panel pairs at dist < edge_ratio * h_source get a
        # composite subdivided rule (a small panel's collocation point
        # close to a LARGE neighbor panel makes the 1/r^2 kernels
        # near-singular; the default deg-10 rule silently
        # under-integrates there). The oscillation cap arithmetic in
        # tier_cap is unchanged: with len(osc_limits) relaxation steps
        # the strictest cap lands on the deg-10 tier, never forcing
        # far pairs onto the near rule.
        if near_tier is not None:
            tiers = (tuple(near_tier),) + tuple(tiers)

        # distance-adaptive tiers (quad_mode="adaptive"): per-tier
        # sub-grids, selected per (collocation, source-face) pair by
        # distance / source-element size
        if quad_mode == "adaptive":
            self.h = np.maximum.reduce([faces.a, faces.b, faces.c])
            self.tier_edges = np.array([t[0] for t in tiers[:-1]])
            self.osc_limits = np.asarray(osc_limits, dtype=float)
            self.tier_grids = []
            for _, deg in tiers:
                if isinstance(deg, (tuple, list)):
                    ref, wi = simplex_rule_composite(*deg)
                    grids = subgrid_from_rule(faces, ref, wi)
                elif deg == nint:
                    grids = (self.xia, self.yia, self.zia, self.wi)
                else:
                    ref, wi = simplex_rule(deg)
                    grids = subgrid_from_rule(faces, ref, wi)
                self.tier_grids.append(grids)
        elif quad_mode == "full":
            self.tier_edges = self.tier_grids = self.h = None
            self.osc_limits = None
        else:
            raise ValueError("quad_mode must be 'full' or 'adaptive'")

        # self-integration point sets; the weakly singular G/B kernels
        # get a finer polar rule (their annulus integrand varies more
        # than the traction kernel's; see tests/test_self_convergence.py)
        if self_scheme == "grid":
            self.refs, self.wis = self_ref_grid(nxi)
            self.polar = self.polar_weak = None
        elif self_scheme == "polar":
            self.refs = self.wis = None
            self.polar = [polar_self_points(faces, i, ntheta, nrho, ridfac)
                          for i in range(faces.n)]
            self.polar_weak = [polar_self_points(faces, i, ntheta_weak,
                                                 nrho_weak, ridfac)
                               for i in range(faces.n)]
        else:
            raise ValueError("self_scheme must be 'grid' or 'polar'")

    def tier_cap(self, k_abs):
        """Maximum allowed tier index per source face for wavenumber
        magnitude k_abs (oscillation resolution cap): tier t (0 = full
        degree-10 rule) is only relaxed to lower-order tiers while
        |k|*h stays below osc_limits."""
        kh = k_abs * self.h
        n_low = len(self.tier_grids) - 1
        return n_low - np.searchsorted(self.osc_limits, kh,
                                       side="left").clip(max=n_low)


def int_self_trac(vp, vs, rho, w, faces, i, wis, refs, ridfac=RIDFAC):
    """Self traction integral for face i (int_self_trac_tri_vec.m).
    Returns 3x3 block (before the +eye(3) added by the caller)."""
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi = _self_points(faces, i, refs, ridfac)
    T = green_traction_tensor(vp, vs, rho, w, xi, yi, zi, xs, ys, zs, n1, n2, n3)
    fac = 2.0
    ST = np.array([[(t * wis).sum() * faces.area[i] * fac for t in row]
                   for row in (T[0:3], T[3:6], T[6:9])])
    return ST - 0.5 * np.eye(3)


def int_self_G(vp, vs, rho, w, faces, i, wis, refs, ridfac=RIDFAC):
    """Self displacement integral for face i (int_self_G_tri_vec.m):
    numerical part + analytic disk term."""
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi = _self_points(faces, i, refs, ridfac)
    G = bmat_func(vp, vs, rho, w, xi, yi, zi, xs, ys, zs)
    fac = 2.0
    ST = np.array([[(g * wis).sum() * faces.area[i] * fac for g in row]
                   for row in (G[0:3], G[3:6], G[6:9])])
    return ST + g_singular_value(vp, vs, rho, w, faces.r[i], n1, n2, n3)


def int_self_A(vp, w, faces, i, wis, refs, ridfac=RIDFAC):
    """Self integral of the acoustic normal-derivative kernel
    (int_self_A_tri_vec.m). Scalar."""
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi = _self_points(faces, i, refs, ridfac)
    Gpn = greens_func_deri_p(vp, w, xi, yi, zi, xs, ys, zs, n1, n2, n3)
    ST = (Gpn * wis).sum() * faces.area[i] * 2.0
    return ST - 0.5


def int_self_B(vp, w, faces, i, wis, refs, ridfac=RIDFAC):
    """Self integral of the acoustic single-layer kernel
    (int_self_B_tri_vec.m): numerical part + analytic disk term. Scalar."""
    kp = w / vp
    xs, ys, zs = faces.ic[i]
    xi, yi, zi = _self_points(faces, i, refs, ridfac)
    Gpn = greens_func_p(vp, w, xi, yi, zi, xs, ys, zs)
    ST = (Gpn * wis).sum() * faces.area[i] * 2.0
    R = faces.r[i]
    B_singular = -0.5j * (-1 + np.exp(1j * kp * np.sqrt(R ** 2))) / kp
    return ST + B_singular


def int_self_trac_polar(vp, vs, rho, w, faces, i, pts):
    """Polar-quadrature version of int_self_trac (same CPV exclusion +
    jump term; only the numerical quadrature differs)."""
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi, wq = pts
    T = green_traction_tensor(vp, vs, rho, w, xi, yi, zi, xs, ys, zs, n1, n2, n3)
    ST = np.array([[(t * wq).sum() for t in row]
                   for row in (T[0:3], T[3:6], T[6:9])])
    return ST - 0.5 * np.eye(3)


def int_self_G_polar(vp, vs, rho, w, faces, i, pts):
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi, wq = pts
    G = bmat_func(vp, vs, rho, w, xi, yi, zi, xs, ys, zs)
    ST = np.array([[(g * wq).sum() for g in row]
                   for row in (G[0:3], G[3:6], G[6:9])])
    return ST + g_singular_value(vp, vs, rho, w, faces.r[i], n1, n2, n3)


def int_self_A_polar(vp, w, faces, i, pts):
    xs, ys, zs = faces.ic[i]
    n1, n2, n3 = faces.nvec[i]
    xi, yi, zi, wq = pts
    Gpn = greens_func_deri_p(vp, w, xi, yi, zi, xs, ys, zs, n1, n2, n3)
    return (Gpn * wq).sum() - 0.5


def int_self_B_polar(vp, w, faces, i, pts):
    kp = w / vp
    xs, ys, zs = faces.ic[i]
    xi, yi, zi, wq = pts
    Gp = greens_func_p(vp, w, xi, yi, zi, xs, ys, zs)
    R = faces.r[i]
    B_singular = -0.5j * (-1 + np.exp(1j * kp * np.sqrt(R ** 2))) / kp
    return (Gp * wq).sum() + B_singular


def _assemble_3x3(faces1, faces2, w, kernel, self_block, geom, k_abs=None):
    """Shared skeleton of cal_T_st.m / cal_G_st.m.

    kernel(xi, yi, zi, xs, ys, zs, n1, n2, n3) -> 9 (N2, nsub) arrays.
    self_block(i) -> 3x3 for face i of faces2 (used on same-surface).
    Row/column layout identical to the MATLAB code:
    row j of block-row r is [st_1r; st_2r; st_3r] over source faces.
    """
    same = faces1 is faces2
    n1_, n2_ = faces1.n, faces2.n

    STs = None
    if same:
        STs = np.empty((n2_, 3, 3), dtype=complex)
        for i in range(n2_):
            STs[i] = self_block(i)

    out = np.zeros((3 * n1_, 3 * n2_), dtype=complex)
    fac = 2.0
    n1a, n2a, n3a, area = geom.n1a, geom.n2a, geom.n3a, geom.area
    adaptive = geom.quad_mode == "adaptive"
    cap = geom.tier_cap(k_abs) if adaptive else None
    for j in range(n1_):
        xs, ys, zs = faces1.ic[j]
        if adaptive:
            dist = np.linalg.norm(faces2.ic - faces1.ic[j], axis=1)
            tier = np.minimum(np.searchsorted(geom.tier_edges,
                                              dist / geom.h, side="right"),
                              cap)
            st = [np.zeros(n2_, dtype=complex) for _ in range(9)]
            for t_id, (xia, yia, zia, wi) in enumerate(geom.tier_grids):
                idx = np.nonzero(tier == t_id)[0]
                if idx.size == 0:
                    continue
                T = kernel(xia[idx], yia[idx], zia[idx], xs, ys, zs,
                           n1a[idx], n2a[idx], n3a[idx])
                for k9, t in enumerate(T):
                    st[k9][idx] = (t * wi).sum(axis=1) * area[idx] * fac
        else:
            T = kernel(geom.xia, geom.yia, geom.zia, xs, ys, zs,
                       n1a, n2a, n3a)
            st = [(t * geom.wi).sum(axis=1) * area * fac for t in T]
        if same:
            for k9 in range(9):
                st[k9][j] = STs[j, k9 // 3, k9 % 3]
        # st order: 11,12,13,21,22,23,31,32,33
        out[j, 0:n2_] = st[0]
        out[j, n2_:2 * n2_] = st[3]
        out[j, 2 * n2_:3 * n2_] = st[6]
        out[j + n1_, 0:n2_] = st[1]
        out[j + n1_, n2_:2 * n2_] = st[4]
        out[j + n1_, 2 * n2_:3 * n2_] = st[7]
        out[j + 2 * n1_, 0:n2_] = st[2]
        out[j + 2 * n1_, n2_:2 * n2_] = st[5]
        out[j + 2 * n1_, 2 * n2_:3 * n2_] = st[8]
    return out


def _geom_for(faces, geom, nint, nxi, self_scheme):
    if geom is not None:
        return geom
    return Geometry(faces, nint=nint, nxi=nxi, self_scheme=self_scheme)


def cal_T_st(faces1, faces2, w, lamda, mu, rho, Q, nint=NINT, nxi=NXI_SELF,
             ridfac=RIDFAC, qp_fac=QP_FAC_LEGACY, geom=None,
             self_scheme="grid", disp_ref_hz=0.0):
    """Traction interaction matrix (cal_T_st.m).
    faces1: collocation surface (rows); faces2: source surface (columns).
    geom: prebuilt Geometry for faces2 (hoisted across frequencies)."""
    vp, vs = wave_speeds(lamda, mu, rho, Q, qp_fac, w=w,
                         disp_ref_hz=disp_ref_hz)
    geom = _geom_for(faces2, geom, nint, nxi, self_scheme)

    def kernel(xi, yi, zi, xs, ys, zs, n1, n2, n3):
        return green_traction_tensor(vp, vs, rho, w, xi, yi, zi,
                                     xs, ys, zs, n1, n2, n3)

    if geom.self_scheme == "grid":
        def self_block(i):
            return int_self_trac(vp, vs, rho, w, faces2, i, geom.wis,
                                 geom.refs, geom.ridfac) + np.eye(3)
    else:
        def self_block(i):
            return int_self_trac_polar(vp, vs, rho, w, faces2, i,
                                       geom.polar[i]) + np.eye(3)

    return _assemble_3x3(faces1, faces2, w, kernel, self_block, geom,
                         k_abs=abs(w / vs))


def cal_G_st(faces1, faces2, w, lamda, mu, rho, Q, nint=NINT, nxi=NXI_SELF,
             ridfac=RIDFAC, qp_fac=QP_FAC_LEGACY, geom=None,
             self_scheme="grid", disp_ref_hz=0.0):
    """Displacement interaction matrix (cal_G_st.m)."""
    vp, vs = wave_speeds(lamda, mu, rho, Q, qp_fac, w=w,
                         disp_ref_hz=disp_ref_hz)
    geom = _geom_for(faces2, geom, nint, nxi, self_scheme)

    def kernel(xi, yi, zi, xs, ys, zs, n1, n2, n3):
        return bmat_func(vp, vs, rho, w, xi, yi, zi, xs, ys, zs)

    if geom.self_scheme == "grid":
        def self_block(i):
            return int_self_G(vp, vs, rho, w, faces2, i, geom.wis,
                              geom.refs, geom.ridfac)
    else:
        def self_block(i):
            return int_self_G_polar(vp, vs, rho, w, faces2, i,
                                    geom.polar_weak[i])

    return _assemble_3x3(faces1, faces2, w, kernel, self_block, geom,
                         k_abs=abs(w / vs))


def _assemble_scalar(faces1, faces2, w, kernel, self_value, geom, k_abs=None):
    """Shared skeleton of cal_A_st.m / cal_B_st.m (scalar acoustic
    kernels; (N1, N2) output)."""
    same = faces1 is faces2
    n1_, n2_ = faces1.n, faces2.n

    STs = None
    if same:
        STs = np.empty(n2_, dtype=complex)
        for i in range(n2_):
            STs[i] = self_value(i)

    out = np.zeros((n1_, n2_), dtype=complex)
    n1a, n2a, n3a, area = geom.n1a, geom.n2a, geom.n3a, geom.area
    adaptive = geom.quad_mode == "adaptive"
    cap = geom.tier_cap(k_abs) if adaptive else None
    for j in range(n1_):
        xs, ys, zs = faces1.ic[j]
        if adaptive:
            dist = np.linalg.norm(faces2.ic - faces1.ic[j], axis=1)
            tier = np.minimum(np.searchsorted(geom.tier_edges,
                                              dist / geom.h, side="right"),
                              cap)
            s = np.zeros(n2_, dtype=complex)
            for t_id, (xia, yia, zia, wi) in enumerate(geom.tier_grids):
                idx = np.nonzero(tier == t_id)[0]
                if idx.size == 0:
                    continue
                Gpn = kernel(xia[idx], yia[idx], zia[idx], xs, ys, zs,
                             n1a[idx], n2a[idx], n3a[idx])
                s[idx] = (Gpn * wi).sum(axis=1) * area[idx] * 2.0
        else:
            Gpn = kernel(geom.xia, geom.yia, geom.zia, xs, ys, zs,
                         n1a, n2a, n3a)
            s = (Gpn * geom.wi).sum(axis=1) * area * 2.0
        if same:
            s[j] = STs[j]
        out[j] = s
    return out


def cal_A_st(faces1, faces2, w, lamda, mu, rho, Q, nint=NINT, nxi=NXI_SELF,
             ridfac=RIDFAC, qp_fac=QP_FAC_LEGACY, geom=None,
             self_scheme="grid", disp_ref_hz=0.0):
    """Acoustic normal-derivative matrix for the fluid layer (cal_A_st.m)."""
    vp, _ = wave_speeds(lamda, mu, rho, Q, qp_fac, w=w,
                        disp_ref_hz=disp_ref_hz)
    geom = _geom_for(faces2, geom, nint, nxi, self_scheme)

    def kernel(xi, yi, zi, xs, ys, zs, n1, n2, n3):
        return greens_func_deri_p(vp, w, xi, yi, zi, xs, ys, zs, n1, n2, n3)

    if geom.self_scheme == "grid":
        def self_value(i):
            return int_self_A(vp, w, faces2, i, geom.wis, geom.refs,
                              geom.ridfac) + 1.0
    else:
        def self_value(i):
            return int_self_A_polar(vp, w, faces2, i, geom.polar[i]) + 1.0

    return _assemble_scalar(faces1, faces2, w, kernel, self_value, geom,
                            k_abs=abs(w / vp))


def cal_B_st(faces1, faces2, w, lamda, mu, rho, Q, nint=NINT, nxi=NXI_SELF,
             ridfac=RIDFAC, qp_fac=QP_FAC_LEGACY, geom=None,
             self_scheme="grid", disp_ref_hz=0.0):
    """Acoustic single-layer matrix for the fluid layer (cal_B_st.m)."""
    vp, _ = wave_speeds(lamda, mu, rho, Q, qp_fac, w=w,
                        disp_ref_hz=disp_ref_hz)
    geom = _geom_for(faces2, geom, nint, nxi, self_scheme)

    def kernel(xi, yi, zi, xs, ys, zs, n1, n2, n3):
        return greens_func_p(vp, w, xi, yi, zi, xs, ys, zs)

    if geom.self_scheme == "grid":
        def self_value(i):
            return int_self_B(vp, w, faces2, i, geom.wis, geom.refs,
                              geom.ridfac)
    else:
        def self_value(i):
            return int_self_B_polar(vp, w, faces2, i, geom.polar_weak[i])

    return _assemble_scalar(faces1, faces2, w, kernel, self_value, geom,
                            k_abs=abs(w / vp))


def cal_traction(faces, w, lamda, mu, rho, Q, nint=NINT, nxi=NXI_SELF,
                 ridfac=RIDFAC, qp_fac=QP_FAC_LEGACY, geom=None,
                 self_scheme="grid", disp_ref_hz=0.0):
    """Full traction matrix TRAC (3N x 3N), port of cal_traction_tri_vec.m
    — identical to the same-surface branch of cal_T_st.

    Unknown ordering: [u_x(1..N); u_y(1..N); u_z(1..N)].
    """
    return cal_T_st(faces, faces, w, lamda, mu, rho, Q, nint, nxi,
                    ridfac, qp_fac, geom=geom, self_scheme=self_scheme,
                    disp_ref_hz=disp_ref_hz)


def smat_func(faces):
    """Normal-projection matrix Smat (Smat_func.m), (N, 3N)."""
    n = faces.n
    S = np.zeros((n, 3 * n))
    idx = np.arange(n)
    S[idx, idx] = faces.nvec[:, 0]
    S[idx, idx + n] = faces.nvec[:, 1]
    S[idx, idx + 2 * n] = faces.nvec[:, 2]
    return S
