"""Moment-fitted RHS (falsification rig, docs/moment_fitted_rhs.md).

Corrects the low-(l,m) vector-spherical-harmonic moments of an
incident-trace RHS block on a spherical boundary to their exact
continuum values, with a minimal area-weighted-norm update:

    dB = Psi (Psi^H W Psi)^-1 (E - Psi^H W b)

Both the discrete moment (Psi^H W b) and the exact moment E (graded
source-frame quadrature of the analytic incident field) are evaluated
with the SAME basis code, so normalization / Condon-Shortley /
conjugation conventions cancel identically in the deficit; the only
structural requirements are that the basis spans the low-l vector
fields (gate: dense-grid Gram ~ I) and that the exact quadrature
converges (gate: two-tier agreement, reported per call).

Layout convention: all field vectors are component-major, matching
source.u0eM — [x(all pts), y(all pts), z(all pts)].
"""

import numpy as np


def norm_legendre(lmax, x, deriv=False):
    """Fully-normalized associated Legendre P_lm(x) with the
    Condon-Shortley phase, normalized so Y_lm = P_lm(cos th) e^{i m ph}
    is orthonormal on the sphere. Returns (lmax+1, lmax+1, N) with
    [l, m] valid for m <= l; with deriv=True also dP_lm/dtheta
    (same shape), from sin(th) dP/dth = l x P_lm - e_lm P_{l-1,m}.
    Near-pole points (sin th < 1e-12) get zeroed derivatives (the
    quadrature grids and mesh incenters never sit there)."""
    x = np.asarray(x, dtype=float)
    s = np.sqrt(np.maximum(0.0, 1.0 - x * x))
    P = np.zeros((lmax + 1, lmax + 1) + x.shape)
    P[0, 0] = 1.0 / np.sqrt(4.0 * np.pi)
    for m in range(1, lmax + 1):
        P[m, m] = -np.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s * P[m - 1, m - 1]
    for m in range(0, lmax):
        P[m + 1, m] = np.sqrt(2.0 * m + 3.0) * x * P[m, m]
    for m in range(0, lmax + 1):
        for l in range(m + 2, lmax + 1):
            a = np.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = np.sqrt(((2.0 * l + 1.0) * ((l - 1.0) ** 2 - m * m))
                        / ((2.0 * l - 3.0) * (l * l - m * m)))
            P[l, m] = a * x * P[l - 1, m] - b * P[l - 2, m]
    if not deriv:
        return P
    ok = s > 1e-12
    si = np.where(ok, s, 1.0)
    dP = np.zeros_like(P)
    for m in range(0, lmax + 1):
        for l in range(m, lmax + 1):
            num = l * x * P[l, m]
            if l > m:
                e = np.sqrt((l * l - m * m) * (2.0 * l + 1.0)
                            / (2.0 * l - 1.0))
                num = num - e * P[l - 1, m]
            dP[l, m] = np.where(ok, num / si, 0.0)
    return P, dP


def basis_labels(lmax):
    """Column labels [(family, l, m)] matching vec_sph_basis order."""
    lab = [("P", l, m) for l in range(lmax + 1) for m in range(-l, l + 1)]
    for fam in ("B", "C"):
        lab += [(fam, l, m) for l in range(1, lmax + 1)
                for m in range(-l, l + 1)]
    return lab


def vec_sph_basis(lmax, pts):
    """Vector spherical harmonics at the unit directions of pts (N,3):
    P_lm = Y_lm rhat, B_lm = grad_S Y_lm / sqrt(l(l+1)),
    C_lm = rhat x B_lm. Returns Psi (3N, K) complex, component-major
    rows, columns ordered as basis_labels(lmax)."""
    pts = np.asarray(pts, dtype=float)
    u = pts / np.linalg.norm(pts, axis=1)[:, None]
    ct = np.clip(u[:, 2], -1.0, 1.0)
    st = np.hypot(u[:, 0], u[:, 1])
    ph = np.arctan2(u[:, 1], u[:, 0])
    ok = st > 1e-12
    si = np.where(ok, st, 1.0)
    cp, sp = np.cos(ph), np.sin(ph)
    n = len(pts)
    rhat = u.T                                     # (3, N)
    that = np.stack([ct * cp, ct * sp, -st])       # theta-hat
    phat = np.stack([-sp, cp, np.zeros(n)])        # phi-hat

    P, dP = norm_legendre(lmax, ct, deriv=True)
    eim = np.exp(1j * np.outer(np.arange(lmax + 1), ph))  # (m, N)

    K = 3 * (lmax + 1) ** 2 - 2
    Psi = np.empty((3 * n, K), dtype=complex)

    def put(col, vec3):                            # vec3 (3, N)
        Psi[0:n, col] = vec3[0]
        Psi[n:2 * n, col] = vec3[1]
        Psi[2 * n:3 * n, col] = vec3[2]

    def scalars(l, m):
        """Y, dY/dth, i m Y / sin th at all points (m may be < 0)."""
        am = abs(m)
        Y = P[l, am] * eim[am]
        t1 = dP[l, am] * eim[am]
        t2 = 1j * am * np.where(ok, P[l, am] / si, 0.0) * eim[am]
        if m < 0:
            sgn = (-1.0) ** am
            Y, t1, t2 = sgn * Y.conj(), sgn * t1.conj(), sgn * t2.conj()
        return Y, t1, t2

    col = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y, _, _ = scalars(l, m)
            put(col, rhat * Y)
            col += 1
    for fam in ("B", "C"):
        for l in range(1, lmax + 1):
            f = 1.0 / np.sqrt(l * (l + 1.0))
            for m in range(-l, l + 1):
                _, t1, t2 = scalars(l, m)
                if fam == "B":
                    put(col, f * (t1 * that + t2 * phat))
                else:
                    put(col, f * (t1 * phat - t2 * that))
                col += 1
    return Psi


def source_frame_grid(R, src_xyz, first_panel, n_gl=16, n_phi=96):
    """Graded product quadrature on the sphere of radius R: colatitude
    measured from the sub-source axis in geometrically-doubling
    Gauss-Legendre panels (edges 0, a, 2a, ... clipped at pi, a =
    first_panel radians), uniform trapezoid in azimuth. Returns
    pts (Q,3) in the EARTH frame and weights (Q,) including R^2
    (sum = 4 pi R^2)."""
    e3 = np.asarray(src_xyz, dtype=float)
    e3 = e3 / np.linalg.norm(e3)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(e3 @ ref) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(e3, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)

    edges = [0.0]
    a = float(first_panel)
    while a < np.pi:
        edges.append(a)
        # doubling growth, capped at 0.5 rad per panel so degree-~20
        # harmonics stay over-resolved by the per-panel GL rule
        a += min(a, 0.5)
    edges.append(np.pi)
    xg, wg = np.polynomial.legendre.leggauss(n_gl)
    th, wth = [], []
    for t0, t1 in zip(edges[:-1], edges[1:]):
        th.append(t0 + (t1 - t0) * 0.5 * (xg + 1.0))
        wth.append(0.5 * (t1 - t0) * wg)
    th = np.concatenate(th)
    wth = np.concatenate(wth)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi

    TH, PH = np.meshgrid(th, phi, indexing="ij")
    WT = (wth * np.sin(th))[:, None] * np.full(n_phi, 2.0 * np.pi / n_phi)
    pts = R * (np.sin(TH)[..., None] * np.cos(PH)[..., None] * e1
               + np.sin(TH)[..., None] * np.sin(PH)[..., None] * e2
               + np.cos(TH)[..., None] * e3)
    return pts.reshape(-1, 3), (R * R * WT).ravel()


def _mh(Psi, w):
    """(K, 3Q) moment operator: E = _mh(Psi, w) @ field."""
    return (np.conj(Psi) * np.tile(w, 3)[:, None]).T


class MomentFit:
    """Per-interface moment-fit context (frequency-independent parts
    precomputed once; fork-shared read-only by the workers)."""

    def __init__(self, lmax, ic, area, R, src_xyz, depth_m,
                 n_gl=(16, 24), n_phi=(96, 128)):
        self.lmax = int(lmax)
        self.labels = basis_labels(self.lmax)
        a = max(float(depth_m), 1.0) / float(R)
        self.pts1, w1 = source_frame_grid(R, src_xyz, a, n_gl[0], n_phi[0])
        self.pts2, w2 = source_frame_grid(R, src_xyz, 0.5 * a,
                                          n_gl[1], n_phi[1])
        self.MH1 = _mh(vec_sph_basis(self.lmax, self.pts1), w1)
        self.MH2 = _mh(vec_sph_basis(self.lmax, self.pts2), w2)
        self.Psi = vec_sph_basis(self.lmax, np.asarray(ic, dtype=float))
        self.MHi = _mh(self.Psi, np.asarray(area, dtype=float))
        gram = self.MHi @ self.Psi
        ev = np.linalg.eigvalsh(gram)
        self.gram_cond = float(ev[-1] / ev[0])
        self.gram_inv = np.linalg.inv(gram)

    def exact_moments(self, field_fn, tier=2):
        """E_k = R^2 int conj(psi_k) . field dOmega on quadrature tier
        1 (base) or 2 (refined). field_fn(pts (Q,3)) -> (3Q,)
        component-major."""
        MH, pts = (self.MH1, self.pts1) if tier == 1 else \
                  (self.MH2, self.pts2)
        return MH @ np.asarray(field_fn(pts), dtype=complex).ravel()

    def correct(self, b, field_fn):
        """Minimal-norm moment correction of the RHS block b (3n,).
        Returns (b + db, diag) with diag: dbnorm = |db|/|b|,
        griddiff = max_k |E1-E2| / max_k |E2| (quadrature convergence),
        deficit = max_k |E-m[b]| / max_k |E| (how wrong sampling was),
        E, delta (per-moment, for channel diagnostics)."""
        b = np.asarray(b, dtype=complex).ravel()
        E1 = self.exact_moments(field_fn, tier=1)
        E2 = self.exact_moments(field_fn, tier=2)
        md = self.MHi @ b
        delta = E2 - md
        db = self.Psi @ (self.gram_inv @ delta)
        sc = np.max(np.abs(E2))
        diag = dict(
            dbnorm=float(np.linalg.norm(db) / np.linalg.norm(b)),
            griddiff=float(np.max(np.abs(E1 - E2)) / sc),
            deficit=float(np.max(np.abs(delta)) / sc),
            E=E2, delta=delta)
        return b + db, diag
