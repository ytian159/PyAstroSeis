"""Mesh generation: quasi-uniform sphere sampling, spherical subdivision,
and spherical-harmonic topography.

Ports of: RandSampleSphere.m, ParticleSampleSphere.m (Anton Semechko),
TriQuad.m, SubdivideSphericalMesh.m, ylm1.m/ylms2.m (MATLAB
legendre(...,'norm') convention), get_phobos_topo.m (Willner et al. 2014
coefficients, TABLEA1.DAT), smooth_topo_rand.m, gen_mesh_ph_topo.m,
gen_mesh_topo_layers.m, gen_layer.m.

Notes on fidelity:
  - The particle optimizer is stochastic (random initialization) and its
    Gauss-Seidel line search takes float-sensitive branches, so meshes
    are not bit-reproducible against MATLAB; they are validated
    geometrically (closed surface, element-size statistics, energy).
    Pass `rng` (numpy Generator) for reproducibility within Python.
  - smooth_topo_rand.m OVERWRITES the analysis coefficients with
    rand() values from MATLAB's rng(4) stream; those values are shipped
    verbatim in data/coefrand_rng4.txt (dumped once by
    tests/dump_oracle2.m) so the "Mars-like" topography is reproduced
    exactly without MATLAB.
  - gen_mesh_ph_topo.m hardcodes r0 = 10.993 km (Phobos) and the
    TABLEA1 l=0 term adds another mean radius, so the body radius is
    ~2x10.993 km regardless of the R parameter. Replicated verbatim;
    see `radius_mode="parameter"` for the sane behavior.
"""

import os

import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import gammaln, lpmv

from .mesh import faces_from_vertices

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
R0_PHOBOS = 10.993e3

_EPS = np.finfo(float).eps


# ---------------------------------------------------------------------------
# sphere sampling and subdivision
# ---------------------------------------------------------------------------

def rand_sample_sphere(n=200, spl="uniform", rng=None):
    """Uniform or stratified random sampling of the unit sphere
    (RandSampleSphere.m)."""
    rng = np.random.default_rng(rng)
    n = int(round(n))
    if n < 3:
        spl = "uniform"
    if spl == "stratified":
        m = int(np.ceil(np.sqrt(n)))
        ds = 2.0 / m
        c = np.arange(-1 + ds / 2, 1, ds)
        Xc, Yc = np.meshgrid(c, c)
        x = ds * (rng.random(m * m) - 0.5) + Xc.ravel(order="F")
        y = ds * (rng.random(m * m) - 0.5) + Yc.ravel(order="F")
        excess = m * m - n
        if excess > 0:
            idx = rng.permutation(m * m)[:excess]
            x = np.delete(x, idx)
            y = np.delete(y, idx)
        lon = (x + 1) * np.pi
        z = y
    elif spl == "uniform":
        z = 2 * rng.random(n) - 1
        lon = 2 * np.pi * rng.random(n)
    else:
        raise ValueError("spl must be 'uniform' or 'stratified'")
    lat = np.arccos(z)
    return np.column_stack([np.cos(lon) * np.sin(lat),
                            np.sin(lon) * np.sin(lat), z])


def _hull_faces(V):
    """Outward-oriented triangulation of points on a sphere
    (fliplr(convhulln(V)) in MATLAB; orientation enforced explicitly).
    Returns 1-based faces."""
    hull = ConvexHull(V)
    F = hull.simplices.copy()
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    nrm = np.cross(B - A, C - A)
    cent = (A + B + C) / 3.0
    flip = np.einsum("ij,ij->i", nrm, cent) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return F + 1


def particle_sample_sphere(N=None, Vo=None, s=1.0, Etol=1e-5, Nitr=1000,
                           rng=None, verbose=False):
    """Quasi-uniform sphere sampling by Riesz s-energy minimization
    (ParticleSampleSphere.m, Gauss-Seidel scheme ported faithfully).

    Returns (V (N,3), Tri 1-based (M,3), Ue_i, Ue)."""
    rng = np.random.default_rng(rng)
    if Vo is not None:
        V = np.asarray(Vo, dtype=float).copy()
        V /= np.linalg.norm(V, axis=1)[:, None]
    else:
        V = rand_sample_sphere(200 if N is None else N, rng=rng)
    n = V.shape[0]

    DOT = np.clip(V @ V.T, -1.0, 1.0)
    GD = np.arccos(DOT)
    np.fill_diagonal(GD, np.inf)
    Ue_ij = 1.0 / (GD ** s + _EPS)
    Ue_i = Ue_ij.sum(axis=1)
    Ue = [Ue_i.sum()]

    a = np.ones(n)
    a_min, a_max = 1e-14, 0.1
    dE = np.inf
    it = 0
    while it < Nitr and dE > Etol:
        it += 1
        idx_sort = np.argsort(-Ue_i, kind="stable")
        for j in idx_sort:
            mask = np.ones(n, dtype=bool)
            mask[j] = False
            DOTj = DOT[mask, j]
            GDj = GD[mask, j]
            dVj = (s / (np.sqrt(1 - DOTj ** 2) + _EPS))[:, None] * V[mask]
            dVj = dVj / (GDj ** (s + 1) + _EPS)[:, None]
            if it < 5:
                ok = (dVj ** 2).sum(axis=1) <= 1e8
                dVj = dVj[ok]
                if dVj.shape[0] == 0:
                    V = V + rng.standard_normal(V.shape) / 1e5
                    V /= np.linalg.norm(V, axis=1)[:, None]
                    DOT = np.clip(V @ V.T, -1.0, 1.0)
                    GD = np.arccos(DOT)
                    break
            dVj = dVj.sum(axis=0)
            dVj_t = dVj - (dVj @ V[j]) * V[j]

            m = 0
            Uj_old = Ue_ij[j].sum()
            while True:
                Vj_new = V[j] - a[j] * dVj_t
                Vj_new = Vj_new / np.linalg.norm(Vj_new)
                DOTj = np.clip(V @ Vj_new, -1.0, 1.0)
                GDj = np.arccos(DOTj)
                GDj[j] = np.inf
                Ue_ij_j = 1.0 / (GDj ** s + _EPS)
                if Ue_ij_j.sum() <= Uj_old:
                    V[j] = Vj_new
                    DOT[j, :] = DOTj
                    DOT[:, j] = DOTj
                    GD[j, :] = GDj
                    GD[:, j] = GDj
                    Ue_ij[j, :] = Ue_ij_j
                    Ue_ij[:, j] = Ue_ij_j
                    if m == 1:
                        a[j] = min(a[j] * 2.5, a_max)
                    break
                if a[j] > a_min:
                    a[j] = max(a[j] / 2.5, a_min)
                else:
                    break
        Ue_i = Ue_ij.sum(axis=1)
        Ue.append(Ue_i.sum())
        if verbose and it % 50 == 0:
            print(f"  particle_sample_sphere iter {it}: E={Ue[-1]:.6f}")
        if it >= 10:
            dE = (Ue[it - 2] - Ue[it]) / 10.0
        if it % 20 == 0:
            a[:] = a_max

    Tri = _hull_faces(V)
    return V, Tri, Ue_i, np.array(Ue)


def tri_quad(F, V):
    """Triangular quadrisection (TriQuad.m). F is 1-based (M,3).
    Returns (F_new 1-based (4M,3), V_new)."""
    F0 = np.asarray(F, dtype=int) - 1
    X = np.asarray(V, dtype=float)
    V1 = (X[F0[:, 0]] + X[F0[:, 1]]) / 2.0
    V2 = (X[F0[:, 1]] + X[F0[:, 2]]) / 2.0
    V3 = (X[F0[:, 2]] + X[F0[:, 0]]) / 2.0
    Vnew = np.vstack([V1, V2, V3])

    # unique rows, 'stable' (first-occurrence order), exact float match
    uniq, first_idx, inv = np.unique(Vnew, axis=0, return_index=True,
                                     return_inverse=True)
    order = np.argsort(first_idx, kind="stable")
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    inv_stable = rank[inv]
    V_stable = uniq[order]

    nx = X.shape[0]
    nt = F0.shape[0]
    i1 = nx + inv_stable[:nt]
    i2 = nx + inv_stable[nt:2 * nt]
    i3 = nx + inv_stable[2 * nt:]

    T1 = np.column_stack([F0[:, 0], i1, i3])
    T2 = np.column_stack([F0[:, 1], i2, i1])
    T3 = np.column_stack([F0[:, 2], i3, i2])
    T4 = np.column_stack([i1, i2, i3])
    Fnew = np.stack([T1, T2, T3, T4], axis=1).reshape(-1, 3)
    return Fnew + 1, np.vstack([X, V_stable])


def subdivide_spherical_mesh(F, V, k=1, opt=True):
    """k rounds of spherical quadrisection (SubdivideSphericalMesh.m)."""
    F = np.asarray(F, dtype=int)
    V = np.asarray(V, dtype=float)
    if k < 1:
        return F, V / np.linalg.norm(V, axis=1)[:, None]
    for i in range(k):
        F, V = tri_quad(F, V)
        if opt or i == k - 1:
            V = V / np.linalg.norm(V, axis=1)[:, None]
    return F, V


# ---------------------------------------------------------------------------
# spherical harmonics (MATLAB legendre(...,'norm') convention)
# ---------------------------------------------------------------------------

def _legendre_norm(l, m, x):
    """MATLAB legendre(l, x, 'norm') row m:
    N_l^m(x) = (-1)^m sqrt((l+1/2)(l-m)!/(l+m)!) P_l^m(x),
    with P_l^m the associated Legendre function including the
    Condon-Shortley phase (same as scipy.special.lpmv)."""
    fac = np.exp(0.5 * (np.log(l + 0.5) + gammaln(l - m + 1)
                        - gammaln(l + m + 1)))
    return (-1.0) ** m * fac * lpmv(m, l, x)


def ylm1(L, m, theta, phi):
    """Fully normalized spherical harmonic, elementwise over paired
    (theta, phi) arrays (ylm1.m)."""
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    mabs = abs(m)
    if mabs > L:
        raise ValueError("|m| > L")
    y = _legendre_norm(L, mabs, np.cos(theta))
    y = (-1.0) ** mabs * y / np.sqrt(2 * np.pi)
    if m < 0:
        y = (-1.0) ** mabs * y
    return y * np.exp(1j * m * phi)


def ylms2(L, m, theta, phi):
    """Fully normalized spherical harmonic on the outer-product grid
    theta x phi (ylms2.m). Returns (ntheta, nphi) complex."""
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    mabs = abs(m)
    if mabs > L:
        raise ValueError("|m| > L")
    y = _legendre_norm(L, mabs, np.cos(theta))
    y = (-1.0) ** mabs * y / np.sqrt(2 * np.pi)
    if m < 0:
        y = (-1.0) ** mabs * y
    return np.outer(y, np.exp(1j * m * phi))


# ---------------------------------------------------------------------------
# topography models
# ---------------------------------------------------------------------------

_TABLEA1 = None


def get_phobos_topo(theta, phi):
    """Phobos shape from spherical-harmonic coefficients (Willner et al.,
    PSS 2014; TABLEA1.DAT). Port of get_phobos_topo.m. NOTE: includes
    the l=0 mean-radius term (~10.99 km), as in the original."""
    global _TABLEA1
    if _TABLEA1 is None:
        _TABLEA1 = np.loadtxt(os.path.join(DATA_DIR, "TABLEA1.DAT"))
    theta = np.asarray(theta, dtype=float).ravel()
    phi = np.asarray(phi, dtype=float).ravel()
    vv = np.zeros(theta.shape)
    for ll, mm, amn, _, bmn, _2 in zip(_TABLEA1[:, 0].astype(int),
                                       _TABLEA1[:, 1].astype(int),
                                       _TABLEA1[:, 2], _TABLEA1[:, 3],
                                       _TABLEA1[:, 4], _TABLEA1[:, 5]):
        fac = -1.0 if mm % 2 == 1 else 1.0
        ctopo = fac * ylm1(ll, mm, theta, phi)
        vv = vv + amn * np.real(ctopo) + bmn * np.imag(ctopo)
    return vv


_COEFRAND = None


def _coefrand_rng4(n):
    """First n values of MATLAB's rand stream after rng(4), dumped once
    by tests/dump_oracle2.m (see module docstring)."""
    global _COEFRAND
    if _COEFRAND is None:
        _COEFRAND = np.loadtxt(os.path.join(DATA_DIR, "coefrand_rng4.txt"))
    if n > _COEFRAND.size:
        raise ValueError(f"only {_COEFRAND.size} rng(4) values shipped")
    return _COEFRAND[:n]


def smooth_topo_rand(llon, llat, hh, lmax):
    """Pseudo-random smooth topography (smooth_topo_rand.m). The MATLAB
    original computes analysis coefficients from `hh` and then discards
    them, replacing them with rng(4) random values — only the synthesis
    is ported; `hh` is accepted for signature compatibility."""
    phi = np.asarray(llon)[0, :] * np.pi / 180.0
    theta = np.asarray(llat)[:, 0] * np.pi / 180.0
    nbasis = (lmax + 1) * (lmax + 2) // 2
    coefrand = _coefrand_rng4(nbasis)
    maxh = 20.0
    topos = np.zeros(theta.size * phi.size, dtype=complex)
    k = 0
    for ell in range(lmax + 1):
        for m in range(ell + 1):
            coef = (coefrand[k] - 0.5) * maxh
            y = ylms2(ell, m, theta, phi)
            fac = 2.0 if m > 0 else 1.0
            topos = topos + coef * y.ravel(order="F") * fac
            k += 1
    return np.real(topos).reshape(np.asarray(llat).shape, order="F")


def _interp2_nearest(llon, llat, Z, phi, theta, fill=0.0):
    """interp2(...,'nearest',fill) on a plaid meshgrid (either grid
    direction)."""
    gx = np.asarray(llon)[0, :]
    gy = np.asarray(llat)[:, 0]
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    ix = np.abs(phi[:, None] - gx[None, :]).argmin(axis=1)
    iy = np.abs(theta[:, None] - gy[None, :]).argmin(axis=1)
    out = np.asarray(Z)[iy, ix]
    bad = ((phi < gx.min()) | (phi > gx.max())
           | (theta < gy.min()) | (theta > gy.max()))
    return np.where(bad, fill, out)


# ---------------------------------------------------------------------------
# mesh builders
# ---------------------------------------------------------------------------

def _mesh_extras(faces, V, Tri, Rm):
    """The auxiliary arrays the MATLAB mesh files carry."""
    ds = faces.area.copy()
    xs0, ys0, zs0 = faces.ic[:, 0], faces.ic[:, 1], faces.ic[:, 2]
    phis = np.arctan2(ys0, xs0)
    thetas = np.pi / 2 - np.arctan2(zs0, np.sqrt(xs0 ** 2 + ys0 ** 2))
    height = np.linalg.norm(faces.ic, axis=1) - Rm
    return dict(numface=faces.n, ds=ds, xs0=xs0, ys0=ys0, zs0=zs0,
                thetas=thetas, phis=phis, height=height, V=V, Tri=Tri)


def gen_mesh_ph_topo(Rm, n, nfold, rng=None, radius_mode="matlab",
                     verbose=False):
    """Phobos-topography mesh (gen_mesh_ph_topo.m).

    radius_mode="matlab" replicates the original: vertex radius is
    r0=10.993 km + SH height (whose l=0 term is another ~10.99 km), so
    Rm only labels the mesh. radius_mode="parameter" scales the shape so
    its mean radius equals Rm (sane behavior for new bodies).

    Returns (faces, extras dict)."""
    V, Tri, _, _ = particle_sample_sphere(N=n, rng=rng, verbose=verbose)
    Tri, V = subdivide_spherical_mesh(Tri, V, 1)
    V = V * Rm
    r = np.linalg.norm(V, axis=1)
    r_xy = np.linalg.norm(V[:, :2], axis=1)
    phi = np.arctan2(V[:, 1], V[:, 0])
    theta = np.arctan2(r_xy, V[:, 2])
    hh = np.real(get_phobos_topo(theta, phi) * nfold)
    rr = R0_PHOBOS + hh
    if radius_mode == "parameter":
        rr = rr * (Rm / rr.mean())
    elif radius_mode != "matlab":
        raise ValueError("radius_mode must be 'matlab' or 'parameter'")
    V = np.column_stack([rr * np.sin(theta) * np.cos(phi),
                         rr * np.sin(theta) * np.sin(phi),
                         rr * np.cos(theta)])
    faces = faces_from_vertices(V, Tri)
    return faces, _mesh_extras(faces, V, Tri, Rm)


_MARSTOPO = None


def gen_mesh_topo_layers(Rm, n, nfold, rng=None, lmax=8, verbose=False):
    """Sphere of radius Rm with pseudo-random Mars-like topography scaled
    by nfold (gen_mesh_topo_layers.m). nfold=0 gives an exact sphere."""
    global _MARSTOPO
    import scipy.io as sio
    if _MARSTOPO is None:
        _MARSTOPO = sio.loadmat(os.path.join(DATA_DIR, "marstopo.mat"))
    llon = _MARSTOPO["llon"]
    llat = 90.0 - _MARSTOPO["llat"]
    aa = _MARSTOPO["aa"]

    V, Tri, _, _ = particle_sample_sphere(N=n, rng=rng, verbose=verbose)
    Tri, V = subdivide_spherical_mesh(Tri, V, 1)
    V = V * Rm

    topos = smooth_topo_rand(llon, llat, aa, lmax)

    r_xy = np.linalg.norm(V[:, :2], axis=1)
    phi = np.arctan2(V[:, 1], V[:, 0]) * 180.0 / np.pi
    phi = np.where(phi < 0, phi + 360.0, phi)
    theta = np.arctan2(r_xy, V[:, 2]) * 180.0 / np.pi
    hh = _interp2_nearest(llon, llat, topos * 20.0, phi, theta, 0.0) * 1e3
    hh = hh * nfold
    rr = Rm + hh
    tr = np.deg2rad(theta)
    pr = np.deg2rad(phi)
    V = np.column_stack([rr * np.sin(tr) * np.cos(pr),
                         rr * np.sin(tr) * np.sin(pr),
                         rr * np.cos(tr)])
    faces = faces_from_vertices(V, Tri)
    return faces, _mesh_extras(faces, V, Tri, Rm)


def gen_layer(rpl, nmesh, nfold, rng=None, verbose=False):
    """Multi-layer spherical meshes (gen_layer.m). Returns a list of
    (faces, extras) tuples, innermost first."""
    out = []
    for r, n, f in zip(rpl, nmesh, nfold):
        out.append(gen_mesh_topo_layers(r, n, f, rng=rng, verbose=verbose))
    return out
