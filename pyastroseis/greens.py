"""Elastodynamic Green's functions (frequency domain, full space).

Faithful ports of:
  - green_traction_tensor.m  (traction Green tensor, Mathematica-generated)
  - greens_functionQ.m       (displacement Green tensor with Q)
  - ricker.m                 (source wavelet, used in post-processing)

The nine Tij expressions are transcribed literally from the MATLAB code;
they are verified element-by-element against a MATLAB oracle dump in
tests/test_oracle.py. Do not "simplify" them here — that is Phase 2 work
(common-subexpression / radial-function form) and must be gated on the
oracle test.
"""

import numpy as np

PI = np.pi


def green_traction_tensor(vp, vs, rho, w, x1, x2, x3, xs, ys, zs, n1, n2, n3):
    """Traction Green tensor Tij — canonical radial-function form.

    Algebraically identical to the Mathematica-generated expressions in
    green_traction_tensor.m (kept verbatim as
    green_traction_tensor_reference below); equivalence is enforced by
    tests/test_kernel_equiv.py and the MATLAB oracle tests. All nine
    components are

        Tij = 1/(4 pi r^4 rho w^2) * [ W*Xij - Q2s*mu*Y1ij - Q2p*Y2ij
                                       + Q3s*mu*Z1ij + Q3p*Z2ij ]

    with shared radial factors (kpr = kp*r, ksr = ks*r):
        W   = mu * (6(Ep-Es) - 6i(Ep*kpr - Es*ksr))
        Q2s = Es*ksr^2      Q2p = Ep*kpr^2
        Q3s = i*Es*ksr^3    Q3p = i*Ep*kpr^3
    and tensor combinations (gn = g.n, gg = gi*gj, a = gi*nj, b = gj*ni):
        X   = 5*gg*gn - a - b - dij*gn
        Y1  = 3a + 2b - 12*gg*gn + 3*dij*gn
        Y2  = lam*b - 2*mu*(a+b) + 12*mu*gg*gn - 2*mu*dij*gn
        Z1  = a - 2*gg*gn + dij*gn
        Z2  = lam*b + 2*mu*gg*gn

    This evaluates ~3x fewer array operations than the generated form.
    """
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    dx = x1 - xs
    dy = x2 - ys
    dz = x3 - zs
    r2 = dx * dx + dy * dy + dz * dz
    r = np.sqrt(r2)
    inv_r = 1.0 / r
    g1 = dx * inv_r
    g2 = dy * inv_r
    g3 = dz * inv_r
    kpr = (w / vp) * r
    ksr = (w / vs) * r
    Ep = np.exp(1j * kpr)
    Es = np.exp(1j * ksr)

    W = mu * (6 * (Ep - Es) - 6j * (Ep * kpr - Es * ksr))
    Q2sm = Es * ksr ** 2 * mu
    Q2p = Ep * kpr ** 2
    Q3sm = 1j * Es * ksr ** 3 * mu
    Q3p = 1j * Ep * kpr ** 3

    gn = g1 * n1 + g2 * n2 + g3 * n3
    pref = (0.25 / PI) / (rho * w ** 2) * inv_r ** 4

    g = (g1, g2, g3)
    n = (n1, n2, n3)
    out = []
    for i in range(3):
        for j in range(3):
            gg_gn = g[i] * g[j] * gn
            a = g[i] * n[j]
            b = g[j] * n[i]
            if i == j:
                X = 5 * gg_gn - a - b - gn
                Y1 = 3 * a + 2 * b - 12 * gg_gn + 3 * gn
                Y2 = lam * b - 2 * mu * (a + b) + 12 * mu * gg_gn - 2 * mu * gn
                Z1 = a - 2 * gg_gn + gn
            else:
                X = 5 * gg_gn - a - b
                Y1 = 3 * a + 2 * b - 12 * gg_gn
                Y2 = lam * b - 2 * mu * (a + b) + 12 * mu * gg_gn
                Z1 = a - 2 * gg_gn
            Z2 = lam * b + 2 * mu * gg_gn
            out.append(pref * (W * X - Q2sm * Y1 - Q2p * Y2
                               + Q3sm * Z1 + Q3p * Z2))
    return tuple(out)


def green_traction_tensor_reference(vp, vs, rho, w, x1, x2, x3,
                                    xs, ys, zs, n1, n2, n3):
    """Verbatim transcription of the Mathematica-generated MATLAB
    expressions (green_traction_tensor.m). Kept as the reference for
    tests/test_kernel_equiv.py; the production kernel above is the
    canonical CSE form.
    """
    mu = rho * vs * vs
    lam = rho * vp * vp - 2 * mu
    r = np.sqrt((x1 - xs) ** 2 + (x2 - ys) ** 2 + (x3 - zs) ** 2)
    g1 = (x1 - xs) / r
    g2 = (x2 - ys) / r
    g3 = (x3 - zs) / r
    kp = w / vp
    ks = w / vs
    Ep = np.exp(1j * kp * r)
    Es = np.exp(1j * ks * r)

    iww = 1.0 / (rho * w ** 2)
    c = (1.0 / 4.0) / PI * r ** (-4)

    T11 = c * (Ep * (2 * mu * (g2 * n2 + g3 * n3) * (-3 + kp * r * (3j + kp * r))
                     + g1 * n1 * (kp ** 2 * lam * r ** 2 * (-1 + 1j * kp * r)
                                  + 6 * mu * (-3 + kp * r * (3j + kp * r)))
                     + 2 * g1 ** 3 * mu * n1 * (15 + 1j * kp * r * (-15 + kp * r * (6j + kp * r)))
                     + 2 * g1 ** 2 * mu * (g2 * n2 + g3 * n3) * (15 + 1j * kp * r * (-15 + kp * r * (6j + kp * r))))
               + Es * mu * (2 * g1 ** 3 * n1 * (-15 + ks * r * (15j + ks * r * (6 - 1j * ks * r)))
                            + 2 * g1 ** 2 * (g2 * n2 + g3 * n3) * (-15 + ks * r * (15j + ks * r * (6 - 1j * ks * r)))
                            + (g2 * n2 + g3 * n3) * (6 + 1j * ks * r * (-6 + ks * r * (3j + ks * r)))
                            + 2 * g1 * n1 * (9 + 1j * ks * r * (-9 + ks * r * (4j + ks * r))))) * iww

    T21 = c * (6 * (Ep - Es) * mu * (-g1 * n2 + 5 * g1 * g2 ** 2 * n2
                                     + g2 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g3 * n3))
               + (-6j) * (Ep * kp - Es * ks) * mu * (-g1 * n2 + 5 * g1 * g2 ** 2 * n2
                                                     + g2 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g3 * n3)) * r
               - (Es * ks ** 2 * mu * (2 * g1 * n2 - 12 * g1 * g2 ** 2 * n2
                                       + 3 * g2 * (n1 - 4 * g1 ** 2 * n1 - 4 * g1 * g3 * n3))
                  + Ep * kp ** 2 * (g1 * (lam - 2 * mu) * n2 + 12 * g1 * g2 ** 2 * mu * n2
                                    + 2 * g2 * mu * (-n1 + 6 * g1 ** 2 * n1 + 6 * g1 * g3 * n3))) * r ** 2
               + 1j * (Es * g2 * ks ** 3 * mu * (n1 - 2 * g1 ** 2 * n1 - 2 * g1 * (g2 * n2 + g3 * n3))
                       + Ep * g1 * kp ** 3 * (lam * n2 + 2 * g2 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T31 = c * (6 * (Ep - Es) * mu * (g3 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g2 * n2)
                                     + g1 * (-1 + 5 * g3 ** 2) * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * (g3 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g2 * n2)
                                                     + g1 * (-1 + 5 * g3 ** 2) * n3) * r
               - (Es * ks ** 2 * mu * (3 * g3 * (n1 - 4 * g1 ** 2 * n1 - 4 * g1 * g2 * n2)
                                       + 2 * g1 * (1 - 6 * g3 ** 2) * n3)
                  + Ep * kp ** 2 * (2 * g3 * mu * ((-1 + 6 * g1 ** 2) * n1 + 6 * g1 * g2 * n2)
                                    + g1 * (lam + 2 * (-1 + 6 * g3 ** 2) * mu) * n3)) * r ** 2
               + 1j * (Es * g3 * ks ** 3 * mu * (n1 - 2 * g1 ** 2 * n1 - 2 * g1 * (g2 * n2 + g3 * n3))
                       + Ep * g1 * kp ** 3 * (lam * n3 + 2 * g3 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T12 = c * (6 * (Ep - Es) * mu * (-g1 * n2 + 5 * g1 * g2 ** 2 * n2
                                     + g2 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g3 * n3))
               + (-6j) * (Ep * kp - Es * ks) * mu * (-g1 * n2 + 5 * g1 * g2 ** 2 * n2
                                                     + g2 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g3 * n3)) * r
               - (Ep * kp ** 2 * (g2 * (lam - 2 * mu + 12 * g1 ** 2 * mu) * n1
                                  - 2 * g1 * mu * n2 + 12 * g1 * g2 ** 2 * mu * n2
                                  + 12 * g1 * g2 * g3 * mu * n3)
                  + Es * ks ** 2 * mu * (3 * g1 * n2 - 12 * g1 * g2 ** 2 * n2
                                         + 2 * g2 * (n1 - 6 * g1 ** 2 * n1 - 6 * g1 * g3 * n3))) * r ** 2
               + 1j * (Es * g1 * ks ** 3 * mu * (n2 - 2 * g2 * (g1 * n1 + g2 * n2 + g3 * n3))
                       + Ep * g2 * kp ** 3 * (lam * n1 + 2 * g1 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T22 = c * (6 * (Ep - Es) * mu * (g1 * (-1 + 5 * g2 ** 2) * n1 - 3 * g2 * n2 + 5 * g2 ** 3 * n2
                                     - g3 * n3 + 5 * g2 ** 2 * g3 * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * (g1 * (-1 + 5 * g2 ** 2) * n1 - 3 * g2 * n2
                                                     + 5 * g2 ** 3 * n2 - g3 * n3 + 5 * g2 ** 2 * g3 * n3) * r
               - (Ep * kp ** 2 * (2 * g1 * (-1 + 6 * g2 ** 2) * mu * n1
                                  + g2 * (lam - 6 * mu + 12 * g2 ** 2 * mu) * n2
                                  + 2 * (-1 + 6 * g2 ** 2) * g3 * mu * n3)
                  + Es * ks ** 2 * mu * (3 * g1 * (1 - 4 * g2 ** 2) * n1 + 3 * g3 * n3
                                         - 4 * g2 * (-2 * n2 + 3 * g2 ** 2 * n2 + 3 * g2 * g3 * n3))) * r ** 2
               + 1j * (Ep * g2 * kp ** 3 * (lam * n2 + 2 * g2 * mu * (g1 * n1 + g2 * n2 + g3 * n3))
                       + Es * ks ** 3 * mu * (g1 * (n1 - 2 * g2 ** 2 * n1) + g3 * n3
                                              - 2 * g2 * ((-1 + g2 ** 2) * n2 + g2 * g3 * n3))) * r ** 3) * iww

    T32 = c * (6 * (Ep - Es) * mu * (5 * g1 * g2 * g3 * n1 + (-1 + 5 * g2 ** 2) * g3 * n2
                                     + g2 * (-1 + 5 * g3 ** 2) * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * (5 * g1 * g2 * g3 * n1 + (-1 + 5 * g2 ** 2) * g3 * n2
                                                     + g2 * (-1 + 5 * g3 ** 2) * n3) * r
               - (Es * ks ** 2 * mu * (3 * g3 * (n2 - 4 * g2 * (g1 * n1 + g2 * n2))
                                       + 2 * g2 * (1 - 6 * g3 ** 2) * n3)
                  + Ep * kp ** 2 * (2 * g3 * mu * (-n2 + 6 * g2 * (g1 * n1 + g2 * n2))
                                    + g2 * (lam + 2 * (-1 + 6 * g3 ** 2) * mu) * n3)) * r ** 2
               + 1j * (Es * g3 * ks ** 3 * mu * (n2 - 2 * g2 * (g1 * n1 + g2 * n2 + g3 * n3))
                       + Ep * g2 * kp ** 3 * (lam * n3 + 2 * g3 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T13 = c * (6 * (Ep - Es) * mu * (g3 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g2 * n2)
                                     + g1 * (-1 + 5 * g3 ** 2) * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * (g3 * ((-1 + 5 * g1 ** 2) * n1 + 5 * g1 * g2 * n2)
                                                     + g1 * (-1 + 5 * g3 ** 2) * n3) * r
               - (Es * ks ** 2 * mu * (2 * g3 * (n1 - 6 * g1 ** 2 * n1 - 6 * g1 * g2 * n2)
                                       + 3 * g1 * (1 - 4 * g3 ** 2) * n3)
                  + Ep * kp ** 2 * (g3 * (lam + 2 * (-1 + 6 * g1 ** 2) * mu) * n1
                                    + 12 * g1 * g2 * g3 * mu * n2
                                    + 2 * g1 * (-1 + 6 * g3 ** 2) * mu * n3)) * r ** 2
               + 1j * (Es * g1 * ks ** 3 * mu * (n3 - 2 * g3 * (g1 * n1 + g2 * n2 + g3 * n3))
                       + Ep * g3 * kp ** 3 * (lam * n1 + 2 * g1 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T23 = c * (6 * (Ep - Es) * mu * (5 * g1 * g2 * g3 * n1 + (-1 + 5 * g2 ** 2) * g3 * n2
                                     + g2 * (-1 + 5 * g3 ** 2) * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * (5 * g1 * g2 * g3 * n1 + (-1 + 5 * g2 ** 2) * g3 * n2
                                                     + g2 * (-1 + 5 * g3 ** 2) * n3) * r
               - (Es * ks ** 2 * mu * (2 * g3 * (n2 - 6 * g2 * (g1 * n1 + g2 * n2))
                                       + 3 * g2 * (1 - 4 * g3 ** 2) * n3)
                  + Ep * kp ** 2 * (12 * g1 * g2 * g3 * mu * n1
                                    + g3 * (lam - 2 * mu + 12 * g2 ** 2 * mu) * n2
                                    + 2 * g2 * (-1 + 6 * g3 ** 2) * mu * n3)) * r ** 2
               + 1j * (Es * g2 * ks ** 3 * mu * (n3 - 2 * g3 * (g1 * n1 + g2 * n2 + g3 * n3))
                       + Ep * g3 * kp ** 3 * (lam * n2 + 2 * g2 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    T33 = c * (6 * (Ep - Es) * mu * ((-1 + 5 * g3 ** 2) * (g1 * n1 + g2 * n2)
                                     + g3 * (-3 + 5 * g3 ** 2) * n3)
               + (-6j) * (Ep * kp - Es * ks) * mu * ((-1 + 5 * g3 ** 2) * (g1 * n1 + g2 * n2)
                                                     + g3 * (-3 + 5 * g3 ** 2) * n3) * r
               - ((2 * Ep * (-1 + 6 * g3 ** 2) * kp ** 2 + 3 * Es * (1 - 4 * g3 ** 2) * ks ** 2)
                  * mu * (g1 * n1 + g2 * n2)
                  + g3 * (4 * Es * (2 - 3 * g3 ** 2) * ks ** 2 * mu
                          + Ep * kp ** 2 * (lam + 6 * (-1 + 2 * g3 ** 2) * mu)) * n3) * r ** 2
               + 1j * (Es * ks ** 3 * mu * (-(-1 + 2 * g3 ** 2) * (g1 * n1 + g2 * n2)
                                            - 2 * g3 * (-1 + g3 ** 2) * n3)
                       + Ep * g3 * kp ** 3 * (lam * n3 + 2 * g3 * mu * (g1 * n1 + g2 * n2 + g3 * n3))) * r ** 3) * iww

    return T11, T12, T13, T21, T22, T23, T31, T32, T33


def greens_functionQ(w, rho, mu, lamda, x0, x, y, z, Q, qp_fac=9.0 / 4.0):
    """Displacement Green tensor with attenuation (greens_functionQ.m),
    vectorized over field points.

    x0 is the (3,) source position; x, y, z are (N,) field points.
    Returns G with shape (3, 3, N): G[i,j] = displacement component i
    for a unit force in direction j.

    qp_fac: Qp = qp_fac * Q. Default 9/4 is the original MATLAB value
    (pure-shear-attenuation relation hardcoded for a Poisson solid).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    Qs = Q
    Qp = Qs * qp_fac
    kp = w / np.sqrt((2 * mu + lamda) / rho) * (1 + 1j / (2 * Qp))
    ks = w / np.sqrt(mu / rho) * (1 + 1j / (2 * Qs))

    r = np.sqrt((x0[0] - x) ** 2 + (x0[1] - y) ** 2 + (x0[2] - z) ** 2)
    xr = np.stack([x - x0[0], y - x0[1], z - x0[2]])   # (3,N)
    ts_prar = np.exp(1j * ks * r) / (4 * PI * r ** 3)
    tp_prar = np.exp(1j * kp * r) / (4 * PI * r ** 3)
    gamma = xr / r
    green_coef = 1.0 / (rho * w ** 2)
    gs = ts_prar * r ** 2

    G = np.empty((3, 3) + r.shape, dtype=complex)
    for i in range(3):
        for j in range(3):
            sig = 1.0 if i == j else 0.0
            gs2 = ts_prar * (((3 - ks ** 2 * r ** 2) * gamma[i] * gamma[j] - sig)
                             + 1j * ks * r * (sig - 3 * gamma[i] * gamma[j]))
            gp2 = tp_prar * (((3 - kp ** 2 * r ** 2) * gamma[i] * gamma[j] - sig)
                             + 1j * kp * r * (sig - 3 * gamma[i] * gamma[j]))
            G[i, j] = green_coef * (gs * ks ** 2 * sig + gs2 - gp2)
    return G


def ricker(f0, ts, dt, nt):
    """Ricker wavelet, port of ricker.m. Returns (w, t)."""
    a = (PI * PI) * (f0 * f0)
    t = np.arange(nt) * dt
    tp = t - ts
    arg = a * tp ** 2
    w = (1.0 - 2 * a * tp * tp) * np.exp(-arg)
    return w, t


def bmat_func(vp, vs, rho, w, x, y, z, x0, y0, z0):
    """Displacement Green tensor Bij (Bmat_func.m), broadcastable.
    Returns nine arrays (B is symmetric: B21=B12 etc., as in MATLAB)."""
    R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    g1 = (x - x0) / R
    g2 = (y - y0) / R
    g3 = (z - z0) / R
    kp = w / vp
    ks = w / vs
    Gs = np.exp(1j * ks * R) / (4 * PI * R)
    Gp = np.exp(1j * kp * R) / (4 * PI * R)
    iww = 1.0 / (rho * w ** 2)

    def diag(gsq):
        return R ** (-2) * (Gp * (1 - 1j * kp * R + gsq * (-3 + 3j * kp * R + kp ** 2 * R ** 2))
                            + Gs * (-1 + 1j * ks * R + ks ** 2 * R ** 2
                                    - gsq * (-3 + 3j * ks * R + ks ** 2 * R ** 2))) * iww

    def off(ga, gb):
        return ga * gb * R ** (-2) * (Gp * (-3 + 3j * kp * R + kp ** 2 * R ** 2)
                                      - Gs * (-3 + 3j * ks * R + ks ** 2 * R ** 2)) * iww

    B11 = diag(g1 ** 2)
    B22 = diag(g2 ** 2)
    B33 = diag(g3 ** 2)
    B12 = off(g1, g2)
    B13 = off(g1, g3)
    B23 = off(g2, g3)
    return B11, B12, B13, B12, B22, B23, B13, B23, B33


def greens_deri_src(vp, vs, rho, w, x, y, z, x0, y0, z0):
    """Source-coordinate derivatives of the displacement Green tensor
    (greens_deri_src.m), for moment-tensor sources. Vectorized over the
    field points (x,y,z); source at (x0,y0,z0).

    Returns gi1, gi2, gi3, each shape (3, 3, N): the 27 entries of
    d G_ij / d x_n. The Mathematica-generated MATLAB expressions reduce
    to four term patterns (A/B/C/D below); equality with the original is
    enforced by tests/test_oracle2.py.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    R = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)
    g1 = (x - x0) / R
    g2 = (y - y0) / R
    g3 = (z - z0) / R
    kp = w / vp
    ks = w / vs
    Gs = np.exp(1j * ks * R) / (4 * PI * R)
    Gp = np.exp(1j * kp * R) / (4 * PI * R)
    R3 = R ** (-3)
    iww = 1.0 / (rho * w ** 2)

    Wp = -15 + 15j * kp * R + 6 * kp ** 2 * R ** 2 - 1j * kp ** 3 * R ** 3
    Ws = 15 - 15j * ks * R - 6 * ks ** 2 * R ** 2 + 1j * ks ** 3 * R ** 3

    def A(pref, gsq):
        return pref * R3 * (Gp * (9 - 9j * kp * R - 3 * kp ** 2 * R ** 2 + gsq * Wp)
                            + Gs * (-9 + 9j * ks * R + 4 * ks ** 2 * R ** 2
                                    - 1j * ks ** 3 * R ** 3 + gsq * Ws)) * iww

    def B(pref, gsq):
        return pref * R3 * (Gp * (3 - 3j * kp * R - kp ** 2 * R ** 2 + gsq * Wp)
                            + Gs * (-3 + 3j * ks * R + 2 * ks ** 2 * R ** 2
                                    - 1j * ks ** 3 * R ** 3 + gsq * Ws)) * iww

    def C(pref, gsq):
        return pref * R3 * (Gp * (3 - 3j * kp * R - kp ** 2 * R ** 2 + gsq * Wp)
                            + Gs * (-3 + 3j * ks * R + ks ** 2 * R ** 2 + gsq * Ws)) * iww

    D = g1 * g2 * g3 * R3 * (Gp * Wp + Gs * Ws) * iww

    g1s, g2s, g3s = g1 ** 2, g2 ** 2, g3 ** 2
    gi1 = np.array([[A(g1, g1s), B(g2, g1s), B(g3, g1s)],
                    [C(g2, g1s), C(g1, g2s), D],
                    [C(g3, g1s), D, C(g1, g3s)]])
    gi2 = np.array([[C(g2, g1s), C(g1, g2s), D],
                    [B(g1, g2s), A(g2, g2s), B(g3, g2s)],
                    [D, C(g3, g2s), C(g2, g3s)]])
    gi3 = np.array([[C(g3, g1s), D, C(g1, g3s)],
                    [D, C(g3, g2s), C(g2, g3s)],
                    [B(g1, g3s), B(g2, g3s), A(g3, g3s)]])
    return gi1, gi2, gi3


def greens_func_p(vp, w, x1, x2, x3, xs, ys, zs):
    """Acoustic (P) Green function (greens_func_p.m)."""
    r = np.sqrt((x1 - xs) ** 2 + (x2 - ys) ** 2 + (x3 - zs) ** 2)
    kp = w / vp
    return np.exp(1j * kp * r) / (4 * PI * r)


def greens_func_deri_p(vp, w, x1, x2, x3, xs, ys, zs, n1, n2, n3):
    """Normal derivative of the acoustic Green function
    (greens_func_deri_p.m)."""
    r = np.sqrt((x1 - xs) ** 2 + (x2 - ys) ** 2 + (x3 - zs) ** 2)
    kp = w / vp
    Ep = np.exp(1j * kp * r) / (4 * PI * r)
    g1 = (x1 - xs) / r
    g2 = (x2 - ys) / r
    g3 = (x3 - zs) / r
    return Ep * (-1.0 / r + 1j * kp) * (g1 * n1 + g2 * n2 + g3 * n3)


def g_singular_value(vp, vs, rho, w, R, n1, n2, n3):
    """Analytic disk integral of the displacement Green tensor around the
    singular point, rotated to the face frame (G_singular_value.m)."""
    kp = w / vp
    ks = w / vs
    value1 = (1.0 / 4.0) * R ** (-1) * (2j * ks * R
                                        + np.exp(1j * kp * R) * (1 - 1j * kp * R)
                                        + np.exp(1j * ks * R) * (-1 - 1j * ks * R)) / (rho * w ** 2)
    value2 = (1.0 / 2.0) * R ** (-1) * (-np.exp(1j * kp * R) + 1j * kp * R
                                        + np.exp(1j * ks * R) * (1 - 1j * ks * R)) / (rho * w ** 2)
    Galy = np.diag([value1, value1, value2])
    phi_rot = np.arctan2(n2, n1)
    theta_rot = np.arctan2(np.sqrt(n1 ** 2 + n2 ** 2), n3)
    cz, sz = np.cos(phi_rot), np.sin(phi_rot)
    cy, sy = np.cos(theta_rot), np.sin(theta_rot)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rm = Rz @ Ry
    return Rm @ Galy @ Rm.T
