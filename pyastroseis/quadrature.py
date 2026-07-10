"""Xiao-Gimbutas symmetric quadrature on triangles.

Faithful port of lib_BEM/quad_integration (John Burkardt's MATLAB port
of the Hong Xiao & Zydrunas Gimbutas FORTRAN77 code; GNU LGPL).
Only the degree-10 rule table is included — the only one AstroSeis
uses (nint = 10 hardcoded in cal_traction_tri_vec.m).
"""

import numpy as np

_EPS = np.finfo(float).eps
_SQRT_EPS = np.sqrt(_EPS)

RULE_FULL_SIZE = {10: 25}
RULE_COMPRESSED_SIZE = {10: 7}


def _rule10():
    """Compressed degree-10 rule data (rule10.m)."""
    x = np.array([
        0.00000000000000000000000000000000,
        0.00000000000000000000000000000000,
        -0.69780686931593427582366555189730,
        0.00000000000000000000000000000000,
        -0.30903100009613455447142535490429,
        0.00000000000000000000000000000000,
        0.00000000000000000000000000000000])
    y = np.array([
        -0.56063064349133316993278274030104,
        0.10883996591237330573849553975247e+01,
        -0.51720719429149531106975531479846,
        0.00000000000000000000000000000000,
        -0.51225507594767383123122215407598,
        0.51562570796758001502147445483555,
        -0.32874839651011214404597359718045])
    w = np.array([
        0.64438869372269120316756176676570e-02,
        0.42018026730627469477523487256964e-02,
        0.38116505989607355372802824839739e-01,
        0.18340560543312397707554978722912e-01,
        0.50983455788600478135692116331631e-01,
        0.51743930451848519084272818465418e-01,
        0.49515526441757000856785778879781e-01])
    return x, y, w


_RULE_TABLES = {10: _rule10}


def _quaerotate(x, y):
    """120-degree rotation (quaerotate.m)."""
    theta = 2.0 * np.pi / 3.0
    a11 = np.cos(theta)
    a22 = np.cos(theta)
    a12 = -np.sin(theta)
    a21 = -a12
    return a11 * x + a12 * y, a21 * x + a22 * y


def _quaeinside(iitype, x, y):
    """Point-in-region tests on the D3 reference triangle (quaeinside.m)."""
    s = np.sqrt(3.0)
    if iitype == 2:                      # lower-left 1/6
        nbool = 1
        if x < -1.0 or 0.0 < x:
            nbool = 0
        if y < -1.0 / s - 1.0e-30 or x / s < y:
            nbool = 0
    elif iitype == 1:                    # lower 1/3
        nbool = 0
        if x <= 0.0 and -1.0 <= x and -1.0 / s <= y and y <= x / s:
            nbool = 1
        if 0.0 <= x and x <= 1.0 and -1.0 / s <= y and y <= -x / s:
            nbool = 1
    elif iitype == 0:                    # whole triangle
        nbool = 1
        if y < -1.0 / s:
            nbool = 0
        if s * x + 2.0 / s < y:
            nbool = 0
        if -s * x + 2.0 / s < y:
            nbool = 0
    else:
        raise ValueError("iitype must be 0, 1 or 2")
    return nbool


def _quaequad0(mmax):
    """Retrieve compressed rule and reflect nodes into canonical
    positions (quaequad0.m)."""
    if mmax not in _RULE_TABLES:
        raise NotImplementedError(
            f"rule table for degree {mmax} not ported (only {sorted(_RULE_TABLES)})")
    x, y, w = (a.copy() for a in _RULE_TABLES[mmax]())
    for i in range(len(x)):
        if _quaeinside(2, x[i], y[i]) == 1:
            pass
        elif _quaeinside(1, x[i], y[i]) == 1:
            x[i] = -x[i]
        elif _quaeinside(0, x[i], y[i]) == 1:
            x[i], y[i] = _quaerotate(x[i], y[i])
        else:
            raise RuntimeError("quaequad0: point does not lie inside triangle")
    return x, y, w


def _quaenodes(xs, ys, ws):
    """Expand compressed nodes to the full reference triangle
    (quaenodes.m)."""
    xo, yo, wo = [], [], []
    for i in range(len(xs)):
        if xs[i] ** 2 + ys[i] ** 2 < _EPS:
            xo.append(xs[i]); yo.append(ys[i]); wo.append(ws[i])
        elif xs[i] ** 2 < _EPS or abs(ys[i] - xs[i] / np.sqrt(3.0)) < _SQRT_EPS:
            x0, y0, w0 = xs[i], ys[i], ws[i] / 3.0
            xo.append(x0); yo.append(y0); wo.append(w0)
            x1, y1 = _quaerotate(x0, y0)
            xo.append(x1); yo.append(y1); wo.append(w0)
            x2, y2 = _quaerotate(x1, y1)
            xo.append(x2); yo.append(y2); wo.append(w0)
        else:
            x0, y0, w0 = xs[i], ys[i], ws[i] / 6.0
            xo.append(x0); yo.append(y0); wo.append(w0)
            x1, y1 = _quaerotate(x0, y0)
            xo.append(x1); yo.append(y1); wo.append(w0)
            x2, y2 = _quaerotate(x1, y1)
            xo.append(x2); yo.append(y2); wo.append(w0)
            xo.append(-x0); yo.append(y0); wo.append(w0)
            x1, y1 = _quaerotate(-x0, y0)
            xo.append(x1); yo.append(y1); wo.append(w0)
            x2, y2 = _quaerotate(x1, y1)
            xo.append(x2); yo.append(y2); wo.append(w0)
    return np.array(xo), np.array(yo), np.array(wo)


def quaequad(itype, mmax):
    """Full symmetric rule on the D3 reference triangle (quaequad.m).
    Only itype = 0 (whole triangle) is needed/ported."""
    if itype != 0:
        raise NotImplementedError("only itype=0 is ported")
    x, y, w = _quaequad0(mmax)
    xs, ys, ws = _quaenodes(x, y, w)
    return np.vstack([xs, ys]), ws


def _triasimp(x, y):
    """Map D3 reference triangle -> unit simplex coords (triasimp.m)."""
    scale = 1.0 / np.sqrt(3.0)
    u = 0.5 * (x + 1.0) - 0.5 * scale * (y + scale)
    v = 1.0 * scale * (y + scale)
    return u, v


def _triangle_area(v1, v2, v3):
    """Signed area of a 2-D triangle (triangle_area.m)."""
    return 0.5 * ((v2[0] - v1[0]) * (v3[1] - v1[1])
                  - (v3[0] - v1[0]) * (v2[1] - v1[1]))


def triasymq(n, vert1, vert2, vert3):
    """Symmetric quadrature on a user 2-D triangle (triasymq.m).

    Returns (rnodes, weights): rnodes is (2, npts), weights sum to the
    triangle area.
    """
    zs, whts = quaequad(0, n)
    area = abs(_triangle_area(vert1, vert2, vert3))
    scale = area / whts.sum()
    u, v = _triasimp(zs[0, :], zs[1, :])
    x = (vert2[0] - vert1[0]) * u + (vert3[0] - vert1[0]) * v + vert1[0]
    y = (vert2[1] - vert1[1]) * u + (vert3[1] - vert1[1]) * v + vert1[1]
    return np.vstack([x, y]), whts * scale


def simplex_rule(degree):
    """Quadrature rule on the unit simplex (0,0)-(1,0)-(0,1) exact for
    polynomials of the given total degree. Weights sum to the simplex
    area 1/2. Used by the distance-adaptive assembly for mid/far element
    pairs; degree 10 is the Xiao-Gimbutas rule the MATLAB code uses.

    degree 1: centroid (1 pt); 2: 3-pt symmetric; 5: 7-pt Radon;
    10: 25-pt Xiao-Gimbutas.
    """
    if degree == 10:
        return triasymq(10, (0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    if degree == 1:
        ref = np.array([[1.0 / 3.0], [1.0 / 3.0]])
        w = np.array([0.5])
        return ref, w
    if degree == 2:
        ref = np.array([[1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0]])
        w = np.full(3, 1.0 / 6.0)
        return ref, w
    if degree == 5:  # Radon's 7-point rule
        s15 = np.sqrt(15.0)
        a1 = (6.0 + s15) / 21.0
        a2 = (6.0 - s15) / 21.0
        w1 = (155.0 + s15) / 2400.0
        w2 = (155.0 - s15) / 2400.0
        ref = np.array([
            [1.0 / 3.0, a1, 1 - 2 * a1, a1, a2, 1 - 2 * a2, a2],
            [1.0 / 3.0, a1, a1, 1 - 2 * a1, a2, a2, 1 - 2 * a2]])
        w = np.array([9.0 / 80.0, w1, w1, w1, w2, w2, w2])
        return ref, w
    raise ValueError(f"no simplex rule for degree {degree} (1, 2, 5, 10)")
