"""Triangular surface mesh handling.

Struct-of-arrays replacement for the MATLAB 1xN `face` struct array.
`faces_from_vertices` is a faithful port of mesh2face_convert.m.
"""

from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class Faces:
    """All per-face geometric quantities, shape (N, ...) arrays."""
    A: np.ndarray          # (N,3) vertex 1
    B: np.ndarray          # (N,3) vertex 2
    C: np.ndarray          # (N,3) vertex 3
    nvec: np.ndarray       # (N,3) outward unit normal
    ic: np.ndarray         # (N,3) incenter (collocation point)
    area: np.ndarray       # (N,)
    r: np.ndarray          # (N,) inradius
    a: np.ndarray          # (N,) side |B-C|
    b: np.ndarray          # (N,) side |C-A|
    c: np.ndarray          # (N,) side |B-A|

    @property
    def n(self):
        return self.A.shape[0]


def faces_from_struct_array(fs):
    """Convert a loaded MATLAB `face` struct array (mat_struct objects)
    to Faces."""
    fs = np.atleast_1d(fs)
    n = fs.size

    def field(name, dim):
        if dim == 3:
            out = np.empty((n, 3))
            for i in range(n):
                out[i] = np.asarray(getattr(fs[i], name), dtype=float).ravel()
        else:
            out = np.empty(n)
            for i in range(n):
                out[i] = float(getattr(fs[i], name))
        return out

    return Faces(
        A=field("A", 3), B=field("B", 3), C=field("C", 3),
        nvec=field("nvec", 3), ic=field("ic", 3),
        area=field("area", 1), r=field("r", 1),
        a=field("a", 1), b=field("b", 1), c=field("c", 1),
    )


def load_faces_mat(path):
    """Load a mesh: .mat (MATLAB `face` struct or V/Tri arrays) or a
    Wavefront .obj triangulated shape model (standard format for
    published asteroid shapes)."""
    if str(path).lower().endswith(".obj"):
        return load_shape_obj(path)
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    if "face" in m:
        return faces_from_struct_array(m["face"])
    if "V" in m and "Tri" in m:
        return faces_from_vertices(m["V"], m["Tri"])
    raise ValueError(f"{path}: no 'face' struct or V/Tri arrays found")


def load_shape_obj(path, scale=1.0):
    """Load a triangulated Wavefront .obj shape model (v/f records; f may
    use v/vt/vn syntax). scale multiplies vertex coordinates (e.g. 1e3
    for a model in km when the solver works in meters)."""
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            t = line.split()
            if not t:
                continue
            if t[0] == "v":
                V.append([float(v) for v in t[1:4]])
            elif t[0] == "f":
                if len(t) != 4:
                    raise ValueError(f"{path}: non-triangular face: "
                                     f"{line.strip()}")
                F.append([int(v.split("/")[0]) for v in t[1:4]])
    if not V or not F:
        raise ValueError(f"{path}: no vertices/faces found")
    return faces_from_vertices(np.asarray(V, dtype=float) * scale,
                               np.asarray(F, dtype=int))


def faces_from_vertices(V, Tri):
    """Build Faces from vertices V (nv,3) and triangles Tri (N,3),
    1-based indices as in MATLAB. Port of mesh2face_convert.m."""
    V = np.asarray(V, dtype=float)
    Tri = np.asarray(Tri, dtype=int)
    A = V[Tri[:, 0] - 1]
    B = V[Tri[:, 1] - 1]
    C = V[Tri[:, 2] - 1]

    a = np.linalg.norm(B - C, axis=1)
    b = np.linalg.norm(C - A, axis=1)
    c = np.linalg.norm(B - A, axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    r = area / s  # inradius

    vab = B - A
    vab = vab / np.linalg.norm(vab, axis=1)[:, None]
    vac = C - A
    vac = vac / np.linalg.norm(vac, axis=1)[:, None]
    vn = np.cross(vab, vac)
    vn = vn / np.linalg.norm(vn, axis=1)[:, None]
    flip = np.einsum("ij,ij->i", vn, A) < 0  # set outward normal
    vn[flip] = -vn[flip]

    cost = np.einsum("ij,ij->i", vab, vac)
    sinhalft = np.sqrt((1.0 - cost) / 2.0)
    dis = r / sinhalft
    bivec = vab + vac
    bivec = bivec / np.linalg.norm(bivec, axis=1)[:, None]
    ic = A + dis[:, None] * bivec

    return Faces(A=A, B=B, C=C, nvec=vn, ic=ic, area=area, r=r, a=a, b=b, c=c)
