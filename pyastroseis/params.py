"""AstroSeis parameter-file reader.

Replicates the fixed-line-order format read by AstroSeis.m
('#' starts a comment, textscan-style)."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Params:
    mesh_file: str
    R: float
    nmesh: int
    nfold: float
    out_mesh_name: str
    output_file: str
    vp: float
    vs: float
    rho: float
    Q: float
    nt: int
    dt: float
    f0: float
    source_type: str
    source_scale: float
    fsrc: np.ndarray = field(default=None)      # (3,) force (scaled)
    M: np.ndarray = field(default=None)         # (3,3) moment (scaled)
    h: float = 0.0                              # source depth (m)
    lat: float = 0.0
    lon: float = 0.0


def _tokens(line):
    return line.split("#")[0].split()


def resolve_path(name, param_file):
    """Resolve a file referenced by a parameter file: first relative to
    the parameter file's directory, then the CWD (MATLAB behavior)."""
    import os
    if not isinstance(param_file, (str, os.PathLike)):
        return name
    cand = os.path.join(os.path.dirname(os.path.abspath(param_file)), name)
    if os.path.exists(cand):
        return cand
    return name


def read_config(path):
    """Modern YAML configuration. Returns Params, or ParamsLC when a
    `core` block is present. Example:

        mesh: my_mesh.mat
        output: out_Q570.mat
        medium: {vp: 6000, vs: 3000, rho: 3000, Q: 570}
        # core: {vp: 8000, vs: 0, rho: 4000}      # -> liquid-core solver
        time: {nt: 500, dt: 0.1, f0: 0.3}
        source:
          type: single            # single | moment
          scale: 0                # amplitude = 10^scale
          force: [1, 1, 1]
          moment: [1, 0, 0, 1, 0, 1]   # Mxx Mxy Mxz Myy Myz Mzz
          depth: 2000
          lat: 0
          lon: 180
    """
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    src = c["source"]
    tm = c["time"]
    scale = 10.0 ** float(src.get("scale", 0))
    fsrc = scale * np.array(src.get("force", [0.0, 0.0, 0.0]), dtype=float)
    m = [float(v) for v in src.get("moment", [1, 0, 0, 1, 0, 1])]
    M = scale * np.array([[m[0], m[1], m[2]],
                          [m[1], m[3], m[4]],
                          [m[2], m[4], m[5]]])
    if src["type"] not in ("single", "moment"):
        raise ValueError(f"source type not recognized: {src['type']}")
    common = dict(out_mesh_name="", output_file=c["output"],
                  nt=int(tm["nt"]), dt=float(tm["dt"]), f0=float(tm["f0"]),
                  source_type=src["type"],
                  source_scale=float(src.get("scale", 0)),
                  fsrc=fsrc, M=M, h=float(src["depth"]),
                  lat=float(src.get("lat", 0)), lon=float(src.get("lon", 0)))
    med = c["medium"]
    if "core" in c:
        core = c["core"]
        return ParamsLC(mesh_file=c["mesh"], r_core=0.0, nmesh_core=0,
                        nfold_core=0.0, r_surf=0.0, nmesh_surf=0,
                        nfold_surf=0.0,
                        vp1=float(med["vp"]), vs1=float(med["vs"]),
                        rho1=float(med["rho"]), Q=float(med["Q"]),
                        vp2=float(core["vp"]), vs2=float(core.get("vs", 0)),
                        rho2=float(core["rho"]), **common)
    return Params(mesh_file=c["mesh"], R=0.0, nmesh=0, nfold=0.0,
                  vp=float(med["vp"]), vs=float(med["vs"]),
                  rho=float(med["rho"]), Q=float(med["Q"]), **common)


def read_params(path):
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    t = [_tokens(ln) for ln in lines]

    mesh_file = t[0][0]
    R, nmesh, nfold = float(t[1][0]), int(float(t[1][1])), float(t[1][2])
    out_mesh_name = t[2][0]
    output_file = t[3][0]
    vp, vs, rho, Q = (float(v) for v in t[4][:4])
    nt, dt, f0 = int(float(t[5][0])), float(t[5][1]), float(t[5][2])
    source_type = t[6][0]
    source_scale = float(t[6][1])
    scale = 10.0 ** source_scale
    fsrc = scale * np.array([float(v) for v in t[7][:3]])
    m = [float(v) for v in t[8][:6]]
    M = scale * np.array([[m[0], m[1], m[2]],
                          [m[1], m[3], m[4]],
                          [m[2], m[4], m[5]]])
    h, lat, lon = (float(v) for v in t[9][:3])

    if source_type not in ("single", "moment"):
        raise ValueError(f"source type not recognized: {source_type}")

    return Params(mesh_file=mesh_file, R=R, nmesh=nmesh, nfold=nfold,
                  out_mesh_name=out_mesh_name, output_file=output_file,
                  vp=vp, vs=vs, rho=rho, Q=Q, nt=nt, dt=dt, f0=f0,
                  source_type=source_type, source_scale=source_scale,
                  fsrc=fsrc, M=M, h=h, lat=lat, lon=lon)


@dataclass
class ParamsLC:
    mesh_file: str
    r_core: float
    nmesh_core: int
    nfold_core: float
    r_surf: float
    nmesh_surf: int
    nfold_surf: float
    out_mesh_name: str
    output_file: str
    vp1: float
    vs1: float
    rho1: float
    Q: float
    vp2: float
    vs2: float
    rho2: float
    nt: int
    dt: float
    f0: float
    source_type: str
    source_scale: float
    fsrc: np.ndarray = field(default=None)
    M: np.ndarray = field(default=None)
    h: float = 0.0
    lat: float = 0.0
    lon: float = 0.0


def read_params_lc(path):
    """Liquid-core parameter file (AstroSeis_liquidcore.m line order)."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    t = [_tokens(ln) for ln in lines]

    mesh_file = t[0][0]
    r_core, nmesh_core, nfold_core = (float(t[1][0]), int(float(t[1][1])),
                                      float(t[1][2]))
    r_surf, nmesh_surf, nfold_surf = (float(t[2][0]), int(float(t[2][1])),
                                      float(t[2][2]))
    out_mesh_name = t[3][0]
    output_file = t[4][0]
    vp1, vs1, rho1, Q = (float(v) for v in t[5][:4])
    vp2, vs2, rho2 = (float(v) for v in t[6][:3])
    nt, dt, f0 = int(float(t[7][0])), float(t[7][1]), float(t[7][2])
    source_type = t[8][0]
    source_scale = float(t[8][1])
    scale = 10.0 ** source_scale
    fsrc = scale * np.array([float(v) for v in t[9][:3]])
    m = [float(v) for v in t[10][:6]]
    M = scale * np.array([[m[0], m[1], m[2]],
                          [m[1], m[3], m[4]],
                          [m[2], m[4], m[5]]])
    h, lat, lon = (float(v) for v in t[11][:3])

    if source_type not in ("single", "moment"):
        raise ValueError(f"source type not recognized: {source_type}")

    return ParamsLC(mesh_file=mesh_file, r_core=r_core,
                    nmesh_core=nmesh_core, nfold_core=nfold_core,
                    r_surf=r_surf, nmesh_surf=nmesh_surf,
                    nfold_surf=nfold_surf, out_mesh_name=out_mesh_name,
                    output_file=output_file, vp1=vp1, vs1=vs1, rho1=rho1,
                    Q=Q, vp2=vp2, vs2=vs2, rho2=rho2, nt=nt, dt=dt, f0=f0,
                    source_type=source_type, source_scale=source_scale,
                    fsrc=fsrc, M=M, h=h, lat=lat, lon=lon)
