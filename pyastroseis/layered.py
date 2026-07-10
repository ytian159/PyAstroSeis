"""Layered-model configuration (rung 2): YAML-described nested-shell
models built on domains.nested_shell_model.

YAML schema (see examples/config_threelayer.yml)::

    layers:                  # innermost first; SI units
      - {r: 2200e3, nmesh: 24, vp: 8000, vs: 4500, rho: 4000, Q: 500}
      - {r: 4300e3, nmesh: 90, vp: 7000, vs: 3900, rho: 3500, Q: 300}
      - {r: 6371e3, nmesh: 200, vp: 6000, vs: 3000, rho: 3000, Q: 200}
    f0: 3.0e-4               # reference frequency [Hz] -> w0 = 2*pi*f0
    seed: 1                  # mesh generator seed (optional)

`vs: 0` marks a fluid layer (innermost only). Meshes are exact
spheres from meshgen.gen_layer (nfold=0), one deterministic RNG per
layer (seed + layer index).
"""

import numpy as np

from .domains import Material, nested_shell_model
from .liquidcore import flip_normals
from .meshgen import gen_layer
from .solver import qp_factors


def read_layered_config(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "layers" not in cfg or len(cfg["layers"]) < 1:
        raise ValueError("layered config needs a non-empty 'layers' list")
    radii = [float(la["r"]) for la in cfg["layers"]]
    if radii != sorted(radii):
        raise ValueError("layers must be listed innermost first "
                         "(increasing r)")
    return cfg


def _outward(faces):
    if np.median(np.sum(faces.nvec * faces.ic, axis=1)) < 0:
        return flip_normals(faces)
    return faces


def layer_meshes(layers, seed=1):
    """One exact-sphere mesh per layer boundary, innermost first."""
    out = []
    for i, la in enumerate(layers):
        rng = np.random.default_rng(seed + i)
        faces = gen_layer((float(la["r"]),), (int(la["nmesh"]),), (0,),
                          rng=rng)[0][0]
        out.append(_outward(faces))
    return out


def layer_materials(layers, qp_mode="physical"):
    mats = []
    for la in layers:
        vp, vs = float(la["vp"]), float(la["vs"])
        rho, Q = float(la["rho"]), float(la["Q"])
        if vs == 0.0:
            qf = 1.0 if qp_mode == "physical" else 2.5
            mats.append(Material.acoustic(vp, rho, Q, qp_fac=qf))
        else:
            qk, _ = qp_factors(qp_mode, vp, vs)
            mats.append(Material.solid(vp, vs, rho, Q, qp_fac=qk))
    return mats


def incident_outer_source(model, ifaces, field):
    """Incident-field dict for a source located in the OUTERMOST
    layer: only the outer region's equations (the ("u", iface) rows of
    the free surface and of the first interface below it) carry the
    incident field; every other block is zero. field(faces) must
    return the (3n,) incident displacement on a mesh."""
    inc = {}
    for kind, iface in model.blocks:
        n = iface.faces.n
        if kind == "u" and (iface is ifaces[-1]
                            or (len(ifaces) > 1 and iface is ifaces[-2])):
            inc[(kind, iface)] = field(iface.faces)
        elif kind == "p":
            inc[(kind, iface)] = np.zeros(n, dtype=complex)
        else:
            inc[(kind, iface)] = np.zeros(3 * n, dtype=complex)
    return inc


def build_layered_model(cfg, qp_mode="physical", faces_list=None,
                        **opts):
    """cfg: dict from read_layered_config (or equivalent). Returns
    (model, interfaces, faces_list). Pass faces_list to reuse
    prebuilt meshes (e.g. loaded from files)."""
    layers = cfg["layers"]
    if faces_list is None:
        faces_list = layer_meshes(layers, seed=int(cfg.get("seed", 1)))
    mats = layer_materials(layers, qp_mode=qp_mode)
    w0 = 2.0 * np.pi * float(cfg["f0"])
    model, ifaces = nested_shell_model(faces_list, mats, w0, **opts)
    return model, ifaces, faces_list
