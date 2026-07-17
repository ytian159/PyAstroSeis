"""Layered-model configuration (rungs 2 + 2b): YAML-described
nested-shell models built on domains.nested_shell_model.

YAML schema (see examples/config_threelayer.yml)::

    layers:                  # innermost first; SI units
      - {r: 2200e3, nmesh: 24, vp: 8000, vs: 4500, rho: 4000, Q: 500}
      - {r: 4300e3, vp: 7000, vs: 3900, rho: 3500, Q: 300}
      - {r: 6371e3, vp: 6000, vs: 3000, rho: 3000, Q: 200,
         perturb: {type: ylm, l: 2, m: 0, amp: 10e3}}
    f0: 3.0e-4               # reference frequency [Hz] -> w0 = 2*pi*f0
    fmax: 5.3e-4             # band top [Hz]; drives auto nmesh
    epw: 10                  # elements per min wavelength (default 10)
    seed: 1                  # mesh generator seed (optional)

`vs: 0` marks a fluid layer (anywhere below the surface: innermost =
liquid core, internal = fluid annulus; not outermost, no two adjacent
fluids). `nmesh` is optional per layer: when omitted it is chosen by
the wavelength rule (auto_nmesh) so the interface mesh resolves the
slowest adjacent wavelength at `fmax` with `epw` elements. `perturb`
(optional, per layer) adds radial relief to that layer's OUTER
boundary: type "ylm" ({l, m, amp}) or "random" ({lmax, amp, seed,
lmin}); amp [m] is the peak radial perturbation, and amp: 0 is a
BITWISE no-op. Meshes come from meshgen (gen_layer nfold=0 /
gen_mesh_relief), one deterministic RNG per layer (seed + index).
"""

import math

import numpy as np

from .domains import Material, nested_shell_model
from .liquidcore import flip_normals
from .meshgen import gen_layer, gen_mesh_relief, relief_random, relief_ylm
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
    if any("nmesh" not in la for la in cfg["layers"]) \
            and "fmax" not in cfg:
        raise ValueError("layers without 'nmesh' need a top-level "
                         "'fmax' for the auto-mesh rule")
    return cfg


def auto_nmesh(layers, fmax, epw=10.0, nmesh_min=12,
               hr_reflector=0.09, hr_welded=0.2):
    """Mesh parameter per layer boundary: wavelength rule + curvature
    floor.

    Wavelength: with face count F = 8*nmesh - 16 (meshgen
    construction) and mean face size h = sqrt(4 pi r^2 / F), require
    h <= (v_min / fmax) / epw, where v_min is the minimum of vs (vp
    for a fluid) over the layers touching the boundary.

    Curvature: boundaries that CONFINE modes (the free surface and
    fluid-solid interfaces — near-perfect reflectors) bias the
    trapped-mode eigenfrequencies by O((h/R)^2) (measured +5.9% at
    h/R 0.26, +2.9% at 0.18 on the 0T2 control, order 2.0; ~0.7% at
    the validated h/R 0.09), so they additionally require
    h/R <= hr_reflector. Welded interfaces are transmissive and only
    need h/R <= hr_welded (0.13-0.21 passed the rung-1/2 DSM
    arbitrations). Layers carrying an explicit 'nmesh' keep it."""
    out = []
    n = len(layers)
    for i, la in enumerate(layers):
        if la.get("nmesh"):
            out.append(int(la["nmesh"]))
            continue
        cands = layers[i:i + 2]
        v = min((float(c["vs"]) if float(c["vs"]) > 0
                 else float(c["vp"])) for c in cands)
        h = v / float(fmax) / float(epw)
        F = 4.0 * math.pi * float(la["r"]) ** 2 / h ** 2
        reflector = (i == n - 1) or any(float(c["vs"]) == 0.0
                                        for c in cands)
        hr = float(hr_reflector) if reflector else float(hr_welded)
        F = max(F, 4.0 * math.pi / hr ** 2)
        out.append(max(int(nmesh_min), int(math.ceil((F + 16.0) / 8.0))))
    return out


def _outward(faces):
    if np.median(np.sum(faces.nvec * faces.ic, axis=1)) < 0:
        return flip_normals(faces)
    return faces


def _relief_from_spec(spec):
    typ = spec.get("type", "ylm")
    amp = float(spec.get("amp", 0.0))
    if typ == "ylm":
        return relief_ylm(int(spec["l"]), int(spec.get("m", 0)), amp)
    if typ == "random":
        return relief_random(int(spec.get("lmax", 8)), amp,
                             int(spec["seed"]),
                             lmin=int(spec.get("lmin", 1)))
    raise ValueError(f"unknown perturb type {typ!r} "
                     "(supported: ylm, random)")


def layer_meshes(layers, seed=1):
    """One mesh per layer boundary, innermost first: an exact sphere,
    or a radially perturbed sphere when the layer carries a 'perturb'
    spec (same construction path; amp=0 is bitwise identical)."""
    out = []
    for i, la in enumerate(layers):
        rng = np.random.default_rng(seed + i)
        pert = la.get("perturb")
        if pert is not None:
            faces, _ = gen_mesh_relief(float(la["r"]), int(la["nmesh"]),
                                       _relief_from_spec(pert), rng=rng)
        else:
            faces = gen_layer((float(la["r"]),), (int(la["nmesh"]),),
                              (0,), rng=rng)[0][0]
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
        elif kind in ("p", "un"):
            inc[(kind, iface)] = np.zeros(n, dtype=complex)
        else:
            inc[(kind, iface)] = np.zeros(3 * n, dtype=complex)
    return inc


def incident_solid_layer_source(model, ifaces, layer, field):
    """Incident-field dict for a source anywhere in SOLID layer
    `layer` (0-based, innermost first): by the representation theorem
    the incident term enters exactly the row blocks OWNED by the
    source region — its equation collocated on each of its bounding
    interfaces (("u", iface) rows where it is the first-registered
    solid, ("t", iface) rows where it is the second) — and nothing
    else. For layer == len(ifaces)-1 this reproduces
    incident_outer_source bitwise (gate in tests/test_elim.py).
    field(faces) -> (3n,) incident displacement, evaluated with the
    SOURCE layer's material."""
    name = "layer%d" % layer
    inc = {}
    for kind, iface in model.blocks:
        n = iface.faces.n
        own = None
        if kind == "u" and iface in model._solid_of:
            own = model._solid_of[iface][0].name
        elif kind == "t":
            own = model._solid_b_of[iface][0].name
        if own == name:
            inc[(kind, iface)] = field(iface.faces)
        elif kind in ("p", "un"):
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
        if any("nmesh" not in la for la in layers):
            nms = auto_nmesh(layers, float(cfg["fmax"]),
                             epw=float(cfg.get("epw", 10.0)))
            layers = [dict(la, nmesh=nm) for la, nm in zip(layers, nms)]
        faces_list = layer_meshes(layers, seed=int(cfg.get("seed", 1)))
    mats = layer_materials(layers, qp_mode=qp_mode)
    w0 = 2.0 * np.pi * float(cfg["f0"])
    model, ifaces = nested_shell_model(faces_list, mats, w0, **opts)
    return model, ifaces, faces_list


def scattered_solid_layer_source(model, ifaces, layer, mat, w,
                                 src_xyz, mts, geom_opts=None,
                                 refine_spec=((8.0, 2), (30.0, 1)),
                                 chunk=400):
    """RHS matrix B (model.size, len(mts)) for the SCATTERED-FIELD
    source formulation (rung C, docs/fast_methods_notes.md section 7).

    Unknowns are reinterpreted as the SCATTERED field within the
    source region (and on its boundary interfaces) and the total
    field elsewhere:

      * rows OWNED by the source region (its representation equation
        collocated on its boundaries):
            B = - sum_{bnd} int_bnd G(x, y) t_inc,out(y) dS(y)
        with t_inc the ANALYTIC incident traction (source_traction
        .t0eM) w.r.t. the region-outward normal, evaluated on
        distance-adaptively subdivided panels (node-level sampling;
        centroid sampling recreates the small-|kd| disease),
      * rows of OTHER regions whose equations couple to a source-
        region boundary displacement unknown:
            B -= [assembled block] @ u_inc(interface)
        (those unknowns are now scattered-field values; the other
        region needs the total field),
      * everything else: 0. Receivers must add u_inc back
        (total = scattered + incident on the source-region side).

    refine_spec: ((dist_over_h, levels), ...) sorted inner-first;
    panels with |ic - src| < dist_over_h * h get that subdivision.
    """
    import numpy as np

    from .assembly import Geometry, cal_G_st
    from .mesh import Faces, refine_faces
    from .source import u0eM
    from .source_traction import t0eM

    geom_opts = dict(geom_opts or {})
    # the refined integration panels bypass the self-block machinery
    # (collocation points sit ON parent panels): a near-singular
    # composite tier is REQUIRED, not optional (rung-4b machinery)
    geom_opts.setdefault("near_tier", (1.5, (10, 2)))
    name = "layer%d" % layer
    region = next(r for r in model.regions if r.name == name)
    xs, ys, zs = src_xyz
    src = np.array([xs, ys, zs], dtype=float)
    ncols = len(mts)
    B = np.zeros((model.size, ncols), dtype=complex)

    # ---- weak -int G t_inc over the source region's boundary ----
    for ifc, sc in region.interfaces:
        if ifc.condition not in ("free", "fluid_solid"):
            raise NotImplementedError(
                "scattered source next to %s interface"
                % ifc.condition)
        fc0 = ifc.faces
        dist = np.linalg.norm(fc0.ic - src[None, :], axis=1)
        h = np.maximum.reduce([fc0.a, fc0.b, fc0.c])
        lev = np.zeros(fc0.n, dtype=int)
        for d_over_h, lv in sorted(refine_spec, key=lambda t: t[0]):
            m = (dist < d_over_h * h) & (lev == 0)
            lev[m] = lv
        fref, _ = refine_faces(fc0, lev)
        # incident traction at sub-panel incenters, region-outward
        tvec = np.empty((3 * fref.n, ncols), dtype=complex)
        for col, M in enumerate(mts):
            t = t0eM(fref.ic, sc * fref.nvec, w, mat.rho, mat.mu,
                     mat.lamda, xs, ys, zs, mat.Q, M,
                     qp_fac=mat.qp_fac, disp_ref_hz=mat.disp_ref_hz)
            tvec[:fref.n, col] = t[:, 0]
            tvec[fref.n:2 * fref.n, col] = t[:, 1]
            tvec[2 * fref.n:, col] = t[:, 2]
        for ifr, sr in region.interfaces:
            row = ("u", ifr)
            if row not in model._slices:
                continue
            if model._solid_of[ifr][0] is not region:
                continue
            fr = model._oriented[(ifr, sr)]
            acc = np.zeros((3 * fr.n, ncols), dtype=complex)
            for j0 in range(0, fref.n, chunk):
                j1 = min(j0 + chunk, fref.n)
                sub = Faces(A=fref.A[j0:j1], B=fref.B[j0:j1],
                            C=fref.C[j0:j1], nvec=fref.nvec[j0:j1],
                            ic=fref.ic[j0:j1], area=fref.area[j0:j1],
                            r=fref.r[j0:j1], a=fref.a[j0:j1],
                            b=fref.b[j0:j1], c=fref.c[j0:j1])
                geo = Geometry(sub, **geom_opts)
                Gb = cal_G_st(fr, sub, w, mat.lamda, mat.mu, mat.rho,
                              mat.Q, qp_fac=mat.qp_fac, geom=geo,
                              disp_ref_hz=mat.disp_ref_hz)
                idx = np.concatenate([np.arange(j0, j1),
                                      fref.n + np.arange(j0, j1),
                                      2 * fref.n + np.arange(j0, j1)])
                acc += Gb @ tvec[idx, :]
            B[model._slices[row], :] -= acc

    # ---- corrections: other regions coupling to boundary u ----
    bnd_u_cols = set()
    for ifc, sc in region.interfaces:
        if ("u", ifc) in model._slices:
            bnd_u_cols.add(("u", ifc))

    def _row_owned_by_source(row):
        kind, ifr = row
        if kind == "u" and ifr in model._solid_of:
            return model._solid_of[ifr][0] is region
        if kind == "t":
            return model._solid_b_of[ifr][0] is region
        return False              # p / un rows are fluid-owned

    def skip(row, col):
        return not (col in bnd_u_cols
                    and not _row_owned_by_source(row))

    blocks = model.assemble_blocks(w, skip=skip)
    u_inc_cache = {}
    for (row, col), blk in blocks.items():
        ifc = col[1]
        if col not in u_inc_cache:
            u_inc_cache[col] = np.stack([
                u0eM(ifc.faces, w, mat.rho, mat.mu, mat.lamda,
                     xs, ys, zs, mat.Q, M, qp_fac=mat.qp_fac,
                     disp_ref_hz=mat.disp_ref_hz)
                for M in mts], axis=1)
        B[model._slices[row], :] -= blk @ u_inc_cache[col]
    return B
