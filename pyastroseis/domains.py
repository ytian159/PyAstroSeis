"""Multi-domain BEM: Region / Interface abstraction (rungs 0-1 of the
multi-layer roadmap, docs/multilayer_roadmap.md).

Rung-0 contract: MultiDomainModel reproduces the two existing solver
configurations BITWISE —
  * one solid region + free surface   == assembly.cal_traction
  * solid shell + fluid core          == liquidcore.liq_core
(gate: tests/test_rung0.py).

Rung 1 adds welded solid-solid interfaces: displacement u is shared
between the two sides and the interface traction becomes an unknown
t = sigma . n_canonical (canonical = the stored mesh normals), stored
scaled as t' = t / traction_scale (rho*c*w0, same form as liq_core's
scale_fac) for conditioning. Each solid's representation equation
carries -sign * G * scale on the t' columns; the first-registered
solid's equation fills the ("u", iface) rows, the second's the
("t", iface) rows. Gate: tests/test_rung1.py (transparent interface,
A/B swap, interface refinement).

Conventions (checked at model build time):
  * fluid_solid interface: the stored mesh normals point OUT of the
    fluid region (the fluid side registers it with sign=+1), matching
    liq_core's use of smat_func on the canonical mesh;
  * free: exactly one adjacent solid; fluid_solid: one solid + one
    fluid; welded: exactly two solids with opposite signs;
  * a fluid region is bounded by fluid_solid interfaces only.

Unknown/row layout: all pressure blocks first (fluid regions in the
order given, their interfaces in the order given), then displacement
blocks in first-appearance order over the regions, then traction
blocks (welded interfaces, first-appearance order). With regions
(fluid core, solid shell) this reproduces liq_core's
x = [p; u_core; u_surf]; with a single solid region it is just u.
Each block's rows hold the equation of the adjacent fluid (for p) or
solid (for u; second solid for t) region collocated on that interface.
"""

import dataclasses

import numpy as np

from .assembly import (NINT, NXI_SELF, QP_FAC_LEGACY, Geometry, cal_A_st,
                       cal_B_st, cal_G_st, cal_T_st, smat_func)
from .liquidcore import flip_normals

FREE = "free"
FLUID_SOLID = "fluid_solid"
WELDED = "welded"
FLUID_FLUID = "fluid_fluid"


@dataclasses.dataclass(frozen=True)
class Material:
    """Homogeneous (visco)elastic or acoustic material.

    qp_fac: Qp = qp_fac * Qs inside the kernels (see solver.qp_factors;
    physical = 0.75*(vp/vs)^2, legacy MATLAB = 2.5). For a fluid the
    acoustic kernels use Q directly when qp_fac = 1.
    """
    lamda: float
    mu: float
    rho: float
    Q: float
    qp_fac: float = QP_FAC_LEGACY
    fluid: bool = False
    # causal constant-Q physical dispersion reference frequency [Hz]
    # (0 = off, historical non-dispersive constant-Q; 1.0 = DSM/PREM
    # convention). See assembly.wave_speeds.
    disp_ref_hz: float = 0.0

    @classmethod
    def solid(cls, vp, vs, rho, Q, qp_fac=QP_FAC_LEGACY,
              disp_ref_hz=0.0):
        mu = rho * vs * vs
        return cls(rho * vp * vp - 2 * mu, mu, rho, Q, qp_fac,
                   disp_ref_hz=disp_ref_hz)

    @classmethod
    def acoustic(cls, vp, rho, Q, qp_fac=1.0, disp_ref_hz=0.0):
        return cls(rho * vp * vp, 0.0, rho, Q, qp_fac, fluid=True,
                   disp_ref_hz=disp_ref_hz)


@dataclasses.dataclass(frozen=True, eq=False)
class Interface:
    """A closed surface mesh plus its boundary condition. The stored
    normals define the canonical orientation (Smat / traction side)."""
    faces: object
    condition: str = FREE
    name: str = ""


@dataclasses.dataclass(frozen=True, eq=False)
class Region:
    """Homogeneous region bounded by interfaces; each entry is
    (Interface, sign) with sign=+1 if the interface's stored normals
    point out of this region."""
    material: Material
    interfaces: tuple
    name: str = ""


class MultiDomainModel:
    """General multi-region assembler. Builds all frequency-independent
    geometry once (Geometry hoist, fork-shareable); assemble()/solve()
    are then called per frequency.

    w0: reference angular frequency for the fluid row scaling and the
    welded traction column scaling (liq_core's scale_fac form);
    required when any region is fluid or any interface is welded.
    """

    def __init__(self, regions, w0=None, self_scheme="polar",
                 quad_mode="adaptive", nint=NINT, nxi=NXI_SELF,
                 near_tier=None):
        self.regions = tuple(regions)
        self.w0 = w0

        # adjacency lists per interface, in region registration order
        solids_of, fluids_of = {}, {}
        for reg in self.regions:
            for iface, sign in reg.interfaces:
                side = fluids_of if reg.material.fluid else solids_of
                side.setdefault(iface, []).append((reg, sign))

        # unknown/row blocks: p blocks, then u blocks, then t blocks,
        # then un blocks (fluid-fluid scalar normal displacement
        # q = u . n_canonical; docs/fluid_fluid_derivation.md)
        p_ifaces, u_ifaces, seen_p, seen = [], [], set(), set()
        for reg in self.regions:
            for iface, _ in reg.interfaces:
                if reg.material.fluid and iface not in seen_p:
                    seen_p.add(iface)
                    p_ifaces.append(iface)
                if iface not in seen and iface.condition != FLUID_FLUID:
                    seen.add(iface)
                    u_ifaces.append(iface)
        t_ifaces = [i for i in u_ifaces if i.condition == WELDED]
        un_ifaces = [i for i in p_ifaces if i.condition == FLUID_FLUID]

        self._solid_of, self._solid_b_of = {}, {}
        self._fluid_of, self._fluid_b_of = {}, {}
        for iface in u_ifaces + un_ifaces:
            sol = solids_of.get(iface, [])
            flu = fluids_of.get(iface, [])
            if len(flu) > 1 and iface.condition != FLUID_FLUID:
                raise ValueError(
                    f"interface {iface.name!r} has two fluid sides but "
                    f"is not declared {FLUID_FLUID!r}")
            if iface.condition == FREE:
                if len(sol) != 1 or flu:
                    raise ValueError(f"free interface {iface.name!r} needs "
                                     "exactly one adjacent solid region")
            elif iface.condition == FLUID_SOLID:
                if len(sol) != 1 or len(flu) != 1:
                    raise ValueError(f"fluid_solid interface {iface.name!r} "
                                     "needs one solid and one fluid side")
                if sol[0][1] * flu[0][1] >= 0:
                    raise ValueError(
                        f"fluid_solid interface {iface.name!r}: the solid "
                        "and fluid sides must register opposite "
                        "orientation signs")
            elif iface.condition == WELDED:
                if len(sol) != 2 or flu:
                    raise ValueError(f"welded interface {iface.name!r} "
                                     "needs exactly two adjacent solid "
                                     "regions and no fluid side")
                if sol[0][1] * sol[1][1] >= 0:
                    raise ValueError(
                        f"welded interface {iface.name!r}: the two solid "
                        "sides must register opposite orientation signs")
                self._solid_b_of[iface] = sol[1]
            elif iface.condition == FLUID_FLUID:
                if len(flu) != 2 or sol:
                    raise ValueError(
                        f"fluid_fluid interface {iface.name!r} needs "
                        "exactly two fluid sides and no solid side")
                if flu[0][1] * flu[1][1] >= 0:
                    raise ValueError(
                        f"fluid_fluid interface {iface.name!r}: the two "
                        "fluid sides must register opposite orientation "
                        "signs")
                self._fluid_b_of[iface] = flu[1]
            else:
                raise NotImplementedError(
                    f"interface condition {iface.condition!r} is not "
                    "supported (free / fluid_solid / welded / "
                    "fluid_fluid)")
            if sol:
                self._solid_of[iface] = sol[0]
            if flu:
                self._fluid_of[iface] = flu[0]

        self.blocks = [("p", i) for i in p_ifaces] \
            + [("u", i) for i in u_ifaces] \
            + [("t", i) for i in t_ifaces] \
            + [("un", i) for i in un_ifaces]
        self._slices, off = {}, 0
        for kind, iface in self.blocks:
            n = iface.faces.n if kind in ("p", "un") \
                else 3 * iface.faces.n
            self._slices[(kind, iface)] = slice(off, off + n)
            off += n
        self.size = off

        # oriented meshes + hoisted Geometry per (interface, sign) usage;
        # sign=+1 reuses the canonical Faces object so the assembly
        # routines' `faces1 is faces2` self-block fast path fires
        self._oriented, self._geom = {}, {}
        for reg in self.regions:
            for iface, sign in reg.interfaces:
                key = (iface, sign)
                if key in self._geom:
                    continue
                fc = iface.faces if sign > 0 else flip_normals(iface.faces)
                self._oriented[key] = fc
                self._geom[key] = Geometry(fc, nint=nint, nxi=nxi,
                                           self_scheme=self_scheme,
                                           quad_mode=quad_mode,
                                           near_tier=near_tier)

        self._smat = {i: smat_func(i.faces) for i in p_ifaces}

        # fluid row scaling, verbatim liq_core scale_fac (rho*c*w0 with
        # c from the P modulus of the solid adjacent to the fluid's
        # first interface). Built for EVERY fluid region — the second
        # fluid of a fluid-fluid pair owns the "un" rows and needs a
        # scale too; a region bounded only by fluid-fluid interfaces
        # has no adjacent solid and falls back to its own P modulus
        # (same rho*c*w0 form).
        self._scale = {}
        for reg in self.regions:
            if not reg.material.fluid:
                continue
            if w0 is None:
                raise ValueError("w0 is required when a region is fluid "
                                 "(fluid row scaling)")
            sol = self._solid_of.get(reg.interfaces[0][0])
            if sol is not None:
                smat_reg = sol[0].material
                self._scale[reg] = reg.material.rho \
                    * np.sqrt(smat_reg.lamda + 2 * smat_reg.mu) \
                    / np.sqrt(reg.material.rho) * w0
            else:
                self._scale[reg] = np.sqrt(
                    reg.material.rho * reg.material.lamda) * w0

        # welded traction column scaling (same rho*c*w0 form, from the
        # first-registered solid's material)
        self._tscale = {}
        for iface in t_ifaces:
            if w0 is None:
                raise ValueError("w0 is required when an interface is "
                                 "welded (traction column scaling)")
            ma = self._solid_of[iface][0].material
            self._tscale[iface] = np.sqrt(ma.rho * (ma.lamda + 2 * ma.mu)) \
                * w0

    def block_slice(self, kind, iface):
        """Rows/columns of one unknown block ("p", "u" or "t")."""
        return self._slices[(kind, iface)]

    def traction_scale(self, iface):
        """Column scale of a welded interface's traction block: the
        physical traction is traction_scale * x[block_slice("t", iface)]
        (t = sigma . n_canonical)."""
        return self._tscale[iface]

    def _solid_entries(self, reg, ifr, sr, w, row=None, skip=None):
        """One solid region's representation equation collocated on
        interface ifr (used for both "u" and "t" row blocks): yields
        (col_block, matrix) pairs. skip(row, col) -> True suppresses
        that block's kernel evaluation (rung-3 block caching)."""
        m = reg.material
        fr = self._oriented[(ifr, sr)]

        def keep(col):
            return not (skip and skip(row, col))

        for ifc, sc in reg.interfaces:
            fc = self._oriented[(ifc, sc)]
            geo = self._geom[(ifc, sc)]
            if keep(("u", ifc)):
                Tb = cal_T_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                              qp_fac=m.qp_fac, geom=geo,
                              disp_ref_hz=m.disp_ref_hz)
                yield ("u", ifc), Tb
            if ifc.condition == FLUID_SOLID and keep(("p", ifc)):
                # traction on the solid from the fluid, w.r.t. the
                # region's outward normal: t = -p n_out = -sc * p n_can,
                # so the -G t term contributes +sc * G (Smat^T p)
                Gb = cal_G_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                              qp_fac=m.qp_fac, geom=geo,
                              disp_ref_hz=m.disp_ref_hz)
                yield ("p", ifc), sc * Gb @ self._smat[ifc].T
            elif ifc.condition == WELDED and keep(("t", ifc)):
                # T u - G t_region = u0 with t_region = sc * t_canonical
                # and t_canonical = tscale * t'
                Gb = cal_G_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                              qp_fac=m.qp_fac, geom=geo,
                              disp_ref_hz=m.disp_ref_hz)
                yield ("t", ifc), (-sc * self._tscale[ifc]) * Gb

    def _fluid_entries(self, reg, ifr, sr, w, row=None, skip=None):
        """One fluid region's representation equation collocated on
        interface ifr (used for both "p" and "un" row blocks): yields
        (col_block, matrix) pairs, mirroring _solid_entries."""
        m = reg.material
        fr = self._oriented[(ifr, sr)]
        scale = self._scale[reg]

        def keep(col):
            return not (skip and skip(row, col))

        for ifc, sc in reg.interfaces:
            fc = self._oriented[(ifc, sc)]
            geo = self._geom[(ifc, sc)]
            if keep(("p", ifc)):
                Ab = cal_A_st(fr, fc, w, m.lamda, m.mu, m.rho,
                              m.Q, qp_fac=m.qp_fac, geom=geo,
                              disp_ref_hz=m.disp_ref_hz)
                yield ("p", ifc), Ab / scale
            if ifc.condition == FLUID_FLUID:
                if keep(("un", ifc)):
                    Bb = cal_B_st(fr, fc, w, m.lamda, m.mu, m.rho,
                                  m.Q, qp_fac=m.qp_fac, geom=geo,
                                  disp_ref_hz=m.disp_ref_hz)
                    # u . n_fluid_out = sc * q (q = u . n_canonical is
                    # the scalar unknown; no Smat)
                    yield ("un", ifc), \
                        -sc * m.rho * w ** 2 * Bb / scale
            elif keep(("u", ifc)):
                Bb = cal_B_st(fr, fc, w, m.lamda, m.mu, m.rho,
                              m.Q, qp_fac=m.qp_fac, geom=geo,
                              disp_ref_hz=m.disp_ref_hz)
                # u . n_fluid_out = sc * (Smat u): the B coupling
                # carries the fluid's orientation sign
                yield ("u", ifc), \
                    -sc * m.rho * w ** 2 * (Bb @ self._smat[ifc]) \
                    / scale

    def _block_entries(self, w, skip=None):
        """Yield (row_block, col_block, matrix) for every structurally
        nonzero block of the system matrix at angular frequency w.
        skip: optional predicate skip(row_block, col_block) -> bool to
        suppress computing selected blocks (rung-3 block caching)."""
        for kind_r, ifr in self.blocks:
            row = (kind_r, ifr)
            if kind_r in ("p", "un"):
                # "p": first-registered fluid; "un": the second fluid's
                # equation on a fluid-fluid interface
                reg, sr = (self._fluid_of[ifr] if kind_r == "p"
                           else self._fluid_b_of[ifr])
                for col, M in self._fluid_entries(reg, ifr, sr, w,
                                                  row=row, skip=skip):
                    yield row, col, M
            else:
                # "u": first-registered solid; "t": the second solid's
                # equation on a welded interface
                reg, sr = (self._solid_of[ifr] if kind_r == "u"
                           else self._solid_b_of[ifr])
                for col, M in self._solid_entries(reg, ifr, sr, w,
                                                  row=row, skip=skip):
                    yield row, col, M

    def assemble_blocks(self, w, skip=None):
        """All nonzero blocks as {(row_block, col_block): matrix}."""
        return {(r, c): M for r, c, M in self._block_entries(w, skip=skip)}

    def assemble(self, w):
        """Assemble the coupled system matrix for angular frequency w."""
        A = np.zeros((self.size, self.size), dtype=complex)
        for row, col, M in self._block_entries(w):
            A[self._slices[row], self._slices[col]] = M
        return A

    def assemble_rhs(self, incident):
        """Right-hand side from incident fields per row block:
        incident[(kind, iface)] = vector. "p" rows take P0 (scaled here
        like liq_core); "u" rows the incident displacement of the
        first-registered adjacent solid; "t" rows (welded) that of the
        second solid (zero when the source is not in that region)."""
        b = np.empty(self.size, dtype=complex)
        for kind, iface in self.blocks:
            v = np.asarray(incident[(kind, iface)], dtype=complex).ravel()
            if kind == "p":
                v = v / self._scale[self._fluid_of[iface][0]]
            elif kind == "un":
                v = v / self._scale[self._fluid_b_of[iface][0]]
            b[self._slices[(kind, iface)]] = v
        return b

    def solve(self, w, incident):
        return np.linalg.solve(self.assemble(w), self.assemble_rhs(incident))


def homogeneous_model(faces, material, **opts):
    """One solid region with a free surface (== run_case physics).
    Returns (model, surface_interface)."""
    surf = Interface(faces, FREE, "surface")
    model = MultiDomainModel(
        (Region(material, ((surf, +1),), "body"),), **opts)
    return model, surf


def liquid_core_model(face_surf, face_core, mat_solid, mat_fluid, w0,
                      **opts):
    """Solid shell + fluid core (== run_case_lc physics), block order
    [p_core; u_core; u_surf]. Returns (model, surface, core)."""
    surf = Interface(face_surf, FREE, "surface")
    core = Interface(face_core, FLUID_SOLID, "core")
    model = MultiDomainModel(
        (Region(mat_fluid, ((core, +1),), "fluid core"),
         Region(mat_solid, ((core, -1), (surf, +1)), "solid shell")),
        w0=w0, **opts)
    return model, surf, core


def welded_two_layer_model(face_surf, face_core, mat_shell, mat_core, w0,
                           **opts):
    """Solid shell welded to a solid core (rung 1), block order
    [u_core; u_surf; t_core]. The shell is registered first, so
    ("u", core) rows carry the shell's equation and ("t", core) rows
    the core's. Returns (model, surface, core)."""
    surf = Interface(face_surf, FREE, "surface")
    core = Interface(face_core, WELDED, "core")
    model = MultiDomainModel(
        (Region(mat_shell, ((core, -1), (surf, +1)), "shell"),
         Region(mat_core, ((core, +1),), "core")),
        w0=w0, **opts)
    return model, surf, core


def nested_shell_model(faces_list, materials, w0, **opts):
    """N nested shells (rungs 2 + 2b), innermost layer first.

    faces_list[i] is the OUTER boundary mesh of layer i (stored
    normals pointing away from the center), so faces_list[-1] is the
    free surface; materials[i] is layer i's material. Consecutive
    solid layers are welded; fluid layers may sit anywhere below the
    surface (innermost = liquid core, internal = fluid annulus /
    subsurface ocean); a fluid-solid contact becomes fluid_solid and
    two adjacent fluid layers a fluid_fluid interface (shared p +
    shared scalar normal displacement "un" unknowns — a graded /
    staircase outer core; docs/fluid_fluid_derivation.md). Not
    supported: fluid outermost (free fluid surface). Regions are
    registered outermost-first, so for each internal interface the
    OUTER layer's equation fills the ("u", iface) rows (fluid_fluid:
    the ("p", iface) rows) and the inner layer's the ("t", iface)
    rows (fluid_fluid: the ("un", iface) rows); the 2-layer solid
    case reproduces welded_two_layer_model and the fluid-core case
    reproduces liquid_core_model exactly.

    Returns (model, interfaces) with interfaces innermost-first
    (interfaces[-1] = the free surface)."""
    n = len(faces_list)
    if len(materials) != n:
        raise ValueError("need one material per layer")
    for i, m in enumerate(materials):
        if m.fluid and i == n - 1:
            raise NotImplementedError(
                "outermost layer cannot be fluid (a free fluid "
                "surface / ocean top is a later rung)")
    ifaces = []
    for i in range(n):
        if i == n - 1:
            ifaces.append(Interface(faces_list[i], FREE, "surface"))
        elif materials[i].fluid and materials[i + 1].fluid:
            ifaces.append(Interface(faces_list[i], FLUID_FLUID,
                                    "interface%d" % i))
        elif materials[i].fluid or materials[i + 1].fluid:
            ifaces.append(Interface(faces_list[i], FLUID_SOLID,
                                    "interface%d" % i))
        else:
            ifaces.append(Interface(faces_list[i], WELDED,
                                    "interface%d" % i))
    model = nested_shell_model_from_ifaces(ifaces, materials, w0, **opts)
    return model, ifaces


def nested_shell_model_from_ifaces(ifaces, materials, w0, **opts):
    """Nested-shell model over an EXISTING innermost-first Interface
    list (conditions already set). Reusing Interface objects keeps
    assemble_blocks cache keys valid across models that share
    boundaries — the rung-3 block-caching hook: to perturb boundary k,
    replace ifaces[k] with a new Interface carrying the perturbed mesh
    and rebuild; every block not touching the new object is reusable."""
    n = len(ifaces)
    regions = []
    for i in range(n - 1, -1, -1):
        bounds = (((ifaces[i - 1], -1),) if i > 0 else ()) \
            + ((ifaces[i], +1),)
        regions.append(Region(materials[i], bounds, "layer%d" % i))
    if materials[0].fluid:
        # fluid region first, matching liquid_core_model registration
        regions = regions[-1:] + regions[:-1]
    return MultiDomainModel(tuple(regions), w0=w0, **opts)
