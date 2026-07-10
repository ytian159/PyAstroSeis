"""Multi-domain BEM: Region / Interface abstraction (rung 0 of the
multi-layer roadmap, docs/multilayer_roadmap.md).

Rung-0 contract: MultiDomainModel reproduces the two existing solver
configurations BITWISE —
  * one solid region + free surface   == assembly.cal_traction
  * solid shell + fluid core          == liquidcore.liq_core
(gate: tests/test_rung0.py). New physics (welded interfaces, nested
shells) arrives in later rungs behind the same gate discipline.

Conventions (checked at model build time):
  * fluid_solid interface: the stored mesh normals point OUT of the
    fluid region (the fluid side registers it with sign=+1), matching
    liq_core's use of smat_func on the canonical mesh;
  * every interface has exactly one adjacent solid region; two solid
    sides (welded) is rung 1;
  * a fluid region is bounded by fluid_solid interfaces only.

Unknown/row layout: all pressure blocks first (fluid regions in the
order given, their interfaces in the order given), then displacement
blocks in first-appearance order over the regions. With regions
(fluid core, solid shell) this reproduces liq_core's
x = [p; u_core; u_surf]; with a single solid region it is just u.
Each block's rows hold the equation of the adjacent fluid (for p) or
solid (for u) region collocated on that interface.
"""

import dataclasses

import numpy as np

from .assembly import (NINT, NXI_SELF, QP_FAC_LEGACY, Geometry, cal_A_st,
                       cal_B_st, cal_G_st, cal_T_st, smat_func)
from .liquidcore import flip_normals

FREE = "free"
FLUID_SOLID = "fluid_solid"
WELDED = "welded"


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

    @classmethod
    def solid(cls, vp, vs, rho, Q, qp_fac=QP_FAC_LEGACY):
        mu = rho * vs * vs
        return cls(rho * vp * vp - 2 * mu, mu, rho, Q, qp_fac)

    @classmethod
    def acoustic(cls, vp, rho, Q, qp_fac=1.0):
        return cls(rho * vp * vp, 0.0, rho, Q, qp_fac, fluid=True)


@dataclasses.dataclass(frozen=True, eq=False)
class Interface:
    """A closed surface mesh plus its boundary condition. The stored
    normals define the canonical orientation (the Smat side)."""
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

    w0: reference angular frequency for the fluid row scaling
    (liq_core's scale_fac); required when any region is fluid.
    """

    def __init__(self, regions, w0=None, self_scheme="polar",
                 quad_mode="adaptive", nint=NINT, nxi=NXI_SELF):
        self.regions = tuple(regions)
        self.w0 = w0

        # adjacency: one solid and at most one fluid region per interface
        self._solid_of, self._fluid_of = {}, {}
        for reg in self.regions:
            for iface, sign in reg.interfaces:
                side = self._fluid_of if reg.material.fluid \
                    else self._solid_of
                if iface in side:
                    raise NotImplementedError(
                        f"interface {iface.name!r} has two "
                        f"{'fluid' if reg.material.fluid else 'solid'} "
                        "sides (welded/fluid-fluid interfaces are a "
                        "later rung)")
                side[iface] = (reg, sign)

        # unknown/row blocks: p blocks first, then u blocks
        p_ifaces, u_ifaces, seen = [], [], set()
        for reg in self.regions:
            for iface, _ in reg.interfaces:
                if reg.material.fluid:
                    p_ifaces.append(iface)
                if iface not in seen:
                    seen.add(iface)
                    u_ifaces.append(iface)

        for iface in u_ifaces:
            has_solid = iface in self._solid_of
            has_fluid = iface in self._fluid_of
            if iface.condition == FREE:
                if not has_solid or has_fluid:
                    raise ValueError(f"free interface {iface.name!r} needs "
                                     "exactly one adjacent solid region")
            elif iface.condition == FLUID_SOLID:
                if not (has_solid and has_fluid):
                    raise ValueError(f"fluid_solid interface {iface.name!r} "
                                     "needs one solid and one fluid side")
                if self._fluid_of[iface][1] <= 0:
                    raise ValueError(
                        f"fluid_solid interface {iface.name!r}: stored "
                        "normals must point out of the fluid region "
                        "(fluid side sign=+1, liq_core Smat convention)")
            else:
                raise NotImplementedError(
                    f"interface condition {iface.condition!r} is a later "
                    "rung (rung 0 supports free / fluid_solid)")

        self.blocks = [("p", i) for i in p_ifaces] \
            + [("u", i) for i in u_ifaces]
        self._slices, off = {}, 0
        for kind, iface in self.blocks:
            n = iface.faces.n if kind == "p" else 3 * iface.faces.n
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
                                           quad_mode=quad_mode)

        self._smat = {i: smat_func(i.faces) for i in p_ifaces}

        # fluid row scaling, verbatim liq_core scale_fac (rho*c*w0 with
        # c from the adjacent solid's P modulus)
        self._scale = {}
        for iface in p_ifaces:
            freg = self._fluid_of[iface][0]
            if freg in self._scale:
                continue
            if w0 is None:
                raise ValueError("w0 is required when a region is fluid "
                                 "(fluid row scaling)")
            smat_reg = self._solid_of[freg.interfaces[0][0]][0].material
            self._scale[freg] = freg.material.rho \
                * np.sqrt(smat_reg.lamda + 2 * smat_reg.mu) \
                / np.sqrt(freg.material.rho) * w0

    def block_slice(self, kind, iface):
        """Rows/columns of one unknown block ("p" or "u" on iface)."""
        return self._slices[(kind, iface)]

    def assemble(self, w):
        """Assemble the coupled system matrix for angular frequency w."""
        A = np.zeros((self.size, self.size), dtype=complex)
        for kind_r, ifr in self.blocks:
            rows = self._slices[(kind_r, ifr)]
            if kind_r == "p":
                reg, sr = self._fluid_of[ifr]
                m = reg.material
                fr = self._oriented[(ifr, sr)]
                scale = self._scale[reg]
                for ifc, sc in reg.interfaces:
                    fc = self._oriented[(ifc, sc)]
                    geo = self._geom[(ifc, sc)]
                    Ab = cal_A_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                                  qp_fac=m.qp_fac, geom=geo)
                    Bb = cal_B_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                                  qp_fac=m.qp_fac, geom=geo)
                    A[rows, self._slices[("p", ifc)]] = Ab / scale
                    A[rows, self._slices[("u", ifc)]] = \
                        -m.rho * w ** 2 * (Bb @ self._smat[ifc]) / scale
            else:
                reg, sr = self._solid_of[ifr]
                m = reg.material
                fr = self._oriented[(ifr, sr)]
                for ifc, sc in reg.interfaces:
                    fc = self._oriented[(ifc, sc)]
                    geo = self._geom[(ifc, sc)]
                    Tb = cal_T_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                                  qp_fac=m.qp_fac, geom=geo)
                    A[rows, self._slices[("u", ifc)]] = Tb
                    if ifc.condition == FLUID_SOLID:
                        Gb = cal_G_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                                      qp_fac=m.qp_fac, geom=geo)
                        A[rows, self._slices[("p", ifc)]] = \
                            -Gb @ self._smat[ifc].T
        return A

    def assemble_rhs(self, incident):
        """Right-hand side from incident fields per row block:
        incident[(kind, iface)] = vector (P0 for "p" rows — scaled here
        like liq_core — u0 for "u" rows)."""
        b = np.empty(self.size, dtype=complex)
        for kind, iface in self.blocks:
            v = np.asarray(incident[(kind, iface)], dtype=complex).ravel()
            if kind == "p":
                v = v / self._scale[self._fluid_of[iface][0]]
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
