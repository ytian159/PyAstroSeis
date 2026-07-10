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
                 quad_mode="adaptive", nint=NINT, nxi=NXI_SELF):
        self.regions = tuple(regions)
        self.w0 = w0

        # adjacency lists per interface, in region registration order
        solids_of, fluids_of = {}, {}
        for reg in self.regions:
            for iface, sign in reg.interfaces:
                side = fluids_of if reg.material.fluid else solids_of
                side.setdefault(iface, []).append((reg, sign))

        # unknown/row blocks: p blocks, then u blocks, then t blocks
        p_ifaces, u_ifaces, seen = [], [], set()
        for reg in self.regions:
            for iface, _ in reg.interfaces:
                if reg.material.fluid:
                    p_ifaces.append(iface)
                if iface not in seen:
                    seen.add(iface)
                    u_ifaces.append(iface)
        t_ifaces = [i for i in u_ifaces if i.condition == WELDED]

        self._solid_of, self._solid_b_of, self._fluid_of = {}, {}, {}
        for iface in u_ifaces:
            sol = solids_of.get(iface, [])
            flu = fluids_of.get(iface, [])
            if len(flu) > 1:
                raise NotImplementedError(
                    f"interface {iface.name!r} has two fluid sides "
                    "(fluid-fluid interfaces are a later rung)")
            if iface.condition == FREE:
                if len(sol) != 1 or flu:
                    raise ValueError(f"free interface {iface.name!r} needs "
                                     "exactly one adjacent solid region")
            elif iface.condition == FLUID_SOLID:
                if len(sol) != 1 or len(flu) != 1:
                    raise ValueError(f"fluid_solid interface {iface.name!r} "
                                     "needs one solid and one fluid side")
                if flu[0][1] <= 0:
                    raise ValueError(
                        f"fluid_solid interface {iface.name!r}: stored "
                        "normals must point out of the fluid region "
                        "(fluid side sign=+1, liq_core Smat convention)")
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
            else:
                raise NotImplementedError(
                    f"interface condition {iface.condition!r} is not "
                    "supported (rung 1 supports free / fluid_solid / "
                    "welded)")
            if sol:
                self._solid_of[iface] = sol[0]
            if flu:
                self._fluid_of[iface] = flu[0]

        self.blocks = [("p", i) for i in p_ifaces] \
            + [("u", i) for i in u_ifaces] \
            + [("t", i) for i in t_ifaces]
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

    def _solid_rows(self, A, rows, reg, ifr, sr, w):
        """One solid region's representation equation collocated on
        interface ifr (used for both "u" and "t" row blocks)."""
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
            elif ifc.condition == WELDED:
                # T u - G t_region = u0 with t_region = sc * t_canonical
                # and t_canonical = tscale * t'
                Gb = cal_G_st(fr, fc, w, m.lamda, m.mu, m.rho, m.Q,
                              qp_fac=m.qp_fac, geom=geo)
                A[rows, self._slices[("t", ifc)]] = \
                    (-sc * self._tscale[ifc]) * Gb

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
            elif kind_r == "u":
                reg, sr = self._solid_of[ifr]
                self._solid_rows(A, rows, reg, ifr, sr, w)
            else:   # "t": the second solid's equation on a welded iface
                reg, sr = self._solid_b_of[ifr]
                self._solid_rows(A, rows, reg, ifr, sr, w)
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
