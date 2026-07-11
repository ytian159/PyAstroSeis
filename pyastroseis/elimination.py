"""Shell-by-shell block elimination for nested-shell models.

The nested-shell system (domains.nested_shell_model) is block
tridiagonal when the unknowns are grouped by interface: every equation
is a region's representation formula collocated on one of its bounding
interfaces, and a region touches at most two consecutive interfaces,
so the rows grouped on interface i couple only to the unknowns on
interfaces i-1, i, i+1. ShellElimination exploits this with a
generalized Thomas algorithm (forward elimination innermost-first,
optional back substitution), turning the dense O((sum_i m_i)^3) solve
into O(sum_i m_i^3) with peak memory of a few interface-sized blocks —
the enabler for graded profiles approximated by many constant shells.

The blocks come from MultiDomainModel.assemble_blocks (exactly the
matrices the dense assemble() scatters), so the eliminated solution
equals the dense np.linalg.solve up to LU rounding (gate:
tests/test_elim.py). A precomputed/patched block dict can be passed to
solve() directly — the hook for rung-3 boundary-perturbation ensembles
where only the blocks touching the perturbed interface are
reassembled (cached_blocks).
"""

import numpy as np

from .domains import FLUID_SOLID, WELDED


class ShellElimination:
    """model: MultiDomainModel; ifaces: its interfaces ordered so that
    consecutive entries are adjacent (nested_shell_model's return value,
    innermost first; the last interface is the free surface)."""

    def __init__(self, model, ifaces):
        self.model = model
        self.ifaces = list(ifaces)
        self._gi = {}
        for g, iface in enumerate(self.ifaces):
            if iface in self._gi:
                raise ValueError("duplicate interface in ifaces")
            self._gi[iface] = g
        self.groups = [[blk for blk in model.blocks if blk[1] is iface]
                       for iface in self.ifaces]
        nblk = sum(len(g) for g in self.groups)
        if nblk != len(model.blocks):
            raise ValueError("model has unknown blocks on interfaces "
                             "not listed in ifaces")
        self.local, self.sizes = [], []
        for keys in self.groups:
            sl, off = {}, 0
            for k in keys:
                s = model.block_slice(*k)
                sl[k] = slice(off, off + (s.stop - s.start))
                off = sl[k].stop
            self.local.append(sl)
            self.sizes.append(off)

    def _check_tridiagonal(self, blocks):
        for (rk, ck) in blocks:
            if abs(self._gi[rk[1]] - self._gi[ck[1]]) > 1:
                raise ValueError(
                    "system is not block tridiagonal in the given "
                    f"interface order: rows {rk} couple to {ck}")

    def _group_matrix(self, blocks, i, j):
        """Dense (sizes[i], sizes[j]) coupling of interface group j into
        the equations grouped on interface i; None if structurally 0."""
        found = False
        A = np.zeros((self.sizes[i], self.sizes[j]), dtype=complex)
        for rk in self.groups[i]:
            for ck in self.groups[j]:
                M = blocks.get((rk, ck))
                if M is not None:
                    A[self.local[i][rk], self.local[j][ck]] = M
                    found = True
        return A if found else None

    def _group_rhs(self, B, i):
        out = np.zeros((self.sizes[i], B.shape[1]), dtype=complex)
        for k in self.groups[i]:
            out[self.local[i][k]] = B[self.model.block_slice(*k)]
        return out

    def solve(self, w=None, b=None, blocks=None, full=True):
        """Solve the coupled system for RHS b (global vector from
        assemble_rhs, or a (size, nrhs) matrix of stacked RHSs).

        blocks: pass a precomputed {(row_block, col_block): matrix}
        dict (assemble_blocks output, possibly cache-patched) to skip
        assembly; otherwise w is required and the blocks are assembled
        here.

        full=True returns the complete solution, shaped like the dense
        np.linalg.solve result. full=False runs the forward sweep only
        and returns just the OUTERMOST interface group's unknowns
        (surface displacement for a free outer surface) without storing
        the back-substitution couplings — peak memory stays at a few
        interface-sized blocks."""
        if blocks is None:
            if w is None:
                raise ValueError("need w when blocks are not supplied")
            blocks = self.model.assemble_blocks(w)
        self._check_tridiagonal(blocks)
        b = np.asarray(b, dtype=complex)
        one = (b.ndim == 1)
        B = b[:, None] if one else b
        if B.shape[0] != self.model.size:
            raise ValueError("RHS length does not match model size")

        n = len(self.groups)
        gs = [None] * n
        Cs = [None] * n
        for i in range(n):
            D = self._group_matrix(blocks, i, i)
            if D is None:
                raise ValueError(f"empty diagonal group {i}")
            r = self._group_rhs(B, i)
            if i > 0:
                L = self._group_matrix(blocks, i, i - 1)
                if L is not None:
                    if Cs[i - 1] is not None:
                        D = D - L @ Cs[i - 1]
                    r = r - L @ gs[i - 1]
                if not full:
                    gs[i - 1] = Cs[i - 1] = None
            U = self._group_matrix(blocks, i, i + 1) if i < n - 1 else None
            if U is not None:
                m = U.shape[1]
                sol = np.linalg.solve(D, np.hstack([U, r]))
                Cs[i], gs[i] = sol[:, :m], sol[:, m:]
            else:
                gs[i] = np.linalg.solve(D, r)

        if not full:
            x_out = gs[-1]
            return x_out[:, 0] if one else x_out

        # back substitution: x_{n-1} = g_{n-1}; x_i = g_i - C_i x_{i+1}
        xs = [None] * n
        xs[-1] = gs[-1]
        for i in range(n - 2, -1, -1):
            xs[i] = gs[i] if Cs[i] is None else gs[i] - Cs[i] @ xs[i + 1]

        X = np.empty((self.model.size, B.shape[1]), dtype=complex)
        for i, keys in enumerate(self.groups):
            for k in keys:
                X[self.model.block_slice(*k)] = xs[i][self.local[i][k]]
        return X[:, 0] if one else X


def cached_blocks(model, w, cache, invalid_ifaces=()):
    """Assemble the block dict at frequency w, reusing entries from
    `cache` (a previous assemble_blocks result at the SAME w and
    materials) except those whose row or column interface is in
    invalid_ifaces (the perturbed boundaries). Returns (blocks, nnew)
    where nnew counts freshly computed blocks."""
    bad = set(invalid_ifaces)

    def skip(row, col):
        key = (row, col)
        return key in cache and row[1] not in bad and col[1] not in bad

    fresh = model.assemble_blocks(w, skip=skip)
    blocks = {}
    for key in _structural_keys(model):
        if key in fresh:
            blocks[key] = fresh[key]
        elif key in cache and skip(*key):
            blocks[key] = cache[key]
    return blocks, len(fresh)


def _structural_keys(model):
    """(row_block, col_block) keys of every structurally nonzero block
    (no kernel evaluation)."""
    keys = []
    for kind_r, ifr in model.blocks:
        row = (kind_r, ifr)
        if kind_r == "p":
            reg, _sr = model._fluid_of[ifr]
        elif kind_r == "u":
            reg, _sr = model._solid_of[ifr]
        else:
            reg, _sr = model._solid_b_of[ifr]
        for ifc, _sc in reg.interfaces:
            if kind_r == "p":
                keys.append((row, ("p", ifc)))
                keys.append((row, ("u", ifc)))
            else:
                keys.append((row, ("u", ifc)))
                if ifc.condition == FLUID_SOLID:
                    keys.append((row, ("p", ifc)))
                elif ifc.condition == WELDED:
                    keys.append((row, ("t", ifc)))
    return keys
