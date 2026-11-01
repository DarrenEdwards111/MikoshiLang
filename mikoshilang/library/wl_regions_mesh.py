"""Wolfram batch W22: mesh regions -- MeshRegion / BoundaryMeshRegion (inert, validated), mesh queries
(MeshCoordinates, MeshCells, MeshPrimitives, MeshCellCount, MeshCellIndex, AdjacentMeshCells, NearestMeshCells),
mesh constructions (DelaunayMesh, VoronoiMesh, ConvexHullMesh, ConvexHullRegion, ArrayMesh, DiscretizeRegion,
BoundaryDiscretizeRegion, DiscretizeGraphics, BoundaryDiscretizeGraphics, BoundaryMesh), components and the
Euler characteristic.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  Mesh cells are given by integer indices into the coordinate list (1-based, as in the
Wolfram Language).  The region of a MeshRegion is the union of its cells; that of a BoundaryMeshRegion the
full-dimensional region its closed boundary encloses.  Lower-dimensional cells derived from higher ones (edges of
triangles, ...) are listed in order of first appearance; Wolfram's internal order of derived cells is not
documented, so cell indices of derived cells (and the coordinate order of constructed meshes) may differ.
"""
from __future__ import annotations

import itertools

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, error
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_geomprims as G
from .wl_geomprims import sx, pt, pts, out, num, concrete, _is_coord_list
from . import wl_regions as W
from .wl_regions import (S, sgn, le, lt, eqz, vsub, vadd, vmul, dot, n2, norm, simp, smin, smax, is_zero_vec,
                         affine_rank, region_of, PointsR, SimplexR, Polygon2R, Polygon3R, PolyhedronR,
                         UnionCellsR, EmptyR, convex_hull_2d, convex_hull_3d, _nz_count, _cross3)

AREA = "geometry: mesh regions (W22)"

CELL_DIM = {"Point": 0, "Line": 1, "Triangle": 2, "Polygon": 2, "Tetrahedron": 3, "Hexahedron": 3, "Prism": 3,
            "Pyramid": 3}
WRAPPERS = ("Style", "Labeled", "Annotation", "Tooltip", "Property")
# faces (0-based local indices, outward when the cell is positively oriented) of the 3-D cell types
TET_FACES = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]]
HEX_FACES = [[3, 2, 1, 0], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [4, 5, 6, 7]]
PRISM_FACES = [[0, 2, 5, 3], [1, 0, 3, 4], [5, 2, 1, 4], [0, 1, 2], [3, 5, 4]]
PYRAMID_FACES = [[3, 2, 1, 0], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
CELL_FACES = {"Tetrahedron": TET_FACES, "Hexahedron": HEX_FACES, "Prism": PRISM_FACES, "Pyramid": PYRAMID_FACES}
CELL_NV = {"Tetrahedron": 4, "Hexahedron": 8, "Prism": 6, "Pyramid": 5, "Triangle": 3}
MESH_OPTS = None    # any trailing options are accepted (presentation options) and kept


class Mesh:
    """Parsed mesh: coordinates (SymPy), explicit cells [(head, idx tuple 0-based)], boundary flag."""

    def __init__(self, P, cells, boundary, expr, f):
        self.P, self.cells, self.boundary, self.expr, self.f = P, cells, boundary, expr, f
        self.edim = len(P[0])
        self._all = None

    @property
    def dim(self):
        if self.boundary:
            return self.edim
        return max(CELL_DIM[h] for h, _ in self.cells)

    def top_dim(self):
        return max(CELL_DIM[h] for h, _ in self.cells)

    def all_cells(self):
        """{d: [(head, idx)]} including derived lower-dimensional cells (unique, order of first appearance)."""
        if self._all is not None:
            return self._all
        res = {d: [] for d in range(4)}
        seen = {d: set() for d in range(4)}

        def add(d, h, idx):
            key = frozenset(idx) if d < 3 else tuple(sorted(idx))
            if d == 1:
                key = frozenset(idx)
            elif d == 2:
                key = tuple(sorted(idx))
            if key in seen[d]:
                return
            seen[d].add(key)
            res[d].append((h, tuple(idx)))
        for h, idx in self.cells:
            add(CELL_DIM[h], h, idx)
        for d in (3, 2, 1):
            for h, idx in list(res[d]):
                for fh, fidx in faces_of(h, idx):
                    add(d - 1, fh, fidx)
        res[0] = [("Point", (i,)) for i in range(len(self.P))]
        top = max(d for d in range(4) if res[d]) if any(res[d] for d in range(4)) else 0
        self._all = {d: res[d] for d in range(top + 1)}
        if self.boundary:
            self._all = dict(self._all)
        return self._all

    def count(self, d):
        a = self.all_cells()
        return len(a.get(d, []))


def faces_of(h, idx):
    if h in ("Triangle", "Polygon"):
        n = len(idx)
        return [("Line", (idx[i], idx[(i + 1) % n])) for i in range(n)]
    if h == "Line":
        return [("Point", (i,)) for i in idx]
    if h in CELL_FACES:
        return [("Polygon" if len(fc) > 3 else "Triangle", tuple(idx[j] for j in fc)) for fc in CELL_FACES[h]]
    return []


def _unwrap(c):
    while isinstance(c, Expr) and isinstance(c.head, str) and c.head in WRAPPERS and c.args:
        c = c.args[0]
    return c


def _int_idx(v, n, f):
    if not K.is_int(v) or isinstance(v, bool):
        if isinstance(v, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: cell vertices must be integer indices, not {K_input(v)}")
    i = int(v)
    if not 1 <= i <= n:
        raise ValueError(f"{f}: vertex index {i} is out of range 1..{n}")
    return i - 1


def parse_cells(cl, n, f):
    """List of mesh cells h[{i, ...}] (or h[{{i, ...}, ...}], wrappers allowed) -> [(head, 0-based tuple)]."""
    items = cl.args if K.is_list(cl) else [cl]
    res = []
    for c in items:
        c = _unwrap(c)
        if K.is_list(c):
            res += parse_cells(c, n, f)
            continue
        if not isinstance(c, Expr) or not isinstance(c.head, str) or c.head not in CELL_DIM or len(c.args) != 1:
            if isinstance(c, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: {K_input(c)} is not a mesh cell (Point, Line, Triangle, Polygon, Tetrahedron, "
                            f"Hexahedron, Prism or Pyramid with vertex indices)")
        h, a = c.head, c.args[0]
        if not K.is_list(a):
            res.append((h, (_int_idx(a, n, f),)) if h == "Point" else None)
            if res[-1] is None:
                raise TypeError(f"{f}: the vertices of {h} must be a list of indices")
            continue
        groups = a.args if a.args and K.is_list(a.args[0]) else [a]
        for g in groups:
            if not K.is_list(g):
                raise TypeError(f"{f}: the vertices of {h} must be a list of indices")
            idx = tuple(_int_idx(v, n, f) for v in g.args)
            if h == "Point":
                res += [("Point", (i,)) for i in idx]
            elif h == "Line":
                if len(idx) < 2:
                    raise ValueError(f"{f}: a Line cell needs at least 2 vertices")
                res += [("Line", (idx[i], idx[i + 1])) for i in range(len(idx) - 1)]
            elif h in ("Triangle", "Polygon"):
                if len(idx) < 3 or (h == "Triangle" and len(idx) != 3):
                    raise ValueError(f"{f}: a {h} cell needs {'3' if h == 'Triangle' else 'at least 3'} vertices")
                res.append((h, idx))
            else:
                if len(idx) != CELL_NV[h]:
                    raise ValueError(f"{f}: a {h} cell needs {CELL_NV[h]} vertices")
                res.append((h, idx))
    return res


def mesh_of(x, f):
    """Mesh object for MeshRegion[...] / BoundaryMeshRegion[...]; TypeError if x is not one."""
    if not (isinstance(x, Expr) and x.head in ("MeshRegion", "BoundaryMeshRegion")):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {K_input(x)} is not a MeshRegion or BoundaryMeshRegion")
    a, _opts = split_trailing_options(list(x.args))
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head in ("MeshRegion", "BoundaryMeshRegion"):
        return mesh_of(a[0], f)
    if len(a) < 2:
        raise ValueError(f"{f}: expected {x.head}[{{p1, p2, ...}}, {{cell1, ...}}]")
    if x.head == "MeshRegion" and len(a) != 2:
        raise ValueError(f"{f}: expected MeshRegion[{{p1, p2, ...}}, {{cell1, ...}}]")
    P = pts(a[0], f, "coordinates")
    concrete(P)
    if len(P[0]) > 3:
        raise ValueError(f"{f}: mesh regions are embedded in 1, 2 or 3 dimensions")
    cells = []
    for cl in a[1:]:
        cells += parse_cells(cl, len(P), f)
    if not cells:
        raise ValueError(f"{f}: the mesh needs at least one cell")
    for h, idx in cells:
        d = CELL_DIM[h]
        if d > len(P[0]):
            raise ValueError(f"{f}: {h} cells need at least {d}-dimensional coordinates")
    m = Mesh(P, cells, x.head == "BoundaryMeshRegion", x, f)
    if m.boundary:
        _check_closed(m, f)
    return m


def _check_closed(m, f):
    n = m.edim
    want = n - 1
    if any(CELL_DIM[h] != want for h, _ in m.cells):
        raise ValueError(f"{f}: a BoundaryMeshRegion in {n} dimensions needs {want}-dimensional boundary cells")
    if n == 1:
        if len({i for _, idx in m.cells for i in idx}) % 2:
            raise ValueError(f"{f}: the boundary points must come in pairs")
        return
    cnt = {}
    for h, idx in m.cells:
        for fh, fidx in faces_of(h, idx):
            k = frozenset(fidx)
            cnt[k] = cnt.get(k, 0) + 1
    if any(v % 2 for v in cnt.values()):
        raise ValueError(f"{f}: the boundary is not closed")


# ───────────────────────────── conversion to the region model ─────────────────────────────

def cell_region(m, h, idx, f):
    V = [m.P[i] for i in idx]
    d = CELL_DIM[h]
    if d == 0:
        return PointsR(V)
    if d == 1 or (d == 2 and len(V) == 3 and m.edim == 2) or h == "Tetrahedron" or \
            (h == "Triangle"):
        return SimplexR([V], f=f)
    if d == 2:
        return Polygon2R([(V, [])], f=f) if m.edim == 2 else Polygon3R([(V, [])], f=f)
    return PolyhedronR(V, [list(fc) for fc in CELL_FACES[h]], f=f)


def _loops(edges):
    """Closed vertex loops from undirected edges (every vertex of even degree; degree-2 assumed)."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used = set()
    loops = []
    for a, b in edges:
        if frozenset((a, b)) in used:
            continue
        loop = [a]
        prev, cur = a, b
        used.add(frozenset((a, b)))
        while cur != a:
            loop.append(cur)
            nxt = [v for v in adj[cur] if frozenset((cur, v)) not in used]
            if not nxt:
                raise ValueError("Mesh: the boundary is not closed")
            prev, cur = cur, nxt[0]
            used.add(frozenset((prev, cur)))
        loops.append(loop)
    return loops


def boundary_region(m, f):
    n = m.edim
    if n == 1:
        xs = sorted({m.P[i][0] for _, idx in m.cells for i in idx}, key=num)
        segs = [[[xs[i]], [xs[i + 1]]] for i in range(0, len(xs), 2)]
        return SimplexR(segs, f=f) if len(segs) > 1 else SimplexR(segs, f=f)
    if n == 2:
        loops = [[m.P[i] for i in lp] for lp in _loops([idx for _, idx in m.cells])]
        # nesting depth decides outer rings / holes
        comps = []
        depth = []
        for i, r in enumerate(loops):
            d = sum(1 for j, s in enumerate(loops) if j != i and W.ring_contains(s, r[0]) > 0)
            depth.append(d)
        outers = [i for i in range(len(loops)) if depth[i] % 2 == 0]
        for i in outers:
            holes = [loops[j] for j in range(len(loops)) if depth[j] == depth[i] + 1 and
                     W.ring_contains(loops[i], loops[j][0]) > 0]
            comps.append((loops[i], holes))
        return Polygon2R(comps, f=f)
    F = [list(idx) for _, idx in m.cells]
    comps = G._components(F)
    if len(comps) == 1:
        return PolyhedronR(m.P, F, f=f)
    parts = [PolyhedronR(m.P, [F[i] for i in c], f=f) for c in comps]
    depth = [sum(1 for j, q in enumerate(parts) if j != i and q._inside_closed(p.V[p.F[0][0]]))
             for i, p in enumerate(parts)]
    if max(depth) == 0:
        return UnionCellsR(parts, 3, f=f)
    if max(depth) > 1 or sum(1 for d in depth if d == 0) > 1:
        raise NotImplementedError(f"{f}: nested boundary components beyond one level are not supported")
    outer = parts[depth.index(0)]
    voids = [(p.V, p.F) for p, d in zip(parts, depth) if d == 1]
    return PolyhedronR(outer.V, outer.F, f=f, voids=voids)


def mesh_region(x, f):
    m = mesh_of(x, f)
    if m.boundary:
        r = boundary_region(m, f)
    else:
        cells = [cell_region(m, h, idx, f) for h, idx in m.cells]
        r = UnionCellsR(cells, m.edim, f=f, boundary_fn=lambda: boundary_mesh_expr(m, f))
        if len(cells) == 1:
            r.convex = cells[0].convex
    r.expr = x
    r.kind = x.head
    return r


W.CONVERTERS["MeshRegion"] = mesh_region
W.CONVERTERS["BoundaryMeshRegion"] = mesh_region


# ───────────────────────────── output helpers ─────────────────────────────

def cell_expr(h, idx):
    return Expr(h, L(*[i + 1 for i in idx]))


def mesh_expr(P, cells, head="MeshRegion", inputs=()):
    return Expr(head, out(P, *inputs), L(*[cell_expr(h, idx) for h, idx in cells]))


def compact(P, cells):
    """Keep only used coordinates (original order) and re-index."""
    used = sorted({i for _, idx in cells for i in idx})
    pos = {k: j for j, k in enumerate(used)}
    return [P[k] for k in used], [(h, tuple(pos[i] for i in idx)) for h, idx in cells]


def boundary_cells(m, f):
    """Boundary (d-1)-cells of the top-dimensional cells of a full-dimensional MeshRegion (used exactly once)."""
    top = m.top_dim()
    if top != m.edim:
        raise NotImplementedError(f"{f}: boundaries of lower-dimensional meshes are not supported")
    cnt, first = {}, {}
    order = []
    for h, idx in m.cells:
        if CELL_DIM[h] != top:
            continue
        sgn_ = _cell_orientation(m, h, idx)
        for fh, fidx in faces_of(h, idx):
            if sgn_ < 0:
                fidx = tuple(reversed(fidx))
            k = frozenset(fidx)
            if k not in cnt:
                order.append(k)
                first[k] = (fh, fidx)
            cnt[k] = cnt.get(k, 0) + 1
    if top == 1:
        return [("Point", first[k][1]) for k in order if cnt[k] == 1]
    return [first[k] for k in order if cnt[k] == 1]


def _cell_orientation(m, h, idx):
    V = [m.P[i] for i in idx]
    if CELL_DIM[h] == 2 and m.edim == 2:
        return 1 if sgn(W.signed_area(V)) >= 0 else -1
    if h == "Tetrahedron":
        a, b, c, d = V
        return 1 if sgn(dot(_cross3(vsub(b, a), vsub(c, a)), vsub(d, a))) >= 0 else -1
    return 1


def boundary_mesh_expr(m, f):
    if m.boundary:
        return m.expr
    bc = boundary_cells(m, f)
    if not bc:
        return Expr("EmptyRegion", m.edim)
    if m.edim == 2:
        loops = _loops([idx for _, idx in bc])
        P, cells = compact(m.P, [("Line", tuple(lp) + (lp[0],)) for lp in loops])
        return Expr("BoundaryMeshRegion", out(P), L(*[Expr("Line", L(*[i + 1 for i in idx])) for h, idx in cells]))
    P, cells = compact(m.P, bc)
    return mesh_expr(P, cells, "BoundaryMeshRegion")


# ───────────────────────────── MeshRegion / BoundaryMeshRegion ─────────────────────────────

@fn("MeshRegion", area=AREA, status="added",
    notes="MeshRegion[{p1, p2, ...}, {cell1, ...}] represents a mesh region: cells Point, Line, Triangle, "
          "Polygon, Tetrahedron, Hexahedron, Prism, Pyramid with 1-based vertex indices (Line[{i, j, k}] is a "
          "polyline of segments, h[{{...}, ...}] several cells, wrappers such as Style[cell, ...] allowed); "
          "coordinates in 1, 2 or 3 dimensions; presentation options are kept. The region is the union of its "
          "cells. Inert after validation: non-cells and non-integer indices raise TypeError, out-of-range "
          "indices, cells too high-dimensional for the coordinates and empty meshes ValueError. "
          "MeshRegion[mreg] gives mreg. Check: exact; tests compare region functions on meshes with the "
          "equivalent primitives.")
def _mesh_region(pts_: MatrixOf[Real] | Expr, *rest: ListOf[Expr] | Option) -> Expr:
    f = "MeshRegion"
    if isinstance(pts_, Expr) and pts_.head in ("MeshRegion", "BoundaryMeshRegion") and not \
            [r for r in rest if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed"))]:
        return pts_
    mesh_of(Expr(f, pts_, *rest), f)
    raise Unevaluated()


@fn("BoundaryMeshRegion", area=AREA, status="added",
    notes="BoundaryMeshRegion[{p1, ...}, {bcell1, ...}] (or several boundaries BoundaryMeshRegion[pts, b1, b2, "
          "...]) represents the full-dimensional region enclosed by closed boundary cells: Point pairs in 1-D, "
          "Line loops in 2-D (nested loops are holes), Polygon / Triangle surfaces in 3-D (nested closed "
          "surfaces are voids). Inert after validation: cells of the wrong dimension and open boundaries "
          "(a face used an odd number of times) raise ValueError, non-cells TypeError. Check: exact; tests "
          "compare Area / Volume with the enclosed primitives.")
def _boundary_mesh_region(pts_: MatrixOf[Real] | Expr, *rest: ListOf[Expr] | Option) -> Expr:
    f = "BoundaryMeshRegion"
    if isinstance(pts_, Expr) and pts_.head == "BoundaryMeshRegion" and not rest:
        return pts_
    mesh_of(Expr(f, pts_, *rest), f)
    raise Unevaluated()


def _mesh_q(x, head):
    if not (isinstance(x, Expr) and x.head == head):
        return False
    try:
        mesh_of(x, head + "Q")
        return True
    except (TypeError, ValueError, Unevaluated):
        return False


@fn("MeshRegionQ", area=AREA, status="added",
    notes="MeshRegionQ[x] gives True if x is a valid MeshRegion and False otherwise (including "
          "BoundaryMeshRegion and invalid cell data). Check: exact; tests compare valid and invalid meshes.")
def _mesh_region_q(x: Expr) -> Boolean:
    return _mesh_q(x, "MeshRegion")


@fn("BoundaryMeshRegionQ", area=AREA, status="added",
    notes="BoundaryMeshRegionQ[x] gives True if x is a valid BoundaryMeshRegion (closed boundary) and False "
          "otherwise. Check: exact; tests compare closed, open and non-mesh inputs.")
def _boundary_mesh_region_q(x: Expr) -> Boolean:
    return _mesh_q(x, "BoundaryMeshRegion")


@fn("MeshCoordinates", area=AREA, status="added",
    notes="MeshCoordinates[mreg] gives the coordinate list of a MeshRegion or BoundaryMeshRegion. A non-mesh "
          "raises TypeError. Check: exact; tests compare the input coordinates.")
def _mesh_coordinates(mreg: Expr) -> MatrixOf[Real]:
    m = mesh_of(mreg, "MeshCoordinates")
    return out(m.P, mreg)


def cell_spec(m, spec, f, allow_explicit=True):
    """Cell specification -> list of (d, i) (1-based i) in order."""
    A = m.all_cells()
    maxd = max(A)

    def ids(d, ispec):
        if d not in A:
            if d > 3 or d < 0:
                raise ValueError(f"{f}: invalid cell dimension {d}")
            return []
        n = len(A[d])
        if isinstance(ispec, Symbol) and ispec.name == "All":
            return [(d, i) for i in range(1, n + 1)]
        if K.is_list(ispec):
            return [x for v in ispec.args for x in ids(d, v)]
        if not K.is_int(ispec) or isinstance(ispec, bool):
            raise TypeError(f"{f}: cell indices must be integers or All, not {K_input(ispec)}")
        i = int(ispec)
        if i < 0:
            i = n + 1 + i
        if not 1 <= i <= n:
            raise ValueError(f"{f}: cell index {int(ispec)} is out of range 1..{n} for dimension {d}")
        return [(d, i)]

    def dims(dspec):
        if isinstance(dspec, Symbol) and dspec.name == "All":
            return list(range(maxd + 1))
        if K.is_list(dspec):
            return [int(v) for v in dspec.args]
        if not K.is_int(dspec) or isinstance(dspec, bool):
            raise TypeError(f"{f}: the cell dimension must be an integer or All, not {K_input(dspec)}")
        return [int(dspec)]
    if K.is_int(spec) and not isinstance(spec, bool):
        d = int(spec)
        if d < 0 or d > 3:
            raise ValueError(f"{f}: invalid cell dimension {d}")
        return ids(d, Symbol("All"))
    if isinstance(spec, Symbol) and spec.name == "All":
        return [x for d in range(maxd + 1) for x in ids(d, spec)]
    if K.is_list(spec) and len(spec.args) == 2 and not K.is_list(spec.args[0]) and \
            not (isinstance(spec.args[0], Expr) and spec.args[0].head in CELL_DIM):
        return [x for d in dims(spec.args[0]) for x in ids(d, spec.args[1])]
    if K.is_list(spec):
        return [x for s in spec.args for x in cell_spec(m, s, f)]
    if allow_explicit and isinstance(spec, Expr) and spec.head in CELL_DIM:
        res = []
        for h, idx in parse_cells(spec, len(m.P), f):
            d = CELL_DIM[h]
            key = frozenset(idx)
            for i, (h2, idx2) in enumerate(A.get(d, [])):
                if frozenset(idx2) == key:
                    res.append((d, i + 1))
                    break
            else:
                raise ValueError(f"{f}: the cell {K_input(spec)} is not in the mesh")
        return res
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{f}: invalid cell specification {K_input(spec)}")


def _single(spec):
    return K.is_list(spec) and len(spec.args) == 2 and K.is_int(spec.args[0]) and K.is_int(spec.args[1])


@fn("MeshCells", area=AREA, status="added",
    notes="MeshCells[mreg, d] gives the cells of dimension d (h[{i1, ...}] with 1-based coordinate indices), "
          "MeshCells[mreg, {d, i}] one cell, MeshCells[mreg, {d, {i1, ...}}] / {d, All} / All several. 0-cells "
          "are Point[{i}] for every coordinate; derived lower-dimensional cells (edges of triangles, faces of "
          "tetrahedra) come in order of first appearance. Out-of-range indices raise ValueError, bad "
          "specifications TypeError. Check: exact; tests compare cell lists and MeshCellCount.")
def _mesh_cells(mreg: Expr, spec: Integer | ListOf[Expr] | Expr) -> ListOf[Expr] | Expr:
    f = "MeshCells"
    m = mesh_of(mreg, f)
    A = m.all_cells()
    ids = cell_spec(m, spec, f)
    res = [cell_expr(*A[d][i - 1]) for d, i in ids]
    return res[0] if _single(spec) else L(*res)


def cell_prim(m, h, idx, inputs):
    V = [m.P[i] for i in idx]
    if h == "Point":
        return Expr("Point", out(V[0], *inputs))
    if h == "Triangle":
        return Expr("Polygon", out(V, *inputs))
    return Expr(h, out(V, *inputs))


@fn("MeshPrimitives", area=AREA, status="added",
    notes="MeshPrimitives[mreg, d] gives the cells of dimension d as primitives with explicit coordinates "
          "(Point[p], Line[{p, q}], Polygon[{...}] for 2-cells, Tetrahedron[{...}] etc. for 3-cells); "
          "MeshPrimitives[mreg, {d, i}] one primitive and the other cell specifications as in MeshCells. "
          "Check: exact; tests compare with MeshCells and MeshCoordinates.")
def _mesh_primitives(mreg: Expr, spec: Integer | ListOf[Expr] | Expr) -> ListOf[Expr] | Expr:
    f = "MeshPrimitives"
    m = mesh_of(mreg, f)
    A = m.all_cells()
    ids = cell_spec(m, spec, f)
    res = [cell_prim(m, *A[d][i - 1], (mreg,)) for d, i in ids]
    return res[0] if _single(spec) else L(*res)


@fn("MeshCellCount", area=AREA, status="added",
    notes="MeshCellCount[mreg] gives {c0, c1, ...}, the number of cells of each dimension (derived cells "
          "included); MeshCellCount[mreg, d] one count; MeshCellCount[mreg, cellspec] the number of cells "
          "specified. Check: exact; tests compare the vertex / edge / face counts of a triangulated square and "
          "a tetrahedron (4, 6, 4, 1).")
def _mesh_cell_count(mreg: Expr, *spec: Integer | ListOf[Expr]) -> ListOf[NonNegInteger] | NonNegInteger:
    f = "MeshCellCount"
    m = mesh_of(mreg, f)
    A = m.all_cells()
    if not spec:
        return L(*[len(A[d]) for d in sorted(A)])
    if len(spec) > 1:
        raise ValueError(f"{f}: expected MeshCellCount[mreg, d]")
    s = spec[0]
    if K.is_int(s) and not isinstance(s, bool):
        d = int(s)
        if d < 0:
            raise ValueError(f"{f}: invalid cell dimension {d}")
        return len(A.get(d, []))
    return len(cell_spec(m, s, f))


@fn("MeshCellIndex", area=AREA, status="added",
    notes="MeshCellIndex[mreg, d] gives the cell indices {{d, 1}, ..., {d, n}} of the d-dimensional cells; "
          "MeshCellIndex[mreg, cellspec] those of the specified cells, including explicit cells h[{i1, ...}] "
          "(matched as vertex sets; a cell not in the mesh raises ValueError). Check: exact; tests compare "
          "round trips with MeshCells.")
def _mesh_cell_index(mreg: Expr, spec: Integer | ListOf[Expr] | Expr) -> MatrixOf[Integer] | ListOf[Integer]:
    f = "MeshCellIndex"
    m = mesh_of(mreg, f)
    ids = cell_spec(m, spec, f)
    if isinstance(spec, Expr) and spec.head in CELL_DIM and len(ids) == 1:
        return L(*ids[0])
    return L(*[L(d, i) for d, i in ids])


def _cell_obj(m, d, i, f):
    h, idx = m.all_cells()[d][i - 1]
    return cell_region(m, h, idx, f)


def _incidence(m):
    """vertex sets of all cells by dimension."""
    A = m.all_cells()
    return {d: [frozenset(idx) for _, idx in A[d]] for d in A}


@fn("AdjacentMeshCells", area=AREA, status="added",
    notes="AdjacentMeshCells[mreg, cellspec, d] gives the indices {d, i} of the d-dimensional cells adjacent "
          "to the specified cells: for a lower d the faces of the cell, for a higher d the cells having it as a "
          "face, for the same d the other cells sharing a (d-1)-dimensional face (for points: joined by an "
          "edge). Sorted by index, without duplicates and without the cells themselves. Check: exact; tests "
          "compare a triangulated square's adjacencies.")
def _adjacent_mesh_cells(mreg: Expr, spec: ListOf[Expr] | Expr, d: NonNegInteger) -> MatrixOf[Integer]:
    f = "AdjacentMeshCells"
    m = mesh_of(mreg, f)
    if not K.is_int(d) or isinstance(d, bool):
        if isinstance(d, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the target dimension must be an integer")
    d = int(d)
    I_ = _incidence(m)
    if d not in I_:
        if d < 0 or d > 3:
            raise ValueError(f"{f}: invalid cell dimension {d}")
        return L()
    src = cell_spec(m, spec, f)
    res = set()
    for k, i in src:
        s = I_[k][i - 1]
        for j, t in enumerate(I_[d]):
            if (d, j + 1) == (k, i):
                continue
            if d < k and t <= s:
                res.add(j + 1)
            elif d > k and s <= t:
                res.add(j + 1)
            elif d == k:
                if k == 0:
                    if any(e == s | t for e in I_.get(1, [])):
                        res.add(j + 1)
                elif any(fc <= s and fc <= t for fc in I_[k - 1]):
                    res.add(j + 1)
    return L(*[L(d, j) for j in sorted(res)])


@fn("NearestMeshCells", area=AREA, status="added",
    notes="NearestMeshCells[mreg, p] gives {d, {i1, ...}}: the 0-dimensional cells (vertices) nearest to p "
          "(all ties); NearestMeshCells[{mreg, d}, p] uses the d-dimensional cells (distance to the closest point "
          "of the cell), NearestMeshCells[.., p, n] the n nearest (sorted by distance, ties by index), "
          "[.., p, {n, r}] at most n within distance r. Exact distances. Check: exact; tests compare hand "
          "distances to the vertices and edges of a square mesh.")
def _nearest_mesh_cells(spec: Expr | ListOf[Expr], p: ListOf[Real], *rest: PosInteger | ListOf[Real]) -> \
        ListOf[Expr]:
    f = "NearestMeshCells"
    if K.is_list(spec) and len(spec.args) == 2:
        mr, d = spec.args
        if not K.is_int(d) or isinstance(d, bool):
            raise TypeError(f"{f}: the cell dimension must be an integer")
        d = int(d)
    else:
        mr, d = spec, 0
    m = mesh_of(mr, f)
    q = pt(p, f, dim=m.edim)
    concrete(q)
    A = m.all_cells()
    if d not in A:
        raise ValueError(f"{f}: the mesh has no cells of dimension {d}")
    dists = [(simp(_cell_obj(m, d, i + 1, f).nearest(q)[0]), i + 1) for i in range(len(A[d]))]
    if not rest:
        best = smin([a for a, _ in dists])
        return L(d, L(*[i for a, i in dists if eqz(a - best)]))
    if len(rest) > 1:
        raise ValueError(f"{f}: expected NearestMeshCells[mreg, p, n]")
    nr = rest[0]
    rr = None
    if K.is_list(nr):
        if len(nr.args) != 2:
            raise ValueError(f"{f}: expected {{n, r}}")
        nr, rr = nr.args
        rr = sx(rr, f, "radius")
        concrete(rr)
    if isinstance(nr, Symbol) and nr.name == "All":
        n = len(dists)
    elif K.is_int(nr) and not isinstance(nr, bool) and int(nr) >= 1:
        n = int(nr)
    else:
        raise ValueError(f"{f}: the count must be a positive integer or All")
    import functools

    def cmp(a, b):
        s = sgn(a[0] - b[0])
        return s if s else a[1] - b[1]
    order = sorted(dists, key=functools.cmp_to_key(cmp))
    if rr is not None:
        order = [x for x in order if le(x[0], rr ** 2)]
    return L(d, L(*[i for _, i in order[:n]]))


@fn("HighlightMesh", area=AREA, status="added",
    notes="HighlightMesh[mreg, cellspec] / HighlightMesh[mreg, {w1[spec1], ...}] represents mreg with the "
          "specified cells highlighted (a display construct). Inert after validation: the mesh must be valid "
          "and the specifications valid cell specifications (wrappers Style / Labeled allowed), else TypeError "
          "/ ValueError. Check: exact; tests compare validation and the unchanged region measure.")
def _highlight_mesh(mreg: Expr, spec: Expr, *opts: Option) -> Expr:
    f = "HighlightMesh"
    m = mesh_of(mreg, f)
    items = spec.args if K.is_list(spec) and spec.args and not _single(spec) and not \
        (len(spec.args) == 2 and K.is_int(spec.args[0])) else [spec]
    for it in items:
        cell_spec(m, _unwrap(it), f)
    raise Unevaluated()


# ───────────────────────────── components ─────────────────────────────

def _uf_components(n, groups):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for g in groups:
        for v in g[1:]:
            ra, rb = find(g[0]), find(v)
            if ra != rb:
                parent[rb] = ra
    return find


@fn("ConnectedMeshComponents", area=AREA, status="added",
    notes="ConnectedMeshComponents[mreg] gives the path-connected components of a mesh region as mesh regions "
          "(cells sharing a vertex are connected), ordered by their first cell, each with only its own "
          "coordinates (original order). A BoundaryMeshRegion gives BoundaryMeshRegion components (a component "
          "with holes keeps them). Check: exact; tests compare component counts and measures summing to the "
          "whole.")
def _connected_mesh_components(mreg: Expr) -> ListOf[Expr]:
    f = "ConnectedMeshComponents"
    m = mesh_of(mreg, f)
    find = _uf_components(len(m.P), [list(idx) for _, idx in m.cells])
    groups = {}
    order = []
    for h, idx in m.cells:
        r = find(idx[0])
        if r not in groups:
            groups[r] = []
            order.append(r)
        groups[r].append((h, idx))
    head = "BoundaryMeshRegion" if m.boundary else "MeshRegion"
    if m.boundary and m.edim >= 2:
        # holes belong to the component that encloses them
        reg = [boundary_region(Mesh(m.P, groups[r], True, None, f), f) for r in order]
        merged = []
        used = set()
        for i, r in enumerate(order):
            if i in used:
                continue
            cells = list(groups[r])
            for j, r2 in enumerate(order):
                if j != i and j not in used and reg[i].member(m.P[groups[r2][0][1][0]]) and \
                        not reg[j].member(m.P[groups[r][0][1][0]]):
                    cells += groups[r2]
                    used.add(j)
            used.add(i)
            merged.append(cells)
        comps = merged
    else:
        comps = [groups[r] for r in order]
    res = []
    for cells in comps:
        P, cs = compact(m.P, cells)
        res.append(mesh_expr(P, cs, head, (mreg,)))
    return L(*res)


@fn("DimensionalMeshComponents", area=AREA, status="added",
    notes="DimensionalMeshComponents[mreg] gives {r0, r1, ..., rn} (n the embedding dimension): r_d is the mesh "
          "of the given d-dimensional cells that are not faces of higher-dimensional cells, or EmptyRegion[n] "
          "when there are none. A BoundaryMeshRegion gives EmptyRegion below dimension n and itself at n. "
          "Check: exact; tests compare a mesh with a triangle, a dangling edge and an isolated point.")
def _dimensional_mesh_components(mreg: Expr) -> ListOf[Expr]:
    f = "DimensionalMeshComponents"
    m = mesh_of(mreg, f)
    n = m.edim
    if m.boundary:
        return L(*([Expr("EmptyRegion", n)] * n + [mreg]))
    I_ = {}
    for h, idx in m.cells:
        I_.setdefault(CELL_DIM[h], []).append((h, idx))
    faces = {d: set() for d in range(4)}
    A = Mesh(m.P, m.cells, False, None, f).all_cells()
    for d in A:
        if d >= 1:
            for h, idx in A[d]:
                for fh, fidx in faces_of(h, idx):
                    faces[d - 1].add(frozenset(fidx))
    res = []
    for d in range(n + 1):
        cs = [(h, idx) for h, idx in I_.get(d, []) if frozenset(idx) not in faces[d]]
        if not cs:
            res.append(Expr("EmptyRegion", n))
        else:
            P, cc = compact(m.P, cs)
            res.append(mesh_expr(P, cc, "MeshRegion", (mreg,)))
    return L(*res)


@fn("BoundaryMesh", area=AREA, status="added",
    notes="BoundaryMesh[mreg] gives the boundary of a full-dimensional MeshRegion as a BoundaryMeshRegion (its "
          "regular closure: the (n-1)-faces used by exactly one n-cell; lower-dimensional parts are dropped). "
          "A BoundaryMeshRegion is returned unchanged; meshes without full-dimensional cells raise "
          "NotImplementedError. Check: exact; tests compare Area[BoundaryMesh[m]] with Area[m] and the "
          "boundary edge count.")
def _boundary_mesh(mreg: Expr, *opts: Option) -> Expr:
    f = "BoundaryMesh"
    m = mesh_of(mreg, f)
    if m.boundary:
        return mreg
    return boundary_mesh_expr(m, f)


@fn("EulerCharacteristic", area=AREA, status="added",
    notes="EulerCharacteristic[poly] gives V - E + F for a polyhedron (Polyhedron, Tetrahedron, Cube, the other "
          "regular solids, Cuboid, Hexahedron, Prism, Pyramid, ...) and the alternating cell count "
          "c0 - c1 + c2 - c3 for a MeshRegion / BoundaryMeshRegion (the boundary complex for the latter's "
          "cells). Non-polyhedra raise TypeError. Check: exact; tests compare 2 for convex solids, 0 for a "
          "torus-like polyhedron and the Euler-Poincare formula on meshes.")
def _euler_characteristic(poly: Expr) -> Integer:
    f = "EulerCharacteristic"
    if isinstance(poly, Expr) and poly.head in ("MeshRegion", "BoundaryMeshRegion"):
        m = mesh_of(poly, f)
        A = m.all_cells()
        return sum((-1) ** d * len(A[d]) for d in A) if not m.boundary else \
            sum((-1) ** d * len(A[d]) for d in A)
    if isinstance(poly, Expr) and poly.head in G._PLATONIC:
        a, _ = split_trailing_options(list(poly.args))
        G._check_args(poly.head, list(poly.args), allow_index=False)
        return 2
    V, F = G.polyhedron_data(poly, f)
    E = {frozenset((fc[i], fc[(i + 1) % len(fc)])) for fc in F for i in range(len(fc))}
    used = {i for fc in F for i in fc}
    return len(used) - len(E) + len(F)


# ───────────────────────────── constructions ─────────────────────────────

def _points_arg(x, f, dims=(1, 2, 3)):
    P = pts(x, f)
    concrete(P)
    if len(P[0]) not in dims:
        raise ValueError(f"{f}: the points must be {' or '.join(map(str, dims))}-dimensional")
    return P


def _dedupe(P):
    U = []
    for p in P:
        if not any(is_zero_vec(vsub(p, q)) for q in U):
            U.append(p)
    return U


def hull_mesh(P, f, inputs):
    """ConvexHullMesh of exact points -> expression."""
    P = _dedupe(P)
    n = len(P[0])
    rk = affine_rank(P)
    if rk == 0:
        return Expr("MeshRegion", out([P[0]], *inputs), L(Expr("Point", L(1))))
    if rk == 1:
        a, b = W._extreme_pair(P)
        return Expr("MeshRegion", out([a, b], *inputs), L(Expr("Line", L(1, 2))))
    if n == 1:
        lo, hi = smin([p[0] for p in P]), smax([p[0] for p in P])
        return Expr("BoundaryMeshRegion", out([[lo], [hi]], *inputs), L(Expr("Point", L(1)), Expr("Point", L(2))))
    if n == 2:
        idx = convex_hull_2d(P)
        idx_sorted = sorted(idx)
        pos = {k: j + 1 for j, k in enumerate(idx_sorted)}
        return Expr("BoundaryMeshRegion", out([P[k] for k in idx_sorted], *inputs),
                    L(Expr("Line", L(*([pos[k] for k in idx] + [pos[idx[0]]])))))
    if rk == 2:
        ring = W._convex_ring3(P, f)
        return Expr("MeshRegion", out(ring, *inputs), L(Expr("Polygon", L(*range(1, len(ring) + 1)))))
    vs, fs = convex_hull_3d(P, f)
    keep = sorted(vs)
    pos = {k: j + 1 for j, k in enumerate(keep)}
    return Expr("BoundaryMeshRegion", out([P[k] for k in keep], *inputs),
                L(*[Expr("Polygon", L(*[pos[i] for i in fc])) for fc in fs]))


def _points_of_region(x, f):
    if K.is_list(x):
        return _points_arg(x, f)
    r = region_of(x, f)
    V = r.verts()
    if V is None or not r.bounded:
        raise NotImplementedError(f"{f}: the convex hull of {r.kind} regions is not supported")
    return V


@fn("ConvexHullMesh", area=AREA, status="added",
    notes="ConvexHullMesh[{p1, p2, ...}] gives the convex hull as a BoundaryMeshRegion: end points in 1-D, a "
          "closed Line loop (counterclockwise) in 2-D, convex Polygon faces (outward) in 3-D, keeping only the "
          "hull vertices (in input order); degenerate inputs give a MeshRegion (Point, Line or planar "
          "Polygon). ConvexHullMesh[mreg] / [polytope] uses the vertices. Exact hulls; more than 60 extreme "
          "points in 3-D need SciPy (else NotImplementedError 'requires SciPy'). Check: exact; tests compare "
          "Area / Volume of the hull with hand values and the vertex count.")
def _convex_hull_mesh(x: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    f = "ConvexHullMesh"
    P = _points_of_region(x, f)
    return hull_mesh(P, f, (x,))


@fn("ConvexHullRegion", area=AREA, status="added",
    notes="ConvexHullRegion[{p1, p2, ...}] gives the convex hull as a region: Point, Line, Polygon (2-D or "
          "planar in 3-D) or Polyhedron (outward faces). ConvexHullRegion[reg] gives reg itself when it is "
          "convex (Disk, Ball, convex polygons, ...) and the hull of its vertices for polytopes and point sets; "
          "other regions raise NotImplementedError. Check: exact; tests compare the hull of a square with "
          "interior points and RegionEqual with the input for convex regions.")
def _convex_hull_region(x: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    f = "ConvexHullRegion"
    if not K.is_list(x):
        r = region_of(x, f)
        if r.convex:
            return x
        V = r.verts()
        if V is None:
            raise NotImplementedError(f"{f}: the convex hull of {r.kind} regions is not supported")
        P = V
    else:
        P = _points_arg(x, f)
    P = _dedupe(P)
    hr = W.convex_hull_region(P, f)
    e = W.region_expr(hr, (x,))
    if isinstance(hr, PolyhedronR):
        vs = sorted({i for fc in hr.F for i in fc})
        pos = {k: j + 1 for j, k in enumerate(vs)}
        return Expr("Polyhedron", out([hr.V[k] for k in vs], x), L(*[L(*[pos[i] for i in fc]) for fc in hr.F]))
    return e


def _circumsphere_contains(S_, q):
    """sign of |q - c|^2 - r^2 for the circumsphere of the simplex S_ (exact)."""
    c = W._circumcentre(S_)
    return sgn(n2(vsub(q, c)) - n2(vsub(S_[0], c)))


@fn("DelaunayMesh", area=AREA, status="added",
    notes="DelaunayMesh[{p1, p2, ...}] gives the Delaunay mesh as a MeshRegion: intervals between sorted points "
          "in 1-D (exact), triangles (counterclockwise) in 2-D and tetrahedra (positively oriented) in 3-D. "
          "Duplicate points are removed; collinear 2-D / coplanar 3-D input gives the lower-dimensional "
          "Delaunay mesh. 2-D and 3-D use scipy.spatial.Delaunay (NotImplementedError 'requires SciPy' without "
          "it) and then verify exactly that no input point lies strictly inside any circumsphere and that the "
          "cells cover the convex hull (ArithmeticError otherwise). Cocircular points admit several Delaunay "
          "meshes; the one returned is valid but may differ from Wolfram's, and cell order may differ. Check: "
          "self-check (empty circumsphere and area / volume of the hull).")
def _delaunay_mesh(points: MatrixOf[Real], *opts: Option) -> Expr:
    f = "DelaunayMesh"
    P = _dedupe(_points_arg(points, f))
    n = len(P[0])
    if len(P) < 2:
        raise ValueError(f"{f}: at least two distinct points are needed")
    rk = affine_rank(P)
    if rk == 1 or n == 1:
        a, b = W._extreme_pair(P)
        d = vsub(b, a)
        order = sorted(range(len(P)), key=lambda i: num(dot(vsub(P[i], a), d)))
        return Expr("MeshRegion", out(P, points), L(*[Expr("Line", L(order[i] + 1, order[i + 1] + 1))
                                                      for i in range(len(order) - 1)]))
    if n == 3 and rk == 2:
        raise NotImplementedError(f"{f}: Delaunay meshes of coplanar 3-D points are not supported")
    G._require_scipy(f)
    import numpy as np
    from scipy.spatial import Delaunay
    A = np.array([[float(num(c)) for c in p] for p in P])
    try:
        tri = Delaunay(A)
    except Exception as e:
        raise ArithmeticError(f"{f}: the Delaunay triangulation failed ({e})")
    cells = []
    for s in tri.simplices:
        idx = [int(i) for i in s]
        V = [P[i] for i in idx]
        if affine_rank(V) < n:
            continue
        if n == 2 and sgn(W.orient2(*V)) < 0:
            idx = [idx[0], idx[2], idx[1]]
        if n == 3:
            a, b, c, d = [P[i] for i in idx]
            if sgn(dot(_cross3(vsub(b, a), vsub(c, a)), vsub(d, a))) < 0:
                idx = [idx[0], idx[2], idx[1], idx[3]]
        cells.append(idx)
    for idx in cells:
        S_ = [P[i] for i in idx]
        for j in range(len(P)):
            if j not in idx and _circumsphere_contains(S_, P[j]) < 0:
                raise ArithmeticError(f"{f}: the triangulation failed the empty-circumsphere verification")
    hull = W.convex_hull_region(P, f)
    tot = sum((abs(W.simplex_integral([P[i] for i in idx], 1, W.X(n))) for idx in cells), sp.Integer(0))
    if not eqz(tot - hull.measure()):
        raise ArithmeticError(f"{f}: the cells do not cover the convex hull")
    cells.sort()
    h = "Triangle" if n == 2 else "Tetrahedron"
    return Expr("MeshRegion", out(P, points), L(*[Expr(h, L(*[i + 1 for i in idx])) for idx in cells]))


def _bounds_arg(b, n, f):
    if not K.is_list(b) or len(b.args) != n or not all(K.is_list(r) and len(r.args) == 2 for r in b.args):
        raise ValueError(f"{f}: the bounds must be {{{{xmin, xmax}}, ...}} with {n} ranges")
    res = []
    for r in b.args:
        lo, hi = sx(r.args[0], f, "bound"), sx(r.args[1], f, "bound")
        concrete(lo, hi)
        if not lt(lo, hi):
            raise ValueError(f"{f}: each range must have min < max")
        res.append((lo, hi))
    return res


def _clip_poly(poly, a, b):
    """Clip a convex polygon (CCW list of points) by a.x <= b (exact Sutherland-Hodgman)."""
    res = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        sp_, sq = dot(a, p) - b, dot(a, q) - b
        inp, inq = sgn(sp_) <= 0, sgn(sq) <= 0
        if inp:
            res.append(p)
        if inp != inq and not (eqz(sp_) or eqz(sq)):
            t = sp_ / (sp_ - sq)
            res.append([simp(c) for c in vadd(p, vmul(t, vsub(q, p)))])
    return _dedupe_ring(res)


def _dedupe_ring(R_):
    out_ = []
    for p in R_:
        if not out_ or not is_zero_vec(vsub(p, out_[-1])):
            out_.append(p)
    if len(out_) > 1 and is_zero_vec(vsub(out_[0], out_[-1])):
        out_.pop()
    return out_


@fn("VoronoiMesh", area=AREA, status="added",
    notes="VoronoiMesh[{p1, ...}, {{xmin, xmax}, ...}] gives the Voronoi mesh clipped to the bounds as a "
          "MeshRegion: intervals in 1-D, convex polygons (counterclockwise) in 2-D, one cell per distinct point "
          "in input order, computed by exact half-plane clipping (each cell is {x : |x - pi| <= |x - pj|}); "
          "coordinates are the distinct cell vertices in order of first appearance. VoronoiMesh[pts] without "
          "bounds raises NotImplementedError (Wolfram's automatic range is not documented); 3-D raises "
          "NotImplementedError; points outside the bounds ValueError. Check: exact; tests verify the cell "
          "areas sum to the box area and each vertex is equidistant from its generators.")
def _voronoi_mesh(points: MatrixOf[Real], *rest: MatrixOf[Real] | Option) -> Expr:
    f = "VoronoiMesh"
    P = _dedupe(_points_arg(points, f))
    n = len(P[0])
    a, _opts = split_trailing_options(list(rest))
    if not a:
        raise NotImplementedError(f"{f}: Wolfram's automatic range is not documented; give the bounds "
                                  f"VoronoiMesh[pts, {{{{xmin, xmax}}, ...}}]")
    B = _bounds_arg(a[0], n, f)
    for p in P:
        if not all(le(lo, c) and le(c, hi) for c, (lo, hi) in zip(p, B)):
            raise ValueError(f"{f}: the point {out(p)} lies outside the bounds")
    if n == 1:
        xs = sorted(range(len(P)), key=lambda i: num(P[i][0]))
        cuts = {}
        for k, i in enumerate(xs):
            lo = B[0][0] if k == 0 else (P[xs[k - 1]][0] + P[i][0]) / 2
            hi = B[0][1] if k == len(xs) - 1 else (P[i][0] + P[xs[k + 1]][0]) / 2
            cuts[i] = (simp(lo), simp(hi))
        coords, cells = [], []

        def vid(v):
            for j, q in enumerate(coords):
                if is_zero_vec(vsub(q, v)):
                    return j
            coords.append(v)
            return len(coords) - 1
        for i in range(len(P)):
            lo, hi = cuts[i]
            cells.append(("Line", (vid([lo]), vid([hi]))))
        return mesh_expr(coords, cells, "MeshRegion", (points,))
    if n != 2:
        raise NotImplementedError(f"{f}: Voronoi meshes in {n} dimensions are not supported")
    box = [[B[0][0], B[1][0]], [B[0][1], B[1][0]], [B[0][1], B[1][1]], [B[0][0], B[1][1]]]
    coords, cells = [], []

    def vid(v):
        for j, q in enumerate(coords):
            if is_zero_vec(vsub(q, v)):
                return j
        coords.append(v)
        return len(coords) - 1
    for i, p in enumerate(P):
        poly = [list(v) for v in box]
        for j, q in enumerate(P):
            if i == j:
                continue
            av = vsub(q, p)
            bv = (n2(q) - n2(p)) / 2
            poly = _clip_poly(poly, av, bv)
            if len(poly) < 3:
                break
        if len(poly) >= 3:
            cells.append(("Polygon", tuple(vid(v) for v in poly)))
    tot = sum((W.signed_area([coords[k] for k in idx]) for _, idx in cells), sp.Integer(0))
    if not eqz(tot - (B[0][1] - B[0][0]) * (B[1][1] - B[1][0])):
        raise ArithmeticError(f"{f}: the Voronoi cells do not tile the bounds")
    return mesh_expr(coords, cells, "MeshRegion", (points,))


def _opt_dict(opts, f, allowed):
    d = {}
    for o in opts:
        nm = opt_name(o)
        if nm not in allowed:
            raise ValueError(f"{f}: unknown option {nm}")
        d[nm] = o.args[1]
    return d


@fn("ArrayMesh", area=AREA, status="added",
    notes="ArrayMesh[array] gives a MeshRegion with a unit cell for every nonzero entry: intervals [i - 1, i] "
          "for a vector, unit squares for a matrix with rows laid out downwards and columns across (entry "
          "{i, j} of an m-row matrix covers [j - 1, j] x [m - i, m - i + 1]). Options DataRange -> "
          "{{xmin, xmax}, ...} rescales the grid and DataReversed -> True puts row 1 at the bottom; other "
          "presentation options are kept. Coordinates are the used grid points in order of first appearance. "
          "Rank-3 arrays raise NotImplementedError, an all-zero array gives EmptyRegion, non-numeric entries "
          "TypeError. Check: exact; tests compare Area with the number of nonzero entries and RegionMember of "
          "cell centres.")
def _array_mesh(array: ListOf[Real] | MatrixOf[Real], *opts: Option) -> Expr:
    f = "ArrayMesh"
    if not K.is_list(array):
        if isinstance(array, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the argument must be an array")
    o = {}
    for x in opts:
        if not (isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed")):
            raise TypeError(f"{f}: {K_input(x)} is not an option")
        o[opt_name(x)] = x.args[1]
    rank = 1
    rows = [array]
    if array.args and K.is_list(array.args[0]):
        rank = 2
        if array.args[0].args and K.is_list(array.args[0].args[0]):
            raise NotImplementedError(f"{f}: rank-3 arrays are not supported")
        rows = list(array.args)
        if len({len(r.args) for r in rows}) != 1:
            raise ValueError(f"{f}: the array must be rectangular")
    vals = [[sx(v, f, "array entry") for v in r.args] for r in rows]
    concrete(vals)
    rev = o.get("DataReversed", False)
    if rev not in (True, False):
        raise ValueError(f"{f}: DataReversed must be True or False")
    m = len(vals)
    ncol = len(vals[0])
    cellsq = []
    for i in range(m):
        for j in range(ncol):
            if not eqz(vals[i][j]):
                cellsq.append((i, j))
    if not cellsq:
        return Expr("EmptyRegion", rank)
    dr = o.get("DataRange", Symbol("Automatic"))
    sizes = [ncol] if rank == 1 else [ncol, m]
    if isinstance(dr, Symbol) and dr.name == "Automatic":
        rng = [(sp.Integer(0), sp.Integer(s)) for s in sizes]
    else:
        rng = _bounds_arg(dr if rank == 2 else (L(dr) if not K.is_list(dr.args[0]) else dr), rank, f)

    def X_(k):
        return simp(rng[0][0] + (rng[0][1] - rng[0][0]) * k / sizes[0])

    def Y_(k):
        return simp(rng[1][0] + (rng[1][1] - rng[1][0]) * k / sizes[1])
    coords, cells = [], []
    index = {}

    def vid(key, v):
        if key not in index:
            index[key] = len(coords)
            coords.append(v)
        return index[key]
    for i, j in cellsq:
        if rank == 1:
            cells.append(("Line", (vid((j,), [X_(j)]), vid((j + 1,), [X_(j + 1)]))))
        else:
            r0 = (i if rev else m - 1 - i)
            c = [(j, r0), (j + 1, r0), (j + 1, r0 + 1), (j, r0 + 1)]
            cells.append(("Polygon", tuple(vid(k, [X_(k[0]), Y_(k[1])]) for k in c)))
    keep = [x for x in opts if opt_name(x) not in ("DataRange", "DataReversed")]
    e = mesh_expr(coords, cells, "MeshRegion", (array,) + tuple(opts))
    return Expr("MeshRegion", *e.args, *keep)


def discretize(r, x, f, boundary):
    """Exact mesh of a polytope region object (no approximation)."""
    ins = (x,)
    if isinstance(r, W.OrientFreeR):
        r.need_orient()
    if isinstance(r, EmptyR):
        return Expr("EmptyRegion", r.edim)
    if isinstance(r, PointsR):
        if boundary:
            raise ValueError(f"{f}: a point set has no full-dimensional boundary mesh")
        P = _dedupe(r.P)
        return mesh_expr(P, [("Point", (i,)) for i in range(len(P))], "MeshRegion", ins)
    if isinstance(r, W.BoxR):
        V = r.verts()
        if r.dim != r.edim:
            hr = W.convex_hull_region(_dedupe(V), f)
            return discretize(hr, x, f, boundary)
        if r.edim == 1:
            P = [r.lo, r.hi]
            return mesh_expr(P, [("Point", (0,)), ("Point", (1,))], "BoundaryMeshRegion", ins) if boundary else \
                mesh_expr(P, [("Line", (0, 1))], "MeshRegion", ins)
        if r.edim == 2:
            P = [[r.lo[0], r.lo[1]], [r.hi[0], r.lo[1]], [r.hi[0], r.hi[1]], [r.lo[0], r.hi[1]]]
            if boundary:
                return Expr("BoundaryMeshRegion", out(P, *ins), L(Expr("Line", L(1, 2, 3, 4, 1))))
            return mesh_expr(P, [("Polygon", (0, 1, 2, 3))], "MeshRegion", ins)
        if r.edim == 3:
            Vh, Fh = G.polyhedron_data(Expr("Cuboid", out(r.lo), out(r.hi)), f)
            if boundary:
                return mesh_expr(Vh, [("Polygon", tuple(fc)) for fc in Fh], "BoundaryMeshRegion", ins)
            return mesh_expr(Vh, [("Hexahedron", tuple(range(8)))], "MeshRegion", ins)
    if isinstance(r, SimplexR):
        P = []
        cells = []

        def vid(v):
            for j, q in enumerate(P):
                if is_zero_vec(vsub(q, v)):
                    return j
            P.append(v)
            return len(P) - 1
        for s in r.S:
            idx = tuple(vid(v) for v in s)
            cells.append(({1: "Line", 2: "Triangle", 3: "Tetrahedron"}[len(s) - 1], idx))
        if boundary:
            m = Mesh(P, cells, False, None, f)
            return boundary_mesh_expr(m, f)
        return mesh_expr(P, cells, "MeshRegion", ins)
    if isinstance(r, (Polygon2R, Polygon3R)):
        P, cells = [], []

        def vid(v):
            for j, q in enumerate(P):
                if is_zero_vec(vsub(q, v)):
                    return j
            P.append(v)
            return len(P) - 1
        if boundary:
            if isinstance(r, Polygon3R):
                raise ValueError(f"{f}: a planar polygon in 3-D has no full-dimensional boundary mesh")
            for o, hs in r.C:
                for ring in [G._orient_ring(o, True)] + [G._orient_ring(h, False) for h in hs]:
                    ids = [vid(v) for v in ring]
                    cells.append(("Line", tuple(ids) + (ids[0],)))
            return Expr("BoundaryMeshRegion", out(P, *ins), L(*[Expr("Line", L(*[i + 1 for i in idx]))
                                                                for _, idx in cells]))
        for o, hs in r.C:
            if hs:
                raise NotImplementedError(f"{f}: meshing polygons with holes is not supported")
            ring = G._orient_ring(o, True) if r.edim == 2 else o
            cells.append(("Polygon", tuple(vid(v) for v in ring)))
        return mesh_expr(P, cells, "MeshRegion", ins)
    if isinstance(r, PolyhedronR):
        if r.voids:
            raise NotImplementedError(f"{f}: polyhedra with voids are not supported")
        if boundary:
            return mesh_expr(r.V, [("Polygon", tuple(fc)) for fc in r.F], "BoundaryMeshRegion", ins)
        if not r.convex:
            raise NotImplementedError(f"{f}: tetrahedralizing non-convex polyhedra is not supported")
        o = 0
        cells = []
        for T in r.tris:
            ids = [next(j for j, q in enumerate(r.V) if is_zero_vec(vsub(q, v))) for v in T]
            if o in ids:
                continue
            a, b, c = ids
            cells.append(("Tetrahedron", (o, a, c, b) if sgn(dot(_cross3(vsub(r.V[a], r.V[o]), vsub(r.V[c], r.V[o])),
                                                                  vsub(r.V[b], r.V[o]))) > 0 else (o, a, b, c)))
        return mesh_expr(r.V, cells, "MeshRegion", ins)
    if isinstance(r, W.ParallelotopeR):
        V = r.verts()
        hr = W.convex_hull_region(_dedupe(V), f)
        return discretize(hr, x, f, boundary)
    if isinstance(r, UnionCellsR) and isinstance(x, Expr) and x.head in ("MeshRegion", "BoundaryMeshRegion"):
        return boundary_mesh_expr(mesh_of(x, f), f) if boundary else x
    raise NotImplementedError(f"{f}: exact discretization of {r.kind} regions is not supported (curved regions "
                              f"need Wolfram's approximation scheme)")


def _disc_common(f, reg, rest, boundary):
    a, opts = split_trailing_options(list(rest))
    if isinstance(reg, Expr) and reg.head in ("MeshRegion", "BoundaryMeshRegion") and not a:
        m = mesh_of(reg, f)
        if boundary:
            return reg if m.boundary else boundary_mesh_expr(m, f)
        return reg
    r = region_of(reg, f)
    if a:
        B = _bounds_arg(a[0], r.edim, f)
        box = W.BoxR([b[0] for b in B], [b[1] for b in B], f=f)
        r = W.BoolR("Intersection", [r, box], f=f)
        if r.res is None:
            raise NotImplementedError(f"{f}: cannot intersect this region with the bounds exactly")
        r = r.res
    elif not r.bounded:
        raise ValueError(f"{f}: the region is unbounded; give bounds {{{{xmin, xmax}}, ...}}")
    return discretize(r, reg, f, boundary)


@fn("DiscretizeRegion", area=AREA, status="added",
    notes="DiscretizeRegion[reg] gives an exact MeshRegion of a piecewise-linear region: Point cells for point "
          "sets, Line cells for segments, the polygon itself as one Polygon cell (2-D, counterclockwise), "
          "Triangle / Tetrahedron simplices, one Hexahedron for a Cuboid, a vertex-fan tetrahedralization of "
          "convex polyhedra; DiscretizeRegion[reg, {{xmin, xmax}, ...}] intersects with the box first (exact "
          "for half-spaces and convex polytopes). Curved regions (Disk, Ball, ...), polygons with holes and "
          "non-convex polyhedra raise NotImplementedError (Wolfram's approximation / meshing scheme is not "
          "documented); unbounded regions without bounds ValueError. Check: exact; tests compare the measure of "
          "the mesh with the region's.")
def _discretize_region(reg: Expr, *rest: MatrixOf[Real] | Option) -> Expr:
    return _disc_common("DiscretizeRegion", reg, rest, False)


@fn("BoundaryDiscretizeRegion", area=AREA, status="added",
    notes="BoundaryDiscretizeRegion[reg] gives an exact BoundaryMeshRegion of a full-dimensional "
          "piecewise-linear region: end points in 1-D, closed Line loops in 2-D (outer counterclockwise, holes "
          "clockwise), outward Polygon faces in 3-D; with bounds the region is first intersected with the box. "
          "Curved regions raise NotImplementedError, lower-dimensional regions ValueError. Check: exact; tests "
          "compare Area / Volume of the boundary mesh region with the region's.")
def _boundary_discretize_region(reg: Expr, *rest: MatrixOf[Real] | Option) -> Expr:
    return _disc_common("BoundaryDiscretizeRegion", reg, rest, True)


def _graphics_prims(g, f):
    if not (isinstance(g, Expr) and g.head in ("Graphics", "Graphics3D")):
        if _looks(g):
            return [g]
        if isinstance(g, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected Graphics[...] or Graphics3D[...], not {K_input(g)}")
    prims = []

    def walk(x):
        if K.is_list(x):
            for y in x.args:
                walk(y)
        elif isinstance(x, Expr) and isinstance(x.head, str) and x.head in W.REGION_HEADS:
            prims.append(x)
        elif isinstance(x, Expr) and x.head in ("Style", "Annotation", "Tooltip"):
            walk(x.args[0])
        # directives (colours, EdgeForm, ...) are ignored
    if g.args:
        walk(g.args[0])
    if not prims:
        raise ValueError(f"{f}: the graphics contain no geometric primitives")
    return prims


def _looks(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head in W.REGION_HEADS


def _disc_graphics(f, g, boundary):
    prims = _graphics_prims(g, f)
    if len(prims) == 1:
        return _disc_common("BoundaryDiscretizeRegion" if boundary else "DiscretizeRegion", prims[0], (), boundary)
    rs = [region_of(p, f) for p in prims]
    for a, b in itertools.combinations(rs, 2):
        if W.disjoint(a, b) is not True:
            raise NotImplementedError(f"{f}: overlapping or touching primitives are not supported (their union "
                                      f"needs Boolean mesh operations)")
    meshes = [mesh_of(discretize(r, p, f, boundary), f) for r, p in zip(rs, prims)]
    P, cells = [], []
    for m in meshes:
        off = len(P)
        P += m.P
        cells += [(h, tuple(i + off for i in idx)) for h, idx in m.cells]
    return mesh_expr(P, cells, "BoundaryMeshRegion" if boundary else "MeshRegion", (g,))


@fn("DiscretizeGraphics", area=AREA, status="added",
    notes="DiscretizeGraphics[g] gives a MeshRegion of the geometric primitives in Graphics / Graphics3D g (or a "
          "single primitive), directives ignored, each primitive discretized exactly as in DiscretizeRegion; "
          "several primitives must be proved disjoint (else NotImplementedError: their union needs Boolean "
          "mesh operations). Curved primitives raise NotImplementedError; graphics without primitives "
          "ValueError; non-graphics TypeError. Check: exact; tests compare the mesh measure with the sum of "
          "primitive measures.")
def _discretize_graphics(g: Expr, *opts: Option) -> Expr:
    return _disc_graphics("DiscretizeGraphics", g, False)


@fn("BoundaryDiscretizeGraphics", area=AREA, status="added",
    notes="BoundaryDiscretizeGraphics[g] gives a BoundaryMeshRegion of the full-dimensional primitives in g "
          "(as BoundaryDiscretizeRegion, disjoint primitives combined). Curved primitives raise "
          "NotImplementedError. Check: exact; tests compare the enclosed measure with the primitives'.")
def _boundary_discretize_graphics(g: Expr, *opts: Option) -> Expr:
    return _disc_graphics("BoundaryDiscretizeGraphics", g, True)


# ───────────────────────────── inert option / annotation symbols ─────────────────────────────

def _inert(name, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    return _opt


for _nm, _txt in [
    ("MaxCellMeasure", "Option of DiscretizeRegion / mesh generators bounding the measure of each cell (a number, "
                       "{d -> m, ...} or Automatic)"),
    ("MeshCellHighlight", "Option of MeshRegion / BoundaryMeshRegion giving highlighting for cells "
                          "({cellspec -> style, ...})"),
    ("MeshCellLabel", "Option of MeshRegion / BoundaryMeshRegion giving labels for cells ({cellspec -> label, ...})"),
    ("MeshCellMarker", "Option of MeshRegion / BoundaryMeshRegion giving integer markers for cells"),
    ("MeshCellShapeFunction", "Option of MeshRegion / BoundaryMeshRegion giving functions to draw cells"),
    ("MeshCellStyle", "Option of MeshRegion / BoundaryMeshRegion giving styles for cells "
                      "({cellspec -> style, ...})"),
    ("MeshQualityGoal", "Option of DiscretizeRegion and related mesh generators giving the cell quality goal "
                        "(a number between 0 and 1, \"Maximal\" or Automatic)"),
    ("MeshRefinementFunction", "Option of DiscretizeRegion giving a function f[vertices, measure] that returns "
                               "True when a cell must be refined"),
    ("NormalsFunction", "Option of plotting functions (Plot3D, ParametricPlot3D, ...) giving the vertex normals "
                        "(a function, None or Automatic)"),
    ("MeshCellMeasure", "Annotation key of MeshRegion / BoundaryMeshRegion giving the measures of cells (read "
                        "with AnnotationValue[{mreg, cellspec}, MeshCellMeasure] in the Wolfram Language; here "
                        "use RegionMeasure of MeshPrimitives)"),
    ("MeshCellCentroid", "Annotation key of MeshRegion / BoundaryMeshRegion giving the centroids of cells (read "
                         "with AnnotationValue in the Wolfram Language; here use RegionCentroid of "
                         "MeshPrimitives)"),
]:
    _inert(_nm, _txt + ". Inert as an expression; no function of this batch reads it. Check: reference "
                       "(Wolfram documentation) in tests.")


# ───────────────────────────── skipped (algorithms not documented) ─────────────────────────────

for _nm, _why in [
    ("ConcaveHullMesh", "the concave (alpha-shape) hull depends on Wolfram's undocumented automatic alpha choice"),
    ("FindMeshDefects", "Wolfram's catalogue of defect types and tolerances is not documented"),
    ("GradientFittedMesh", "the gradient-fitting reconstruction algorithm is not documented"),
    ("MeshCellQuality", "the quality measure used by the Wolfram Language is not documented"),
    ("ReconstructionMesh", "the surface reconstruction method and its parameters are not documented"),
    ("Remesh", "the remeshing algorithm and its target cell sizes are not documented"),
    ("RepairMesh", "the repair heuristics (which defects are fixed and how) are not documented"),
    ("SimplifyMesh", "the mesh decimation algorithm and its error criteria are not documented"),
    ("SmoothMesh", "the smoothing operator and its default parameters are not documented"),
    ("SubdivisionRegion", "subdivision-surface regions need Wolfram's subdivision scheme and limit evaluation"),
    ("RegionFit", "the fitting methods and their output parametrizations are not documented"),
    ("TriangulateMesh", "Wolfram's quality triangulation (Steiner points, MaxCellMeasure refinement) is not "
                        "documented, so the cells could not match"),
]:
    error(_nm, AREA, f"{_nm} is not implemented: {_why}")
