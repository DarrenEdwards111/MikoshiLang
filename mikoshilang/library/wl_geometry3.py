"""Wolfram batch X06: Geometry III -- mesh processing (ConcaveHullMesh, SubdivisionRegion, SmoothMesh, SimplifyMesh,
Remesh, TriangulateMesh, RepairMesh, FindMeshDefects, ReconstructionMesh, GradientFittedMesh, EstimatedPointNormals,
MeshCellQuality, MeshConnectivityGraph, ListSurfacePlot3D).  The 12 mesh heads W22 reserved as errors are
implemented here (their error registrations in wl_regions_mesh were removed).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  Each head uses an explicit published method, named in its notes together with the default
parameters; where Wolfram's own defaults are undocumented the results are those of the stated method and can differ
from Wolfram's (for example in vertex placement or cell order).  Numerical results are machine reals.
"""
from __future__ import annotations

import heapq
import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, REGISTRY, STATUS, record
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_regions_mesh as M
from . import wl_geomprims as G
from .wl_geometry3_util import (int_arg, real_arg, options, is_auto, is_sym, mesh_np, mesh_out, coords_out, compact,
                                dedupe_points, require_scipy, require_points, normals_arg, fnum, MESH_PRESENTATION)

AREA = "geometry: mesh processing (X06)"


def _np():
    import numpy as np
    return np


def _method(v, f, allowed, subopts):
    """(method name, {suboption: value}) from Method -> "name" | {"name", "sub" -> v, ...} | Automatic."""
    if is_auto(v):
        return None, {}
    sub = {}
    if K.is_list(v) and v.args and isinstance(v.args[0], str):
        name = v.args[0]
        for r in v.args[1:]:
            n = opt_name(r)
            if n not in subopts:
                raise ValueError(f"{f}: {K_input(r)} is not a known suboption of Method -> \"{name}\"")
            sub[n] = r.args[1]
    elif isinstance(v, str):
        name = v
    else:
        raise ValueError(f"{f}: Method must be Automatic, a method name or {{name, opt -> value, ...}}")
    if name not in allowed:
        raise NotImplementedError(f"{f}: Method -> \"{name}\" is not supported (supported: "
                                  f"{', '.join(repr(a) for a in allowed)})")
    return name, sub


def _sub_int(sub, key, dflt, f, lo=1):
    return dflt if key not in sub or is_auto(sub[key]) else int_arg(sub[key], f, key, lo)


def _sub_real(sub, key, dflt, f, lo=0.0):
    return dflt if key not in sub or is_auto(sub[key]) else real_arg(sub[key], f, key, lo, lo_open=True)


def _is_mesh(x):
    return isinstance(x, Expr) and x.head in ("MeshRegion", "BoundaryMeshRegion")


def _mesh_arg(x, f):
    if not _is_mesh(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {K_input(x)} is not a MeshRegion or BoundaryMeshRegion")
    return mesh_np(x, f)


def _top_cells(m):
    top = m.top_dim()
    return top, [(h, idx) for h, idx in m.cells if M.CELL_DIM[h] == top]


# ═══════════════════════════════ MeshConnectivityGraph ═══════════════════════════════

def _cells_of_dim(m, d):
    a = m.all_cells()
    if d not in a:
        return []
    return [frozenset(idx) for _, idx in a[d]]


@fn("MeshConnectivityGraph", area=AREA, status="added",
    notes="MeshConnectivityGraph[mr] (= [mr, 0]), MeshConnectivityGraph[mr, d], MeshConnectivityGraph[mr, d, r] "
          "and MeshConnectivityGraph[mr, {d, e}, r]: a Graph whose vertices {d, i} are the cells of dimension d "
          "(index i as in MeshCells[mr, d]); [mr, d] joins cells sharing a (d - 1)-cell (points sharing a line for "
          "d = 0); [mr, {d, e}, r] has a directed edge {d, i} -> {e, j} when some r-cell is a subset or superset "
          "of both cells (Wolfram's definition), [mr, d, r] is its undirected version (no self-loops). Derived "
          "lower-dimensional cells are numbered in order of first appearance (W22), which may differ from "
          "Wolfram's numbering. Graph options are kept. Dimensions outside 0..3 raise ValueError; non-meshes "
          "TypeError. Check: exact (combinatorial); tests compare edge counts with hand counts (square split in "
          "two triangles: 5 edges between its 4 points; 2 triangles sharing an edge).")
def _mesh_connectivity_graph(mr: Expr, *rest: Integer | ListOf[Integer] | Option) -> Expr:
    f = "MeshConnectivityGraph"
    pos, gopts = split_trailing_options(list(rest))
    m, _ = _mesh_arg(mr, f)
    if len(pos) > 2:
        raise ValueError(f"{f}: expected MeshConnectivityGraph[mr, d] or MeshConnectivityGraph[mr, {{d, e}}, r]")
    spec = pos[0] if pos else 0
    if K.is_list(spec):
        if len(spec.args) != 2 or len(pos) != 2:
            raise ValueError(f"{f}: {{d, e}} needs a third argument r")
        d, e = (int_arg(t, f, "the cell dimension", 0, 3) for t in spec.args)
        r = int_arg(pos[1], f, "the shared dimension", 0, 3)
        directed = True
    else:
        d = int_arg(spec, f, "the cell dimension", 0, 3)
        e = d
        if len(pos) == 2:
            r = int_arg(pos[1], f, "the shared dimension", 0, 3)
        else:
            r = 1 if d == 0 else d - 1
        directed = False
    A, B, C = _cells_of_dim(m, d), _cells_of_dim(m, e), _cells_of_dim(m, r)
    verts = [L(d, i + 1) for i in range(len(A))]
    if (d, e) != (d, d) or e != d:
        verts += [L(e, j + 1) for j in range(len(B))]
    rel = {}
    for k, c in enumerate(C):
        rel[k] = c
    # for each r-cell, the d-cells and e-cells related (subset or superset)
    def related(cells, c):
        return [i for i, a in enumerate(cells) if c <= a or a <= c]
    edges = set()
    for c in C:
        ra, rb = related(A, c), related(B, c)
        for i in ra:
            for j in rb:
                if d == e and i == j:
                    continue
                if directed:
                    edges.add((i, j))
                else:
                    edges.add((min(i, j), max(i, j)))
    head = "DirectedEdge" if directed else "UndirectedEdge"
    E = [Expr(head, L(d, i + 1), L(e, j + 1)) for i, j in sorted(edges)]
    return Expr("Graph", L(*verts), L(*E), *gopts)


# ═══════════════════════════════ MeshCellQuality ═══════════════════════════════

def _quality(P, h, idx, f):
    np = _np()
    V = P[list(idx)]
    d = M.CELL_DIM[h]
    if d <= 1:
        return 1.0
    if h == "Triangle" or (h == "Polygon" and len(idx) == 3):
        a, b, c = V[0], V[1], V[2]
        if len(a) == 2:
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
        else:
            area = np.linalg.norm(np.cross(b - a, c - a)) / 2
        s = sum(np.sum((p - q) ** 2) for p, q in ((a, b), (b, c), (c, a)))
        return 0.0 if s == 0 else float(4 * math.sqrt(3) * area / s)
    if h == "Tetrahedron":
        a, b, c, e = V
        vol = abs(np.dot(b - a, np.cross(c - a, e - a))) / 6
        l2 = [np.sum((V[i] - V[j]) ** 2) for i, j in itertools.combinations(range(4), 2)]
        lrms = math.sqrt(sum(l2) / 6)
        return 0.0 if lrms == 0 else float(6 * math.sqrt(2) * vol / lrms ** 3)
    raise NotImplementedError(f"{f}: MeshCellQuality of {h} cells with {len(idx)} vertices is not supported "
                              "(only points, lines, triangles and tetrahedra)")


def mesh_cell_quality(mreg, spec, f="AnnotationValue"):
    """MeshCellQuality values for {mreg, spec} (spec as in MeshCells)."""
    m, P = mesh_np(mreg, f)
    a = m.all_cells()
    return [_quality(P, *a[d][i - 1], f) for d, i in M.cell_spec(m, spec, f)]


@fn("MeshCellQuality", area=AREA, status="added",
    notes="MeshCellQuality is the annotation key read with AnnotationValue[{mreg, cellspec}, MeshCellQuality] "
          "(cellspec as in MeshCells: d, {d, i}, {d, All}, {{d, i}, ...} or explicit cells). Values: 1 for points "
          "and lines; 4 Sqrt[3] A / (l1^2 + l2^2 + l3^2) for triangles; 6 Sqrt[2] V / l_rms^3 for tetrahedra (1 for "
          "regular, 0 for degenerate cells). These formulas reproduce every value in Wolfram's documentation "
          "(0.866025, 0.968246, 0.918559). Polygons with more than 3 vertices and hexahedra / prisms / pyramids "
          "raise NotImplementedError. As a symbol MeshCellQuality is inert. Check: reference (Wolfram "
          "documentation values in tests) and exact values 1 for an equilateral triangle and a regular "
          "tetrahedron.")
def _mesh_cell_quality(*args: Expr) -> Expr:
    raise Unevaluated()


def _install_annotation_hook():
    e = REGISTRY.get("AnnotationValue")
    if e is None or getattr(e, "_x06", False):
        return
    old = list(e.impls)

    def _annotation_value(spec: Expr, key: Expr) -> Expr:
        if K.is_list(spec) and len(spec.args) == 2 and _is_mesh(spec.args[0]):
            keys = key.args if K.is_list(key) else [key]
            res = []
            for k in keys:
                if not is_sym(k, "MeshCellQuality"):
                    raise NotImplementedError(f"AnnotationValue: only MeshCellQuality is supported for mesh "
                                              f"regions (got {K_input(k)})")
                v = mesh_cell_quality(spec.args[0], spec.args[1])
                res.append(_spec_shape(spec.args[1], v))
            return L(*res) if K.is_list(key) else res[0]
        for fn_, lo, hi in old:
            if lo <= 2 and (hi is None or hi >= 2):
                return fn_(spec, key)
        raise NotImplementedError("AnnotationValue: unsupported call")
    from .core import _arity
    e.impls = [(_annotation_value, *_arity(_annotation_value))]
    e._x06 = True
    area, status, notes = STATUS["AnnotationValue"]
    record("AnnotationValue", area, status, notes.replace(
        " Check:", " AnnotationValue[{mreg, cellspec}, MeshCellQuality] gives mesh cell qualities (X06, see "
                   "MeshCellQuality). Check:", 1))


def _spec_shape(spec, vals):
    """A single cell spec {d, i} / explicit cell gives a number, others a list."""
    if K.is_list(spec) and len(spec.args) == 2 and all(K.is_int(t) for t in spec.args):
        return vals[0]
    if isinstance(spec, Expr) and isinstance(spec.head, str) and spec.head in M.CELL_DIM and \
            not (spec.args and K.is_list(spec.args[0]) and spec.args[0].args and K.is_list(spec.args[0].args[0])):
        return vals[0]
    return L(*vals)


_install_annotation_hook()


# ═══════════════════════════════ alpha complexes: ConcaveHullMesh ═══════════════════════════════

def _circumradius2(Q):
    """Squared radius and centre of the smallest circumsphere of the simplex with vertex rows Q (k+1 points in
    R^n, k <= n)."""
    np = _np()
    a = Q[0]
    A = Q[1:] - a
    if len(A) == 0:
        return 0.0, a.copy()
    G_ = A @ A.T
    rhs = 0.5 * np.einsum("ij,ij->i", A, A)
    try:
        lam = np.linalg.solve(G_, rhs)
    except np.linalg.LinAlgError:
        return math.inf, a.copy()
    c = a + lam @ A
    return float(np.sum((c - a) ** 2)), c


def alpha_complex(P, f):
    """Delaunay complex of P with alpha filtration values (squared radii of the smallest empty circumscribing
    balls): {k: {frozenset(simplex): value}} for k = 0..n (Edelsbrunner-Muecke; GUDHI's attached-face rule)."""
    np = _np()
    n = P.shape[1]
    if n == 1:
        order = np.argsort(P[:, 0], kind="stable")
        cx = {0: {frozenset([int(i)]): 0.0 for i in range(len(P))}, 1: {}}
        for a, b in zip(order[:-1], order[1:]):
            cx[1][frozenset([int(a), int(b)])] = float((P[b, 0] - P[a, 0]) / 2) ** 2
        return cx
    require_scipy(f)
    from scipy.spatial import Delaunay
    try:
        tri = Delaunay(P)
    except Exception as e:
        raise ValueError(f"{f}: the points cannot be triangulated ({e.__class__.__name__}); they may be degenerate "
                         f"(collinear / coplanar)")
    cx = {k: {} for k in range(n + 1)}
    for s in tri.simplices:
        S_ = frozenset(int(i) for i in s)
        r2, _ = _circumradius2(P[sorted(S_)])
        if r2 == math.inf:
            continue
        cx[n][S_] = r2
    for k in range(n - 1, -1, -1):
        for tau, ft in cx[k + 1].items():
            for v in tau:
                sig = tau - {v}
                if k == 0:
                    cx[0].setdefault(sig, 0.0)
                    continue
                r2, c = _circumradius2(P[sorted(sig)])
                attached = float(np.sum((P[v] - c) ** 2)) < r2 * (1 - 1e-12)
                cur = cx[k].get(sig)
                if attached:
                    cx[k][sig] = ft if cur is None else min(cur, ft)
                elif cur is None:
                    cx[k][sig] = r2
                else:
                    cx[k][sig] = min(cur, max(r2, 0.0)) if cur > r2 else cur
    for i in range(len(P)):
        cx[0].setdefault(frozenset([i]), 0.0)
    return cx


def _components_ok(cells, k, npts):
    """True when the k-cells use every point and are connected through shared (k-1)-faces (vertices for k = 1)."""
    if not cells:
        return False
    if len({i for c in cells for i in c}) != npts:
        return False
    if k == 0:
        return npts == 1
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    seen = {}
    for ci, c in enumerate(cells):
        faces = [frozenset([v]) for v in c] if k == 1 else [c - {v} for v in c]
        for fc in faces:
            if fc in seen:
                parent[find(ci)] = find(seen[fc])
            else:
                seen[fc] = ci
    return len({find(i) for i in range(len(cells))}) == 1


def default_alpha(cx, k, npts):
    """Smallest alpha whose k-cells cover all points in one component (CGAL's optimal alpha for one solid
    component); the largest value when no alpha achieves it."""
    vals = sorted(set(cx[k].values()))
    if not vals:
        return 0.0
    lo, hi = 0, len(vals) - 1
    if not _components_ok([c for c, v in cx[k].items() if v <= vals[hi]], k, npts):
        return math.sqrt(vals[hi])
    while lo < hi:
        mid = (lo + hi) // 2
        if _components_ok([c for c, v in cx[k].items() if v <= vals[mid]], k, npts):
            hi = mid
        else:
            lo = mid + 1
    return math.sqrt(vals[lo])


_SIMPLEX_HEAD = {0: "Point", 1: "Line", 2: "Triangle", 3: "Tetrahedron"}


def _oriented(P, idx):
    """Simplex vertex order with positive orientation for full-dimensional simplices."""
    np = _np()
    idx = sorted(idx)
    n = P.shape[1]
    if len(idx) == n + 1 and n >= 2:
        D = np.linalg.det(P[idx[1:]] - P[idx[0]])
        if D < 0:
            idx[1], idx[2] = idx[2], idx[1]
    return tuple(idx)


def _points_or_point(x, f, dims=(1, 2, 3)):
    if isinstance(x, Expr) and x.head == "Point":
        A, _ = require_points(x, f, dims)
        return A
    A, _ = require_points(x, f, dims)
    return A


@fn("ConcaveHullMesh", area=AREA, status="added",
    notes="ConcaveHullMesh[pts], [pts, alpha], [pts, alpha, d] (pts a list of 1-, 2- or 3-D points or Point[...]): "
          "the alpha shape (Edelsbrunner-Muecke). Cells of the Delaunay mesh are kept when the smallest empty ball "
          "through their vertices has radius <= alpha (alpha-complex filtration with GUDHI's attached-face rule); "
          "the result is the MeshRegion of the kept d-cells (d defaults to the embedding dimension; d = All gives "
          "all maximal kept cells). Default alpha (Automatic): the smallest alpha whose d-cells use every point "
          "and form one connected component (CGAL's optimal alpha; Wolfram's automatic choice is undocumented, so "
          "the default shape can differ from Wolfram's). Coordinates are the used points in input order; "
          "duplicates are removed. 2-D and 3-D need SciPy (Qhull Delaunay; NotImplementedError 'requires SciPy' "
          "otherwise); degenerate point sets ValueError; alpha <= 0 ValueError; no cell within alpha ValueError; "
          "symbolic input stays unevaluated. Check: exact (filtration by construction); tests compare with the "
          "convex hull for large alpha (area of the Delaunay mesh), hand-computed 1-D alpha shapes, and the "
          "documented definition (every kept triangle has circumradius <= alpha).")
def _concave_hull_mesh(points: MatrixOf[Real] | Expr, *rest: Real | Integer | Expr | Option) -> Expr:
    f = "ConcaveHullMesh"
    pos, _o, keep = options(rest, f, set(), MESH_PRESENTATION)
    if len(pos) > 2:
        raise ValueError(f"{f}: expected ConcaveHullMesh[pts, alpha, d]")
    P = _points_or_point(points, f)
    P, _ = dedupe_points(P)
    n = P.shape[1]
    if len(P) < 2:
        raise ValueError(f"{f}: at least two distinct points are needed")
    d = n
    if len(pos) == 2:
        d = "All" if is_sym(pos[1], "All") else int_arg(pos[1], f, "the cell dimension", 0, n)
    cx = alpha_complex(P, f)
    kdim = n if d == "All" else d
    if len(pos) >= 1 and not is_auto(pos[0]):
        alpha = real_arg(pos[0], f, "alpha", 0, lo_open=True)
    else:
        alpha = default_alpha(cx, kdim, len(P))
    a2 = alpha * alpha * (1 + 1e-12)
    if d == "All":
        kept = {k: [s for s, v in cx[k].items() if v <= a2] for k in cx}
        cells = []
        for k in range(n, -1, -1):
            for s in kept[k]:
                if any(s < t for t in kept.get(k + 1, [])):
                    continue
                cells.append((k, s))
    else:
        cells = [(d, s) for s, v in cx[d].items() if v <= a2]
    if not cells:
        raise ValueError(f"{f}: no {d}-dimensional cell lies within alpha = {alpha}")
    cells.sort(key=lambda t: (-t[0], sorted(t[1])))
    out = [(_SIMPLEX_HEAD[k], _oriented(P, s)) for k, s in cells]
    Q, out = compact(list(P), out)
    return mesh_out(Q, out, opts=keep)


# ═══════════════════════════════ shared mesh topology helpers ═══════════════════════════════

def to_mesh(x, f, boundary=False):
    """Mesh expression for a MeshRegion / BoundaryMeshRegion or an exactly discretizable region (polygons,
    polyhedra, boxes, simplices); curved regions raise NotImplementedError (via DiscretizeRegion)."""
    if _is_mesh(x):
        return x
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not isinstance(x, Expr):
        raise TypeError(f"{f}: {K_input(x)} is not a region")
    return M._disc_common(f, x, (), boundary)


def edge_faces(faces):
    """{frozenset edge: [face indices]} for polygon faces (index tuples)."""
    E = {}
    for fi, fc in enumerate(faces):
        n = len(fc)
        for i in range(n):
            E.setdefault(frozenset((fc[i], fc[(i + 1) % n])), []).append(fi)
    return E


def _manifold(E, f):
    for e, fl_ in E.items():
        if len(fl_) > 2:
            raise ValueError(f"{f}: the mesh is not manifold (an edge is shared by {len(fl_)} faces)")


def polylines(edges, f):
    """Chains of a 1-D cell complex: list of (vertex list, closed?).  Branch points raise NotImplementedError."""
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    if any(len(v) > 2 for v in adj.values()):
        raise NotImplementedError(f"{f}: curves with branch points (a vertex on more than two lines) are not "
                                  "supported")
    seen, out_ = set(), []
    starts = [v for v in adj if len(adj[v]) == 1] + list(adj)
    for s in starts:
        if s in seen:
            continue
        chain, prev, cur = [s], None, s
        seen.add(s)
        while True:
            nxt = [w for w in adj[cur] if w != prev and w not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        closed = len(chain) > 2 and chain[0] in adj[chain[-1]] and len(adj[s]) == 2
        out_.append((chain, closed))
    return out_


def _orient_chain(chain, closed, edges):
    """Keep the direction of the input lines (reverse the chain if its first edge runs backwards)."""
    if len(chain) < 2:
        return chain
    dirs = set(edges)
    if (chain[0], chain[1]) not in dirs and (chain[1], chain[0]) in dirs:
        return chain[::-1] if not closed else [chain[0]] + chain[1:][::-1]
    return chain


def _line_edges(m):
    E = []
    for h, idx in m.cells:
        if h == "Line":
            E += list(zip(idx[:-1], idx[1:]))
        elif M.CELL_DIM[h] != 0:
            return None
    return E


# ═══════════════════════════════ SubdivisionRegion ═══════════════════════════════

def chaikin(Pl, closed, n):
    np = _np()
    Q = np.array(Pl, dtype=float)
    for _ in range(n):
        if closed:
            Rn = np.roll(Q, -1, axis=0)
            new = np.empty((2 * len(Q), Q.shape[1]))
            new[0::2] = 0.75 * Q + 0.25 * Rn
            new[1::2] = 0.25 * Q + 0.75 * Rn
            Q = new
        elif len(Q) > 2:
            a, b = Q[:-1], Q[1:]
            q, r = 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b
            mid = np.empty((2 * len(a), Q.shape[1]))
            mid[0::2], mid[1::2] = q, r
            Q = np.vstack([Q[:1], mid[1:-1], Q[-1:]])
    return Q


def loop_subdivide(P, faces, f):
    """One step of Loop subdivision (Loop 1987) with the standard boundary rules (crease curves = cubic
    B-spline)."""
    np = _np()
    E = edge_faces(faces)
    _manifold(E, f)
    nV = len(P)
    nb = {i: set() for i in range(nV)}
    bnb = {i: set() for i in range(nV)}
    for e, fl_ in E.items():
        a, b = tuple(e)
        nb[a].add(b)
        nb[b].add(a)
        if len(fl_) == 1:
            bnb[a].add(b)
            bnb[b].add(a)
    newP = list(np.zeros_like(P))
    for v in range(nV):
        if bnb[v]:
            if len(bnb[v]) == 2:
                b1, b2 = bnb[v]
                newP[v] = 0.75 * P[v] + 0.125 * (P[b1] + P[b2])
            else:
                newP[v] = P[v].copy()
        elif nb[v]:
            k = len(nb[v])
            beta = (5 / 8 - (3 / 8 + 0.25 * math.cos(2 * math.pi / k)) ** 2) / k
            newP[v] = (1 - k * beta) * P[v] + beta * sum(P[w] for w in nb[v])
        else:
            newP[v] = P[v].copy()
    eid = {}
    for e, fl_ in E.items():
        a, b = tuple(e)
        if len(fl_) == 2:
            opp = [next(w for w in faces[fi] if w not in e) for fi in fl_]
            p = 0.375 * (P[a] + P[b]) + 0.125 * (P[opp[0]] + P[opp[1]])
        else:
            p = 0.5 * (P[a] + P[b])
        eid[e] = len(newP)
        newP.append(p)
    nf = []
    for a, b, c in faces:
        ab, bc, ca = eid[frozenset((a, b))], eid[frozenset((b, c))], eid[frozenset((c, a))]
        nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(newP), nf


def catmull_clark(P, faces, f):
    """One step of Catmull-Clark subdivision (1978) with boundary edges as cubic B-spline creases."""
    np = _np()
    E = edge_faces(faces)
    _manifold(E, f)
    nV = len(P)
    F = [P[list(fc)].mean(axis=0) for fc in faces]
    newP = []
    vfaces = {i: [] for i in range(nV)}
    for fi, fc in enumerate(faces):
        for v in fc:
            vfaces[v].append(fi)
    vedges = {i: [] for i in range(nV)}
    for e in E:
        for v in e:
            vedges[v].append(e)
    for v in range(nV):
        be = [e for e in vedges[v] if len(E[e]) == 1]
        if be:
            if len(be) == 2:
                w = [next(iter(e - {v})) for e in be]
                newP.append(0.75 * P[v] + 0.125 * (P[w[0]] + P[w[1]]))
            else:
                newP.append(P[v].copy())
        elif vfaces[v]:
            n = len(vfaces[v])
            Fa = sum(F[fi] for fi in vfaces[v]) / n
            Ra = sum((P[a] + P[b]) / 2 for a, b in (tuple(e) for e in vedges[v])) / len(vedges[v])
            newP.append((Fa + 2 * Ra + (n - 3) * P[v]) / n)
        else:
            newP.append(P[v].copy())
    eid = {}
    for e, fl_ in E.items():
        a, b = tuple(e)
        if len(fl_) == 2:
            p = (P[a] + P[b] + F[fl_[0]] + F[fl_[1]]) / 4
        else:
            p = (P[a] + P[b]) / 2
        eid[e] = len(newP)
        newP.append(p)
    fid = []
    for fp in F:
        fid.append(len(newP))
        newP.append(fp)
    nf = []
    for fi, fc in enumerate(faces):
        n = len(fc)
        for i in range(n):
            v, nx, pv = fc[i], fc[(i + 1) % n], fc[i - 1]
            nf.append((v, eid[frozenset((v, nx))], fid[fi], eid[frozenset((pv, v))]))
    return np.array(newP), nf


def _euler(faces):
    V = len({v for fc in faces for v in fc})
    return V - len(edge_faces(faces)) + len(faces)


@fn("SubdivisionRegion", area=AREA, status="added",
    notes="SubdivisionRegion[mesh, n] and SubdivisionRegion[mesh, n, proc]: n steps of a subdivision scheme on a "
          "MeshRegion / BoundaryMeshRegion (or an exactly discretizable region such as Polygon, Rectangle, "
          "Polyhedron, Cuboid). Schemes: \"Loop\" (Loop 1987; triangle surfaces), \"CatmullClark\" "
          "(Catmull-Clark 1978; polygon surfaces; the final non-planar quads are returned split into two triangles along the vertex-to-face-point diagonal) and \"Chaikin\" (corner cutting 1/4-3/4; curves "
          "and the boundary loops of 2-D regions). Boundary edges of open surfaces follow the cubic B-spline crease "
          "rules (Loop: 3/4, 1/8, 1/8 and midpoints; Catmull-Clark: the same), open curves keep their end points. "
          "Defaults: Chaikin for curves and 2-D regions, Loop for all-triangle surfaces, Catmull-Clark otherwise "
          "(Wolfram's default choice is undocumented). 2-D filled regions give a BoundaryMeshRegion of the "
          "subdivided boundary loops, 3-D solids the subdivided boundary surface. SubdivisionRegion[mesh] and "
          "SubdivisionRegion[mesh, Infinity] (the limit region) are kept inert after validation; \"QuadTri\" and "
          "solid (tetrahedral) MeshRegions raise NotImplementedError; non-manifold surfaces ValueError; a "
          "scheme that does not fit the mesh ValueError. Region presentation options are kept. Check: exact "
          "(scheme by construction; the Euler characteristic of a surface is preserved and verified, "
          "ArithmeticError otherwise); tests compare one Loop step of a tetrahedron and one Chaikin step of a "
          "square with hand computations.")
def _subdivision_region(mesh: Expr, *rest: NonNegInteger | Expr | String | Option) -> Expr:
    f = "SubdivisionRegion"
    pos, _o, keep = options(rest, f, set(), MESH_PRESENTATION)
    if len(pos) > 2:
        raise ValueError(f"{f}: expected SubdivisionRegion[mesh, n, proc]")
    proc = None
    if len(pos) == 2:
        if not isinstance(pos[1], str):
            raise TypeError(f"{f}: the scheme must be a string such as \"Loop\", not {K_input(pos[1])}")
        if pos[1] == "QuadTri":
            raise NotImplementedError(f"{f}: the \"QuadTri\" scheme is not supported")
        if pos[1] not in ("Loop", "CatmullClark", "Chaikin"):
            raise ValueError(f"{f}: unknown subdivision scheme \"{pos[1]}\"")
        proc = pos[1]
    mx = to_mesh(mesh, f)
    m, P = mesh_np(mx, f)
    if not pos or is_sym(pos[0], "Infinity"):
        raise Unevaluated()
    n = int_arg(pos[0], f, "the number of subdivision steps", 0, 12)
    top = m.top_dim()
    edges = _line_edges(m)
    if top == 3:
        raise NotImplementedError(f"{f}: subdivision of solid (tetrahedral / hexahedral) meshes is not supported; "
                                  "use the BoundaryMeshRegion of the solid")
    if top == 1 or (m.edim == 2 and top == 2):
        if proc not in (None, "Chaikin"):
            raise ValueError(f"{f}: the \"{proc}\" scheme is for surfaces; curves and 2-D regions use \"Chaikin\"")
        if top == 2:
            bm = M.mesh_of(M.boundary_mesh_expr(m, f), f) if not m.boundary else m
            edges = _line_edges(bm)
            P = mesh_np(bm.expr, f)[1] if bm is not m else P
            head = "BoundaryMeshRegion"
        else:
            if edges is None:
                raise NotImplementedError(f"{f}: mixed-dimensional meshes are not supported")
            head = mx.head
        chains = polylines(edges, f)
        allp, cells = [], []
        for chain, closed in chains:
            chain = _orient_chain(chain, closed, edges)
            Q = chaikin(P[chain], closed, n)
            k0 = len(allp)
            allp += list(Q)
            ids = list(range(k0, k0 + len(Q)))
            if closed:
                ids.append(k0)
            cells.append(("Line", tuple(ids)))
        return mesh_out(allp, cells, head, keep)
    if top != 2:
        raise NotImplementedError(f"{f}: point sets cannot be subdivided")
    if proc == "Chaikin":
        raise ValueError(f"{f}: \"Chaikin\" is for curves and 2-D regions, not surfaces")
    faces = [tuple(idx) for h, idx in m.cells if M.CELL_DIM[h] == 2]
    if len(faces) != len(m.cells):
        raise NotImplementedError(f"{f}: mixed-dimensional meshes are not supported")
    tri = all(len(fc) == 3 for fc in faces)
    proc = proc or ("Loop" if tri else "CatmullClark")
    if proc == "Loop" and not tri:
        raise ValueError(f"{f}: the \"Loop\" scheme needs a triangle mesh (use \"CatmullClark\")")
    chi = _euler(faces)
    for _ in range(n):
        P, faces = (loop_subdivide if proc == "Loop" else catmull_clark)(P, faces, f)
    if _euler(faces) != chi:
        raise ArithmeticError(f"{f}: self-check failed (the Euler characteristic changed)")
    if n == 0:
        return mesh_out(P, [("Polygon" if len(fc) > 3 else "Triangle", fc) for fc in faces], mx.head, keep)
    if proc == "CatmullClark":
        # quads (v, e_next, F, e_prev) are not planar: split along the v-F diagonal
        faces = [t for v, a, c, b in faces for t in ((v, a, c), (v, c, b))]
    return mesh_out(P, [("Triangle", fc) for fc in faces], mx.head, keep)


# ═══════════════════════════════ SmoothMesh ═══════════════════════════════

def _boundary_vertices(m):
    """Vertices on the boundary of the top-dimensional cells (faces used once); none for BoundaryMeshRegions."""
    if m.boundary:
        return set()
    top, cells = _top_cells(m)
    cnt = {}
    for h, idx in cells:
        fs = [(i,) for i in idx] if top == 1 else [fi for _, fi in M.faces_of(h, idx)]
        for fc in fs:
            k = frozenset(fc)
            cnt[k] = cnt.get(k, 0) + 1
    return {v for k, c in cnt.items() if c == 1 for v in k}


def laplacian_weights(m, P, uniform):
    """Row-normalised Laplacian weights {i: [(j, w)]}: cotangent weights (clamped at 0) on triangle surfaces,
    uniform (umbrella) weights otherwise or when "UniformLaplace" -> True."""
    np = _np()
    W_ = {}
    edges = [idx for _, idx in m.all_cells().get(1, [])]
    tris = [idx for h, idx in m.cells if M.CELL_DIM[h] == 2 and len(idx) == 3]
    cot_ok = not uniform and len(tris) == len(m.cells) and tris
    if cot_ok:
        acc = {}
        for t in tris:
            for k in range(3):
                i, j, o = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
                u, v = P[i] - P[o], P[j] - P[o]
                cr = np.linalg.norm(np.cross(u, v)) if len(u) == 3 else abs(u[0] * v[1] - u[1] * v[0])
                c = 0.0 if cr == 0 else float(np.dot(u, v) / cr)
                for a, b in ((i, j), (j, i)):
                    acc[(a, b)] = acc.get((a, b), 0.0) + 0.5 * c
        for (a, b), w in acc.items():
            W_.setdefault(a, []).append((b, max(w, 0.0)))
    else:
        for a, b in edges:
            W_.setdefault(a, []).append((b, 1.0))
            W_.setdefault(b, []).append((a, 1.0))
    out_ = {}
    for i, lst in W_.items():
        s = sum(w for _, w in lst)
        if s <= 0:
            lst, s = [(j, 1.0) for j, _ in lst], float(len(lst))
        out_[i] = [(j, w / s) for j, w in lst]
    return out_


def _signs(m, P):
    np = _np()
    top, cells = _top_cells(m)
    if top != m.edim or top < 2:
        return None
    res = []
    for h, idx in cells:
        if h == "Tetrahedron":
            a, b, c, d = P[list(idx)]
            res.append(np.sign(np.dot(b - a, np.cross(c - a, d - a))))
        elif top == 2 and len(idx) == 3:
            a, b, c = P[list(idx)]
            res.append(np.sign((b - a)[0] * (c - a)[1] - (b - a)[1] * (c - a)[0]))
    return res


@fn("SmoothMesh", area=AREA, status="added",
    notes="SmoothMesh[mesh] Laplacian smoothing of a MeshRegion / BoundaryMeshRegion (curves, surfaces, 2-D and "
          "tetrahedral meshes); the cells are unchanged, the coordinates move. Method -> \"ExplicitLaplacian\" "
          "(default; x <- x + t L x) or \"ImplicitLaplacian\" (backward Euler (I - t L) x' = x, Desbrun et al. "
          "1999), with suboptions \"UniformLaplace\" (default False: cotangent weights on triangle meshes, clamped "
          "at 0; True or non-triangle meshes: uniform umbrella weights; weights are row-normalised), "
          "\"MaxIterations\" (default 1) and \"TimeStep\" (default 0.5). Vertices on the boundary of an open mesh "
          "stay fixed (all vertices of a BoundaryMeshRegion move). Wolfram's default parameters are undocumented, "
          "so the amount of smoothing can differ. The implicit method uses a dense solve (sparse with SciPy above "
          "2000 vertices; NotImplementedError 'requires SciPy' without it). Unknown suboptions ValueError; other "
          "methods NotImplementedError; non-meshes TypeError. Check: self-check that no 2-D triangle or "
          "tetrahedron is inverted (ArithmeticError otherwise); tests verify fixed boundary vertices, an "
          "interior vertex moving to the neighbour average with t = 1 and uniform weights, and invariance of a "
          "flat regular grid.")
def _smooth_mesh(mesh: Expr, *opts: Option) -> Expr:
    f = "SmoothMesh"
    np = _np()
    pos, o, keep = options(opts, f, {"Method"}, MESH_PRESENTATION - {"Method"})
    if pos:
        raise ValueError(f"{f}: expected SmoothMesh[mesh, opts]")
    m, P = _mesh_arg(mesh, f)
    meth, sub = _method(o.get("Method"), f, ("ExplicitLaplacian", "ImplicitLaplacian"),
                        {"UniformLaplace", "MaxIterations", "TimeStep"})
    meth = meth or "ExplicitLaplacian"
    uni = sub.get("UniformLaplace", Symbol("Automatic"))
    if is_auto(uni):
        uni = not (m.edim == 3 and m.top_dim() == 2)
    elif is_sym(uni, "True") or is_sym(uni, "False") or isinstance(uni, bool):
        uni = uni is True or is_sym(uni, "True")
    else:
        raise TypeError(f"{f}: \"UniformLaplace\" must be True, False or Automatic")
    it = _sub_int(sub, "MaxIterations", 1, f, 0)
    t = _sub_real(sub, "TimeStep", 0.5, f)
    fixed = _boundary_vertices(m)
    s0 = _signs(m, P)
    for _ in range(it):
        Wt = laplacian_weights(m, P, uni)
        free = [i for i in range(len(P)) if i in Wt and i not in fixed]
        if meth == "ExplicitLaplacian":
            Q = P.copy()
            for i in free:
                Q[i] = P[i] + t * (sum(w * P[j] for j, w in Wt[i]) - P[i])
            P = Q
        else:
            n = len(P)
            if n > 2000:
                require_scipy(f)
                from scipy.sparse import lil_matrix
                from scipy.sparse.linalg import spsolve
                A = lil_matrix((n, n))
            else:
                A = np.zeros((n, n))
            for i in range(n):
                A[i, i] = 1.0
            for i in free:
                A[i, i] = 1.0 + t
                for j, w in Wt[i]:
                    A[i, j] = A[i, j] - t * w
            P = spsolve(A.tocsr(), P) if n > 2000 else np.linalg.solve(A, P)
            P = P.reshape(n, -1)
    s1 = _signs(m, P)
    if s0 is not None and any(a != b for a, b in zip(s0, s1)):
        raise ArithmeticError(f"{f}: self-check failed (smoothing inverted cells; use a smaller \"TimeStep\")")
    return mesh_out(P, m.cells, mesh.head, keep)


# ═══════════════════════════════ EstimatedPointNormals ═══════════════════════════════

def _knn(P, k):
    np = _np()
    try:
        from scipy.spatial import cKDTree
        _, I = cKDTree(P).query(P, k + 1)
        return [list(r[1:]) if np.ndim(r) else [] for r in I]
    except ImportError:
        D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return [list(np.argsort(D[i], kind="stable")[1:k + 1]) for i in range(len(P))]


def orient_normals(P, N, nbrs):
    """Consistent orientation (Hoppe et al. 1992): propagate along a maximum-|n_i . n_j| spanning tree of the
    neighbour graph; each component is then flipped to point away from the centroid of all points (ties: the
    point with the largest last coordinate gets a normal with non-negative last component)."""
    np = _np()
    n = len(P)
    adj = {i: set() for i in range(n)}
    for i, lst in enumerate(nbrs):
        for j in lst:
            j = int(j)
            if j != i:
                adj[i].add(j)
                adj[j].add(i)
    seen = [False] * n
    c = P.mean(axis=0)
    for s in sorted(range(n), key=lambda i: (-P[i][-1], i)):
        if seen[s]:
            continue
        comp = [s]
        seen[s] = True
        h = [(-abs(float(np.dot(N[s], N[j]))), s, j) for j in adj[s]]
        heapq.heapify(h)
        while h:
            _, a, b = heapq.heappop(h)
            if seen[b]:
                continue
            seen[b] = True
            comp.append(b)
            if np.dot(N[a], N[b]) < 0:
                N[b] = -N[b]
            for j in adj[b]:
                if not seen[j]:
                    heapq.heappush(h, (-abs(float(np.dot(N[b], N[j]))), b, j))
        if N[s][-1] < 0:
            for i in comp:
                N[i] = -N[i]
        out_score = sum(float(np.dot(N[i], P[i] - c)) for i in comp)
        scale = sum(float(np.linalg.norm(P[i] - c)) for i in comp) or 1.0
        if out_score < -1e-9 * scale:
            for i in comp:
                N[i] = -N[i]
    return N


def plane_fit_normals(P, k=None):
    np = _np()
    n, d = P.shape
    if n < 2:
        raise ValueError("EstimatedPointNormals: at least two points are needed")
    k = min(n - 1, k or 2 * d)
    nb = _knn(P, k)
    N = np.zeros_like(P)
    for i in range(n):
        Q = P[[i] + [int(j) for j in nb[i]]]
        C = np.cov((Q - Q.mean(axis=0)).T, bias=True).reshape(d, d)
        w, V = np.linalg.eigh(C)
        N[i] = V[:, 0]
    return orient_normals(P, N, nb)


def face_weighted_normals(m, P, f):
    np = _np()
    N = np.zeros_like(P)
    top, cells = _top_cells(m)
    if m.edim == 3 and top == 2:
        for h, idx in cells:
            for k in range(1, len(idx) - 1):
                a, b, c = idx[0], idx[k], idx[k + 1]
                nv = np.cross(P[b] - P[a], P[c] - P[a])
                for v in (a, b, c):
                    N[v] += nv
    elif m.edim == 2 and top == 1:
        for h, idx in cells:
            for a, b in zip(idx[:-1], idx[1:]):
                e = P[b] - P[a]
                nv = np.array([e[1], -e[0]])
                N[a] += nv
                N[b] += nv
    else:
        raise NotImplementedError(f"{f}: \"FaceWeighted\" normals need a surface in 3D or a curve in 2D; use "
                                  "Method -> \"PlaneFitting\"")
    nn = np.linalg.norm(N, axis=1)
    if np.any(nn == 0):
        raise ValueError(f"{f}: some vertices have no well-defined normal (isolated or degenerate cells)")
    return N / nn[:, None]


@fn("EstimatedPointNormals", area=AREA, status="added",
    notes="EstimatedPointNormals[{p1, p2, ...}] (points of any dimension, or Point objects) and "
          "EstimatedPointNormals[mreg]: unit normals. Method -> \"PlaneFitting\" (default for points): principal "
          "component analysis of each point and its k = min(n - 1, 2 d) nearest neighbours (smallest "
          "eigenvector), oriented consistently along a maximum-|n_i . n_j| spanning tree of the neighbour graph "
          "(Hoppe et al. 1992), each connected part then pointing away from the centroid (ties: positive last "
          "component at the point with the largest last coordinate). Method -> \"FaceWeighted\" (default for "
          "meshes): area-weighted sum of the adjacent face normals (surfaces in 3D, curves in 2D, following the "
          "cell orientation). Wolfram's neighbourhood size and orientation rule are undocumented, so signs and "
          "values on noisy data can differ. Nearest neighbours use SciPy when available (brute force "
          "otherwise). Fewer than two points ValueError; FaceWeighted on other meshes NotImplementedError. "
          "Check: self-check of unit length (ArithmeticError otherwise); tests compare circle and sphere points "
          "with the exact outward normals and collinear points with {0, 1}.")
def _estimated_point_normals(points: MatrixOf[Real] | Expr, *opts: Option) -> MatrixOf[Real]:
    f = "EstimatedPointNormals"
    np = _np()
    pos, o, _k = options(opts, f, {"Method"})
    if pos:
        raise ValueError(f"{f}: expected EstimatedPointNormals[points, opts]")
    meth, _sub = _method(o.get("Method"), f, ("PlaneFitting", "FaceWeighted"), set())
    if _is_mesh(points):
        m, P = mesh_np(points, f)
        N = face_weighted_normals(m, P, f) if meth in (None, "FaceWeighted") else plane_fit_normals(P)
    else:
        if K.is_list(points) and points.args and all(isinstance(p, Expr) and p.head == "Point" and len(p.args) == 1
                                                     and K.is_list(p.args[0]) for p in points.args):
            points = L(*[p.args[0] for p in points.args])
        P, _ = require_points(points, f, tuple(range(1, 65)))
        if meth == "FaceWeighted":
            raise ValueError(f"{f}: \"FaceWeighted\" needs a mesh region")
        N = plane_fit_normals(P)
    if not np.allclose(np.linalg.norm(N, axis=1), 1.0, atol=1e-9):
        raise ArithmeticError(f"{f}: self-check failed (normals are not unit vectors)")
    return coords_out(N)
