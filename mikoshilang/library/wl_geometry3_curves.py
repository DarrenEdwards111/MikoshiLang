"""Wolfram batch X06: Geometry III -- shortest curves (FindShortestCurve, ShortestCurveDistance), curve paths
(FindCurvePath), affine region matching (FindRegionTransform), sample-consensus fitting (RegionFit) and
HighlightRegion.  See wl_geometry3 for the conventions used."""
from __future__ import annotations

import heapq
import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_regions as W
from . import wl_regions_mesh as M
from . import wl_geomprims as G
from .wl_geomprims import sx, num, concrete, out, pts
from .wl_regions_xform import homog
from .wl_geometry3_util import (int_arg, real_arg, options, is_auto, is_sym, require_points, dedupe_points,
                                require_scipy, fnum, coords_out)
from .wl_geometry3 import AREA, _is_mesh, _knn, plane_fit_normals


def _np():
    import numpy as np
    return np


def _is_numeric_pt(p):
    return all(isinstance(c, sp.Basic) and c.is_number for c in p)


def _point(x, f, n, what):
    if not K.is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {what} must be a list of coordinates, not {K_input(x)}")
    p = [sx(c, f, what) for c in x.args]
    if len(p) != n:
        raise ValueError(f"{f}: {what} must have {n} coordinates (the embedding dimension of the region)")
    return p


def _norm(v):
    return sp.sqrt(sum(sp.expand(c) ** 2 for c in v))


# ═══════════════════════════════ shortest curves ═══════════════════════════════

def _boundary_segments_2d(reg, f):
    """Boundary segments [(a, b)] (exact points) of a 2-D polygonal region, or None."""
    x = reg
    if _is_mesh(x):
        m = M.mesh_of(x, f)
        if m.edim != 2 or m.top_dim() != 2 and not m.boundary:
            return None
        bm = M.mesh_of(M.boundary_mesh_expr(m, f), f) if not m.boundary else m
        segs = []
        for h, idx in bm.cells:
            if h == "Line":
                segs += [(bm.P[a], bm.P[b]) for a, b in zip(idx[:-1], idx[1:])]
        return segs
    r = W.region_of(x, f)
    if isinstance(r, W.Polygon2R):
        segs = []
        for o, hs in r.C:
            for ring in [o] + hs:
                segs += [(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))]
        return segs
    return None


def _seg_cross_params(P, Q, A, B):
    """Parameters t in [0, 1] along P->Q where it meets the segments A->B (arrays)."""
    np = _np()
    d = Q - P
    e = B - A
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    w = A - P
    ts = []
    nz = np.abs(den) > 1e-14
    t = np.where(nz, (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / np.where(nz, den, 1), np.nan)
    u = np.where(nz, (w[:, 0] * d[1] - w[:, 1] * d[0]) / np.where(nz, den, 1), np.nan)
    ok = nz & (t >= -1e-12) & (t <= 1 + 1e-12) & (u >= -1e-12) & (u <= 1 + 1e-12)
    ts += list(t[ok])
    # collinear overlaps: add the endpoints of the overlapping segments
    col = ~nz & (np.abs(w[:, 0] * d[1] - w[:, 1] * d[0]) < 1e-14)
    dd = float(d @ d)
    if dd > 0:
        for k in np.nonzero(col)[0]:
            for X in (A[k], B[k]):
                tt = float((X - P) @ d) / dd
                if -1e-12 <= tt <= 1 + 1e-12:
                    ts.append(tt)
    return ts


def _inside_closed(p, A, B):
    """p in the closed polygonal region bounded by the segments (even-odd rule; boundary counts as inside)."""
    np = _np()
    e = B - A
    w = p - A
    ee = (e * e).sum(1)
    t = np.clip((w * e).sum(1) / np.where(ee == 0, 1, ee), 0, 1)
    dist = np.linalg.norm(A + e * t[:, None] - p, axis=1)
    if dist.min() < 1e-10:
        return True
    y = p[1]
    c = ((A[:, 1] > y) != (B[:, 1] > y))
    xi = A[:, 0] + (y - A[:, 1]) * (B[:, 0] - A[:, 0]) / np.where(B[:, 1] == A[:, 1], 1, B[:, 1] - A[:, 1])
    return bool(np.sum(c & (xi > p[0])) % 2)


def _visible(P, Q, A, B):
    np = _np()
    ts = sorted({0.0, 1.0} | {min(1.0, max(0.0, t)) for t in _seg_cross_params(P, Q, A, B)})
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a < 1e-12:
            continue
        if not _inside_closed(P + (Q - P) * (a + b) / 2, A, B):
            return False
    return True


def visibility_path(segs, s, t, f):
    """Shortest path between s and t inside a closed polygonal region (visibility graph + Dijkstra; Lee 1978):
    list of exact points, or None if t is unreachable."""
    np = _np()
    A = np.array([[float(num(c)) for c in a] for a, _ in segs])
    B = np.array([[float(num(c)) for c in b] for _, b in segs])
    verts = []
    seen = set()
    for a, b in segs:
        for v in (a, b):
            k = tuple(v)
            if k not in seen:
                seen.add(k)
                verts.append(list(v))
    nodes = [s, t] + verts
    X = np.array([[float(num(c)) for c in p] for p in nodes])
    for p, q in ((X[0], s), (X[1], t)):
        if not _inside_closed(p, A, B):
            raise ValueError(f"{f}: the point {K_input(out(q))} is not in the region")
    n = len(nodes)
    dist = [math.inf] * n
    prev = [-1] * n
    dist[0] = 0.0
    h = [(0.0, 0)]
    done = [False] * n
    while h:
        d, i = heapq.heappop(h)
        if done[i]:
            continue
        done[i] = True
        if i == 1:
            break
        for j in range(n):
            if done[j] or j == i:
                continue
            nd = d + float(np.linalg.norm(X[j] - X[i]))
            if nd < dist[j] - 1e-15 and _visible(X[i], X[j], A, B):
                dist[j] = nd
                prev[j] = i
                heapq.heappush(h, (nd, j))
    if not done[1]:
        return None
    path, k = [], 1
    while k != -1:
        path.append(nodes[k])
        k = prev[k]
    return path[::-1]


def _graph_path_1d(reg, s, t, f):
    """Shortest path along a 1-D cell complex (Line, mesh of lines, 1-D intervals)."""
    np = _np()
    if _is_mesh(reg):
        m = M.mesh_of(reg, f)
        if m.boundary and m.edim > 1:
            return None
        if m.boundary and m.edim == 1:
            P = sorted(m.P, key=lambda p: num(p[0]))
            segs = [(P[k], P[k + 1]) for k in range(0, len(P) - 1, 2)]
        else:
            if m.top_dim() != 1:
                return None
            segs = [(m.P[a], m.P[b]) for h, idx in m.cells if h == "Line" for a, b in zip(idx[:-1], idx[1:])]
    elif isinstance(reg, Expr) and reg.head == "Line":
        a, _ = split_trailing_options(list(reg.args))
        lines = pts(a[0], f) if not (K.is_list(a[0]) and a[0].args and K.is_list(a[0].args[0]) and
                                     a[0].args[0].args and K.is_list(a[0].args[0].args[0])) else None
        if lines is None:
            segs = []
            for ln in a[0].args:
                Pl = pts(ln, f)
                segs += list(zip(Pl[:-1], Pl[1:]))
        else:
            segs = list(zip(lines[:-1], lines[1:]))
    else:
        return None
    concrete(*[c for sg in segs for p in sg for c in p])
    # nodes: segment endpoints; s, t inserted into the segments containing them
    key = {}
    nodes = []

    def nid(p):
        k = tuple(sp.nsimplify(c, rational=True) if isinstance(c, sp.Float) else c for c in p)
        if k not in key:
            key[k] = len(nodes)
            nodes.append(list(p))
        return key[k]
    adj = {}

    def add(a, b):
        i, j = nid(a), nid(b)
        w = _norm([x - y for x, y in zip(a, b)])
        adj.setdefault(i, []).append((j, w))
        adj.setdefault(j, []).append((i, w))
    placed = {0: False, 1: False}
    S_, T_ = list(s), list(t)
    for a, b in segs:
        cuts = [a]
        for q, k in ((S_, 0), (T_, 1)):
            d = [y - x for x, y in zip(a, b)]
            dd = sum(c * c for c in d)
            if dd == 0:
                continue
            tt = sum((qq - x) * c for qq, x, c in zip(q, a, d)) / dd
            foot = [x + tt * c for x, c in zip(a, d)]
            if 0 <= num(tt) <= 1 and num(_norm([u - v for u, v in zip(foot, q)])) < 1e-12:
                cuts.append(q)
                placed[k] = True
        cuts.append(b)
        cuts = sorted(cuts, key=lambda p: num(sum((u - v) ** 2 for u, v in zip(p, a))))
        for u, v in zip(cuts[:-1], cuts[1:]):
            if num(_norm([x - y for x, y in zip(u, v)])) > 0:
                add(u, v)
            else:
                nid(u)
    for k, q in ((0, s), (1, t)):
        if not placed[k]:
            raise ValueError(f"{f}: the point {K_input(out(q))} is not on the region")
    i0, i1 = nid(S_), nid(T_)
    dist = {i0: sp.Integer(0)}
    prev = {i0: None}
    h = [(0.0, i0)]
    done = set()
    while h:
        d, i = heapq.heappop(h)
        if i in done:
            continue
        done.add(i)
        if i == i1:
            break
        for j, w in adj.get(i, []):
            nd = dist[i] + w
            if j not in dist or num(nd) < num(dist[j]) - 1e-15:
                dist[j] = nd
                prev[j] = i
                heapq.heappush(h, (float(num(nd)), j))
    if i1 not in done:
        return "disconnected"
    path, k = [], i1
    while k is not None:
        path.append(nodes[k])
        k = prev[k]
    return path[::-1]


def _convex_mesh(x, f):
    """True when a full-dimensional mesh / polygonal region is convex (exact supporting-plane test)."""
    try:
        r = W.region_of(x, f)
    except (NotImplementedError, ValueError):
        return False
    if r.convex:
        return True
    if not _is_mesh(x):
        return False
    m = M.mesh_of(x, f)
    if m.edim == 1 or (m.top_dim() != m.edim and not m.boundary):
        return False
    bm = m if m.boundary else M.mesh_of(M.boundary_mesh_expr(m, f), f)
    P = bm.P
    for h, idx in bm.cells:
        if bm.edim == 2 and h == "Line":
            for a, b in zip(idx[:-1], idx[1:]):
                pa, pb = P[a], P[b]
                sg = {G.fsign((pb[0] - pa[0]) * (q[1] - pa[1]) - (pb[1] - pa[1]) * (q[0] - pa[0])) for q in P}
                if 1 in sg and -1 in sg:
                    return False
        elif bm.edim == 3 and M.CELL_DIM[h] == 2:
            a, b, c = (P[i] for i in idx[:3])
            nrm = G._cross([y - x for x, y in zip(a, b)], [y - x for x, y in zip(a, c)])
            sg = {G.fsign(G._dot(nrm, [y - x for x, y in zip(a, q)])) for q in P}
            if 1 in sg and -1 in sg:
                return False
    return True


def shortest(reg, s_, t_, f):
    """('line', [points]) | ('arc2', c, r, a1, a2, length) | ('arc3', ...) | ('none',) for the shortest curve."""
    r = W.region_of(reg, f)
    n = r.edim
    s = _point(s_, f, n, "the start point")
    t = _point(t_, f, n, "the end point")
    sym = not (_is_numeric_pt(s) and _is_numeric_pt(t))
    if isinstance(r, W.SphereR) and n in (2, 3):
        c, rad = r.c, r.r
        u = [a - b for a, b in zip(s, c)]
        v = [a - b for a, b in zip(t, c)]
        if not sym:
            for q, w in ((s, u), (t, v)):
                if not W.eqz(sum(x * x for x in w) - rad ** 2):
                    raise ValueError(f"{f}: the point {K_input(out(q))} is not on the region")
        if n == 2:
            if sym:
                raise Unevaluated()
            a1 = sp.atan2(u[1], u[0])
            a2 = sp.atan2(v[1], v[0])
            d = sp.simplify(a2 - a1)
            if num(d) > math.pi:
                d = d - 2 * sp.pi
            elif num(d) <= -math.pi:
                d = d + 2 * sp.pi
            ang = sp.simplify(abs(d))
            if num(d) >= 0:
                return ("arc2", c, rad, a1, sp.simplify(a1 + d), sp.simplify(rad * ang))
            return ("arc2", c, rad, sp.simplify(a1 + d), a1, sp.simplify(rad * ang))
        cosang = sum(x * y for x, y in zip(u, v)) / rad ** 2
        ang = sp.acos(sp.simplify(cosang))
        return ("arc3", c, rad, u, v, ang, sp.simplify(rad * ang))
    if sym:
        if r.convex:
            return ("line", [s, t])
        raise Unevaluated()
    concrete(*s, *t)
    if r.convex or _convex_mesh(reg, f):
        for q in (s, t):
            if not r.member(q):
                raise ValueError(f"{f}: the point {K_input(out(q))} is not in the region")
        return ("line", [s, t])
    p1 = _graph_path_1d(reg, s, t, f)
    if p1 == "disconnected":
        return ("none",)
    if p1 is not None:
        return ("line", p1)
    if n == 2:
        segs = _boundary_segments_2d(reg, f)
        if segs is not None:
            path = visibility_path(segs, s, t, f)
            return ("none",) if path is None else ("line", path)
    raise NotImplementedError(f"{f}: shortest curves on {W.region_of(reg, f).kind} regions of this kind are not "
                              "supported (convex regions, circles, spheres, 1-D cell complexes and 2-D polygonal "
                              "regions are)")


def _length(res):
    if res[0] == "line":
        P = res[1]
        return sp.simplify(sum((_norm([a - b for a, b in zip(p, q)]) for p, q in zip(P[:-1], P[1:])),
                               sp.Integer(0)))
    if res[0] == "none":
        return sp.oo
    return res[-1]


_SHORT_NOTE = ("Regions: convex regions (Disk, Ball, Rectangle, Cuboid, Triangle, convex polygons, polyhedra and "
               "meshes; symbolic end points allowed): the segment; Circle and Sphere: the shorter great-circle "
               "arc (antipodal points: the arc through the plane of the first coordinate axis not parallel to "
               "them); 1-D cell complexes (Line, meshes of lines, 1-D BoundaryMeshRegions): Dijkstra on the "
               "segment graph; 2-D polygonal regions with holes (Polygon, MeshRegion, BoundaryMeshRegion, e.g. "
               "MengerMesh[n]): the visibility-graph shortest path (Lee 1978; numerical visibility tests, exact "
               "vertices). Other regions (Annulus, Torus, curved or implicit regions, surfaces of polyhedra) raise "
               "NotImplementedError; points outside the region ValueError. ")


@fn("FindShortestCurve", area=AREA, status="added",
    notes="FindShortestCurve[reg, s, t] the shortest curve from s to t in reg: Line[{s, ..., t}] for polygonal "
          "paths, Circle[c, r, {a1, a2}] for circle arcs, ParametricRegion[{c + r (Cos[u] e1 + Sin[u] e2), "
          "0 <= u <= theta}, {u}] for great-circle arcs on spheres. " + _SHORT_NOTE + "Points in different "
          "components raise ValueError. Options (AccuracyGoal, PrecisionGoal, WorkingPrecision, PerformanceGoal, "
          "Assumptions, GenerateConditions) are accepted and do not change the (exact) result. Check: exact; "
          "tests compare with Wolfram's documented results (Circle[] quarter arc, the polyline path, the square "
          "with a square hole) and ArcLength of the result with ShortestCurveDistance.")
def _find_shortest_curve(reg: Expr, s: ListOf[Real], t: ListOf[Real], *opts: Option) -> Expr:
    f = "FindShortestCurve"
    options(opts, f, {"AccuracyGoal", "PrecisionGoal", "WorkingPrecision", "PerformanceGoal", "Assumptions",
                      "GenerateConditions"})
    res = shortest(reg, s, t, f)
    if res[0] == "none":
        raise ValueError(f"{f}: the points are in different connected components of the region")
    ins = (reg, s, t)
    if res[0] == "line":
        return Expr("Line", out(res[1], *ins))
    if res[0] == "arc2":
        _, c, rad, a1, a2, _ln = res
        return Expr("Circle", out(c, *ins), out(rad, *ins), L(out(a1, *ins), out(a2, *ins)))
    _, c, rad, u, v, ang, _ln = res
    e1 = [x / rad for x in u]
    w = [y - sp.cos(ang) * x for x, y in zip(u, v)]
    nw = _norm(w)
    if W.eqz(nw):
        if W.eqz(ang):
            return Expr("Point", out([a + b for a, b in zip(c, u)]))
        k = next(i for i in range(3) if not W.eqz(e1[i] ** 2 - 1)) if any(not W.eqz(e1[i] ** 2 - 1)
                                                                            for i in range(3)) else 0
        ax = [sp.Integer(1 if i == k else 0) for i in range(3)]
        w = [a - sum(x * y for x, y in zip(ax, e1)) * b for a, b in zip(ax, e1)]
        nw = _norm(w)
    e2 = [sp.simplify(x / nw) for x in w]
    uu = Symbol("u")
    su = sp.Symbol("u")
    comp = [sp.simplify(ci + rad * (sp.cos(su) * a + sp.sin(su) * b)) for ci, a, b in zip(c, e1, e2)]
    from ..math_ops import from_sympy
    return Expr("ParametricRegion", L(*[from_sympy(x) for x in comp]), L(L(uu, 0, from_sympy(ang))))


@fn("ShortestCurveDistance", area=AREA, status="added",
    notes="ShortestCurveDistance[reg, s, t] the length of the shortest curve from s to t in reg (ArcLength of "
          "FindShortestCurve; symbolic end points give formulas, e.g. ArcCos[x] on Sphere[] from {1, 0, 0}). "
          + _SHORT_NOTE + "Points in different components give Infinity. Options are accepted as in "
          "FindShortestCurve. Check: exact; tests compare with Wolfram's documented values (Sqrt[x^2 + "
          "(-1 + y)^2] on Disk[], 2 Sqrt[2] on the polyline, 0.681322 in Ball[]).")
def _shortest_curve_distance(reg: Expr, s: ListOf[Real], t: ListOf[Real], *opts: Option) -> Real:
    f = "ShortestCurveDistance"
    options(opts, f, {"AccuracyGoal", "PrecisionGoal", "WorkingPrecision", "PerformanceGoal", "Assumptions",
                      "GenerateConditions"})
    res = shortest(reg, s, t, f)
    v = _length(res)
    if v == sp.oo:
        return Symbol("Infinity")
    return out(v, reg, s, t)


# ═══════════════════════════════ FindCurvePath ═══════════════════════════════

def nn_crust_edges(P):
    """NN-crust (Dey-Kumar 1999): each point joins its nearest neighbour q and the nearest point r with
    angle(q - p, r - p) > 90 degrees."""
    np = _np()
    n = len(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    E = set()
    for i in range(n):
        q = int(np.argmin(D[i]))
        E.add((min(i, q), max(i, q)))
        v = P[q] - P[i]
        best, bd = None, np.inf
        for j in np.argsort(D[i], kind="stable"):
            j = int(j)
            if j == q or not np.isfinite(D[i, j]):
                continue
            if np.dot(P[j] - P[i], v) < 0:
                best, bd = j, D[i, j]
                break
        if best is not None:
            E.add((min(i, best), max(i, best)))
    return E


def curve_paths(n, E, D):
    """Chains from an edge set: each vertex keeps at most its two shortest edges (an edge survives only if both
    ends keep it); paths start at their smallest end index, closed curves at their smallest index and repeat
    it; isolated points are single-point curves.  Curves are listed by their first index."""
    inc = {i: [] for i in range(n)}
    for a, b in E:
        inc[a].append((D[a][b], b))
        inc[b].append((D[a][b], a))
    keep = {i: {w for _, w in sorted(inc[i])[:2]} for i in range(n)}
    adj = {i: sorted(w for w in keep[i] if i in keep[w]) for i in range(n)}
    seen, curves = set(), []
    for s in sorted(range(n), key=lambda i: (len(adj[i]) != 1, i)):
        if s in seen:
            continue
        if len(adj[s]) == 2 and any(len(adj[v]) == 1 for v in _component(adj, s)):
            continue
        chain = [s]
        seen.add(s)
        prev, cur = None, s
        while True:
            nxt = [w for w in adj[cur] if w != prev and w not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            chain.append(cur)
            seen.add(cur)
        if len(chain) > 2 and chain[0] in adj[chain[-1]]:
            chain.append(chain[0])
        curves.append(chain)
    return sorted(curves, key=lambda c: c[0])


def _component(adj, s):
    st, seen = [s], {s}
    while st:
        v = st.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                st.append(w)
    return seen


@fn("FindCurvePath", area=AREA, status="added",
    notes="FindCurvePath[{p1, p2, ...}] (2-D or 3-D points) orders the points along curves: a list of index lists "
          "(1-based), closed curves repeating their first index. Method: the NN-crust of Dey and Kumar (1999) "
          "(nearest neighbour plus nearest 'half-neighbour' at more than 90 degrees), keeping at most the two "
          "shortest edges at each point; open curves start at their lower end index, closed curves at their "
          "smallest index, proceeding to the smaller-index neighbour; curves are sorted by first index. "
          "Wolfram's method is undocumented, so for sparse or noisy samples the curves can differ. Duplicate "
          "points ValueError; fewer than 1 point ValueError. Check: exact for well-sampled curves (the "
          "NN-crust theorem: an epsilon-sample with epsilon < 1/3 is reconstructed); tests reorder shuffled "
          "circle points and an open arc.")
def _find_curve_path(points: MatrixOf[Real], *opts: Option) -> MatrixOf[Integer]:
    f = "FindCurvePath"
    np = _np()
    options(opts, f, set())
    P, _ = require_points(points, f, (2, 3), lo=1)
    U, idx = dedupe_points(P)
    if len(U) != len(P):
        raise ValueError(f"{f}: the points must be distinct")
    n = len(P)
    if n == 1:
        return L(L(1))
    E = nn_crust_edges(P)
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    curves = curve_paths(n, E, D)
    return L(*[L(*[i + 1 for i in c]) for c in curves])


# ═══════════════════════════════ HighlightRegion ═══════════════════════════════

_WRAPPERS = ("Style", "Labeled", "Tooltip", "Button", "PopupWindow", "Hyperlink", "StatusArea", "EventHandler",
             "Annotation", "Mouseover")


def _unwrap_roi(x):
    while isinstance(x, Expr) and x.head in _WRAPPERS and x.args:
        x = x.args[0]
    return x


@fn("HighlightRegion", area=AREA, status="added",
    notes="HighlightRegion[reg, roi] (roi a region or list of regions, optionally in Style, Labeled, Tooltip, "
          "Button, PopupWindow, ... wrappers) is validated and kept as an inert region expression: region "
          "functions treat it as reg (RegionMeasure, RegionMember, ... see W22), display is a front-end "
          "matter. Both must be regions of the same embedding dimension (TypeError / ValueError otherwise); "
          "symbolic arguments stay unevaluated. Region options are kept. Check: exact; tests verify Area and "
          "RegionMember through the wrapper.")
def _highlight_region(reg: Expr, roi: Expr | ListOf[Expr], *opts: Option) -> Expr:
    f = "HighlightRegion"
    r = W.region_of(reg, f)
    items = roi.args if K.is_list(roi) else [roi]
    for it in items:
        x = _unwrap_roi(it)
        q = W.region_of(x, f)
        if q.edim != r.edim:
            raise ValueError(f"{f}: the highlighted region must have the embedding dimension {r.edim} of the "
                             "region")
    raise Unevaluated()


def _hl_converter(x, f):
    if not x.args:
        raise ValueError(f"{f}: HighlightRegion needs a region")
    return W.region_of(x.args[0], f)


W.CONVERTERS["HighlightRegion"] = _hl_converter
W.REGION_HEADS.add("HighlightRegion")


# ═══════════════════════════════ FindRegionTransform ═══════════════════════════════

def _key(p):
    return tuple(sp.nsimplify(c) if isinstance(c, sp.Float) else sp.simplify(c) for c in p)


def _structure(r, x, f):
    """(vertex list, cells as tuples of vertex indices or None) describing a polytope region exactly."""
    if isinstance(r, W.PointsR):
        return [list(p) for p in W._nz_count(r.P)], None
    if isinstance(r, W.Polygon2R):
        V, cells = [], []
        for o, hs in r.C:
            for ring in [o] + hs:
                i0 = len(V)
                V += [list(p) for p in ring]
                cells.append(("ring", tuple(range(i0, len(V)))))
        return V, cells
    if isinstance(r, W.PolyhedronR):
        return [list(p) for p in r.V], [("set", tuple(fc)) for fc in r.F]
    if isinstance(r, W.SimplexR):
        V, cells = [], []
        for s in r.S:
            i0 = len(V)
            V += [list(p) for p in s]
            cells.append(("set", tuple(range(i0, len(V)))))
        return V, cells
    if isinstance(r, W.UnionCellsR):
        V, cells = [], []
        for c in r.cells:
            v, cs = _structure(c, None, f)
            i0 = len(V)
            V += v
            cells += [(k, tuple(i0 + i for i in t)) for k, t in (cs or [("set", tuple(range(len(v))))])]
        return V, cells
    V = r.verts() if r.convex else None
    if V is None:
        raise NotImplementedError(f"{f}: affine matching of {r.kind} regions is not supported (points, polygons, "
                                  "polyhedra, boxes, simplices, meshes, balls, disks and ellipsoids are)")
    return [list(p) for p in W._nz_count(V)], None


def _canon_cells(V, cells):
    """Exact comparison key of a structure: frozensets of cells (rings up to rotation / reversal)."""
    keys = [_key(p) for p in V]
    if cells is None:
        return frozenset(keys)
    res = []
    for kind, t in cells:
        pts_ = [keys[i] for i in t]
        if kind == "ring":
            n = len(pts_)
            cands = []
            for seq in (pts_, pts_[::-1]):
                for s in range(n):
                    cands.append(tuple(seq[s:] + seq[:s]))
            res.append(("ring", min(cands, key=lambda c: str(c))))
        else:
            res.append(("set", frozenset(pts_)))
    return frozenset(res)


def _affine_from(Pb, Qb):
    """Exact affine map (A, b) with A p_i + b = q_i for affinely independent p_0..p_d (d = dimension)."""
    d = len(Pb[0])
    Pm = sp.Matrix([[c for c in p] + [1] for p in Pb])
    Qm = sp.Matrix([list(q) for q in Qb])
    X = Pm.solve(Qm)  # (d+1) x d
    A = X[:d, :].T
    b = [X[d, i] for i in range(d)]
    return A, b


def _apply(A, b, p):
    v = A * sp.Matrix(p)
    return [sp.simplify(v[i] + b[i]) for i in range(len(b))]


def _similarity_2pts(p0, p1, q0, q1):
    """2-D similarity z -> a z + c mapping p0 -> q0, p1 -> q1 (complex arithmetic)."""
    zp0, zp1 = p0[0] + sp.I * p0[1], p1[0] + sp.I * p1[1]
    zq0, zq1 = q0[0] + sp.I * q0[1], q1[0] + sp.I * q1[1]
    a = sp.simplify((zq1 - zq0) / (zp1 - zp0))
    c = sp.simplify(zq0 - a * zp0)
    ar, ai = sp.re(a), sp.im(a)
    return sp.Matrix([[ar, -ai], [ai, ar]]), [sp.re(c), sp.im(c)]


def _ellipsoid_parts(r):
    if isinstance(r, W.EllipsoidR):
        return sp.Matrix(r.c), r.Lm
    return None


def find_transform(x1, x2, f):
    np = _np()
    if isinstance(x1, Expr) and isinstance(x2, Expr) and x1.head == x2.head == "Point" and \
            len(x1.args) == len(x2.args) == 1:
        a1, a2 = x1.args[0], x2.args[0]
        P = pts(a1, f) if K.is_list(a1) and a1.args and K.is_list(a1.args[0]) else [pts(L(a1), f)[0]]
        Q = pts(a2, f) if K.is_list(a2) and a2.args and K.is_list(a2.args[0]) else [pts(L(a2), f)[0]]
        if len(P) != len(Q) or len(P[0]) != len(Q[0]):
            raise ValueError(f"{f}: the point sets must have the same size and dimension")
        if len(P) == 1:
            return _points_map(P, Q, f)
    if K.is_list(x1) or K.is_list(x2):
        if not (K.is_list(x1) and K.is_list(x2)):
            raise TypeError(f"{f}: give two point lists or two regions")
        P = pts(x1, f)
        Q = pts(x2, f)
        if len(P) != len(Q) or len(P[0]) != len(Q[0]):
            raise ValueError(f"{f}: the point lists must have the same length and dimension")
        return _points_map(P, Q, f)
    r1, r2 = W.region_of(x1, f), W.region_of(x2, f)
    if r1.edim != r2.edim:
        raise ValueError(f"{f}: the regions must have the same embedding dimension")
    e1, e2 = _ellipsoid_parts(r1), _ellipsoid_parts(r2)
    if e1 is not None or e2 is not None:
        if e1 is None or e2 is None:
            raise ValueError(f"{f}: no affine map takes {r1.kind} to {r2.kind}")
        (c1, L1), (c2, L2) = e1, e2
        A = (L2 * L1.inv()).applyfunc(sp.simplify)
        b = list((c2 - A * c1).applyfunc(sp.simplify))
        return A, b
    if isinstance(r1, W.SphereR) and isinstance(r2, W.SphereR):
        A = sp.eye(r1.edim) * (r2.r / r1.r)
        b = [c2 - A[0, 0] * c1 for c1, c2 in zip(r1.c, r2.c)]
        return A, b
    if isinstance(r1, W.PointsR) and isinstance(r2, W.PointsR) and len(r1.P) == len(r2.P):
        return _points_map(r1.P, r2.P, f)
    V1, C1 = _structure(r1, x1, f)
    V2, C2 = _structure(r2, x2, f)
    cv1 = bool(r1.convex) or C1 is None or _convex_mesh(x1, f)
    cv2 = bool(r2.convex) or C2 is None or _convex_mesh(x2, f)
    if cv1 != cv2:
        raise ValueError(f"{f}: the regions are not affinely equivalent (only one is convex)")
    if cv1 and W.rdim(r1, f) == W.rdim(r2, f) == r1.edim:
        # a convex polytope is determined by its vertices: compare vertex sets
        V1 = [list(p) for p in W._nz_count(r1.verts())]
        V2 = [list(p) for p in W._nz_count(r2.verts())]
        C1 = C2 = None
    if len(V1) != len(V2):
        raise ValueError(f"{f}: the regions are not affinely equivalent (different vertex counts)")
    if len(V1) > 16:
        raise NotImplementedError(f"{f}: affine matching of polytopes with more than 16 vertices is not supported")
    concrete(*[c for p in V1 + V2 for c in p])
    d = r1.edim
    # an affinely independent base of V1
    base = [0]
    for i in range(1, len(V1)):
        cand = base + [i]
        if W.affine_rank([V1[j] for j in cand]) == len(cand) - 1:
            base = cand
        if len(base) == d + 1:
            break
    if len(base) != d + 1:
        raise NotImplementedError(f"{f}: lower-dimensional polytopes are not supported")
    F1 = np.array([[float(num(c)) for c in p] for p in V1])
    F2 = np.array([[float(num(c)) for c in p] for p in V2])
    target = _canon_cells(V2, C2)
    target_f = sorted(map(tuple, np.round(F2, 9)))
    Pb = F1[base]
    Ph = np.hstack([Pb, np.ones((d + 1, 1))])
    for perm in itertools.permutations(range(len(V2)), d + 1):
        Qb = F2[list(perm)]
        try:
            X = np.linalg.solve(Ph, Qb)
        except np.linalg.LinAlgError:
            continue
        if abs(np.linalg.det(X[:d, :].T)) < 1e-12:
            continue
        img = F1 @ X[:d, :] + X[d, :]
        if sorted(map(tuple, np.round(img, 9))) != target_f:
            continue
        A, b = _affine_from([V1[i] for i in base], [V2[j] for j in perm])
        if W.eqz(A.det()):
            continue
        imgV = [_apply(A, b, p) for p in V1]
        if _canon_cells(imgV, C1) == target:
            return A, b
    raise ValueError(f"{f}: the regions are not affinely equivalent")


def _points_map(P, Q, f):
    d = len(P[0])
    n = len(P)
    rk = W.affine_rank(P) if all(_is_numeric_pt(p) for p in P) else min(n - 1, d)
    if n == 1:
        return sp.eye(d), [q - p for p, q in zip(P[0], Q[0])]
    if rk == d and n >= d + 1:
        base = [0]
        for i in range(1, n):
            if W.affine_rank([P[j] for j in base + [i]]) == len(base):
                base.append(i)
            if len(base) == d + 1:
                break
        A, b = _affine_from([P[i] for i in base], [Q[i] for i in base])
    elif d == 2 and rk == 1:
        A, b = _similarity_2pts(P[0], P[1], Q[0], Q[1])
    else:
        raise NotImplementedError(f"{f}: point sets of lower affine rank are supported only for one point and "
                                  "for collinear points in 2-D")
    for p, q in zip(P, Q):
        if not all(W.eqz(a - c) for a, c in zip(_apply(A, b, p), q)):
            raise ValueError(f"{f}: no affine map takes the first point list to the second")
    return A, b


@fn("FindRegionTransform", area=AREA, status="added",
    notes="FindRegionTransform[reg1, reg2] a TransformationFunction of an affine map taking reg1 onto reg2. "
          "Point lists {p1, ...} -> {q1, ...}: the map with p_i -> q_i (exact affine map for affinely spanning "
          "points, a translation for one point (symbolic coordinates allowed), the similarity z -> a z + c for "
          "collinear 2-D points; inconsistent lists ValueError). Balls, disks and ellipsoids (any dimension): "
          "x -> c2 + L2 L1^-1 (x - c1) with L the ellipsoid matrices; spheres / circles: a scaling and "
          "translation. Polytopes (Point sets, Polygon, Triangle, Rectangle, Cuboid, Parallelogram, Simplex, "
          "Polyhedron, MeshRegion, BoundaryMeshRegion; at most 16 vertices): the first vertex correspondence "
          "(of an affinely independent base of reg1's vertices to ordered vertex tuples of reg2) whose image "
          "has exactly reg2's cells (rings up to rotation and reversal). When several affine maps exist "
          "(symmetries) Wolfram's choice is undocumented, so the matrix can differ from Wolfram's while the "
          "image is the same. Regions that are not affinely equivalent raise ValueError, other region kinds "
          "NotImplementedError. WorkingPrecision is accepted (results are exact for exact input). Check: "
          "self-check that the map takes reg1's vertices / centre data onto reg2's (built into the search); "
          "tests apply the result and compare with reg2.")
def _find_region_transform(reg1: Expr | MatrixOf[Real], reg2: Expr | MatrixOf[Real], *opts: Option) -> Expr:
    f = "FindRegionTransform"
    options(opts, f, {"WorkingPrecision"})
    A, b = find_transform(reg1, reg2, f)
    M_ = homog(sp.Matrix(A), b).applyfunc(sp.simplify)
    return Expr("TransformationFunction", out([[M_[i, j] for j in range(M_.shape[1])] for i in range(M_.shape[0])],
                                              reg1, reg2))


# ═══════════════════════════════ RegionFit ═══════════════════════════════

_MODELS = {"Line": (2, (2, 3)), "Circle": (3, (2,)), "Plane": (3, (3,)), "Sphere": (4, (3,)),
           "Cylinder": (2, (3,)), "Cone": (3, (3,))}
_PROPS = ("BestFit", "Points", "Inliers", "Outliers", "DistanceVariance", "Distances")


def _unit(v):
    np = _np()
    n = np.linalg.norm(v)
    return None if n == 0 else v / n


def _model_min(kind, S, N):
    """Model parameters from a minimal sample (or None if degenerate)."""
    np = _np()
    if kind == "Line":
        d = _unit(S[1] - S[0])
        return None if d is None else ("Line", S[0], d)
    if kind == "Circle":
        a, b, c = S
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-14:
            return None
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        cc = np.array([ux, uy])
        return ("Circle", cc, float(np.linalg.norm(a - cc)))
    if kind == "Plane":
        n = _unit(np.cross(S[1] - S[0], S[2] - S[0]))
        return None if n is None else ("Plane", S.mean(axis=0), n)
    if kind == "Sphere":
        A = 2 * (S[1:] - S[0])
        rhs = (S[1:] ** 2).sum(1) - (S[0] ** 2).sum()
        if abs(np.linalg.det(A)) < 1e-12:
            return None
        c = np.linalg.solve(A, rhs)
        return ("Sphere", c, float(np.linalg.norm(S[0] - c)))
    if kind == "Cylinder":
        n1, n2 = N
        a = _unit(np.cross(n1, n2))
        if a is None:
            return None
        # closest points of the normal lines p_i + t n_i
        p1, p2 = S
        w = p1 - p2
        b = n1 @ n2
        den = 1 - b * b
        if den < 1e-12:
            return None
        t1 = (b * (n2 @ w) - (n1 @ w)) / den
        c = p1 + t1 * n1
        r = float(np.linalg.norm(np.cross(p1 - c, a)))
        return ("Cylinder", c, a, r)
    if kind == "Cone":
        # apex: intersection of the three tangent planes n_i . (x - p_i) = 0
        Nm = np.array(N)
        if abs(np.linalg.det(Nm)) < 1e-12:
            return None
        apex = np.linalg.solve(Nm, (Nm * S).sum(1))
        U = [_unit(p - apex) for p in S]
        if any(u is None for u in U):
            return None
        ax = _unit(np.cross(U[1] - U[0], U[2] - U[0]))
        if ax is None:
            return None
        if ax @ (S.mean(axis=0) - apex) < 0:
            ax = -ax
        th = float(np.mean([math.acos(max(-1.0, min(1.0, u @ ax))) for u in U]))
        if not 0 < th < math.pi / 2:
            return None
        return ("Cone", apex, ax, th)
    raise AssertionError(kind)


def _dist(model, P):
    np = _np()
    k = model[0]
    if k == "Line":
        _, p, d = model
        w = P - p
        return np.linalg.norm(w - np.outer(w @ d, d), axis=1)
    if k in ("Circle", "Sphere"):
        _, c, r = model
        return np.abs(np.linalg.norm(P - c, axis=1) - r)
    if k == "Plane":
        _, c, n = model
        return np.abs((P - c) @ n)
    if k == "Cylinder":
        _, c, a, r = model
        w = P - c
        return np.abs(np.linalg.norm(w - np.outer(w @ a, a), axis=1) - r)
    if k == "Cone":
        _, apex, a, th = model
        w = P - apex
        h = w @ a
        rho = np.linalg.norm(w - np.outer(h, a), axis=1)
        d = np.abs(rho * math.cos(th) - h * math.sin(th))
        behind = h * math.cos(th) + rho * math.sin(th) < 0
        return np.where(behind, np.linalg.norm(w, axis=1), d)
    raise AssertionError(k)


def _refit(model, P, N):
    """Least-squares refit on the inliers: PCA for lines / planes, geometric (Gauss-Newton via SciPy
    least_squares when available) for circles, spheres, cylinders and cones."""
    np = _np()
    k = model[0]
    if len(P) == 0:
        return model
    if k == "Line" and len(P) >= 2:
        c = P.mean(axis=0)
        _, _, Vt = np.linalg.svd(P - c)
        d = Vt[0]
        return ("Line", c, d)
    if k == "Plane" and len(P) >= 3:
        c = P.mean(axis=0)
        _, _, Vt = np.linalg.svd(P - c)
        n = Vt[-1]
        if N is not None:
            if n @ N.mean(axis=0) < 0:
                n = -n
        elif n @ model[2] < 0:
            n = -n
        return ("Plane", c, n)
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return model
    if k in ("Circle", "Sphere"):
        x0 = np.append(model[1], model[2])
        res = least_squares(lambda x: np.linalg.norm(P - x[:-1], axis=1) - x[-1], x0)
        return (k, res.x[:-1], float(abs(res.x[-1])))
    if k == "Cylinder":
        _, c, a, r = model

        def fun(x):
            aa = _unit(x[3:6])
            w = P - x[:3]
            return np.linalg.norm(w - np.outer(w @ aa, aa), axis=1) - x[6]
        res = least_squares(fun, np.concatenate([c, a, [r]]))
        return ("Cylinder", res.x[:3], _unit(res.x[3:6]), float(abs(res.x[6])))
    if k == "Cone":
        _, apex, a, th = model

        def fun(x):
            return _dist(("Cone", x[:3], _unit(x[3:6]), x[6]), P)
        res = least_squares(fun, np.concatenate([apex, a, [th]]))
        return ("Cone", res.x[:3], _unit(res.x[3:6]), float(res.x[6]))
    return model


def _model_expr(model, P):
    np = _np()
    k = model[0]
    v = lambda a: L(*[fnum(c) for c in a])  # noqa: E731
    if k == "Line":
        _, c, d = model
        t = (P - c) @ d
        q0, q1 = c + d * t[0], c + d * t[1] if len(t) > 1 and abs(t[1] - t[0]) > 0 else c + d
        return Expr("InfiniteLine", L(v(q0), v(q1)))
    if k == "Circle":
        return Expr("Circle", v(model[1]), fnum(model[2]))
    if k == "Sphere":
        return Expr("Sphere", v(model[1]), fnum(model[2]))
    if k == "Plane":
        return Expr("Hyperplane", v(model[2]), v(model[1]))
    if k == "Cylinder":
        _, c, a, r = model
        t = (P - c) @ a
        return Expr("Cylinder", L(v(c + a * t.min()), v(c + a * t.max())), fnum(r))
    if k == "Cone":
        _, apex, a, th = model
        h = float(((P - apex) @ a).max())
        return Expr("Cone", L(v(apex + a * h), v(apex)), fnum(h * math.tan(th)))
    raise AssertionError(k)


def ransac(P, N, kind, tol, method, conf, f):
    np = _np()
    m, _ = _MODELS[kind]
    n = len(P)
    if n < m:
        raise ValueError(f"{f}: a {kind} needs at least {m} points")
    rng = G._np_rng()
    total = math.comb(n, m)
    exhaustive = total <= 3000
    samples = itertools.combinations(range(n), m) if exhaustive else None
    best, bscore = None, None
    it, max_it = 0, 3000
    while True:
        if exhaustive:
            s = next(samples, None)
            if s is None:
                break
        else:
            if it >= max_it:
                break
            s = tuple(sorted(rng.choice(n, m, replace=False)))
        it += 1
        mod = _model_min(kind, P[list(s)], None if N is None else N[list(s)])
        if mod is None:
            continue
        d = _dist(mod, P)
        if method == "RANSAC":
            score = (-(d <= tol).sum(), float(np.minimum(d, tol).sum()))
        elif method == "MSAC":
            score = (float(np.minimum(d ** 2, tol ** 2).sum()),)
        else:  # LMEDS
            score = (float(np.median(d ** 2)),)
        if bscore is None or score < bscore:
            best, bscore = mod, score
            if not exhaustive and method != "LMEDS":
                w = (d <= tol).mean()
                if 0 < w < 1:
                    need = math.log(1 - conf) / math.log(1 - w ** m)
                    max_it = min(max_it, max(int(math.ceil(need)), 50))
                elif w == 1:
                    break
    if best is None:
        raise ValueError(f"{f}: the points are degenerate for a {kind} fit")
    if method == "LMEDS":
        d = _dist(best, P)
        sigma = 1.4826 * (1 + 5 / max(n - m, 1)) * math.sqrt(float(np.median(d ** 2)))
        tol = max(2.5 * sigma, tol if tol else 0.0)
    inl = _dist(best, P) <= tol
    ref = _refit(best, P[inl], None if N is None else N[inl])
    inl2 = _dist(ref, P) <= tol
    if inl2.sum() >= inl.sum():
        best, inl = ref, inl2
    return best, inl


@fn("RegionFit", area=AREA, status="added",
    notes="RegionFit[pts, model] and RegionFit[pts, model, prop] sample-consensus fitting of \"Line\" (2-D, 3-D; "
          "InfiniteLine through the projections of the first two points), \"Circle\" (2-D; Circle), \"Plane\" "
          "(Hyperplane[n, c] with c the inlier centroid), \"Sphere\", \"Cylinder\" (Cylinder spanning the inliers "
          "along the axis) and \"Cone\" (Cone[{base centre, apex}, r] reaching the farthest inlier). Minimal "
          "samples: 2, 3, 3, 4 points; cylinders 2 and cones 3 points with normals (VertexNormals in Point[...], "
          "else estimated by plane fitting). All C(n, m) samples are tried when there are at most 3000 of them, "
          "otherwise seeded random samples (SeedRandom) with the adaptive RANSAC stopping rule for "
          "ConfidenceLevel (default 0.99, at most 3000 samples). Method: \"RANSAC\" (default; most inliers, ties "
          "by truncated distance sum), \"MSAC\" (truncated squared loss) or \"LMEDS\" (least median of squares, "
          "inliers within 2.5 robust sigma); the others raise NotImplementedError. Tolerance (inlier distance) "
          "Automatic = 0.02 times the bounding-box diagonal. The best model is refit to its inliers (PCA for "
          "lines and planes, geometric least squares with SciPy for the curved models) when that keeps at least "
          "as many inliers. Properties: \"BestFit\", \"Points\", \"Inliers\", \"Outliers\", \"Distances\", "
          "\"DistanceVariance\" (Variance of the distances) or a list. 3-D \"Circle\" raises NotImplementedError. "
          "Wolfram's tolerance and sampling are undocumented, so results on noisy data can differ (exact data "
          "give the exact model). Check: self-check that every inlier is within the tolerance of the returned "
          "model (ArithmeticError otherwise); tests fit exact data with outliers.")
def _region_fit(points: MatrixOf[Real] | Expr, model: String, *rest: String | ListOf[String] | Option) -> Expr:
    f = "RegionFit"
    np = _np()
    pos, o, _k = options(rest, f, {"ConfidenceLevel", "Method", "PerformanceGoal", "Tolerance", "VertexNormals",
                                   "WorkingPrecision"})
    if len(pos) > 1:
        raise ValueError(f"{f}: expected RegionFit[points, model, prop]")
    if not isinstance(model, str):
        if isinstance(model, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the model must be a string such as \"Line\"")
    if model not in _MODELS:
        raise ValueError(f"{f}: unknown model \"{model}\" (known: {', '.join(_MODELS)})")
    x = points
    if K.is_list(x) and x.args and all(isinstance(p, Expr) and p.head == "Point" for p in x.args):
        x = L(*[p.args[0] for p in x.args])
    P, N = require_points(x, f, (2, 3), lo=1)
    d = P.shape[1]
    if model == "Circle" and d == 3:
        raise NotImplementedError(f"{f}: 3-D circle fitting is not supported")
    if d not in _MODELS[model][1]:
        raise ValueError(f"{f}: the \"{model}\" model needs {' or '.join(map(str, _MODELS[model][1]))}-D points")
    vn = o.get("VertexNormals")
    if vn is not None and not is_sym(vn, "None") and not is_auto(vn):
        N = normals_arg_(vn, len(P), d, f)
    if model in ("Cylinder", "Cone"):
        if N is None:
            N = plane_fit_normals(P)
        require_scipy(f)
    meth = o.get("Method")
    meth = "RANSAC" if is_auto(meth) else meth
    if meth not in ("RANSAC", "MSAC", "LMEDS"):
        if meth in ("RRANSAC", "RMSAC", "MLESAC", "PROSAC"):
            raise NotImplementedError(f"{f}: Method -> \"{meth}\" is not supported")
        raise ValueError(f"{f}: unknown Method {K_input(meth)}")
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    tol = 0.02 * diag if is_auto(o.get("Tolerance")) else real_arg(o["Tolerance"], f, "Tolerance", 0)
    conf = 0.99 if is_auto(o.get("ConfidenceLevel")) else real_arg(o["ConfidenceLevel"], f, "ConfidenceLevel", 0,
                                                                   1, True, True)
    best, inl = ransac(P, N, model, tol, meth, conf, f)
    D = _dist(best, P)
    if meth != "LMEDS" and np.any(D[inl] > tol * (1 + 1e-9)):
        raise ArithmeticError(f"{f}: self-check failed (an inlier is not within the tolerance)")
    props = pos[0] if pos else "BestFit"

    def prop(p):
        if p == "BestFit":
            return _model_expr(best, P[inl] if inl.sum() >= 2 else P)
        if p == "Points":
            return coords_out(P)
        if p == "Inliers":
            return coords_out(P[inl])
        if p == "Outliers":
            return coords_out(P[~inl])
        if p == "Distances":
            return L(*[fnum(v) for v in D])
        if p == "DistanceVariance":
            return fnum(np.var(D, ddof=1)) if len(D) > 1 else 0.0
        if isinstance(p, str):
            raise ValueError(f"{f}: unknown property \"{p}\" (known: {', '.join(_PROPS)})")
        raise TypeError(f"{f}: the property must be a string")
    if K.is_list(props):
        return L(*[prop(p) for p in props.args])
    return prop(props)


def normals_arg_(v, n, d, f):
    from .wl_geometry3_util import normals_arg
    return normals_arg(v, n, d, f)
