"""Wolfram batch X06: Geometry III -- triangulation (TriangulateMesh) and surface reconstruction
(ReconstructionMesh, GradientFittedMesh, ListSurfacePlot3D).  See wl_geometry3 for the conventions used."""
from __future__ import annotations

import math

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from .wl_graphics_color import K_input
from . import wl_regions_mesh as M
from .wl_geometry3_util import (int_arg, real_arg, options, is_auto, is_sym, mesh_np, mesh_out, compact,
                                MESH_PRESENTATION, dedupe_points, require_scipy, require_points, normals_arg)
from .wl_geometry3 import (_mesh_arg, _method, _sub_int, _sub_real, edge_faces, AREA, alpha_complex, _oriented,
                           _SIMPLEX_HEAD, plane_fit_normals, _is_mesh)


def _np():
    import numpy as np
    return np


# ═══════════════════════════════ TriangulateMesh ═══════════════════════════════

def _loops_2d(m, P, f):
    """Boundary loops (lists of 0-based vertex indices) of a 2-D full-dimensional mesh."""
    if m.boundary:
        segs = []
        for h, idx in m.cells:
            if h == "Line":
                segs += list(zip(idx[:-1], idx[1:]))
    else:
        if m.top_dim() != 2:
            raise ValueError(f"{f}: a 2-D mesh of polygons or a BoundaryMeshRegion is needed")
        segs = [tuple(idx) for _, idx in M.boundary_cells(m, f)]
    return segs


def _seg_key(a, b):
    return (a, b) if a < b else (b, a)


def _tri_area(P, t):
    a, b, c = P[t[0]], P[t[1]], P[t[2]]
    return ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2


def _inside_even_odd(p, P, segs):
    x, y = p
    c = False
    for a, b in segs:
        (x1, y1), (x2, y2) = P[a], P[b]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                c = not c
    return c


def conforming_delaunay(P, segs, f, max_area=None, quality=True, max_points=20000):
    """Conforming Delaunay triangulation of a planar straight-line graph (segments split at midpoints until
    they are Delaunay edges), with optional Ruppert refinement (Ruppert 1995: circumcentres of triangles with
    circumradius / shortest edge > sqrt 2, i.e. minimum angle < 20.7 degrees, or area > max_area; encroached
    segments split first).  Triangles outside the region (even-odd rule on the segments) are removed."""
    np = _np()
    from scipy.spatial import Delaunay
    P = [np.array(p, dtype=float) for p in P]
    segs = {_seg_key(a, b) for a, b in segs}
    nb_in = len(P)
    # small input angles (< 60 degrees) are exempt from the quality test (Shewchuk's termination guard)
    inc = {}
    for a, b in segs:
        inc.setdefault(a, []).append(b)
        inc.setdefault(b, []).append(a)
    small = set()
    for v, ws in inc.items():
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                u, w = P[ws[i]] - P[v], P[ws[j]] - P[v]
                c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
                if c > 0.5:
                    small.add(v)
    keep = None
    for _round in range(200):
        A = np.array(P)
        tri = Delaunay(A)
        edges = set()
        for s in tri.simplices:
            for k in range(3):
                edges.add(_seg_key(int(s[k]), int(s[(k + 1) % 3])))
        missing = [s for s in segs if s not in edges]
        if missing:
            for a, b in missing:
                P.append((P[a] + P[b]) / 2)
                c = len(P) - 1
                segs.discard((a, b))
                segs.update({_seg_key(a, c), _seg_key(c, b)})
            continue
        keep = []
        for s in tri.simplices:
            t = tuple(int(i) for i in s)
            cen = A[list(t)].mean(axis=0)
            if _inside_even_odd(cen, A, segs):
                if _tri_area(A, t) < 0:
                    t = (t[0], t[2], t[1])
                keep.append(t)
        if not quality and max_area is None:
            break
        bad = []
        for t in keep:
            a, b, c = A[list(t)]
            ar = abs(_tri_area(A, t))
            ls = sorted(np.linalg.norm(x - y) for x, y in ((a, b), (b, c), (c, a)))
            R_ = ls[0] * ls[1] * ls[2] / (4 * ar) if ar > 0 else math.inf
            skinny = quality and R_ / ls[0] > math.sqrt(2) + 1e-9 and not (set(t) & small)
            if skinny or (max_area is not None and ar > max_area):
                bad.append((ar, t))
        if not bad:
            break
        if len(P) > max_points:
            raise ValueError(f"{f}: the refinement needs more than {max_points} points; use a larger "
                             "MaxCellMeasure")
        new_pts, split = [], set()
        for _, t in sorted(bad, reverse=True):
            a, b, c = A[list(t)]
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if d == 0:
                continue
            ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
            uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
            cc = np.array([ux, uy])
            enc = [s for s in segs if np.linalg.norm(cc - (A[s[0]] + A[s[1]]) / 2) <
                   np.linalg.norm(A[s[0]] - A[s[1]]) / 2 * (1 - 1e-12)]
            if enc:
                split.update(enc)
            elif _inside_even_odd(cc, A, segs):
                if all(np.linalg.norm(cc - q) > 1e-9 for q in new_pts):
                    new_pts.append(cc)
        for a, b in split:
            P.append((P[a] + P[b]) / 2)
            c = len(P) - 1
            segs.discard((a, b))
            segs.update({_seg_key(a, c), _seg_key(c, b)})
        if not split:
            P += new_pts
        if not split and not new_pts:
            break
    else:
        raise ValueError(f"{f}: the triangulation did not converge")
    return np.array(P), keep


def _max_measure(v, f):
    if v is None or is_auto(v):
        return None
    if is_sym(v, "Infinity"):
        return math.inf
    if K.is_list(v) and len(v.args) == 1 and isinstance(v.args[0], Expr) and v.args[0].head == "Rule":
        v = v.args[0].args[1]
    return real_arg(v, f, "MaxCellMeasure", 0, lo_open=True)


@fn("TriangulateMesh", area=AREA, status="added",
    notes="TriangulateMesh[mr] simplicial mesh of a mesh region. 1-D: line cells (BoundaryMeshRegion of points or "
          "MeshRegion of lines), split into equal pieces no longer than MaxCellMeasure. 2-D (BoundaryMeshRegion "
          "or MeshRegion of polygons, holes allowed): conforming Delaunay triangulation (segments split at "
          "midpoints until they are Delaunay edges; SciPy Qhull), refined by Ruppert's algorithm to minimum "
          "angle 20.7 degrees (vertices with input angles below 60 degrees exempt) and cell area <= "
          "MaxCellMeasure; MaxCellMeasure -> Infinity gives the plain conforming Delaunay triangulation. 3-D: "
          "convex BoundaryMeshRegions are tetrahedralized by the Delaunay tetrahedra of their vertices "
          "(no refinement; finite MaxCellMeasure raises NotImplementedError), non-convex solids raise "
          "NotImplementedError; polygon surface meshes are fan-triangulated per convex face. Defaults: "
          "MaxCellMeasure Automatic (quality refinement only), Method -> Automatic = \"ConformingDelaunay\"; "
          "\"ConstrainedDelaunay\", \"ConstrainedQuality\", MeshQualityGoal other than Automatic and "
          "MeshRefinementFunction other than None raise NotImplementedError; PerformanceGoal is accepted and has "
          "no effect. Wolfram's Steiner point placement is undocumented, so cells differ. 2-D and 3-D need SciPy. "
          "Check: self-check that the cell measures add up to the region's measure (ArithmeticError "
          "otherwise); tests verify areas, minimum angles and cell counts of simple polygons.")
def _triangulate_mesh(mr: Expr, *opts: Option) -> Expr:
    f = "TriangulateMesh"
    np = _np()
    pos, o, keep = options(opts, f, {"MaxCellMeasure", "MeshQualityGoal", "MeshRefinementFunction", "Method",
                                     "PerformanceGoal"}, MESH_PRESENTATION - {"Method"})
    if pos:
        raise ValueError(f"{f}: expected TriangulateMesh[mr, opts]")
    if not is_auto(o.get("MeshQualityGoal")):
        raise NotImplementedError(f"{f}: MeshQualityGoal other than Automatic is not supported")
    mrf = o.get("MeshRefinementFunction")
    if mrf is not None and not is_sym(mrf, "None"):
        raise NotImplementedError(f"{f}: MeshRefinementFunction is not supported")
    meth, _sub = _method(o.get("Method"), f, ("ConformingDelaunay", "ConstrainedDelaunay", "ConstrainedQuality"),
                         set())
    if meth not in (None, "ConformingDelaunay"):
        raise NotImplementedError(f"{f}: Method -> \"{meth}\" is not supported (use \"ConformingDelaunay\")")
    m, P = _mesh_arg(mr, f)
    mcm = _max_measure(o.get("MaxCellMeasure"), f)
    n = m.edim
    top = m.top_dim()
    if n == 1 or (top == 1 and not m.boundary):
        if m.boundary and n == 1:
            ends = sorted(range(len(P)), key=lambda i: P[i][0])
            segs = [(ends[k], ends[k + 1]) for k in range(0, len(ends) - 1, 2)]
        else:
            segs = [(a, b) for h, idx in m.cells if h == "Line" for a, b in zip(idx[:-1], idx[1:])]
            if len(segs) != sum(1 for h, _ in m.cells if M.CELL_DIM[h] > 0) and not segs:
                raise ValueError(f"{f}: no line cells to triangulate")
        Q, cells = [p for p in P], []
        for a, b in segs:
            ln = float(np.linalg.norm(P[b] - P[a]))
            k = 1 if mcm is None or mcm == math.inf else max(1, math.ceil(ln / mcm - 1e-12))
            prev = a
            for j in range(1, k):
                Q.append(P[a] + (P[b] - P[a]) * j / k)
                cells.append(("Line", (prev, len(Q) - 1)))
                prev = len(Q) - 1
            cells.append(("Line", (prev, b)))
        Q, cells = compact(Q, cells)
        return mesh_out(Q, cells, "MeshRegion", keep)
    require_scipy(f)
    if n == 2:
        segs = _loops_2d(m, P, f)
        U, idx = dedupe_points(P)
        segs = [(idx[a], idx[b]) for a, b in segs if idx[a] != idx[b]]
        quality = mcm != math.inf
        Q, tris = conforming_delaunay(U, segs, f, None if mcm in (None, math.inf) else mcm, quality)
        Q, cells = compact(list(Q), [("Triangle", t) for t in tris])
        Q = np.array(Q)
        tot = sum(abs(_tri_area(Q, t)) for _, t in cells)
        segs_area = sum((U[a][0] * U[b][1] - U[b][0] * U[a][1]) / 2 for a, b in segs)
        if m.boundary:
            ref = abs(segs_area)
        else:
            ref = sum(abs(sum(_tri_area(P, (idx_[0], idx_[k], idx_[k + 1])) for k in range(1, len(idx_) - 1)))
                      for h, idx_ in m.cells if M.CELL_DIM[h] == 2)
        if abs(tot - ref) > 1e-9 * max(ref, 1e-300):
            raise ArithmeticError(f"{f}: self-check failed (triangle areas {tot} do not add up to {ref})")
        return mesh_out(Q, cells, "MeshRegion", keep)
    # 3-D
    if top == 2 and not m.boundary:
        cells = []
        for h, idx in m.cells:
            cells += [("Triangle", (idx[0], idx[k], idx[k + 1])) for k in range(1, len(idx) - 1)]
        if mcm not in (None, math.inf):
            raise NotImplementedError(f"{f}: refinement of surface meshes (finite MaxCellMeasure) is not supported")
        return mesh_out(P, cells, "MeshRegion", keep)
    if top == 3:
        if mcm not in (None, math.inf):
            raise NotImplementedError(f"{f}: refinement of solid meshes (finite MaxCellMeasure) is not supported")
        return mr if not keep else mesh_out(P, m.cells, "MeshRegion", keep)
    if mcm not in (None, math.inf):
        raise NotImplementedError(f"{f}: refinement of solid meshes (finite MaxCellMeasure) is not supported")
    from scipy.spatial import Delaunay, ConvexHull
    hull = ConvexHull(P)
    vol_hull = float(hull.volume)
    faces = [idx for h, idx in m.cells if M.CELL_DIM[h] == 2]
    vol = 0.0
    for fc in faces:
        for k in range(1, len(fc) - 1):
            vol += float(np.dot(P[fc[0]], np.cross(P[fc[k]], P[fc[k + 1]]))) / 6
    if abs(abs(vol) - vol_hull) > 1e-9 * vol_hull:
        raise NotImplementedError(f"{f}: tetrahedralization of non-convex solids is not supported")
    U, _ = dedupe_points(P)
    tets = [_oriented(U, [int(i) for i in s]) for s in Delaunay(U).simplices]
    tv = sum(abs(float(np.linalg.det(U[list(t[1:])] - U[t[0]]))) / 6 for t in tets)
    if abs(tv - vol_hull) > 1e-9 * vol_hull:
        raise ArithmeticError(f"{f}: self-check failed (tetrahedron volumes do not add up)")
    Q, cells = compact(list(U), [("Tetrahedron", t) for t in tets])
    return mesh_out(Q, cells, "MeshRegion", keep)


# ═══════════════════════════════ Poisson reconstruction ═══════════════════════════════

_KUHN = [(0, 1, 3, 7), (0, 1, 5, 7), (0, 2, 3, 7), (0, 2, 6, 7), (0, 4, 5, 7), (0, 4, 6, 7)]
_CORNER = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


def marching_tetrahedra(F, origin, h, iso):
    """Triangles of the iso-surface F = iso of a grid function (each cube split into 6 Kuhn tetrahedra; vertices
    linearly interpolated on edges, shared between cells).  Triangles are oriented with normals towards
    increasing F."""
    np = _np()
    nx, ny, nz = F.shape
    G = F - iso
    pts, key, tris = [], {}, []

    def vid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in key:
            fa, fb = G[a], G[b]
            t = fa / (fa - fb)
            pa, pb = np.array(a, dtype=float), np.array(b, dtype=float)
            pts.append(origin + h * (pa + t * (pb - pa)))
            key[k] = len(pts) - 1
        return key[k]
    pos = G > 0
    mixed = np.zeros((nx - 1, ny - 1, nz - 1), dtype=bool)
    for c in _CORNER:
        for d in _CORNER:
            mixed |= pos[c[0]:nx - 1 + c[0], c[1]:ny - 1 + c[1], c[2]:nz - 1 + c[2]] != \
                pos[d[0]:nx - 1 + d[0], d[1]:ny - 1 + d[1], d[2]:nz - 1 + d[2]]
    for i, j, k in zip(*np.nonzero(mixed)):
        corners = [(i + a, j + b, k + c) for a, b, c in _CORNER]
        for tet in _KUHN:
            V = [corners[q] for q in tet]
            ins = [v for v in V if G[v] > 0]
            out_ = [v for v in V if G[v] <= 0]
            if not ins or not out_:
                continue
            if len(ins) == 1 or len(out_) == 1:
                lone = ins[0] if len(ins) == 1 else out_[0]
                rest = out_ if len(ins) == 1 else ins
                t = [vid(lone, r) for r in rest]
                tris.append(tuple(t))
            else:
                a, b = ins
                c, d = out_
                tris.append((vid(a, c), vid(a, d), vid(b, d)))
                tris.append((vid(a, c), vid(b, d), vid(b, c)))
    if not tris:
        return np.zeros((0, 3)), []
    P = np.array(pts)
    # orient by the gradient direction
    gx, gy, gz = np.gradient(G, h)
    out_t = []
    for t in tris:
        a, b, c = P[list(t)]
        n = np.cross(b - a, c - a)
        q = ((a + b + c) / 3 - origin) / h
        ii = tuple(np.clip(np.round(q).astype(int), 0, np.array(F.shape) - 1))
        g = np.array([gx[ii], gy[ii], gz[ii]])
        out_t.append(t if np.dot(n, g) >= 0 else (t[0], t[2], t[1]))
    return P, out_t


def poisson_indicator(P, N, res, f):
    """Indicator function on a res^3 grid (Kazhdan-Bolitho-Hoppe 2006, single level): normals splatted
    trilinearly and Gaussian-smoothed, divergence by central differences, Poisson equation solved with a
    Dirichlet DST-I solver; iso value = mean indicator at the samples."""
    np = _np()
    require_scipy(f)
    from scipy.fft import dstn, idstn
    from scipy.ndimage import gaussian_filter, map_coordinates
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    pad = 0.15 * span
    origin = (lo + hi) / 2 - (span / 2 + pad)
    h = (span + 2 * pad) / (res - 1)
    V = np.zeros((3, res, res, res))
    q = (P - origin) / h
    i0 = np.floor(q).astype(int)
    fr = q - i0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (fr[:, 0] if dx else 1 - fr[:, 0]) * (fr[:, 1] if dy else 1 - fr[:, 1]) * \
                    (fr[:, 2] if dz else 1 - fr[:, 2])
                ii = np.clip(i0 + [dx, dy, dz], 0, res - 1)
                for c in range(3):
                    np.add.at(V[c], (ii[:, 0], ii[:, 1], ii[:, 2]), w * N[:, c])
    # sample density normalisation (area per sample ~ (surface area) / n): use the grid cell area
    V = np.array([gaussian_filter(V[c], 1.0, mode="constant") for c in range(3)])
    div = np.gradient(V[0], h, axis=0) + np.gradient(V[1], h, axis=1) + np.gradient(V[2], h, axis=2)
    inner = div[1:-1, 1:-1, 1:-1]
    m = res - 2
    D = dstn(inner, type=1)
    k = np.arange(1, m + 1)
    lam = (2 * np.cos(np.pi * k / (m + 1)) - 2) / h ** 2
    den = lam[:, None, None] + lam[None, :, None] + lam[None, None, :]
    X = idstn(D / den, type=1)
    chi = np.zeros((res, res, res))
    chi[1:-1, 1:-1, 1:-1] = X
    iso = float(np.mean(map_coordinates(chi, q.T, order=1)))
    return chi, origin, h, iso


def _grid_res(goal, f, default=40):
    if goal is None or is_auto(goal):
        return default
    if goal == "Speed":
        return 28
    if goal == "Quality":
        return 64
    raise ValueError(f"{f}: PerformanceGoal must be Automatic, \"Speed\" or \"Quality\"")


def _points_normals(x, o, f):
    np = _np()
    P, N = require_points(x, f, (3,), lo=4)
    vn = o.get("VertexNormals")
    if vn is not None and not is_auto(vn):
        N = normals_arg(vn, len(P), 3, f)
    if N is None:
        N = plane_fit_normals(P)
    return P, N


def poisson_mesh(P, N, res, f):
    np = _np()
    chi, origin, h, iso = poisson_indicator(P, N, res, f)
    Q, tris = marching_tetrahedra(chi, origin, h, iso)
    if not tris:
        raise ValueError(f"{f}: no surface was found; check the point normals")
    Q, cells = compact(list(Q), [("Triangle", t) for t in tris])
    return np.array(Q), cells


@fn("GradientFittedMesh", area=AREA, status="added",
    notes="GradientFittedMesh[{p1, ...}] and GradientFittedMesh[Point[pts, VertexNormals -> normals]] for 3-D "
          "points: screened-free Poisson surface reconstruction (Kazhdan-Bolitho-Hoppe 2006, single-level "
          "version): normals (given, VertexNormals option, or estimated as in EstimatedPointNormals) are "
          "splatted trilinearly on a cubic grid over the padded bounding box and Gaussian-smoothed (sigma = 1 "
          "cell), the Poisson equation Laplacian(chi) = div(V) is solved with zero Dirichlet boundary values "
          "(DST-I), and the iso-surface chi = mean(chi at the samples) is extracted by marching tetrahedra (Kuhn "
          "split). Grid size: PerformanceGoal Automatic 40, \"Speed\" 28, \"Quality\" 64 points per side. The "
          "result is a triangle MeshRegion whose normals point along the given normals. Wolfram's octree solver "
          "is undocumented, so vertex positions differ (resolution-limited). 2-D points and fewer than 4 points "
          "raise ValueError; needs SciPy (NotImplementedError 'requires SciPy' otherwise). Check: self-check "
          "that the surface is closed (every edge on two triangles; ArithmeticError otherwise); tests "
          "reconstruct sphere points (vertices within one grid cell of the unit sphere, enclosed volume near "
          "4 Pi / 3).")
def _gradient_fitted_mesh(points: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    f = "GradientFittedMesh"
    pos, o, keep = options(opts, f, {"PerformanceGoal", "VertexNormals"}, MESH_PRESENTATION)
    if pos:
        raise ValueError(f"{f}: expected GradientFittedMesh[points, opts]")
    require_scipy(f)
    P, N = _points_normals(points, o, f)
    Q, cells = poisson_mesh(P, N, _grid_res(o.get("PerformanceGoal"), f), f)
    E = edge_faces([c for _, c in cells])
    if any(len(v) != 2 for v in E.values()):
        raise ArithmeticError(f"{f}: self-check failed (the reconstructed surface is not closed)")
    return mesh_out(Q, cells, "MeshRegion", keep)


# ═══════════════════════════════ ReconstructionMesh ═══════════════════════════════

def emst_alpha(P):
    """Half the longest edge of the Euclidean minimum spanning tree (the smallest alpha at which the alpha
    complex is connected)."""
    np = _np()
    n = len(P)
    if n < 2:
        return 0.0
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)) if n <= 3000 else None
    if D is None:
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import Delaunay
        tri = Delaunay(P)
        from scipy.sparse import coo_matrix
        r, c = [], []
        for s in tri.simplices:
            for a in s:
                for b in s:
                    if a < b:
                        r.append(a)
                        c.append(b)
        w = np.linalg.norm(P[r] - P[c], axis=1)
        T = minimum_spanning_tree(coo_matrix((w, (r, c)), shape=(n, n)))
        return float(T.data.max()) / 2
    inT = np.zeros(n, dtype=bool)
    inT[0] = True
    best = D[0].copy()
    mx = 0.0
    for _ in range(n - 1):
        b = np.where(inT, np.inf, best)
        j = int(np.argmin(b))
        mx = max(mx, float(b[j]))
        inT[j] = True
        best = np.minimum(best, D[j])
    return mx / 2


def alpha_boundary(P, alpha, f):
    cx = alpha_complex(P, f)
    n = P.shape[1]
    a2 = alpha * alpha * (1 + 1e-9)
    kept = {k: {s for s, v in cx[k].items() if v <= a2} for k in cx}
    cnt = {}
    for t in kept.get(n, ()):
        for v in t:
            fc = t - {v}
            cnt[fc] = cnt.get(fc, 0) + 1
    cells = [(n - 1, fc) for fc, c in cnt.items() if c == 1]
    covered = set(cnt)
    for k in range(n - 1, 0, -1):
        for s in kept.get(k, ()):
            if s in covered:
                continue
            if k < n - 1 and any(s < t for t in kept.get(k + 1, ())):
                continue
            if k == n - 1 and s in cnt:
                continue
            cells.append((k, s))
    return cells, kept


def _orient_boundary(P, cells, kept):
    """Orient the boundary (n-1)-cells outward of the kept n-cells."""
    np = _np()
    n = P.shape[1]
    top = {}
    for t in kept.get(n, ()):
        for v in t:
            top[t - {v}] = t
    out_ = []
    for k, s in cells:
        idx = sorted(s)
        if k == n - 1 and s in top and n >= 2:
            opp = next(iter(top[s] - s))
            if n == 2:
                a, b = P[idx[0]], P[idx[1]]
                cr = (b[0] - a[0]) * (P[opp][1] - a[1]) - (b[1] - a[1]) * (P[opp][0] - a[0])
                if cr > 0:
                    idx = idx[::-1]
            else:
                a, b, c = P[idx]
                if np.dot(np.cross(b - a, c - a), P[opp] - a) > 0:
                    idx = [idx[0], idx[2], idx[1]]
        out_.append((_SIMPLEX_HEAD[k], tuple(idx)))
    return out_


def crust_2d(P, f):
    """Crust of Amenta-Bern-Eppstein (1998): Delaunay edges of P plus its Voronoi vertices joining two samples."""
    np = _np()
    from scipy.spatial import Delaunay
    tri = Delaunay(P)
    C = []
    for s in tri.simplices:
        a, b, c = P[s]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if d == 0:
            continue
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        C.append((ux, uy))
    A = np.vstack([P, np.array(C)]) if C else P
    t2 = Delaunay(A, qhull_options="QJ") if len(C) else tri
    n = len(P)
    E = set()
    for s in t2.simplices:
        for k in range(3):
            a, b = int(s[k]), int(s[(k + 1) % 3])
            if a < n and b < n:
                E.add((min(a, b), max(a, b)))
    return [("Line", e) for e in sorted(E)]


@fn("ReconstructionMesh", area=AREA, status="added",
    notes="ReconstructionMesh[{p1, p2, ...}] (2-D or 3-D points) gives a mesh approximating the sampled shape. "
          "Method -> \"AlphaShape\" (default): the boundary of the alpha complex (as in ConcaveHullMesh) at "
          "alpha = half the longest Euclidean-minimum-spanning-tree edge (the smallest alpha at which the complex "
          "is connected) or the suboption \"Alpha\" -> a: the outward-oriented boundary (n-1)-cells of the kept "
          "n-cells together with the maximal kept lower-dimensional cells, so curves sampled in the plane or in "
          "space give lines and surfaces give triangles. \"Crust\" (2-D only): the crust of Amenta, Bern and "
          "Eppstein (1998) (3-D raises NotImplementedError). \"Poisson\" (3-D only): GradientFittedMesh (option "
          "VertexNormals). Wolfram's automatic choices are undocumented, so cells can differ. ProgressReporting "
          "is accepted and ignored. Needs SciPy. Check: exact for the alpha shape (definition); tests reconstruct "
          "CirclePoints[16] as its 16-gon, a densely sampled circle by the crust, and sphere samples by "
          "Poisson.")
def _reconstruction_mesh(points: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    f = "ReconstructionMesh"
    np = _np()
    pos, o, keep = options(opts, f, {"Method", "ProgressReporting", "VertexNormals"},
                           MESH_PRESENTATION - {"Method"})
    if pos:
        raise ValueError(f"{f}: expected ReconstructionMesh[points, opts]")
    meth, sub = _method(o.get("Method"), f, ("AlphaShape", "Crust", "Poisson"), {"Alpha"})
    require_scipy(f)
    P, _ = require_points(points, f, (2, 3), lo=2)
    P, _ = dedupe_points(P)
    if meth == "Poisson":
        if P.shape[1] != 3:
            raise NotImplementedError(f"{f}: Method -> \"Poisson\" needs 3-D points")
        P, N = _points_normals(points, o, f)
        Q, cells = poisson_mesh(P, N, 40, f)
        return mesh_out(Q, cells, "MeshRegion", keep)
    if meth == "Crust":
        if P.shape[1] != 2:
            raise NotImplementedError(f"{f}: Method -> \"Crust\" is implemented for 2-D points only")
        if len(P) < 3:
            raise ValueError(f"{f}: at least 3 points are needed")
        cells = crust_2d(P, f)
        Q, cells = compact(list(P), cells)
        return mesh_out(Q, cells, "MeshRegion", keep)
    alpha = _sub_real(sub, "Alpha", None, f) if "Alpha" in sub else emst_alpha(P)
    if len(P) <= P.shape[1]:
        cells = [(1, frozenset(e)) for e in zip(range(len(P) - 1), range(1, len(P)))]
        kept = {}
    else:
        cells, kept = alpha_boundary(P, alpha, f)
    if not cells:
        raise ValueError(f"{f}: the alpha shape is empty")
    out_ = _orient_boundary(P, cells, kept)
    Q, out_ = compact(list(P), out_)
    return mesh_out(Q, out_, "MeshRegion", keep)


# ═══════════════════════════════ ListSurfacePlot3D ═══════════════════════════════

@fn("ListSurfacePlot3D", area=AREA, status="added",
    notes="ListSurfacePlot3D[{{x1, y1, z1}, ...}] Graphics3D of a smooth surface fitted to the points (order "
          "independent): single-level Poisson reconstruction as in GradientFittedMesh with normals estimated by "
          "plane fitting (EstimatedPointNormals), so the surface approximates but need not contain the points "
          "(as documented) and is closed. MaxPlotPoints sets the grid size per side (Automatic 30, Infinity 64, "
          "or an integer 8..128). Output Graphics3D[{style, GraphicsComplex[pts, {EdgeForm[], Polygon[...]}]}] "
          "with Axes -> True, BoxRatios -> Automatic and the data ranges as PlotRange; mesh lines (Mesh -> "
          "Automatic) are not drawn. Options: PlotStyle, MaxPlotPoints, PlotRange and Graphics3D options; others "
          "raise NotImplementedError. Wolfram's reconstruction method is undocumented, so the surface differs in "
          "detail. Needs SciPy. Check: reference (tests verify that sphere samples give vertices within a grid "
          "cell of the sphere).")
def _list_surface_plot3d(data: MatrixOf[Real], *opts: Option) -> Expr:
    f = "ListSurfacePlot3D"
    np = _np()
    from . import wl_plots2d as P2
    pos, o_ = P2._list_args((data,) + opts, f)
    o, gopts = P2._opts3d(f, o_, ("PlotStyle", "MaxPlotPoints"))
    require_scipy(f)
    mpp = o["MaxPlotPoints"]
    if is_auto(mpp):
        res = 30
    elif is_sym(mpp, "Infinity"):
        res = 64
    else:
        res = int_arg(mpp, f, "MaxPlotPoints", 8, 128)
    P, _ = require_points(pos[0], f, (3,), lo=4)
    P, _ = dedupe_points(P)
    N = plane_fit_normals(P)
    Q, cells = poisson_mesh(P, N, res, f)
    st = P2.surface_styles(o["PlotStyle"], 1, f)[0]
    content = []
    if st is not None:
        gc = P2.surface_complex(Q, [c for _, c in cells])
        if gc is not None:
            content.append(L(st, gc))
    prange = P2.resolve_plot_range(o["PlotRange"], f, P2.bounds3(P), dims=3)
    return P2._graphics3d(content, gopts, prange)
