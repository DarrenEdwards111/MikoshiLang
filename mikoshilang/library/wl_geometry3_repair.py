"""Wolfram batch X06: Geometry III -- mesh defects and repair (FindMeshDefects, RepairMesh), decimation
(SimplifyMesh) and remeshing (Remesh) of surface meshes.  See wl_geometry3 for the conventions used."""
from __future__ import annotations

import heapq
import math

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from . import wl_regions_mesh as M
from .wl_geometry3_util import (int_arg, real_arg, options, is_auto, is_sym, mesh_np, mesh_out, compact,
                                MESH_PRESENTATION, dedupe_points)
from .wl_geometry3 import _mesh_arg, _method, _sub_int, _sub_real, edge_faces, AREA

DEFECTS = ("HoleEdges", "TJunctionEdges", "TinyFaces", "OverlappingFaces", "IsolatedVertices", "DanglingEdges",
           "SingularEdges", "SingularVertices", "TinyComponents", "FlippedFaces")
COLORS = {"HoleEdges": "Red", "TJunctionEdges": "Orange", "TinyFaces": "Magenta", "OverlappingFaces": "Purple",
          "IsolatedVertices": "Blue", "DanglingEdges": "Cyan", "SingularEdges": "Green",
          "SingularVertices": "Brown", "TinyComponents": "Pink", "FlippedFaces": "Yellow"}
TINY_AREA = 1e-10      # relative to the squared bounding-box diagonal
TINY_COMPONENT = 1e-3  # relative to the area of the largest component
TJ_TOL = 1e-9          # relative to the bounding-box diagonal


def _np():
    import numpy as np
    return np


class Surf:
    """Polygon surface view of a mesh: coordinates, faces (0-based tuples), explicit lines and points."""

    def __init__(self, m, P, f):
        np = _np()
        self.m, self.P, self.f = m, P, f
        if m.top_dim() == 3:
            raise NotImplementedError(f"{f}: solid (tetrahedral / hexahedral) meshes are not supported; use the "
                                      "boundary surface")
        self.faces = [tuple(idx) for h, idx in m.cells if M.CELL_DIM[h] == 2]
        self.lines = [tuple(idx) for h, idx in m.cells if h == "Line"]
        self.points = [idx[0] for h, idx in m.cells if h == "Point"]
        self.diag = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) or 1.0

    def area(self, fc):
        np = _np()
        P = self.P
        if P.shape[1] == 2:
            x, y = P[list(fc), 0], P[list(fc), 1]
            return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
        v = np.zeros(3)
        for k in range(1, len(fc) - 1):
            v += np.cross(P[fc[k]] - P[fc[0]], P[fc[k + 1]] - P[fc[0]])
        return float(np.linalg.norm(v)) / 2

    def normal(self, fc):
        np = _np()
        P = self.P
        v = np.zeros(3)
        for k in range(1, len(fc) - 1):
            v += np.cross(P[fc[k]] - P[fc[0]], P[fc[k + 1]] - P[fc[0]])
        return v


def _edges_of(fc):
    n = len(fc)
    return [(fc[i], fc[(i + 1) % n]) for i in range(n)]


def face_components(faces):
    """Connected components of faces through shared edges: list of face-index lists."""
    E = edge_faces(faces)
    parent = list(range(len(faces)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for fl_ in E.values():
        for b in fl_[1:]:
            parent[find(b)] = find(fl_[0])
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def orientation_classes(S):
    """Per component: (faces whose orientation must flip to agree with the reference orientation).  Reference:
    the majority orientation reached by propagation over manifold edges; closed components are then made
    outward (positive signed volume)."""
    np = _np()
    faces = S.faces
    E = edge_faces(faces)
    flips = set()
    for comp in face_components(faces):
        par = {comp[0]: 0}
        stack = [comp[0]]
        while stack:
            a = stack.pop()
            for u, v in _edges_of(faces[a]):
                fl_ = E[frozenset((u, v))]
                if len(fl_) != 2:
                    continue
                b = fl_[0] if fl_[1] == a else fl_[1]
                if b in par:
                    continue
                same = (u, v) in _edges_of(faces[b])   # same direction -> opposite orientation
                par[b] = par[a] ^ (1 if same else 0)
                stack.append(b)
        ones = [i for i in comp if par.get(i, 0) == 1]
        zeros = [i for i in comp if par.get(i, 0) == 0]
        bad = ones if len(ones) < len(zeros) or (len(ones) == len(zeros) and ones and min(zeros) < min(ones)) \
            else zeros
        closed = all(len(E[frozenset(e)]) == 2 for i in comp for e in _edges_of(faces[i]))
        if closed and S.P.shape[1] == 3:
            vol = 0.0
            for i in comp:
                fc = faces[i] if i not in bad else faces[i][::-1]
                for k in range(1, len(fc) - 1):
                    vol += float(np.dot(S.P[fc[0]], np.cross(S.P[fc[k]], S.P[fc[k + 1]])))
            if vol < 0:
                bad = [i for i in comp if i not in set(bad)]
        flips.update(bad)
    return flips


def _coplanar_overlap(S, i, j):
    """Coplanar faces i, j overlap with positive area (Sutherland-Hodgman clipping in the common plane)."""
    np = _np()
    P = S.P
    A, B = S.faces[i], S.faces[j]
    if P.shape[1] == 3:
        n = S.normal(A)
        nn = np.linalg.norm(n)
        if nn == 0:
            return False
        n = n / nn
        nb = S.normal(B)
        nbn = np.linalg.norm(nb)
        if nbn == 0 or abs(abs(np.dot(n, nb / nbn)) - 1) > 1e-9:
            return False
        if max(abs(float(np.dot(P[v] - P[A[0]], n))) for v in B) > 1e-9 * S.diag:
            return False
        u = P[A[1]] - P[A[0]]
        u = u / np.linalg.norm(u)
        w = np.cross(n, u)
        pa = [(float(np.dot(P[v], u)), float(np.dot(P[v], w))) for v in A]
        pb = [(float(np.dot(P[v], u)), float(np.dot(P[v], w))) for v in B]
    else:
        pa, pb = [tuple(P[v]) for v in A], [tuple(P[v]) for v in B]

    def sarea(poly):
        return sum(poly[k][0] * poly[(k + 1) % len(poly)][1] - poly[(k + 1) % len(poly)][0] * poly[k][1]
                   for k in range(len(poly))) / 2
    if sarea(pa) < 0:
        pa = pa[::-1]
    if sarea(pb) < 0:
        pb = pb[::-1]
    res = pb
    for k in range(len(pa)):
        a, b = pa[k], pa[(k + 1) % len(pa)]

        def inside(p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0
        new = []
        for t in range(len(res)):
            p, q = res[t], res[(t + 1) % len(res)]
            ip, iq = inside(p), inside(q)
            if ip:
                new.append(p)
            if ip != iq:
                d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
                s = d1 / (d1 - d2)
                new.append((p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])))
        res = new
        if not res:
            return False
    return len(res) >= 3 and abs(sarea(res)) > 1e-9 * S.diag ** 2


def find_defects(S, kinds):
    """{kind: [(d, cell vertex tuple)]} for the requested defect kinds."""
    np = _np()
    P, faces = S.P, S.faces
    E = edge_faces(faces)
    out = {}
    used_edges = set(E)
    if "HoleEdges" in kinds:
        res = []
        for fi, fc in enumerate(faces):
            for u, v in _edges_of(fc):
                if len(E[frozenset((u, v))]) == 1:
                    res.append((1, (u, v)))
        out["HoleEdges"] = res
    if "SingularEdges" in kinds:
        out["SingularEdges"] = [(1, tuple(sorted(e))) for e, fl_ in E.items() if len(fl_) > 2]
    if "TinyFaces" in kinds:
        tol = TINY_AREA * S.diag ** 2
        out["TinyFaces"] = [(2, fc) for fc in faces if S.area(fc) <= tol]
    if "IsolatedVertices" in kinds:
        inedge = {v for e in used_edges for v in e} | {v for ln in S.lines for v in ln}
        out["IsolatedVertices"] = [(0, (v,)) for v in range(len(P)) if v not in inedge]
    if "DanglingEdges" in kinds:
        res = []
        for ln in S.lines:
            for u, v in zip(ln[:-1], ln[1:]):
                if frozenset((u, v)) not in used_edges:
                    res.append((1, (u, v)))
        out["DanglingEdges"] = res if faces else []
    if "SingularVertices" in kinds:
        res = []
        vf = {}
        for fi, fc in enumerate(faces):
            for v in fc:
                vf.setdefault(v, []).append(fi)
        for v, fl_ in vf.items():
            # faces around v connected through the edges containing v
            par = {i: i for i in fl_}

            def find(a):
                while par[a] != a:
                    a = par[a]
                return a
            for e, fe in E.items():
                if v in e:
                    fe2 = [i for i in fe if i in par]
                    for b in fe2[1:]:
                        par[find(b)] = find(fe2[0])
            if len({find(i) for i in fl_}) > 1:
                res.append((0, (v,)))
        out["SingularVertices"] = res
    if "TJunctionEdges" in kinds:
        res = []
        tol = TJ_TOL * S.diag
        verts = sorted({v for fc in faces for v in fc} | {v for ln in S.lines for v in ln})
        for e in E:
            a, b = tuple(e)
            d = P[b] - P[a]
            L2 = float(np.dot(d, d))
            if L2 == 0:
                continue
            for v in verts:
                if v in e:
                    continue
                t = float(np.dot(P[v] - P[a], d)) / L2
                if 1e-9 < t < 1 - 1e-9 and np.linalg.norm(P[a] + t * d - P[v]) <= tol:
                    res.append((1, tuple(sorted(e))))
                    break
        out["TJunctionEdges"] = res
    if "OverlappingFaces" in kinds:
        res = set()
        lo = np.array([P[list(fc)].min(axis=0) for fc in faces]) if faces else None
        hi = np.array([P[list(fc)].max(axis=0) for fc in faces]) if faces else None
        for i in range(len(faces)):
            for j in range(i + 1, len(faces)):
                if np.any(lo[j] > hi[i] + 1e-12) or np.any(lo[i] > hi[j] + 1e-12):
                    continue
                if _coplanar_overlap(S, i, j):
                    res.update((i, j))
        out["OverlappingFaces"] = [(2, faces[i]) for i in sorted(res)]
    if "TinyComponents" in kinds:
        comps = face_components(faces)
        areas = [sum(S.area(faces[i]) for i in c) for c in comps]
        big = max(areas) if areas else 0.0
        out["TinyComponents"] = [(2, faces[i]) for c, a in zip(comps, areas) if len(comps) > 1 and
                                 a < TINY_COMPONENT * big for i in c]
    if "FlippedFaces" in kinds:
        fl_ = orientation_classes(S) if S.P.shape[1] == 3 else set()
        out["FlippedFaces"] = [(2, faces[i]) for i in sorted(fl_)]
    return out


def Rule(a, b):
    return Expr("Rule", a, b)


_HEAD = {0: "Point", 1: "Line", 2: "Polygon"}


def _kinds(spec, f):
    if spec is None or is_auto(spec) or is_sym(spec, "All"):
        return list(DEFECTS)
    items = list(spec.args) if K.is_list(spec) else [spec]
    for k in items:
        if not isinstance(k, str):
            if isinstance(k, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: defect types must be strings, not {K_input(k)}")
        if k not in DEFECTS:
            raise ValueError(f"{f}: unknown defect type \"{k}\" (known: {', '.join(DEFECTS)})")
    return items


def _cell_index(m, d, cell):
    A = m.all_cells()
    key = frozenset(cell)
    for i, (h, idx) in enumerate(A.get(d, [])):
        if frozenset(idx) == key:
            return i + 1
    return None


@fn("FindMeshDefects", area=AREA, status="added",
    notes="FindMeshDefects[mreg], [mreg, def | {def1, ...}], [mreg, defs, fmt] for surface (and curve / point) "
          "MeshRegions and BoundaryMeshRegions. Defect types and the criteria used: \"HoleEdges\" (edges of "
          "exactly one face), \"SingularEdges\" (more than two faces), \"TJunctionEdges\" (a mesh vertex lies "
          "inside the edge within 1e-9 of the bounding-box diagonal), \"TinyFaces\" (area <= 1e-10 diag^2), "
          "\"OverlappingFaces\" (coplanar faces whose intersection has positive area, found by polygon "
          "clipping), \"IsolatedVertices\" (vertices on no edge or line), \"DanglingEdges\" (line cells that are "
          "not face edges, in meshes with faces), \"SingularVertices\" (the faces around the vertex form more than "
          "one fan), \"TinyComponents\" (edge-connected components with area < 1e-3 of the largest), "
          "\"FlippedFaces\" (faces against the majority orientation of their component; closed components are "
          "made outward). Wolfram's tolerances are undocumented. Formats: Automatic (the mesh with "
          "MeshCellHighlight for the defect cells, one colour per type; PlotLegends -> Automatic wraps it in "
          "Legended with a SwatchLegend), \"Cell\" (Association type -> cells, only types found) and "
          "\"CellIndex\" (type -> {d, i} indices as in MeshCells). Solid meshes NotImplementedError; unknown "
          "types ValueError; non-meshes TypeError. Check: exact (combinatorial tests); tests use meshes with "
          "known holes, a flipped face, a T junction, an isolated vertex and a singular (bow-tie) vertex.")
def _find_mesh_defects(mreg: Expr, *rest: String | ListOf[String] | Option) -> Expr:
    f = "FindMeshDefects"
    pos, o, keep = options(rest, f, {"PlotLegends"}, MESH_PRESENTATION)
    if len(pos) > 2:
        raise ValueError(f"{f}: expected FindMeshDefects[mreg, defs, fmt]")
    m, P = _mesh_arg(mreg, f)
    kinds = _kinds(pos[0] if pos else None, f)
    fmt = pos[1] if len(pos) == 2 else Symbol("Automatic")
    if not (is_auto(fmt) or fmt in ("Cell", "CellIndex")):
        raise ValueError(f"{f}: the format must be Automatic, \"Cell\" or \"CellIndex\"")
    S = Surf(m, P, f)
    D = find_defects(S, set(kinds))
    found = [(k, D[k]) for k in kinds if D.get(k)]
    if fmt == "Cell":
        return Expr("Association", *[Rule(k, L(*[Expr(_HEAD[d], L(*[i + 1 for i in c])) for d, c in v]))
                                     for k, v in found])
    idxs = []
    for k, v in found:
        ii = []
        for d, c in v:
            i = _cell_index(m, d, c)
            if i is not None and [d, i] not in ii:
                ii.append([d, i])
        idxs.append((k, ii))
    if fmt == "CellIndex":
        return Expr("Association", *[Rule(k, L(*[L(*t) for t in ii])) for k, ii in idxs])
    hl = [Rule(L(d, i), Symbol(COLORS[k])) for k, ii in idxs for d, i in ii]
    a, old = split_trailing_options(list(mreg.args))
    old = [r for r in old if opt_name(r) != "MeshCellHighlight" and opt_name(r) not in {opt_name(k) for k in keep}]
    res = Expr(mreg.head, *a, *old, *([Rule(Symbol("MeshCellHighlight"), L(*hl))] if hl else []), *keep)
    leg = o.get("PlotLegends")
    if leg is None or is_sym(leg, "None"):
        return res
    if not is_auto(leg):
        raise NotImplementedError(f"{f}: only PlotLegends -> Automatic or None is supported")
    return Expr("Legended", res, Expr("SwatchLegend", L(*[Symbol(COLORS[k]) for k, _ in found]),
                                      L(*[k for k, _ in found])))


def _boundary_loops(faces):
    """Directed boundary loops (in the direction opposite to the face edges, i.e. the hole's outward walk)."""
    E = edge_faces(faces)
    nxt = {}
    for fc in faces:
        for u, v in _edges_of(fc):
            if len(E[frozenset((u, v))]) == 1:
                if v in nxt:
                    raise NotImplementedError("RepairMesh: holes whose boundary touches itself (a singular vertex) "
                                              "cannot be filled; repair \"SingularVertices\" first")
                nxt[v] = u
    loops, seen = [], set()
    for s in list(nxt):
        if s in seen:
            continue
        lp, c = [], s
        while c not in seen:
            seen.add(c)
            lp.append(c)
            if c not in nxt:
                raise NotImplementedError("RepairMesh: an open boundary chain could not be closed")
            c = nxt[c]
        loops.append(lp)
    return loops


def _fan(P, fc, tol):
    """Triangles of a convex polygon (with collinear vertices) as a fan from a vertex giving no degenerate
    triangle; a centroid fan (new vertex) when no such vertex exists."""
    np = _np()

    def area(a, b, c):
        u, v = P[b] - P[a], P[c] - P[a]
        return float(np.linalg.norm(np.cross(u, v))) / 2 if len(u) == 3 else abs(u[0] * v[1] - u[1] * v[0]) / 2
    n = len(fc)
    for s in range(n):
        g = fc[s:] + fc[:s]
        tris = [(g[0], g[k], g[k + 1]) for k in range(1, n - 1)]
        if all(area(*t) > tol for t in tris):
            return tris
    P.append(np.mean([P[v] for v in fc], axis=0))
    c = len(P) - 1
    return [(fc[k], fc[(k + 1) % n], c) for k in range(n)]


def repair(S, kinds, f):
    np = _np()
    P = [p for p in S.P]
    faces, lines, points = list(S.faces), list(S.lines), list(S.points)
    found = find_defects(S, set(kinds))
    if found.get("SingularEdges"):
        raise NotImplementedError(f"{f}: repairing \"SingularEdges\" is not supported")
    if found.get("OverlappingFaces"):
        keys = {}
        for fc in faces:
            keys.setdefault(frozenset(fc), []).append(fc)
        if any(len(v) == 1 for k, v in keys.items() if any(set(c) == set(k) for _, c in found["OverlappingFaces"])):
            raise NotImplementedError(f"{f}: only exactly duplicated overlapping faces can be repaired")
        seen = set()
        nf = []
        for fc in faces:
            k = frozenset(fc)
            if k not in seen:
                seen.add(k)
                nf.append(fc)
        faces = nf
    if "DanglingEdges" in kinds and faces:
        E = edge_faces(faces)
        lines = [ln for ln in lines if not all(frozenset((u, v)) not in E for u, v in zip(ln[:-1], ln[1:]))]
        lines = [ln for ln in lines if all(frozenset((u, v)) in E for u, v in zip(ln[:-1], ln[1:]))]
    if "TinyComponents" in kinds and found.get("TinyComponents"):
        bad = {frozenset(c) for _, c in found["TinyComponents"]}
        faces = [fc for fc in faces if frozenset(fc) not in bad]
    if "TinyFaces" in kinds:
        tol = TINY_AREA * S.diag ** 2
        for _ in range(len(faces) + 1):
            S2 = Surf.__new__(Surf)
            S2.P = np.array(P)
            bad = [fc for fc in faces if Surf.area(S2, fc) <= tol]
            if not bad:
                break
            fc = bad[0]
            u, v = min(_edges_of(fc), key=lambda e: float(np.linalg.norm(P[e[0]] - P[e[1]])))
            faces = [tuple(dict.fromkeys(u if w == v else w for w in g)) for g in faces]
            faces = [g for g in faces if len(g) >= 3]
            lines = [tuple(u if w == v else w for w in ln) for ln in lines]
        else:
            raise ArithmeticError(f"{f}: self-check failed (tiny faces remain)")
    if "TJunctionEdges" in kinds:
        tol = TJ_TOL * S.diag
        for _ in range(10 * len(P) + 10):
            changed = False
            verts = sorted({v for fc in faces for v in fc})
            for fi, fc in enumerate(faces):
                for k, (a, b) in enumerate(_edges_of(fc)):
                    d = P[b] - P[a]
                    L2 = float(np.dot(d, d))
                    hit = None
                    for v in verts:
                        if v in (a, b):
                            continue
                        t = float(np.dot(P[v] - P[a], d)) / L2 if L2 else -1
                        if 1e-9 < t < 1 - 1e-9 and np.linalg.norm(P[a] + t * d - P[v]) <= tol:
                            hit = v
                            break
                    if hit is not None:
                        faces[fi] = fc[:k + 1] + (hit,) + fc[k + 1:]
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break
        # split polygons with inserted vertices back into triangles where the input was triangles
        nf = []
        for fc in faces:
            if len(fc) > 3 and all(len(g) == 3 for g in S.faces):
                nf += _fan(P, fc, TINY_AREA * S.diag ** 2)
            else:
                nf.append(fc)
        faces = nf
    if "SingularVertices" in kinds:
        S3 = Surf.__new__(Surf)
        S3.P, S3.faces, S3.lines, S3.points, S3.diag = np.array(P), faces, [], [], S.diag
        sv = [c[0] for _, c in find_defects(S3, {"SingularVertices"})["SingularVertices"]]
        E = edge_faces(faces)
        for v in sv:
            fl_ = [i for i, fc in enumerate(faces) if v in fc]
            par = {i: i for i in fl_}

            def find(a):
                while par[a] != a:
                    a = par[a]
                return a
            for e, fe in E.items():
                if v in e:
                    fe2 = [i for i in fe if i in par]
                    for b in fe2[1:]:
                        par[find(b)] = find(fe2[0])
            groups = {}
            for i in fl_:
                groups.setdefault(find(i), []).append(i)
            for g in list(groups.values())[1:]:
                P.append(P[v].copy())
                nv = len(P) - 1
                for i in g:
                    faces[i] = tuple(nv if w == v else w for w in faces[i])
    if "FlippedFaces" in kinds and S.P.shape[1] == 3:
        S4 = Surf.__new__(Surf)
        S4.P, S4.faces = np.array(P), faces
        fl_ = orientation_classes(S4)
        faces = [fc[::-1] if i in fl_ else fc for i, fc in enumerate(faces)]
    if "HoleEdges" in kinds and faces:
        for lp in _boundary_loops(faces):
            if len(lp) == 3:
                faces.append((lp[0], lp[1], lp[2]))
                continue
            P.append(np.mean([P[v] for v in lp], axis=0))
            c = len(P) - 1
            for k in range(len(lp)):
                faces.append((lp[k], lp[(k + 1) % len(lp)], c))
    if "IsolatedVertices" in kinds:
        points = []
    cells = [("Polygon" if len(fc) > 3 else "Triangle", fc) for fc in faces] + \
            [("Line", ln) for ln in lines] + [("Point", (p,)) for p in points]
    if "IsolatedVertices" in kinds:
        P, cells = compact(P, cells)
    return np.array(P), cells


@fn("RepairMesh", area=AREA, status="added",
    notes="RepairMesh[mreg] and RepairMesh[mreg, def | {def1, ...}] (the defect types and criteria of "
          "FindMeshDefects; default: all types). Repairs: \"IsolatedVertices\" removes unused coordinates and point "
          "cells; \"DanglingEdges\" removes line cells; \"TinyComponents\" removes them; \"TinyFaces\" collapses the "
          "shortest edge of each tiny face; \"TJunctionEdges\" inserts the T vertex into the faces along the edge "
          "(re-triangulated for triangle meshes); \"SingularVertices\" duplicates the vertex for each fan; "
          "\"FlippedFaces\" reverses them; \"HoleEdges\" fills each boundary loop with a fan around its centroid "
          "(a single triangle for 3-loops); exactly duplicated \"OverlappingFaces\" are removed. Repairing "
          "\"SingularEdges\" and other overlaps raises NotImplementedError when such defects are present. The "
          "result keeps the input head and presentation options. Wolfram's repair heuristics are undocumented, "
          "so repaired cells can differ. Check: self-check that the requested defects are gone afterwards "
          "(ArithmeticError otherwise); tests fill a hole of an open cube and fix a flipped face.")
def _repair_mesh(mreg: Expr, *rest: String | ListOf[String] | Option) -> Expr:
    f = "RepairMesh"
    pos, _o, keep = options(rest, f, set(), MESH_PRESENTATION)
    if len(pos) > 1:
        raise ValueError(f"{f}: expected RepairMesh[mreg, defs]")
    m, P = _mesh_arg(mreg, f)
    kinds = _kinds(pos[0] if pos else None, f)
    S = Surf(m, P, f)
    Q, cells = repair(S, kinds, f)
    m2 = M.Mesh([list(map(float, q)) for q in Q], cells, False, None, f)
    S2 = Surf(m2, Q, f)
    left = find_defects(S2, set(kinds) - {"OverlappingFaces"})
    if any(v for v in left.values()):
        bad = [k for k, v in left.items() if v]
        raise ArithmeticError(f"{f}: self-check failed (defects remain: {', '.join(bad)})")
    head = mreg.head
    if head == "BoundaryMeshRegion" and S2.faces and any(find_defects(S2, {"HoleEdges"}).values()):
        head = "MeshRegion"
    return mesh_out(Q, cells, head, keep)


# ═══════════════════════════════ editable triangle mesh (SimplifyMesh, Remesh) ═══════════════════════════════

class TriMesh:
    """Triangle surface with vertex -> face incidence, supporting edge collapse / split / flip."""

    def __init__(self, P, tris):
        np = _np()
        self.P = [np.array(p, dtype=float) for p in P]
        self.F = [tuple(t) for t in tris]
        self.vf = {}
        for i, t in enumerate(self.F):
            for v in t:
                self.vf.setdefault(v, set()).add(i)
        self.alive_v = set(self.vf)

    def faces_of(self, u, v):
        return [i for i in self.vf.get(u, ()) & self.vf.get(v, ())]

    def nbrs(self, u):
        return {w for i in self.vf.get(u, ()) for w in self.F[i] if w != u}

    def is_boundary_edge(self, u, v):
        return len(self.faces_of(u, v)) == 1

    def boundary_vertices(self):
        out = set()
        for u in self.alive_v:
            for w in self.nbrs(u):
                if self.is_boundary_edge(u, w):
                    out.add(u)
                    break
        return out

    def edges(self):
        E = set()
        for t in (self.F[i] for v in self.alive_v for i in self.vf[v]):
            for k in range(3):
                E.add(frozenset((t[k], t[(k + 1) % 3])))
        return E

    def nfaces(self):
        return len({i for v in self.alive_v for i in self.vf[v]})

    def fnormal(self, t, P=None):
        np = _np()
        P = P or self.P
        return np.cross(P[t[1]] - P[t[0]], P[t[2]] - P[t[0]])

    def _replace(self, i, t):
        for v in self.F[i]:
            self.vf[v].discard(i)
        self.F[i] = t
        if t is not None:
            for v in t:
                self.vf.setdefault(v, set()).add(i)

    def can_collapse(self, u, v, p, bnd):
        np = _np()
        shared = self.faces_of(u, v)
        if not shared or len(shared) > 2:
            return False
        if len(self.nbrs(u) & self.nbrs(v)) != len(shared):
            return False
        if u in bnd and v in bnd and len(shared) != 1:
            return False
        Q = list(self.P)
        Q[u] = p
        Q[v] = p
        for w in (u, v):
            for i in self.vf[w]:
                if i in shared:
                    continue
                t = self.F[i]
                old = self.fnormal(t)
                new = self.fnormal(t, Q)
                if np.dot(old, new) <= 1e-12 * (np.dot(old, old) + 1e-300):
                    return False
        return True

    def collapse(self, u, v, p):
        """Merge u into v at position p."""
        for i in list(self.faces_of(u, v)):
            self._replace(i, None)
        for i in list(self.vf.get(u, ())):
            self._replace(i, tuple(v if x == u else x for x in self.F[i]))
        self.P[v] = p
        self.vf.pop(u, None)
        self.alive_v.discard(u)

    def split(self, u, v):
        self.P.append((self.P[u] + self.P[v]) / 2)
        w = len(self.P) - 1
        self.alive_v.add(w)
        self.vf[w] = set()
        for i in list(self.faces_of(u, v)):
            t = self.F[i]
            k = next(k for k in range(3) if {t[k], t[(k + 1) % 3]} == {u, v})
            a, b, c = t[k], t[(k + 1) % 3], t[(k + 2) % 3]
            self._replace(i, (a, w, c))
            self.F.append((w, b, c))
            j = len(self.F) - 1
            for x in self.F[j]:
                self.vf.setdefault(x, set()).add(j)
        return w

    def flip(self, u, v):
        np = _np()
        sh = self.faces_of(u, v)
        if len(sh) != 2:
            return False
        t1, t2 = self.F[sh[0]], self.F[sh[1]]

        def has(t):
            return any((t[k], t[(k + 1) % 3]) == (u, v) for k in range(3))
        if not has(t1):
            t1, t2 = t2, t1
            sh = sh[::-1]
            if not has(t1):
                return False
        a = next(x for x in t1 if x not in (u, v))
        b = next(x for x in t2 if x not in (u, v))
        if a == b or b in self.nbrs(a):
            return False
        n1, n2 = (u, b, a), (v, a, b)
        old = self.fnormal(t1) + self.fnormal(t2)
        if np.dot(self.fnormal(n1), old) <= 0 or np.dot(self.fnormal(n2), old) <= 0:
            return False
        self._replace(sh[0], n1)
        self._replace(sh[1], n2)
        return True

    def result(self):
        tris = [t for t in self.F if t is not None]
        return compact(self.P, [("Triangle", t) for t in tris])


def _tri_input(m, P, f):
    if m.edim != 3 or m.top_dim() != 2:
        raise NotImplementedError(f"{f}: only triangle / polygon surface meshes in 3D are supported")
    tris = []
    for h, idx in m.cells:
        if M.CELL_DIM[h] != 2:
            raise NotImplementedError(f"{f}: mixed-dimensional meshes are not supported")
        tris += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    E = edge_faces(tris)
    if any(len(v) > 2 for v in E.values()):
        raise ValueError(f"{f}: the surface is not manifold (an edge has more than two faces)")
    return tris


def _euler_tris(tris):
    V = len({v for t in tris for v in t})
    return V - len(edge_faces(tris)) + len(tris)


# ═══════════════════════════════ SimplifyMesh ═══════════════════════════════

_SIMPLIFY_CRIT = ("MaxVertexCount", "MaxEdgeCount", "MaxTriangleCount", "MinEdgeLength")
_SIMPLIFY_UNSUP = ("MaxIncidentEdgeCount", "MinTriangleArea", "TriangleQuality", "MaxNormalDeviation",
                   "MaxHausdorffDistance")


def qem_simplify(T, crit, target, f):
    np = _np()
    Q = {v: np.zeros((4, 4)) for v in T.alive_v}
    bnd = T.boundary_vertices()
    for t in (T.F[i] for i in range(len(T.F)) if T.F[i] is not None):
        n = T.fnormal(t)
        a = np.linalg.norm(n)
        if a == 0:
            continue
        n = n / a
        p = np.append(n, -np.dot(n, T.P[t[0]]))
        K_ = np.outer(p, p) * a
        for v in t:
            Q[v] += K_
        for k in range(3):
            u, v = t[k], t[(k + 1) % 3]
            if T.is_boundary_edge(u, v):
                e = T.P[v] - T.P[u]
                m_ = np.cross(e, n)
                mn = np.linalg.norm(m_)
                if mn:
                    m_ = m_ / mn
                    p2 = np.append(m_, -np.dot(m_, T.P[u]))
                    for w in (u, v):
                        Q[w] += 1e3 * a * np.outer(p2, p2)

    def best(u, v):
        Qe = Q[u] + Q[v]
        A = Qe.copy()
        A[3] = [0, 0, 0, 1]
        cands = [T.P[u], T.P[v], (T.P[u] + T.P[v]) / 2]
        try:
            if abs(np.linalg.det(A)) > 1e-12:
                cands.insert(0, np.linalg.solve(A, [0, 0, 0, 1])[:3])
        except np.linalg.LinAlgError:
            pass
        if u in bnd and v not in bnd:
            cands = [T.P[u]]
        elif v in bnd and u not in bnd:
            cands = [T.P[v]]
        costs = [(float(np.append(c, 1) @ Qe @ np.append(c, 1)), k) for k, c in enumerate(cands)]
        c, k = min(costs)
        return max(c, 0.0), cands[k]

    def count():
        if crit == "MaxVertexCount":
            return len(T.alive_v)
        if crit == "MaxTriangleCount":
            return T.nfaces()
        return len(T.edges())
    ver = {}
    heap = []

    def push(u, v):
        e = frozenset((u, v))
        ver[e] = ver.get(e, 0) + 1
        if crit == "MinEdgeLength" and np.linalg.norm(T.P[u] - T.P[v]) >= target:
            return
        c, p = best(u, v)
        heapq.heappush(heap, (c, ver[e], tuple(sorted((u, v))), p))
    for e in T.edges():
        push(*tuple(e))
    cur = None if crit == "MinEdgeLength" else count()
    while heap:
        if crit != "MinEdgeLength" and cur <= target:
            break
        c, vv, (u, v), p = heapq.heappop(heap)
        e = frozenset((u, v))
        if ver.get(e) != vv or u not in T.alive_v or v not in T.alive_v:
            continue
        if not T.can_collapse(u, v, p, bnd):
            continue
        if u in bnd:
            u, v = v, u
        T.collapse(u, v, p)
        Q[v] = Q[v] + Q[u]
        for w in T.nbrs(v):
            push(v, w)
        if crit != "MinEdgeLength":
            cur = count()
    if crit != "MinEdgeLength" and count() > target:
        raise ValueError(f"{f}: the mesh cannot be reduced to {target} {crit[3:-5].lower()}s without changing its "
                         f"topology (reached {count()})")


@fn("SimplifyMesh", area=AREA, status="added",
    notes="SimplifyMesh[mesh] and SimplifyMesh[mesh, {crit, n}] for triangle / polygon surface meshes in 3D "
          "(polygons are fan-triangulated). Method: Garland-Heckbert quadric error metric edge collapse (1997), "
          "optimal placement (endpoint / midpoint fallback), link condition and face-normal flip check, boundary "
          "edges preserved by weighted perpendicular constraint planes and boundary vertices kept on the boundary. "
          "Criteria: {\"MaxVertexCount\" | \"MaxEdgeCount\" | \"MaxTriangleCount\", n} (0 < n < 1: proportion of the "
          "current count, rounded) and {\"MinEdgeLength\", l} (collapses edges shorter than l while possible); "
          "default {\"MaxTriangleCount\", 0.5} (Wolfram's default reduction is undocumented). The other documented "
          "criteria (MaxIncidentEdgeCount, MinTriangleArea, TriangleQuality, MaxNormalDeviation, "
          "MaxHausdorffDistance) raise NotImplementedError; an unreachable count ValueError; non-manifold input "
          "ValueError. The result keeps the input head and presentation options. Check: self-check that the "
          "count criterion holds and the Euler characteristic is unchanged (ArithmeticError otherwise); tests "
          "decimate a subdivided sphere and a flat grid (flat grids stay flat).")
def _simplify_mesh(mesh: Expr, *rest: ListOf[Expr] | Option) -> Expr:
    f = "SimplifyMesh"
    np = _np()
    pos, _o, keep = options(rest, f, set(), MESH_PRESENTATION)
    if len(pos) > 1:
        raise ValueError(f"{f}: expected SimplifyMesh[mesh, {{crit, n}}]")
    m, P = _mesh_arg(mesh, f)
    crit, nval = "MaxTriangleCount", 0.5
    if pos:
        c = pos[0]
        if not (K.is_list(c) and len(c.args) == 2 and isinstance(c.args[0], str)):
            if isinstance(c, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: the criterion must be {{\"name\", value}}, not {K_input(c)}")
        crit = c.args[0]
        if crit in _SIMPLIFY_UNSUP:
            raise NotImplementedError(f"{f}: the criterion \"{crit}\" is not supported")
        if crit not in _SIMPLIFY_CRIT:
            raise ValueError(f"{f}: unknown criterion \"{crit}\"")
        nval = real_arg(c.args[1], f, crit, 0, lo_open=True)
    tris = _tri_input(m, P, f)
    chi = _euler_tris(tris)
    T = TriMesh(P, tris)
    if crit == "MinEdgeLength":
        target = nval
    else:
        cur = {"MaxVertexCount": len(T.alive_v), "MaxTriangleCount": T.nfaces(), "MaxEdgeCount": len(T.edges())}[crit]
        if nval < 1:
            target = max(1, int(round(nval * cur)))
        elif float(nval).is_integer():
            target = int(nval)
        else:
            raise ValueError(f"{f}: {crit} must be a proportion between 0 and 1 or a positive integer")
    qem_simplify(T, crit, target, f)
    Q, cells = T.result()
    tri2 = [c for _, c in cells]
    got = {"MaxVertexCount": len(Q), "MaxTriangleCount": len(tri2), "MaxEdgeCount": len(edge_faces(tri2))}
    if _euler_tris(tri2) != chi or (crit in got and got[crit] > target):
        raise ArithmeticError(f"{f}: self-check failed (criterion or Euler characteristic)")
    return mesh_out(Q, cells, mesh.head, keep)


# ═══════════════════════════════ Remesh ═══════════════════════════════

def closest_on_tris(X, A, B, C):
    """Closest points of the rows of X on the triangles (A, B, C) (Ericson, Real-Time Collision Detection 5.1.5),
    vectorised over the triangles for each point."""
    np = _np()
    out_ = np.empty_like(X)
    AB, AC = B - A, C - A
    for n, p in enumerate(X):
        AP = p - A
        d1, d2 = (AB * AP).sum(1), (AC * AP).sum(1)
        BP = p - B
        d3, d4 = (AB * BP).sum(1), (AC * BP).sum(1)
        CP = p - C
        d5, d6 = (AB * CP).sum(1), (AC * CP).sum(1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        den = va + vb + vc
        den = np.where(den == 0, 1e-300, den)
        v = vb / den
        w = vc / den
        Q = A + AB * v[:, None] + AC * w[:, None]
        # regions outside: clamp to edges / vertices
        def seg(P0, P1):
            d = P1 - P0
            dd = (d * d).sum(1)
            t = np.clip(((p - P0) * d).sum(1) / np.where(dd == 0, 1, dd), 0, 1)
            return P0 + d * t[:, None]
        inside = (va >= 0) & (vb >= 0) & (vc >= 0)
        cand = [np.where(inside[:, None], Q, np.inf), seg(A, B), seg(B, C), seg(C, A)]
        best, bd = None, None
        for Qc in cand:
            dist = ((Qc - p) ** 2).sum(1)
            dist = np.where(np.isfinite(dist), dist, np.inf)
            k = int(np.argmin(dist))
            if bd is None or dist[k] < bd:
                best, bd = Qc[k], dist[k]
        out_[n] = best
    return out_


def feature_edges(T, cos_tol=math.cos(math.pi / 4)):
    """Boundary edges and sharp edges (dihedral angle between the face normals above 45 degrees)."""
    np = _np()
    out_ = set()
    for e in T.edges():
        u, v = tuple(e)
        sh = T.faces_of(u, v)
        if len(sh) != 2:
            out_.add(e)
            continue
        n1, n2 = T.fnormal(T.F[sh[0]]), T.fnormal(T.F[sh[1]])
        d = np.linalg.norm(n1) * np.linalg.norm(n2)
        if d == 0 or np.dot(n1, n2) / d < cos_tol:
            out_.add(e)
    return out_


def isotropic_remesh(T, Lt, iters, A, B, C):
    np = _np()
    hi, lo = 4 / 3 * Lt, 4 / 5 * Lt
    fe = feature_edges(T)
    fv = {v for e in fe for v in e}
    for _ in range(iters):
        # split long edges (feature edges stay feature edges)
        for e in list(T.edges()):
            u, v = tuple(e)
            if u in T.alive_v and v in T.alive_v and T.faces_of(u, v) and \
                    np.linalg.norm(T.P[u] - T.P[v]) > hi:
                w = T.split(u, v)
                if e in fe:
                    fe.discard(e)
                    fe.update({frozenset((u, w)), frozenset((w, v))})
                    fv.add(w)
        # collapse short edges: the removed vertex is never a feature vertex; new edges must stay below hi
        for e in sorted(T.edges(), key=lambda e: float(np.linalg.norm(T.P[min(e)] - T.P[max(e)]))):
            u, v = tuple(e)
            if u not in T.alive_v or v not in T.alive_v or not T.faces_of(u, v):
                continue
            if np.linalg.norm(T.P[u] - T.P[v]) >= lo:
                continue
            if u in fv:
                u, v = v, u
            if u in fv:
                continue
            p = T.P[v] if v in fv else (T.P[u] + T.P[v]) / 2
            nb = (T.nbrs(u) | T.nbrs(v)) - {u, v}
            if any(np.linalg.norm(T.P[w] - p) > hi for w in nb):
                continue
            if len(T.nbrs(u) & T.nbrs(v)) + 2 > len(nb) or len(T.alive_v) <= 4:
                continue
            if T.can_collapse(u, v, p, fv):
                T.collapse(u, v, p)
        # valence-improving flips (not across feature edges)
        def dev(x, val):
            return abs(val - (4 if x in fv else 6))
        for e in list(T.edges()):
            if e in fe:
                continue
            u, v = tuple(e)
            sh = T.faces_of(u, v)
            if len(sh) != 2:
                continue
            a = next(x for x in T.F[sh[0]] if x not in (u, v))
            b = next(x for x in T.F[sh[1]] if x not in (u, v))
            val = {x: len(T.nbrs(x)) for x in (u, v, a, b)}
            before = sum(dev(x, val[x]) ** 2 for x in val)
            after = dev(u, val[u] - 1) ** 2 + dev(v, val[v] - 1) ** 2 + dev(a, val[a] + 1) ** 2 + \
                dev(b, val[b] + 1) ** 2
            if after < before and val[u] > 3 and val[v] > 3:
                T.flip(u, v)
        # tangential relaxation of non-feature vertices, projected back onto the input surface
        newP = {}
        for x in T.alive_v:
            if x in fv:
                continue
            nb = T.nbrs(x)
            if not nb:
                continue
            c = np.mean([T.P[w] for w in nb], axis=0)
            n = sum(T.fnormal(T.F[i]) for i in T.vf[x])
            nn = np.linalg.norm(n)
            if nn:
                n = n / nn
                c = c - np.dot(c - T.P[x], n) * n
            newP[x] = c
        if newP:
            ks = list(newP)
            proj = closest_on_tris(np.array([newP[k] for k in ks]), A, B, C)
            old = {k: T.P[k] for k in ks}
            for k, q in zip(ks, proj):
                T.P[k] = q
            for k in ks:
                Q = list(T.P)
                Q[k] = old[k]
                if any(np.dot(T.fnormal(T.F[i]), T.fnormal(T.F[i], Q)) <= 0 for i in T.vf[k]):
                    T.P[k] = old[k]


@fn("Remesh", area=AREA, status="added",
    notes="Remesh[mesh] isotropic remeshing of a triangle / polygon surface mesh in 3D (Botsch-Kobbelt 2004): "
          "repeated passes of splitting edges longer than 4/3 L, collapsing edges shorter than 4/5 L, "
          "valence-improving edge flips (target 6 inside, 4 on the boundary), tangential relaxation and "
          "projection onto the input surface. Boundary edges and sharp edges (dihedral angle above 45 degrees) "
          "are feature lines: they are only split, their vertices stay fixed and are never collapsed away. "
          "Method -> \"Uniform\" (default) with suboptions \"EdgeLength\" (L, default the mean edge length) and "
          "\"MaxIterations\" (default 5); \"Adaptive\" raises NotImplementedError. Wolfram's remeshing parameters "
          "are undocumented, so cell counts differ. Planar, curve, solid and non-manifold meshes raise "
          "NotImplementedError / ValueError. Presentation options are kept. Check: self-check that the Euler "
          "characteristic is unchanged and every vertex lies on the input surface (ArithmeticError otherwise); "
          "tests verify the area of a remeshed flat square and edge lengths within [4/5 L, 4/3 L] for most "
          "edges.")
def _remesh(mesh: Expr, *opts: Option) -> Expr:
    f = "Remesh"
    np = _np()
    pos, o, keep = options(opts, f, {"Method"}, MESH_PRESENTATION - {"Method"})
    if pos:
        raise ValueError(f"{f}: expected Remesh[mesh, opts]")
    m, P = _mesh_arg(mesh, f)
    meth, sub = _method(o.get("Method"), f, ("Uniform", "Adaptive"), {"EdgeLength", "MaxIterations"})
    if meth == "Adaptive":
        raise NotImplementedError(f"{f}: Method -> \"Adaptive\" is not supported (use \"Uniform\")")
    tris = _tri_input(m, P, f)
    chi = _euler_tris(tris)
    lens = [np.linalg.norm(P[u] - P[v]) for u, v in (tuple(e) for e in edge_faces(tris))]
    Lt = _sub_real(sub, "EdgeLength", float(np.mean(lens)), f)
    it = _sub_int(sub, "MaxIterations", 5, f, 0)
    tri_arr = np.array(tris)
    A, B, C = P[tri_arr[:, 0]], P[tri_arr[:, 1]], P[tri_arr[:, 2]]
    if sum(np.linalg.norm(np.cross(B - A, C - A), axis=1)) / 2 / Lt ** 2 > 2e5:
        raise ValueError(f"{f}: the target edge length is too small for the surface (more than about 10^5 "
                         "triangles)")
    T = TriMesh(P, tris)
    isotropic_remesh(T, Lt, it, A, B, C)
    Q, cells = T.result()
    tri2 = [c for _, c in cells]
    Qa = np.array(Q)
    d = np.linalg.norm(closest_on_tris(Qa, A, B, C) - Qa, axis=1).max() if len(Qa) else 0
    diag = float(np.linalg.norm(P.max(0) - P.min(0))) or 1.0
    if _euler_tris(tri2) != chi or d > 1e-7 * diag:
        raise ArithmeticError(f"{f}: self-check failed (topology changed or vertices left the surface)")
    return mesh_out(Q, cells, mesh.head, keep)
