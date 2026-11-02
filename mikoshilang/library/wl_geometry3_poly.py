"""Wolfram batch X06 (part): fractal meshes (CantorMesh, MengerMesh, SierpinskiMesh), polyhedron operations
(GeodesicPolyhedron, BeveledPolyhedron, SnubPolyhedron, CanonicalizePolyhedron, OuterPolyhedron, InnerPolyhedron,
PolyhedronDecomposition) and SpherePoints.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).  The constructions are explicit and stated in each head's notes; where Wolfram's own
parameters are undocumented (bevel / snub ratios, point orderings) the defaults used here are given and may differ
from Wolfram's.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R
from .wl_graphics_color import K_input
from . import wl_geomprims as G
from .wl_geomprims import out, num, concrete
from .wl_geometry3_util import (int_arg, real_arg, options, is_auto, is_sym, coords_out, mesh_out, MESH_PRESENTATION,
                                fnum)
from . import wl_regions_mesh as M

AREA = "geometry: fractal meshes, polyhedra & sphere points (X06)"
MAX_CELLS = 200000


# ───────────────────────────── fractal meshes ─────────────────────────────

def _data_range(v, d, f):
    """[(lo, hi)] * d from a DataRange setting (Automatic -> None)."""
    if is_auto(v):
        return None
    rows = v.args if K.is_list(v) else None
    if rows is not None and d == 1 and len(rows) == 2 and not K.is_list(rows[0]):
        rows = [v]
    if rows is None or len(rows) != d or not all(K.is_list(r) and len(r.args) == 2 for r in rows):
        raise ValueError(f"{f}: DataRange must be Automatic or {{{{min, max}}, ...}} with {d} range(s)")
    res = []
    for r in rows:
        a = real_arg(r.args[0], f, "DataRange bound")
        b = real_arg(r.args[1], f, "DataRange bound")
        if not a < b:
            raise ValueError(f"{f}: each DataRange needs min < max")
        res.append((a, b))
    return res


def _rescale(P, box, rr):
    """Map the box [(lo, hi)] of the construction onto the ranges rr."""
    if rr is None:
        return P
    return [[rr[k][0] + (p[k] - box[k][0]) / (box[k][1] - box[k][0]) * (rr[k][1] - rr[k][0]) for k in range(len(p))]
            for p in P]


def _fractal_args(rest, f, dflt_d, dims):
    pos, o, keep = options(rest, f, {"DataRange"}, MESH_PRESENTATION)
    if len(pos) > 1:
        raise ValueError(f"{f}: expected {f}[n] or {f}[n, d]")
    d = int_arg(pos[0], f, "the dimension", 1) if pos else dflt_d
    if d not in dims:
        raise NotImplementedError(f"{f}: dimension {d} is not supported (only {', '.join(map(str, dims))})")
    return d, _data_range(o.get("DataRange"), d, f), keep


def _check_count(c, f):
    if c > MAX_CELLS:
        raise NotImplementedError(f"{f}: the mesh would have {c} cells (more than {MAX_CELLS} are not supported)")


def _box_mesh(boxes, d, rr, keep):
    """MeshRegion of axis-aligned boxes [(corner (Fractions), size)], coordinates sorted lexicographically."""
    corners = []
    for c, s in boxes:
        if d == 1:
            corners.append([(c[0],), (c[0] + s,)])
        elif d == 2:
            x, y = c
            corners.append([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])
        else:
            x, y, z = c
            corners.append([(x, y, z), (x + s, y, z), (x + s, y + s, z), (x, y + s, z),
                            (x, y, z + s), (x + s, y, z + s), (x + s, y + s, z + s), (x, y + s, z + s)])
    pts = sorted({p for cs in corners for p in cs})
    pos = {p: i for i, p in enumerate(pts)}
    head = {1: "Line", 2: "Polygon", 3: "Hexahedron"}[d]
    cells = [(head, tuple(pos[p] for p in cs)) for cs in corners]
    P = _rescale([[float(c) for c in p] for p in pts], [(0, 1)] * d, rr)
    return mesh_out(P, cells, opts=keep)


def _cantor_boxes(n, d):
    iv = [(Fraction(0), Fraction(1))]
    for _ in range(n):
        iv = [w for a, b in iv for w in ((a, a + (b - a) / 3), (b - (b - a) / 3, b))]
    s = iv[0][1] - iv[0][0]
    return [(tuple(a for a, _ in combo), s) for combo in itertools.product(iv, repeat=d)]


@fn("CantorMesh", area=AREA, status="added",
    notes="CantorMesh[n] / CantorMesh[n, d] (d = 1, 2, 3): the n-th step Cantor set (dust) from the unit interval / "
          "square / cube by removing middle thirds; cells Line (1-D), Polygon squares (2-D) or Hexahedron cubes "
          "(3-D), machine-real coordinates sorted lexicographically (as in Wolfram's examples). Option DataRange "
          "-> {{xmin, xmax}, ...} rescales; MeshRegion presentation options are kept. Symbolic n stays "
          "unevaluated; negative n ValueError; d > 3 NotImplementedError; more than 200000 cells "
          "NotImplementedError. Check: exact (construction); tests compare the measure (2/3)^(n d) (Wolfram "
          "documentation: lengths 2/3, 4/9, 8/27, ...) and the cell count 2^(n d).")
def _cantor_mesh(n: NonNegInteger, *rest: PosInteger | Option) -> Expr:
    f = "CantorMesh"
    k = int_arg(n, f, "the step n", 0)
    d, rr, keep = _fractal_args(rest, f, 1, (1, 2, 3))
    _check_count(2 ** (k * d), f)
    return _box_mesh(_cantor_boxes(k, d), d, rr, keep)


def _menger_boxes(n, d):
    boxes = [(tuple(Fraction(0) for _ in range(d)), Fraction(1))]
    for _ in range(n):
        nb = []
        for c, s in boxes:
            t = s / 3
            for idx in itertools.product(range(3), repeat=d):
                if sum(1 for i in idx if i == 1) >= 2:
                    continue
                nb.append((tuple(c[k] + idx[k] * t for k in range(d)), t))
        boxes = nb
    return boxes


@fn("MengerMesh", area=AREA, status="added",
    notes="MengerMesh[n] / MengerMesh[n, 2] (Sierpinski carpet: 8^n squares) and MengerMesh[n, 3] (Menger sponge: "
          "20^n cubes): subcells whose index has two or more middle (1) positions are removed at each step; unit "
          "square / cube, Polygon or Hexahedron cells, machine-real coordinates sorted lexicographically (matches "
          "Wolfram's MeshCoordinates[MengerMesh[1]] example). DataRange rescales; presentation options are kept. "
          "Symbolic n stays unevaluated; negative n ValueError; d other than 2 or 3 NotImplementedError; more than "
          "200000 cells NotImplementedError. Check: exact (construction); tests compare Area (8/9)^n and Volume "
          "(20/27)^n (Wolfram documentation: 8/9, 64/81, ...).")
def _menger_mesh(n: NonNegInteger, *rest: PosInteger | Option) -> Expr:
    f = "MengerMesh"
    k = int_arg(n, f, "the step n", 0)
    d, rr, keep = _fractal_args(rest, f, 2, (2, 3))
    _check_count((8 if d == 2 else 20) ** k, f)
    return _box_mesh(_menger_boxes(k, d), d, rr, keep)


_S3 = math.sqrt(3.0)
_SIERP2 = [(0.0, 0.0), (1.0, 0.0), (0.5, _S3 / 2)]
_SIERP3 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, _S3 / 2, 0.0), (0.5, _S3 / 6, math.sqrt(2.0 / 3.0))]


def _mid(a, b):
    return tuple((x + y) / 2 for x, y in zip(a, b))


def _sierpinski(n, d):
    simp = [tuple(_SIERP2)] if d == 2 else [tuple(_SIERP3)]
    for _ in range(n):
        new = []
        for s in simp:
            if d == 2:
                a, b, c = s
                ab, ac, bc = _mid(a, b), _mid(a, c), _mid(b, c)
                new += [(a, ab, ac), (ab, b, bc), (ac, bc, c)]
            else:
                a, b, c, e = s
                ab, ac, ae, bc, be, ce = _mid(a, b), _mid(a, c), _mid(a, e), _mid(b, c), _mid(b, e), _mid(c, e)
                new += [(a, ab, ac, ae), (ab, b, bc, be), (ac, bc, c, ce), (ae, be, ce, e)]
        simp = new
    return simp


@fn("SierpinskiMesh", area=AREA, status="added",
    notes="SierpinskiMesh[n] / SierpinskiMesh[n, 2]: the n-th step Sierpinski triangle (3^n Triangle cells) from the "
          "triangle (0,0), (1,0), (1/2, Sqrt[3]/2); SierpinskiMesh[n, 3]: the Sierpinski tetrahedron (4^n "
          "Tetrahedron cells) from the regular tetrahedron with base (0,0,0), (1,0,0), (1/2, Sqrt[3]/2, 0) and apex "
          "(1/2, Sqrt[3]/6, Sqrt[2/3]). Machine-real coordinates numbered in order of first appearance in the "
          "recursive subdivision (corner, edge midpoints), which reproduces Wolfram's MeshCoordinates of "
          "SierpinskiMesh[1]; the 3-D base tetrahedron is this batch's choice (Wolfram's is not documented). "
          "DataRange rescales the bounding box; presentation options are kept. Symbolic n stays unevaluated; "
          "negative n ValueError; other d NotImplementedError; more than 200000 cells NotImplementedError. "
          "Check: exact (construction); tests compare Area 3 Sqrt[3]/16, 9 Sqrt[3]/64, 27 Sqrt[3]/256 (Wolfram "
          "documentation) and Volume (1/2)^n Sqrt[2]/12.")
def _sierpinski_mesh(n: NonNegInteger, *rest: PosInteger | Option) -> Expr:
    f = "SierpinskiMesh"
    k = int_arg(n, f, "the step n", 0)
    d, rr, keep = _fractal_args(rest, f, 2, (2, 3))
    _check_count((3 if d == 2 else 4) ** k, f)
    simp = _sierpinski(k, d)
    key, P, cells = {}, [], []
    for s in simp:
        idx = []
        for p in s:
            q = tuple(round(c, 12) + 0.0 for c in p)
            if q not in key:
                key[q] = len(P)
                P.append(list(p))
            idx.append(key[q])
        cells.append(("Triangle" if d == 2 else "Tetrahedron", tuple(idx)))
    base = _SIERP2 if d == 2 else _SIERP3
    box = [(min(p[i] for p in base), max(p[i] for p in base)) for i in range(d)]
    return mesh_out(_rescale(P, box, rr), cells, opts=keep)


# ───────────────────────────── polyhedra ─────────────────────────────

def _unit_solid(name):
    """Vertices (floats, circumradius 1) and outward faces of a base solid for GeodesicPolyhedron."""
    import numpy as np
    V0 = G._platonic_vertices(name)
    V = np.array([[num(c) for c in p] for p in V0], dtype=float)
    V /= np.linalg.norm(V, axis=1)[:, None]
    Fc = G._hull_faces(V0)
    F = []
    for fc in Fc:
        a, b, c = V[fc[0]], V[fc[1]], V[fc[2]]
        if np.dot(np.cross(b - a, c - a), a) < 0:
            fc = fc[::-1]
        F.append(list(fc))
    return V, F


@fn("GeodesicPolyhedron", area=AREA, status="added",
    notes="GeodesicPolyhedron[n] (= GeodesicPolyhedron[\"Icosahedron\", n]) and GeodesicPolyhedron[\"poly\", n] "
          "for \"Tetrahedron\", \"Octahedron\", \"Icosahedron\": every face of the solid inscribed in the unit "
          "sphere is divided into n^2 triangles by the frequency-n barycentric grid and the grid points are "
          "projected radially onto the unit sphere (Class I geodesic, as described in Wolfram's documentation); "
          "10 n^2 + 2 vertices and 20 n^2 faces for the icosahedron. Machine-real coordinates; outward faces. The "
          "orientation of the base solid is this batch's (Wolfram's is not documented), so coordinates may differ "
          "by a rotation; volumes and areas match (Volume[GeodesicPolyhedron[2]] = 3.65871). Polyhedron options "
          "(VertexColors, ...) are kept. Symbolic n stays unevaluated; n < 1 ValueError; other names "
          "ValueError. Check: self-check (all vertices on the unit sphere and V - E + F == 2; ArithmeticError) "
          "and reference volumes / areas from the Wolfram documentation in tests.")
def _geodesic_polyhedron(*args: String | PosInteger | Option) -> Expr:
    f = "GeodesicPolyhedron"
    import numpy as np
    pos, _o, keep = options(args, f, set(), {"BaseStyle", "TextureMapping", "VertexColors", "VertexNormals",
                                             "VertexTextureCoordinates"})
    if len(pos) == 1:
        name, n = "Icosahedron", pos[0]
    elif len(pos) == 2:
        name, n = pos
        if not isinstance(name, str):
            if isinstance(name, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: the base polyhedron must be a string such as \"Icosahedron\", not "
                            f"{K_input(name)}")
        if name not in ("Tetrahedron", "Octahedron", "Icosahedron"):
            raise ValueError(f"{f}: the base polyhedron must be \"Tetrahedron\", \"Octahedron\" or \"Icosahedron\"")
    else:
        raise ValueError(f"{f}: expected GeodesicPolyhedron[n] or GeodesicPolyhedron[\"poly\", n]")
    k = int_arg(n, f, "the order n", 1)
    if 20 * k * k > MAX_CELLS:
        raise NotImplementedError(f"{f}: order {k} gives too many faces")
    V, F = _unit_solid(name)
    key, P, faces = {}, [], []

    def vid(p):
        p = p / np.linalg.norm(p)
        q = tuple(np.round(p, 11) + 0.0)
        if q not in key:
            key[q] = len(P)
            P.append(p)
        return key[q]
    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        grid = {}
        for i in range(k + 1):
            for j in range(k + 1 - i):
                grid[i, j] = vid(A + (B - A) * i / k + (C - A) * j / k)
        for i in range(k):
            for j in range(k - i):
                faces.append([grid[i, j], grid[i + 1, j], grid[i, j + 1]])
                if i + j < k - 1:
                    faces.append([grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
    P = np.array(P)
    E = {frozenset((fc[i], fc[(i + 1) % 3])) for fc in faces for i in range(3)}
    if np.max(np.abs(np.linalg.norm(P, axis=1) - 1)) > 1e-12 or len(P) - len(E) + len(faces) != 2:
        raise ArithmeticError(f"{f}: result failed verification")
    return Expr("Polyhedron", coords_out(P), L(*[L(*[i + 1 for i in fc]) for fc in faces]), *keep)


def _closed_poly(poly, f):
    """(V exact, F outward) of a closed polyhedron given as a polyhedron expression or a 3-D BoundaryMeshRegion."""
    if isinstance(poly, Expr) and poly.head in ("BoundaryMeshRegion", "MeshRegion"):
        m = M.mesh_of(poly, f)
        if m.edim != 3 or any(M.CELL_DIM[h] != 2 for h, _ in m.cells):
            raise ValueError(f"{f}: a mesh argument must be a closed surface of polygons in 3-D")
        V, F = m.P, [list(idx) for _, idx in m.cells]
    else:
        V, F = G._poly_arg(poly, f, need_orientation=True)
    concrete(V)
    return V, G._orient(V, F, f)


def _ratio(rest, f, dflt, hi):
    pos, _o, keep = options(rest, f, set(), {"VertexColors", "VertexNormals", "VertexTextureCoordinates"})
    if len(pos) > 1:
        raise ValueError(f"{f}: expected {f}[poly] or {f}[poly, l]")
    if not pos:
        return dflt, keep, ()
    v = G.sx(pos[0], f, "ratio")
    concrete(v)
    if not (0 < num(v) < hi):
        raise ValueError(f"{f}: the ratio must satisfy 0 < l < {hi}")
    return v, keep, (pos[0],)


def _euler(Vn, F):
    E = {frozenset((fc[k], fc[(k + 1) % len(fc)])) for fc in F for k in range(len(fc))}
    return Vn - len(E) + len(F)


def _poly_expr(V, F, inputs, keep):
    used = sorted({i for fc in F for i in fc})
    pos = {k: j for j, k in enumerate(used)}
    return Expr("Polyhedron", out([V[k] for k in used], *inputs), L(*[L(*[pos[i] + 1 for i in fc]) for fc in F]),
                *keep)


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _scl(t, a):
    return [t * x for x in a]


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _corner_faces(F):
    """{vertex: [(face index, position of the vertex in the face)] in cyclic order around the vertex}."""
    nxt = {}
    for fi, fc in enumerate(F):
        k = len(fc)
        for i in range(k):
            # around vertex v = fc[i]: the face fi is followed by the face across the edge (v, fc[i - 1])
            nxt.setdefault(fc[i], {})[fc[(i + 1) % k]] = (fc[i - 1], fi, i)
    res = {}
    for v, d in nxt.items():
        start = next(iter(d))
        order, w = [], start
        while True:
            w2, fi, i = d[w]
            order.append((fi, i))
            w = w2
            if w == start:
                break
        if len(order) != len(d):
            raise ValueError("the polyhedron has a singular vertex (its faces do not form a single fan)")
        res[v] = order
    return res


def _edge_faces(F):
    """{(v, w) directed: (face index, position i with F[fi][i] = v, F[fi][i+1] = w)}."""
    res = {}
    for fi, fc in enumerate(F):
        for i in range(len(fc)):
            res[(fc[i], fc[(i + 1) % len(fc)])] = (fi, i)
    return res


@fn("BeveledPolyhedron", area=AREA, status="added",
    notes="BeveledPolyhedron[poly, l] for a closed, orientable polyhedron (Polyhedron, Cube, Cuboid, Tetrahedron"
          "[{...}], Hexahedron, Prism, Pyramid, Simplex, or a 3-D BoundaryMeshRegion): the Conway bevel (edge "
          "truncation) built as the truncation, at ratio l along its edges, of the rectification (edge-midpoint "
          "polyhedron): one vertex per flag (face, vertex, edge), a 2k-gon for every k-gon face, a 2d-gon for every "
          "degree-d vertex and a quadrilateral for every edge (dodecahedron: 120 points and 62 faces, tetrahedron: "
          "24 and 14, as in Wolfram's examples). 0 < l < 1/2; BeveledPolyhedron[poly] uses l = 1/3 (Wolfram's "
          "default ratio is undocumented, so default coordinates may differ). Vertex and edge faces are planar for "
          "symmetric inputs but need not be planar in general (as in Wolfram's mesh example). Regular solids "
          "other than the axis-aligned Cube raise NotImplementedError (Wolfram's orientation is not available); "
          "open or singular surfaces ValueError; symbolic coordinates stay unevaluated. Check: self-check (the "
          "result is a closed surface with the Euler characteristic of the input; ArithmeticError) and the "
          "documented point / face counts in tests.")
def _beveled_polyhedron(poly: Expr, *rest: Real | Option) -> Expr:
    f = "BeveledPolyhedron"
    l_, keep, inputs = _ratio(rest, f, sp.Rational(1, 3), 0.5)
    V, F = _closed_poly(poly, f)
    newV, key = [], {}

    def mid(a, b):
        return _scl(sp.Rational(1, 2), _add(V[a], V[b]))

    def P(fi, i, j):
        """Point on the rectified edge from midpoint of edge i to midpoint of edge j of face fi."""
        k = (fi, i, j)
        if k not in key:
            fc = F[fi]
            n = len(fc)
            mi, mj = mid(fc[i], fc[(i + 1) % n]), mid(fc[j], fc[(j + 1) % n])
            key[k] = len(newV)
            newV.append([sp.nsimplify(c) if not isinstance(c, sp.Float) else c for c in _add(mi, _scl(l_, _sub(mj, mi)))])
        return key[k]

    newF = []
    for fi, fc in enumerate(F):
        n = len(fc)
        face = []
        for i in range(n):
            face += [P(fi, i, (i + 1) % n), P(fi, (i + 1) % n, i)]
        newF.append(face)
    try:
        corners = _corner_faces(F)
    except ValueError as e:
        raise ValueError(f"{f}: {e}")
    for v, around in corners.items():
        ring = []
        for fi, i in around:
            n = len(F[fi])
            # edges of face fi at vertex v = F[fi][i]: edge i (v -> next) and edge i - 1 (prev -> v)
            ring += [P(fi, i, (i - 1) % n), P(fi, (i - 1) % n, i)]
        newF.append(ring[::-1])
    ef = _edge_faces(F)
    for (a, b), (fi, i) in ef.items():
        if a > b:
            continue
        gi, j = ef[(b, a)]
        n1, n2 = len(F[fi]), len(F[gi])
        newF.append([P(fi, i, (i + 1) % n1), P(fi, i, (i - 1) % n1), P(gi, j, (j + 1) % n2), P(gi, j, (j - 1) % n2)])
    try:
        newF = G._orient(newV, newF, f)
    except ValueError:
        raise ArithmeticError(f"{f}: result failed verification")
    if _euler(len(newV), newF) != _euler(len({i for fc in F for i in fc}), F):
        raise ArithmeticError(f"{f}: result failed verification")
    return _poly_expr(newV, newF, (poly,) + inputs, keep)


@fn("SnubPolyhedron", area=AREA, status="added",
    notes="SnubPolyhedron[poly, l] for a closed, orientable polyhedron: the Conway snub with one vertex per corner "
          "(face f, vertex v) at q = v + (l/2) (w - v) + (l/2) (c - v), where w follows v in f and c is the vertex "
          "average of f; faces are the shrunken, twisted original faces, one face per original vertex and two "
          "triangles per original edge (cube: 24 points, 38 faces). 0 < l < 1; SnubPolyhedron[poly] uses l = 1/2 "
          "(Wolfram's construction parameters are undocumented, so coordinates differ from Wolfram's; the "
          "combinatorics are those of the snub). Vertex faces of degree > 3 need not be planar for irregular "
          "inputs. Same input forms and errors as BeveledPolyhedron. Check: self-check (closed surface with the "
          "input's Euler characteristic; ArithmeticError) and face counts in tests.")
def _snub_polyhedron(poly: Expr, *rest: Real | Option) -> Expr:
    f = "SnubPolyhedron"
    l_, keep, inputs = _ratio(rest, f, sp.Rational(1, 2), 1)
    V, F = _closed_poly(poly, f)
    h = l_ / 2
    q, newV = {}, []
    for fi, fc in enumerate(F):
        n = len(fc)
        c = _scl(sp.Rational(1, n), [sum(V[i][k] for i in fc) for k in range(3)])
        for i in range(n):
            v, w = V[fc[i]], V[fc[(i + 1) % n]]
            q[fi, i] = len(newV)
            newV.append([sp.nsimplify(x) if not isinstance(x, sp.Float) else x
                         for x in _add(v, _add(_scl(h, _sub(w, v)), _scl(h, _sub(c, v))))])
    newF = [[q[fi, i] for i in range(len(fc))] for fi, fc in enumerate(F)]
    try:
        corners = _corner_faces(F)
    except ValueError as e:
        raise ValueError(f"{f}: {e}")
    for v, around in corners.items():
        newF.append([q[fi, i] for fi, i in around][::-1])
    ef = _edge_faces(F)
    for (a, b), (fi, i) in ef.items():
        if a > b:
            continue
        gi, j = ef[(b, a)]
        i1, j1 = (i + 1) % len(F[fi]), (j + 1) % len(F[gi])
        newF.append([q[fi, i], q[gi, j], q[fi, i1]])
        newF.append([q[fi, i], q[gi, j1], q[gi, j]])
    try:
        newF = G._orient(newV, newF, f)
    except ValueError:
        raise ArithmeticError(f"{f}: result failed verification")
    if _euler(len(newV), newF) != _euler(len({i for fc in F for i in fc}), F):
        raise ArithmeticError(f"{f}: result failed verification")
    return _poly_expr(newV, newF, (poly,) + inputs, keep)


# ── canonical form: shells, nesting, outer and inner boundaries ──

def _np(V):
    import numpy as np
    return np.array([[num(c) for c in p] for p in V], dtype=float)


def _faces_cross(Vn, F, f):
    """True when two faces that share no vertex intersect (float test with a relative tolerance)."""
    import numpy as np
    tris = [(fc[0], fc[k], fc[k + 1]) for fc in F for k in range(1, len(fc) - 1)]
    tri_face = [fi for fi, fc in enumerate(F) for k in range(1, len(fc) - 1)]
    edges = {}
    for fi, fc in enumerate(F):
        for k in range(len(fc)):
            e = frozenset((fc[k], fc[(k + 1) % len(fc)]))
            edges.setdefault(e, set()).add(fi)
    if not tris:
        return False
    T = np.array(tris)
    A, B, C = Vn[T[:, 0]], Vn[T[:, 1]], Vn[T[:, 2]]
    scale = max(1.0, float(np.abs(Vn).max()))
    eps = 1e-9 * scale
    face_verts = [set(fc) for fc in F]
    for e, fs in edges.items():
        a, b = tuple(e)
        p, d = Vn[a], Vn[b] - Vn[a]
        # Moller-Trumbore for all triangles at once
        e1, e2 = B - A, C - A
        h = np.cross(np.broadcast_to(d, e2.shape), e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-14 * scale ** 2
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = p - A
        u = np.einsum("ij,ij->i", s, h) * inv
        qv = np.cross(s, e1)
        v = (qv @ d) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        hit = ok & (u > 1e-9) & (v > 1e-9) & (u + v < 1 - 1e-9) & (t > 1e-9) & (t < 1 - 1e-9)
        for ti in np.nonzero(hit)[0]:
            fi = tri_face[ti]
            if a in face_verts[fi] or b in face_verts[fi] or fi in fs:
                continue
            return True
    return False


def _shells(poly, f):
    """Canonical decomposition: (V, [(outer faces, [void faces, ...]), ...]) with every shell oriented outward
    (as a standalone polyhedron)."""
    V, F = G._poly_arg(poly, f, need_orientation=True)
    concrete(V)
    F = G._orient(V, F, f)
    comps = G._components(F)
    Vn = _np(V)
    if _faces_cross(Vn, F, f):
        raise NotImplementedError(f"{f}: polyhedra with intersecting faces are not supported")
    from .wl_regions import PolyhedronR
    parts = [PolyhedronR(V, [F[i] for i in c], f=f) for c in comps]
    shells = [[F[i] for i in c] for c in comps]
    n = len(shells)
    inside = [[False] * n for _ in range(n)]
    for i in range(n):
        p = V[shells[i][0][0]]
        for j in range(n):
            if i != j:
                inside[i][j] = bool(parts[j]._inside_closed(p))
    depth = [sum(inside[i]) for i in range(n)]
    res = []
    for i in range(n):
        if depth[i] % 2:
            continue
        voids = [shells[j] for j in range(n) if depth[j] == depth[i] + 1 and inside[j][i]]
        res.append((shells[i], voids))
    return V, res


def _canon_expr(V, entries, inputs):
    used = []
    for outer, voids in entries:
        for fc in outer + [fc for v in voids for fc in v]:
            for i in fc:
                if i not in used:
                    used.append(i)
    order = sorted(used)
    pos = {k: j + 1 for j, k in enumerate(order)}

    def fl_(F):
        return L(*[L(*[pos[i] for i in fc]) for fc in F])
    items = []
    for outer, voids in entries:
        items.append(Expr("Rule", fl_(outer), L(*[fl_(v) for v in voids])) if voids else fl_(outer))
    body = items[0] if len(items) == 1 else L(*items)
    return Expr("Polyhedron", out([V[k] for k in order], *inputs), body)


@fn("CanonicalizePolyhedron", area=AREA, status="added",
    notes="CanonicalizePolyhedron[poly] gives Polyhedron[{p1, ...}, faces] with shared coordinates (duplicates "
          "merged, kept in order of first appearance, which reproduces Wolfram's PolyhedronCoordinates example), "
          "every closed shell oriented outward, and nesting resolved: one shell gives Polyhedron[pts, {f1, ...}], "
          "a shell with voids Polyhedron[pts, outer -> {inner1, ...}] (voids oriented as standalone polyhedra, as "
          "in Wolfram's example), several shells a list of such entries. Accepts Polyhedron in all forms, "
          "Tetrahedron[{...}], Cube, Cuboid, Hexahedron, Prism, Pyramid, Simplex. Faces that cross other faces "
          "raise NotImplementedError; open surfaces ValueError; regular solids other than the axis-aligned Cube "
          "NotImplementedError (orientation unknown); symbolic coordinates stay unevaluated. Check: exact "
          "(orientation by exact signed volume, nesting by exact point-in-polyhedron); tests compare Volume with "
          "the input and the documented void example (27 - 1 = 26).")
def _canonicalize_polyhedron(poly: Expr) -> Expr:
    f = "CanonicalizePolyhedron"
    V, entries = _shells(poly, f)
    return _canon_expr(V, entries, (poly,))


@fn("OuterPolyhedron", area=AREA, status="added",
    notes="OuterPolyhedron[poly] gives the outer shells of the canonical form (CanonicalizePolyhedron) as "
          "Polyhedron[pts, {{faces1}, ...}] (one shell: Polyhedron[pts, faces]), dropping voids; coordinates in "
          "order of first appearance. Same input forms and errors as CanonicalizePolyhedron. Check: exact; tests "
          "compare with the input for simple polyhedra (Wolfram documentation) and Volume of the outer cube of the "
          "void example (27).")
def _outer_polyhedron(poly: Expr) -> Expr:
    f = "OuterPolyhedron"
    V, entries = _shells(poly, f)
    return _canon_expr(V, [(o, []) for o, _ in entries], (poly,))


@fn("InnerPolyhedron", area=AREA, status="added",
    notes="InnerPolyhedron[poly] gives the voids of the canonical form as Polyhedron[pts, faces] (each void oriented "
          "as a standalone polyhedron; several voids as a list of face lists), or EmptyRegion[3] when there is "
          "none (Wolfram documentation). Same input forms and errors as CanonicalizePolyhedron. Check: exact; "
          "tests compare the void of the documented example (Volume 1) and EmptyRegion[3] for simple polyhedra.")
def _inner_polyhedron(poly: Expr) -> Expr:
    f = "InnerPolyhedron"
    V, entries = _shells(poly, f)
    voids = [v for _, vs in entries for v in vs]
    if not voids:
        return Expr("EmptyRegion", 3)
    return _canon_expr(V, [(v, []) for v in voids], (poly,))


def _tet_volume(a, b, c, d):
    return G._dot(G._cross(_sub(b, a), _sub(c, a)), _sub(d, a)) / 6


@fn("PolyhedronDecomposition", area=AREA, status="added",
    notes="PolyhedronDecomposition[poly] for a union of disjoint convex shells (no voids) gives a list of "
          "tetrahedra Polyhedron[{p1, p2, p3, p4}, faces] with disjoint interiors: each shell is coned from its "
          "first vertex over a fan triangulation of the faces not containing it (the unit cube gives 6 "
          "tetrahedra, as in Wolfram's example). Non-convex shells and polyhedra with voids raise "
          "NotImplementedError (they need a constrained tetrahedralization); other errors as "
          "CanonicalizePolyhedron. Check: self-check (the tetrahedron volumes add up exactly to each shell's "
          "volume; ArithmeticError).")
def _polyhedron_decomposition(poly: Expr) -> ListOf[Expr]:
    f = "PolyhedronDecomposition"
    V, entries = _shells(poly, f)
    res = []
    for outer, voids in entries:
        if voids:
            raise NotImplementedError(f"{f}: polyhedra with voids are not supported")
        if not G._convex_polyhedron(V, outer, f):
            raise NotImplementedError(f"{f}: non-convex polyhedra are not supported")
        v0 = outer[0][0]
        vol = sp.Integer(0)
        tets = []
        for fc in outer:
            if v0 in fc:
                continue
            for k in range(1, len(fc) - 1):
                t = [v0, fc[0], fc[k], fc[k + 1]]
                w = _tet_volume(*[V[i] for i in t])
                if G.fsign(w) == 0:
                    continue
                if G.fsign(w) < 0:
                    t = [t[0], t[1], t[3], t[2]]
                    w = -w
                vol += w
                tets.append(t)
        total = G._signed_volume(V, outer)
        if G.fsign(sp.simplify(vol - total)) != 0:
            raise ArithmeticError(f"{f}: result failed verification")
        for t in tets:
            res.append(Expr("Polyhedron", out([V[i] for i in t], poly),
                            L(*[L(*fc) for fc in G._TET_FACES])))
    return L(*res)


# ───────────────────────────── SpherePoints ─────────────────────────────

def _exact_sphere_points(n):
    one, z = sp.Integer(1), sp.Integer(0)
    if n == 1:
        return [[z, z, one]]
    if n == 3:
        return [[sp.cos(2 * sp.pi * k / 3), sp.sin(2 * sp.pi * k / 3), z] for k in range(3)]
    if n == 4:
        s = 1 / sp.sqrt(3)
        return [[s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]]
    if n == 12:
        phi = (1 + sp.sqrt(5)) / 2
        r = sp.sqrt(1 + phi ** 2)
        V = []
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                V += [[z, s1 / r, s2 * phi / r], [s1 / r, s2 * phi / r, z], [s2 * phi / r, z, s1 / r]]
        return [[sp.radsimp(c) for c in p] for p in V]
    if 2 <= n <= 7 and n != 3 and n != 4:
        m = n - 2
        P = [[z, z, -one], [z, z, one]]
        ang = [0]
        for k in range(1, m):
            ang.append(-((k + 1) // 2) if k % 2 else (k + 1) // 2)
        for a in ang[:m]:
            t = 2 * sp.pi * a / m
            P.append([sp.cos(t), sp.sin(t), z])
        return P
    return None


@fn("SpherePoints", area=AREA, status="added",
    notes="SpherePoints[n] gives n points on the unit sphere: exactly equally spaced, exact coordinates, for n = 1, "
          "2 (poles), 3 (equilateral triangle on the equator), 4 (regular tetrahedron), 5, 6, 7 (the poles and a "
          "regular (n - 2)-gon on the equator: triangular bipyramid, octahedron, pentagonal bipyramid; "
          "SpherePoints[7] reproduces Wolfram's documented output) and 12 (icosahedron); otherwise the Fibonacci "
          "(golden-spiral) lattice z_k = 1 - (2 k + 1)/n, longitude k * 2 Pi/phi^2 (approximately equally spaced, "
          "machine reals; Wolfram's algorithm is undocumented, so these coordinates differ from Wolfram's). Option "
          "WorkingPrecision -> MachinePrecision or p (digits). Symbolic n stays unevaluated; n < 1 ValueError. "
          "Check: exact for the polyhedral cases (unit norms), self-check (unit norms; ArithmeticError) for the "
          "lattice; tests check the minimum pairwise distance.")
def _sphere_points(n: PosInteger, *opts: Option) -> MatrixOf[Real]:
    f = "SpherePoints"
    pos, o, _ = options((n,) + tuple(opts), f, {"WorkingPrecision"})
    k = int_arg(pos[0], f, "the number of points", 1)
    wp = o.get("WorkingPrecision")
    prec = None
    if wp is not None:
        if is_sym(wp, "MachinePrecision"):
            prec = "machine"
        else:
            prec = int_arg(wp, f, "WorkingPrecision", 1) if K.is_int(wp) else real_arg(wp, f, "WorkingPrecision", 1)
    P = _exact_sphere_points(k)
    if P is None:
        import numpy as np
        if prec is not None and prec != "machine":
            g = (3 - sp.sqrt(5)) * sp.pi
            P = []
            for j in range(k):
                z_ = 1 - sp.Rational(2 * j + 1, k)
                r = sp.sqrt(1 - z_ ** 2)
                P.append([r * sp.cos(g * j), r * sp.sin(g * j), z_])
            return L(*[L(*[R(sp.Float(sp.N(c, int(prec) + 5), int(prec))) for c in p]) for p in P])
        j = np.arange(k)
        z_ = 1 - (2 * j + 1) / k
        r = np.sqrt(1 - z_ ** 2)
        th = j * math.pi * (3 - math.sqrt(5))
        A = np.stack([r * np.cos(th), r * np.sin(th), z_], axis=1)
        if np.max(np.abs(np.linalg.norm(A, axis=1) - 1)) > 1e-12:
            raise ArithmeticError(f"{f}: result failed verification")
        return coords_out(A)
    if prec is None:
        return out(P)
    if prec == "machine":
        return L(*[L(*[fnum(float(sp.N(c))) for c in p]) for p in P])
    return L(*[L(*[R(sp.Float(sp.N(c, int(prec) + 5), int(prec))) for c in p]) for p in P])
