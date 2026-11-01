"""Differential tests for Wolfram batch X06 (geometry III: mesh processing, fractal meshes, polyhedra and geometric
solvers; modules wl_geometry3, wl_geometry3_repair, wl_geometry3_recon, wl_geometry3_curves, wl_geometry3_scene and
wl_geometry3_poly).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented argument forms and
  examples, e.g. ``FindShortestCurve[Circle[], {1, 0}, {0, 1}] == Circle[{0, 0}, 1, {0, Pi/2}]``,
  ``MeshCellQuality`` of the triangle {{1, 0}, {2, 1}, {0, 1}} = 0.866025, ``Volume[GeodesicPolyhedron[2]] =
  3.65871`` and ``SurfaceArea[GeodesicPolyhedron[2]] = 11.6659``, ``SpherePoints[7]`` (two poles and a regular
  pentagon), CantorMesh / MengerMesh / SierpinskiMesh measures (2/3)^n, (8/9)^n, 3 Sqrt[3]/16 ..., the documented
  default of MengerMesh (the 2-D carpet), ``CanonicalizePolyhedron[Cube[]]`` = the cube itself, ``InnerPolyhedron``
  of a polyhedron without voids = EmptyRegion[3].
* Classical geometry computed by hand: Chaikin corner cutting of the unit square (area 7/8), Loop / Catmull-Clark
  vertex counts (V + E (+ F)), Euler characteristics, the truncated cuboctahedron (48 vertices, 26 faces, the bevel of
  a cube), the snub cube (24 vertices, 38 faces), triangle quality 4 Sqrt[3] A / (sum of squared edges), the 3-4-5
  triangle, Thales' theorem, the isosceles right triangle example (area 48), Ceva / mass-point ratios (12/5),
  shortest paths (Sqrt[5] across a rectangle, Pi/2 on the unit circle and sphere), least-squares lines / circles /
  planes through exact data.
* SciPy (independent Qhull) only to produce the Delaunay triangulations the mesh routines start from; paths that
  need SciPy are filtered out on the minimal install and replaced by ``Raises(NotImplementedError)`` cases.

Symbol names used as placeholders are prefixed ``x06`` so they cannot collide with other test modules.
"""
import importlib.util
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None

T, V, A, N = TypeError, ValueError, ArithmeticError, NotImplementedError


def _fl(e):
    if isinstance(e, Expr) and e.head == "List":
        return [x for a in e.args for x in _fl(a)]
    return [float(ev(f"N[{e}]")) if isinstance(e, Expr) else float(e)]


def Num(v, tol=1e-9):
    def ok(got):
        try:
            return abs(_fl(got)[0] - v) <= tol * max(1.0, abs(v))
        except Exception:
            return False
    return Pred(ok, f"~{v}")


def NumList(vs, tol=1e-9):
    def ok(got):
        try:
            g = _fl(got)
        except Exception:
            return False
        return len(g) == len(vs) and all(abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(g, vs))
    return Pred(ok, f"~{vs}")


def is_head(head, n=None):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head and (n is None or len(g.args) == n), f"{head}[...]")


TRUE = Pred(lambda g: g is True, "True")

OPEN_TET = "MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {Triangle[{1,2,3}],Triangle[{1,2,4}],Triangle[{1,3,4}]}]"
TET = "MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {Triangle[{1,3,2}],Triangle[{1,2,4}],Triangle[{1,4,3}],Triangle[{2,3,4}]}]"
FLIP_TET = "MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {Triangle[{1,3,2}],Triangle[{1,2,4}],Triangle[{1,4,3}],Triangle[{2,4,3}]}]"
SQ2 = "MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, {Triangle[{1,2,3}],Triangle[{1,3,4}]}]"
TRI = "MeshRegion[{{1,0},{2,1},{0,1}}, Triangle[{1,2,3}]]"
CUBE = "BoundaryDiscretizeRegion[Cube[]]"
SUBCUBE = "SubdivisionRegion[BoundaryDiscretizeRegion[Cube[]], 3]"
VOID = ("Polyhedron[{{0,0,0},{2,0,0},{0,2,0},{0,0,2},{1/2,1/2,1/2},{1,1/2,1/2},{1/2,1,1/2},{1/2,1/2,1}}, "
        "{{1,3,2},{1,2,4},{1,4,3},{2,3,4}} -> {{{5,6,7},{5,8,6},{5,7,8},{6,8,7}}}]")
S345 = "GeometricScene[{a, b, c}, {a == {0, 0}, b == {3, 0}, PlanarAngle[{b, a, c}] == 90 Degree, EuclideanDistance[a, c] == 4}]"
THALES = "GeometricScene[{a, b, c, o}, {CircleThrough[{a, b, c}, o], o == Midpoint[{a, c}]}]"

CASES = [
    # ── MeshConnectivityGraph (exact combinatorics; hand counts)
    ("VertexCount[MeshConnectivityGraph[%s, 2]]" % SQ2, "2"),                       # two triangles
    ("EdgeCount[MeshConnectivityGraph[%s, 0]]" % SQ2, "5"),                         # 4 sides + diagonal
    ("EdgeCount[MeshConnectivityGraph[%s, 1, 2]]" % SQ2, "6"),                      # 3 edges per triangle
    ("EdgeCount[MeshConnectivityGraph[%s]]" % TRI, "3"),                            # Check: K3 between edges
    ("MeshConnectivityGraph[x06m]", "MeshConnectivityGraph[x06m]"),
    ("MeshConnectivityGraph[3]", Raises(T)),
    ("MeshConnectivityGraph[%s, 5]" % TRI, Raises(V)),
    # ── MeshCellQuality (Wolfram doc: 0.866025; 4 Sqrt[3] A / sum l^2; tet 6 Sqrt[2] V / l_rms^3)
    ("MeshCellQuality[%s]" % TRI, "MeshCellQuality[%s]" % TRI),                     # inert annotation key
    ("AnnotationValue[{%s, 2}, MeshCellQuality]" % TRI, NumList([math.sqrt(3) / 2], 1e-12)),
    ("AnnotationValue[{MeshRegion[{{0,0},{1,0},{1/2,Sqrt[3]/2}}, Triangle[{1,2,3}]], 2}, MeshCellQuality]",
     NumList([1.0], 1e-12)),                                                         # Check: equilateral = 1
    ("AnnotationValue[{MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, Tetrahedron[{1,2,3,4}]], 3}, MeshCellQuality]",
     NumList([6 * math.sqrt(2) / 6 / 1.5 ** 1.5], 1e-12)),                           # corner tetrahedron
    ("MeshCellQuality[x06m]", "MeshCellQuality[x06m]"),
    ("AnnotationValue[{%s, 5}, MeshCellQuality]" % TRI, Raises(V)),
    # ── SubdivisionRegion (hand computations)
    ("Area[SubdivisionRegion[MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, Polygon[{1,2,3,4}]], 1]]", Num(0.875)),  # Chaikin
    ("MeshCellCount[SubdivisionRegion[MeshRegion[{{0,0},{1,0},{0,1}}, Triangle[{1,2,3}]], 2]]", "{12, 12}"),
    ("MeshCellCount[SubdivisionRegion[%s, 1]]" % TET, "{10, 24, 16}"),               # Loop: V+E, 4F
    ("MeshCellCount[SubdivisionRegion[%s, 1]]" % CUBE, "{26, 72, 48}"),              # Catmull-Clark V+E+F
    ("MeshCellCount[SubdivisionRegion[MeshRegion[{{0,0},{1,0},{2,0}}, Line[{1,2,3}]], 1]]", "{4, 3}"),
    ("Module[{m = SubdivisionRegion[%s, 2]}, First[#] - #[[2]] + Last[#] &[MeshCellCount[m]]]" % TET, "2"),  # Check: Euler
    ("SubdivisionRegion[x06m, 1]", "SubdivisionRegion[x06m, 1]"),
    ("SubdivisionRegion[%s, x06n]" % TET, Pred(lambda g: isinstance(g, Expr) and g.head == "SubdivisionRegion", "inert")),
    ("SubdivisionRegion[%s, -1]" % TET, Raises(V)),
    ('SubdivisionRegion[%s, 1, "Foo"]' % TET, Raises(V)),
    ('SubdivisionRegion[%s, 1, "QuadTri"]' % TET, Raises(N)),
    ("SubdivisionRegion[%s, 1, 3]" % TET, Raises(T)),
    # ── SmoothMesh
    ("MeshCoordinates[SmoothMesh[MeshRegion[{{0,0},{1,0},{2,0},{3,0}}, Line[{1,2,3,4}]]]]",
     NumList([0, 0, 1, 0, 2, 0, 3, 0])),                                             # flat: invariant
    ("MeshCoordinates[SmoothMesh[MeshRegion[{{0,0},{1,1},{2,0}}, Line[{1,2,3}]]]]",
     NumList([0, 0, 1, 0.5, 2, 0])),                                                 # t = 1/2 toward average
    ("Volume[SmoothMesh[%s]]" % CUBE, Num(8 / 27)),                                  # umbrella: corners to 2/3
    ("SmoothMesh[x06m]", "SmoothMesh[x06m]"),
    ("SmoothMesh[3]", Raises(T)),
    ('SmoothMesh[%s, Method -> "Foo"]' % CUBE, Raises(N)),
    ("SmoothMesh[%s, Foo -> 1]" % CUBE, Raises(V)),
    # ── EstimatedPointNormals (exact normals of planes / circles)
    ("EstimatedPointNormals[{{0,0,0},{1,0,0},{0,1,0},{1,1,0},{0.5,0.5,0}}]", NumList([0, 0, 1] * 5)),
    ("EstimatedPointNormals[CirclePoints[8]]", Pred(lambda g: all(
        abs(abs(sum(a * b for a, b in zip(_fl(n), _fl(p)))) - 1) < 1e-9 for n, p in zip(g.args, ev("N[CirclePoints[8]]").args)),
        "radial normals")),
    ("EstimatedPointNormals[MeshRegion[{{0,0,0},{1,0,0},{0,1,0}}, Triangle[{1,2,3}]]]", NumList([0, 0, 1] * 3)),
    ("Map[Norm, EstimatedPointNormals[N[SpherePoints[20]]]]", NumList([1.0] * 20)),  # Check: unit length
    ("EstimatedPointNormals[x06p]", "EstimatedPointNormals[x06p]"),
    ('EstimatedPointNormals["a"]', Raises(T)),
    ("EstimatedPointNormals[{{0,0,0}}]", Raises(V)),
    # ── FindMeshDefects
    ('FindMeshDefects[%s, "HoleEdges", "CellIndex"]' % OPEN_TET, '<|"HoleEdges" -> {{1, 2}, {1, 4}, {1, 6}}|>'),
    ('FindMeshDefects[%s, "FlippedFaces", "CellIndex"]' % FLIP_TET, '<|"FlippedFaces" -> {{2, 4}}|>'),
    ('FindMeshDefects[%s, Automatic, "CellIndex"]' % CUBE, "<||>"),                   # closed cube: no defects
    ('FindMeshDefects[MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{5,5,5}}, {Triangle[{1,3,2}],Triangle[{1,2,4}],'
     'Triangle[{1,4,3}],Triangle[{2,3,4}],Point[5]}], "IsolatedVertices", "CellIndex"]',
     Pred(lambda g: "IsolatedVertices" in str(g), "isolated vertex found")),
    ("FindMeshDefects[x06m]", "FindMeshDefects[x06m]"),
    ("FindMeshDefects[3]", Raises(T)),
    ('FindMeshDefects[%s, "Foo"]' % CUBE, Raises(V)),
    # ── RepairMesh
    ("MeshCellCount[RepairMesh[%s]]" % OPEN_TET, "{4, 6, 4}"),                       # hole filled
    ('FindMeshDefects[RepairMesh[%s], Automatic, "CellIndex"]' % OPEN_TET, "<||>"),   # Check: defects gone
    ('MeshCells[RepairMesh[%s, "FlippedFaces"], 2]' % FLIP_TET,
     "{Triangle[{1, 3, 2}], Triangle[{1, 2, 4}], Triangle[{1, 4, 3}], Triangle[{3, 4, 2}]}"),
    ('MeshCoordinates[RepairMesh[MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{5,5,5}}, {Triangle[{1,3,2}],'
     'Triangle[{1,2,4}],Triangle[{1,4,3}],Triangle[{2,3,4}],Point[5]}], "IsolatedVertices"]]',
     NumList([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1])),
    ("RepairMesh[x06m]", "RepairMesh[x06m]"),
    ("RepairMesh[3]", Raises(T)),
    ('RepairMesh[%s, {"Foo"}]' % CUBE, Raises(V)),
    ('RepairMesh[MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{0,-1,0}}, {Triangle[{1,2,3}],Triangle[{1,2,4}],'
     'Triangle[{1,2,5}]}], "SingularEdges"]', Raises(N)),
    # ── SimplifyMesh (QEM)
    ("MeshCellCount[%s][[3]]" % SUBCUBE, "768"),
    ("MeshCellCount[SimplifyMesh[%s]][[3]]" % SUBCUBE, "384"),                       # default half the triangles
    ('MeshCellCount[SimplifyMesh[%s, {"MaxTriangleCount", 100}]][[3]]' % SUBCUBE, "100"),
    ("Module[{c = MeshCellCount[SimplifyMesh[%s]]}, c[[1]] - c[[2]] + c[[3]]]" % SUBCUBE, "2"),  # Check: Euler
    ("SimplifyMesh[x06m]", "SimplifyMesh[x06m]"),
    ("SimplifyMesh[3]", Raises(T)),
    ('SimplifyMesh[%s, {"Foo", 3}]' % CUBE, Raises(V)),
    # ── Remesh (Botsch-Kobbelt, feature preserving)
    ("Volume[Remesh[%s]]" % CUBE, Num(1.0)),
    ("Volume[Remesh[BoundaryDiscretizeRegion[Cuboid[{0,0,0},{2,1,1}]]]]", Num(2.0)),
    ("Module[{c = MeshCellCount[Remesh[%s]]}, c[[1]] - c[[2]] + c[[3]]]" % CUBE, "2"),  # Check: Euler
    ("Remesh[x06m]", "Remesh[x06m]"),
    ("Remesh[3]", Raises(T)),
    ('Remesh[%s, Method -> "Adaptive"]' % CUBE, Raises(N)),
    # ── CantorMesh (Wolfram doc: measures (2/3)^n)
    ("MeshCellCount[CantorMesh[2]]", "{8, 4}"),
    ("RegionMeasure[CantorMesh[3]]", Num(8 / 27)),
    ("MeshCellCount[CantorMesh[2, 2]][[3]]", "16"),                                   # Check: 2^(n d) cells
    ("RegionMeasure[CantorMesh[2, 2]]", Num((4 / 9) ** 2)),
    ("CantorMesh[x06n]", "CantorMesh[x06n]"),
    ('CantorMesh["a"]', Raises(T)),
    ("CantorMesh[-1]", Raises(V)),
    ("CantorMesh[1, 5]", Raises(N)),
    # ── MengerMesh (default dimension 2: the carpet)
    ("MeshCellCount[MengerMesh[1]]", "{16, 24, 8}"),
    ("Area[MengerMesh[2]]", Num((8 / 9) ** 2)),
    ("Volume[MengerMesh[1, 3]]", Num(20 / 27)),                                       # Check: (20/27)^n
    ("MengerMesh[x06n]", "MengerMesh[x06n]"),
    ('MengerMesh["a"]', Raises(T)),
    ("MengerMesh[-1]", Raises(V)),
    ("MengerMesh[1, 4]", Raises(N)),
    # ── SierpinskiMesh
    ("MeshCellCount[SierpinskiMesh[2]]", "{15, 27, 9}"),
    ("Area[SierpinskiMesh[1]]", Num(3 * math.sqrt(3) / 16)),
    ("Area[SierpinskiMesh[2]]", Num(9 * math.sqrt(3) / 64)),                          # Check: doc areas
    ("SierpinskiMesh[x06n]", "SierpinskiMesh[x06n]"),
    ('SierpinskiMesh["a"]', Raises(T)),
    ("SierpinskiMesh[-1]", Raises(V)),
    ("SierpinskiMesh[2, 4]", Raises(N)),
    # ── GeodesicPolyhedron (Wolfram doc values)
    ("Volume[GeodesicPolyhedron[2]]", Num(3.65871, 1e-5)),
    ("SurfaceArea[GeodesicPolyhedron[2]]", Num(11.6659, 1e-5)),
    ("Length[GeodesicPolyhedron[3][[2]]]", "180"),                                   # 20 n^2 faces
    ("Max[Abs[Norm /@ GeodesicPolyhedron[3][[1]] - 1]] < 10^-12", "True"),           # Check: on the sphere
    ("GeodesicPolyhedron[x06n]", "GeodesicPolyhedron[x06n]"),
    ("GeodesicPolyhedron[0]", Raises(V)),
    ('GeodesicPolyhedron["a"]', Raises(T)),
    # ── BeveledPolyhedron (truncated cuboctahedron = bevelled cube)
    ("Length[BeveledPolyhedron[Cube[]][[1]]]", "48"),
    ("Length[BeveledPolyhedron[Cube[]][[2]]]", "26"),
    ("Module[{p = BeveledPolyhedron[Cube[]]}, Length[p[[1]]] - Total[Length /@ p[[2]]]/2 + Length[p[[2]]]]", "2"),
    ("BeveledPolyhedron[x06p]", "BeveledPolyhedron[x06p]"),
    ("BeveledPolyhedron[Disk[]]", Raises(T)),
    ("BeveledPolyhedron[Dodecahedron[]]", Raises(N)),
    # ── SnubPolyhedron (snub cube)
    ("Length[SnubPolyhedron[Cube[]][[1]]]", "24"),
    ("Length[SnubPolyhedron[Cube[]][[2]]]", "38"),
    ("Module[{p = SnubPolyhedron[Cube[]]}, Length[p[[1]]] - Total[Length /@ p[[2]]]/2 + Length[p[[2]]]]", "2"),
    ("SnubPolyhedron[x06p]", "SnubPolyhedron[x06p]"),
    ("SnubPolyhedron[Disk[]]", Raises(T)),
    ("SnubPolyhedron[Icosahedron[]]", Raises(N)),
    # ── CanonicalizePolyhedron / OuterPolyhedron / InnerPolyhedron (documented void example)
    ("Volume[CanonicalizePolyhedron[Cube[]]]", "1"),
    ("Volume[CanonicalizePolyhedron[%s]]" % VOID, Num(4 / 3 - 1 / 48)),
    ("Volume[CanonicalizePolyhedron[Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,2,3},{1,4,2},{1,3,4},{2,4,3}}]]]",
     "1/6"),                                                                        # Check: inward faces reoriented
    ("CanonicalizePolyhedron[x06p]", "CanonicalizePolyhedron[x06p]"),
    ("CanonicalizePolyhedron[Disk[]]", Raises(T)),
    ("CanonicalizePolyhedron[Polyhedron[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,6/5}}, "
     "{{1,4,3,2},{1,2,6,5},{2,3,7,6},{3,4,8,7},{4,1,5,8},{5,6,7,8}}]]", Raises(V)),
    ("Volume[OuterPolyhedron[%s]]" % VOID, "4/3"),
    ("OuterPolyhedron[Cube[]] == CanonicalizePolyhedron[Cube[]]", "True"),
    ("Volume[OuterPolyhedron[Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,3,2},{1,2,4},{1,4,3},{2,3,4}}]]]", "1/6"),
    ("OuterPolyhedron[x06p]", "OuterPolyhedron[x06p]"),
    ("OuterPolyhedron[Disk[]]", Raises(T)),
    ("Volume[InnerPolyhedron[%s]]" % VOID, "1/48"),
    ("InnerPolyhedron[Cube[]]", "EmptyRegion[3]"),
    ("InnerPolyhedron[Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,3,2},{1,2,4},{1,4,3},{2,3,4}}]]", "EmptyRegion[3]"),
    ("InnerPolyhedron[x06p]", "InnerPolyhedron[x06p]"),
    ("InnerPolyhedron[Disk[]]", Raises(T)),
    # ── PolyhedronDecomposition
    ("Length[PolyhedronDecomposition[Cube[]]]", "6"),
    ("Total[Volume /@ PolyhedronDecomposition[Cube[]]]", "1"),                        # Check: volumes add up
    ("Total[Volume /@ PolyhedronDecomposition[Polyhedron[{{0,0,0},{2,0,0},{0,2,0},{0,0,2}}, {{1,3,2},{1,2,4},{1,4,3},{2,3,4}}]]]",
     "4/3"),
    ("PolyhedronDecomposition[x06p]", "PolyhedronDecomposition[x06p]"),
    ("PolyhedronDecomposition[Disk[]]", Raises(T)),
    ("PolyhedronDecomposition[Polyhedron[{{0,0,0},{2,0,0},{2,2,0},{0,2,0},{1,1,-1},{0,0,1},{2,0,1},{2,2,1},{0,2,1},"
     "{1,1,1/2}}, {{1,2,6},{2,7,6},{2,3,7},{3,8,7},{3,4,8},{4,9,8},{4,1,9},{1,6,9},{6,7,10},{7,8,10},{8,9,10},"
     "{9,6,10},{1,5,2},{2,5,3},{3,5,4},{4,5,1}}]]", Raises(N)),
    # ── SpherePoints (Wolfram doc: SpherePoints[7] = poles + pentagon)
    ("SpherePoints[7][[;; 3]]", "{{0, 0, -1}, {0, 0, 1}, {1, 0, 0}}"),
    ("Simplify /@ (Norm /@ SpherePoints[7])", "{1, 1, 1, 1, 1, 1, 1}"),                   # Check: unit norms
    ("Norm[Total[SpherePoints[4]]]", "0"),                                           # regular tetrahedron
    ("Length[SpherePoints[100]]", "100"),
    ("Max[Abs[Norm /@ SpherePoints[50] - 1]] < 10^-12", "True"),
    ("SpherePoints[x06n]", "SpherePoints[x06n]"),
    ("SpherePoints[0]", Raises(V)),
    ('SpherePoints["a"]', Raises((T, V))),
    # ── FindShortestCurve / ShortestCurveDistance (Wolfram doc examples)
    ("FindShortestCurve[Circle[], {1, 0}, {0, 1}]", "Circle[{0, 0}, 1, {0, Pi/2}]"),
    ("FindShortestCurve[Rectangle[{0, 0}, {2, 1}], {0, 0}, {2, 1}]", "Line[{{0, 0}, {2, 1}}]"),
    ("FindShortestCurve[Polygon[{{0,0},{2,0},{2,2},{1,1/2},{0,2}}], {1/10, 3/2}, {19/10, 3/2}]",
     "Line[{{1/10, 3/2}, {1, 1/2}, {19/10, 3/2}}]"),
    ("ArcLength[FindShortestCurve[Polygon[{{0,0},{2,0},{2,2},{1,1/2},{0,2}}], {1/10, 3/2}, {19/10, 3/2}]] == "
     "ShortestCurveDistance[Polygon[{{0,0},{2,0},{2,2},{1,1/2},{0,2}}], {1/10, 3/2}, {19/10, 3/2}]", "True"),  # Check
    ("FindShortestCurve[x06r, {0, 0}, {1, 1}]", "FindShortestCurve[x06r, {0, 0}, {1, 1}]"),
    ("FindShortestCurve[3, {0, 0}, {1, 1}]", Raises(T)),
    ("FindShortestCurve[Rectangle[], {0, 0}, {5, 5}]", Raises(V)),
    ("ShortestCurveDistance[Circle[], {1, 0}, {0, 1}]", "Pi/2"),
    ("ShortestCurveDistance[Rectangle[{0, 0}, {2, 1}], {0, 0}, {2, 1}]", "Sqrt[5]"),
    ("ShortestCurveDistance[Sphere[], {1, 0, 0}, {0, 0, 1}]", "Pi/2"),
    ("ShortestCurveDistance[Disk[], {x06x, 0}, {0, 1}]", "Sqrt[1 + x06x^2]"),          # doc: symbolic end point
    ("ShortestCurveDistance[Polygon[{{0,0},{2,0},{2,2},{1,1/2},{0,2}}], {1/10, 3/2}, {19/10, 3/2}]",
     Num(2 * math.hypot(0.9, 1.0))),
    ("ShortestCurveDistance[x06r, {0, 0}, {1, 1}]", "ShortestCurveDistance[x06r, {0, 0}, {1, 1}]"),
    ("ShortestCurveDistance[3, {0, 0}, {1, 1}]", Raises(T)),
    ("ShortestCurveDistance[Rectangle[], {0, 0}, {5, 5}]", Raises(V)),
    # ── FindCurvePath (NN-crust)
    ("FindCurvePath[CirclePoints[6]]", "{{1, 2, 3, 4, 5, 6, 1}}"),
    ("FindCurvePath[{{0,0},{1,0},{2,0},{10,0},{11,0}}]", "{{1, 2, 3, 4, 5}}"),
    ("FindCurvePath[CirclePoints[8][[{3, 7, 1, 5, 2, 8, 4, 6}]]]",                    # Check: shuffled circle
     Pred(lambda g: len(g.args) == 1 and len(g.args[0].args) == 9, "one closed 8-cycle")),
    ("FindCurvePath[{{0,0}}]", "{{1}}"),
    ("FindCurvePath[x06p]", "FindCurvePath[x06p]"),
    ('FindCurvePath["a"]', Raises(T)),
    # ── HighlightRegion (inert display wrapper)
    ("HighlightRegion[Disk[], Point[{0, 0}]]", "HighlightRegion[Disk[], Point[{0, 0}]]"),
    ("Area[HighlightRegion[Disk[], Point[{0, 0}]]]", "Pi"),                         # Check: region through wrapper
    ("RegionMember[HighlightRegion[Disk[], Point[{0, 0}]], {1/2, 0}]", "True"),
    ("HighlightRegion[x06r, x06s]", "HighlightRegion[x06r, x06s]"),
    ("HighlightRegion[Disk[], {0, 0}]", Raises(T)),
    # ── FindRegionTransform
    ("Module[{t = FindRegionTransform[Rectangle[], Parallelogram[{0,0},{{1,1},{1,0}}]]}, "
     "Sort[t /@ {{0,0},{1,0},{1,1},{0,1}}] == Sort[{{0,0},{1,1},{2,1},{1,0}}]]", "True"),   # Check: maps vertices
    ("FindRegionTransform[{{0,0},{1,0},{0,1}}, {{1,1},{2,1},{1,2}}][{1/3, 1/3}]", "{4/3, 4/3}"),
    ("FindRegionTransform[Disk[], Disk[{1, 2}, 3]][{1, 0}]", "{4, 2}"),
    ("FindRegionTransform[x06r, Disk[]]", "FindRegionTransform[x06r, Disk[]]"),
    ("FindRegionTransform[3, Disk[]]", Raises(T)),
    ("FindRegionTransform[Disk[], Rectangle[]]", Raises(V)),
    # ── RegionFit
    ('RegionFit[{{0,1},{1,3},{2,5},{3,7}}, "Line"]',
     Pred(lambda g: g.head == "InfiniteLine" and abs(_fl(g.args[0].args[1])[1] - 2 * _fl(g.args[0].args[1])[0] - 1) < 1e-9,
          "y = 2 x + 1")),
    ('RegionFit[CirclePoints[{1, 2}, 3, 7], "Circle"]', Pred(lambda g: g.head == "Circle" and
     max(abs(a - b) for a, b in zip(_fl(g.args[0]) + _fl(g.args[1]), [1, 2, 3])) < 1e-9, "Circle[{1, 2}, 3]")),
    ('RegionFit[{{0,0,1},{1,0,1},{0,1,1},{1,1,1}}, "Plane"]', Pred(lambda g: g.head == "Hyperplane" and
     abs(abs(_fl(g.args[0])[2]) - 1) < 1e-12, "z = 1")),
    ('RegionFit[N[SpherePoints[10]], "Sphere"]', Pred(lambda g: g.head == "Sphere" and abs(_fl(g.args[1])[0] - 1) < 1e-9,
                                                      "unit sphere")),
    ('RegionFit[{{0,1},{1,3},{2,5},{3,7},{4,9},{5,11},{2,-20}}, "Line", Method -> "RANSAC"]',   # Check: outlier ignored
     Pred(lambda g: g.head == "InfiniteLine" and abs(_fl(g.args[0].args[1])[1] - 2 * _fl(g.args[0].args[1])[0] - 1) < 1e-6,
          "robust y = 2 x + 1")),
    ('RegionFit[x06p, "Line"]', 'RegionFit[x06p, "Line"]'),
    ('RegionFit["a", "Line"]', Raises(T)),
    ('RegionFit[{{0,1},{1,3}}, "Foo"]', Raises(V)),
    ('RegionFit[{{0,1},{1,3},{2,5}}, "Line", Method -> "Foo"]', Raises(V)),
    ('RegionFit[{{0,0,0},{1,0,0},{0,1,0},{1,1,0}}, "Circle"]', Raises(N)),
    # ── RandomInstance / GeometricSolveValues / FindGeometricConjectures (classical theorems)
    ("SeedRandom[6]; {a, b, c} /. RandomInstance[%s][[1, 1]]" % S345, NumList([0, 0, 3, 0, 0, -4], 1e-9)),
    ("SeedRandom[6]; Length[RandomInstance[%s, 3]]" % THALES, "3"),
    ("SeedRandom[7]; Module[{r = Association[RandomInstance[%s][[1, 1]]]}, "
     "Abs[(r[a] - r[b]) . (r[c] - r[b])] < 10^-8]" % THALES, "True"),                 # Check: Thales on the instance
    ("RandomInstance[x06s]", "RandomInstance[x06s]"),
    ("RandomInstance[3]", Raises(T)),
    ("RandomInstance[GeometricScene[{a, b}, {EuclideanDistance[a, b] == 1, EuclideanDistance[a, b] == 2}]]", Raises(V)),
    ("RandomInstance[GeometricScene[{a, b, c}, {GeometricAssertion[{a, b, c}, \"Counterclockwise\"]}]]", Raises(N)),
    ("SeedRandom[1]; GeometricSolveValues[%s, EuclideanDistance[b, c]]" % S345, Num(5.0, 1e-8)),
    ("SeedRandom[2]; GeometricSolveValues[GeometricScene[{a, b, c, d, e, f}, {PlanarAngle[{b, a, c}] == 90 Degree, "
     "PlanarAngle[{a, c, b}] == 45 Degree, Line[{a, d, c}], Line[{b, f, c}], Line[{a, e, b}], Line[{e, f}], "
     "GeometricAssertion[{Line[{d, f}], Line[{b, c}]}, \"Perpendicular\"], EuclideanDistance[a, d] == 4, "
     "EuclideanDistance[d, c] == 8, EuclideanDistance[a, e] == 10}], Area[Polygon[{a, e, f, d}]]]", Num(48.0, 1e-7)),
    ("SeedRandom[3]; GeometricSolveValues[%s, PlanarAngle[{a, b, c}]]" % THALES, Num(math.pi / 2, 1e-8)),
    ("SeedRandom[4]; GeometricSolveValues[%s, {EuclideanDistance[a, b], EuclideanDistance[a, c]}]" % S345,
     NumList([3.0, 4.0], 1e-8)),
    ("GeometricSolveValues[x06s, x06e]", "GeometricSolveValues[x06s, x06e]"),
    ("GeometricSolveValues[3, x06e]", Raises(T)),
    ("SeedRandom[5]; GeometricSolveValues[GeometricScene[{a, b, c}, {Triangle[{a, b, c}]}], EuclideanDistance[a, b]]",
     Raises(V)),
    ("GeometricSolveValues[GeometricScene[{a, b, c}, {GeometricAssertion[{a, b, c}, \"Counterclockwise\"]}], EuclideanDistance[a, b]]",
     Raises(N)),
    ("SeedRandom[5]; MemberQ[FindGeometricConjectures[%s][[3]], PlanarAngle[{a, b, c}] == Pi/2]" % THALES, "True"),
    ("SeedRandom[5]; MemberQ[FindGeometricConjectures[%s][[3]], GeometricAssertion[{a, c, o}, \"Collinear\"]]" % THALES,
     "True"),
    ("SeedRandom[5]; FindGeometricConjectures[%s, _GeometricAssertion, 1][[3]]" % THALES,
     '{GeometricAssertion[{a, c, o}, "Collinear"]}'),
    ("SeedRandom[5]; MemberQ[FindGeometricConjectures[%s][[3]], EuclideanDistance[b, o] == EuclideanDistance[c, o]]"
     % THALES, "True"),                                                              # Check: circumcentre
    ("FindGeometricConjectures[x06s]", "FindGeometricConjectures[x06s]"),
    ("FindGeometricConjectures[3]", Raises(T)),
    ('FindGeometricConjectures[%s, Foo -> 1]' % THALES, Raises(V)),
    ('FindGeometricConjectures[%s, Method -> "Foo"]' % THALES, Raises(N)),
]

# ── SciPy-backed heads
SCIPY_CASES = [
    # ConcaveHullMesh (alpha shapes)
    ("ConcaveHullMesh[{{0,0},{1,0},{0,1}}]", "MeshRegion[{{0., 0.}, {1., 0.}, {0., 1.}}, {Triangle[{1, 2, 3}]}]"),
    ("Area[ConcaveHullMesh[{{0,0},{2,0},{2,2},{0,2},{1,0},{2,1},{1,2},{0,1},{1,1}}]]", Num(4.0)),
    ("Area[ConcaveHullMesh[{{0,0},{2,0},{2,2},{0,2},{1,0},{2,1},{1,2},{0,1},{1,1}}, 100]] == "
     "Area[ConvexHullMesh[{{0,0},{2,0},{2,2},{0,2},{1,0},{2,1},{1,2},{0,1},{1,1}}]]", "True"),  # Check: large alpha
    ("ConcaveHullMesh[{{0,0},{2,0},{2,2},{0,2},{1,0},{2,1},{1,2},{0,1},{1,1}}, 0.6]", Raises(V)),
    ("ConcaveHullMesh[{{0,0},{1,0},{0,1}}, -1]", Raises(V)),
    # TriangulateMesh
    ("Area[TriangulateMesh[MeshRegion[{{0,0},{2,0},{2,2},{1,1},{0,2}}, Polygon[{1,2,3,4,5}]]]]", Num(3.0)),
    ("MeshCellCount[TriangulateMesh[MeshRegion[{{0,0},{2,0},{2,2},{1,1},{0,2}}, Polygon[{1,2,3,4,5}]]]][[3]]", "3"),
    ("Volume[TriangulateMesh[%s]]" % CUBE, Num(1.0)),                                 # Check: measure preserved
    ("MeshCellCount[TriangulateMesh[MeshRegion[{{0,0},{1,0}}, Line[{1,2}]], MaxCellMeasure -> 0.25]]", "{5, 4}"),
    ("TriangulateMesh[MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, Polygon[{1,2,3,4}]], Foo -> 1]", Raises(V)),
    ('TriangulateMesh[MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, Polygon[{1,2,3,4}]], Method -> "Foo"]', Raises(N)),
    # ReconstructionMesh
    ("MeshCellCount[ReconstructionMesh[CirclePoints[16]]]", "{16, 16}"),            # doc: the 16-gon
    ('MeshCellCount[ReconstructionMesh[CirclePoints[12], Method -> "Crust"]]', "{12, 12}"),
    ("Max[Abs[Norm /@ MeshCoordinates[ReconstructionMesh[N[CirclePoints[16]]]] - 1]] < 10^-12", "True"),
    ('ReconstructionMesh[CirclePoints[12], Method -> "Foo"]', Raises(N)),
    ('ReconstructionMesh[SpherePoints[30], Method -> "Crust"]', Raises(N)),
    # GradientFittedMesh (Poisson)
    ('Module[{m = GradientFittedMesh[SpherePoints[150], PerformanceGoal -> "Speed"]}, '
     'Abs[Area[m] - 4 Pi] < 0.2 && Max[Abs[Norm /@ MeshCoordinates[m] - 1]] < 0.1]', "True"),   # Check: closed sphere
    ('GradientFittedMesh[SpherePoints[150], PerformanceGoal -> "Foo"]', Raises(V)),
    ("GradientFittedMesh[{{1,2,3}}]", Raises(V)),
    # ListSurfacePlot3D
    ("Head[ListSurfacePlot3D[SpherePoints[150], MaxPlotPoints -> 20]]", "Graphics3D"),
    ("ListSurfacePlot3D[SpherePoints[150], Foo -> 1]", Raises(V)),
    ("ListSurfacePlot3D[SpherePoints[150], MaxPlotPoints -> 2]", Raises(V)),
]

NO_SCIPY_CASES = [
    ("ConcaveHullMesh[{{0,0},{1,0},{0,1}}]", Raises(N, "SciPy")),
    ("TriangulateMesh[MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, Polygon[{1,2,3,4}]]]", Raises(N, "SciPy")),
    ("ReconstructionMesh[CirclePoints[16]]", Raises(N, "SciPy")),
    ("GradientFittedMesh[SpherePoints[150]]", Raises(N, "SciPy")),
    ("ListSurfacePlot3D[SpherePoints[150]]", Raises(N, "SciPy")),
]

# heads whose symbolic / type cases do not need SciPy
COMMON = [
    ("ConcaveHullMesh[x06p]", "ConcaveHullMesh[x06p]"),
    ('ConcaveHullMesh["a"]', Raises(T)),
    ("TriangulateMesh[x06m]", "TriangulateMesh[x06m]"),
    ("TriangulateMesh[3]", Raises(T)),
    ("ReconstructionMesh[x06p]", "ReconstructionMesh[x06p]"),
    ('ReconstructionMesh["a"]', Raises(T)),
    ("GradientFittedMesh[x06p]", "GradientFittedMesh[x06p]"),
    ('GradientFittedMesh["a"]', Raises(T)),
    ("ListSurfacePlot3D[x06p]", "ListSurfacePlot3D[x06p]"),
    ('ListSurfacePlot3D["a"]', Raises(T)),
]

CASES += COMMON + (SCIPY_CASES if HAVE_SCIPY else NO_SCIPY_CASES)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x06(case):
    check(case)
