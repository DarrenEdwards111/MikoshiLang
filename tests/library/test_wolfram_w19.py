"""Differential tests for Wolfram batch W19 (geometry I: primitives, special regions, splines, polyhedra and
synthetic geometry; module wl_geomprims).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented argument forms and
  defaults (``Annulus[] == Annulus[{0, 0}, {1/2, 1}]``, ``Cylinder[] == Cylinder[{{0, 0, -1}, {0, 0, 1}}]``,
  ``CirclePoints`` starting at angle Pi/n - Pi/2, Hexahedron/Prism/Pyramid face lists, uniform B-spline knots),
  and the documented example values ``BernsteinBasis[4, 3, 0.5] == 0.25``, ``BSplineBasis[3, 0.5] == 0.666667``,
  ``GeometricTest[{{2, 3}, {4, 6}, {-2, -3}}, "Collinear"] == True``,
  ``GeometricTest[Polygon[{{0, 0}, {5, 1}, {4, 4}, {-2, 0}}], "Convex"] == True``.
* Classical geometry (exact identities): the 3-4-5 right triangle (area 6, inradius 1, circumradius 5/2,
  circumcentre at the hypotenuse midpoint, orthocentre at the right-angle vertex), the regular tetrahedron
  (dihedral angle ArcCos[1/3], vertex solid angle ArcCos[23/27]), the cube (dihedral angle Pi/2, vertex solid angle
  Pi/2), the octant solid angle Pi/2, the spherical cap solid angle 2 Pi (1 - Cos[a]), Euler's formula
  V - E + F = 2 - 2 g, the cuboctahedron volume 5/6 of the unit cube.
* scipy.interpolate.BSpline (independent B-spline implementation) for BSplineBasis / BSplineFunction values.
* Independent re-implementations written here (Bernstein sums, point-in-region tests for random samples).

Symbol names used as symbolic placeholders are prefixed ``w19`` so they cannot collide with other test modules.
"""
import importlib.util
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr


def Num(v, tol=1e-9):
    def ok(got):
        try:
            return abs(float(got) - v) <= tol * max(1.0, abs(v))
        except Exception:
            return False
    return Pred(ok, f"~{v}")


def _flat(e):
    if isinstance(e, Expr) and e.head == "List":
        return [x for a in e.args for x in _flat(a)]
    return [float(e)]


def NumList(vs, tol=1e-9):
    def ok(got):
        try:
            g = _flat(got)
        except Exception:
            return False
        w = [x for v in vs for x in (v if isinstance(v, (list, tuple)) else [v])]
        return len(g) == len(w) and all(abs(a - b) <= tol * max(1.0, abs(b)) for a, b in zip(g, w))
    return Pred(ok, f"~{vs}")


def is_head(head, n=None):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head and (n is None or len(g.args) == n), f"{head}[...]")


def shape_pts(n, d, test=None):
    """A list of n points of dimension d (machine numbers), each passing test(point) if given."""
    def ok(g):
        if not (isinstance(g, Expr) and g.head == "List" and len(g.args) == n):
            return False
        for p in g.args:
            if not (isinstance(p, Expr) and p.head == "List" and len(p.args) == d and
                    all(isinstance(c, float) for c in p.args)):
                return False
            if test is not None and not test([float(c) for c in p.args]):
                return False
        return True
    return Pred(ok, f"{n} points in R^{d}")


T = "Triangle[{{0, 0}, {4, 0}, {0, 3}}]"       # the 3-4-5 right triangle
P345 = "{{0, 0}, {4, 0}, {0, 3}}"
SQ = "Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]"
HOLED = "Polygon[{{0, 0}, {4, 0}, {4, 4}, {0, 4}} -> {{{1, 1}, {2, 1}, {2, 2}}}]"
TET = "Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {1/2, Sqrt[3]/2, 0}, {1/2, Sqrt[3]/6, Sqrt[6]/3}}]"
TORUS = ("Polyhedron[{{0,0,0},{3,0,0},{3,3,0},{0,3,0},{1,1,0},{2,1,0},{2,2,0},{1,2,0},"
         "{0,0,1},{3,0,1},{3,3,1},{0,3,1},{1,1,1},{2,1,1},{2,2,1},{1,2,1}}, "
         "{{1,2,6,5},{2,3,7,6},{3,4,8,7},{4,1,5,8},{9,13,14,10},{10,14,15,11},{11,15,16,12},{12,16,13,9},"
         "{1,9,10,2},{2,10,11,3},{3,11,12,4},{4,12,9,1},{5,6,14,13},{6,7,15,14},{7,8,16,15},{8,5,13,16}}]")

# --- primitives: inert regions, validated arguments, graphics dimensions (ref/<Name> argument forms) ----------
# (valid form, symbolic form, (bad form, error class), graphics case)
PRIM = [
    ("Point[{1, 2}]", "Point[{w19a, w19b}]", ("Point[{\"x\", 1}]", TypeError), ("Graphics3D[Point[{1, 2}]]", ValueError)),
    ("Line[{{0, 0}, {1, 1}}]", "Line[{{0, 0}, {w19a, 1}}]", ("Line[{{0, 0}, {1, 1, 1}}]", ValueError),
     ("Graphics[Line[{{0, 0}, {1, 1}}]]", None)),
    ("Polygon[{{0, 0}, {1, 0}, {0, 1}}]", "Polygon[{{0, 0}, {w19a, 0}, {0, 1}}]",
     ("Polygon[{{0, 0}, {1, 0}, {0, 1}}, {{1, 2, 9}}]", ValueError), ("Graphics[Polygon[{{0, 0}, {1, 0}, {0, 1}}]]", None)),
    ("Triangle[]", "Triangle[{{0, 0}, {w19a, 0}, {0, 1}}]", ("Triangle[{{0, 0}, {1, 0}}]", ValueError),
     ("Graphics[Triangle[]]", None)),
    ("Rectangle[{0, 0}, {1, 1}, RoundingRadius -> 0.1]", "Rectangle[{w19a, 0}, {1, 1}]",
     ("Rectangle[{0, 0}, {1, 1}, RoundingRadius -> -1]", ValueError), ("Graphics3D[Rectangle[]]", ValueError)),
    ("Cuboid[{0, 0, 0}, {1, 1, 1}]", "Cuboid[{w19a, 0, 0}]", ("Cuboid[{0, 0}, {1, 1, 1}]", ValueError),
     ("Graphics3D[Cuboid[]]", None)),
    ("Circle[{0, 0}, 2, {0, Pi}]", "Circle[{0, 0}, w19r]", ("Circle[{0, 0}, -1]", ValueError), ("Graphics[Circle[]]", None)),
    ("Disk[{0, 0}, {2, 1}]", "Disk[{0, 0}, w19r]", ("Disk[{0, 0}, \"r\"]", TypeError), ("Graphics3D[Disk[]]", ValueError)),
    ("DiskSegment[{0, 0}, 1, {0, Pi/2}]", "DiskSegment[{0, 0}, w19r, {0, Pi/2}]", ("DiskSegment[{0, 0}, 1]", ValueError),
     ("Graphics[DiskSegment[{0, 0}, 1, {0, Pi/2}]]", None)),
    ("Sphere[{0, 0, 0}, 2]", "Sphere[{0, 0, 0}, w19r]", ("Sphere[{0, 0, 0}, \"a\"]", TypeError), ("Graphics3D[Sphere[]]", None)),
    ("Ball[{0, 0, 0}, 2]", "Ball[{w19a, 0, 0}]", ("Ball[{0, 0, 0}, -2]", ValueError), ("Graphics[Ball[]]", ValueError)),
    ("Ellipsoid[{0, 0}, {1, 2}]", "Ellipsoid[{0, 0}, {w19r, 2}]", ("Ellipsoid[{0, 0}, {{1, 2}, {2, 1}}]", ValueError),
     ("Graphics[Ellipsoid[{0, 0}, {1, 2}]]", None)),
    ("Annulus[{0, 0}, {1, 2}]", "Annulus[{0, 0}, {w19r, 2}]", ("Annulus[{0, 0}, {2, 1}]", ValueError), ("Graphics[Annulus[]]", None)),
    ("SphericalShell[{0, 0, 0}, {1, 2}]", "SphericalShell[{0, 0, 0}, {w19r, 2}]",
     ("SphericalShell[{0, 0, 0}, {2, 1}]", ValueError), ("Graphics3D[SphericalShell[]]", None)),
    ("Torus[{0, 0, 0}, {1, 2}]", "Torus[{0, 0, 0}, {w19r, 2}]", ("Torus[{0, 0, 0}, {2, 1}]", ValueError),
     ("Graphics[Torus[]]", ValueError)),
    ("FilledTorus[]", "FilledTorus[{0, 0, 0}, {w19r, 2}]", ("FilledTorus[{0, 0, 0}, 1]", TypeError),
     ("Graphics3D[FilledTorus[]]", None)),
    ("Cube[2]", "Cube[w19a]", ("Cube[-1]", ValueError), ("Graphics[Cube[]]", ValueError)),
    ("Tetrahedron[]", "Tetrahedron[w19a]", ("Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]", ValueError),
     ("Graphics3D[Tetrahedron[]]", None)),
    ("Octahedron[{0, 0, 0}, 1]", "Octahedron[w19a]", ("Octahedron[0]", ValueError), ("Graphics[Octahedron[]]", ValueError)),
    ("Dodecahedron[{1, 2}]", "Dodecahedron[w19a]", ("Dodecahedron[{1, 2, 3}, {1}]", ValueError),
     ("Graphics3D[Dodecahedron[]]", None)),
    ("Icosahedron[]", "Icosahedron[w19a]", ("Icosahedron[\"a\"]", TypeError), ("Graphics3D[Icosahedron[]]", None)),
    ("Hexahedron[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}}]",
     "Hexahedron[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,w19a}}]",
     ("Hexahedron[{{0, 0, 0}}]", ValueError), ("Graphics[Hexahedron[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}}]]", ValueError)),
    ("Prism[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}}]", "Prism[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,w19a}}]",
     ("Prism[{{0, 0, 0}, {1, 0, 0}}]", ValueError), ("Graphics3D[Prism[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,0,1},{0,1,1}}]]", None)),
    ("Pyramid[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{1/2,1/2,1}}]", "Pyramid[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{1/2,1/2,w19a}}]",
     ("Pyramid[{{0, 0, 0}}]", ValueError), ("Graphics3D[Pyramid[{{0,0,0},{1,0,0},{1,1,0},{0,1,0},{1/2,1/2,1}}]]", None)),
    ("Cone[{{0, 0, 0}, {0, 0, 1}}, 1]", "Cone[{{0, 0, 0}, {0, 0, 1}}, w19r]", ("Cone[{{0, 0, 0}, {0, 0, 1}}, -1]", ValueError),
     ("Graphics[Cone[]]", ValueError)),
    ("Cylinder[{{0, 0, 0}, {0, 0, 1}}, 1]", "Cylinder[{{0, 0, 0}, {0, 0, 1}}, w19r]",
     ("Cylinder[{{0, 0, 0}, {0, 0, 1}}, -1]", ValueError), ("Graphics3D[Cylinder[]]", None)),
    ("CapsuleShape[{{0, 0, 0}, {0, 0, 1}}, 1]", "CapsuleShape[{{0, 0, 0}, {0, 0, 1}}, w19r]",
     ("CapsuleShape[{{0, 0, 0}, {0, 0, 1}}, -1]", ValueError), ("Graphics3D[CapsuleShape[]]", None)),
    ("StadiumShape[{{0, 0}, {1, 0}}, 1]", "StadiumShape[{{0, 0}, {1, 0}}, w19r]",
     ("StadiumShape[{{0, 0}, {1, 0}}, -1]", ValueError), ("Graphics[StadiumShape[]]", None)),
    ("Parallelogram[{0, 0}, {{1, 0}, {1, 1}}]", "Parallelogram[{0, 0}, {{w19a, 0}, {1, 1}}]",
     ("Parallelogram[{0, 0}, {{1, 0}}]", ValueError), ("Graphics[Parallelogram[]]", None)),
    ("Parallelepiped[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]", "Parallelepiped[{0, 0, 0}, {{w19a, 0, 0}, {0, 1, 0}, {0, 0, 1}}]",
     ("Parallelepiped[{0, 0, 0}, {{1, 0, 0}, {0, 1}}]", ValueError), ("Graphics3D[Parallelepiped[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]", None)),
    ("Simplex[3]", "Simplex[{{0, 0}, {w19a, 0}, {0, 1}}]", ("Simplex[0]", ValueError), ("Graphics[Simplex[{{0, 0}, {1, 0}, {0, 1}}]]", None)),
    ("Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,2,3},{1,2,4},{2,3,4},{1,3,4}}]",
     "Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,w19a}}, {{1,2,3},{1,2,4},{2,3,4},{1,3,4}}]",
     ("Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,2,3},{1,2,4},{2,3,4},{1,3,9}}]", ValueError),
     ("Graphics[Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {{1,2,3},{1,2,4},{2,3,4},{1,3,4}}]]", ValueError)),
    ("UniformPolyhedron[\"Cube\"]", "UniformPolyhedron[w19a]", ("UniformPolyhedron[{1, 2, 3, 4, 5}]", TypeError),
     ("Graphics3D[UniformPolyhedron[\"Cube\"]]", None)),
    ("RegularPolygon[5]", "RegularPolygon[{0, 0}, w19r, 5]", ("RegularPolygon[2]", ValueError), ("Graphics[RegularPolygon[5]]", None)),
    ("BezierCurve[{{0, 0}, {1, 1}, {2, 0}}]", "BezierCurve[{{0, 0}, {w19a, 1}, {2, 0}}]",
     ("BezierCurve[{{0, 0}, {1, 1}, {2, 0}}, SplineDegree -> 0]", ValueError), ("Graphics[BezierCurve[{{0, 0}, {1, 1}, {2, 0}}]]", None)),
    ("BSplineCurve[{{0, 0}, {1, 1}, {2, 0}}, SplineClosed -> True]", "BSplineCurve[{{0, 0}, {w19a, 1}, {2, 0}}]",
     ("BSplineCurve[{{0, 0}, {1, 1}, {2, 0}}, SplineClosed -> 3]", ValueError), ("Graphics[BSplineCurve[{{0, 0}, {1, 1}}]]", None)),
    ("BSplineSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,1}}}]", "BSplineSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,w19a}}}]",
     ("BSplineSurface[{{{0,0,0},{1,0,0}},{{0,1,0}}}]", ValueError), ("Graphics[BSplineSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,1}}}]]", ValueError)),
    ("BezierSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,1}}}]", "BezierSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,w19a}}}]",
     ("BezierSurface[{{{0,0,0},{1,0,0}},{{0,1,0}}}]", ValueError), ("Graphics3D[BezierSurface[{{{0,0,0},{1,0,0}},{{0,1,0},{1,1,1}}}]]", None)),
    ("FilledCurve[{Line[{{0, 0}, {1, 0}, {0, 1}}]}]", "FilledCurve[{Line[{{0, 0}, {w19a, 0}, {0, 1}}]}]",
     ("FilledCurve[{Disk[]}]", TypeError), ("Graphics[FilledCurve[{Line[{{0, 0}, {1, 0}, {0, 1}}]}]]", None)),
    ("JoinedCurve[{Line[{{0, 0}, {1, 0}}], BezierCurve[{{1, 0}, {2, 1}, {0, 1}}]}]", "JoinedCurve[{Line[{{0, 0}, {w19a, 0}}]}]",
     ("JoinedCurve[{Disk[]}]", TypeError), ("Graphics[JoinedCurve[{Line[{{0, 0}, {1, 0}}]}]]", None)),
    ("Tube[{{0, 0, 0}, {1, 1, 1}}, 0.1]", "Tube[{{0, 0, 0}, {1, 1, 1}}, w19r]", ("Tube[{{0, 0, 0}, {1, 1, 1}}, -0.1]", ValueError),
     ("Graphics3D[Tube[{{0, 0, 0}, {1, 1, 1}}]]", None)),
    ("Arrow[{{0, 0}, {1, 1}}, {0.1, 0.2}]", "Arrow[{{0, 0}, {w19a, 1}}]", ("Arrow[\"a\"]", TypeError),
     ("Graphics[Arrow[{{0, 0}, {1, 1}}]]", None)),
    ("HalfLine[{0, 0}, {1, 1}]", "HalfLine[{0, 0}, {w19a, 1}]", ("HalfLine[{{0, 0}, {1, 1, 1}}]", ValueError),
     ("Graphics[HalfLine[{0, 0}, {1, 1}]]", None)),
    ("InfiniteLine[{{0, 0}, {1, 1}}]", "InfiniteLine[{0, 0}, {w19a, 1}]", ("InfiniteLine[{{0, 0}, {1, 1, 1}}]", ValueError),
     ("Graphics[InfiniteLine[{{0, 0}, {1, 1}}]]", None)),
    ("InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]", "InfinitePlane[{{0, 0, 0}, {w19a, 0, 0}, {0, 1, 0}}]",
     ("InfinitePlane[{{0, 0, 0}, {1, 0, 0}}]", ValueError), ("Graphics3D[InfinitePlane[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}]]", None)),
    ("HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]", "HalfPlane[{{0, 0}, {w19a, 0}}, {0, 1}]", ("HalfPlane[{{0, 0}, {1, 0}}]", ValueError),
     ("Graphics[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]", None)),
    ("HalfSpace[{0, 0, 1}, 0]", "HalfSpace[{0, 0, 1}, w19c]", ("HalfSpace[{0, 0, 1}, \"c\"]", TypeError),
     ("Graphics3D[HalfSpace[{0, 0, 1}, 0]]", None)),
    ("Hyperplane[{1, 1}, 1]", "Hyperplane[{1, 1}, w19c]", ("Hyperplane[{1, 1}, \"c\"]", TypeError), ("Graphics[Hyperplane[{1, 1}, 1]]", None)),
    ("AffineSpace[{{0, 0, 0}, {1, 0, 0}}]", "AffineSpace[{{0, 0, 0}, {w19a, 0, 0}}]", ("AffineSpace[\"a\"]", TypeError),
     ("Graphics3D[AffineSpace[{{0, 0, 0}, {1, 0, 0}}]]", None)),
    ("AffineHalfSpace[{{0, 0, 0}, {1, 0, 0}}, {0, 1, 0}]", "AffineHalfSpace[{{0, 0, 0}, {w19a, 0, 0}}, {0, 1, 0}]",
     ("AffineHalfSpace[\"a\", {0, 1, 0}]", TypeError), ("Graphics3D[AffineHalfSpace[{{0, 0, 0}, {1, 0, 0}}, {0, 1, 0}]]", None)),
    ("ConicHullRegion[{{0, 0}, {1, 0}, {0, 1}}]", "ConicHullRegion[{{0, 0}, {w19a, 0}, {0, 1}}]", ("ConicHullRegion[\"a\"]", TypeError),
     ("Graphics[ConicHullRegion[{{0, 0}, {1, 0}, {0, 1}}]]", None)),
    ("EmptyRegion[2]", "EmptyRegion[w19n]", ("EmptyRegion[0]", ValueError), ("Graphics[EmptyRegion[2]]", None)),
    ("FullRegion[3]", "FullRegion[w19n]", ("FullRegion[-1]", ValueError), ("Graphics3D[FullRegion[3]]", None)),
    ("PolarCurve[1 + Cos[w19t], {w19t, 0, 2 Pi}]", "PolarCurve[w19r, {w19t, 0, 2 Pi}]",
     ("PolarCurve[{Cos[w19t], Sin[w19t]}, {w19t, 0, 2 Pi}]", TypeError), ("Graphics[PolarCurve[1, {w19t, 0, Pi}]]", None)),
    ("FilledPolarCurve[1 + Cos[w19t], {w19t, 0, 2 Pi}]", "FilledPolarCurve[w19r, {w19t, 0, Pi}]",
     ("FilledPolarCurve[{1, 2}, {w19t, 0, 2 Pi}]", TypeError), ("Graphics[FilledPolarCurve[1, {w19t, 0, Pi}]]", None)),
]

PRIMITIVES = []
for _ok, _sym, (_bad, _exc), (_g, _gexc) in PRIM:
    PRIMITIVES += [(_ok, _ok), (_sym, _sym), (_bad, Raises(_exc))]
    PRIMITIVES.append((_g, Raises(_gexc)) if _gexc else (_g, _g))

PRIMITIVES += [
    ("Line[{1, 2}]", "Line[{1, 2}]"),                    # GraphicsComplex vertex-index forms (ref/GraphicsComplex)
    ("Polygon[{1, 2, 3}]", "Polygon[{1, 2, 3}]"),
    ("Normal[GraphicsComplex[{{0, 0}, {1, 0}, {0, 1}}, Polygon[{1, 2, 3}]]]", "Polygon[{{0, 0}, {1, 0}, {0, 1}}]"),
    ("ConvexPolygonQ[Polygon[{1, 2, 3}]]", Raises(TypeError)),
]

OPTION_SYMBOLS = []
for _o, _v in [("RoundingRadius", "0.2"), ("SplineDegree", "3"), ("SplineKnots", "\"Clamped\""),
               ("SplineWeights", "{1, 2, 1}"), ("SplineClosed", "True"), ("CurveClosed", "True"),
               ("RevolutionAxis", "{0, 0, 1}"), ("GeometricStylingRules", "{}")]:
    OPTION_SYMBOLS += [(f"{_o}[]", f"{_o}[]"), (f"{_o}[{_v}]", f"{_o}[{_v}]"), (f"{_o}[w19x]", f"{_o}[w19x]")]
OPTION_SYMBOLS += [
    ("RoundingRadius -> 0.2", "RoundingRadius -> 0.2"),
    ("BSplineCurve[{{0, 0}, {1, 1}, {2, 0}}, SplineKnots -> {0, 0, 0, 1, 1, 1}, SplineWeights -> {1, 2, 1}]",
     "BSplineCurve[{{0, 0}, {1, 1}, {2, 0}}, SplineKnots -> {0, 0, 0, 1, 1, 1}, SplineWeights -> {1, 2, 1}]"),
    ("FilledCurve[{Line[{{0, 0}, {1, 0}, {0, 1}}]}, CurveClosed -> True]", "FilledCurve[{Line[{{0, 0}, {1, 0}, {0, 1}}]}, CurveClosed -> True]"),
]

SCENES = [
    ("GeometricScene[{w19a, w19b, w19c}, {Triangle[{w19a, w19b, w19c}]}]",
     "GeometricScene[{w19a, w19b, w19c}, {Triangle[{w19a, w19b, w19c}]}]"),
    ("GeometricScene[{w19a -> {0, 0}, w19b}, {}]", "GeometricScene[{w19a -> {0, 0}, w19b}, {}]"),
    ("GeometricScene[{{w19a, w19b}, {w19t}}, {}, GeometricStylingRules -> {}]",
     "GeometricScene[{{w19a, w19b}, {w19t}}, {}, GeometricStylingRules -> {}]"),
    ("GeometricScene[\"x\", {}]", Raises(TypeError)),
    ("GeometricStep[{Triangle[{w19a, w19b, w19c}]}]", "GeometricStep[{Triangle[{w19a, w19b, w19c}]}]"),
    ("GeometricStep[{Triangle[{w19a, w19b, w19c}]}, \"label\"]", "GeometricStep[{Triangle[{w19a, w19b, w19c}]}, \"label\"]"),
    ("GeometricStep[{}, \"empty\"]", "GeometricStep[{}, \"empty\"]"),
    ("GeometricStep[\"x\"]", Raises(TypeError)),
    ("GeometricAssertion[Triangle[{w19a, w19b, w19c}], \"Equilateral\"]",
     "GeometricAssertion[Triangle[{w19a, w19b, w19c}], \"Equilateral\"]"),
    ("GeometricAssertion[{InfiniteLine[{w19a, w19b}], InfiniteLine[{w19c, w19d}]}, \"Perpendicular\"]",
     "GeometricAssertion[{InfiniteLine[{w19a, w19b}], InfiniteLine[{w19c, w19d}]}, \"Perpendicular\"]"),
    ("GeometricAssertion[w19a, {\"Inside\", Circle[w19b, 1]}, \"Distinct\"]",
     "GeometricAssertion[w19a, {\"Inside\", Circle[w19b, 1]}, \"Distinct\"]"),
    ("GeometricAssertion[Triangle[{w19a, w19b, w19c}], \"Bogus\"]", Raises(ValueError)),
    ("GeometricAssertion[Triangle[{w19a, w19b, w19c}], 3]", Raises(TypeError)),
    ("GeometricAssertion[w19a, \"Inside\"]", Raises(ValueError)),
]

# --- synthetic geometry: angles, vectors, orientation (exact classical values) ---------------------------------
SYNTHETIC = [
    ("AngleVector[Pi/3]", "{1/2, Sqrt[3]/2}"),
    ("AngleVector[{1, 1}, {2, Pi/2}]", "{1, 3}"),
    ("AngleVector[{2, Pi/4}]", "{Sqrt[2], Sqrt[2]}"),
    ("AngleVector[0.5]", [math.cos(0.5), math.sin(0.5)]),
    ("AngleVector[w19t]", "{Cos[w19t], Sin[w19t]}"),
    ("AngleVector[\"a\"]", Raises(TypeError)),
    # CirclePoints: documented first angle Pi/n - Pi/2 (ref/CirclePoints)
    ("CirclePoints[4]", "{{Sqrt[2]/2, -Sqrt[2]/2}, {Sqrt[2]/2, Sqrt[2]/2}, {-Sqrt[2]/2, Sqrt[2]/2}, {-Sqrt[2]/2, -Sqrt[2]/2}}"),
    ("CirclePoints[{1, 0}, 3]", "{{1, 0}, {-1/2, Sqrt[3]/2}, {-1/2, -Sqrt[3]/2}}"),
    ("CirclePoints[{0, 0}, {2, Pi/2}, 4]", "{{0, 2}, {-2, 0}, {0, -2}, {2, 0}}"),
    ("Total[CirclePoints[7]]", Pred(lambda g: all(abs(float(c)) < 1e-12 for c in ev("N[Total[CirclePoints[7]]]").args), "sum 0")),
    ("CirclePoints[0]", Raises(ValueError)),
    ("PlanarAngle[{{1, 0}, {0, 0}, {0, 1}}]", "Pi/2"),
    ("PlanarAngle[{0, 0} -> {{1, 0}, {0, 1}}]", "Pi/2"),
    ("PlanarAngle[{{1, 0}, {0, 0}, {0, 1}}, \"Clockwise\"]", "3 Pi/2"),
    ("PlanarAngle[{{1, 0}, {0, 0}, {1, Sqrt[3]}}]", "Pi/3"),
    ("PlanarAngle[{{1, 0, 0}, {0, 0, 0}, {0, 1, 1}}]", "Pi/2"),
    ("PlanarAngle[{{1., 0}, {0, 0}, {0, 1}}]", math.pi / 2),
    ("PlanarAngle[{{w19a, 0}, {0, 0}, {0, 1}}]", "PlanarAngle[{{w19a, 0}, {0, 0}, {0, 1}}]"),
    ("PlanarAngle[{{1, 0}, {0, 0}, {0, 1}}, \"Bad\"]", Raises(ValueError)),
    ("AngleBisector[{{1, 0}, {0, 0}, {0, 1}}]", "InfiniteLine[{0, 0}, {1, 1}]"),
    ("AngleBisector[{{1, 0}, {0, 0}, {0, 1}}, \"Exterior\"]", "InfiniteLine[{0, 0}, {1, -1}]"),
    ("AngleBisector[{{2, 0}, {0, 0}, {0, 5}}]", "InfiniteLine[{0, 0}, {1, 1}]"),
    ("AngleBisector[{{1, 0}, {0, 0}, {0, 1}}, \"Bad\"]", Raises(ValueError)),
    ("PerpendicularBisector[{{0, 0}, {2, 0}}]", "InfiniteLine[{1, 0}, {0, 2}]"),
    ("PerpendicularBisector[Line[{{0, 0}, {0, 2}}]]", "InfiniteLine[{0, 1}, {-2, 0}]"),
    ("PerpendicularBisector[{{1, 1}, {3, 3}}]", "InfiniteLine[{2, 2}, {-2, 2}]"),
    ("PerpendicularBisector[Line[{{0, 0, 0}, {2, 0, 0}}]]", Raises(ValueError)),
    ("InfiniteLineThrough[{{0, 0}, {1, 1}, {2, 2}}]", "InfiniteLine[{{0, 0}, {1, 1}}]"),
    ("InfiniteLineThrough[{{0, 0, 0}, {0, 0, 1}}]", "InfiniteLine[{{0, 0, 0}, {0, 0, 1}}]"),
    ("InfiniteLineThrough[{{1, 2}, {3, 4}}]", "InfiniteLine[{{1, 2}, {3, 4}}]"),
    ("InfiniteLineThrough[{{0, 0}, {1, 1}, {2, 3}}]", Raises(ValueError)),
    ("PositivelyOrientedPoints[{{0, 0}, {1, 0}, {0, 1}}]", True),
    ("PositivelyOrientedPoints[{{0, 0}, {0, 1}, {1, 0}}]", False),
    ("PositivelyOrientedPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]", True),
    ("PositivelyOrientedPoints[{{0, 0}, {w19a, 0}, {0, 1}}]", "PositivelyOrientedPoints[{{0, 0}, {w19a, 0}, {0, 1}}]"),
    ("PositivelyOrientedPoints[{{0, 0}, {1, 0}, {0, 1}, {1, 1}}]", Raises(NotImplementedError)),
    ("NegativelyOrientedPoints[{{0, 0}, {1, 0}, {0, 1}}]", False),
    ("NegativelyOrientedPoints[{{0, 0}, {0, 1}, {1, 0}}]", True),
    ("NegativelyOrientedPoints[{{0, 0}, {1, 1}, {2, 2}}]", False),
    ("NegativelyOrientedPoints[{1, 2}]", Raises(TypeError)),
    ("CollinearPoints[{{0, 0}, {1, 1}, {2, 2}}]", True),
    ("CollinearPoints[{{0, 0}, {1, 1}, {2, 3}}]", False),
    ("CollinearPoints[{{0, 0, 0}, {1, 2, 3}, {2, 4, 6}, {-1, -2, -3}}]", True),
    ("CollinearPoints[{{0., 0.}, {0.1, 0.2}, {0.3, 0.6}}]", True),
    ("CollinearPoints[{{0, 0}, {1, 1}, {w19a, 2}}]", "CollinearPoints[{{0, 0}, {1, 1}, {w19a, 2}}]"),
    ("CollinearPoints[{{0, 0}, {1, 1, 1}}]", Raises(ValueError)),
    ("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}]", True),
    ("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 1}}]", False),
    ("CoplanarPoints[{{1, 1, 1}, {2, 2, 2}, {3, 3, 4}, {0, 0, 1}}]", True),
    ("CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, w19a}}]", "CoplanarPoints[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, w19a}}]"),
    ("CoplanarPoints[\"a\"]", Raises(TypeError)),
    ("DihedralAngle[{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {0, 1, 0}}]", "Pi/2"),
    ("DihedralAngle[{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {1, 1, 0}}]", "Pi/4"),
    ("DihedralAngle[{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {-1, 0, 5}}]", "Pi"),
    ("DihedralAngle[{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {w19a, 1, 0}}]", "DihedralAngle[{{0, 0, 0}, {0, 0, 1}}, {{1, 0, 0}, {w19a, 1, 0}}]"),
    ("DihedralAngle[{{0, 0, 0}, {0, 0, 0}}, {{1, 0, 0}, {0, 1, 0}}]", Raises(ValueError)),
    # solid angles: octant Pi/2; spherical cap 2 Pi (1 - Cos[a]) with Sin[a] = 1/2 -> 2 Pi (1 - Sqrt[3]/2)
    ("SolidAngle[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]", "Pi/2"),
    ("SolidAngle[{0, 0, 0}, Ball[{0, 0, 2}, 1]]", "2 Pi (1 - Sqrt[3]/2)"),
    ("SolidAngle[{0, 0}, {{1, 0}, {0, 1}}]", "Pi/2"),
    ("SolidAngle[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, w19a}}]", "SolidAngle[{0, 0, 0}, {{1, 0, 0}, {0, 1, 0}, {0, 0, w19a}}]"),
    ("SolidAngle[{0, 0, 0}, Cylinder[]]", Raises(NotImplementedError)),
    # spherical points are {polar angle, azimuth} (ref/SphericalDistance)
    ("SphericalDistance[{0, 0}, {Pi/2, 0}]", "Pi/2"),
    ("SphericalDistance[{Pi/2, 0}, {Pi/2, Pi}]", "Pi"),
    ("SphericalDistance[{Pi/2, 0}, {Pi/2, Pi/3}]", "Pi/3"),
    ("SphericalDistance[{1., 0.}, {1., 0.5}]",
     2 * math.asin(math.sqrt(math.sin(1) ** 2 * math.sin(0.25) ** 2))),
    ("SphericalDistance[{w19a, 0}, {w19a, 0}]", Pred(lambda g: g is not None, "formula")),
    ("SphericalDistance[{0}, {1, 2}]", Raises((TypeError, ValueError))),
    ("SphericalAngle[{Pi/2, 0} -> {{0, 0}, {Pi/2, Pi/2}}]", "-Pi/2"),
    ("SphericalAngle[{Pi/2, 0} -> {{Pi/2, Pi/2}, {0, 0}}]", "Pi/2"),
    ("SphericalAngle[{0, 0} -> {{Pi/2, 0}, {Pi/2, Pi/2}}]", "Pi/2"),
    ("SphericalAngle[{0, 0} -> {{Pi/2, 0}, {Pi/2, Pi/4}}]", "Pi/4"),
    ("SphericalAngle[{0, 0} -> {{0, Pi/2}, {Pi/2, 0}}]", Raises(ValueError)),
]

# --- triangles (classical exact values of the 3-4-5 right triangle and equilateral triangles) -----------------
TRIANGLES = [
    ("SSSTriangle[3, 4, 5]", "Triangle[{{0, 0}, {5, 0}, {16/5, 12/5}}]"),
    ("SSSTriangle[1, 1, 1]", "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"),
    ("TriangleMeasurement[SSSTriangle[2, 3, 4], \"Perimeter\"]", "9"),
    ("SSSTriangle[w19a, w19b, w19c]", is_head("Triangle", 1)),
    ("SSSTriangle[1, 1, 3]", Raises(ValueError)),
    ("SSSTriangle[\"a\", 1, 1]", Raises(TypeError)),
    ("SASTriangle[1, Pi/2, 1]", "Triangle[{{0, 0}, {Sqrt[2], 0}, {Sqrt[2]/2, Sqrt[2]/2}}]"),
    ("SASTriangle[1, Pi/3, 1]", "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"),
    ("TriangleMeasurement[SASTriangle[3, Pi/2, 4], \"Area\"]", "6"),
    ("SASTriangle[1, Pi, 1]", Raises(ValueError)),
    ("ASATriangle[Pi/3, 1, Pi/3]", "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"),
    ("ASATriangle[Pi/2, 1, Pi/4]", "Triangle[{{0, 0}, {1, 0}, {0, 1}}]"),
    ("TriangleMeasurement[ASATriangle[Pi/4, 2, Pi/4], \"Area\"]", "1"),
    ("ASATriangle[Pi/2, 1, Pi/2]", Raises(ValueError)),
    ("AASTriangle[Pi/2, Pi/4, 1]", "Triangle[{{0, 0}, {Sqrt[2]/2, 0}, {0, Sqrt[2]/2}}]"),
    ("AASTriangle[Pi/3, Pi/3, 1]", "Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}]"),
    ("TriangleMeasurement[AASTriangle[Pi/2, Pi/6, 2], \"Area\"]", "Sqrt[3]/2"),
    ("AASTriangle[Pi, Pi/3, 1]", Raises(ValueError)),
    ("TriangleCenter[" + P345 + "]", "{4/3, 1}"),
    ("TriangleCenter[" + P345 + ", \"Incenter\"]", "{1, 1}"),
    ("TriangleCenter[" + P345 + ", \"Circumcenter\"]", "{2, 3/2}"),
    ("TriangleCenter[" + P345 + ", \"Orthocenter\"]", "{0, 0}"),
    ("TriangleCenter[" + P345 + ", \"NinePointCenter\"]", "{1, 3/4}"),
    ("TriangleCenter[" + P345 + ", {\"Midpoint\", 1}]", "{2, 3/2}"),
    ("TriangleCenter[" + T + ", \"Centroid\"]", "{4/3, 1}"),
    ("TriangleCenter[{{0, 0}, {w19a, 0}, {0, 3}}]", "TriangleCenter[{{0, 0}, {w19a, 0}, {0, 3}}]"),
    ("TriangleCenter[" + P345 + ", \"Bogus\"]", Raises(ValueError)),
    ("TriangleMeasurement[" + P345 + ", \"Area\"]", "6"),
    ("TriangleMeasurement[" + P345 + ", \"Inradius\"]", "1"),
    ("TriangleMeasurement[" + P345 + ", \"Circumradius\"]", "5/2"),
    ("TriangleMeasurement[" + P345 + ", {\"InteriorAngle\", 1}]", "Pi/2"),
    ("TriangleMeasurement[" + P345 + ", \"Perimeter\"]", "12"),
    ("TriangleMeasurement[{{0., 0.}, {4., 0.}, {0., 3.}}, \"Area\"]", 6.0),
    ("TriangleMeasurement[{{0, 0}, {w19a, 0}, {0, 3}}, \"Area\"]", "TriangleMeasurement[{{0, 0}, {w19a, 0}, {0, 3}}, \"Area\"]"),
    ("TriangleMeasurement[" + P345 + ", \"Bogus\"]", Raises(ValueError)),
    ("TriangleConstruct[" + P345 + ", \"Circumcircle\"]", "Circle[{2, 3/2}, 5/2]"),
    ("TriangleConstruct[" + P345 + ", \"Incircle\"]", "Circle[{1, 1}, 1]"),
    ("TriangleConstruct[" + P345 + ", \"MedialTriangle\"]", "Triangle[{{2, 3/2}, {0, 3/2}, {2, 0}}]"),
    ("TriangleConstruct[" + P345 + ", {\"Altitude\", 1}]", "Line[{{0, 0}, {36/25, 48/25}}]"),
    ("TriangleConstruct[{{0, 0}, {w19a, 0}, {0, 3}}, \"Incircle\"]", "TriangleConstruct[{{0, 0}, {w19a, 0}, {0, 3}}, \"Incircle\"]"),
    ("TriangleConstruct[" + P345 + ", \"Bogus\"]", Raises(ValueError)),
    ("TriangleConstruct[{{0, 0}, {1, 1}, {2, 2}}, \"Incircle\"]", Raises(ValueError)),
]

# --- circles and spheres through / around points --------------------------------------------------------------
SPHERES = [
    ("Circumsphere[" + P345 + "]", "Sphere[{2, 3/2}, 5/2]"),
    ("Circumsphere[Cube[]]", "Sphere[{0, 0, 0}, Sqrt[3]/2]"),
    ("Circumsphere[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}}]", "Sphere[{0, 0, 0}, 1]"),
    ("Circumsphere[{{0, 0}, {w19a, 0}, {0, 3}}]", "Circumsphere[{{0, 0}, {w19a, 0}, {0, 3}}]"),
    ("Circumsphere[{{0, 0}, {1, 1}, {2, 2}}]", Raises(ValueError)),
    ("Insphere[" + P345 + "]", "Sphere[{1, 1}, 1]"),
    ("Insphere[Cube[]]", "Sphere[{0, 0, 0}, 1/2]"),
    ("Insphere[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]", "Sphere[{1/2 - Sqrt[3]/6, 1/2 - Sqrt[3]/6, 1/2 - Sqrt[3]/6}, 1/2 - Sqrt[3]/6]"),
    ("Insphere[Polygon[{{0, 0}, {2, 0}, {2, 1}, {0, 1}}]]", Raises(ValueError)),
    ("CircleThrough[{{1, 0}, {0, 1}, {-1, 0}}]", "Circle[{0, 0}, 1]"),
    ("CircleThrough[" + P345 + "]", "Circle[{2, 3/2}, 5/2]"),
    ("CircleThrough[{{0, 0}, {2, 0}}, {1, 0}]", "Circle[{1, 0}, 1]"),
    ("CircleThrough[{{0, 0}, {2, 0}}]", Raises(ValueError)),
    ("CircleThrough[{{0, 0}, {1, 1}, {2, 2}}]", Raises(ValueError)),
    ("CircularArcThrough[{{1, 0}, {0, 1}, {-1, 0}}]", "Circle[{0, 0}, 1, {0, Pi}]"),
    ("CircularArcThrough[{{-1, 0}, {0, 1}, {1, 0}}]", "Circle[{0, 0}, 1, {0, Pi}]"),
    ("CircularArcThrough[{{0, -1}, {1, 0}, {0, 1}}]", "Circle[{0, 0}, 1, {-Pi/2, Pi/2}]"),
    ("CircularArcThrough[{{0, 0}, {1, 1}, {2, 2}}]", Raises(ValueError)),
    ("CircumscribedBall[{{0, 0}, {4, 0}, {0, 3}, {1, 1}}]", "Ball[{2, 3/2}, 5/2]"),
    ("CircumscribedBall[{{-1, 0}, {1, 0}, {0, 1/2}}]", "Ball[{0, 0}, 1]"),
    ("CircumscribedBall[{{0, 0, 0}, {2, 0, 0}}]", "Ball[{1, 0, 0}, 1]"),
    ("CircumscribedBall[{{0, 0}, {w19a, 0}}]", "CircumscribedBall[{{0, 0}, {w19a, 0}}]"),
    ("CircumscribedBall[\"a\"]", Raises(TypeError)),
    ("InscribedBall[{{0, 0}, {4, 0}, {0, 3}, {1, 1}}]", "Ball[{1, 1}, 1]"),
    ("InscribedBall[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]", "Ball[{1, 1}, 1]"),
    ("InscribedBall[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]", "Ball[{1/2 - Sqrt[3]/6, 1/2 - Sqrt[3]/6, 1/2 - Sqrt[3]/6}, 1/2 - Sqrt[3]/6]"),
    ("InscribedBall[{{0, 0}, {1, 1}, {2, 2}}]", Raises(ValueError)),
]

# --- polygons -------------------------------------------------------------------------------------------------
POLYGONS = [
    ("ConvexPolygonQ[" + SQ + "]", True),
    ("ConvexPolygonQ[Polygon[{{0, 0}, {2, 0}, {1, 1/2}, {1, 2}}]]", False),
    ("ConvexPolygonQ[RegularPolygon[6]]", True),
    ("ConvexPolygonQ[" + HOLED + "]", False),
    ("ConvexPolygonQ[Polygon[{{0, 0}, {w19a, 0}, {0, 1}}]]", "ConvexPolygonQ[Polygon[{{0, 0}, {w19a, 0}, {0, 1}}]]"),
    ("ConvexPolygonQ[Disk[]]", Raises(TypeError)),
    ("SimplePolygonQ[Polygon[{{0, 0}, {1, 1}, {1, 0}, {0, 1}}]]", False),
    ("SimplePolygonQ[" + SQ + "]", True),
    ("SimplePolygonQ[Polygon[{{0, 0}, {2, 0}, {1, 1/2}, {1, 2}}]]", True),
    ("SimplePolygonQ[" + HOLED + "]", False),
    ("SimplePolygonQ[\"a\"]", Raises(TypeError)),
    ("PolygonAngle[" + SQ + "]", "{Pi/2, Pi/2, Pi/2, Pi/2}"),
    ("PolygonAngle[" + SQ + ", {1, 0}]", "Pi/2"),
    ("PolygonAngle[Triangle[], 1]", "Pi/2"),
    ("PolygonAngle[RegularPolygon[5], 1]", "3 Pi/5"),
    ("PolygonAngle[RegularPolygon[7]]", "Table[5 Pi/7, 7]"),
    ("N[Total[PolygonAngle[Polygon[{{0, 0}, {3, 0}, {0, 1}}]]]]", math.pi),
    ("PolygonAngle[" + SQ + ", {5, 5}]", Raises(ValueError)),
    ("PolygonCoordinates[Triangle[]]", "{{0, 0}, {0, 1}, {1, 0}}"),
    ("PolygonCoordinates[Rectangle[{0, 0}, {2, 1}]]", "{{0, 0}, {0, 1}, {2, 0}, {2, 1}}"),
    ("PolygonCoordinates[" + HOLED + "]", "{{0, 0}, {0, 4}, {1, 1}, {2, 1}, {2, 2}, {4, 0}, {4, 4}}"),
    ("PolygonCoordinates[Polygon[{{0, 0}, {w19a, 0}, {0, 1}}]]", "PolygonCoordinates[Polygon[{{0, 0}, {w19a, 0}, {0, 1}}]]"),
    ("PolygonCoordinates[Cube[]]", Raises(TypeError)),
    ("CanonicalizePolygon[Polygon[{{0, 0}, {0, 1}, {1, 0}}]]", "Polygon[{{0, 0}, {0, 1}, {1, 0}}, {3, 2, 1}]"),
    ("CanonicalizePolygon[" + HOLED + "]",
     "Polygon[{{0, 0}, {0, 4}, {1, 1}, {2, 1}, {2, 2}, {4, 0}, {4, 4}}, {1, 6, 7, 2} -> {{5, 4, 3}}]"),
    ("CanonicalizePolygon[" + SQ + "]", "Polygon[{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {1, 3, 4, 2}]"),
    ("CanonicalizePolygon[Disk[]]", Raises(TypeError)),
    ("OuterPolygon[" + HOLED + "]", "Polygon[{{0, 0}, {0, 4}, {4, 0}, {4, 4}}, {1, 3, 4, 2}]"),
    ("OuterPolygon[Triangle[]]", "Polygon[{{0, 0}, {0, 1}, {1, 0}}, {1, 3, 2}]"),
    ("PolygonCoordinates[OuterPolygon[" + HOLED + "]]", "{{0, 0}, {0, 4}, {4, 0}, {4, 4}}"),
    ("OuterPolygon[Disk[]]", Raises(TypeError)),
    ("InnerPolygon[" + HOLED + "]", "Polygon[{{1, 1}, {2, 1}, {2, 2}}, {1, 2, 3}]"),
    ("InnerPolygon[Triangle[]]", "EmptyRegion[2]"),
    ("InnerPolygon[Polygon[{{0, 0}, {6, 0}, {6, 6}, {0, 6}} -> {{{1, 1}, {2, 1}, {2, 2}}, {{4, 4}, {5, 4}, {5, 5}}}]]",
     Pred(lambda g: g.head == "Polygon", "Polygon")),
    ("InnerPolygon[Disk[]]", Raises(TypeError)),
    ("WindingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}], {1/2, 1/2}]", 1),
    ("WindingCount[Line[{{0, 0}, {0, 1}, {1, 1}, {1, 0}, {0, 0}}], {1/2, 1/2}]", -1),
    ("WindingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}], {{1/2, 1/2}, {3, 3}}]", "{1, 0}"),
    ("WindingCount[Line[{{0, 0}, {1, 1}, {1, 0}, {0, 1}, {0, 0}}], {1/2, 1/4}]", 0),
    ("WindingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 0}}], {w19a, 0}]", "WindingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 0}}], {w19a, 0}]"),
    ("WindingCount[Line[{{0, 0}, {1, 0}, {1, 1}}], {0, 0}]", Raises(ValueError)),
    ("WindingCount[\"a\", {0, 0}]", Raises(TypeError)),
    ("CrossingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}], {1/2, 1/2}]", 1),
    ("CrossingCount[Line[{{1, 0}, {0, 3/2}, {-1, 0}, {1, 0}}], {0, 1}]", 1),
    ("CrossingCount[Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}], {3, 3}]", 0),
    ("CrossingCount[Line[{{1, 0}, {0, 2}, {-1, 0}, {0, 1}, {1, 0}}], {0, 1/2}]", 0),
    ("CrossingCount[\"a\", {0, 0}]", Raises(TypeError)),
    ("WindingPolygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]", "Polygon[{{0, 0}, {0, 1}, {1, 0}, {1, 1}}, {1, 3, 4, 2}]"),
    ("WindingPolygon[{{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, {{1/2, 1/2}, {3/2, 1/2}, {3/2, 3/2}, {1/2, 3/2}}}, \"OddRule\"]",
     "Polygon[{{0, 0}, {0, 2}, {1/2, 1/2}, {1/2, 3/2}, {3/2, 1/2}, {3/2, 3/2}, {2, 0}, {2, 2}}, {1, 7, 8, 2} -> {{4, 6, 5, 3}}]"),
    ("WindingPolygon[{{0, 0}, {0, 1}, {1, 1}, {1, 0}}, \"PositiveRule\"]", "EmptyRegion[2]"),
    ("WindingPolygon[{{0, 0}, {1, 0}, {1, 1}}, \"Bogus\"]", Raises(ValueError)),
    ("CrossingPolygon[{{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, {{1/2, 1/2}, {3/2, 1/2}, {3/2, 3/2}, {1/2, 3/2}}}]",
     "Polygon[{{0, 0}, {0, 2}, {1/2, 1/2}, {1/2, 3/2}, {3/2, 1/2}, {3/2, 3/2}, {2, 0}, {2, 2}}, {1, 7, 8, 2} -> {{4, 6, 5, 3}}]"),
    ("CrossingPolygon[{{0, 0}, {1, 0}, {0, 1}}]", "Polygon[{{0, 0}, {0, 1}, {1, 0}}, {1, 3, 2}]"),
    ("CrossingPolygon[Polygon[{{0, 0}, {1, 0}, {0, 1}}]]", "Polygon[{{0, 0}, {0, 1}, {1, 0}}, {1, 3, 2}]"),
    ("CrossingPolygon[{{0, 0}, {1, 1}, {1, 0}, {0, 1}}]", Raises(NotImplementedError)),
    ("PolygonDecomposition[Polygon[{{0, 0}, {2, 0}, {2, 2}, {1, 1}, {0, 2}}]]",
     "Polygon[{{{2, 0}, {2, 2}, {1, 1}, {0, 0}}, {{0, 0}, {1, 1}, {0, 2}}}]"),
    ("PolygonDecomposition[Polygon[{{0, 0}, {2, 0}, {2, 2}, {1, 1}, {0, 2}}], \"Triangle\"]",
     Pred(lambda g: g.head == "Polygon" and len(g.args[0].args) == 3, "3 triangles")),
    ("PolygonDecomposition[" + SQ + "]", Pred(lambda g: g.head == "Polygon" and len(g.args[0].args) in (1, 4), "convex piece")),
    ("PolygonDecomposition[Polygon[{{0, 0}, {1, 0}, {1, 1}}], \"Bogus\"]", Raises(ValueError)),
]

# --- polyhedra (exact classical values; Euler's formula) ------------------------------------------------------
POLYHEDRA = [
    ("PolyhedronCoordinates[Cube[]]", "Tuples[{-1/2, 1/2}, 3]"),
    ("PolyhedronCoordinates[Tetrahedron[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]]", "{{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 0}}"),
    ("PolyhedronCoordinates[Cuboid[{0, 0, 0}, {1, 2, 3}]]", "Tuples[{{0, 1}, {0, 2}, {0, 3}}]"),
    ("PolyhedronCoordinates[Cube[{0, 0, w19a}, 1]]", "PolyhedronCoordinates[Cube[{0, 0, w19a}, 1]]"),
    ("PolyhedronCoordinates[Icosahedron[]]", Raises(NotImplementedError)),
    ("PolyhedronCoordinates[Disk[]]", Raises(TypeError)),
    ("PolyhedronGenus[Cube[]]", 0),
    ("PolyhedronGenus[Dodecahedron[]]", 0),
    ("PolyhedronGenus[" + TORUS + "]", 1),
    ("PolyhedronGenus[Polyhedron[{{0,0,0},{1,0,0},{0,1,0}}, {{1,2,3}}]]", Raises(ValueError)),
    ("ConvexPolyhedronQ[Icosahedron[]]", True),
    ("ConvexPolyhedronQ[" + TET + "]", True),
    ("ConvexPolyhedronQ[" + TORUS + "]", False),
    ("ConvexPolyhedronQ[AugmentedPolyhedron[Cuboid[], -1/4]]", False),
    ("ConvexPolyhedronQ[Polyhedron[{{0,0,0},{1,0,0},{0,1,0}}, {{1,2,3}}]]", Raises(ValueError)),
    ("SimplePolyhedronQ[Cube[]]", True),
    ("SimplePolyhedronQ[" + TORUS + "]", False),
    ("SimplePolyhedronQ[Polyhedron[{{0,0,0},{1,0,0},{0,1,0}}, {{1,2,3}}]]", False),
    ("SimplePolyhedronQ[AugmentedPolyhedron[Cuboid[], -1/4]]", True),
    ("SimplePolyhedronQ[\"a\"]", Raises(TypeError)),
    ("PolyhedronAngle[Cube[], {1/2, 1/2, 1/2}]", "Pi/2"),
    ("PolyhedronAngle[Cube[], {{1/2, 1/2, 1/2}, {1/2, 1/2, -1/2}}]", "Pi/2"),
    ("PolyhedronAngle[" + TET + ", {{0, 0, 0}, {1, 0, 0}}]", "ArcCos[1/3]"),
    ("N[PolyhedronAngle[" + TET + ", {0, 0, 0}]]", math.acos(23 / 27)),
    ("PolyhedronAngle[" + TORUS + ", {{1, 1, 0}, {1, 1, 1}}]", "3 Pi/2"),
    ("PolyhedronAngle[Cube[], {5, 5, 5}]", Raises(ValueError)),
    ("PolyhedronAngle[" + TORUS + ", {1, 1, 0}]", Raises(NotImplementedError)),
    ("DualPolyhedron[Cube[]]", Pred(lambda g: g.head == "Polyhedron" and len(g.args[0].args) == 6 and len(g.args[1].args) == 8,
                                    "octahedron: 6 vertices, 8 faces")),
    ("PolyhedronCoordinates[DualPolyhedron[Cube[]]]", "{{-1/2, 0, 0}, {0, -1/2, 0}, {0, 0, -1/2}, {0, 0, 1/2}, {0, 1/2, 0}, {1/2, 0, 0}}"),
    ("ConvexPolyhedronQ[DualPolyhedron[Icosahedron[]]]", Raises(NotImplementedError)),
    ("PolyhedronGenus[DualPolyhedron[Cuboid[{0, 0, 0}, {2, 1, 1}]]]", 0),
    ("DualPolyhedron[" + TORUS + "]", Raises(NotImplementedError)),
    ("TruncatedPolyhedron[Cube[], 1/2]", Pred(lambda g: g.head == "Polyhedron" and len(g.args[0].args) == 12 and len(g.args[1].args) == 14,
                                              "cuboctahedron: 12 vertices, 14 faces")),
    ("TruncatedPolyhedron[Cuboid[], 1/3]", Pred(lambda g: len(g.args[0].args) == 24 and len(g.args[1].args) == 14, "24 vertices, 14 faces")),
    ("ConvexPolyhedronQ[TruncatedPolyhedron[Cube[], 1/4]]", True),
    ("TruncatedPolyhedron[Cube[]]", Raises(NotImplementedError)),
    ("TruncatedPolyhedron[Cube[], 2]", Raises(ValueError)),
    ("TruncatedPolyhedron[Cube[], w19a]", "TruncatedPolyhedron[Cube[], w19a]"),
    ("AugmentedPolyhedron[Cuboid[], 1]", Pred(lambda g: len(g.args[0].args) == 14 and len(g.args[1].args) == 24, "14 vertices, 24 faces")),
    ("PolyhedronGenus[AugmentedPolyhedron[Cube[], 1/2]]", 0),
    ("PolyhedronCoordinates[AugmentedPolyhedron[Cuboid[], 1]][[-1]]", "{2, 1/2, 1/2}"),
    ("AugmentedPolyhedron[Cube[]]", Raises(NotImplementedError)),
    ("AugmentedPolyhedron[Cube[], 1, 2]", Raises(ValueError)),
]

# --- splines: documented examples, scipy.interpolate.BSpline and explicit Bernstein sums ---------------------
SPLINES = [
    ("BernsteinBasis[4, 3, 0.5]", 0.25),                      # ref/BernsteinBasis example
    ("BernsteinBasis[3, 1, 1/3]", "4/9"),
    ("BernsteinBasis[3, 1, 2]", "0"),
    ("Sum[BernsteinBasis[5, k, 3/10], {k, 0, 5}]", "1"),
    ("BernsteinBasis[3, 1, w19x]", "BernsteinBasis[3, 1, w19x]"),
    ("BernsteinBasis[3/2, 1, 0.5]", Raises(TypeError)),
    ("BernsteinBasis[-1, 0, 0.5]", Raises(ValueError)),
    ("BSplineBasis[3, 0.5]", Num(2 / 3)),                     # ref/BSplineBasis example (0.666667)
    ("BSplineBasis[{3, {0, 0, 0, 0, 1/3, 2/3, 1, 1, 1, 1}}, 2, 0.5]", Num(0.46875)),   # scipy basis_element
    ("BSplineBasis[{3, {0, 0, 0, 0, 1/3, 2/3, 1, 1, 1, 1}}, 5, 1]", "1"),
    ("BSplineBasis[2, 1, 1/2]", "1/8"),                       # uniform knots {1/3, 2/3, 1, 4/3}
    ("Total[Table[BSplineBasis[{2, {0, 0, 0, 1/4, 1/2, 3/4, 1, 1, 1}}, i, 3/5], {i, 0, 5}]]", "1"),
    ("BSplineBasis[3, w19x]", "BSplineBasis[3, w19x]"),
    ("BSplineBasis[{3, {0, 0, 0, 0, 1/3, 2/3, 1, 1, 1, 1}}, 6, 1]", Raises(ValueError)),
    ("BSplineBasis[{1, {0, 1, 1/2}}, 0, 0.5]", Raises(ValueError)),
    ("BSplineBasis[\"a\", 0.5]", Raises(TypeError)),
    ("BezierFunction[{{0, 0}, {1, 1}, {2, 0}, {3, 2}}][0.5]", [1.5, 0.625]),
    ("BezierFunction[{{0, 0}, {1, 1}, {2, 0}, {3, 2}}][1/2]", "{3/2, 5/8}"),
    ("BezierFunction[{{0, 0}, {1, 1}, {2, 0}, {3, 2}}][1]", "{3, 2}"),
    ("BezierFunction[{{{0, 0, 0}, {0, 1, 0}}, {{1, 0, 0}, {1, 1, 1}}}][1/2, 1/2]", "{1/2, 1/2, 1/4}"),
    ("BezierFunction[{{0, 0}, {1, 1}}]", "BezierFunction[{{0, 0}, {1, 1}}]"),
    ("BezierFunction[{{0, 0}, {1, 1}}][w19t]", "BezierFunction[{{0, 0}, {1, 1}}][w19t]"),
    ("BezierFunction[{{0, 0}, {1, 1}}][2]", Raises(NotImplementedError)),
    ("BezierFunction[{{0, 0}, {1, 1}}][1/2, 1/2]", Raises(ValueError)),
    ("BezierFunction[\"a\"]", Raises(TypeError)),
    ("BSplineFunction[{{1, 1}, {2, 3}, {3, -1}, {4, 1}, {5, 0}}][0.5]", [3.0, 0.5]),   # scipy BSpline, clamped cubic
    ("BSplineFunction[{{1, 1}, {2, 3}, {3, -1}, {4, 1}, {5, 0}}][0]", "{1, 1}"),
    ("BSplineFunction[{{1, 1}, {2, 3}, {3, -1}, {4, 1}, {5, 0}}][1]", "{5, 0}"),
    ("BSplineFunction[{{0, 0}, {1, 1}, {2, 0}}, SplineDegree -> 2, SplineWeights -> {1, 2, 1}][1/2]", "{1, 2/3}"),
    ("BSplineFunction[{{0, 1}, {2, 3}}, 2][1/2, 1/2]", "3/2"),
    ("BSplineFunction[{{0, 0}, {1, 2}, {2, 0}, {3, 2}}, SplineKnots -> {0, 0, 0, 1/2, 1, 1, 1}][1/4]", "{7/8, 5/4}"),    # scipy BSpline: [0.875, 1.25]
    ("BSplineFunction[{{0, 0}, {1, 1}}]", "BSplineFunction[{{0, 0}, {1, 1}}]"),
    ("BSplineFunction[{{0, 0}, {1, 1}}][w19t]", "BSplineFunction[{{0, 0}, {1, 1}}][w19t]"),
    ("BSplineFunction[{{0, 0}, {1, 1}, {2, 0}}, SplineClosed -> True]", Raises(NotImplementedError)),
    ("BSplineFunction[{{0, 0}, {1, 1}, {2, 0}}, SplineKnots -> {0, 1}]", Raises(ValueError)),
    ("BSplineFunction[\"a\"]", Raises(TypeError)),
]


def _in_disk(c, r):
    return lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 <= r * r + 1e-9


def _in_box(lo, hi):
    return lambda p: all(a - 1e-12 <= x <= b + 1e-12 for x, a, b in zip(p, lo, hi))


# --- random geometry: membership of every sample (independent tests written here) ----------------------------
RANDOM = [
    ("RandomPoint[Disk[{5, 5}, 1], 50]", shape_pts(50, 2, _in_disk((5, 5), 1))),
    ("RandomPoint[Ball[{1, 1, 1}, 2], 20]", shape_pts(20, 3, lambda p: sum((x - 1) ** 2 for x in p) <= 4 + 1e-9)),
    ("RandomPoint[Circle[], 20]", shape_pts(20, 2, lambda p: abs(p[0] ** 2 + p[1] ** 2 - 1) < 1e-9)),
    ("RandomPoint[Rectangle[{0, 0}, {2, 1}], 30]", shape_pts(30, 2, _in_box((0, 0), (2, 1)))),
    ("RandomPoint[Cube[], 30]", shape_pts(30, 3, _in_box((-.5, -.5, -.5), (.5, .5, .5)))),
    ("RandomPoint[Triangle[], 30]", shape_pts(30, 2, lambda p: p[0] >= -1e-12 and p[1] >= -1e-12 and p[0] + p[1] <= 1 + 1e-12)),
    ("RandomPoint[Annulus[], 30]", shape_pts(30, 2, lambda p: 0.25 - 1e-9 <= p[0] ** 2 + p[1] ** 2 <= 1 + 1e-9)),
    ("RandomPoint[Cylinder[], 30]", shape_pts(30, 3, lambda p: p[0] ** 2 + p[1] ** 2 <= 1 + 1e-9 and -1 <= p[2] <= 1)),
    ("RandomPoint[" + HOLED + ", 30]", shape_pts(30, 2, lambda p: 0 <= p[0] <= 4 and 0 <= p[1] <= 4 and
                                                 not (1 < p[0] < 2 and 1 < p[1] < p[0]))),
    ("RandomPoint[InfiniteLine[{0, 0}, {1, 1}], 10, {{0, 1}, {0, 2}}]", shape_pts(10, 2, lambda p: abs(p[0] - p[1]) < 1e-12 and 0 <= p[0] <= 1)),
    ("RandomPoint[HalfSpace[{0, 1}, 0], 10, {{-1, 1}, {-1, 1}}]", shape_pts(10, 2, lambda p: p[1] <= 1e-12)),
    ("RandomPoint[Circle[], {3, 4}]", Pred(lambda g: len(g.args) == 3 and all(len(r.args) == 4 for r in g.args), "3 x 4 array")),
    ("RandomPoint[Disk[]]", Pred(lambda g: g.head == "List" and len(g.args) == 2 and isinstance(g.args[0], float), "one point")),
    ("RandomPoint[Point[{{1}, {2}, {3}}], 10]", shape_pts(10, 1, lambda p: p[0] in (1.0, 2.0, 3.0))),
    ("RandomPoint[HalfSpace[{0, 1}, 0]]", Raises(ValueError)),
    ("RandomPoint[Torus[]]", Raises(NotImplementedError)),
    ("RandomPoint[Disk[], -1]", Raises(ValueError)),
    ("RandomPoint[Disk[], \"a\"]", Raises(TypeError)),
    ("RandomPoint[w19reg]", "RandomPoint[w19reg]"),
    ("SimplePolygonQ[RandomPolygon[20]]", True),
    ("ConvexPolygonQ[RandomPolygon[{\"Convex\", 30}]]", True),
    ("Length[First[RandomPolygon[{\"StarShaped\", 12}]]]", 12),
    ("Length[RandomPolygon[{\"ConvexHull\", 10}, 4]]", 4),
    ("RandomPolygon[{\"Convex\", 5}, DataRange -> {{2, 3}, {5, 7}}]",
     Pred(lambda g: all(2 - 1e-9 <= float(p.args[0]) <= 3 + 1e-9 and 5 - 1e-9 <= float(p.args[1]) <= 7 + 1e-9 for p in g.args[0].args),
          "inside DataRange")),
    ("RandomPolygon[{\"Bogus\", 5}]", Raises(ValueError)),
    ("RandomPolygon[2]", Raises(ValueError)),
    ("RandomPolygon[{\"ConvexHull\", NormalDistribution[], 5}]", Raises(NotImplementedError)),
    ("RandomPolygon[\"a\"]", Raises(TypeError)),
    ("ConvexPolyhedronQ[RandomPolyhedron[{\"ConvexHull\", 20}]]", True),
    ("PolyhedronGenus[RandomPolyhedron[{\"ConvexHull\", 12}]]", 0),
    ("Length[RandomPolyhedron[{\"ConvexHull\", 8}, 3]]", 3),
    ("RandomPolyhedron[{\"ConvexHull\", 3}]", Raises(ValueError)),
    ("RandomPolyhedron[{\"ConvexHull\", NormalDistribution[], 8}]", Raises(NotImplementedError)),
    ("RandomPolyhedron[{\"ConvexHull\", 8}, \"a\"]", Raises(TypeError)),
]

# --- GeometricTest: documented examples and classical facts ---------------------------------------------------
GEOTEST = [
    ("GeometricTest[{{2, 3}, {4, 6}, {-2, -3}}, \"Collinear\"]", True),          # ref/GeometricTest
    ("GeometricTest[{{1, 0}, {4, 2}, {8, 8}}, \"Collinear\"]", False),           # ref/GeometricTest
    ("GeometricTest[Polygon[{{0, 0}, {5, 1}, {4, 4}, {-2, 0}}], \"Convex\"]", True),   # ref/GeometricTest
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], InfiniteLine[{{1, 1}, {2, 2}}]}, \"Parallel\"]", False),  # ref/GeometricTest
    ("GeometricTest[Triangle[{{0, 0}, {1, 0}, {1/2, Sqrt[3]/2}}], \"Equilateral\", \"Counterclockwise\", \"Regular\"]", True),
    ("GeometricTest[Polygon[{{0, 0}, {2, 0}, {2, 1}, {0, 1}}], \"Rectangle\", \"Cyclic\", \"Equiangular\"]", True),
    ("GeometricTest[Polygon[{{0, 0}, {2, 0}, {3, 1}, {1, 1}}], \"Parallelogram\"]", True),
    ("GeometricTest[Polygon[{{0, 0}, {2, 0}, {3, 1}, {1, 1}}], \"Rectangle\"]", False),
    ("GeometricTest[Point[{0, 0}], {\"Inside\", Circle[{0, 0}, 1]}]", True),
    ("GeometricTest[{Line[{{2, 2}, {3, 3}}], Circle[{5, 5}, 1]}, {\"Outside\", Circle[{0, 0}, 1]}]", True),
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], InfiniteLine[{{0, 2}, {2, 0}}], InfiniteLine[{{1, 0}, {1, 5}}]}, \"Concurrent\"]", True),
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 1}}], InfiniteLine[{{0, 2}, {2, 0}}]}, {\"Concurrent\", {1, 1}}]", True),
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], InfiniteLine[{{1, 1}, {2, 1}}]}, \"MatchingParallel\", \"Nondegenerate\"]", True),
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], InfiniteLine[{{1, 1}, {0, 1}}]}, \"Antiparallel\"]", True),
    ("GeometricTest[{InfiniteLine[{{0, 0}, {1, 0}}], InfiniteLine[{{0, 0}, {0, 1}}]}, \"Perpendicular\"]", True),
    ("GeometricTest[{Triangle[{{0, 0}, {1, 0}, {0, 1}}], Triangle[{{0, 0}, {2, 0}, {0, 2}}]}, \"Similar\"]", True),
    ("GeometricTest[{Triangle[{{0, 0}, {1, 0}, {0, 1}}], Triangle[{{0, 0}, {2, 0}, {0, 2}}]}, \"Congruent\"]", False),
    ("GeometricTest[{Circle[{0, 0}, 1], Circle[{3, 0}, 2]}, \"PairwiseTangent\"]", True),
    ("GeometricTest[{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}, {\"EqualAngles\", {0, 0}}]", True),
    ("GeometricTest[{Point[{0, 1}], Point[{2, 3}]}, {\"SameSide\", InfiniteLine[{{0, 0}, {1, 0}}]}]", True),
    ("GeometricTest[{Triangle[{{0, 0}, {1, 0}, {0, 1}}], Circle[{0, -5}, 1]}, {\"OppositeSides\", InfiniteLine[{{0, -1}, {1, -1}}]}]", True),
    ("GeometricTest[Line[{{0, 0}, {-1, 2}}], \"Leftward\", \"Upward\"]", True),
    ("GeometricTest[Line[{{0, 0}, {1, 0}}], \"Horizontal\", \"Rightward\"]", True),
    ("GeometricTest[{{0, 0}, {1, 0}, {0, 1}}, \"Counterclockwise\"]", True),
    ("GeometricTest[{{0, 0}, {1, 0}, {0, 1}}, \"Clockwise\"]", False),
    ("GeometricTest[{{0, 0}, {1, 0}, {1, 0}}, \"Distinct\"]", False),
    ("GeometricTest[Polygon[{{0, 0}, {1, 1}, {1, 0}, {0, 1}}], \"Simple\"]", False),
    ("GeometricTest[{{w19a, w19b}, {w19c, w19d}, {w19e, w19f}}, \"Collinear\"]",
     "GeometricTest[{{w19a, w19b}, {w19c, w19d}, {w19e, w19f}}, \"Collinear\"]"),
    ("GeometricTest[{{0, 0}, {1, 1}}, \"Bogus\"]", Raises(ValueError)),
    ("GeometricTest[{{0, 0}, {1, 1}}, 3]", Raises(TypeError)),
    ("GeometricTest[{Circle[{0, 0}, 1], Circle[{2, 0}, 1]}, \"Tangent\"]", Raises(NotImplementedError)),
]

# --- Check methods: self-checks and exact identities exercised explicitly -------------------------------------
CHECKS = [
    # nice_angle exactness: interior angles of a regular n-gon are exactly (n - 2) Pi / n
    ("PolygonAngle[RegularPolygon[9], 1]", "7 Pi/9"),
    # Euler characteristic self-check of TruncatedPolyhedron; volume of the cuboctahedron is 5/6 of the unit cube
    ("PolyhedronGenus[TruncatedPolyhedron[Cube[], 1/2]]", 0),
    # SSSTriangle self-check: the constructed triangle has the requested sides
    ("TriangleMeasurement[SSSTriangle[5, 6, 7], \"Perimeter\"]", "18"),
    # BSplineFunction vs. explicit Cox-de Boor sum with BSplineBasis
    ("BSplineFunction[{{0, 0}, {1, 2}, {2, 0}, {3, 2}}, SplineKnots -> {0, 0, 0, 1/2, 1, 1, 1}][3/10] == "
     "Total[Table[{{0, 0}, {1, 2}, {2, 0}, {3, 2}}[[i + 1]] BSplineBasis[{2, {0, 0, 0, 1/2, 1, 1, 1}}, i, 3/10], {i, 0, 3}]]", True),
    # Insphere touches every face of the 3-4-5 triangle: radius = area / semiperimeter
    ("Insphere[" + P345 + "][[2]] == TriangleMeasurement[" + P345 + ", \"Area\"] / 6", True),
    # CircumscribedBall contains every point
    ("Max[Map[Norm[# - {2, 3/2}] &, {{0, 0}, {4, 0}, {0, 3}, {1, 1}}]] == 5/2", True),
]

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
NEEDS_SCIPY = ("InscribedBall[", "RandomPolyhedron[")

CASES = (PRIMITIVES + OPTION_SYMBOLS + SCENES + SYNTHETIC + TRIANGLES + SPHERES + POLYGONS + POLYHEDRA + SPLINES
         + RANDOM + GEOTEST + CHECKS)
if not HAVE_SCIPY:  # minimal install: these heads raise the clear 'requires SciPy' error instead
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_SCIPY)]
    CASES += [("InscribedBall[{{0, 0}, {4, 0}, {0, 3}}]", Raises(NotImplementedError, "requires SciPy")),
              ("RandomPolyhedron[{\"ConvexHull\", 8}]", Raises(NotImplementedError, "requires SciPy"))]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w19(case):
    check(case)


def test_w19_seedrandom_repeatable():
    a = ev("SeedRandom[19]; RandomPoint[Disk[], 3]")
    b = ev("SeedRandom[19]; RandomPoint[Disk[], 3]")
    assert a == b


def test_w19_random_point_uniform_mean():
    """Empirical mean of uniform samples in a triangle is its centroid (standard error ~ 0.004)."""
    pts_ = ev("SeedRandom[7]; RandomPoint[Triangle[{{0, 0}, {3, 0}, {0, 3}}], 4000]")
    mx = sum(float(p.args[0]) for p in pts_.args) / len(pts_.args)
    my = sum(float(p.args[1]) for p in pts_.args) / len(pts_.args)
    assert abs(mx - 1) < 0.05 and abs(my - 1) < 0.05


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w19_bspline_matches_scipy():
    import numpy as np
    from scipy.interpolate import BSpline
    ctrl = np.array([[1, 1], [2, 3], [3, -1], [4, 1], [5, 0]], dtype=float)
    knots = [0, 0, 0, 0, 0.5, 1, 1, 1, 1]
    ref = BSpline(knots, ctrl, 3)
    for u in (0.1, 0.37, 0.5, 0.82):
        got = ev(f"BSplineFunction[{{{{1, 1}}, {{2, 3}}, {{3, -1}}, {{4, 1}}, {{5, 0}}}}][{u}]")
        assert np.allclose([float(c) for c in got.args], ref(u), atol=1e-12)


def test_w19_registry_for_later_batches():
    from mikoshilang.library import wl_geomprims as g
    from mikoshilang.parser import parse
    from mikoshilang.evaluate import evaluate
    V, F = g.polyhedron_data(evaluate(parse("Cuboid[]")), "test")
    assert len(V) == 8 and len(F) == 6
    assert len(g.polygon_data(evaluate(parse(HOLED)), "test")) == 2
    d, s = g.region_sampler(evaluate(parse("Disk[]")), "test")
    assert d == 2
