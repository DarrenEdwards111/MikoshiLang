"""Differential tests for Wolfram batch W22 (geometry II: region computation, measures, meshes, geodesy and
geometric transforms; modules wl_regions, wl_regions_mesh, wl_regions_xform, wl_regions_geo).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented argument forms,
  defaults and examples, e.g. ``RegionMeasure[Disk[]] == Pi``, ``Area[Disk[{0, 0}, {2, 3}]] == 6 Pi``,
  ``RegionMeanCurvature[Sphere[], {1, 0, 0}] == 1``, ``RegionCentroid`` of an unbounded region giving
  ``Indeterminate``, ``RotationTransform[Pi/2][{1, 0}] == {0, 1}``, ``EulerMatrix`` / ``RollPitchYawMatrix``
  conventions (z-y-z and z-y-x axis sequences), ``MeshCellCount`` of a triangulated square ``{4, 5, 2}``,
  ``BoundaryMesh`` giving a BoundaryMeshRegion, ``DimensionalMeshComponents`` padding with ``EmptyRegion``,
  ``FromDMS[{20, 32, 15.828}] == 20.5377``, ``CoordinateBounds`` / ``CoordinateBoundsArray`` examples.
* Classical geometry (exact closed forms): areas / volumes / surface areas of discs, balls, cones, cylinders,
  capsules, ellipsoids and Platonic solids (Volume[Icosahedron[]] = 5 (3 + Sqrt[5]) / 12, Volume[Octahedron[]] =
  Sqrt[2]/3, Volume[Tetrahedron[]] = Sqrt[2]/12), centroids (half disc 4/(3 Pi), cone at a quarter of the height),
  moments of inertia (unit ball 8 Pi/15, unit cube 1/6 about its centre), curvatures of spheres (1/r, Gaussian
  1/r^2) and cylinders (1/r and 0), Euler characteristics (V - E + F), inclusion-exclusion for Boolean regions.
* GeographicLib 2.1 (Karney's geodesic algorithms, WGS84) for GeoDistance / GeoDirection / GeoDestination values:
  London (51.5, 0) to Paris (48.85, 2.35) s12 = 339174.0901525508 m, azi1 = 149.42970597571184 deg; one degree of
  longitude on the equator 111319.49079327357 m; quarter meridian 10001965.729 m; direct problem from (10, 20),
  azimuth 45 deg, 500 km -> (13.178676456668079, 23.259772528898296).  GeographicLib is used only to produce these
  constants; it is not a dependency of the package or of these tests.
* SciPy (independent Qhull) for the Delaunay / convex hull comparisons in the explicit test functions below.

Symbol names used as symbolic placeholders are prefixed ``w22`` so they cannot collide with other test modules.
"""
import importlib.util

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr
from mikoshilang.math_ops import to_sympy

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def _f(e):
    return float(to_sympy(e))


def Num(v, tol=1e-9):
    def ok(got):
        try:
            return abs(float(got) - v) <= tol * max(1.0, abs(v))
        except Exception:
            return False
    return Pred(ok, f"~{v}")


def _flat(e):
    if isinstance(e, Expr) and e.head in ("List", "GeoPosition"):
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


def QNum(v, unit, tol=1e-9):
    def ok(got):
        return (isinstance(got, Expr) and got.head == "Quantity" and str(got.args[1]).strip('"') == unit
                and abs(float(got.args[0]) - v) <= tol * max(1.0, abs(v)))
    return Pred(ok, f"Quantity[~{v}, {unit}]")


def is_head(head, n=None):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head and (n is None or len(g.args) == n), f"{head}[...]")


T, V, A, I, N = TypeError, ValueError, ArithmeticError, NotImplementedError, NotImplementedError

# Meshes used throughout (inlined in every case so each case is self-contained).
SQ = "MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, {Triangle[{1,2,3}], Triangle[{1,3,4}]}]"
TRI = "MeshRegion[{{0,0},{1,0},{0,1}}, {Triangle[{1,2,3}]}]"
TET = "MeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {Tetrahedron[{1,2,3,4}]}]"
TETB = ("BoundaryMeshRegion[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}, {Triangle[{1,3,2}], Triangle[{1,2,4}], "
        "Triangle[{2,3,4}], Triangle[{1,4,3}]}]")
HOLED = ("BoundaryMeshRegion[{{0,0},{4,0},{4,4},{0,4},{1,1},{2,1},{2,2},{1,2}}, Line[{1,2,3,4,1}], "
         "Line[{5,6,7,8,5}]]")
BTRI = "BoundaryMeshRegion[{{0,0},{1,0},{1,1}}, {Line[{1,2,3,1}]}]"
HP = "HalfPlane[{0,0},{1,0},{0,1}]"
PTS = "{{0,0},{1,0},{0,1},{1,1},{1/2,1/3}}"

# --- measures ----------------------------------------------------------------------------------------------------
MEASURES = [
    ("RegionMeasure[Disk[]]", "Pi"),
    ("RegionMeasure[Line[{{0,0},{3,4}}]]", "5"),
    ("RegionMeasure[Sphere[]]", "4 Pi"),
    ("RegionMeasure[Point[{1,2}]]", "1"),
    ("RegionMeasure[%HP%]", "Infinity"),
    ("RegionMeasure[Disk[], 1]", "Infinity"),
    ("RegionMeasure[Disk[], 3]", "0"),
    ("RegionMeasure[ImplicitRegion[w22x^2+w22y^2<=4, {w22x,w22y}]]", "4 Pi"),
    ("RegionMeasure[ImplicitRegion[0<=w22x<=3 || 5<=w22x<=6, {w22x}]]", "4"),
    ("RegionMeasure[%SQ%]", "1"),
    ("RegionMeasure[w22r]", "RegionMeasure[w22r]"),
    ("RegionMeasure[5]", Raises(T, "not a geometric region")),
    ("RegionMeasure[Disk[], 1.5]", Raises(T, "non-negative integer")),
    ("RegionMeasure[ParametricRegion[{Cos[w22t], Sin[w22t]}, {{w22t,0,Pi}}]]", Raises(N, "not supported")),
    ("Area[Rectangle[{0,0},{2,3}]]", "6"),
    ("Area[Triangle[{{0,0},{3,0},{0,4}}]]", "6"),
    ("Area[Disk[{0,0},{2,3}]]", "6 Pi"),
    ("Area[Annulus[]]", "3 Pi/4"),
    ("Area[Disk[{0,0},1,{0,Pi/2}]]", "Pi/4"),
    ("Area[Sphere[]]", "4 Pi"),
    ("Area[Ball[]]", "Undefined"),
    ("Area[ImplicitRegion[w22x^2/4+w22y^2<=1,{w22x,w22y}]]", "2 Pi"),
    ("Area[ImplicitRegion[0<=w22x<=1 && 0<=w22y<=w22x, {w22x,w22y}]]", "1/2"),
    ("Area[{w22u Cos[w22v], w22u Sin[w22v]}, {w22u,0,1}, {w22v,0,2Pi}]", "Pi"),
    ("Area[%HOLED%]", "15"),
    ("Area[RegionDifference[Disk[], Disk[{0,0},1/2]]]", "3 Pi/4"),
    ("Area[RegionUnion[Disk[], Disk[{3,0}]]]", "2 Pi"),
    ("Area[RegionUnion[Rectangle[], Rectangle[{1/2,0},{3/2,1}]]]", "3/2"),
    ("Area[w22r]", "Area[w22r]"),
    ("Area[3]", Raises(T, "not a geometric region")),
    ("Area[RegionUnion[Disk[], Disk[{1,0}]]]", Raises(N, "not supported")),
    ("Volume[Cuboid[{0,0,0},{1,2,3}]]", "6"),
    ("Volume[Ball[{0,0,0},2]]", "32 Pi/3"),
    ("Volume[Cylinder[]]", "2 Pi"),
    ("Volume[Cone[]]", "2 Pi/3"),
    ("Volume[Tetrahedron[]]", "Sqrt[2]/12"),
    ("Volume[Octahedron[]]", "Sqrt[2]/3"),
    ("Volume[Icosahedron[]]", "5 (3 + Sqrt[5])/12"),
    ("Volume[Dodecahedron[]]", "(15 + 7 Sqrt[5])/4"),
    ("Volume[Ellipsoid[{0,0,0},{1,2,3}]]", "8 Pi"),
    ("Volume[CapsuleShape[]]", "10 Pi/3"),
    ("Volume[FilledTorus[]]", "3 Pi^2/32"),
    ("Volume[Disk[]]", "Undefined"),
    ("Volume[ImplicitRegion[w22x^2+w22y^2+w22z^2<=1, {w22x,w22y,w22z}]]", "4 Pi/3"),
    ("Volume[{w22u Sin[w22t] Cos[w22p], w22u Sin[w22t] Sin[w22p], w22u Cos[w22t]}, {w22u,0,1}, {w22t,0,Pi}, "
     "{w22p,0,2Pi}]", "4 Pi/3"),
    ("Volume[%TET%]", "1/6"),
    ("Volume[%TETB%]", "1/6"),
    ("Volume[CSGRegion[\"Union\", {Cuboid[], Cuboid[{1/2,1/2,1/2},{2,2,2}]}]]", "17/4"),
    ("Volume[w22r]", "Volume[w22r]"),
    ("Volume[\"a\"]", Raises(T, "not a geometric region")),
    ("Volume[RegionUnion[Ball[], Ball[{1,0,0}]]]", Raises(N, "not supported")),
    ("Perimeter[Disk[]]", "2 Pi"),
    ("Perimeter[Rectangle[{0,0},{2,3}]]", "10"),
    ("Perimeter[Triangle[{{0,0},{3,0},{0,4}}]]", "12"),
    ("Perimeter[Annulus[]]", "3 Pi"),
    ("Perimeter[%HOLED%]", "20"),
    ("Perimeter[Ball[]]", "Undefined"),
    ("Perimeter[w22r]", "Perimeter[w22r]"),
    ("Perimeter[{w22x,w22y}]", Raises(T, "not a geometric region")),
    ("Perimeter[{Cos[w22t],Sin[w22t]},{w22t,0,2Pi}]", Raises(V, "expected Perimeter")),
    ("Perimeter[RegionUnion[Disk[], Disk[{1,0}]]]", Raises(N)),
    ("SurfaceArea[Ball[]]", "4 Pi"),
    ("SurfaceArea[Cuboid[{0,0,0},{1,2,3}]]", "22"),
    ("SurfaceArea[Cylinder[]]", "6 Pi"),
    ("SurfaceArea[Cone[]]", "Pi (1 + Sqrt[5])"),
    ("SurfaceArea[Tetrahedron[]]", "Sqrt[3]"),
    ("SurfaceArea[Dodecahedron[]]", "3 Sqrt[25 + 10 Sqrt[5]]"),
    ("SurfaceArea[Disk[]]", "Undefined"),
    ("SurfaceArea[w22r]", "SurfaceArea[w22r]"),
    ("SurfaceArea[5]", Raises(T, "not a geometric region")),
    ("SurfaceArea[{Cos[w22u],Sin[w22u],w22v},{w22u,0,2Pi},{w22v,0,1}]", Raises(V, "expected SurfaceArea")),
    ("SurfaceArea[RegionUnion[Ball[], Ball[{1,0,0}]]]", Raises(N)),
]

# --- centroids, moments, dimensions, bounds -----------------------------------------------------------------------
MOMENTS = [
    ("RegionCentroid[Disk[{1,2}]]", "{1, 2}"),
    ("RegionCentroid[Triangle[{{0,0},{3,0},{0,3}}]]", "{1, 1}"),
    ("RegionCentroid[Disk[{0,0},1,{0,Pi}]]", "{0, 4/(3 Pi)}"),
    ("RegionCentroid[Cone[]]", "{0, 0, -1/2}"),
    ("RegionCentroid[%HP%]", "{Indeterminate, Indeterminate}"),
    ("RegionCentroid[Line[{{0,0},{2,0},{2,2}}]]", "{3/2, 1/2}"),
    ("RegionCentroid[Point[{{0,0},{2,4}}]]", "{1, 2}"),
    ("RegionCentroid[%SQ%]", "{1/2, 1/2}"),
    ("RegionCentroid[RegionUnion[Rectangle[], Rectangle[{1/2,0},{3/2,1}]]]", "{3/4, 1/2}"),
    ("RegionCentroid[w22r]", "RegionCentroid[w22r]"),
    ("RegionCentroid[5]", Raises(T, "not a geometric region")),
    ("RegionCentroid[RegionUnion[Disk[], Disk[{1,0}]]]", Raises(N)),
    ("RegionMoment[Rectangle[], {1,0}]", "1/2"),
    ("RegionMoment[Disk[], {2,0}]", "Pi/4"),
    ("RegionMoment[Rectangle[], {1,1}]", "1/4"),
    ("RegionMoment[Cuboid[], {1,1,1}]", "1/8"),
    ("RegionMoment[w22r, {1,0}]", "RegionMoment[w22r, {1, 0}]"),
    ("RegionMoment[Disk[], {1,0,0}]", Raises(V, "2 exponents")),
    ("RegionMoment[Disk[], {-1,0}]", Raises(N, "non-negative integer")),
    ("RegionMoment[5, {1,0}]", Raises(T, "not a geometric region")),
    ("MomentOfInertia[Ball[]]", "8 Pi/15 IdentityMatrix[3]"),
    ("MomentOfInertia[Cuboid[]]", "IdentityMatrix[3]/6"),
    ("MomentOfInertia[Disk[]]", "{{Pi/4, 0}, {0, Pi/4}}"),
    ("MomentOfInertia[Cuboid[], {1/2,1/2,1/2}, {0,0,1}]", "1/6"),
    ("MomentOfInertia[Ball[], {0,0,0}, {0,0,1}]", "8 Pi/15"),
    ("MomentOfInertia[Line[{{0,0},{1,0}}]]", "{{0, 0}, {0, 1/12}}"),
    ("MomentOfInertia[w22r]", "MomentOfInertia[w22r]"),
    ("MomentOfInertia[5]", Raises(T, "not a geometric region")),
    ("MomentOfInertia[RegionUnion[Ball[], Ball[{1,0,0}]]]", Raises(N)),
    ("RegionDimension[Disk[]]", "2"),
    ("RegionDimension[Sphere[]]", "2"),
    ("RegionDimension[Point[{1,2,3}]]", "0"),
    ("RegionDimension[%TET%]", "3"),
    ("RegionDimension[RegionProduct[Disk[], Point[{0}]]]", "2"),
    ("RegionDimension[w22r]", "RegionDimension[w22r]"),
    ("RegionDimension[5]", Raises(T, "not a geometric region")),
    ("RegionDimension[ImplicitRegion[w22x^2+w22y^2==1,{w22x,w22y}]]", Raises(N, "cannot be determined")),
    ("RegionEmbeddingDimension[Disk[]]", "2"),
    ("RegionEmbeddingDimension[Line[{{0,0,0},{1,1,1}}]]", "3"),
    ("RegionEmbeddingDimension[%TETB%]", "3"),
    ("RegionEmbeddingDimension[w22r]", "RegionEmbeddingDimension[w22r]"),
    ("RegionEmbeddingDimension[\"a\"]", Raises(T, "not a geometric region")),
    ("RegionBounds[Disk[]]", "{{-1, 1}, {-1, 1}}"),
    ("RegionBounds[Triangle[{{0,0},{3,0},{1,2}}]]", "{{0, 3}, {0, 2}}"),
    ("RegionBounds[%HP%]", "{{-Infinity, Infinity}, {0, Infinity}}"),
    ("RegionBounds[%HOLED%]", "{{0, 4}, {0, 4}}"),
    ("RegionBounds[w22r]", "RegionBounds[w22r]"),
    ("RegionBounds[5]", Raises(T, "not a geometric region")),
    ("RegionBounds[Disk[], \"Sufficient\"]", Raises(N, "bound types")),
]

# --- membership, distances, nearest points ------------------------------------------------------------------------
DISTANCES = [
    ("RegionMember[Disk[], {1/2,1/2}]", "True"),
    ("RegionMember[Disk[], {1,1}]", "False"),
    ("RegionMember[Disk[], {1,0}]", "True"),
    ("RegionMember[Ball[], {{0,0,0},{2,0,0}}]", "{True, False}"),
    ("RegionMember[ImplicitRegion[w22x^2+w22y^2<=1, {w22x,w22y}], {1/2,1/2}]", "True"),
    ("RegionMember[ParametricRegion[{2 w22s, w22t},{{w22s,0,1},{w22t,0,1}}], {1, 1/2}]", "True"),
    ("RegionMember[%HOLED%, {3/2,3/2}]", "False"),
    ("RegionMember[%HOLED%, {3,3}]", "True"),
    ("RegionMember[Disk[], {w22x,w22y}]", "RegionMember[Disk[], {w22x, w22y}]"),
    ("RegionMember[Disk[], {1,2,3}]", Raises(V, "must have 2 coordinates")),
    ("RegionMember[Disk[], \"a\"]", Raises(T, "point must be a list")),
    ("RegionMemberFunction[Disk[]][{1/2,0}]", "True"),
    ("RegionMemberFunction[Disk[]][{{0,0},{3,3}}]", "{True, False}"),
    ("RegionMemberFunction[Rectangle[]][{2,2}]", "False"),
    ("RegionMemberFunction[Disk[]]", "RegionMemberFunction[Disk[]]"),
    ("RegionMemberFunction[Disk[]][{1,2,3}]", Raises(V, "must have 2 coordinates")),
    ("RegionMemberFunction[5]", Raises(T, "not a geometric region")),
    ("RegionDistance[Disk[], {3,4}]", "4"),
    ("RegionDistance[Rectangle[], {2,2}]", "Sqrt[2]"),
    ("RegionDistance[Disk[], {0,0}]", "0"),
    ("RegionDistance[Disk[], Disk[{5,0}]]", "3"),
    ("RegionDistance[Sphere[], {0,0,0}]", "1"),
    ("RegionDistance[Disk[], {w22x,w22y}]", "RegionDistance[Disk[], {w22x, w22y}]"),
    ("RegionDistance[Disk[], {1,2,3}]", Raises(V, "must have 2 coordinates")),
    ("RegionDistance[5, {1,2}]", Raises(T, "not a geometric region")),
    ("RegionDistanceFunction[Disk[]][{3,4}]", "4"),
    ("RegionDistanceFunction[Rectangle[]][{{2,2},{1/2,1/2}}]", "{Sqrt[2], 0}"),
    ("RegionDistanceFunction[Line[{{0,0},{1,0}}]][{1/2,3}]", "3"),
    ("RegionDistanceFunction[Disk[]]", "RegionDistanceFunction[Disk[]]"),
    ("RegionDistanceFunction[Disk[]][{1}]", Raises(V, "must have 2 coordinates")),
    ("RegionDistanceFunction[5]", Raises(T, "not a geometric region")),
    ("SignedRegionDistance[Disk[], {3,4}]", "4"),
    ("SignedRegionDistance[Disk[], {0,0}]", "-1"),
    ("SignedRegionDistance[Rectangle[], {1/2,1/4}]", "-1/4"),
    ("SignedRegionDistance[Disk[], {w22x,w22y}]", "SignedRegionDistance[Disk[], {w22x, w22y}]"),
    ("SignedRegionDistance[Line[{{0,0},{1,0}}], {0,1}]", Raises(V, "full-dimensional")),
    ("SignedRegionDistance[5, {0,1}]", Raises(T, "not a geometric region")),
    ("RegionNearest[Disk[], {3,4}]", "{3/5, 4/5}"),
    ("RegionNearest[Rectangle[], {2,1/2}]", "{1, 1/2}"),
    ("RegionNearest[Disk[], {0,0}]", "{0, 0}"),
    ("RegionNearest[Disk[], {w22x,w22y}]", "RegionNearest[Disk[], {w22x, w22y}]"),
    ("RegionNearest[Disk[], {1}]", Raises(V, "must have 2 coordinates")),
    ("RegionNearest[5, {1,1}]", Raises(T, "not a geometric region")),
    ("RegionNearestFunction[Disk[]][{3,4}]", "{3/5, 4/5}"),
    ("RegionNearestFunction[Rectangle[]][{{2,1/2},{-1,-1}}]", "{{1, 1/2}, {0, 0}}"),
    ("RegionNearestFunction[Sphere[]][{0,0,2}]", "{0, 0, 1}"),
    ("RegionNearestFunction[Disk[]]", "RegionNearestFunction[Disk[]]"),
    ("RegionNearestFunction[Disk[]][{1,2,3}]", Raises(V, "must have 2 coordinates")),
    ("RegionNearestFunction[5]", Raises(T, "not a geometric region")),
    ("RegionFarthestDistance[Disk[], {3,4}]", "6"),
    ("RegionFarthestDistance[Rectangle[], Disk[{0,0},1/2]]", "1/2 + Sqrt[2]"),
    ("RegionFarthestDistance[Disk[], {3,3}]", "1 + 3 Sqrt[2]"),
    ("RegionFarthestDistance[%HP%, {0,0}]", "Infinity"),
    ("RegionFarthestDistance[Disk[], {w22x,w22y}]", "RegionFarthestDistance[Disk[], {w22x, w22y}]"),
    ("RegionFarthestDistance[5, {0,0}]", Raises(T, "not a geometric region")),
    ("RegionFarthestDistance[Disk[], {1,2,3}]", Raises(V)),
    ("RegionHausdorffDistance[Disk[], Disk[{1,0}]]", "1"),
    ("RegionHausdorffDistance[Rectangle[], Rectangle[{0,0},{2,1}]]", "1"),
    ("RegionHausdorffDistance[Disk[], Disk[{0,0},3]]", "2"),
    ("RegionHausdorffDistance[Disk[], w22r]", "RegionHausdorffDistance[Disk[], w22r]"),
    ("RegionHausdorffDistance[Disk[], Ball[]]", Raises(V, "same embedding dimension")),
    ("RegionHausdorffDistance[Disk[], 5]", Raises(T, "not a geometric region")),
]

# --- boundaries and predicates ------------------------------------------------------------------------------------
PREDICATES = [
    ("RegionBoundary[Disk[]]", "Circle[{0, 0}, 1]"),
    ("RegionBoundary[Ball[{1,2,3},2]]", "Sphere[{1, 2, 3}, 2]"),
    ("RegionBoundary[Rectangle[]]", "Line[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {0, 0}}]"),
    ("RegionBoundary[Sphere[]]", "Sphere[]"),
    ("RegionBoundary[%SQ%]", "BoundaryMeshRegion[{{0,0},{1,0},{1,1},{0,1}}, {Line[{1,2,3,4,1}]}]"),
    ("Perimeter[RegionBoundary[Rectangle[{0,0},{2,3}]]]", "Undefined"),
    ("RegionBoundary[w22r]", "RegionBoundary[w22r]"),
    ("RegionBoundary[5]", Raises(T, "not a geometric region")),
    ("RegionQ[Disk[]]", "True"),
    ("RegionQ[5]", "False"),
    ("RegionQ[ImplicitRegion[w22x^2+w22y^2<=1,{w22x,w22y}]]", "True"),
    ("RegionQ[%SQ%]", "True"),
    ("RegionQ[w22r]", "False"),
    ("RegionQ[Disk[{0,0},-1]]", Raises(V, "non-negative")),
    ("ConstantRegionQ[Disk[]]", "True"),
    ("ConstantRegionQ[Disk[{w22a,0}]]", "False"),
    ("ConstantRegionQ[5]", "False"),
    ("ConstantRegionQ[%TET%]", "True"),
    ("BoundedRegionQ[Disk[]]", "True"),
    ("BoundedRegionQ[%HP%]", "False"),
    ("BoundedRegionQ[5]", "False"),
    ("BoundedRegionQ[%HOLED%]", "True"),
    ("BoundedRegionQ[ImplicitRegion[w22x>0,{w22x}]]", Raises(N, "cannot decide")),
    ("ConvexRegionQ[Disk[]]", "True"),
    ("ConvexRegionQ[Annulus[]]", "False"),
    ("ConvexRegionQ[Polygon[{{0,0},{2,0},{1,1},{2,2},{0,2}}]]", "False"),
    ("ConvexRegionQ[Tetrahedron[]]", "True"),
    ("ConvexRegionQ[5]", "False"),
    ("SolidRegionQ[Ball[]]", "True"),
    ("SolidRegionQ[Sphere[]]", "False"),
    ("SolidRegionQ[Disk[]]", "False"),
    ("SolidRegionQ[%TET%]", "True"),
    ("SolidRegionQ[5]", "False"),
    ("RegionWithin[Disk[], Disk[{0,0},1/2]]", "True"),
    ("RegionWithin[Disk[], Rectangle[]]", "False"),
    ("RegionWithin[Rectangle[{0,0},{2,2}], Disk[{1,1},1]]", "True"),
    ("RegionWithin[Disk[], w22r]", "RegionWithin[Disk[], w22r]"),
    ("RegionWithin[Disk[], Ball[]]", Raises(V, "same embedding dimension")),
    ("RegionWithin[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionDisjoint[Disk[], Disk[{3,0}]]", "True"),
    ("RegionDisjoint[Disk[], Disk[{1,0}]]", "False"),
    ("RegionDisjoint[Disk[], Disk[{2,0}]]", "False"),
    ("RegionDisjoint[Disk[], Rectangle[{2,2},{3,3}], Rectangle[{5,5},{6,6}]]", "True"),
    ("RegionDisjoint[w22r, Disk[]]", "RegionDisjoint[w22r, Disk[]]"),
    ("RegionDisjoint[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionEqual[Disk[], Ball[{0,0}]]", "True"),
    ("RegionEqual[Rectangle[], Polygon[{{0,0},{1,0},{1,1},{0,1}}]]", "True"),
    ("RegionEqual[Disk[], Disk[{0,0},2]]", "False"),
    ("RegionEqual[w22r, Disk[]]", "RegionEqual[w22r, Disk[]]"),
    ("RegionEqual[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionEqual[Disk[], ImplicitRegion[w22x^2+w22y^2<=1,{w22x,w22y}]]", Raises(N, "cannot decide")),
    ("RegionCongruent[Disk[], Disk[{3,4}]]", "True"),
    ("RegionCongruent[Rectangle[{0,0},{1,2}], Rectangle[{0,0},{2,1}]]", "True"),
    ("RegionCongruent[Disk[], Disk[{0,0},2]]", "False"),
    ("RegionCongruent[Triangle[{{0,0},{3,0},{0,4}}], Triangle[{{1,1},{1,4},{5,1}}]]", "True"),
    ("RegionCongruent[w22r, Disk[]]", "RegionCongruent[w22r, Disk[]]"),
    ("RegionCongruent[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionSimilar[Disk[], Disk[{3,4},5]]", "True"),
    ("RegionSimilar[Rectangle[{0,0},{1,2}], Rectangle[{0,0},{4,2}]]", "True"),
    ("RegionSimilar[Rectangle[{0,0},{1,2}], Rectangle[{0,0},{3,2}]]", "False"),
    ("RegionSimilar[Disk[], Rectangle[]]", "False"),
    ("RegionSimilar[w22r, Disk[]]", "RegionSimilar[w22r, Disk[]]"),
    ("RegionSimilar[Disk[], 5]", Raises(T, "not a geometric region")),
]

# --- region constructors (inert, validated) -----------------------------------------------------------------------
CONSTRUCTORS = [
    ("ImplicitRegion[w22x^2+w22y^2<=1, {w22x,w22y}]", is_head("ImplicitRegion", 2)),
    ("ImplicitRegion[w22x>0, w22x]", is_head("ImplicitRegion", 2)),
    ("ImplicitRegion[w22x^2+w22y^2<=1, {{w22x,-1,1},w22y}]", is_head("ImplicitRegion", 2)),
    ("ImplicitRegion[w22c, {w22x}]", "ImplicitRegion[w22c, {w22x}]"),
    ("ImplicitRegion[w22x^2<=1, 5]", Raises(T, "variables must be a list")),
    ("ParametricRegion[{Cos[w22t],Sin[w22t]},{{w22t,0,2Pi}}]", is_head("ParametricRegion", 2)),
    ("ParametricRegion[{w22s,w22t},{{w22s,0,1},{w22t,0,1}}]", is_head("ParametricRegion", 2)),
    ("Area[ParametricRegion[{2 w22s, w22t},{{w22s,0,1},{w22t,0,1}}]]", "2"),
    ("ParametricRegion[{w22s,w22t},5]", Raises(T, "variables must be a list")),
    ("Region[Disk[]]", "Region[Disk[]]"),
    ("Region[Disk[], Axes->True]", "Region[Disk[], Axes -> True]"),
    ("Area[Region[Rectangle[{0,0},{2,3}]]]", "6"),
    ("Region[w22r]", "Region[w22r]"),
    ("Region[5]", Raises(T, "not a geometric region")),
    ("RegionUnion[Disk[], Disk[{3,0}]]", "RegionUnion[Disk[], Disk[{3, 0}]]"),
    ("RegionUnion[Disk[]]", "Disk[]"),
    ("RegionMember[RegionUnion[Disk[], Disk[{3,0}]], {3,0}]", "True"),
    ("RegionUnion[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionUnion[Disk[], Ball[]]", Raises(V, "same embedding dimension")),
    ("RegionIntersection[Disk[], Rectangle[]]", "RegionIntersection[Disk[], Rectangle[]]"),
    ("Area[RegionIntersection[Rectangle[], Rectangle[{1/2,1/2},{2,2}]]]", "1/4"),
    ("RegionMember[RegionIntersection[Disk[], Rectangle[]], {1/2,1/2}]", "True"),
    ("RegionIntersection[Disk[], 5]", Raises(T, "not a geometric region")),
    ("RegionIntersection[Disk[], Ball[]]", Raises(V, "same embedding dimension")),
    ("RegionDifference[Disk[], Disk[{0,0},1/2]]", "RegionDifference[Disk[], Disk[{0, 0}, 1/2]]"),
    ("Area[RegionDifference[Rectangle[{0,0},{2,2}], Rectangle[]]]", "3"),
    ("Area[RegionDifference[Rectangle[], Rectangle[{1/2,1/2},{2,2}]]]", "3/4"),
    ("RegionDifference[Disk[], \"a\"]", Raises(T, "not a geometric region")),
    ("RegionDifference[Disk[]]", Raises(V, "expected 2")),
    ("RegionSymmetricDifference[Rectangle[], Rectangle[{1/2,0},{3/2,1}]]",
     "RegionSymmetricDifference[Rectangle[], Rectangle[{1/2, 0}, {3/2, 1}]]"),
    ("Area[RegionSymmetricDifference[Rectangle[], Rectangle[{1/2,0},{3/2,1}]]]", "1"),
    ("RegionMember[RegionSymmetricDifference[Rectangle[], Rectangle[{1/2,0},{3/2,1}]], {3/4,1/2}]", "False"),
    ("RegionSymmetricDifference[Disk[], 5]", Raises(T, "not a geometric region")),
    ("BooleanRegion[And, {Disk[], Rectangle[]}]", "BooleanRegion[And, {Disk[], Rectangle[]}]"),
    ("RegionMember[BooleanRegion[#1 && !#2 &, {Disk[], Disk[{0,0},1/2]}], {3/4,0}]", "True"),
    ("RegionMember[BooleanRegion[Xor, {Rectangle[], Rectangle[{1/2,0},{3/2,1}]}], {1/4,1/2}]", "True"),
    ("BooleanRegion[And, Disk[]]", Raises(T, "must be a list")),
    ("BooleanRegion[And, {Disk[], 5}]", Raises(T, "not a geometric region")),
    ("TransformedRegion[Disk[], TranslationTransform[{1,2}]]",
     "TransformedRegion[Disk[], TransformationFunction[{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}]]"),
    ("Area[TransformedRegion[Disk[], ScalingTransform[{2,3}]]]", "6 Pi"),
    ("RegionCentroid[TransformedRegion[Disk[], TranslationTransform[{1,2}]]]", "{1, 2}"),
    ("TransformedRegion[5, TranslationTransform[{1,2}]]", Raises(T, "not a geometric region")),
    ("TransformedRegion[Disk[], 5]", Raises(T, "not a transformation")),
    ("InverseTransformedRegion[Disk[], TranslationTransform[{1,2}]]",
     "InverseTransformedRegion[Disk[], TransformationFunction[{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}]]"),
    ("RegionCentroid[InverseTransformedRegion[Disk[], TranslationTransform[{1,2}]]]", "{-1, -2}"),
    ("Area[InverseTransformedRegion[Disk[], ScalingTransform[{2,2}]]]", "Pi/4"),
    ("RegionMember[InverseTransformedRegion[Disk[], {2 #[[1]], #[[2]]} &], {1/4, 1/2}]", "True"),
    ("RegionMember[InverseTransformedRegion[Disk[], {2 #[[1]], #[[2]]} &], {3/4, 0}]", "False"),
    ("InverseTransformedRegion[5, TranslationTransform[{1,2}]]", Raises(T, "not a geometric region")),
    ("RegionProduct[Line[{{0},{1}}], Line[{{0},{2}}]]", "RegionProduct[Line[{{0}, {1}}], Line[{{0}, {2}}]]"),
    ("Area[RegionProduct[Line[{{0},{1}}], Line[{{0},{2}}]]]", "2"),
    ("Volume[RegionProduct[Disk[], Line[{{0},{2}}]]]", "2 Pi"),
    ("RegionProduct[Disk[], 5]", Raises(T, "not a geometric region")),
    ("CSGRegion[\"Union\", {Ball[], Cuboid[]}]", "CSGRegion[\"Union\", {Ball[], Cuboid[]}]"),
    ("Volume[CSGRegion[\"Difference\", {Cuboid[{0,0,0},{2,2,2}], Cuboid[]}]]", "7"),
    ("Volume[CSGRegion[\"Intersection\", {Cuboid[], Cuboid[{1/2,1/2,1/2},{2,2,2}]}]]", "1/8"),
    ("CSGRegion[\"Foo\", {Ball[], Cuboid[]}]", Raises(V, "unknown CSG operation")),
    ("CSGRegion[\"Union\", {Ball[], 5}]", Raises(T, "not a geometric region")),
    ("CSGRegionQ[CSGRegion[\"Union\", {Ball[], Cuboid[]}]]", "True"),
    ("CSGRegionQ[Ball[]]", "False"),
    ("CSGRegionQ[5]", "False"),
    ("CSGRegionTree[CSGRegion[\"Union\", {Ball[], Cuboid[]}]]",
     "Tree[\"Union\", {Tree[Ball[], None], Tree[Cuboid[], None]}]"),
    ("CSGRegionTree[CSGRegion[\"Union\", {CSGRegion[\"Intersection\",{Ball[], Cuboid[]}], Cuboid[{2,2,2}]}]]",
     "Tree[\"Union\", {Tree[\"Intersection\", {Tree[Ball[], None], Tree[Cuboid[], None]}], "
     "Tree[Cuboid[{2, 2, 2}], None]}]"),
    ("CSGRegionTree[CSGRegion[\"Difference\", {Ball[], Cuboid[]}]]",
     "Tree[\"Difference\", {Tree[Ball[], None], Tree[Cuboid[], None]}]"),
    ("CSGRegionTree[w22r]", "CSGRegionTree[w22r]"),
    ("CSGRegionTree[Ball[]]", Raises(T, "not a CSGRegion")),
]

# --- derived regions ----------------------------------------------------------------------------------------------
DERIVED = [
    ("RegionDilation[Disk[], 1]", "Disk[{0, 0}, 2]"),
    ("RegionDilation[Point[{0,0}], 2]", "Disk[{0, 0}, 2]"),
    ("RegionDilation[Ball[{1,1,1},1/2], 1/2]", "Ball[{1, 1, 1}, 1]"),
    ("RegionDilation[Disk[], w22x]", "RegionDilation[Disk[], w22x]"),
    ("RegionDilation[Disk[], -1]", Raises(V, "non-negative")),
    ("RegionDilation[Rectangle[], 1]", Raises(N, "not supported")),
    ("RegionDilation[5, 1]", Raises(T, "not a geometric region")),
    ("RegionErosion[Disk[], 1/2]", "Disk[{0, 0}, 1/2]"),
    ("RegionErosion[Rectangle[{0,0},{2,2}], 1/2]", "Rectangle[{1/2, 1/2}, {3/2, 3/2}]"),
    ("RegionErosion[Disk[], 2]", "EmptyRegion[2]"),
    ("RegionErosion[Disk[], w22x]", "RegionErosion[Disk[], w22x]"),
    ("RegionErosion[Disk[], -1]", Raises(V, "non-negative")),
    ("RegionErosion[5, 1]", Raises(T, "not a geometric region")),
    ("RegionResize[Disk[], 2]", "Disk[{0, 0}, 1]"),
    ("RegionResize[Rectangle[], {2,3}]", "Rectangle[{0, 0}, {2, 3}]"),
    ("RegionResize[Rectangle[], {{0,4},{0,2}}]", "Rectangle[{0, 0}, {4, 2}]"),
    ("RegionResize[Rectangle[], {2,Automatic}]", "Rectangle[{0, 0}, {2, 2}]"),
    ("RegionResize[Rectangle[{0,0},{2,1}], {4}]", "Rectangle[{0, 0}, {4, 2}]"),
    ("RegionResize[Disk[], w22x]", "RegionResize[Disk[], w22x]"),
    ("RegionResize[Disk[], -1]", Raises(V, "must be positive")),
    ("RegionResize[5, 2]", Raises(T, "not a geometric region")),
    ("BoundingRegion[{{0,0},{2,0},{1,3}}]", "Rectangle[{0, 0}, {2, 3}]"),
    ("BoundingRegion[{{0,0},{2,0},{1,3}}, \"MinDisk\"]", "Disk[{1, 4/3}, 5/3]"),
    ("BoundingRegion[{{0,0},{2,0},{1,3},{1,1}}, \"MinConvexPolygon\"]", "Polygon[{{0, 0}, {2, 0}, {1, 3}}]"),
    ("BoundingRegion[{{0},{3},{1}}, \"MinInterval\"]", "Line[{{0}, {3}}]"),
    ("BoundingRegion[{{0,0,0},{1,2,3}}, \"MinCuboid\"]", "Cuboid[{0, 0, 0}, {1, 2, 3}]"),
    ("BoundingRegion[{{0,0},{1,0},{1,1},{0,1}}, \"MinBall\"]", "Disk[{1/2, 1/2}, Sqrt[2]/2]"),
    ("BoundingRegion[Disk[]]", "Rectangle[{-1, -1}, {1, 1}]"),
    ("BoundingRegion[w22p]", "BoundingRegion[w22p]"),
    ("BoundingRegion[{{0,0},{2,0},{1,3}}, \"Foo\"]", Raises(V, "unknown form")),
    ("BoundingRegion[{{0,0},{2,0},{1,3}}, \"FastOrientedRectangle\"]", Raises(N, "not supported")),
    ("BoundingRegion[5]", Raises(T, "not a geometric region")),
    ("CanonicalizeRegion[Ball[{0,0}]]", "Ball[{0, 0}]"),
    ("CanonicalizeRegion[Rectangle[{1,1},{0,0}]]", "Rectangle[{0, 0}, {1, 1}]"),
    ("CanonicalizeRegion[Disk[{1,2},0]]", "Point[{1, 2}]"),
    ("CanonicalizeRegion[Triangle[{{0,0},{1,1},{2,2}}]]", "Line[{{0, 0}, {2, 2}}]"),
    ("CanonicalizeRegion[Annulus[{0,0},{0,1}]]", "Disk[{0, 0}, 1]"),
    ("CanonicalizeRegion[w22r]", "CanonicalizeRegion[w22r]"),
    ("CanonicalizeRegion[5]", Raises(T, "not a geometric region")),
    ("RegionConvert[Disk[], \"Implicit\"]", "ImplicitRegion[FormalX^2 + FormalY^2 <= 1, {FormalX, FormalY}]"),
    ("RegionConvert[Rectangle[], \"Mesh\"]", "MeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Polygon[{1, 2, 3, 4}]}]"),
    ("RegionConvert[Rectangle[], \"BoundaryMesh\"]",
     "BoundaryMeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Line[{1, 2, 3, 4, 1}]}]"),
    ("RegionConvert[w22r, \"Implicit\"]", "RegionConvert[w22r, \"Implicit\"]"),
    ("RegionConvert[Disk[], \"Foo\"]", Raises(V, "unknown form")),
    ("RegionConvert[Disk[], \"Parametric\"]", Raises(N, "not supported")),
    ("RegionConvert[5, \"Implicit\"]", Raises(T, "not a geometric region")),
    ("ShellRegion[Ball[], 1/2]", "SphericalShell[{0, 0, 0}, {1/2, 1}]"),
    ("Volume[ShellRegion[Ball[], 1/2]]", "7 Pi/6"),
    ("ShellRegion[Disk[], 1/2]", "Annulus[{0, 0}, {1/2, 1}]"),
    ("ShellRegion[w22r, 1]", "ShellRegion[w22r, 1]"),
    ("ShellRegion[Ball[], -1]", Raises(V, "must be positive")),
    ("ShellRegion[Ball[]]", Raises(N, "default shell thickness")),
    ("ShellRegion[5, 1]", Raises(T, "not a geometric region")),
]

# --- curvatures ---------------------------------------------------------------------------------------------------
CURVATURES = [
    ("RegionGaussianCurvature[Sphere[{0,0,0},2], {2,0,0}]", "1/4"),
    ("RegionGaussianCurvature[Cylinder[], {1,0,0}]", "0"),
    ("RegionGaussianCurvature[ImplicitRegion[w22x^2+w22y^2+w22z^2==4,{w22x,w22y,w22z}], {0,0,2}]", "1/4"),
    ("RegionGaussianCurvature[Sphere[], {w22x,w22y,w22z}]", "RegionGaussianCurvature[Sphere[], {w22x, w22y, w22z}]"),
    ("RegionGaussianCurvature[Sphere[], {2,0,0}]", Raises(V, "not on the surface")),
    ("RegionGaussianCurvature[5, {1,0,0}]", Raises(T, "not a geometric region")),
    ("RegionGaussianCurvature[Ellipsoid[{0,0,0},{1,2,3}], {1,0,0}]", Raises(N, "not supported")),
    ("RegionMeanCurvature[Sphere[], {1,0,0}]", "1"),
    ("RegionMeanCurvature[Sphere[{0,0,0},2], {0,2,0}]", "1/2"),
    ("RegionMeanCurvature[ImplicitRegion[w22x^2+w22y^2+w22z^2==4,{w22x,w22y,w22z}], {0,0,2}]", "1/2"),
    ("RegionMeanCurvature[Cylinder[{{0,0,-1},{0,0,1}},2], {2,0,0}]", "1/4"),
    ("RegionMeanCurvature[Sphere[], {w22x,0,0}]", "RegionMeanCurvature[Sphere[], {w22x, 0, 0}]"),
    ("RegionMeanCurvature[Sphere[], {1,0}]", Raises(V, "must have 3 coordinates")),
    ("RegionMeanCurvature[Circle[{0,0},2], {2,0}]", Raises(N, "surfaces in 3 dimensions")),
    ("RegionMeanCurvature[5, {1,0,0}]", Raises(T, "not a geometric region")),
    ("RegionMaxCurvature[Cylinder[{{0,0,-1},{0,0,1}},2], {2,0,0}]", "1/2"),
    ("RegionMaxCurvature[Sphere[{0,0,0},3], {3,0,0}]", "1/3"),
    ("RegionMaxCurvature[Sphere[], {0,0,1}]", "1"),
    ("RegionMaxCurvature[Sphere[], {w22x,0,0}]", "RegionMaxCurvature[Sphere[], {w22x, 0, 0}]"),
    ("RegionMaxCurvature[Sphere[], {2,0,0}]", Raises(V, "not on the surface")),
    ("RegionMaxCurvature[5, {1,0,0}]", Raises(T, "not a geometric region")),
    ("RegionMaxCurvature[Ellipsoid[{0,0,0},{1,2,3}], {1,0,0}]", Raises(N, "not supported")),
    ("RegionMinCurvature[Cylinder[{{0,0,-1},{0,0,1}},2], {2,0,0}]", "0"),
    ("RegionMinCurvature[Sphere[{0,0,0},3], {3,0,0}]", "1/3"),
    ("RegionMinCurvature[Sphere[], {0,1,0}]", "1"),
    ("RegionMinCurvature[Sphere[], {w22x,0,0}]", "RegionMinCurvature[Sphere[], {w22x, 0, 0}]"),
    ("RegionMinCurvature[Sphere[], {2,0,0}]", Raises(V, "not on the surface")),
    ("RegionMinCurvature[5, {1,0,0}]", Raises(T, "not a geometric region")),
    ("RegionMinCurvature[Ellipsoid[{0,0,0},{1,2,3}], {1,0,0}]", Raises(N, "not supported")),
]

# --- inert option / annotation symbols ----------------------------------------------------------------------------
INERT = [(s, s) for s in ("SphericalRegion", "RegionSize", "MaxCellMeasure", "MeshCellHighlight", "MeshCellLabel",
                          "MeshCellMarker", "MeshCellShapeFunction", "MeshCellStyle", "MeshQualityGoal",
                          "MeshRefinementFunction", "NormalsFunction", "MeshCellMeasure", "MeshCellCentroid")]
INERT += [(f"{s} -> 2", f"{s} -> 2") for s, _ in INERT]
INERT += [(f"{s}[w22q]", f"{s}[w22q]") for s, _ in INERT[:13]]
INERT += [
    ("DiscretizeRegion[Rectangle[], MaxCellMeasure -> 0.1]", "MeshRegion[{{0,0},{1,0},{1,1},{0,1}}, {Polygon[{1,2,3,4}]}]"),
    ("MeshRegion[{{0,0},{1,0},{0,1}}, {Triangle[{1,2,3}]}, MeshCellStyle -> {1 -> Red}]",
     "MeshRegion[{{0,0},{1,0},{0,1}}, {Triangle[{1,2,3}]}, MeshCellStyle -> {1 -> Red}]"),
]

# --- meshes -------------------------------------------------------------------------------------------------------
MESHES = [
    ("%SQ%", is_head("MeshRegion", 2)),
    ("Area[%SQ%]", "1"),
    ("Volume[%TET%]", "1/6"),
    ("MeshRegion[w22p, {Triangle[{1,2,3}]}]", "MeshRegion[w22p, {Triangle[{1, 2, 3}]}]"),
    ("MeshRegion[\"a\", {Triangle[{1,2,3}]}]", Raises(T, "coordinates must be a list")),
    ("MeshRegion[{{0,0},{1,0},{0,1}}, {Triangle[{1,2,4}]}]", Raises(V, "out of range")),
    ("MeshRegion[{{0,0},{1,0},{0,1}}, {Foo[{1,2,3}]}]", Raises(T, "not a mesh cell")),
    ("%HOLED%", is_head("BoundaryMeshRegion", 3)),
    ("Area[%BTRI%]", "1/2"),
    ("BoundaryMeshRegion[w22p, {Line[{1,2,3,1}]}]", "BoundaryMeshRegion[w22p, {Line[{1, 2, 3, 1}]}]"),
    ("BoundaryMeshRegion[\"a\", {Line[{1,2,3,1}]}]", Raises(T, "coordinates must be a list")),
    ("BoundaryMeshRegion[{{0,0},{1,0},{1,1}}, {Line[{1,2,3}]}]", Raises(V, "not closed")),
    ("MeshRegionQ[%SQ%]", "True"),
    ("MeshRegionQ[Disk[]]", "False"),
    ("MeshRegionQ[5]", "False"),
    ("MeshRegionQ[%HOLED%]", "False"),
    ("BoundaryMeshRegionQ[%HOLED%]", "True"),
    ("BoundaryMeshRegionQ[%TRI%]", "False"),
    ("BoundaryMeshRegionQ[w22r]", "False"),
    ("MeshCoordinates[%SQ%]", "{{0, 0}, {1, 0}, {1, 1}, {0, 1}}"),
    ("MeshCoordinates[%BTRI%]", "{{0, 0}, {1, 0}, {1, 1}}"),
    ("MeshCoordinates[RegionConvert[Triangle[], \"Mesh\"]]", "{{0, 0}, {1, 0}, {0, 1}}"),
    ("MeshCoordinates[w22r]", "MeshCoordinates[w22r]"),
    ("MeshCoordinates[Disk[]]", Raises(T, "not a MeshRegion")),
    ("MeshCells[%SQ%, 1]", "{Line[{1, 2}], Line[{2, 3}], Line[{3, 1}], Line[{3, 4}], Line[{4, 1}]}"),
    ("MeshCells[%SQ%, {2,1}]", "Triangle[{1, 2, 3}]"),
    ("MeshCells[%TRI%, 0]", "{Point[{1}], Point[{2}], Point[{3}]}"),
    ("MeshCells[%TRI%, {1,All}]", "{Line[{1, 2}], Line[{2, 3}], Line[{3, 1}]}"),
    ("MeshCells[w22r, 1]", "MeshCells[w22r, 1]"),
    ("MeshCells[%TRI%, 5]", Raises(V, "invalid cell dimension")),
    ("MeshCells[%TRI%, {2,5}]", Raises(V, "out of range")),
    ("MeshCells[Disk[], 1]", Raises(T, "not a MeshRegion")),
    ("MeshPrimitives[%SQ%, 2]", "{Polygon[{{0, 0}, {1, 0}, {1, 1}}], Polygon[{{0, 0}, {1, 1}, {0, 1}}]}"),
    ("MeshPrimitives[%TRI%, 1]", "{Line[{{0, 0}, {1, 0}}], Line[{{1, 0}, {0, 1}}], Line[{{0, 1}, {0, 0}}]}"),
    ("MeshPrimitives[%TRI%, 0]", "{Point[{0, 0}], Point[{1, 0}], Point[{0, 1}]}"),
    ("MeshPrimitives[w22r, 1]", "MeshPrimitives[w22r, 1]"),
    ("MeshPrimitives[%TRI%, 7]", Raises(V, "invalid cell dimension")),
    ("MeshPrimitives[Disk[], 1]", Raises(T, "not a MeshRegion")),
    ("MeshCellCount[%SQ%]", "{4, 5, 2}"),
    ("MeshCellCount[%TET%]", "{4, 6, 4, 1}"),
    ("MeshCellCount[%TRI%, 1]", "3"),
    ("MeshCellCount[w22r]", "MeshCellCount[w22r]"),
    ("MeshCellCount[5]", Raises(T, "not a MeshRegion")),
    ("MeshCellIndex[%SQ%, Line[{1,3}]]", "{1, 3}"),
    ("MeshCellIndex[%TRI%, Line[{3,2}]]", "{1, 2}"),
    ("MeshCellIndex[%TRI%, {Point[{1}], Triangle[{1,2,3}]}]", "{{0, 1}, {2, 1}}"),
    ("MeshCellIndex[%TRI%, 1]", "{{1, 1}, {1, 2}, {1, 3}}"),
    ("MeshCellIndex[w22r, Line[{1,2}]]", "MeshCellIndex[w22r, Line[{1, 2}]]"),
    ("MeshCellIndex[%TRI%, Line[{1,5}]]", Raises(V, "out of range")),
    ("MeshCellIndex[Disk[], 1]", Raises(T, "not a MeshRegion")),
    ("AdjacentMeshCells[%SQ%, {2,1}, 2]", "{{2, 2}}"),
    ("AdjacentMeshCells[%SQ%, {0,1}, 1]", "{{1, 1}, {1, 3}, {1, 5}}"),
    ("AdjacentMeshCells[%TRI%, {2,1}, 0]", "{{0, 1}, {0, 2}, {0, 3}}"),
    ("AdjacentMeshCells[w22r, {2,1}, 0]", "AdjacentMeshCells[w22r, {2, 1}, 0]"),
    ("AdjacentMeshCells[%TRI%, {2,7}, 0]", Raises(V, "out of range")),
    ("AdjacentMeshCells[Disk[], {2,1}, 0]", Raises(T, "not a MeshRegion")),
    ("NearestMeshCells[%SQ%, {0.9, 0.1}]", "{0, {2}}"),
    ("NearestMeshCells[{%SQ%,1}, {2, 1/2}]", "{1, {2}}"),
    ("NearestMeshCells[{%TRI%,2}, {3,3}]", "{2, {1}}"),
    ("NearestMeshCells[w22r, {0,0}]", "NearestMeshCells[w22r, {0, 0}]"),
    ("NearestMeshCells[%TRI%, {1,2,3}]", Raises(V, "must have 2 coordinates")),
    ("NearestMeshCells[Disk[], {0,0}]", Raises(T, "not a MeshRegion")),
    ("HighlightMesh[%TRI%, {2,1}]", "HighlightMesh[%TRI%, {2, 1}]"),
    ("HighlightMesh[%TRI%, Style[{1,All},Red]]", "HighlightMesh[%TRI%, Style[{1, All}, Red]]"),
    ("HighlightMesh[w22r, 1]", "HighlightMesh[w22r, 1]"),
    ("HighlightMesh[5, 1]", Raises(T, "not a MeshRegion")),
    ("ConnectedMeshComponents[MeshRegion[{{0,0},{1,0},{0,1},{5,5},{6,5}}, {Triangle[{1,2,3}], Line[{4,5}]}]]",
     "{MeshRegion[{{0, 0}, {1, 0}, {0, 1}}, {Triangle[{1, 2, 3}]}], MeshRegion[{{5, 5}, {6, 5}}, {Line[{1, 2}]}]}"),
    ("Length[ConnectedMeshComponents[%SQ%]]", "1"),
    ("Length[ConnectedMeshComponents[BoundaryMeshRegion[{{0,0},{1,0},{1,1},{5,5},{6,5},{6,6}}, "
     "{Line[{1,2,3,1}]}, {Line[{4,5,6,4}]}]]]", "2"),
    ("ConnectedMeshComponents[w22r]", "ConnectedMeshComponents[w22r]"),
    ("ConnectedMeshComponents[Disk[]]", Raises(T, "not a MeshRegion")),
    ("DimensionalMeshComponents[MeshRegion[{{0,0},{1,0},{0,1},{5,5},{6,5},{9,9}}, "
     "{Triangle[{1,2,3}], Line[{4,5}], Point[{6}]}]]",
     "{MeshRegion[{{9, 9}}, {Point[{1}]}], MeshRegion[{{5, 5}, {6, 5}}, {Line[{1, 2}]}], "
     "MeshRegion[{{0, 0}, {1, 0}, {0, 1}}, {Triangle[{1, 2, 3}]}]}"),
    ("DimensionalMeshComponents[%TRI%]", "{EmptyRegion[2], EmptyRegion[2], %TRI%}"),
    ("Length[DimensionalMeshComponents[%TET%]]", "4"),
    ("DimensionalMeshComponents[w22r]", "DimensionalMeshComponents[w22r]"),
    ("DimensionalMeshComponents[5]", Raises(T, "not a MeshRegion")),
    ("BoundaryMesh[%SQ%]", "BoundaryMeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Line[{1, 2, 3, 4, 1}]}]"),
    ("BoundaryMesh[%TET%]", "%TETB%"),
    ("Perimeter[BoundaryMesh[%SQ%]]", "4"),
    ("BoundaryMesh[%BTRI%]", "%BTRI%"),
    ("BoundaryMesh[w22r]", "BoundaryMesh[w22r]"),
    ("BoundaryMesh[Disk[]]", Raises(T, "not a MeshRegion")),
    ("BoundaryMesh[MeshRegion[{{0,0},{1,0},{2,0}}, {Line[{1,2}],Line[{2,3}]}]]", Raises(N, "lower-dimensional")),
    ("EulerCharacteristic[Cube[]]", "2"),
    ("EulerCharacteristic[%SQ%]", "1"),
    ("EulerCharacteristic[%HOLED%]", "0"),
    ("EulerCharacteristic[Polyhedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}},{{1,3,2},{1,2,4},{2,3,4},{1,4,3}}]]", "2"),
    ("EulerCharacteristic[w22r]", "EulerCharacteristic[w22r]"),
    ("EulerCharacteristic[5]", Raises(T, "expected a polyhedron")),
    ("DelaunayMesh[{{0},{3},{1}}]", "MeshRegion[{{0}, {3}, {1}}, {Line[{1, 3}], Line[{3, 2}]}]"),
    ("DelaunayMesh[{{0,0},{1,1},{2,2}}]", "MeshRegion[{{0, 0}, {1, 1}, {2, 2}}, {Line[{1, 2}], Line[{2, 3}]}]"),
    ("DelaunayMesh[w22p]", "DelaunayMesh[w22p]"),
    ("DelaunayMesh[\"a\"]", Raises(T, "points must be a list")),
    ("VoronoiMesh[{{0,0},{1,0},{0,1}}, {{-1,2},{-1,2}}]",
     "MeshRegion[{{-1, -1}, {1/2, -1}, {1/2, 1/2}, {-1, 1/2}, {2, -1}, {2, 2}, {-1, 2}}, "
     "{Polygon[{1, 2, 3, 4}], Polygon[{2, 5, 6, 3}], Polygon[{6, 7, 4, 3}]}]"),
    ("VoronoiMesh[{{0},{2}}, {{-1,3}}]", "MeshRegion[{{-1}, {1}, {3}}, {Line[{1, 2}], Line[{2, 3}]}]"),
    ("Area[VoronoiMesh[{{0,0},{1,0},{0,1},{1,1}}, {{-1,2},{-1,2}}]]", "9"),
    ("VoronoiMesh[w22p]", "VoronoiMesh[w22p]"),
    ("VoronoiMesh[\"a\"]", Raises(T, "points must be a list")),
    ("VoronoiMesh[{{0,0},{1,0}}, {{0,1}}]", Raises(V, "with 2 ranges")),
    ("VoronoiMesh[{{0,0},{1,0}}]", Raises(N, "automatic range")),
    ("ConvexHullMesh[{{0,0},{2,0},{1,1},{2,2},{0,2}}]",
     "BoundaryMeshRegion[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}, {Line[{1, 2, 3, 4, 1}]}]"),
    ("Area[ConvexHullMesh[{{0,0},{2,0},{1,1},{2,2},{0,2}}]]", "4"),
    ("Volume[ConvexHullMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1/10,1/10,1/10}}]]", "1/6"),
    ("ConvexHullMesh[{{0,0},{1,1}}]", "MeshRegion[{{0, 0}, {1, 1}}, {Line[{1, 2}]}]"),
    ("ConvexHullMesh[w22p]", "ConvexHullMesh[w22p]"),
    ("ConvexHullMesh[\"a\"]", Raises(T, "not a geometric region")),
    ("ConvexHullRegion[{{0,0},{2,0},{1,1},{2,2},{0,2}}]", "Polygon[{{0, 0}, {2, 0}, {2, 2}, {0, 2}}]"),
    ("ConvexHullRegion[{{0,0},{1,1},{2,2}}]", "Line[{{0, 0}, {2, 2}}]"),
    ("ConvexHullRegion[Disk[]]", "Disk[]"),
    ("ConvexHullRegion[w22p]", "ConvexHullRegion[w22p]"),
    ("ConvexHullRegion[\"a\"]", Raises(T, "not a geometric region")),
    ("ConvexHullRegion[Annulus[]]", Raises(N, "not supported")),
    ("ArrayMesh[{1,0,1}]", "MeshRegion[{{0}, {1}, {2}, {3}}, {Line[{1, 2}], Line[{3, 4}]}]"),
    ("Area[ArrayMesh[{{1,0},{1,1}}]]", "3"),
    ("ArrayMesh[{{0,0}}]", "EmptyRegion[2]"),
    ("Area[ArrayMesh[{{1,1}}, DataRange->{{0,2},{0,1}}]]", "2"),
    ("ArrayMesh[w22p]", "ArrayMesh[w22p]"),
    ("ArrayMesh[\"a\"]", Raises(T, "must be an array")),
    ("ArrayMesh[{{{1}}}]", Raises(N, "rank-3")),
    ("DiscretizeRegion[Rectangle[]]", "MeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Polygon[{1, 2, 3, 4}]}]"),
    ("DiscretizeRegion[Triangle[]]", "MeshRegion[{{0, 0}, {1, 0}, {0, 1}}, {Triangle[{1, 2, 3}]}]"),
    ("Volume[DiscretizeRegion[Cuboid[]]]", "1"),
    ("Area[DiscretizeRegion[%HP%, {{0,2},{0,2}}]]", "4"),
    ("DiscretizeRegion[Point[{{0,0},{1,1}}]]", "MeshRegion[{{0, 0}, {1, 1}}, {Point[{1}], Point[{2}]}]"),
    ("DiscretizeRegion[w22r]", "DiscretizeRegion[w22r]"),
    ("DiscretizeRegion[5]", Raises(T, "not a geometric region")),
    ("DiscretizeRegion[%HP%]", Raises(V, "unbounded")),
    ("DiscretizeRegion[Disk[]]", Raises(N, "curved regions")),
    ("BoundaryDiscretizeRegion[Rectangle[]]",
     "BoundaryMeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Line[{1, 2, 3, 4, 1}]}]"),
    ("BoundaryDiscretizeRegion[Triangle[]]", "BoundaryMeshRegion[{{0, 0}, {1, 0}, {0, 1}}, {Line[{1, 2, 3, 1}]}]"),
    ("Volume[BoundaryDiscretizeRegion[Cuboid[{0,0,0},{1,2,3}]]]", "6"),
    ("BoundaryDiscretizeRegion[w22r]", "BoundaryDiscretizeRegion[w22r]"),
    ("BoundaryDiscretizeRegion[5]", Raises(T, "not a geometric region")),
    ("BoundaryDiscretizeRegion[Disk[]]", Raises(N, "curved regions")),
    ("DiscretizeGraphics[Graphics[{Red, Rectangle[], Triangle[{{2,0},{3,0},{2,1}}]}]]",
     "MeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {2, 0}, {3, 0}, {2, 1}}, {Polygon[{1, 2, 3, 4}], "
     "Triangle[{5, 6, 7}]}]"),
    ("Area[DiscretizeGraphics[Graphics[{Red, Rectangle[], Triangle[{{2,0},{3,0},{2,1}}]}]]]", "3/2"),
    ("Volume[DiscretizeGraphics[Graphics3D[{Cuboid[]}]]]", "1"),
    ("DiscretizeGraphics[w22g]", "DiscretizeGraphics[w22g]"),
    ("DiscretizeGraphics[5]", Raises(T, "expected Graphics")),
    ("DiscretizeGraphics[Graphics[{Disk[]}]]", Raises(N, "curved regions")),
    ("BoundaryDiscretizeGraphics[Graphics[{Rectangle[]}]]",
     "BoundaryMeshRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}, {Line[{1, 2, 3, 4, 1}]}]"),
    ("Area[BoundaryDiscretizeGraphics[Graphics[{Rectangle[{0,0},{2,3}]}]]]", "6"),
    ("Volume[BoundaryDiscretizeGraphics[Graphics3D[{Cuboid[]}]]]", "1"),
    ("BoundaryDiscretizeGraphics[w22g]", "BoundaryDiscretizeGraphics[w22g]"),
    ("BoundaryDiscretizeGraphics[5]", Raises(T, "expected Graphics")),
    ("BoundaryDiscretizeGraphics[Graphics[{Disk[]}]]", Raises(N, "curved regions")),
]
SCIPY_MESHES = [
    ("Area[DelaunayMesh[%PTS%]]", "1"),
    ("DelaunayMesh[%PTS%]", "MeshRegion[%PTS%, {Triangle[{2, 5, 1}], Triangle[{4, 5, 2}], Triangle[{5, 3, 1}], "
     "Triangle[{5, 4, 3}]}]"),
    ("Volume[DelaunayMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}}]]", "1/2"),
    ("MeshCellCount[DelaunayMesh[{{0,0,0},{1,0,0},{0,1,0},{0,0,1},{1,1,1}}], 3]", "3"),
]
NEEDS_SCIPY = ("DelaunayMesh[%PTS%", "DelaunayMesh[{{0,0,0}")

# --- geometric transforms -----------------------------------------------------------------------------------------
TRANSFORMS = [
    ("TranslationTransform[{1,2}]", "TransformationFunction[{{1, 0, 1}, {0, 1, 2}, {0, 0, 1}}]"),
    ("TranslationTransform[{1,2}][{3,4}]", "{4, 6}"),
    ("TranslationTransform[{w22a,w22b}][{0,0}]", "{w22a, w22b}"),
    ("TranslationTransform[w22v]", "TranslationTransform[w22v]"),
    ("TranslationTransform[5]", Raises(T, "must be a list")),
    ("RotationTransform[Pi/2][{1,0}]", "{0, 1}"),
    ("RotationTransform[Pi/2, {0,0,1}][{1,0,0}]", "{0, 1, 0}"),
    ("RotationTransform[Pi/2, {1,1}][{2,1}]", "{1, 2}"),
    ("RotationTransform[{{1,0,0},{0,1,0}}][{1,0,0}]", "{0, 1, 0}"),
    ("RotationTransform[w22t]", "TransformationFunction[{{Cos[w22t], -Sin[w22t], 0}, {Sin[w22t], Cos[w22t], 0}, "
     "{0, 0, 1}}]"),
    ("RotationTransform[Pi/2, {0,0,0}]", Raises(V, "must be nonzero")),
    ("RotationTransform[\"a\"]", Raises(T, "angle must be")),
    ("ScalingTransform[{2,3}][{1,1}]", "{2, 3}"),
    ("ScalingTransform[2, {1,0}][{1,1}]", "{2, 1}"),
    ("ScalingTransform[{2,2}, {1,1}][{2,2}]", "{3, 3}"),
    ("ScalingTransform[w22s]", "ScalingTransform[w22s]"),
    ("ScalingTransform[\"a\"]", Raises(T, "scale factor")),
    ("ScalingTransform[2]", Raises(V, "expected ScalingTransform")),
    ("ReflectionTransform[{1,0}][{3,4}]", "{-3, 4}"),
    ("ReflectionTransform[{0,1},{0,1}][{3,4}]", "{3, -2}"),
    ("ReflectionTransform[{1,1}][{1,0}]", "{0, -1}"),
    ("ReflectionTransform[w22v]", "ReflectionTransform[w22v]"),
    ("ReflectionTransform[{0,0}]", Raises(V, "must be nonzero")),
    ("ReflectionTransform[\"a\"]", Raises(T, "normal vector")),
    ("ShearingTransform[Pi/4, {1,0}, {0,1}][{0,1}]", "{1, 1}"),
    ("ShearingTransform[w22t, {1,0}, {0,1}]", "TransformationFunction[{{1, Tan[w22t], 0}, {0, 1, 0}, {0, 0, 1}}]"),
    ("ShearingTransform[Pi/4, {1,0}, {0,1}][{2,0}]", "{2, 0}"),
    ("ShearingTransform[\"a\", {1,0}, {0,1}]", Raises(T, "angle must be")),
    ("ShearingTransform[Pi/4, {1,0}]", Raises(V, "expected 3")),
    ("ShearingTransform[Pi/4, {1,0}, {1,0}]", Raises(N, "orthogonal")),
    ("RescalingTransform[{{0,1},{0,1}}, {{0,2},{0,4}}][{1/2,1/2}]", "{1, 2}"),
    ("RescalingTransform[{{0,1}}][{1/2}]", "{1/2}"),
    ("RescalingTransform[{{2,4}}][{3}]", "{1/2}"),
    ("RescalingTransform[w22b]", "RescalingTransform[w22b]"),
    ("RescalingTransform[{{0,0}}]", Raises(V, "degenerate")),
    ("RescalingTransform[\"a\"]", Raises(T, "source ranges")),
    ("AffineTransform[{{{1,2},{3,4}},{5,6}}][{1,1}]", "{8, 13}"),
    ("AffineTransform[{{1,2},{3,4}}]", "TransformationFunction[{{1, 2, 0}, {3, 4, 0}, {0, 0, 1}}]"),
    ("AffineTransform[{{1,0},{0,1}}][{w22a,w22b}]", "{w22a, w22b}"),
    ("AffineTransform[w22m]", "AffineTransform[w22m]"),
    ("AffineTransform[\"a\"]", Raises(T, "must be a matrix")),
    ("AffineTransform[{{1,2},{3}}]", Raises(V, "rectangular")),
    ("LinearFractionalTransform[{{{1,0},{0,1}},{0,0},{1,1},0}][{1,1}]", "{1/2, 1/2}"),
    ("LinearFractionalTransform[{{1,2},{3,4}}]", "TransformationFunction[{{1, 2}, {3, 4}}]"),
    ("LinearFractionalTransform[{{1,0},{1,1}}][{1}]", "{1/2}"),
    ("LinearFractionalTransform[w22m]", "LinearFractionalTransform[w22m]"),
    ("LinearFractionalTransform[\"a\"]", Raises(T, "must be a matrix")),
    ("LinearFractionalTransform[{{{1,0},{0,1}},{0,0},{0,0},0}][{1,1}]", Raises(V, "mapped to infinity")),
    ("TransformationFunction[{{1,0,1},{0,1,2},{0,0,1}}][{1,1}]", "{2, 3}"),
    ("Composition[TranslationTransform[{1,0}], ScalingTransform[{2,2}]][{1,1}]", "{3, 2}"),
    ("TransformationFunction[{{2,0,0},{0,2,0},{0,0,1}}][{{1,1},{2,3}}]", "{{2, 2}, {4, 6}}"),
    ("TransformationFunction[w22m]", "TransformationFunction[w22m]"),
    ("TransformationFunction[{{1,0,1},{0,1,2},{0,0,1}}][w22p]", "TransformationFunction[{{1,0,1},{0,1,2},{0,0,1}}][w22p]"),
    ("TransformationFunction[\"a\"]", Raises(T, "must be a matrix")),
    ("TransformationFunction[{{1,2},{3,4},{5,6}}]", Raises(V, "must be square")),
    ("EulerMatrix[{Pi/2,0,0}]", "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"),
    ("EulerMatrix[{Pi/2,0,0},{1,2,3}]", "{{1, 0, 0}, {0, 0, -1}, {0, 1, 0}}"),
    ("EulerMatrix[{w22a,w22b,w22c}]", "RotationMatrix[w22a,{0,0,1}].RotationMatrix[w22b,{0,1,0}]."
     "RotationMatrix[w22c,{0,0,1}]"),
    ("EulerMatrix[w22e]", "EulerMatrix[w22e]"),
    ("EulerMatrix[\"a\"]", Raises(T, "angle list")),
    ("EulerMatrix[{1,2}]", Raises(V, "3 components")),
    ("EulerAngles[RotationMatrix[Pi/3,{0,0,1}]]", "{Pi/3, 0, 0}"),
    ("EulerAngles[N[EulerMatrix[{0.5,0.3,0.2}]]]", NumList([0.5, 0.3, 0.2])),
    ("EulerAngles[N[EulerMatrix[{0.5,0.3,0.2},{1,2,1}]],{1,2,1}]", NumList([0.5, 0.3, 0.2])),
    ("EulerAngles[w22m]", "EulerAngles[w22m]"),
    ("EulerAngles[\"a\"]", Raises(T, "must be a matrix")),
    ("EulerAngles[{{2,0,0},{0,1,0},{0,0,1}}]", Raises(V, "not a rotation matrix")),
    ("RollPitchYawMatrix[{Pi/2,0,0}]", "{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}"),
    ("RollPitchYawMatrix[{w22a,w22b,w22c},{1,2,3}]", "RotationMatrix[w22c,{0,0,1}].RotationMatrix[w22b,{0,1,0}]."
     "RotationMatrix[w22a,{1,0,0}]"),
    ("RollPitchYawMatrix[{0,Pi/2,0}]", "RotationMatrix[Pi/2,{0,1,0}]"),
    ("RollPitchYawMatrix[w22e]", "RollPitchYawMatrix[w22e]"),
    ("RollPitchYawMatrix[\"a\"]", Raises(T, "angle list")),
    ("RollPitchYawMatrix[{1,2}]", Raises(V, "3 components")),
    ("RollPitchYawAngles[RollPitchYawMatrix[{1/2,1/3,1/4}]]", "{1/2, 1/3, 1/4}"),
    ("RollPitchYawAngles[N[RollPitchYawMatrix[{0.1,0.2,0.3}]]]", NumList([0.1, 0.2, 0.3])),
    ("RollPitchYawAngles[N[RollPitchYawMatrix[{0.1,0.2,0.3},{1,2,3}]],{1,2,3}]", NumList([0.1, 0.2, 0.3])),
    ("RollPitchYawAngles[w22m]", "RollPitchYawAngles[w22m]"),
    ("RollPitchYawAngles[\"a\"]", Raises(T, "must be a matrix")),
    ("RollPitchYawAngles[{{2,0,0},{0,1,0},{0,0,1}}]", Raises(V, "not a rotation matrix")),
    ("GeometricTransformation[Disk[], TranslationTransform[{1,0}]]",
     "GeometricTransformation[Disk[], TransformationFunction[{{1, 0, 1}, {0, 1, 0}, {0, 0, 1}}]]"),
    ("GeometricTransformation[Disk[], {{1,0},{0,1}}]", "GeometricTransformation[Disk[], {{1, 0}, {0, 1}}]"),
    ("GeometricTransformation[Disk[], {{{1,0},{0,1}},{1,1}}]",
     "GeometricTransformation[Disk[], {{{1, 0}, {0, 1}}, {1, 1}}]"),
    ("GeometricTransformation[Disk[], \"a\"]", Raises(T, "not a transformation")),
    ("GeometricTransformation[Disk[]]", Raises(V, "expected 2")),
]

# --- geodesy and coordinates (GeographicLib reference values, WGS84) ----------------------------------------------
GEO = [
    ("GeoDistance[{0,0},{0,1}]", QNum(111.31949079327357, "Kilometers")),
    ("GeoDistance[{0,0},{90,0}]", QNum(10001.965729229534, "Kilometers")),
    ("GeoDistance[{51.5,0},{48.85,2.35}]", QNum(339.1740901525508, "Kilometers")),
    ("GeoDistance[{40.1,-88.2},{51.5,0}]", QNum(6548.19954472546, "Kilometers")),
    ("GeoDistance[{-33.9,151.2},{35.7,139.7}]", QNum(7797.236865157441, "Kilometers")),
    ("GeoDistance[GeoPosition[{0,0}], GeoPosition[{0,1}]]", QNum(111.31949079327357, "Kilometers")),
    ("GeoDistance[{{0,0},{0,1},{0,2}}]", QNum(2 * 111.31949079327357, "Kilometers")),
    ("GeoDistance[w22p, {0,0}]", "GeoDistance[w22p, {0, 0}]"),
    ("GeoDistance[{100,0},{0,0}]", Raises(V, "must lie in")),
    ("GeoDistance[{0,0},{0,180}]", Raises(N, "antipodal")),
    ("GeoDistance[5, {0,0}]", Raises(T)),
    ("GeoDirection[{0,0},{0,1}]", QNum(90.0, "AngularDegrees")),
    ("GeoDirection[{0,0},{1,0}]", QNum(0.0, "AngularDegrees")),
    ("GeoDirection[{51.5,0},{48.85,2.35}]", QNum(149.42970597571184, "AngularDegrees", 1e-8)),
    ("GeoDirection[{40.1,-88.2},{51.5,0}]", QNum(46.73503984801909, "AngularDegrees", 1e-8)),
    ("GeoDirection[w22p, {0,0}]", "GeoDirection[w22p, {0, 0}]"),
    ("GeoDirection[\"a\", {0,0}]", Raises(V, "DMS string")),
    ("GeoDirection[5, {0,0}]", Raises(T)),
    ("GeoDisplacement[{0,0},{0,1}]", Pred(lambda g: g.head == "GeoDisplacement" and abs(float(g.args[0].args[0].args[0])
                                                                                        - 111319.49079327357) < 1e-4,
                                          "GeoDisplacement[{~111319.49 m, ...}]")),
    ("GeoDisplacement[{51.5,0},{48.85,2.35}]",
     Pred(lambda g: abs(float(g.args[0].args[1].args[0]) - 149.42970597571184) < 1e-7, "azimuth ~149.4297")),
    ("GeoDisplacement[{0,0},{1,0}]", Pred(lambda g: abs(float(g.args[0].args[1].args[0])) < 1e-9, "azimuth 0")),
    ("GeoDisplacement[w22p,{0,0}]", "GeoDisplacement[w22p, {0, 0}]"),
    ("GeoDisplacement[\"a\",{0,0}]", Raises(V, "DMS string")),
    ("GeoDisplacement[5,{0,0}]", Raises(T)),
    ("GeoDestination[{10,20}, {500000, 45}]", NumList([13.178676456668079, 23.259772528898296], 1e-9)),
    ("GeoDestination[{10,20}, GeoDisplacement[{500000, 45}]]", NumList([13.178676456668079, 23.259772528898296], 1e-9)),
    ("GeoDestination[{0,0}, {Quantity[111319.49079327357,\"Meters\"], 90}]", NumList([0.0, 1.0], 1e-9)),
    ("GeoDestination[w22p, {1000,0}]", "GeoDestination[w22p, {1000, 0}]"),
    ("GeoDestination[{0,0}, {-1, 0}]", Raises(V, "non-negative")),
    ("GeoDestination[5, {1000,0}]", Raises(T)),
    ("Latitude[{40.1,-88.2}]", "Quantity[40.1, \"AngularDegrees\"]"),
    ("Latitude[GeoPosition[{40.1,-88.2}]]", "Quantity[40.1, \"AngularDegrees\"]"),
    ("Latitude[{-30,10}]", "Quantity[-30, \"AngularDegrees\"]"),
    ("Latitude[w22p]", "Latitude[w22p]"),
    ("Latitude[{100,0}]", Raises(V, "must lie in")),
    ("Latitude[{1,2,3,4}]", Raises(T, "location must be")),
    ("Longitude[{40.1,-88.2}]", "Quantity[-88.2, \"AngularDegrees\"]"),
    ("Longitude[GeoPosition[{1,2}]]", "Quantity[2, \"AngularDegrees\"]"),
    ("Longitude[{0,170}]", "Quantity[170, \"AngularDegrees\"]"),
    ("Longitude[w22p]", "Longitude[w22p]"),
    ("Longitude[{1,2,3,4}]", Raises(T, "location must be")),
    ("Longitude[\"a\"]", Raises(V, "DMS string")),
    ("LatitudeLongitude[{40.1,-88.2}]", "{Quantity[40.1, \"AngularDegrees\"], Quantity[-88.2, \"AngularDegrees\"]}"),
    ("LatitudeLongitude[GeoPosition[{1,2}]]", "{Quantity[1, \"AngularDegrees\"], Quantity[2, \"AngularDegrees\"]}"),
    ("LatitudeLongitude[{0,0}]", "{Quantity[0, \"AngularDegrees\"], Quantity[0, \"AngularDegrees\"]}"),
    ("LatitudeLongitude[w22p]", "LatitudeLongitude[w22p]"),
    ("LatitudeLongitude[\"a\"]", Raises(V, "DMS string")),
    ("LatitudeLongitude[{1,2,3,4}]", Raises(T, "location must be")),
    ("FromDMS[{20,32,15.828}]", Num(20.53773)),
    ("FromDMS[{20,30}]", "41/2"),
    ("FromDMS[{-20,30,0}]", "-39/2"),
    ("FromDMS[{{1,30,0},{2,15,0}}]", "{3/2, 9/4}"),
    ("FromDMS[\"20°32'15.828\\\"\"]", Num(20.53773)),
    ("FromDMS[w22d]", "FromDMS[w22d]"),
    ("FromDMS[\"abc\"]", Raises(V, "not a degrees-minutes-seconds string")),
    ("FromDMS[{1,2,3,4}]", Raises(V, "DMS list must be")),
    ("FromDMS[Disk[]]", Raises(T)),
    ("DMSList[41/2]", "{20, 30, 0}"),
    ("DMSList[20.5375]", NumList([20, 32, 15.0])),
    ("DMSList[-20.5]", NumList([-20, -30, 0.0])),
    ("DMSList[{40.097, -88.246}]", NumList([[40, 5, 49.2], [-88, -14, -45.6]], 1e-8)),
    ("DMSList[w22d]", "DMSList[w22d]"),
    ("DMSList[\"a\"]", Raises(V, "not a degrees-minutes-seconds string")),
    ("DMSList[Disk[]]", Raises(T)),
    ("DMSString[20.537730]", "\"20°32'15.828\\\"\""),
    ("DMSString[{40.097, -88.246}]", "\"40°5'49.200\\\"N 88°14'45.600\\\"W\""),
    ("DMSString[20.5, 2]", "\"20°30'0.00\\\"\""),
    ("DMSString[{1,2,3}]", "\"1°2'3\\\"\""),
    ("DMSString[w22d]", "DMSString[w22d]"),
    ("DMSString[20.5, -1]", Raises(V, "non-negative integer")),
    ("DMSString[Disk[]]", Raises(T)),
    ("CoordinateBounds[{{1,2},{3,0},{2,5}}]", "{{1, 3}, {0, 5}}"),
    ("CoordinateBounds[{{1,2},{3,0},{2,5}}, 1]", "{{0, 4}, {-1, 6}}"),
    ("CoordinateBounds[{{1,2},{3,0}}, Scaled[1/2]]", "{{0, 4}, {-1, 3}}"),
    ("CoordinateBounds[{1,5,3}]", "{{1, 5}}"),
    ("CoordinateBounds[w22p]", "CoordinateBounds[w22p]"),
    ("CoordinateBounds[\"a\"]", Raises(T, "expected an array")),
    ("CoordinateBounds[{{1,2},{3}}]", Raises(V, "same length")),
    ("CoordinateBoundsArray[{{0,2},{0,1}}]", "{{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}}"),
    ("CoordinateBoundsArray[{{0,1}}, 1/2]", "{{0}, {1/2}, {1}}"),
    ("CoordinateBoundsArray[{{0,2}}, Into[4]]", "{{0}, {1/2}, {1}, {3/2}, {2}}"),
    ("CoordinateBoundsArray[{{0,2}}, 1, Center]", "{{1/2}, {3/2}}"),
    ("CoordinateBoundsArray[w22b]", "CoordinateBoundsArray[w22b]"),
    ("CoordinateBoundsArray[{{0,1}}, w22h]", "CoordinateBoundsArray[{{0, 1}}, w22h]"),
    ("CoordinateBoundsArray[\"a\"]", Raises(T, "bounds must be")),
    ("CoordinateBoundsArray[{{0,1}}, 0]", Raises(V, "must be positive")),
    ("CoordinateBoundingBoxArray[{{0,0},{2,1}}]", "{{{0, 0}, {0, 1}}, {{1, 0}, {1, 1}}, {{2, 0}, {2, 1}}}"),
    ("CoordinateBoundingBoxArray[{{0,0},{2,1}}, 1, Center]", "{{{1/2, 1/2}}, {{3/2, 1/2}}}"),
    ("CoordinateBoundingBoxArray[{{0},{1}}, 1/2]", "{{0}, {1/2}, {1}}"),
    ("CoordinateBoundingBoxArray[w22b]", "CoordinateBoundingBoxArray[w22b]"),
    ("CoordinateBoundingBoxArray[{{0,0},{2,1}}, 1, w22o]", "CoordinateBoundingBoxArray[{{0, 0}, {2, 1}}, 1, w22o]"),
    ("CoordinateBoundingBoxArray[\"a\"]", Raises(T, "expected {pmin, pmax}")),
    ("CoordinateBoundingBoxArray[{{0,0},{2,1}}, 0]", Raises(V, "must be positive")),
    ("GeoDisk[{0,0}, Quantity[10,\"Kilometers\"]]", "GeoDisk[{0, 0}, Quantity[10, \"Kilometers\"]]"),
    ("GeoDisk[{10,20}, 5000]", "GeoDisk[{10, 20}, 5000]"),
    ("GeoDisk[GeoPosition[{1,2}], 5]", "GeoDisk[GeoPosition[{1, 2}], 5]"),
    ("GeoDisk[w22p, 5]", "GeoDisk[w22p, 5]"),
    ("GeoDisk[{0,0}, -1]", Raises(V, "non-negative")),
    ("GeoDisk[5, 1]", Raises(T)),
    ("GeoPath[{{0,0},{1,1}}]", "GeoPath[{{0, 0}, {1, 1}}]"),
    ("GeoPath[{{0,0},{1,1},{2,0}}]", "GeoPath[{{0, 0}, {1, 1}, {2, 0}}]"),
    ("GeoPath[{GeoPosition[{0,0}],GeoPosition[{1,1}]}]", "GeoPath[{GeoPosition[{0, 0}], GeoPosition[{1, 1}]}]"),
    ("GeoPath[w22p]", "GeoPath[w22p]"),
    ("GeoPath[\"a\"]", Raises(T, "must be a list")),
    ("GeoPath[{{0,0},{100,0}}]", Raises(V, "must lie in")),
]

# --- Check methods: self-checks and exact identities exercised explicitly -----------------------------------------
CHECKS = [
    # inclusion-exclusion: |A u B| = |A| + |B| - |A n B|
    ("Area[RegionUnion[Rectangle[], Rectangle[{1/2,1/2},{2,2}]]] == Area[Rectangle[]] + Area[Rectangle[{1/2,1/2},"
     "{2,2}]] - Area[RegionIntersection[Rectangle[], Rectangle[{1/2,1/2},{2,2}]]]", "True"),
    # CanonicalizeRegion returns an equal region
    ("RegionEqual[CanonicalizeRegion[Annulus[{0,0},{0,1}]], Annulus[{0,0},{0,1}]]", "True"),
    # discretization preserves measure
    ("Area[DiscretizeRegion[Polygon[{{0,0},{2,0},{1,1},{2,2},{0,2}}]]]", "3"),
    ("Volume[DiscretizeRegion[Tetrahedron[{{0,0,0},{1,0,0},{0,1,0},{0,0,1}}]]]", "1/6"),
    # boundary of the mesh has the perimeter of the region; Euler characteristic V - E + F
    ("Perimeter[BoundaryDiscretizeRegion[Rectangle[{0,0},{2,3}]]]", "10"),
    ("EulerCharacteristic[DiscretizeRegion[Rectangle[]]]", "1"),
    # Voronoi cells tile the bounds (self-check: ArithmeticError otherwise)
    ("Area[VoronoiMesh[{{0,0},{2,0},{1,3}}, {{-1,3},{-1,4}}]]", "20"),
    # EulerAngles / RollPitchYawAngles invert their matrices (self-check)
    ("EulerMatrix[EulerAngles[N[EulerMatrix[{1,2,3}/10]]]] - N[EulerMatrix[{1,2,3}/10]] // Abs // Flatten // Max",
     Pred(lambda g: float(g) < 1e-12, "< 1e-12")),
    # GeoDestination is verified by GeoDistance back to the start point (self-check)
    ("GeoDistance[{10,20}, GeoDestination[{10,20}, {500000, 45}]]", QNum(500.0, "Kilometers", 1e-9)),
    # transforms compose: T . T^-1 is the identity
    ("TranslationTransform[{1,2}][TranslationTransform[{-1,-2}][{5,7}]]", "{5, 7}"),
    ("ReflectionTransform[{1,2}][ReflectionTransform[{1,2}][{3,4}]]", "{3, 4}"),
    # minimal disk contains all points (self-check) and has a support point on its boundary
    ("RegionDistance[RegionBoundary[BoundingRegion[{{0,0},{2,0},{1,3}}, \"MinDisk\"]], {1,3}]", "0"),
    # MomentOfInertia satisfies the parallel axis theorem: I_p = I_c + m d^2
    ("MomentOfInertia[Cuboid[], {0,0,0}, {0,0,1}] - MomentOfInertia[Cuboid[], {1/2,1/2,1/2}, {0,0,1}]", "1/2"),
]

# --- the 12 mesh heads W22 reserved as errors are implemented (and tested) in batch X06 -------------------------
SKIPS = []

_SUBS = {"%SQ%": SQ, "%TRI%": TRI, "%TETB%": TETB, "%TET%": TET, "%HOLED%": HOLED, "%BTRI%": BTRI, "%HP%": HP,
         "%PTS%": PTS}


def _sub(s):
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    return s


def _expand(cases):
    return [(_sub(c[0]), _sub(c[1]) if isinstance(c[1], str) else c[1]) + tuple(c[2:]) for c in cases]


NEEDS_SCIPY = tuple(_sub(s) for s in NEEDS_SCIPY)
CASES = _expand(MEASURES + MOMENTS + DISTANCES + PREDICATES + CONSTRUCTORS + DERIVED + CURVATURES + INERT + MESHES
                + TRANSFORMS + GEO + CHECKS + SKIPS)
if HAVE_SCIPY:
    CASES += _expand(SCIPY_MESHES)
else:  # minimal install: these heads raise the clear 'requires SciPy' error instead
    CASES = [c for c in CASES if not any(n in c[0] for n in NEEDS_SCIPY)]
    CASES += [(_sub("DelaunayMesh[%PTS%]"), Raises(NotImplementedError, "requires SciPy"))]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w22(case):
    check(case)


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w22_delaunay_matches_scipy_area_and_empty_circles():
    """Delaunay triangulation of random points: same total area as SciPy's Qhull hull and empty circumcircles."""
    import numpy as np
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(22)
    P = rng.integers(0, 50, size=(25, 2))
    src = "{" + ",".join("{%d,%d}" % tuple(p) for p in P) + "}"
    m = ev(f"DelaunayMesh[{src}]")
    area = _f(ev(f"Area[DelaunayMesh[{src}]]"))
    assert abs(area - ConvexHull(P).volume) < 1e-9
    tris = [[int(i) - 1 for i in c.args[0].args] for c in m.args[1].args]
    for t in tris:
        a, b, c = (P[i].astype(float) for i in t)
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        r2 = (a[0] - ux) ** 2 + (a[1] - uy) ** 2
        others = [P[i] for i in range(len(P)) if i not in t]
        assert all((q[0] - ux) ** 2 + (q[1] - uy) ** 2 >= r2 - 1e-7 for q in others)


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w22_convex_hull_matches_scipy():
    import numpy as np
    from scipy.spatial import ConvexHull
    rng = np.random.default_rng(5)
    P = rng.integers(-20, 20, size=(30, 3))
    src = "{" + ",".join("{%d,%d,%d}" % tuple(p) for p in P) + "}"
    h = ConvexHull(P)
    assert abs(_f(ev(f"Volume[ConvexHullMesh[{src}]]")) - h.volume) < 1e-9
    assert abs(_f(ev(f"SurfaceArea[ConvexHullMesh[{src}]]")) - h.area) < 1e-9


def test_w22_delaunay_self_check_raises(monkeypatch):
    """The empty-circumsphere self-check raises ArithmeticError when the triangulation is wrong."""
    if not HAVE_SCIPY:
        pytest.skip("needs SciPy")
    from mikoshilang.library import wl_regions_mesh as M
    import scipy.spatial

    class Bad:
        def __init__(self, P, *a, **k):
            self.simplices = [[0, 1, 2], [0, 2, 3]] if len(P) == 4 else [[0, 1, 2]]

    monkeypatch.setattr(scipy.spatial, "Delaunay", Bad)
    with pytest.raises(ArithmeticError):
        ev("DelaunayMesh[{{0,0},{1,0},{3,1},{0,1}}]")
    assert M is not None


def test_w22_exact_sign_decisions():
    """Exact sign decisions used by the region predicates (rational shortcut and algebraic numbers)."""
    from mikoshilang.library import wl_regions as W
    assert W.sgn(W.sp.Rational(-3, 4)) == -1 and W.sgn(W.sp.sqrt(2) - 1) == 1


def test_w22_euler_angles_round_trip_random():
    import random
    rnd = random.Random(22)
    for _ in range(5):
        a = [rnd.uniform(-3, 3), rnd.uniform(0.1, 3), rnd.uniform(-3, 3)]
        got = ev("EulerAngles[N[EulerMatrix[{%r, %r, %r}]]]" % tuple(a))
        back = ev("N[EulerMatrix[{%r, %r, %r}]]" % tuple(float(x) for x in got.args))
        want = ev("N[EulerMatrix[{%r, %r, %r}]]" % tuple(a))
        assert max(abs(float(x) - float(y)) for r1, r2 in zip(back.args, want.args)
                   for x, y in zip(r1.args, r2.args)) < 1e-12


def test_w22_geodesic_symmetry():
    d1 = ev("GeoDistance[{51.5,0},{48.85,2.35}]")
    d2 = ev("GeoDistance[{48.85,2.35},{51.5,0}]")
    assert abs(float(d1.args[0]) - float(d2.args[0])) < 1e-9
