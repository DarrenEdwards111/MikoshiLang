"""Differential tests for Wolfram batch W16 (2-D function, list, log, date, statistical and 3-D plots; module
wl_plots2d).

Plots are compared by their sampled data and primitives, not pixels. Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>), "Details" sections: data layouts
  (ListPlot3D: height f_ij at {j, i}; ListContourPlot3D: farr[[i, j, k]] at {k, j, i} and 3 default contour levels;
  ListPolarPlot: values at equally spaced angles 2 Pi (i - 1)/n; TernaryListPlot: barycentric placement in the
  unit triangle; ArrayPlot3D: i top to bottom, j back to front, k left to right; SphericalPlot3D: theta is the
  polar angle; ListLinePlot3D: each row of an array is a curve; ListStepPlot default step Right).
* AbsoluteTime reference page: AbsoluteTime[{2020, 1, 1}] == 3786825600 (seconds since 1 Jan 1900).
* Mathematical identities: points of Plot[f] satisfy y = f(x); Log plots store Log[y]; normal quantiles
  (Phi^-1(0.1) = -1.2815515655446004, Abramowitz & Stegun table 26.5); the uniform CDF; the unit sphere |p| = 1.

Symbol names used as symbolic placeholders are prefixed ``w16`` so they cannot collide with other test modules.
"""
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr, Symbol

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:  # minimal install
    HAVE_SCIPY = False


# ───────────── helpers on the result structure ─────────────

def unwrap(g):
    return g.args[0] if isinstance(g, Expr) and g.head == "Legended" else g


def find(e, head):
    out = []
    if isinstance(e, Expr):
        if e.head == head:
            out.append(e)
        for a in e.args:
            out += find(a, head)
    return out


def fl(x):
    return float(x)


def coords(e):
    """List of coordinate tuples in a Point/Line argument (single point, list of points or list of lists)."""
    if isinstance(e, Expr) and e.head == "List" and e.args and all(not (isinstance(c, Expr) and c.head == "List")
                                                                    for c in e.args):
        return [tuple(fl(c) for c in e.args)]
    out = []
    for a in e.args:
        out += coords(a)
    return out


def prim_points(g, head):
    return [p for pr in find(unwrap(g), head) for p in coords(pr.args[0])]


def opt(g, name):
    for r in unwrap(g).args[1:]:
        if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == Symbol(name):
            return r.args[1]
    return None


def is_graphics(g, head="Graphics"):
    return isinstance(unwrap(g), Expr) and unwrap(g).head == head


def unevaluated(head):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head, f"stays {head}[...]")


def pts_are(head, want, tol=1e-9, kind="Point", sort=False):
    def f(g):
        got = prim_points(g, kind)
        a, b = (sorted(got), sorted(want)) if sort else (got, want)
        return is_graphics(g, head) and len(a) == len(b) and all(
            all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(p, q)) for p, q in zip(a, b))
    return Pred(f, f"{kind} points {want}")


def on_curve(f, kind="Line", tol=1e-7, head="Graphics"):
    def p(g):
        ps = prim_points(g, kind)
        return is_graphics(g, head) and len(ps) > 2 and all(abs(y - f(x)) <= tol * max(1.0, abs(f(x)))
                                                            for x, y in ps)
    return Pred(p, "samples lie on the curve")


def complex_points(g):
    return [p for gc in find(unwrap(g), "GraphicsComplex") for p in coords(gc.args[0])]


PHI_01 = -1.2815515655446004          # InverseCDF[NormalDistribution[], 0.1] (A&S table 26.5)
T2020 = 3786825600.0                  # AbsoluteTime[{2020, 1, 1}] (AbsoluteTime reference page)

CASES = [
    # ── Plot ──
    ("Plot[x^2, {x, 0, 1}]", on_curve(lambda x: x * x)),
    ("Plot[Sin[x], {x, 0, 2 Pi}]", on_curve(math.sin)),
    ("Plot[x^2, {x, 0, 1}, PlotPoints -> 3, MaxRecursion -> 0]", pts_are("Graphics", [(0, 0), (0.5, 0.25), (1, 1)],
                                                                         kind="Line")),
    ("Plot[{x, 2 x}, {x, 0, 1}]", Pred(lambda g: len(find(g, "Line")) == 2, "two curves")),
    ("Plot[1/x, {x, -1, 1}]", Pred(lambda g: len(find(g, "Line")) >= 2, "split at the pole")),
    ("Plot[Sqrt[x], {x, -1, 1}]", Pred(lambda g: all(x >= -1e-9 for x, _ in prim_points(g, "Line")),
                                       "no samples where Sqrt is not real")),
    ("Plot[w16a x, {x, 0, 1}]", unevaluated("Plot")),
    ("Plot[x, {x, 1, 0}]", Raises(ValueError)),
    ("Plot[x, {x, 0, 1}, PlotTheme -> \"Scientific\"]", Raises(NotImplementedError)),
    ("Plot[x, {x, 0, 1}, w16Bogus -> 1]", Raises(ValueError)),
    ("Plot[x, {x, 0, 1}, PlotLegends -> {\"a\"}]", Pred(lambda g: g.head == "Legended", "legend")),
    # ── LogPlot family: y stored as Log ──
    ("LogPlot[Exp[x], {x, 0, 1}]", on_curve(lambda x: x)),
    ("LogPlot[x^2, {x, 1, 2}]", on_curve(lambda x: 2 * math.log(x))),
    ("LogPlot[w16a x, {x, 1, 2}]", unevaluated("LogPlot")),
    ("LogPlot[x, {x, 2, 1}]", Raises(ValueError)),
    ("LogLinearPlot[x, {x, 1, 10}, PlotPoints -> 3, MaxRecursion -> 0]",
     pts_are("Graphics", [(0, 1), (math.log(10) / 2, math.sqrt(10)), (math.log(10), 10)], kind="Line")),
    ("LogLinearPlot[Log[x], {x, 1, 10}]", on_curve(lambda u: u)),
    ("LogLinearPlot[w16a x, {x, 1, 2}]", unevaluated("LogLinearPlot")),
    ("LogLinearPlot[x, {x, 1, 2}, PlotTheme -> \"Web\"]", Raises(NotImplementedError)),
    ("LogLogPlot[x^2, {x, 1, 10}]", on_curve(lambda u: 2 * u)),
    ("LogLogPlot[x^3, {x, 1, 2}, PlotPoints -> 3, MaxRecursion -> 0]",
     pts_are("Graphics", [(0, 0), (math.log(2) / 2, 1.5 * math.log(2)), (math.log(2), 3 * math.log(2))],
             kind="Line")),
    ("LogLogPlot[w16a x, {x, 1, 2}]", unevaluated("LogLogPlot")),
    ("LogLogPlot[x, {x, 1, 1}]", Raises(ValueError)),
    # ── list plots ──
    ("ListPlot[{1, 2, 3}]", pts_are("Graphics", [(1, 1), (2, 2), (3, 3)])),
    ("ListPlot[{{1, 4}, {2, 5}}]", pts_are("Graphics", [(1, 4), (2, 5)])),
    ("ListPlot[{1, 2}, DataRange -> {0, 10}]", pts_are("Graphics", [(0, 1), (10, 2)])),
    ("ListPlot[{1, w16a, 3}]", unevaluated("ListPlot")),
    ("ListPlot[{1, \"a\"}]", Raises(TypeError)),
    ("ListPlot[{1, 2}, Mesh -> 3]", Raises(NotImplementedError)),
    ("ListLinePlot[{{1, 2}, {2, 4}}]", pts_are("Graphics", [(1, 2), (2, 4)], kind="Line")),
    ("ListLinePlot[{1, None, 3, 4}]", Pred(lambda g: len(find(g, "Line")) == 1 and
                                          prim_points(g, "Line") == [(3.0, 3.0), (4.0, 4.0)], "gap at None")),
    ("ListLinePlot[{{1, 2, 3}, {3, 4, 5}}]", Pred(lambda g: len(find(g, "Line")) == 2, "two datasets")),
    ("ListLinePlot[w16data]", unevaluated("ListLinePlot")),
    ("ListLinePlot[{1, 2}, Filling -> \"x\"]", Raises(ValueError)),
    ("ListStepPlot[{1, 2, 3}]", pts_are("Graphics", [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 3)], kind="Line")),
    ("ListStepPlot[{{1, 1}, {3, 2}}, Left]", Pred(lambda g: prim_points(g, "Line")[:2] == [(1.0, 1.0), (1.0, 2.0)]
                                                  or len(prim_points(g, "Line")) >= 3, "left steps")),
    ("ListStepPlot[{1, 2}, Center]", Pred(lambda g: is_graphics(g), "center steps")),
    ("ListStepPlot[{1, 2}, w16where]", Raises((ValueError, TypeError))),
    ("ListLogPlot[{1, 10, 100}]", pts_are("Graphics", [(1, 0), (2, math.log(10)), (3, math.log(100))])),
    ("ListLogPlot[{{1, E}, {2, E^2}}]", pts_are("Graphics", [(1, 1), (2, 2)])),
    ("ListLogPlot[{1, 0, 10}]", Pred(lambda g: len(prim_points(g, "Point")) == 2, "non-positive values omitted")),
    ("ListLogPlot[{1, \"x\"}]", Raises(TypeError)),
    ("ListLogLinearPlot[{{1, 1}, {10, 2}}]", pts_are("Graphics", [(0, 1), (math.log(10), 2)])),
    ("ListLogLinearPlot[{{E, 1}, {E^3, 2}}]", pts_are("Graphics", [(1, 1), (3, 2)])),
    ("ListLogLinearPlot[{{1, 1}, {w16x, 2}}]", unevaluated("ListLogLinearPlot")),
    ("ListLogLogPlot[{{1, 1}, {10, 100}}]", pts_are("Graphics", [(0, 0), (math.log(10), math.log(100))])),
    ("ListLogLogPlot[{{E, E}, {E^2, E^4}}]", pts_are("Graphics", [(1, 1), (2, 4)])),
    ("ListLogLogPlot[{{1, 1}, {2, \"a\"}}]", Raises(TypeError)),
    ("ListPolarPlot[{1, 1, 1, 1}]", pts_are("Graphics", [(1, 0), (0, 1), (-1, 0), (0, -1)], tol=1e-12)),
    ("ListPolarPlot[{{0, 2}, {Pi/2, 3}}]", pts_are("Graphics", [(2, 0), (0, 3)], tol=1e-12)),
    ("ListPolarPlot[{1, w16r}]", unevaluated("ListPolarPlot")),
    ("ListPolarPlot[{1, 2}, ScalingFunctions -> \"Log\"]", Raises((NotImplementedError, ValueError))),
    ("StackedListPlot[{{1, 2}, {3, 4}}]", pts_are("Graphics", [(1, 1), (2, 2), (1, 4), (2, 6)], kind="Line")),
    ("StackedListPlot[{{1, 1}, {1, 1}, {1, 1}}]", Pred(lambda g: prim_points(g, "Line")[-2:] == [(1.0, 3.0),
                                                                                              (2.0, 3.0)],
                                                       "cumulative sums")),
    ("StackedListPlot[{{1, 2}, {3, w16b}}]", unevaluated("StackedListPlot")),
    ("StackedListPlot[{{1, 2}, {3, \"x\"}}]", Raises(TypeError)),
    # ── date plots ──
    ("DateListPlot[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 2}}]",
     pts_are("Graphics", [(T2020, 1), (T2020 + 86400, 2)], kind="Line")),
    ("DateListPlot[{1, 2, 3}, {2020, 1, 1}]", Pred(lambda g: prim_points(g, "Line")[0] == (T2020, 1.0) and
                                                   prim_points(g, "Line")[1][0] == T2020 + 86400, "daily dates")),
    ("DateListPlot[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 2}}]", Pred(lambda g: opt(g, "Frame") is True, "framed")),
    ("DateListPlot[{{w16d, 1}}]", unevaluated("DateListPlot")),
    ("DateListPlot[{{{2020, 13, 45, 1, 2, 3, 4, 5}, 1}}]", Raises((ValueError, TypeError))),
    ("DateListLogPlot[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 10}}]",
     pts_are("Graphics", [(T2020, 0), (T2020 + 86400, math.log(10))], kind="Line")),
    ("DateListLogPlot[{{{2020, 1, 1}, E}, {{2020, 1, 3}, E^2}}]",
     pts_are("Graphics", [(T2020, 1), (T2020 + 2 * 86400, 2)], kind="Line")),
    ("DateListLogPlot[{{w16d, 1}}]", unevaluated("DateListLogPlot")),
    ("DateListStepPlot[{{{2020, 1, 1}, 1}, {{2020, 1, 3}, 2}}]",
     pts_are("Graphics", [(T2020, 1), (T2020 + 172800, 1), (T2020 + 172800, 2), (T2020 + 345600, 2)], kind="Line")),
    ("DateListStepPlot[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 3}}]", Pred(lambda g: len(prim_points(g, "Line")) == 4,
                                                                      "one step per value")),
    ("DateListStepPlot[{{w16d, 1}}]", unevaluated("DateListStepPlot")),
    ("StackedDateListPlot[{{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 2}}, {{{2020, 1, 1}, 3}, {{2020, 1, 2}, 4}}}]",
     pts_are("Graphics", [(T2020, 1), (T2020 + 86400, 2), (T2020, 4), (T2020 + 86400, 6)], kind="Line")),
    ("StackedDateListPlot[{{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 1}}, {{{2020, 1, 1}, 1}, {{2020, 1, 2}, 1}}}]",
     Pred(lambda g: len(find(g, "Polygon")) == 2, "stacked fillings")),
    ("StackedDateListPlot[{{{{2020, 1, 1}, w16v}}, {{{2020, 1, 1}, 2}}}]", unevaluated("StackedDateListPlot")),
    # ── DiscretePlot ──
    ("DiscretePlot[n^2, {n, 1, 3}]", pts_are("Graphics", [(1, 1), (2, 4), (3, 9)])),
    ("DiscretePlot[n^2, {n, 1, 3}]", Pred(lambda g: len(find(g, "Line")) == 3, "stems to the axis")),
    ("DiscretePlot[1/n, {n, {1, 2, 4}}, Filling -> None]", pts_are("Graphics", [(1, 1), (2, 0.5), (4, 0.25)])),
    ("DiscretePlot[w16a n, {n, 1, 3}]", unevaluated("DiscretePlot")),
    ("DiscretePlot[n, {n, 1, 3, 0}]", Raises(ValueError)),
    ("DiscretePlot[n, {n, 1, 3}, ExtentSize -> Full]", Raises(NotImplementedError)),
    # ── arrays ──
    ("ArrayPlot[{{0, 1}, {1, 0}}]", Pred(lambda g: find(g, "Raster")[0].args[0] == ev(
        "{{{0., 0., 0.}, {1., 1., 1.}}, {{1., 1., 1.}, {0., 0., 0.}}}"), "black 1s, white 0s (row 1 at the top)")),
    ("ArrayPlot[{{1, 2}}]", Pred(lambda g: [fl(c) for c in find(g, "Raster")[0].args[0].args[0].args[0].args] ==
                                 [0.5, 0.5, 0.5], "grey 1 - v/max")),
    ("ArrayPlot[{{1, w16a}}]", unevaluated("ArrayPlot")),
    ("ArrayPlot[{{1, {2}}}]", Raises(ValueError)),
    ("ArrayPlot[3]", Raises(TypeError)),
    ("MatrixPlot[{{1, -1}, {0, 2}}]", Pred(lambda g: is_graphics(g) and opt(g, "Frame") is True, "framed")),
    ("MatrixPlot[{{0, 0}, {0, 0}}]", Pred(lambda g: len(find(g, "Raster")) == 1, "one raster")),
    ("MatrixPlot[{{1, -1}, {0, 2}}]", Pred(lambda g: [fl(c) for c in find(g, "Raster")[0].args[0].args[0].args[0]
                                                      .args] == [0.95, 0.95, 0.95], "0 is white-grey")),
    ("MatrixPlot[{{1, w16a}}]", unevaluated("MatrixPlot")),
    ("MatrixPlot[\"x\"]", Raises(TypeError)),
    ("ReliefPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: len(find(g, "Raster")) == 1, "raster")),
    ("ReliefPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: [fl(v) for v in opt(g, "PlotRange").args[0].args] == [0.5, 2.5],
                                          "cells centred on the indices")),
    ("ReliefPlot[{{1, 1}, {1, 1}}]", Pred(lambda g: is_graphics(g), "flat")),
    ("ReliefPlot[{{1, w16a}}]", unevaluated("ReliefPlot")),
    ("ReliefPlot[{{1, 2}}, ColorFunction -> 3]", Raises(ValueError)),
    # ── 2-D fields ──
    ("DensityPlot[x + y, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 2, MaxRecursion -> 0]",
     Pred(lambda g: complex_points(g) == [(0., 0.), (1., 0.), (0., 1.), (1., 1.)], "grid vertices")),
    ("DensityPlot[x, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 2, MaxRecursion -> 0]",
     Pred(lambda g: [c for c in opt(find(g, "GraphicsComplex")[0], "VertexColors").args][0] ==
          ev("RGBColor[0.26, 0.15, 0.44]"), "minimum gets the first gradient colour")),
    ("DensityPlot[x y, {x, 0, 1}, {y, 0, 1}]", Pred(lambda g: len(complex_points(g)) == 49 * 49, "49 x 49 grid")),
    ("DensityPlot[w16a x, {x, 0, 1}, {y, 0, 1}]", unevaluated("DensityPlot")),
    ("DensityPlot[x, {x, 0, 1}, {x, 0, 1}]", Raises(ValueError)),
    ("DensityPlot[x, {x, 0, 1}, {y, 0, 1}, Mesh -> 3]", Raises(NotImplementedError)),
    ("RegionPlot[x < 0.5, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 3, MaxRecursion -> 0]",
     Pred(lambda g: all(x <= 0.5 + 1e-12 for x, _ in prim_points(g, "Polygon")), "polygons inside the region")),
    ("RegionPlot[x^2 + y^2 < 1, {x, -1, 1}, {y, -1, 1}]",
     Pred(lambda g: all(x * x + y * y <= 1 + 1e-9 for x, y in prim_points(g, "Polygon")), "inside the disk")),
    ("RegionPlot[x > y, {x, 0, 1}, {y, 0, 1}]", Pred(lambda g: len(find(g, "Line")) >= 1, "boundary drawn")),
    ("RegionPlot[x > w16a, {x, 0, 1}, {y, 0, 1}]", unevaluated("RegionPlot")),
    ("RegionPlot[x == y, {x, 0, 1}, {y, 0, 1}]", Raises(NotImplementedError)),
    ("RegionPlot[x > y, {x, 1, 0}, {y, 0, 1}]", Raises(ValueError)),
    ("ListDensityPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: complex_points(g) == [(1., 1.), (2., 1.), (1., 2.),
                                                                              (2., 2.)], "z_ij at {j, i}")),
    ("ListDensityPlot[{{1, 2}, {3, 4}}, DataRange -> {{0, 1}, {0, 1}}]",
     Pred(lambda g: complex_points(g)[-1] == (1., 1.), "DataRange")),
    ("ListDensityPlot[{{1, 2}, {3, w16a}}]", unevaluated("ListDensityPlot")),
    ("ListDensityPlot[{{1, 2}, {3}}]", Raises(ValueError)),
    ("ListContourPlot[{{0, 1}, {1, 2}}, Contours -> {1}]",
     Pred(lambda g: [tuple(map(fl, p)) for p in coords(find(g, "Line")[0].args[0])] in
          ([(2.0, 1.0), (1.0, 2.0)], [(1.0, 2.0), (2.0, 1.0)]) or len(find(g, "Line")) == 0
          or all(abs(x + y - 3) < 1e-9 for x, y in prim_points(g, "Line")), "level 1 is x + y = 3")),
    ("ListContourPlot[{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}, Contours -> {2.5}]",
     Pred(lambda g: all(abs(x + y - 4.5) < 1e-9 for x, y in prim_points(g, "Line")), "linear interpolation")),
    ("ListContourPlot[{{0, 1}, {1, 2}}, Contours -> 3]", Pred(lambda g: len(find(g, "Line")) >= 2, "3 levels")),
    ("ListContourPlot[{{0, 1}, {1, w16a}}]", unevaluated("ListContourPlot")),
    ("ListContourPlot[{{0, 1}, {1, 2}}, Contours -> {\"a\"}]", Raises(ValueError)),
    # ── statistics ──
    ("QuantilePlot[{1, 2, 3, 4, 5}]", Pred(lambda g: abs(prim_points(g, "Point")[0][0] - (3 + math.sqrt(2) * PHI_01))
                                           < 1e-9, "Hazen position of the minimum on the fitted normal")),
    ("QuantilePlot[{1, 2, 3}, NormalDistribution[0, 1]]",
     Pred(lambda g: abs(prim_points(g, "Point")[1][0]) < 1e-12 and [p[1] for p in prim_points(g, "Point")] ==
          [1.0, 2.0, 3.0], "median at 0, sorted data")),
    ("QuantilePlot[{3, 1, 2}, UniformDistribution[{0, 1}]]",
     pts_are("Graphics", [(1 / 6, 1), (0.5, 2), (5 / 6, 3)])),
    ("QuantilePlot[{1, 2, w16a}]", unevaluated("QuantilePlot")),
    ("QuantilePlot[{1, \"a\"}]", Raises(TypeError)),
    ("ProbabilityPlot[{1, 2, 3}, UniformDistribution[{0, 4}]]",
     pts_are("Graphics", [(0.25, 1 / 3), (0.5, 2 / 3), (0.75, 1.0)])),
    ("ProbabilityPlot[{1, 2, 3}, UniformDistribution[{0, 4}]]",
     Pred(lambda g: prim_points(g, "Line") == [(0.0, 0.0), (1.0, 1.0)], "diagonal reference line")),
    ("ProbabilityPlot[{2, 1}, UniformDistribution[{0, 2}]]", pts_are("Graphics", [(0.5, 0.5), (1.0, 1.0)])),
    ("ProbabilityPlot[{1, w16a}]", unevaluated("ProbabilityPlot")),
    ("ProbabilityPlot[{1, \"a\"}]", Raises(TypeError)),
    ("ProbabilityScalePlot[{1, 2, 3}]", Pred(lambda g: abs(prim_points(g, "Point")[1][1] - 1.0) < 1e-12,
                                             "median rank 0.5 at h = 1")),
    ("ProbabilityScalePlot[{1, 2, 3}, \"Exponential\"]",
     Pred(lambda g: abs(prim_points(g, "Point")[0][1] + math.log(1 - 0.7 / 3.4)) < 1e-12, "Benard rank")),
    ("ProbabilityScalePlot[{1, 2, 3}, \"Weibull\"]", Pred(lambda g: prim_points(g, "Point")[0][0] == 0.0, "Log x")),
    ("ProbabilityScalePlot[{1, 2, 3}, \"Bogus\"]", Raises(ValueError)),
    ("ProbabilityScalePlot[{1, 2, 3}, 3]", Raises(TypeError)),
    ("ProbabilityScalePlot[{1, w16a}]", unevaluated("ProbabilityScalePlot")),
    ("PairwiseListPlot[{{1, 2}, {3, 4}, {5, 7}}]",
     Pred(lambda g: g.head == "GraphicsGrid" and coords(find(g.args[0].args[1].args[0], "Point")[0].args[0]) ==
          [(1., 2.), (3., 4.), (5., 7.)], "panel (2, 1) is y1 against y2")),
    ("PairwiseListPlot[{{1, 2, 3}, {3, 4, 5}}]", Pred(lambda g: len(g.args[0].args) == 3, "3 x 3 panels")),
    ("PairwiseListPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: len(find(g, "Text")) == 2, "headers on the diagonal")),
    ("PairwiseListPlot[w16m]", unevaluated("PairwiseListPlot")),
    ("PairwiseListPlot[{{1, \"a\"}, {2, 3}}]", Raises(TypeError)),
    ("PairwiseQuantilePlot[{{1, 2}, {3, 4}, {5, 7}}]",
     Pred(lambda g: g.head == "GraphicsGrid" and len(g.args[0].args) == 2, "2 x 2 panels")),
    ("PairwiseQuantilePlot[{{1, 3}, {2, 2}, {3, 1}}]",
     Pred(lambda g: coords(find(g.args[0].args[1].args[0], "Point")[0].args[0]) == [(1., 1.), (2., 2.), (3., 3.)],
          "sorted against sorted")),
    ("PairwiseQuantilePlot[w16m]", unevaluated("PairwiseQuantilePlot")),
    ("PairwiseQuantilePlot[{{1, \"a\"}, {2, 3}}]", Raises(TypeError)),
    ("PairwiseProbabilityPlot[{{1, 2}, {3, 4}, {5, 7}}]",
     Pred(lambda g: coords(find(g.args[0].args[0].args[1], "Point")[0].args[0]) == [(1 / 3, 1 / 3), (2 / 3, 2 / 3),
                                                                                    (1.0, 1.0)] and
          coords(find(g.args[0].args[1].args[0], "Point")[0].args[0]) == [(1 / 3, 0.0), (2 / 3, 1 / 3), (1.0, 2 / 3)],
          "ECDF of y_i at the order statistics of y_j")),
    ("PairwiseProbabilityPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: g.head == "GraphicsGrid", "grid")),
    ("PairwiseProbabilityPlot[w16m]", unevaluated("PairwiseProbabilityPlot")),
    ("PairwiseProbabilityPlot[{{1, \"a\"}, {2, 3}}]", Raises(TypeError)),
    # ── number line, axes, ternary, timeline, curve paths, grids ──
    ("NumberLinePlot[{1, 2, 3}]", pts_are("Graphics", [(1, 1), (2, 1), (3, 1)])),
    ("NumberLinePlot[x^2 < 4, x]", Pred(lambda g: prim_points(g, "Line") == [(-2.0, 1.0), (2.0, 1.0)],
                                        "solution interval (-2, 2)")),
    ("NumberLinePlot[x == 2 || x^2 == 9, x]", pts_are("Graphics", [(-3, 1), (2, 1), (3, 1)], sort=True)),
    ("NumberLinePlot[{Interval[{1, 2}], x >= 3}, x]", Pred(lambda g: (3.0, 2.0) in prim_points(g, "Point"),
                                                           "second spec at height 2")),
    ("NumberLinePlot[x > w16a, x]", unevaluated("NumberLinePlot")),
    ("NumberLinePlot[x > 1]", Raises(ValueError)),
    ("NumberLinePlot[Sin[x], x]", Raises((NotImplementedError, TypeError))),
    ("NumberLinePlot[\"a\"]", Raises(TypeError)),
    ("ParallelAxisPlot[{{1, 2, 3}, {3, 2, 1}}]", pts_are("Graphics", [(1, 0), (2, 0.5), (3, 1), (1, 1), (2, 0.5),
                                                                      (3, 0)], kind="Line", sort=True)
     if False else Pred(lambda g: [(1.0, 0.0), (2.0, 0.5), (3.0, 1.0)] == coords(find(g, "Line")[3].args[0]),
                        "scaled coordinates")),
    ("ParallelAxisPlot[{{0, 10}, {5, 20}}]", Pred(lambda g: coords(find(g, "Line")[2].args[0]) ==
                                                  [(1.0, 0.0), (2.0, 0.0)], "per-axis min at 0")),
    ("ParallelAxisPlot[{{1, w16a}}]", unevaluated("ParallelAxisPlot")),
    ("ParallelAxisPlot[{{1}, {2}}]", Raises(ValueError)),
    ("RadialAxisPlot[{1, 1, 1, 1}]", Pred(lambda g: all(abs(math.hypot(x, y) - 1) < 1e-12 for x, y in
                                                        coords(find(g, "Line")[-1].args[0])), "radius 1 at the max")),
    ("RadialAxisPlot[{2, 1, 2}]", Pred(lambda g: abs(coords(find(g, "Line")[-1].args[0])[0][1] - 1) < 1e-12,
                                       "first axis points up")),
    ("RadialAxisPlot[{1, w16a, 2}]", unevaluated("RadialAxisPlot")),
    ("RadialAxisPlot[{1, 2}]", Raises(ValueError)),
    ("TernaryListPlot[{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}]",
     pts_are("Graphics", [(0, 0), (1, 0), (0.5, math.sqrt(3) / 2)], tol=1e-12)),
    ("TernaryListPlot[{{1, 1, 2}}]", pts_are("Graphics", [(0.5, math.sqrt(3) / 4)], tol=1e-12)),
    ("TernaryListPlot[{{1, 1, 1}, {2, 2, 2}}]", pts_are("Graphics", [(0.5, math.sqrt(3) / 6)] * 2, tol=1e-12)),
    ("TernaryListPlot[{{1, w16a, 1}}]", unevaluated("TernaryListPlot")),
    ("TernaryListPlot[{1, 2}]", Raises(ValueError)),
    ("TimelinePlot[{{2020, 1, 1}, {2020, 1, 2}}]", pts_are("Graphics", [(T2020, 1), (T2020 + 86400, 1)])),
    ("TimelinePlot[<|\"a\" -> {2020, 1, 1}|>]", Pred(lambda g: find(g, "Text")[0].args[0] == "a", "labelled")),
    ("TimelinePlot[{DateInterval[{{2020, 1, 1}, {2020, 1, 3}}]}]",
     Pred(lambda g: prim_points(g, "Line") == [(T2020, 1.0), (T2020 + 172800, 1.0)], "interval segment")),
    ("TimelinePlot[w16events]", unevaluated("TimelinePlot")),
    ("TimelinePlot[3]", Raises(TypeError)),
    ("ListCurvePathPlot[{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}]",
     Pred(lambda g: len(prim_points(g, "Line")) == 5, "closed loop through 4 points")),
    ("ListCurvePathPlot[{{2, 4}, {0, 0}, {3, 9}, {1, 1}}]",
     Pred(lambda g: prim_points(g, "Line") in ([(0., 0.), (1., 1.), (2., 4.), (3., 9.)],
                                               [(3., 9.), (2., 4.), (1., 1.), (0., 0.)]), "parabola order")),
    ("ListCurvePathPlot[{{0, 0}, {1, 0}, {10, 0}, {11, 0}}]", Pred(lambda g: len(find(g, "Line")) == 2,
                                                                  "gap splits the path")),
    ("ListCurvePathPlot[{{0, 0}, {1, w16a}}]", unevaluated("ListCurvePathPlot")),
    ("ListCurvePathPlot[{{0, 0}}, InterpolationOrder -> 3]", Raises(NotImplementedError)),
    ("PlotGrid[{{Graphics[{}], None}, {Graphics[{}], Graphics[{}]}}]",
     Pred(lambda g: [tuple(map(fl, i.args[1].args)) for i in find(g, "Inset")] ==
          [(0.5, 1.5), (0.5, 0.5), (1.5, 0.5)], "row 1 at the top")),
    ("PlotGrid[{Graphics[{}], Graphics[{}]}]", Pred(lambda g: len(find(g, "Inset")) == 2, "one row")),
    ("PlotGrid[{{ListPlot[{1, 2}]}}]", Pred(lambda g: is_graphics(g), "graphics")),
    ("PlotGrid[w16grid]", unevaluated("PlotGrid")),
    ("PlotGrid[{{3}}]", Raises(TypeError)),
    # ── 3-D list plots ──
    ("ListPlot3D[{{1, 2, 3, 4}, {2, 3, 4, 5}}]", Pred(lambda g: complex_points(g)[:2] == [(1., 1., 1.), (2., 1., 2.)]
                                                       and is_graphics(g, "Graphics3D"), "f_ij at {j, i}")),
    ("ListPlot3D[{{0, 0}, {0, 0}}]", Pred(lambda g: len(find(g, "Polygon")[0].args[0].args) == 2, "2 triangles")),
    ("ListPlot3D[{{1, 2}, {3, 4}}, DataRange -> {{0, 1}, {0, 2}}]",
     Pred(lambda g: complex_points(g)[-1] == (1., 2., 4.), "DataRange")),
    ("ListPlot3D[{{1, 2}, {3, w16a}}]", unevaluated("ListPlot3D")),
    ("ListPlot3D[{{1, 2}, {3, \"a\"}}]", Raises(TypeError)),
    ("ListPlot3D[{{1, 2}, {3, 4}}, Mesh -> 4]", Raises(NotImplementedError)),
    ("ListPointPlot3D[{{1, 2, 3}, {4, 5, 6}}]", pts_are("Graphics3D", [(1, 2, 3), (4, 5, 6)])),
    ("ListPointPlot3D[{{1, 2}, {3, 4}}]", pts_are("Graphics3D", [(1, 1, 1), (2, 1, 2), (1, 2, 3), (2, 2, 4)])),
    ("ListPointPlot3D[{{{0, 0, 0}}, {{1, 1, 1}}}]", Pred(lambda g: len(find(g, "Point")) == 2, "two datasets")),
    ("ListPointPlot3D[{{1, 2, w16a}}]", unevaluated("ListPointPlot3D")),
    ("ListPointPlot3D[{{1, 2, 3}}, Filling -> Top]", Raises(NotImplementedError)),
    ("ListLinePlot3D[{{1, 2, 3}, {2, 3, 1}}]", pts_are("Graphics3D", [(1, 2, 3), (2, 3, 1)], kind="Line")),
    ("ListLinePlot3D[{{1, 2, 3, 4}, {2, 3, 1, 0}}]", Pred(lambda g: len(find(g, "Line")) == 2 and
                                                          prim_points(g, "Line")[4] == (1., 2., 2.), "rows at y = i")),
    ("ListLinePlot3D[{{1, 2, 3}, None, {2, 3, 4}, {3, 4, 5}}]", Pred(lambda g: len(find(g, "Line")) == 1,
                                                                      "gap at None")),
    ("ListLinePlot3D[{{1, 2, w16a}}]", unevaluated("ListLinePlot3D")),
    ("ListLinePlot3D[{{1, 2, \"a\"}}]", Raises(TypeError)),
    ("ArrayPlot3D[{{{1}}}]", Pred(lambda g: find(g, "Cuboid")[0] == ev("Cuboid[{0., 0., 0.}, {1., 1., 1.}]"),
                                  "one unit cube")),
    ("ArrayPlot3D[{{{1, 0}, {0, 1}}, {{0.5, 0}, {0, 0}}}]",
     Pred(lambda g: len(find(g, "Cuboid")) == 3 and ev("Cuboid[{0., 1., 1.}, {1., 2., 2.}]") in find(g, "Cuboid"),
          "a_111 at the top back left; zeros omitted")),
    ("ArrayPlot3D[{{{2, 1}}}]", Pred(lambda g: ev("RGBColor[0.5, 0.5, 0.5]") in find(g, "RGBColor") or
                                     any(c.args[0] == 0.5 for c in find(g, "RGBColor")), "grey 1 - v/max")),
    ("ArrayPlot3D[w16arr]", unevaluated("ArrayPlot3D")),
    ("ArrayPlot3D[{1, 2}]", Raises(ValueError)),
    ("DiscretePlot3D[i + j, {i, 1, 2}, {j, 1, 2}]", pts_are("Graphics3D", [(1, 1, 2), (1, 2, 3), (2, 1, 3),
                                                                            (2, 2, 4)])),
    ("DiscretePlot3D[i j, {i, 1, 2}, {j, 1, 3}]", Pred(lambda g: len(find(g, "Line")) == 6, "stems by default")),
    ("DiscretePlot3D[i, {i, 1, 2}, {j, 1, 2}, Filling -> None]", Pred(lambda g: not find(g, "Line"), "no stems")),
    ("DiscretePlot3D[w16a i, {i, 1, 2}, {j, 1, 2}]", unevaluated("DiscretePlot3D")),
    ("DiscretePlot3D[i, {i, 1, 2}, {i, 1, 2}]", Raises(ValueError)),
    ("DiscretePlot3D[i, {i, 1, 2}, {j, 1, 2}, ExtentSize -> Full]", Raises(NotImplementedError)),
    # ── 3-D functions ──
    ("ParametricPlot3D[{Cos[t], Sin[t], t}, {t, 0, 2 Pi}]",
     Pred(lambda g: all(abs(x * x + y * y - 1) < 1e-9 for x, y, _ in prim_points(g, "Line")), "on the helix")),
    ("ParametricPlot3D[{u, v, u v}, {u, 0, 1}, {v, 0, 1}]",
     Pred(lambda g: all(abs(z - x * y) < 1e-12 for x, y, z in complex_points(g)), "on the saddle")),
    ("ParametricPlot3D[{{t, 0, 0}, {0, t, 0}}, {t, 0, 1}]", Pred(lambda g: len(find(g, "Line")) == 2, "two curves")),
    ("ParametricPlot3D[{w16a t, t, t}, {t, 0, 1}]", unevaluated("ParametricPlot3D")),
    ("ParametricPlot3D[{t, t}, {t, 0, 1}]", Raises(ValueError)),
    ("SphericalPlot3D[1, {t, 0, Pi}, {p, 0, 2 Pi}]",
     Pred(lambda g: all(abs(math.sqrt(x * x + y * y + z * z) - 1) < 1e-12 for x, y, z in complex_points(g)),
          "unit sphere")),
    ("SphericalPlot3D[2, t, p]", Pred(lambda g: complex_points(g)[0] == (0.0, 0.0, 2.0), "theta from the z axis")),
    ("SphericalPlot3D[{1, 2}, {t, 0, Pi}, {p, 0, Pi}]", Pred(lambda g: len(find(g, "GraphicsComplex")) == 2,
                                                             "two surfaces")),
    ("SphericalPlot3D[w16a, {t, 0, Pi}, {p, 0, Pi}]", unevaluated("SphericalPlot3D")),
    ("SphericalPlot3D[1, {t, 0, Pi}, {t, 0, Pi}]", Raises(ValueError)),
    ("ContourPlot3D[x^2 + y^2 + z^2 == 1, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, PlotPoints -> 9]",
     Pred(lambda g: all(abs(math.sqrt(x * x + y * y + z * z) - 1) < 0.1 for x, y, z in complex_points(g)),
          "near the unit sphere")),
    ("ContourPlot3D[x + y + z, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, Contours -> {0}, PlotPoints -> 5]",
     Pred(lambda g: all(abs(x + y + z) < 1e-9 for x, y, z in complex_points(g)), "exact for a linear f")),
    ("ContourPlot3D[x, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, PlotPoints -> 5]",
     Pred(lambda g: len(find(g, "GraphicsComplex")) == 3, "3 default levels")),
    ("ContourPlot3D[x == w16a, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]", unevaluated("ContourPlot3D")),
    ("ContourPlot3D[x, {x, -1, 1}, {y, -1, 1}]", Raises(ValueError)),
    ("ContourPlot3D[x, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, Mesh -> 3]", Raises(NotImplementedError)),
    ("ListContourPlot3D[Table[x + y + z, {z, 1, 3}, {y, 1, 3}, {x, 1, 3}], Contours -> {6}]",
     Pred(lambda g: all(abs(x + y + z - 6) < 1e-9 for x, y, z in complex_points(g)), "farr[[i, j, k]] at {k, j, i}")),
    ("ListContourPlot3D[Table[x, {z, 1, 3}, {y, 1, 3}, {x, 1, 3}]]",
     Pred(lambda g: sorted({round(p[0], 9) for p in complex_points(g)}) == [1.5, 2.0, 2.5], "levels 1.5, 2, 2.5")),
    ("ListContourPlot3D[Table[x, {z, 1, 2}, {y, 1, 2}, {x, 1, 2}], ContourStyle -> None]",
     Pred(lambda g: not find(g, "GraphicsComplex"), "no surfaces")),
    ("ListContourPlot3D[{{{1, 2}, {3, 4}}, {{5, 6}, {7, w16a}}}]", unevaluated("ListContourPlot3D")),
    ("ListContourPlot3D[{{1, 2}, {3, 4}}]", Raises(ValueError)),
    ("RegionPlot3D[x^2 + y^2 + z^2 < 1, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, PlotPoints -> 9]",
     Pred(lambda g: all(math.sqrt(x * x + y * y + z * z) < 1.1 for x, y, z in complex_points(g)), "ball boundary")),
    ("RegionPlot3D[x > 0, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, PlotPoints -> 5]",
     Pred(lambda g: all(abs(x) < 1e-9 for x, _, _ in complex_points(g)) and
          all(x >= -1e-9 for x, _, _ in prim_points(g, "Polygon")), "the plane x = 0 and caps with x >= 0")),
    ("RegionPlot3D[x > 0, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, PlotPoints -> 5]",
     Pred(lambda g: [list(map(fl, r.args)) for r in opt(g, "PlotRange").args] == [[-1, 1]] * 3, "full ranges")),
    ("RegionPlot3D[x > w16a, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]", unevaluated("RegionPlot3D")),
    ("RegionPlot3D[x == 0, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}]", Raises(NotImplementedError)),
    ("DensityPlot3D[x, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 2]",
     Pred(lambda g: [fl(c) for c in find(g, "Raster3D")[0].args[0].args[0].args[0].args[0].args] ==
          [0.26, 0.15, 0.44, 0.15], "minimum cell: first gradient colour, opacity 0.15")),
    ("DensityPlot3D[x, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 3, OpacityFunction -> None]",
     Pred(lambda g: all(fl(c.args[3]) == 1.0 for s in find(g, "Raster3D")[0].args[0].args for r in s.args
                        for c in r.args), "opaque")),
    ("DensityPlot3D[x, {x, 0, 1}, {y, 0, 1}, {z, 0, 2}, PlotPoints -> 2]",
     Pred(lambda g: find(g, "Raster3D")[0].args[1] == ev("{{0., 0., 0.}, {1., 1., 2.}}"), "box")),
    ("DensityPlot3D[w16a x, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("DensityPlot3D")),
    ("DensityPlot3D[x, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, OpacityFunction -> 2]", Raises(ValueError)),
    ("ListDensityPlot3D[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]",
     Pred(lambda g: [fl(c) for c in find(g, "Raster3D")[0].args[0].args[1].args[1].args[1].args[:3]] ==
          [0.94, 0.49, 0.24], "maximum: last gradient colour")),
    ("ListDensityPlot3D[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, DataRange -> {{0, 1}, {0, 1}, {0, 1}}]",
     Pred(lambda g: find(g, "Raster3D")[0].args[1] == ev("{{0., 0., 0.}, {1., 1., 1.}}"), "DataRange")),
    ("ListDensityPlot3D[{{{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}}]", Pred(lambda g: is_graphics(g, "Graphics3D"),
                                                                     "constant data")),
    ("ListDensityPlot3D[{{{1, 2}, {3, 4}}, {{5, 6}, {7, w16a}}}]", unevaluated("ListDensityPlot3D")),
    ("ListDensityPlot3D[{{{1, 2}, {3, 4}}, {{5, 6}}}]", Raises(ValueError)),
    ("SurfaceContourPlot3D[x y, Element[{x, y, z}, Sphere[]]]",
     Pred(lambda g: all(abs(math.sqrt(x * x + y * y + z * z) - 1) < 0.01 for x, y, z in prim_points(g, "Line")),
          "contours on the (meshed) sphere")),
    ("SurfaceContourPlot3D[z, Element[{x, y, z}, Sphere[]], Contours -> {0}]",
     Pred(lambda g: all(abs(z) < 1e-9 for _, _, z in prim_points(g, "Line")), "the equator")),
    ("SurfaceContourPlot3D[First[p], Element[p, Cuboid[]], Contours -> {0.5}]",
     Pred(lambda g: all(abs(x - 0.5) < 1e-9 for x, _, _ in prim_points(g, "Line")), "x = 1/2 on the cube")),
    ("SurfaceContourPlot3D[w16a x, Element[{x, y, z}, Sphere[]]]", unevaluated("SurfaceContourPlot3D")),
    ("SurfaceContourPlot3D[x, Element[{x, y, z}, Torus[]]]", Raises(NotImplementedError)),
    ("SurfaceContourPlot3D[x, {x, y, z}]", Raises(ValueError)),
    ("SurfaceDensityPlot3D[Norm[p], Element[p, Sphere[{0, 0, 0}, 2]]]",
     Pred(lambda g: all(abs(math.sqrt(x * x + y * y + z * z) - 2) < 1e-9 for x, y, z in complex_points(g)),
          "sphere of radius 2")),
    ("SurfaceDensityPlot3D[z, Element[{x, y, z}, Cylinder[]]]",
     Pred(lambda g: all(abs(z) <= 1 + 1e-9 for _, _, z in complex_points(g)), "default cylinder")),
    ("SurfaceDensityPlot3D[First[p], Element[p, Cone[]]]", Pred(lambda g: len(find(g, "VertexColors")) == 0 and
                                                                is_graphics(g, "Graphics3D"), "cone")),
    ("SurfaceDensityPlot3D[w16a x, Element[{x, y, z}, Sphere[]]]", unevaluated("SurfaceDensityPlot3D")),
    ("SurfaceDensityPlot3D[x, Element[{x, y, z}, Torus[]]]", Raises(NotImplementedError)),
]

if HAVE_SCIPY:
    CASES += [
        ("ListPlot3D[{{1, 2, 3}, {2, 1, 4}, {3, 4, 5}, {0, 0, 1}}]",
         Pred(lambda g: complex_points(g)[0] == (1., 2., 3.), "Delaunay of the triples")),
        ("ListDensityPlot[{{0, 0, 1}, {1, 0, 2}, {0, 1, 3}, {1, 1, 4}}]",
         Pred(lambda g: len(complex_points(g)) == 4, "triples")),
    ]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_w16(case):
    check(case)


def test_w16_skipped_symbols():
    from mikoshilang.library import STATUS
    from mikoshilang.library.wl_plots2d import SKIPPED
    assert set(SKIPPED) == {"ChromaticityPlot", "ChromaticityPlot3D", "BioSequencePlot", "SingularValuePlot",
                            "ImageWaveformPlot"}   # ListSurfacePlot3D: implemented in X06
    assert not any(n in STATUS for n in SKIPPED)


def test_w16_cross_check_discards_wrong_compiled_code():
    """Check: the compiled function is compared with the kernel's N; a mismatch falls back to the kernel."""
    import numpy as np
    from mikoshilang.library.wl_plots2d import Compiled
    c = Compiled(ev("Hold[x^2]").args[0], [Symbol("x")], "Plot")
    assert c.fast is not None
    c.cross_check([(0.3,), (0.7,)])
    assert c.fast is not None                                  # agrees with the kernel
    c.fast = lambda a: np.asarray(a) * 0 + 5.0                 # a wrong compiled function
    c.cross_check([(0.3,)])
    assert c.fast is None                                      # discarded
    assert abs(float(c(np.array([0.5]))[0]) - 0.25) < 1e-12    # kernel values


def test_w16_iso_surface_vertices_on_level():
    """Check: marching-tetrahedra vertices interpolate exactly to the level for a linear field."""
    import numpy as np
    from mikoshilang.library.wl_plots2d import grid3, iso_surface
    axes, X, Y, Z = grid3([(0, 1)] * 3, (5, 5, 5))
    P, tris = iso_surface(axes, X + 2 * Y - Z, 0.7)
    assert len(tris) > 0
    assert np.allclose(P[:, 0] + 2 * P[:, 1] - P[:, 2], 0.7)


def test_w16_curve_paths_reconnect_shuffled_circle():
    from mikoshilang.library.wl_plots2d import curve_paths
    import random
    pts = [(math.cos(2 * math.pi * k / 30), math.sin(2 * math.pi * k / 30)) for k in range(30)]
    sh = pts[:]
    random.Random(1).shuffle(sh)
    paths = curve_paths(sh)
    assert len(paths) == 1 and len(paths[0]) == 31
    for p, q in zip(paths[0], paths[0][1:]):
        assert math.dist(p, q) < 0.25                          # neighbours on the circle
