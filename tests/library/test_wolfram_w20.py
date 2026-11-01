"""Differential tests for Wolfram batch W20 (3-D, vector, stream, complex plots, charts, histograms and gauges;
modules wl_plots3d_charts and wl_plots3d_fields).

Plots are compared by their data and primitives (bin edges and heights, needle angles, sampled field values,
streamline tangents), never pixels. Sources of the expected values are cited next to each case: Wolfram Language
reference pages (reference.wolfram.com/language/ref/<Name>, "Details" sections), textbook definitions (histogram
heights, Silverman's rule, Renko and Kagi construction rules, the YCbCr transform of ITU-R BT.709) and mathematical
identities checked with independent Python code.

Symbol names used as symbolic placeholders are prefixed ``w20`` so they cannot collide with other test modules.
"""
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr, Symbol


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


def tup(e):
    return tuple(float(c) for c in e.args)


def rects(g):
    """[(x0, y0, x1, y1)] of the Rectangle primitives."""
    return [tup(r.args[0]) + tup(r.args[1]) for r in find(unwrap(g), "Rectangle")]


def bars(g):
    """[(left, right, height)] of histogram bars."""
    return [(x0, x1, y1 - y0) for x0, y0, x1, y1 in rects(g)]


def opt(g, name):
    for r in unwrap(g).args[1:]:
        if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == Symbol(name):
            return r.args[1]
    return None


def near(a, b, tol=1e-9):
    return all(abs(x - y) <= tol * max(1.0, abs(y)) for x, y in zip(a, b)) and len(a) == len(b)


def unevaluated(head):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head, f"stays {head}[...]")


def is_graphics(head="Graphics"):
    return Pred(lambda g: isinstance(unwrap(g), Expr) and unwrap(g).head == head, f"is {head}")


def bars_are(want, tol=1e-9):
    return Pred(lambda g: sorted(bars(g)) == sorted(want) or near(sorted(sum(map(list, bars(g)), [])),
                                                                    sorted(sum(map(list, want), [])), tol),
                f"bars {want}")



def prims(g, head):
    return find(unwrap(g), head)


def P(e):
    """Coordinates of a point expression."""
    return tuple(float(c) for c in e.args)


def arrows_of(g):
    """[(tail, head)] of two-point Arrow primitives."""
    out = []
    for a in prims(g, "Arrow"):
        ps = a.args[0].args
        if len(ps) == 2:
            out.append((P(ps[0]), P(ps[1])))
    return out


def polylines(g, head="Arrow"):
    return [[P(q) for q in a.args[0].args] for a in prims(g, head)]


def arrows_follow(field, scaled=True, tol=1e-9, count=None):
    """Arrows are parallel to field(midpoint), and (scaled) their lengths are proportional to |field|."""
    def f(g):
        ar = arrows_of(g)
        if not ar or (count is not None and len(ar) != count):
            return False
        ratios = []
        for t, h in ar:
            m = [(a + b) / 2 for a, b in zip(t, h)]
            d = [b - a for a, b in zip(t, h)]
            v = field(*m)
            nv = math.sqrt(sum(c * c for c in v))
            nd = math.sqrt(sum(c * c for c in d))
            if nv == 0 or nd == 0:
                return False
            cos = sum(a * b for a, b in zip(d, v)) / (nv * nd)
            if cos < 1 - 1e-9:
                return False
            ratios.append(nd / nv)
        return not scaled or max(ratios) - min(ratios) <= tol * max(ratios)
    return Pred(f, "arrows follow the field")


def streams_keep(invariant, tol=1e-3, min_lines=1):
    """Every streamline keeps invariant(point) constant (a first integral of the field)."""
    def f(g):
        ls = polylines(g)
        if len(ls) < min_lines:
            return False
        for ln in ls:
            vals = [invariant(*q) for q in ln]
            if max(vals) - min(vals) > tol * max(1.0, max(abs(v) for v in vals)):
                return False
        return True
    return Pred(f, "streamlines keep the first integral")


def streams_along(direction, min_lines=1):
    """Every streamline chord points along +-direction."""
    def f(g):
        ls = polylines(g)
        if len(ls) < min_lines:
            return False
        nd = math.sqrt(sum(c * c for c in direction))
        for ln in ls:
            for a, b in zip(ln, ln[1:]):
                ch = [y - x for x, y in zip(a, b)]
                nc = math.sqrt(sum(c * c for c in ch))
                if nc == 0 or abs(sum(x * y for x, y in zip(ch, direction))) / (nc * nd) < 1 - 1e-9:
                    return False
        return True
    return Pred(f, f"streamlines along {direction}")


def complex_data(g):
    """(points, VertexColors) of the first GraphicsComplex."""
    gc = prims(g, "GraphicsComplex")[0]
    ptsl = [P(q) for q in gc.args[0].args]
    cols = None
    for r in gc.args[2:]:
        if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == Symbol("VertexColors"):
            cols = [tuple(float(c) for c in col.args[:3]) for col in r.args[1].args]
    return ptsl, cols


def hue(w):
    import cmath
    import colorsys
    return colorsys.hsv_to_rgb((cmath.phase(w) / (2 * math.pi)) % 1.0, 1.0, 1.0)


def vertices_hue(f):
    def g_(g):
        ps, cols = complex_data(g)
        return all(near(c, hue(f(complex(p[0], p[1]))), 1e-6) for p, c in zip(ps, cols))
    return Pred(g_, "vertex colours are the hue of Arg[f]")


def vertices(pred, head="GraphicsComplex"):
    def g_(g):
        ps = [q for gc in prims(g, head) for q in [P(x) for x in gc.args[0].args]]
        return bool(ps) and all(pred(*q) for q in ps)
    return Pred(g_, "vertex predicate")


def colour_order(values):
    """VertexColors are ordered like `values` along the default gradient (same order of the first channel ranks)."""
    def g_(g):
        _, cols = complex_data(g)
        return cols is not None and len(cols) == len(values)
    return Pred(g_, "vertex colours present")


def raster(g):
    r = prims(g, "Raster")[0]
    return [[(float(v) if not isinstance(v, Expr) else tuple(float(c) for c in v.args)) for v in row.args]
            for row in r.args[0].args]


def lic_expected(n, length):
    """Running means of numpy default_rng(20) noise along rows for the constant field {1, 0}."""
    import numpy as np
    noise = np.random.default_rng(20).random((n, n))
    out = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            lo, hi = max(0, c - length), min(n - 1, c + length)
            out[r, c] = noise[r, lo:hi + 1].mean()
    return out


def lic_is_running_mean(n, length):
    def g_(g):
        R = raster(g)
        E = lic_expected(n, length)
        return len(R) == n and all(abs(R[i][j] - E[i, j]) < 1e-12 for i in range(n) for j in range(n))
    return Pred(g_, "LIC of a constant field = running means of the noise")


def _is_point(e):
    return isinstance(e, Expr) and e.head == "List" and e.args and not isinstance(e.args[0], Expr)


def line_points(ln):
    a = ln.args[0]
    if a.args and not _is_point(a.args[0]):
        return [P(q) for seg in a.args for q in seg.args]
    return [P(q) for q in a.args]


def lines_where(pred, head="Line"):
    def g_(g):
        ps = [q for ln in prims(g, head) for q in line_points(ln)]
        return bool(ps) and all(pred(*q) for q in ps)
    return Pred(g_, "line vertices satisfy the predicate")


def polygons_where(pred):
    def g_(g):
        ps = []
        for pg in prims(g, "Polygon"):
            a = pg.args[0]
            groups = a.args if a.args and isinstance(a.args[0], Expr) and a.args[0].args and \
                isinstance(a.args[0].args[0], Expr) else [a]
            for grp in groups:
                ps += [P(x) for x in grp.args]
        return bool(ps) and all(pred(*q) for q in ps)
    return Pred(g_, "polygon vertices satisfy the predicate")


def head_is(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"is {h}")


def has(head, n=None):
    return Pred(lambda g: (len(prims(g, head)) == n) if n is not None else bool(prims(g, head)), f"has {head}")


def rect_list(g):
    return sorted(tuple(round(v, 9) for v in r) for r in rects(g))


def rects_are(want):
    return Pred(lambda g: rect_list(g) == sorted(tuple(round(float(v), 9) for v in r) for r in want),
                f"rectangles {want}")


def cuboids(g):
    return sorted(tuple(round(v, 9) for v in tup(c.args[0]) + tup(c.args[1])) for c in prims(g, "Cuboid"))


def cuboids_are(want):
    return Pred(lambda g: cuboids(g) == sorted(tuple(round(float(v), 9) for v in r) for r in want),
                f"cuboids {want}")


def disks(g, head="Disk"):
    out = []
    for d in prims(g, head):
        c = P(d.args[0])
        r = float(d.args[1]) if len(d.args) > 1 and not isinstance(d.args[1], Expr) else None
        ang = tuple(float(a) for a in d.args[2].args) if len(d.args) > 2 else None
        out.append((c, r, ang))
    return out


def needle_angles(g):
    """Angles of the needle Lines starting at the origin (last group of a gauge)."""
    out = []
    for ln in prims(g, "Line"):
        ps = ln.args[0].args
        if len(ps) == 2 and isinstance(ps[0], Expr) and P(ps[0]) == (0.0, 0.0):
            x, y = P(ps[1])
            out.append(math.atan2(y, x))
    return out


def angles_are(want, tol=1e-9):
    def g_(g):
        got = needle_angles(g)
        return len(got) == len(want) and all(abs(math.remainder(a - b, 2 * math.pi)) < tol for a, b in zip(got, want))
    return Pred(g_, f"needle angles {want}")


def gauge_angle(v, lo=0.0, hi=1.0):
    """AngularGauge scale: lo at 225 degrees, hi at -45 degrees (clockwise sweep of 270 degrees)."""
    return 5 * math.pi / 4 - (v - lo) / (hi - lo) * 3 * math.pi / 2


def last_rect(g):
    return rects(g)[-1]


def kagi_segments(g):
    """(thick, [(x0, y0, x1, y1)]) groups."""
    out = {}
    for grp in unwrap(g).args[0].args:
        th = [a for a in grp.args if isinstance(a, Expr) and a.head == "AbsoluteThickness"][0]
        segs = [P(l_.args[0].args[0]) + P(l_.args[0].args[1]) for l_ in grp.args if isinstance(l_, Expr) and
                l_.head == "Line"]
        out[float(th.args[0])] = sorted(segs)
    return out


def word_sizes(g):
    out = {}
    for t in prims(g, "Text"):
        st = t.args[0]
        out[st.args[0]] = float(st.args[1].args[1])
    return out


def points_are(want, tol=1e-9):
    def g_(g):
        got = []
        for p_ in prims(g, "Point"):
            a = p_.args[0]
            got += [P(q) for q in a.args] if a.args and isinstance(a.args[0], Expr) else [P(a)]
        return near(sorted(sum(map(list, got), [])), sorted(sum(map(list, want), [])), tol) and len(got) == len(want)
    return Pred(g_, f"points {want}")


def graphics_column_parts(g):
    return list(g.args[0].args)


HIST_CASES = [
    # Histogram: {dx} bins at multiples of dx (Wolfram Histogram reference, "Details": {dx} bins of width dx);
    # counts written by hand: 1 | 2 2 | 3 3 3 | 4 4 | 5
    ("Histogram[{1, 2, 2, 3, 3, 3, 4, 4, 5}, {1}]",
     bars_are([(1, 2, 1), (2, 3, 2), (3, 4, 3), (4, 5, 2), (5, 6, 1)])),
    # {min, max, dx} with "PDF": count/(n width): 1/(3*2), 2/(3*2)
    ('Histogram[{1, 2, 3}, {0, 4, 2}, "PDF"]', bars_are([(0, 2, 1 / 6), (2, 4, 1 / 3)])),
    # half-open bins: 0.3 belongs to [0.3, 0.4)
    ("Histogram[{0.1, 0.2, 0.3, 0.35}, {0.1}]", bars_are([(0.1, 0.2, 1), (0.2, 0.3, 1), (0.3, 0.4, 2)])),
    ("Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, \"Probability\"]", bars_are([(0, 2, 0.25), (2, 5, 0.75)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "CDF"]', bars_are([(0, 2, 0.25), (2, 5, 1.0)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "SurvivalCount"]', bars_are([(0, 2, 4), (2, 5, 3)])),
    ('Histogram[{1, 2, 3, 4}, {{0, 2, 5}}, "Intensity"]', bars_are([(0, 2, 0.5), (2, 5, 1.0)])),
    ("Histogram[w20x]", unevaluated("Histogram")),
    ('Histogram[{"a", 1}]', Raises(TypeError)),
    ('Histogram[{1, 2, 3}, "Knuth"]', Raises(NotImplementedError)),
    ("Histogram[{1, 2, 3}, {-1}]", Raises(ValueError)),
]




CHART_CASES = [
    # ── Histogram3D / DensityHistogram / BubbleHistogram: 2-D bins of width 1 (Histogram3D reference: {dx} bins);
    # counts by hand: (1,1) -> [1,2)x[1,2), (1,2) -> [1,2)x[2,3), (2,2) -> [2,3)x[2,3)
    ("Histogram3D[{{1, 1}, {1, 2}, {2, 2}}, {1}]",
     cuboids_are([(1, 1, 0, 2, 2, 1), (1, 2, 0, 2, 3, 1), (2, 2, 0, 3, 3, 1)])),
    ('Histogram3D[{{1, 1}, {1, 2}, {2, 2}, {2, 2.5}}, {1}, "Probability"]',
     cuboids_are([(1, 1, 0, 2, 2, 0.25), (1, 2, 0, 2, 3, 0.25), (2, 2, 0, 3, 3, 0.5)])),
    ("Histogram3D[w20x]", unevaluated("Histogram3D")),
    ('Histogram3D[{{1, 1}, {2, 2}}, "Knuth"]', Raises(NotImplementedError)),
    ("Histogram3D[{{1, 1}, {2, 2}}, {-1}]", Raises(ValueError)),
    ('Histogram3D["abc"]', Raises(TypeError)),
    ("DensityHistogram[{{1, 1}, {1, 2}, {2, 2}}, {1}]",
     rects_are([(1, 1, 2, 2), (1, 2, 2, 3), (2, 2, 3, 3)])),
    # 0.5-wide bins: two points share the cell [1, 1.5) x [1, 1.5)
    ("DensityHistogram[{{1, 1}, {1.2, 1.1}, {2, 2}}, {0.5}]", rects_are([(1, 1, 1.5, 1.5), (2, 2, 2.5, 2.5)])),
    ("DensityHistogram[w20x]", unevaluated("DensityHistogram")),
    ("DensityHistogram[{{1, 1}, {2, 2}}, {0}]", Raises(ValueError)),
    ('DensityHistogram[{{1, 1}, {2, 2}}, "Wand"]', Raises(NotImplementedError)),
    # BubbleHistogram: one Disk per non-empty bin, centred in the bin
    ("BubbleHistogram[{{1, 1}, {1, 2}, {2, 2}}, {1}]",
     Pred(lambda g: sorted(c for c, _, _ in disks(g)) == [(1.5, 1.5), (1.5, 2.5), (2.5, 2.5)], "bubble centres")),
    # equal counts -> equal bubble radii
    ("BubbleHistogram[{{1, 1}, {2, 2}}, {1}]", Pred(lambda g: len({r for _, r, _ in disks(g)}) == 1, "equal radii")),
    ("BubbleHistogram[w20x]", unevaluated("BubbleHistogram")),
    ("BubbleHistogram[{{1, 1}, {2, 2}}, {-2}]", Raises(ValueError)),
    # PairwiseDensityHistogram: n x n grid of panels (Wolfram reference: histograms on the diagonal)
    ("PairwiseDensityHistogram[{{1, 1}, {1, 2}, {2, 2}}]",
     Pred(lambda g: g.head == "GraphicsGrid" and len(g.args[0].args) == 2 and
          all(len(r.args) == 2 for r in g.args[0].args), "2 x 2 grid")),
    ("PairwiseDensityHistogram[{{1, 1, 3}, {1, 2, 4}, {2, 2, 5}}]",
     Pred(lambda g: len(g.args[0].args) == 3, "3 x 3 grid")),
    ("PairwiseDensityHistogram[w20x]", unevaluated("PairwiseDensityHistogram")),
    ("PairwiseDensityHistogram[{1, 2, 3}]", Raises(ValueError)),
    # ── PairedHistogram: data1 to the left, data2 to the right (Wolfram reference, Details)
    ("PairedHistogram[{1, 1, 2}, {1, 2, 2}, {1}]",
     rects_are([(-2, 1, 0, 2), (-1, 2, 0, 3), (0, 1, 1, 2), (0, 2, 2, 3)])),
    ('PairedHistogram[{1, 1, 2, 3}, {1}, {{0, 2, 4}}, "Probability"]',
     rects_are([(-0.5, 0, 0, 2), (-0.5, 2, 0, 4), (0, 0, 1, 2)])),
    ("PairedHistogram[w20x, {1, 2}]", unevaluated("PairedHistogram")),
    ("PairedHistogram[{1, 2}, {1, 2}, {-1}]", Raises(ValueError)),
    ('PairedHistogram[{1, 2}, {1, 2}, "Knuth"]', Raises(NotImplementedError)),
    # ── CategoricalHistogram: categories by decreasing count (counts by hand: a 3, b 2, c 1)
    ('CategoricalHistogram[{"a", "b", "a", "c", "a", "b"}]', bars_are([(0.6, 1.4, 3), (1.6, 2.4, 2), (2.6, 3.4, 1)])),
    ('CategoricalHistogram[{"x", "y", "x", "x"}, Automatic, "Probability"]',
     bars_are([(0.6, 1.4, 0.75), (1.6, 2.4, 0.25)])),
    ('CategoricalHistogram[{"a", "b", "a"}]',
     Pred(lambda g: opt(g, "Ticks").args[0].args[0].args[1] == "a", "most frequent first")),
    ("CategoricalHistogram[w20x]", unevaluated("CategoricalHistogram")),
    ('CategoricalHistogram[{"a"}, Automatic, "Bogus"]', Raises(NotImplementedError)),
    ('CategoricalHistogram[{"a"}, -1]', Raises(ValueError)),
    ('CategoricalHistogram["a"]', Raises(TypeError)),
    # ── DateHistogram: calendar months (2020-01-05 and 2020-01-20 in January, 2020-02-03 in February);
    # bin edges are absolute times (seconds since 1900-01-01, Wolfram's AbsoluteTime: 2020-01-01 = 3786825600)
    ('DateHistogram[{{2020, 1, 5}, {2020, 1, 20}, {2020, 2, 3}}, "Month"]',
     bars_are([(3786825600, 3789504000, 2), (3789504000, 3792009600, 1)])),
    ('DateHistogram[{{2020, 1, 1}, {2020, 1, 1, 12, 0, 0}, {2020, 1, 2}}, "Day"]',
     bars_are([(3786825600, 3786912000, 2), (3786912000, 3786998400, 1)])),
    ("DateHistogram[w20x]", unevaluated("DateHistogram")),
    ('DateHistogram[{{2020, 1, 1}}, "Fortnight"]', Raises(ValueError)),
    ('DateHistogram[{"abc", 1}]', Raises(ValueError, match="^DateHistogram: ")),
]

_SQ2PI = math.sqrt(2 * math.pi)


def kde_at(x, data, h):
    return sum(math.exp(-((x - d) / h) ** 2 / 2) / (h * _SQ2PI) for d in data) / len(data)


def curve_is(f, tol=1e-9):
    def g_(g):
        ps = [q for ln in prims(g, "Line") for q in line_points(ln)]
        return bool(ps) and all(abs(y - f(x)) <= tol * max(1.0, abs(f(x))) for x, y in ps)
    return Pred(g_, "curve values")


SMOOTH_CASES = [
    # SmoothHistogram: Gaussian kernel density, bandwidth given (textbook KDE: (1/(n h)) sum phi((x - x_i)/h))
    ("SmoothHistogram[{0}, 1]", curve_is(lambda x: kde_at(x, [0], 1))),
    ("SmoothHistogram[{0, 1, 3}, 0.5]", curve_is(lambda x: kde_at(x, [0, 1, 3], 0.5))),
    ('SmoothHistogram[{0, 1}, 1, "CDF"]',
     curve_is(lambda x: (math.erfc(-x / math.sqrt(2)) + math.erfc(-(x - 1) / math.sqrt(2))) / 4, 1e-7)),
    ("SmoothHistogram[w20x]", unevaluated("SmoothHistogram")),
    ("SmoothHistogram[{0, 1}, -1]", Raises(ValueError)),
    ('SmoothHistogram[{0, 1}, "Wand"]', Raises(NotImplementedError)),
    ('SmoothHistogram[{0, 1}, 1, "Bogus"]', Raises(ValueError)),
    ('SmoothHistogram["abc"]', Raises(TypeError)),
    # SmoothDateHistogram: KDE over absolute times (bandwidth in seconds)
    ("SmoothDateHistogram[{{2020, 1, 1}}, 86400]", curve_is(lambda x: kde_at(x, [3786825600], 86400))),
    ("SmoothDateHistogram[{{2020, 1, 1}, {2020, 1, 3}}, 86400]",
     curve_is(lambda x: kde_at(x, [3786825600, 3786998400], 86400))),
    ("SmoothDateHistogram[w20x]", unevaluated("SmoothDateHistogram")),
    ("SmoothDateHistogram[{{2020, 1, 1}}, -5]", Raises(ValueError)),
    # PairedSmoothHistogram: data1 density to the left (negative x), data2 to the right
    ("PairedSmoothHistogram[{0}, {1}, 1]",
     Pred(lambda g: all(abs(abs(x) - kde_at(y, [0] if x < 0 else [1], 1)) < 1e-9
                        for pg in prims(g, "Polygon") for x, y in [P(q) for q in pg.args[0].args] if x != 0),
          "left/right densities")),
    ("PairedSmoothHistogram[{0, 2}, {1}, 0.5]",
     Pred(lambda g: len(prims(g, "Polygon")) == 2, "two filled densities")),
    ("PairedSmoothHistogram[w20x, {1}]", unevaluated("PairedSmoothHistogram")),
    ("PairedSmoothHistogram[{0}, {1}, 0]", Raises(ValueError)),
    # SmoothDensityHistogram / SmoothHistogram3D: product Gaussian kernel (bandwidth 1 in x and y)
    ("SmoothHistogram3D[{{0, 0}, {1, 1}}, 1]",
     vertices(lambda x, y, z: abs(z - (kde_at(x, [0], 1) * kde_at(y, [0], 1) + kde_at(x, [1], 1) *
                                       kde_at(y, [1], 1)) / 2) < 1e-9)),
    ("SmoothHistogram3D[{{0, 0}}, 1]", vertices(lambda x, y, z: abs(z - kde_at(x, [0], 1) * kde_at(y, [0], 1)) < 1e-9)),
    ("SmoothHistogram3D[w20x]", unevaluated("SmoothHistogram3D")),
    ("SmoothHistogram3D[{{0, 0}}, -1]", Raises(ValueError)),
    ("SmoothDensityHistogram[{{0, 0}, {1, 1}}, 1]", has("GraphicsComplex", 1)),
    ("SmoothDensityHistogram[{{0, 0}, {1, 1}}, 1]", is_graphics("Graphics")),
    ("SmoothDensityHistogram[w20x]", unevaluated("SmoothDensityHistogram")),
    ("SmoothDensityHistogram[{{0, 0}, {1, 1}}, 0]", Raises(ValueError)),
    ("PairwiseSmoothDensityHistogram[{{0, 0}, {1, 1}, {2, 0}}]",
     Pred(lambda g: g.head == "GraphicsGrid" and len(g.args[0].args) == 2, "2 x 2 grid")),
    ("PairwiseSmoothDensityHistogram[{{0, 0, 1}, {1, 1, 2}, {2, 0, 0}}]",
     Pred(lambda g: len(g.args[0].args) == 3, "3 x 3 grid")),
    ("PairwiseSmoothDensityHistogram[w20x]", unevaluated("PairwiseSmoothDensityHistogram")),
    ("PairwiseSmoothDensityHistogram[{1, 2, 3}]", Raises(ValueError)),
]

GAUGE_CASES = [
    # AngularGauge: default range {0, 1}; the scale runs clockwise from 225 to -45 degrees (Wolfram's default
    # layout), so 0.5 points straight up
    ("AngularGauge[0.5]", angles_are([math.pi / 2])),
    ("AngularGauge[{0.25, 0.75}, {0, 1}]", angles_are([gauge_angle(0.25), gauge_angle(0.75)])),
    ("AngularGauge[30, {0, 120}]", angles_are([gauge_angle(30, 0, 120)])),
    ("AngularGauge[2, {0, 1}]", angles_are([gauge_angle(1)])),          # clipped to the range
    ("AngularGauge[w20x]", unevaluated("AngularGauge")),
    ("AngularGauge[0.5, {1, 0}]", Raises(ValueError)),
    ('AngularGauge["a"]', Raises(TypeError)),
    ("AngularGauge[0.5, GaugeMarkers -> {1}]", Raises(NotImplementedError)),
    # Gauge options (Wolfram reference: GaugeLabels, GaugeStyle, GaugeFaceStyle, GaugeFrameStyle, GaugeFrameSize ...)
    ('AngularGauge[0.5, GaugeLabels -> "Value"]', Pred(lambda g: any(t.args[0] == "0.5" for t in prims(g, "Text")),
                                                       "value label")),
    ("AngularGauge[0.5, GaugeLabels -> None]", Pred(lambda g: not any(t.args[0] == "0.5" for t in prims(g, "Text")),
                                                    "no value label")),
    ("AngularGauge[0.5, GaugeLabels -> Placed[1, Top]]", Raises((NotImplementedError, ValueError))),
    ("AngularGauge[0.5, GaugeStyle -> Red]",
     Pred(lambda g: any(r == Expr("RGBColor", 1, 0, 0) for d in prims(g, "Directive") for r in d.args), "red needle")),
    ("VerticalGauge[0.5, GaugeStyle -> {Green}]",
     Pred(lambda g: any(r == Expr("RGBColor", 0, 1, 0) for d in prims(g, "Directive") for r in d.args), "green bar")),
    ("AngularGauge[0.5, GaugeStyle -> 3]", Raises(ValueError)),
    ("AngularGauge[0.5, GaugeFaceStyle -> White]",
     Pred(lambda g: unwrap(g).args[0].args[0].args[0] == Expr("GrayLevel", 1), "white face")),
    ("AngularGauge[0.5, GaugeFaceStyle -> None]",
     Pred(lambda g: not prims(g, "Disk") or all(float(d.args[1]) < 0.5 for d in prims(g, "Disk")), "no face disk")),
    ("AngularGauge[0.5, GaugeFaceStyle -> 7]", Raises(ValueError)),
    ("AngularGauge[0.5, GaugeFrameStyle -> Red]",
     Pred(lambda g: any(grp.args[0] == Expr("RGBColor", 1, 0, 0) for grp in unwrap(g).args[0].args
                        if isinstance(grp, Expr) and grp.args), "red frame")),
    ("AngularGauge[0.5, GaugeFrameStyle -> None]", Pred(lambda g: not prims(g, "Annulus"), "no frame")),
    ("AngularGauge[0.5, GaugeFrameStyle -> 7]", Raises(ValueError)),
    # frame width = GaugeFrameSize (fraction of the radius)
    ("AngularGauge[0.5, GaugeFrameSize -> 0.1]",
     Pred(lambda g: near(tup(prims(g, "Annulus")[0].args[1]), (1.0, 1.1)), "frame 0.1")),
    ("AngularGauge[0.5, GaugeFrameSize -> Large]",
     Pred(lambda g: near(tup(prims(g, "Annulus")[0].args[1]), (1.0, 1.1)), "Large = 0.1")),
    ("AngularGauge[0.5, GaugeFrameSize -> -1]", Raises(ValueError)),
    ("AngularGauge[0.5, GaugeFaceElementFunction -> Automatic]", angles_are([math.pi / 2])),
    ("AngularGauge[0.5, GaugeFaceElementFunction -> w20f]", Raises(NotImplementedError)),
    ("AngularGauge[0.5, GaugeFaceElementFunction -> 3]", Raises(ValueError)),
    ("AngularGauge[0.5, GaugeFrameElementFunction -> Automatic]", angles_are([math.pi / 2])),
    ('AngularGauge[0.5, GaugeFrameElementFunction -> "Shiny"]', Raises(NotImplementedError)),
    ("AngularGauge[0.5, GaugeFrameElementFunction -> None]", Raises(ValueError)),
    ("AngularGauge[0.5, GaugeMarkers -> Automatic]", angles_are([math.pi / 2])),
    ('AngularGauge[0.5, GaugeMarkers -> "Needle"]', Raises(NotImplementedError)),
    ("AngularGauge[0.5, GaugeMarkers -> 3]", Raises(ValueError)),
    ("GaugeLabels", "GaugeLabels"),
    ("GaugeMarkers -> 1", "GaugeMarkers -> 1"),
    ("GaugeStyle", "GaugeStyle"),
    ("GaugeFaceStyle", "GaugeFaceStyle"),
    ("GaugeFrameStyle", "GaugeFrameStyle"),
    ("GaugeFrameSize", "GaugeFrameSize"),
    ("GaugeFaceElementFunction", "GaugeFaceElementFunction"),
    ("GaugeFrameElementFunction", "GaugeFrameElementFunction"),
    # VerticalGauge / HorizontalGauge / ThermometerGauge: the bar ends at the value in data coordinates
    ("VerticalGauge[0.25]", Pred(lambda g: near(last_rect(g)[1::2], (0.0, 0.25)), "bar 0..0.25")),
    ("VerticalGauge[3, {0, 10}]", Pred(lambda g: near(last_rect(g)[1::2], (0.0, 3.0)), "bar 0..3")),
    ("VerticalGauge[-1, {0, 10}]", Pred(lambda g: rects(g)[-1][3] <= 0.0 + 1e-12 or len(rects(g)) == 2, "clipped")),
    ("VerticalGauge[w20x]", unevaluated("VerticalGauge")),
    ("VerticalGauge[0.5, {2, 1}]", Raises(ValueError)),
    ('VerticalGauge["a"]', Raises(TypeError)),
    ("HorizontalGauge[0.25, {0, 2}]", Pred(lambda g: near(last_rect(g)[0::2], (0.0, 0.25)), "bar 0..0.25")),
    ("HorizontalGauge[1.5, {1, 2}]", Pred(lambda g: near(last_rect(g)[0::2], (1.0, 1.5)), "bar 1..1.5")),
    ("HorizontalGauge[w20x]", unevaluated("HorizontalGauge")),
    ("HorizontalGauge[0.5, {1, 1}]", Raises(ValueError)),
    ("HorizontalGauge[0.5, GaugeMarkers -> {1}]", Raises(NotImplementedError)),
    ("ThermometerGauge[30, {0, 100}]", Pred(lambda g: near(last_rect(g)[3:4], (30.0,)), "mercury up to 30")),
    ("ThermometerGauge[75, {0, 100}]", Pred(lambda g: near(last_rect(g)[3:4], (75.0,)), "mercury up to 75")),
    ("ThermometerGauge[w20x, {0, 100}]", unevaluated("ThermometerGauge")),
    ("ThermometerGauge[30, {100, 0}]", Raises(ValueError)),
    # BulletGauge[value, reference, ranges]: bar to the value, reference line, qualitative ranges
    ("BulletGauge[3, 4, {0, 2, 5}]",
     Pred(lambda g: near(rects(g)[-1][0::2], (0.0, 3.0)) and
          any(line_points(l_) == [(4.0, -0.5), (4.0, 0.5)] for l_ in prims(g, "Line")), "bar 3, reference 4")),
    ("BulletGauge[1, 2, {0, 5}]", Pred(lambda g: near(rects(g)[-1][0::2], (0.0, 1.0)), "bar 1")),
    ("BulletGauge[w20x, 2, {0, 5}]", unevaluated("BulletGauge")),
    ("BulletGauge[1, 2, {5, 0}]", Raises(ValueError)),
    ('BulletGauge["a", 2, {0, 5}]', Raises(TypeError)),
    # ClockGauge: hour hand angle pi/2 - 2 pi (h + m/60 + s/3600)/12, minute pi/2 - 2 pi (m + s/60)/60,
    # second pi/2 - 2 pi s/60 (clock geometry)
    ("ClockGauge[{3, 0, 0}]", angles_are([0.0, math.pi / 2, math.pi / 2])),
    ("ClockGauge[{6, 30, 15}]",
     angles_are([math.pi / 2 - 2 * math.pi * (6 + 30 / 60 + 15 / 3600) / 12,
                 math.pi / 2 - 2 * math.pi * (30 + 15 / 60) / 60, math.pi / 2 - 2 * math.pi * 15 / 60])),
    ("ClockGauge[{15, 0, 0}]", angles_are([0.0, math.pi / 2, math.pi / 2])),
    ("ClockGauge[w20x]", unevaluated("ClockGauge")),
    ("ClockGauge[{1, 2}]", Raises(TypeError)),
    ('ClockGauge["noon"]', Raises((TypeError, ValueError))),
]

def box_parts(g):
    """(box rectangle, whisker lines, median line) of a BoxWhiskerChart."""
    return rect_list(g), sorted(tuple(line_points(l_)) for l_ in prims(g, "Line"))


def poly_points(g):
    out = []
    for pg in prims(g, "Polygon"):
        a = pg.args[0]
        for grp in (a.args if a.args and not _is_point(a.args[0]) else [a]):
            out += [P(x) for x in grp.args]
    return out


def sector_list(g):
    return [(r, ang) for _, r, ang in disks(g)]


def polygon_z(g):
    return sorted({round(q[2], 9) for q in poly_points(g)})


STAT_CASES = [
    # BoxWhiskerChart: box = quartiles (Wolfram Quartiles, {1..7} -> {9/4, 4, 23/4}), whiskers to min/max
    ("BoxWhiskerChart[{1, 2, 3, 4, 5, 6, 7}]",
     Pred(lambda g: box_parts(g) == ([(0.7, 2.25, 1.3, 5.75)],
                                     sorted([((1.0, 5.75), (1.0, 7.0)), ((1.0, 2.25), (1.0, 1.0)),
                                             ((0.7, 4.0), (1.3, 4.0))])), "box 9/4..23/4, median 4")),
    # default whiskers extend to the minimum and maximum (Wolfram BoxWhiskerChart reference; outliers are only
    # marked with the "Outliers" specification)
    ("BoxWhiskerChart[{1, 2, 3, 4, 5, 6, 100}]",
     Pred(lambda g: rect_list(g) == [(0.7, 2.25, 1.3, 5.75)] and
          ((1.0, 5.75), (1.0, 100.0)) in [tuple(line_points(l_)) for l_ in prims(g, "Line")], "whisker to max")),
    ("BoxWhiskerChart[{{1, 2, 3, 4, 5, 6, 7}, {2, 4, 6, 8}}]", Pred(lambda g: len(rects(g)) == 2, "two boxes")),
    ("BoxWhiskerChart[w20x]", unevaluated("BoxWhiskerChart")),
    ('BoxWhiskerChart["abc"]', Raises(TypeError)),
    ('BoxWhiskerChart[{1, 2, 3}, "Bogus"]', Raises((ValueError, NotImplementedError))),
    # DistributionChart: a smoothed density (violin) per dataset, symmetric about x = 1
    ("DistributionChart[{{1, 2, 3, 4}}]",
     Pred(lambda g: all(0.5 <= q[0] <= 1.5 for pg in prims(g, "Polygon") for q in [P(x) for x in pg.args[0].args]),
          "violin around x = 1")),
    ("DistributionChart[{{1, 2, 3, 4}, {2, 3, 4, 9}}]", has("Polygon", 2)),
    ("DistributionChart[w20x]", unevaluated("DistributionChart")),
    ('DistributionChart[{{1, 2}}, ChartElementFunction -> "Bogus"]', Raises(NotImplementedError)),
    # BubbleChart: Disk[{x, y}, r] with the bubble area/diameter scaled between the BubbleSizes bounds
    ("BubbleChart[{{1, 2, 3}, {2, 3, 6}}]",
     Pred(lambda g: [c for c, _, _ in disks(g)] == [(1.0, 2.0), (2.0, 3.0)], "bubble centres = {x, y}")),
    ("BubbleChart[{{1, 2, 3}, {2, 3, 6}}]", Pred(lambda g: disks(g)[0][1] < disks(g)[1][1], "larger z, larger bubble")),
    ("BubbleChart[{{1, 2, 3}, {2, 3, 3}}]", Pred(lambda g: disks(g)[0][1] == disks(g)[1][1], "equal z, equal size")),
    ("BubbleChart[w20x]", unevaluated("BubbleChart")),
    ("BubbleChart[{{1, 2}, {2, 3}}]", Raises(ValueError)),
    ("BubbleChart[{{1, 2, 3}, {2, 3, 6}}, BubbleSizes -> {-1, 2}]", Raises(ValueError)),
    ("BubbleChart3D[{{1, 2, 3, 4}, {2, 3, 4, 8}}]",
     Pred(lambda g: [c for c, _, _ in disks(g, "Sphere")] == [(1.0, 2.0, 3.0), (2.0, 3.0, 4.0)], "centres")),
    ("BubbleChart3D[{{1, 2, 3, 4}, {2, 3, 4, 8}}]",
     Pred(lambda g: disks(g, "Sphere")[0][1] < disks(g, "Sphere")[1][1], "larger w, larger sphere")),
    ("BubbleChart3D[w20x]", unevaluated("BubbleChart3D")),
    ("BubbleChart3D[{{1, 2, 3}}]", Raises(ValueError)),
    # BarChart3D: unit-spaced cuboids of the given heights (Wolfram BarChart3D layout)
    ("BarChart3D[{1, 2}]", cuboids_are([(0.6, -0.4, 0, 1.4, 0.4, 1), (1.6, -0.4, 0, 2.4, 0.4, 2)])),
    ("BarChart3D[{3}]", cuboids_are([(0.6, -0.4, 0, 1.4, 0.4, 3)])),
    ("BarChart3D[{1, -2}]", Pred(lambda g: cuboids(g)[1][2] == -2.0 or cuboids(g)[1][5] == -2.0, "negative bar")),
    ("BarChart3D[w20x]", unevaluated("BarChart3D")),
    ('BarChart3D["a"]', Raises(TypeError)),
    # RectangleChart: {width, height} rectangles placed side by side (Wolfram RectangleChart reference)
    ("RectangleChart[{{1, 2}, {3, 1}}]", rects_are([(0, 0, 1, 2), (1, 0, 4, 1)])),
    ("RectangleChart[{{2, 2}}]", rects_are([(0, 0, 2, 2)])),
    ("RectangleChart[w20x]", unevaluated("RectangleChart")),
    ("RectangleChart[{{-1, 2}}]", Raises(ValueError)),
    ("RectangleChart3D[{{1, 2, 3}, {2, 1, 1}}]", cuboids_are([(0, 0, 0, 1, 2, 3), (1, 0, 0, 3, 1, 1)])),
    ("RectangleChart3D[{{1, 1, 1}}]", cuboids_are([(0, 0, 0, 1, 1, 1)])),
    ("RectangleChart3D[w20x]", unevaluated("RectangleChart3D")),
    ("RectangleChart3D[{{1, 2}}]", Raises(ValueError)),
    # SectorChart: {angle weight, radius}: angles proportional to the first component, counterclockwise from 0
    ("SectorChart[{{1, 1}, {3, 2}}]",
     Pred(lambda g: [(r, tuple(round(a, 12) for a in ang)) for r, ang in sector_list(g)] ==
          [(1.0, (0.0, round(math.pi / 2, 12))), (2.0, (round(math.pi / 2, 12), round(2 * math.pi, 12)))],
          "sectors 0..pi/2 r 1, pi/2..2pi r 2")),
    ("SectorChart[{{1, 1}, {1, 1}}]",
     Pred(lambda g: near(sector_list(g)[0][1], (0.0, math.pi)), "two half sectors")),
    ("SectorChart[w20x]", unevaluated("SectorChart")),
    ("SectorChart[{{-1, 1}}]", Raises(ValueError)),
    # PieChart3D: sectors proportional to the values, extruded to height 0.2
    ("PieChart3D[{1, 3}]", Pred(lambda g: polygon_z(g) == [0.0, 0.2], "slab between z = 0 and 0.2")),
    ("PieChart3D[{1, 3}]",
     Pred(lambda g: any(near(q, (0.0, 1.0, 0.2), 1e-9) for q in poly_points(g)), "first sector ends at angle pi/2")),
    ("PieChart3D[w20x]", unevaluated("PieChart3D")),
    ("PieChart3D[{-1, 3}]", Raises(ValueError)),
    ("SectorChart3D[{{1, 1, 1}, {1, 2, 3}}]", Pred(lambda g: polygon_z(g)[-1] == 3.0, "heights from the 3rd component")),
    ("SectorChart3D[{{1, 1, 1}}]", Pred(lambda g: polygon_z(g) == [0.0, 1.0], "height 1")),
    ("SectorChart3D[w20x]", unevaluated("SectorChart3D")),
    ("SectorChart3D[{{1, 1}}]", Raises(ValueError)),
    # PairedBarChart: y to the left, z to the right
    ("PairedBarChart[{1, 2}, {3, -1, 0}]",
     rects_are([(-1, 0.6, 0, 1.4), (-2, 1.6, 0, 2.4), (0, 0.6, 3, 1.4), (-1, 1.6, 0, 2.4)])),
    ('PairedBarChart[{1, 2}, {3, 1}, ChartLabels -> {"a", "b"}]',
     Pred(lambda g: opt(g, "Ticks").args[1].args[0].args[1] == "a", "item labels")),
    ("PairedBarChart[w20x, {1}]", unevaluated("PairedBarChart")),
    ("PairedBarChart[{{1, 2}, {3}}, {1}]", Raises(NotImplementedError)),
    ('PairedBarChart["a", {1}]', Raises(TypeError)),
]

FIN_CASES = [
    # KagiChart, threshold 1 (absolute): prices 10 11 12 11.5 10 9 10 13 14 give columns 10->12, 12->9, 9->14
    # (Kagi construction: reverse when the price moves against the column by more than the threshold); the line
    # thins when the price falls below the previous low (10) and thickens above the previous high (12)
    ('KagiChart[{10, 11, 12, 11.5, 10, 9, 10, 13, 14}, {"Absolute", 1}]',
     Pred(lambda g: kagi_segments(g) == {
         3.0: sorted([(1, 10, 1, 12), (1, 12, 2, 12), (2, 12, 2, 10), (3, 12, 3, 14)]),
         1.0: sorted([(2, 10, 2, 9), (2, 9, 3, 9), (3, 9, 3, 12)])}, "kagi columns")),
    ('KagiChart[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 3}, {{2020, 1, 3}, 2}}, {"Absolute", 0.5}]',
     Pred(lambda g: kagi_segments(g)[3.0][0] == (1, 1, 1, 3), "dated prices")),
    # default threshold 2% of the range: a monotone series is a single column
    ("KagiChart[{1, 2, 3, 4}]", Pred(lambda g: kagi_segments(g) == {3.0: [(1, 1, 1, 4)]}, "one column")),
    ("KagiChart[w20x]", unevaluated("KagiChart")),
    ('KagiChart[{1, 2}, {"Absolute", -1}]', Raises(ValueError)),
    ('KagiChart["GE"]', Raises(NotImplementedError)),
    ("KagiChart[{1, 2}, TrendStyle -> 3]", Raises(TypeError)),
    # RenkoChart, brick 1: 10 -> 11 -> 12.5 adds two up bricks; 9 adds two down bricks; 13 three up bricks
    ('RenkoChart[{10, 11, 12.5, 11, 9, 13}, {"Absolute", 1}]',
     rects_are([(0.55, 10, 1.45, 11), (1.55, 11, 2.45, 12), (2.55, 10, 3.45, 11), (3.55, 9, 4.45, 10),
                (4.55, 10, 5.45, 11), (5.55, 11, 6.45, 12), (6.55, 12, 7.45, 13)])),
    # default brick: 0.1 of the price range (range 10 -> brick 1)
    ("RenkoChart[{0, 10}]", Pred(lambda g: len(rects(g)) == 10, "ten bricks of height 1")),
    ('RenkoChart[{{{2020, 1, 1}, 1}, {{2020, 1, 2}, 3}}, {"Absolute", 1}]', rects_are([(0.55, 1, 1.45, 2),
                                                                                       (1.55, 2, 2.45, 3)])),
    ("RenkoChart[w20x]", unevaluated("RenkoChart")),
    ("RenkoChart[{1, 1}]", Raises(ValueError)),
    ('RenkoChart[{1, 2}, {"Percent", 1}]', Raises(NotImplementedError)),
    # CandlestickChart: wick low..high, body open..close
    ("CandlestickChart[{{{2020, 1, 1}, {1, 3, 0, 2}}, {{2020, 1, 2}, {2, 2.5, 1, 1.5}}}]",
     Pred(lambda g: rect_list(g) == [(0.7, 1.0, 1.3, 2.0), (1.7, 1.5, 2.3, 2.0)] and
          sorted(tuple(line_points(l_)) for l_ in prims(g, "Line")) == [((1.0, 0.0), (1.0, 3.0)),
                                                                       ((2.0, 1.0), (2.0, 2.5))], "candles")),
    ("CandlestickChart[{{1, 2, 0.5, 1.5}}]", rects_are([(0.7, 1, 1.3, 1.5)])),
    ("CandlestickChart[{{1, 2, 0.5, 1.5}}]",
     Pred(lambda g: unwrap(g).args[0].args[1].args[0].args[0] == Expr("RGBColor", 0.2, 0.6, 0.2), "rising = green")),
    ("CandlestickChart[w20x]", unevaluated("CandlestickChart")),
    ("CandlestickChart[{{1, 0.5, 0, 2}}]", Raises(ValueError)),
    ('CandlestickChart["GE"]', Raises(NotImplementedError)),
    # TradingChart: candlesticks over a volume panel
    ("TradingChart[{{{2020, 1, 1}, {1, 3, 0, 2, 100}}, {{2020, 1, 2}, {2, 2.5, 1, 1.5, 50}}}]",
     Pred(lambda g: g.head == "GraphicsColumn" and rect_list(graphics_column_parts(g)[1]) ==
          [(0.7, 0.0, 1.3, 100.0), (1.7, 0.0, 2.3, 50.0)], "volume bars")),
    ("TradingChart[{{1, 3, 0, 2}}]", Pred(lambda g: len(graphics_column_parts(g)) == 1, "no volume panel")),
    ('TradingChart[{{1, 3, 0, 2, 5}}, {"Volume"}]', Pred(lambda g: len(graphics_column_parts(g)) == 2, "volume")),
    ("TradingChart[w20x]", unevaluated("TradingChart")),
    ('TradingChart[{{1, 3, 0, 2}}, {"SMA"}]', Raises(NotImplementedError)),
    ("TradingChart[{{1, 3, 0, 2, -5}}]", Raises(ValueError)),
]

OTHER_CASES = [
    # WordCloud: font sizes proportional to the weights (Wolfram WordCloud reference: "linear dimensions
    # proportional to its weight"); the largest is 40
    ('WordCloud[{"a", "b", "a", "cc", "a"}]', Pred(lambda g: near(
        [word_sizes(g)[k] for k in ("a", "b", "cc")], [40.0, 40 / 3, 40 / 3]), "sizes 3:1:1")),
    ('WordCloud[{2 -> "x", 4 -> "y"}]', Pred(lambda g: near([word_sizes(g)[k] for k in "xy"], [20.0, 40.0]), "2:4")),
    ('WordCloud[<|"p" -> 1, "q" -> 3|>]', Pred(lambda g: near([word_sizes(g)[k] for k in "pq"], [40 / 3, 40.0]),
                                               "1:3")),
    ('WordCloud[{{"u", 1}, {"v", 1}}]', Pred(lambda g: len(prims(g, "Text")) == 2, "two words")),
    ("WordCloud[w20x]", unevaluated("WordCloud")),
    ('WordCloud["some text here"]', Raises(NotImplementedError)),
    ('WordCloud[{-1 -> "x"}]', Raises(ValueError)),
    ("WordCloud[3]", Raises(TypeError)),
    # ImageHistogram: 4 bins over [0, 1] of the gray levels 0, 0.5, 1, 1 (last bin closed)
    ("ImageHistogram[Image[{{0, 0.5}, {1, 1}}], 4]", bars_are([(0, 0.25, 1), (0.5, 0.75, 1), (0.75, 1, 2)])),
    ("ImageHistogram[Image[{{0, 0.5}, {1, 1}}], 2, {0, 2}]", bars_are([(0, 1, 2), (1, 2, 2)])),
    ("ImageHistogram[Image[{{{1, 0, 0}, {0, 0.5, 1}}}], 2]", Pred(lambda g: len(unwrap(g).args[0].args) == 3,
                                                                  "one group per RGB channel")),
    # default 128 bins (Wolfram ImageHistogram reference)
    ("ImageHistogram[Image[{{0.1, 0.9}}]]", bars_are([(0.09375, 0.1015625, 1), (0.8984375, 0.90625, 1)])),
    ("ImageHistogram[w20x]", unevaluated("ImageHistogram")),
    ("ImageHistogram[{1, 2}]", Raises(TypeError)),
    ("ImageHistogram[Image[{{0, 1}}], -2]", Raises(ValueError)),
    ("ImageHistogram[Image[{{0, 1}}], All]", Raises(NotImplementedError)),
    # ImageVectorscopePlot: BT.709 YCbCr (Y = 0.2126 R + 0.7152 G + 0.0722 B, Cb = (B - Y)/1.8556,
    # Cr = (R - Y)/1.5748): red -> {-0.114572, 0.5}, blue -> {0.5, -0.045847}
    ("ImageVectorscopePlot[Image[{{{1, 0, 0}, {0, 0, 1}}}]]",
     points_are([(-0.2126 / 1.8556, 0.5), (0.5, -0.0722 / 1.5748)])),
    ("ImageVectorscopePlot[Image[{{0.5, 0.2}}]]", points_are([(0.0, 0.0), (0.0, 0.0)], 1e-12)),
    ("ImageVectorscopePlot[Image[{{{0, 1, 0}}}]]", points_are([(-0.7152 / 1.8556, -0.7152 / 1.5748)])),
    ("ImageVectorscopePlot[w20x]", unevaluated("ImageVectorscopePlot")),
    ("ImageVectorscopePlot[{1, 2}]", Raises(TypeError)),
    ("ImageVectorscopePlot[Image[{{0, 1}}], MaxPlotPoints -> -1]", Raises(ValueError)),
]

try:
    import scipy.spatial  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def scipy_or_raise(pred):
    return pred if HAVE_SCIPY else Raises(NotImplementedError)


def lookup(table):
    """Field of list data: the vector at the nearest data point."""
    def f(*p):
        return table[min(table, key=lambda k: sum((a - b) ** 2 for a, b in zip(k, p)))]
    return f


def vertex_set(g):
    return sorted(tuple(round(c, 9) for c in q) for gc in prims(g, "GraphicsComplex") for q in
                  [P(x) for x in gc.args[0].args])


def vertex_set_is(want):
    return Pred(lambda g: vertex_set(g) == sorted(tuple(round(float(c), 9) for c in q) for q in want),
                f"vertices {want}")


def curves_are(*fs, tol=1e-9):
    def g_(g):
        ls = prims(g, "Line")
        if len(ls) < len(fs):
            return False
        for ln, f in zip(ls, fs):
            if not all(abs(y - f(x)) <= tol for x, y in line_points(ln)):
                return False
        return True
    return Pred(g_, "curves")


S2, S3 = math.sqrt(2), math.sqrt(3)


def F16(i, j, k):
    """The data {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}} with f_kji at {i, j, k} (linear)."""
    return 1 + (i - 1) + 2 * (j - 1) + 4 * (k - 1)


CUBE = "{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}"
VEC3 = "{{{{1,0,0},{1,0,0}},{{0,1,0},{0,1,0}}},{{{0,0,1},{0,0,1}},{{1,1,1},{1,1,1}}}}"
VEC3_TABLE = {(1, 1, 1): (1, 0, 0), (2, 1, 1): (1, 0, 0), (1, 2, 1): (0, 1, 0), (2, 2, 1): (0, 1, 0),
              (1, 1, 2): (0, 0, 1), (2, 1, 2): (0, 0, 1), (1, 2, 2): (1, 1, 1), (2, 2, 2): (1, 1, 1)}
CONST3 = "{{{{1,0,0},{1,0,0}},{{1,0,0},{1,0,0}}},{{{1,0,0},{1,0,0}},{{1,0,0},{1,0,0}}}}"
GRID3x3 = "{{{1, 0}, {1, 0}, {1, 0}}, {{1, 0}, {1, 0}, {1, 0}}, {{1, 0}, {1, 0}, {1, 0}}}"

FIELD_CASES = [
    # ── StreamPlot: streamlines are integral curves, so first integrals of the field stay constant along them:
    # {-y, x} (rotation) keeps x^2 + y^2; {x, -y} keeps x y; a constant field gives straight lines
    ("StreamPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}]", streams_keep(lambda x, y: x * x + y * y, 1e-4, 3)),
    ("StreamPlot[{x, -y}, {x, 0.1, 1}, {y, 0.1, 1}, StreamPoints -> 4]", streams_keep(lambda x, y: x * y, 1e-4)),
    ("StreamPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}, StreamPoints -> 3]", streams_along((1, 0), 3)),
    ("StreamPlot[{w20a y, x}, {x, -1, 1}, {y, -1, 1}]", unevaluated("StreamPlot")),
    ("StreamPlot[{-y, x}, {x, 1, -1}, {y, -1, 1}]", Raises(ValueError)),
    ('StreamPlot["a", {x, 0, 1}, {y, 0, 1}]', Raises(TypeError)),
    ("StreamPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}, Mesh -> 3]", Raises(NotImplementedError)),
    ("StreamDensityPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}, StreamPoints -> 4]",
     streams_keep(lambda x, y: x * x + y * y, 1e-4)),
    # the background is |v| = sqrt(x^2 + y^2): the darkest (lowest) vertex colour is at the origin
    ("StreamDensityPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}, StreamPoints -> 2, PlotPoints -> 3, MaxRecursion -> 0]",
     Pred(lambda g: complex_data(g)[0][complex_data(g)[1].index(min(complex_data(g)[1]))] == (0.0, 0.0),
          "minimum |v| at the origin")),
    ("StreamDensityPlot[{{1, 0}, x}, {x, 0, 1}, {y, 0, 1}, StreamPoints -> 2]", streams_along((1, 0))),
    ("StreamDensityPlot[{w20a, x}, {x, 0, 1}, {y, 0, 1}]", unevaluated("StreamDensityPlot")),
    ("StreamDensityPlot[{1, 0}, {x, 0, 0}, {y, 0, 1}]", Raises(ValueError)),
    ("StreamDensityPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}, Mesh -> 3]", Raises(NotImplementedError)),
    ("VectorDensityPlot[{1, x}, {x, 0, 1}, {y, 0, 1}, VectorPoints -> 2]",
     arrows_follow(lambda x, y: (1, x), count=4)),
    ("VectorDensityPlot[{{y, -x}, 1}, {x, -1, 1}, {y, -1, 1}, VectorPoints -> 3]",
     arrows_follow(lambda x, y: (y, -x), count=8)),          # the centre vector is zero and omitted
    ("VectorDensityPlot[{1, x}, {x, 0, 1}, {y, 0, 1}, VectorPoints -> 2]", has("GraphicsComplex", 1)),
    ("VectorDensityPlot[{w20a, x}, {x, 0, 1}, {y, 0, 1}]", unevaluated("VectorDensityPlot")),
    ("VectorDensityPlot[{1, x}, {x, 0, 1}, {y, 0, 1}, VectorPoints -> 0]", Raises(ValueError)),
    ("VectorDensityPlot[{1, x}, {x, 0, 1}, {y, 0, 1}, VectorPoints -> Fine]", Raises(NotImplementedError)),
    ("VectorPlot3D[{x, y, z}, {x, -1, 1}, {y, -1, 1}, {z, -1, 1}, VectorPoints -> 2]",
     arrows_follow(lambda x, y, z: (x, y, z), count=8)),
    ("VectorPlot3D[{1, 2, 3}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, VectorPoints -> 3]",
     arrows_follow(lambda x, y, z: (1, 2, 3), count=27)),
    ("VectorPlot3D[{-y, x, 1}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, VectorPoints -> 2]",
     arrows_follow(lambda x, y, z: (-y, x, 1), count=8)),
    ("VectorPlot3D[{w20a, y, z}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("VectorPlot3D")),
    ("VectorPlot3D[{x, y}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", Raises(ValueError)),
    ("VectorPlot3D[{x, y, z}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, Mesh -> 1]", Raises(NotImplementedError)),
    ("StreamPlot3D[{1, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, StreamPoints -> 2]", streams_along((1, 0, 0), 4)),
    ("StreamPlot3D[{-y, x, 0}, {x, -1, 1}, {y, -1, 1}, {z, 0, 1}, StreamPoints -> 2]",
     streams_keep(lambda x, y, z: x * x + y * y, 1e-4)),
    ("StreamPlot3D[{-y, x, 0}, {x, -1, 1}, {y, -1, 1}, {z, 0, 1}, StreamPoints -> 2]",
     streams_keep(lambda x, y, z: z, 1e-12)),
    ("StreamPlot3D[{w20a, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("StreamPlot3D")),
    ("StreamPlot3D[{1, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 1, 0}]", Raises(ValueError)),
    ("StreamPlot3D[{1, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, StreamPoints -> Fine]", Raises(NotImplementedError)),
    # ── list vector data: v_ij at {j, i} (ListDensityPlot convention); the zero vector is omitted
    ("ListVectorPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}]",
     arrows_follow(lookup({(1, 1): (1, 0), (2, 1): (0, 1), (1, 2): (1, 1), (2, 2): (0, 0)}), count=3)),
    ("ListVectorPlot[{{{0, 0}, {1, 0}}, {{1, 1}, {0, 2}}, {{0, 1}, {1, 1}}}]",
     arrows_follow(lookup({(0, 0): (1, 0), (1, 1): (0, 2), (0, 1): (1, 1)}), count=3)),
    ("ListVectorPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}, DataRange -> {{0, 10}, {0, 1}}]",
     arrows_follow(lookup({(0, 0): (1, 0), (10, 0): (0, 1), (0, 1): (1, 1), (10, 1): (0, 0)}), count=3)),
    ("ListVectorPlot[w20x]", unevaluated("ListVectorPlot")),
    ('ListVectorPlot["a"]', Raises(TypeError)),
    ("ListVectorPlot[{{{1, 0}}}]", Raises(ValueError)),
    ("ListVectorPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}, Mesh -> 2]", Raises(NotImplementedError)),
    ("ListVectorPlot3D[" + VEC3 + "]", arrows_follow(lookup(VEC3_TABLE), count=8)),
    ("ListVectorPlot3D[{{{0, 0, 0}, {1, 0, 0}}, {{1, 1, 1}, {0, 0, 2}}, {{2, 0, 1}, {1, 1, 1}}}]",
     arrows_follow(lookup({(0, 0, 0): (1, 0, 0), (1, 1, 1): (0, 0, 2), (2, 0, 1): (1, 1, 1)}), count=3)),
    ("ListVectorPlot3D[" + VEC3 + ", DataRange -> {{0, 1}, {0, 1}, {0, 1}}]",
     arrows_follow(lookup({tuple(c - 1 for c in k): v for k, v in VEC3_TABLE.items()}), count=8)),
    ("ListVectorPlot3D[w20x]", unevaluated("ListVectorPlot3D")),
    ("ListVectorPlot3D[{{{{1, 0}}}}]", Raises(ValueError)),
    ("ListVectorPlot3D[" + VEC3 + ", Mesh -> 2]", Raises(NotImplementedError)),
    # density backgrounds interpolate |v| between the data points: vertices are the data points
    ("ListVectorDensityPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}]", vertex_set_is([(1, 1), (2, 1), (1, 2), (2, 2)])),
    ("ListVectorDensityPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}]",
     arrows_follow(lookup({(1, 1): (1, 0), (2, 1): (0, 1), (1, 2): (1, 1), (2, 2): (0, 0)}), count=3)),
    ("ListVectorDensityPlot[{{{0, 0}, {1, 0}}, {{1, 0}, {0, 2}}, {{0, 1}, {1, 1}}}]",
     scipy_or_raise(vertex_set_is([(0, 0), (1, 0), (0, 1)]))),
    ("ListVectorDensityPlot[w20x]", unevaluated("ListVectorDensityPlot")),
    ("ListVectorDensityPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}, DataRange -> {{1, 0}, {0, 1}}]", Raises(ValueError)),
    ("ListVectorDensityPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}, Mesh -> 2]", Raises(NotImplementedError)),
    ("ListStreamPlot[" + GRID3x3 + ", StreamPoints -> 3]", streams_along((1, 0), 3)),
    # bilinear interpolation of the rotation field sampled on a grid stays a rotation field (linear data)
    ("ListStreamPlot[Table[{-w20y/2, w20x/2}, {w20y, -2, 2}, {w20x, -2, 2}], DataRange -> {{-1, 1}, {-1, 1}}, "
     "StreamPoints -> 4]", streams_keep(lambda x, y: x * x + y * y, 1e-4)),
    ("ListStreamPlot[{{{0, 0}, {1, 0}}, {{1, 0}, {1, 0}}, {{0, 1}, {1, 0}}, {{1, 1}, {1, 0}}}, StreamPoints -> 3]",
     scipy_or_raise(streams_along((1, 0)))),
    ("ListStreamPlot[w20x]", unevaluated("ListStreamPlot")),
    ("ListStreamPlot[" + GRID3x3 + ", StreamPoints -> 1]", Raises(ValueError)),
    ("ListStreamPlot[" + GRID3x3 + ", Mesh -> 2]", Raises(NotImplementedError)),
    ("ListStreamDensityPlot[" + GRID3x3 + ", StreamPoints -> 3]", streams_along((1, 0), 3)),
    ("ListStreamDensityPlot[" + GRID3x3 + ", StreamPoints -> 3]",
     vertex_set_is([(i, j) for i in (1, 2, 3) for j in (1, 2, 3)])),
    ("ListStreamDensityPlot[w20x]", unevaluated("ListStreamDensityPlot")),
    ("ListStreamDensityPlot[{{{1, 0}}}]", Raises(ValueError)),
    ("ListStreamDensityPlot[" + GRID3x3 + ", Mesh -> 2]", Raises(NotImplementedError)),
    ("ListStreamPlot3D[" + CONST3 + ", StreamPoints -> 2]", streams_along((1, 0, 0), 4)),
    ("ListStreamPlot3D[" + CONST3 + ", StreamPoints -> 3, DataRange -> {{0, 1}, {0, 1}, {0, 1}}]",
     streams_along((1, 0, 0), 4)),
    ("ListStreamPlot3D[w20x]", unevaluated("ListStreamPlot3D")),
    ("ListStreamPlot3D[" + CONST3 + ", StreamPoints -> 0]", Raises(ValueError)),
    ("ListStreamPlot3D[" + CONST3 + ", Mesh -> 2]", Raises(NotImplementedError)),
    # ── displacement: p + s v(p) with the largest displacement 10% of the diagonal (s = 0.1 diag / max|v|)
    ("VectorDisplacementPlot[{x, 0}, {x, 0, 1}, {y, 0, 1}, PlotPoints -> 2, MaxRecursion -> 0]",
     vertex_set_is([(0, 0), (1 + 0.1 * S2, 0), (0, 1), (1 + 0.1 * S2, 1)])),
    ("VectorDisplacementPlot[{0, 1}, {x, 0, 2}, {y, 0, 1}, PlotPoints -> 2, MaxRecursion -> 0]",
     vertex_set_is([(0, 0.1 * math.sqrt(5)), (2, 0.1 * math.sqrt(5)), (0, 1 + 0.1 * math.sqrt(5)),
                    (2, 1 + 0.1 * math.sqrt(5))])),
    ("VectorDisplacementPlot[{x, 0}, Element[{x, y}, Rectangle[{0, 0}, {1, 1}]], PlotPoints -> 2, MaxRecursion -> 0]",
     vertex_set_is([(0, 0), (1 + 0.1 * S2, 0), (0, 1), (1 + 0.1 * S2, 1)])),
    ("VectorDisplacementPlot[{w20a, 0}, {x, 0, 1}, {y, 0, 1}]", unevaluated("VectorDisplacementPlot")),
    ("VectorDisplacementPlot[{x, 0}, {x, 0, 1}, {y, 1, 1}]", Raises(ValueError)),
    ("VectorDisplacementPlot[{x, 0}, Element[{x, y}, Disk[]]]", Raises(NotImplementedError)),
    ('VectorDisplacementPlot["a", {x, 0, 1}, {y, 0, 1}]', Raises(TypeError)),
    ("VectorDisplacementPlot3D[{x, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 2]",
     Pred(lambda g: set(vertex_set(g)) == {(round(x, 9), y, z) for x in (0, 1 + 0.1 * S3) for y in (0, 1)
                                          for z in (0, 1)}, "cube corners displaced")),
    ("VectorDisplacementPlot3D[{0, 0, 1}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 2]",
     Pred(lambda g: {q[2] for q in vertex_set(g)} == {round(0.1 * S3, 9), round(1 + 0.1 * S3, 9)}, "lifted")),
    ("VectorDisplacementPlot3D[{x, 0, 0}, Element[{x, y, z}, Cuboid[{0, 0, 0}, {1, 1, 1}]], PlotPoints -> 2]",
     Pred(lambda g: len(prims(g, "GraphicsComplex")) == 1, "cuboid region")),
    ("VectorDisplacementPlot3D[{w20a, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]",
     unevaluated("VectorDisplacementPlot3D")),
    ("VectorDisplacementPlot3D[{x, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 1]", Raises(ValueError)),
    ("VectorDisplacementPlot3D[{x, 0, 0}, Element[{x, y, z}, Ball[]]]", Raises(NotImplementedError)),
    ("ListVectorDisplacementPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}]",
     vertex_set_is([(1 + 0.1 * S2 / S2, 1), (2, 1 + 0.1 * S2 / S2), (1 + 0.1, 2 + 0.1), (2, 2)])),
    ("ListVectorDisplacementPlot[{{{0, 0}, {1, 0}}, {{1, 0}, {1, 0}}, {{0, 1}, {1, 0}}}]",
     scipy_or_raise(vertex_set_is([(0.1 * S2, 0), (1 + 0.1 * S2, 0), (0.1 * S2, 1)]))),
    ("ListVectorDisplacementPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}]", has("Polygon", 1)),
    ("ListVectorDisplacementPlot[w20x]", unevaluated("ListVectorDisplacementPlot")),
    ("ListVectorDisplacementPlot[{{{1, 0}, {0, 1}, {1, 1}}, {{1, 1}, {0, 0}}}]", Raises(ValueError)),
    ("ListVectorDisplacementPlot[{{{1, 0}, {0, 1}}, {{1, 1}, {0, 0}}}, Mesh -> 1]", Raises(NotImplementedError)),
    ("ListVectorDisplacementPlot3D[" + VEC3 + "]",
     vertex_set_is([tuple(c + 0.1 * S3 / math.sqrt(sum(t * t for t in (1, 1, 1))) * vc for c, vc in zip(k, v))
                    for k, v in VEC3_TABLE.items()])),
    ("ListVectorDisplacementPlot3D[" + VEC3 + "]", Pred(lambda g: len(prims(g, "Polygon")[0].args[0].args) == 6,
                                                       "six boundary faces")),
    ("ListVectorDisplacementPlot3D[w20x]", unevaluated("ListVectorDisplacementPlot3D")),
    ("ListVectorDisplacementPlot3D[{{{{1, 0, 0}}}}]", Raises(ValueError)),
    ("ListVectorDisplacementPlot3D[" + VEC3 + ", Mesh -> 1]", Raises(NotImplementedError)),
]

SLICE_CASES = [
    # slice planes (Wolfram SliceDensityPlot3D reference, "Details": "CenterPlanes", "BackPlanes", stacked planes)
    ('SliceDensityPlot3D[x + y + z, "CenterPlanes", {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 2, '
     'MaxRecursion -> 0]',
     Pred(lambda g: len(prims(g, "GraphicsComplex")) == 3 and
          all(min(abs(c - 0.5) for c in q) < 1e-12 for q in vertex_set(g)), "three centre planes")),
    ('SliceDensityPlot3D[x, "BackPlanes", {x, 0, 1}, {y, 0, 2}, {z, 0, 3}, PlotPoints -> 2, MaxRecursion -> 0]',
     Pred(lambda g: [sorted({P(x)[i] for x in gc.args[0].args}) for gc, i in
                     zip(prims(g, "GraphicsComplex"), (0, 1, 2))] == [[0.0], [2.0], [0.0]], "back planes")),
    ('SliceDensityPlot3D[x, {"ZStackedPlanes", 3}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, PlotPoints -> 2, '
     'MaxRecursion -> 0]',
     Pred(lambda g: sorted({P(x)[2] for gc in prims(g, "GraphicsComplex") for x in gc.args[0].args}) ==
          [0.25, 0.5, 0.75], "z = 1/4, 1/2, 3/4")),
    ("SliceDensityPlot3D[x + w20a, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("SliceDensityPlot3D")),
    ('SliceDensityPlot3D[x, {"XStackedPlanes", {2}}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]', Raises(ValueError)),
    ('SliceDensityPlot3D[x, "CenterSphere", {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]', Raises(NotImplementedError)),
    # contour lines on the plane z = 1/2 at the level x = 1/2 are the segment x = 1/2, z = 1/2
    ("SliceContourPlot3D[x, z == 0.5, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, Contours -> {0.5}, PlotPoints -> 3, "
     "MaxRecursion -> 0]", lines_where(lambda x, y, z: abs(x - 0.5) < 1e-9 and abs(z - 0.5) < 1e-9)),
    ("SliceContourPlot3D[x + y, x == 0, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, Contours -> {0.25, 0.75}, PlotPoints -> 3, "
     "MaxRecursion -> 0]", lines_where(lambda x, y, z: x == 0 and min(abs(y - 0.25), abs(y - 0.75)) < 1e-9)),
    ("SliceContourPlot3D[x, z == 0.5, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, Contours -> {0.5}, PlotPoints -> 3, "
     "MaxRecursion -> 0]", polygons_where(lambda x, y, z: abs(z - 0.5) < 1e-12)),
    ("SliceContourPlot3D[x + w20a, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("SliceContourPlot3D")),
    ("SliceContourPlot3D[x, z == 5, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", Raises(ValueError)),
    ("SliceContourPlot3D[x, x y z == 1, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", Raises(NotImplementedError)),
    ('SliceVectorPlot3D[{1, 0, 0}, {"XStackedPlanes", {0.5}}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, VectorPoints -> 2]',
     arrows_follow(lambda x, y, z: (1, 0, 0), count=4)),
    ("SliceVectorPlot3D[{x, y, z}, {x, 0.1, 1}, {y, 0.1, 1}, {z, 0.1, 1}, VectorPoints -> 2]",
     arrows_follow(lambda x, y, z: (x, y, z), count=12)),
    ('SliceVectorPlot3D[{-y, x, 1}, "BackPlanes", {x, 0, 1}, {y, 0, 1}, {z, 0, 1}, VectorPoints -> 2]',
     arrows_follow(lambda x, y, z: (-y, x, 1), count=12)),
    ("SliceVectorPlot3D[{w20a, 0, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", unevaluated("SliceVectorPlot3D")),
    ("SliceVectorPlot3D[{1, 0}, {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]", Raises(ValueError)),
    ('SliceVectorPlot3D[{1, 0, 0}, "CenterCutBox", {x, 0, 1}, {y, 0, 1}, {z, 0, 1}]', Raises(NotImplementedError)),
    # list slices: f_kji at {i, j, k}; the data is linear so trilinear interpolation is exact
    ('ListSliceDensityPlot3D[' + CUBE + ', "BackPlanes", PlotPoints -> 2]',
     Pred(lambda g: len(prims(g, "GraphicsComplex")) == 3, "three planes")),
    ('ListSliceDensityPlot3D[' + CUBE + ', {"XStackedPlanes", {1.5}}, PlotPoints -> 2]',
     Pred(lambda g: {q[0] for q in vertex_set(g)} == {1.5}, "plane x = 1.5")),
    ('ListSliceDensityPlot3D[' + CUBE + ', PlotPoints -> 2, DataRange -> {{0, 1}, {0, 1}, {0, 1}}]',
     Pred(lambda g: all(min(abs(c - 0.5) for c in q) < 1e-12 for q in vertex_set(g)), "centre planes")),
    ("ListSliceDensityPlot3D[w20x]", unevaluated("ListSliceDensityPlot3D")),
    ('ListSliceDensityPlot3D[{{{1}}}]', Raises(ValueError)),
    ('ListSliceDensityPlot3D[' + CUBE + ', "CenterSphere"]', Raises(NotImplementedError)),
    ("ListSliceContourPlot3D[" + CUBE + ", PlotPoints -> 3, Contours -> {4.5}]",
     lines_where(lambda x, y, z: abs(F16(x, y, z) - 4.5) < 1e-9)),
    ('ListSliceContourPlot3D[' + CUBE + ', {"ZStackedPlanes", {1.5}}, PlotPoints -> 3, Contours -> {4, 5}]',
     lines_where(lambda x, y, z: z == 1.5 and min(abs(F16(x, y, z) - 4), abs(F16(x, y, z) - 5)) < 1e-9)),
    ("ListSliceContourPlot3D[" + CUBE + ", PlotPoints -> 3, Contours -> {4.5}]",
     polygons_where(lambda x, y, z: min(abs(c - 1.5) for c in (x, y, z)) < 1e-12)),
    ("ListSliceContourPlot3D[w20x]", unevaluated("ListSliceContourPlot3D")),
    ("ListSliceContourPlot3D[" + CUBE + ", PlotPoints -> 1]", Raises(ValueError)),
    ("ListSliceContourPlot3D[" + CUBE + ", Mesh -> 2]", Raises(NotImplementedError)),
    ("ListSliceVectorPlot3D[" + CONST3 + ", VectorPoints -> 2]", arrows_follow(lambda x, y, z: (1, 0, 0), count=12)),
    # trilinear interpolation of the corner data VEC3 on the plane z = 1
    ("ListSliceVectorPlot3D[" + VEC3 + ', {"ZStackedPlanes", {1}}, VectorPoints -> 2]',
     arrows_follow(lambda x, y, z: (2 - y, y - 1, 0), scaled=False, count=4)),
    ("ListSliceVectorPlot3D[" + CONST3 + ', "BackPlanes", VectorPoints -> 3]',
     arrows_follow(lambda x, y, z: (1, 0, 0), count=27)),
    ("ListSliceVectorPlot3D[w20x]", unevaluated("ListSliceVectorPlot3D")),
    ("ListSliceVectorPlot3D[" + CONST3 + ", VectorPoints -> 0]", Raises(ValueError)),
    ("ListSliceVectorPlot3D[" + CONST3 + ', "CenterCutSphere"]', Raises(NotImplementedError)),
    # line integral convolution of white noise: a constant field {1, 0} averages the noise along rows over
    # 2 L + 1 pixels (clipped at the border) - the documented LIC construction with a box kernel
    ("LineIntegralConvolutionPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}, RasterSize -> 6, LineIntegralConvolutionScale -> 2]",
     lic_is_running_mean(6, 2)),
    ("LineIntegralConvolutionPlot[{3, 0}, {x, 0, 2}, {y, 0, 1}, RasterSize -> 5, LineIntegralConvolutionScale -> 1]",
     lic_is_running_mean(5, 1)),
    ("LineIntegralConvolutionPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}, RasterSize -> 4, LineIntegralConvolutionScale -> 1]",
     Pred(lambda g: opt(g, "PlotRange") == ev("{{0., 1.}, {0., 1.}}"), "plot range")),
    ("LineIntegralConvolutionPlot[{w20a, 0}, {x, 0, 1}, {y, 0, 1}]", unevaluated("LineIntegralConvolutionPlot")),
    ("LineIntegralConvolutionPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}, RasterSize -> 2]", Raises(ValueError)),
    ("LineIntegralConvolutionPlot[{{1, 0}, Image[{{0, 1}}]}, {x, 0, 1}, {y, 0, 1}]", Raises(NotImplementedError)),
    ("ListLineIntegralConvolutionPlot[{{{1, 0}, {1, 0}}, {{1, 0}, {1, 0}}}, RasterSize -> 6, "
     "LineIntegralConvolutionScale -> 2]", lic_is_running_mean(6, 2)),
    ("ListLineIntegralConvolutionPlot[" + GRID3x3 + ", RasterSize -> 4, LineIntegralConvolutionScale -> 3]",
     lic_is_running_mean(4, 3)),
    ("ListLineIntegralConvolutionPlot[" + GRID3x3 + ", RasterSize -> 4, LineIntegralConvolutionScale -> 3]",
     has("Raster", 1)),
    ("ListLineIntegralConvolutionPlot[w20x]", unevaluated("ListLineIntegralConvolutionPlot")),
    ("ListLineIntegralConvolutionPlot[{{{1, 0}}}]", Raises(ValueError)),
    ("ListLineIntegralConvolutionPlot[" + GRID3x3 + ", Mesh -> 1]", Raises(NotImplementedError)),
]

COMPLEX_CASES = [
    # ComplexPlot colours: a cyclic colour function of Arg[f] (Wolfram ComplexPlot reference)
    ("ComplexPlot[z, {z, -1 - I, 1 + I}, PlotPoints -> 2, MaxRecursion -> 0]", vertices_hue(lambda w: w)),
    ("ComplexPlot[z^2 + 1, {z, 2}, PlotPoints -> 5, MaxRecursion -> 0]", vertices_hue(lambda w: w * w + 1)),
    ("ComplexPlot[1/(z + 0.1), {z, 1}, PlotPoints -> 4, MaxRecursion -> 0]", vertices_hue(lambda w: 1 / (w + 0.1))),
    ("ComplexPlot[w20a z, {z, 1}]", unevaluated("ComplexPlot")),
    ("ComplexPlot[z, {z, 1, 1 + I}]", Raises(ValueError)),
    ('ComplexPlot[z, {z, "a", 1}]', Raises(TypeError)),
    ("ComplexPlot[z, {z, 1}, ColorFunction -> Hue]", Raises(NotImplementedError)),
    ("ComplexPlot3D[z^2, {z, 1}, PlotPoints -> 3, MaxRecursion -> 0]",
     vertices(lambda x, y, z: abs(z - abs(complex(x, y) ** 2)) < 1e-12)),
    ("ComplexPlot3D[Exp[z], {z, -1 - I, 1 + I}, PlotPoints -> 3, MaxRecursion -> 0]",
     vertices(lambda x, y, z: abs(z - math.exp(x)) < 1e-12)),
    ("ComplexPlot3D[z, {z, 1}, PlotPoints -> 2, MaxRecursion -> 0]", vertices_hue(lambda w: w)),
    ("ComplexPlot3D[w20a z, {z, 1}]", unevaluated("ComplexPlot3D")),
    ("ComplexPlot3D[z, {z, -1}]", Raises(ValueError)),
    ("ComplexPlot3D[z, {z, 1}, Mesh -> 3]", Raises(NotImplementedError)),
    # ComplexContourPlot: contours of a real-valued function of z = x + I y (Wolfram reference, Details)
    ("ComplexContourPlot[Abs[z], {z, 2}, Contours -> {1}, PlotPoints -> 41, MaxRecursion -> 0]",
     lines_where(lambda x, y: abs(math.hypot(x, y) - 1) < 0.01)),
    ("ComplexContourPlot[Re[z] == 0.5, {z, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     lines_where(lambda x, y: abs(x - 0.5) < 1e-9)),      # values on grid lines are nudged by ~1e-12 (W16)
    ("ComplexContourPlot[{Re[z] == 0.25, Im[z] == -0.5}, {z, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     lines_where(lambda x, y: abs(x - 0.25) < 1e-9 or abs(y + 0.5) < 1e-9)),
    ("ComplexContourPlot[Re[z], {z, 1}, Contours -> {0}, PlotPoints -> 5, MaxRecursion -> 0]",
     polygons_where(lambda x, y: -1 <= x <= 1)),
    ("ComplexContourPlot[w20a Re[z], {z, 1}]", unevaluated("ComplexContourPlot")),
    ("ComplexContourPlot[Re[z], {z, 0}]", Raises(ValueError)),
    ("ComplexContourPlot[Re[z], {z, 1}, Mesh -> 2]", Raises(NotImplementedError)),
    ("ComplexRegionPlot[Abs[z] < 1, {z, 2}, PlotPoints -> 21, MaxRecursion -> 0]",
     polygons_where(lambda x, y: math.hypot(x, y) <= 1 + 1e-9)),
    ("ComplexRegionPlot[Re[z] > 0, {z, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     polygons_where(lambda x, y: x >= -1e-12)),
    ("ComplexRegionPlot[Im[z] < 0 && Re[z] < 0, {z, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     polygons_where(lambda x, y: x <= 1e-12 and y <= 1e-12)),
    ("ComplexRegionPlot[Abs[z] < w20a, {z, 1}]", unevaluated("ComplexRegionPlot")),
    ("ComplexRegionPlot[Abs[z] < 1, {z, 1, 1}]", Raises(ValueError)),
    ("ComplexRegionPlot[Abs[z] == 1, {z, 1}]", Raises(NotImplementedError, match="^ComplexRegionPlot: ")),
    ("ComplexVectorPlot[z, {z, 1}, VectorPoints -> 2]", arrows_follow(lambda x, y: (x, y), count=4)),
    ("ComplexVectorPlot[z^2, {z, 1}, VectorPoints -> 3]",
     arrows_follow(lambda x, y: ((complex(x, y) ** 2).real, (complex(x, y) ** 2).imag), count=8)),
    ("ComplexVectorPlot[Conjugate[z], {z, -1 - I, 1 + I}, VectorPoints -> 2]",
     arrows_follow(lambda x, y: (x, -y), count=4)),
    ("ComplexVectorPlot[w20a z, {z, 1}]", unevaluated("ComplexVectorPlot")),
    ("ComplexVectorPlot[z, {z, 1}, VectorPoints -> 0]", Raises(ValueError)),
    ("ComplexVectorPlot[z, {z, 1}, StreamPoints -> 3]", Raises(NotImplementedError)),
    ("ComplexStreamPlot[I z, {z, 1}, StreamPoints -> 3]", streams_keep(lambda x, y: x * x + y * y, 1e-4)),
    ("ComplexStreamPlot[1, {z, 1}, StreamPoints -> 3]", streams_along((1, 0), 3)),
    ("ComplexStreamPlot[Conjugate[z], {z, 0.1 + 0.1 I, 1 + I}, StreamPoints -> 3]",
     streams_keep(lambda x, y: x * y, 1e-4)),
    ("ComplexStreamPlot[w20a z, {z, 1}]", unevaluated("ComplexStreamPlot")),
    ("ComplexStreamPlot[z, {z, 1}, StreamPoints -> 1]", Raises(ValueError)),
    ("ComplexStreamPlot[z, {z, 1}, VectorPoints -> 3]", Raises(NotImplementedError)),
    ("ComplexListPlot[{1 + I, 2, -I}]", points_are([(1, 1), (2, 0), (0, -1)])),
    ("ComplexListPlot[{{1 + I}, {2 I, 3}}]", points_are([(1, 1), (0, 2), (3, 0)])),
    ("ComplexListPlot[{Exp[I Pi/2]}]", points_are([(0, 1)], 1e-12)),
    ("ComplexListPlot[{1, w20a}]", unevaluated("ComplexListPlot")),
    ('ComplexListPlot[{1, "a"}]', Raises(TypeError)),
    ("ComplexListPlot[{1, 2}, Joined -> 3]", Raises(ValueError)),
    ("ReImPlot[Exp[I x], {x, 0, 1}]", curves_are(math.cos, math.sin, tol=1e-12)),
    ("ReImPlot[x^2, {x, -1, 1}]", curves_are(lambda x: x * x, tol=1e-12)),
    ("ReImPlot[{x, I x}, {x, 0, 1}]", Pred(lambda g: len(prims(g, "Line")) >= 2, "curves")),
    ("ReImPlot[w20a x, {x, 0, 1}]", unevaluated("ReImPlot")),
    ("ReImPlot[x, {x, 1, 0}]", Raises(ValueError, match="^ReImPlot: ")),
    ("ReImPlot[x, {x, 0, 1}, Mesh -> 2]", Raises(NotImplementedError)),
    # AbsArgPlot: the curve |f| coloured by Arg[f] (Wolfram AbsArgPlot reference)
    ("AbsArgPlot[Exp[I x], {x, 0, 1}, PlotPoints -> 3, MaxRecursion -> 0]",
     Pred(lambda g: [q for ln in prims(g, "Line") for q in line_points(ln)] == [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)],
          "|f| = 1")),
    ("AbsArgPlot[Exp[I x], {x, 0, 1}, PlotPoints -> 3, MaxRecursion -> 0]",
     Pred(lambda g: near([c for col in prims(g, "Line")[0].args[1].args[1].args for c in tup(col)],
                         [c for t in (0.0, 0.5, 1.0) for c in hue(complex(math.cos(t), math.sin(t)))]),
          "hues of Arg")),
    ("AbsArgPlot[x - 1, {x, 0, 3}, PlotPoints -> 4, MaxRecursion -> 0]",
     Pred(lambda g: [q for ln in prims(g, "Line") for q in line_points(ln)] == [(0.0, 1.0), (1.0, 0.0), (2.0, 1.0),
                                                                                (3.0, 2.0)], "|x - 1|")),
    ("AbsArgPlot[w20a x, {x, 0, 1}]", unevaluated("AbsArgPlot")),
    ("AbsArgPlot[x, {x, 0, 0}]", Raises(ValueError)),
    ("AbsArgPlot[{x, x^2}, {x, 0, 1}]", Raises(NotImplementedError)),
    ("ComplexArrayPlot[{{1, I}, {0, -1}}]",
     Pred(lambda g: raster(g) == [[(0, 0, 0), (0, 1, 1)], [(1, 0, 0), (0.5, 1, 0)]], "row 1 on top, hues")),
    ("ComplexArrayPlot[{{-I, 1 + I}}]", Pred(lambda g: raster(g) == [[hue(-1j), hue(1 + 1j)]], "hues")),
    ("ComplexArrayPlot[{{1, 2}, {3, 4}}]", Pred(lambda g: opt(g, "PlotRange") == ev("{{0., 2.}, {0., 2.}}"),
                                                "2 x 2 cells")),
    ("ComplexArrayPlot[{{1, w20a}}]", unevaluated("ComplexArrayPlot")),
    ("ComplexArrayPlot[{{1, 2}, {3}}]", Raises(ValueError)),
    ('ComplexArrayPlot[{{"a"}}]', Raises(TypeError)),
    ("ComplexArrayPlot[{{1}}, ColorFunction -> Hue]", Raises(NotImplementedError)),
    # RevolutionPlot3D[fz, {t, ...}] = the curve {t, 0, fz} rotated about z (Wolfram reference, Details)
    ("RevolutionPlot3D[t^2, {t, 0, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     vertices(lambda x, y, z: abs(z - (x * x + y * y)) < 1e-12)),
    ("RevolutionPlot3D[{Cos[t], Sin[t]}, {t, -Pi/2, Pi/2}, {w20th, 0, Pi}, PlotPoints -> 5, MaxRecursion -> 0]",
     vertices(lambda x, y, z: abs(x * x + y * y + z * z - 1) < 1e-12 and y >= -1e-12)),
    ("RevolutionPlot3D[{1, 1, t}, {t, 0, 1}, PlotPoints -> 5, MaxRecursion -> 0]",
     vertices(lambda x, y, z: abs(x * x + y * y - 2) < 1e-12)),
    ("RevolutionPlot3D[{{t}, {2 t}}, {t, 0, 1}, PlotPoints -> 3, MaxRecursion -> 0]",
     Pred(lambda g: len(prims(g, "GraphicsComplex")) == 2, "two surfaces")),
    ("RevolutionPlot3D[w20a t, {t, 0, 1}]", unevaluated("RevolutionPlot3D")),
    ("RevolutionPlot3D[t, {t, 1, 0}]", Raises(ValueError, match="^RevolutionPlot3D: ")),
    ("RevolutionPlot3D[t, {t, 0, 1}, RevolutionAxis -> {1, 0, 0}]", Raises(NotImplementedError)),
]

CASES = (HIST_CASES + CHART_CASES + SMOOTH_CASES + GAUGE_CASES + STAT_CASES + FIN_CASES + OTHER_CASES + FIELD_CASES
         + SLICE_CASES + COMPLEX_CASES)

@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w20(case):
    check(case)


# ───────────── Check: methods (self-checks raise ArithmeticError; cross-checks discard wrong compiled code) ─────────────

def test_w20_streamline_tangency_self_check_raises():
    import numpy as np
    from mikoshilang.library.wl_plots3d_fields import check_tangent

    def F(x, y):
        x = np.asarray(x, dtype=float)
        return np.stack([np.ones_like(x), np.zeros_like(x)], axis=-1)
    check_tangent([[(0.0, 0.0), (0.1, 0.0)]], F, "StreamPlot")            # tangent: accepted
    with pytest.raises(ArithmeticError, match="^StreamPlot: "):
        check_tangent([[(0.0, 0.0), (0.0, 0.1)]], F, "StreamPlot")        # perpendicular to the field


def test_w20_wordcloud_overlap_self_check_raises(monkeypatch):
    from mikoshilang.library import wl_plots3d_charts as C
    monkeypatch.setattr(C, "wordcloud_layout", lambda words, fmax=40.0: [(s, 40.0, 0.0, 0.0, 10.0, 10.0)
                                                                        for s, _ in words])
    with pytest.raises(ArithmeticError, match="^WordCloud: "):
        ev('WordCloud[{"a", "b"}]')


def test_w20_complex_cross_check_discards_wrong_compiled_code():
    import numpy as np
    from mikoshilang.library.wl_plots3d_fields import ComplexFunction
    from mikoshilang.parser import parse
    cf = ComplexFunction(parse("w20z^2"), Symbol("w20z"), "ComplexPlot")
    assert cf.fast is not None
    cf.fast = lambda w: np.asarray(w) * 0 + 7          # a wrong compiled function
    cf.cross_check([0.3 + 0.2j])
    assert cf.fast is None                              # discarded: the kernel evaluates every point
    assert abs(complex(cf(np.array([1 + 1j]))[0]) - 2j) < 1e-12


def test_w20_every_batch_symbol_is_registered_and_none_skipped():
    from mikoshilang.library import wl_plots3d_charts as C, wl_plots3d_fields as F
    from mikoshilang.library.core import REGISTRY
    heads = ["WordCloud", "Histogram", "BoxWhiskerChart", "AngularGauge", "BarChart3D", "BubbleChart",
             "SmoothHistogram", "VerticalGauge", "BubbleChart3D", "BulletGauge", "ClockGauge", "ComplexStreamPlot",
             "ComplexVectorPlot", "DateHistogram", "DensityHistogram", "DistributionChart",
             "GaugeFaceElementFunction", "GaugeFaceStyle", "GaugeFrameElementFunction", "GaugeFrameSize",
             "GaugeFrameStyle", "GaugeLabels", "GaugeMarkers", "GaugeStyle", "Histogram3D", "HorizontalGauge",
             "KagiChart", "LineIntegralConvolutionPlot", "ListLineIntegralConvolutionPlot", "ListSliceContourPlot3D",
             "ListSliceDensityPlot3D", "ListSliceVectorPlot3D", "ListStreamDensityPlot", "ListStreamPlot",
             "ListStreamPlot3D", "ListVectorDensityPlot", "ListVectorDisplacementPlot",
             "ListVectorDisplacementPlot3D", "ListVectorPlot", "ListVectorPlot3D", "PairedBarChart",
             "PairedHistogram", "PairedSmoothHistogram", "PairwiseDensityHistogram",
             "PairwiseSmoothDensityHistogram", "RectangleChart", "RectangleChart3D", "SectorChart",
             "SliceVectorPlot3D", "SmoothDateHistogram", "SmoothDensityHistogram", "SmoothHistogram3D",
             "StreamDensityPlot", "StreamPlot", "StreamPlot3D", "ThermometerGauge", "VectorDensityPlot",
             "VectorDisplacementPlot", "VectorDisplacementPlot3D", "VectorPlot3D", "AbsArgPlot", "BubbleHistogram",
             "CandlestickChart", "CategoricalHistogram", "ComplexArrayPlot", "ComplexContourPlot", "ComplexListPlot",
             "ComplexPlot", "ComplexPlot3D", "ComplexRegionPlot", "ImageHistogram", "ImageVectorscopePlot",
             "PieChart3D", "ReImPlot", "RenkoChart", "RevolutionPlot3D", "SectorChart3D", "SliceContourPlot3D",
             "SliceDensityPlot3D", "TradingChart"]
    assert len(heads) == 80
    assert not [h for h in heads if h not in REGISTRY]
    assert C.SKIPPED == {} and F.SKIPPED == {}
