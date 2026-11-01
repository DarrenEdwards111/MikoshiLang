"""Wolfram batch W20 (part 1): histograms, charts, gauges, financial charts, word clouds and image plots.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). The vector, stream, slice, complex-function and revolution plots of the batch are in
``wl_plots3d_fields``.

Charts evaluate to inert ``Graphics`` / ``Graphics3D`` objects (W14 graphics core) built from ``Rectangle``,
``Cuboid``, ``Disk``, ``Polygon``, ``Line``, ``Point``, ``Text`` and ``GraphicsComplex``. The tests compare the
*data* (bin edges, heights, sector angles, needle angles, brick prices) with independently computed values, never
pixels. Output layout (stable and documented in each head's notes):

* histograms: ``Graphics[{{style_1, Rectangle[{b_i, 0}, {b_(i+1), h_i}], ...}, ...}, opts]`` with one group per
  dataset (zero-height bins are omitted); 3-D: ``Cuboid[{x_i, y_j, 0}, {x_(i+1), y_(j+1), h_ij}]``;
* bins are half-open ``[b_i, b_(i+1))``; heights follow the documented height specifications;
* charts: one ``{style, primitive}`` group per element in data order; gauges: a unit-sized face with the needle,
  bar or hands as the last group.

Binning: ``Automatic`` and the named rules ("Sturges", "Scott", "FreedmanDiaconis") pick a target width from the
rule and round it to a nice width m 10^k (m in 1, 2, 5, as FindDivisions) with bins aligned at multiples of the
width; ``n`` gives the nice width closest to n bins. Wolfram's own automatic choice is not documented, so these
widths can differ from Wolfram's for the same data (the bins are still exact for the stated widths).
Symbols that cannot be done correctly are listed in ``SKIPPED`` at the end of the module.
"""
from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, S
from .wl_graphics_color import parse_color, K_input, inert_option
from . import wl_graphics_core as G
from .wl_graphics_core import is_sym, is_rule, opt_name
from . import wl_plot_options as PO
from . import wl_plots2d as P2
from .wl_plots2d import (ev, rgb, default_color, fl, pt, pts, num, real_arg, is_missing, split_options,
                         finish_graphics, range_expr, AUTO, NONE, GOLDEN, blend, DENSITY_GRADIENT, color_function)

AREA = "plotting: charts, histograms and gauges (W20)"


def _np():
    import numpy as np
    return np


TRUE_ = True
GRAY_EDGE = Expr("EdgeForm", Expr("GrayLevel", 0.5))


# ───────────────────────────── arguments ─────────────────────────────

def split_args(args, fname, lo, hi):
    """(positional args, option rules) with the option rules taken from the end."""
    args = list(args)
    opts = []
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol):
        opts.insert(0, args.pop())
    if not lo <= len(args) <= hi:
        want = str(lo) if lo == hi else f"{lo} to {hi}"
        raise ValueError(f"{fname}: expected {want} positional argument(s), got {len(args)}")
    return args, opts


def exact_value(x):
    """Fraction for an exact rational, float for an inexact real, None otherwise."""
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Expr) and x.head == "Rational":
        try:
            s = S(x)
            return Fraction(int(s.p), int(s.q))
        except Exception:
            return None
    v = num(x)
    return v


def values_of(d, fname, what="data"):
    """Finite real values of a flat dataset in order (Fraction for exact rationals, float otherwise); missing
    values (Missing[], None, non-real numbers) are dropped; symbolic entries -> Unevaluated."""
    d, _style, _w = P2._unwrap(d, fname)
    if isinstance(d, Expr) and d.head == "Association":
        d = L(*[v for _, v in P2._assoc_items(d)])
    if isinstance(d, Expr) and d.head in ("WeightedData", "EventData", "TimeSeries", "EventSeries", "TemporalData",
                                          "Tabular", "Dataset", "Quantity", "QuantityArray"):
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the {what} must be a list of real numbers, not {K_input(d)}")
    out = []
    for x in d.args:
        if K.is_list(x):
            raise ValueError(f"{fname}: the {what} must be a flat list of real numbers")
        v = P2.scalar(x, fname)
        if v is None:
            continue
        e = exact_value(x)
        out.append(e if isinstance(e, Fraction) else v)
    return out


def is_dataset_list(d):
    """{data1, data2, ...}: a non-empty list whose elements are all lists (or wrapped lists)."""
    if not K.is_list(d) or not d.args:
        return False
    return all(K.is_list(P2._unwrap(a, "")[0]) for a in d.args)


def datasets_1d(data, fname):
    """[(values, style)] for one dataset or a list of datasets."""
    data_u, st, _ = P2._unwrap(data, fname)
    if isinstance(data_u, Expr) and data_u.head == "Association" and data_u.args and \
            all(K.is_list(v) for _, v in P2._assoc_items(data_u)):
        return [(values_of(v, fname), None) for _, v in P2._assoc_items(data_u)]
    if is_dataset_list(data_u):
        out = []
        for a in data_u.args:
            _, s, _ = P2._unwrap(a, fname)
            out.append((values_of(a, fname), s))
        return out
    return [(values_of(data, fname), st)]


def fval(x):
    return float(x)


def _frac(x):
    """Exact Fraction for ints/rationals (sympy or kernel), float otherwise."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, sp.Rational):
        return Fraction(int(x.p), int(x.q))
    e = exact_value(x)
    return e


def real_number(x, fname, what):
    """A finite real given as an argument: Fraction (exact) or float; strings/lists raise TypeError, symbols
    stay unevaluated."""
    if isinstance(x, (bool, str)) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    e = exact_value(x)
    if e is None:
        v = real_arg(x, fname, what)
        return v
    return e


def pos_number(x, fname, what):
    v = real_number(x, fname, what)
    if not v > 0:
        raise ValueError(f"{fname}: {what} must be positive, not {K_input(x)}")
    return v


def pos_int(x, fname, what):
    if isinstance(x, bool) or not K.is_int(x):
        if isinstance(x, (Symbol,)) or (isinstance(x, Expr) and not K.is_list(x) and num(x) is None):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    if int(x) < 1:
        raise ValueError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    return int(x)


# ───────────────────────────── styles ─────────────────────────────

def _directive_items(u):
    return list(u.args) if isinstance(u, Expr) and u.head == "Directive" else (list(u.args) if K.is_list(u) else [u])


def chart_styles(o, n, fname, base=None, opacity=None):
    """Directive per chart element / dataset: ChartBaseStyle, default colour i (ColorData[97]), gray edges,
    optional opacity, then ChartStyle (a directive or a list cycled over the elements)."""
    spec = o.get("ChartStyle", AUTO)
    bstyle = o.get("ChartBaseStyle", AUTO)
    if isinstance(spec, (int, str)) and not isinstance(spec, bool):
        raise NotImplementedError(f"{fname}: ChartStyle -> {K_input(spec)} (a colour scheme) is not supported; "
                                  "give directives")
    pre = [] if is_sym(bstyle, "Automatic", "None") else _directive_items(bstyle)
    if K.is_list(spec):
        user = list(spec.args) or [AUTO]
    elif is_sym(spec, "Automatic"):
        user = [AUTO]
    else:
        user = [spec]
    out = []
    for i in range(n):
        items = [default_color(i if base is None else base(i)), GRAY_EDGE]
        if opacity is not None:
            items.append(Expr("Opacity", opacity))
        items = pre + items
        u = user[i % len(user)]
        if not is_sym(u, "Automatic"):
            if is_sym(u, "None"):
                out.append(None)
                continue
            G.directive_like(u, fname, "ChartStyle")
            items += _directive_items(u)
        out.append(Expr("Directive", *items))
    return out


# ───────────────────────────── binning ─────────────────────────────

def dec(x):
    """Exact Fraction of a value (floats by their shortest decimal representation)."""
    if isinstance(x, Fraction):
        return x
    return Fraction(repr(float(x)))


def nice_width(h):
    """The nice width m 10^k (m in 1, 2, 5) nearest to h > 0 on a log scale (Fraction)."""
    e = math.floor(math.log10(h))
    best = None
    for k in (e - 1, e, e + 1):
        for m in (1, 2, 5):
            c = Fraction(m) * (Fraction(10) ** k)
            d = abs(math.log(float(c) / h))
            if best is None or d < best[0] - 1e-12:
                best = (d, c)
    return best[1]


def _nice_width(lo, hi, nbins):
    """Nice width for about nbins bins over [lo, hi]."""
    return nice_width(float(hi - lo) / max(1, int(nbins)))


def _aligned_edges(lo, hi, w):
    """Edges k w covering [lo, hi] with hi strictly inside the last bin (exact arithmetic on decimal values)."""
    ex = isinstance(w, Fraction)
    W = dec(w)
    a = math.floor(dec(lo) / W)
    b = math.floor(dec(hi) / W) + 1
    return [W * k if ex else float(W * k) for k in range(a, b + 1)]


def _quantile_lin(vals, q):
    np = _np()
    return float(np.percentile(np.array([float(v) for v in vals]), q * 100))


def _target_width(vals, rule):
    n = len(vals)
    lo, hi = min(vals), max(vals)
    rng = float(hi - lo)
    if rule == "Sturges":
        k = math.ceil(math.log2(n) + 1) if n > 0 else 1
        return None, k
    if rule == "Scott":
        np = _np()
        sd = float(np.std(np.array([float(v) for v in vals]), ddof=1)) if n > 1 else 0.0
        h = 3.49 * sd * n ** (-1 / 3)
    else:                                             # Freedman-Diaconis (also Automatic)
        iqr = _quantile_lin(vals, 0.75) - _quantile_lin(vals, 0.25)
        h = 2 * iqr * n ** (-1 / 3)
    if not h > 0:
        k = math.ceil(math.log2(n) + 1) if n > 0 else 1
        return None, k
    return h, max(1, round(rng / h)) if rng > 0 else 1


def bin_edges(vals, spec, fname, allow_fun=True, data_expr=None):
    """Bin edges (ascending, Fraction or float) for the values and a Wolfram bin specification."""
    if not vals:
        raise ValueError(f"{fname}: the data has no numeric values")
    lo, hi = min(vals), max(vals)
    if spec is None or is_sym(spec, "Automatic") or spec in ("Sturges", "Scott", "FreedmanDiaconis"):
        rule = "FreedmanDiaconis" if spec is None or is_sym(spec, "Automatic") else spec
        if lo == hi:
            w = Fraction(1)
            return _aligned_edges(_frac(lo) if isinstance(lo, Fraction) else lo,
                                  _frac(hi) if isinstance(hi, Fraction) else hi, w)
        h, k = _target_width(vals, rule)
        w = nice_width(h) if h is not None else _nice_width(lo, hi, k)
        return _aligned_edges(lo, hi, w)
    if isinstance(spec, str):
        if spec in ("Knuth", "Wand"):
            raise NotImplementedError(f"{fname}: the binning method {spec!r} is not supported; use \"Sturges\", "
                                      "\"Scott\", \"FreedmanDiaconis\" or explicit bins")
        raise ValueError(f"{fname}: unknown binning method {spec!r}")
    if K.is_int(spec) and not isinstance(spec, bool):
        n = pos_int(spec, fname, "the number of bins")
        if lo == hi:
            return _aligned_edges(lo, hi, Fraction(1))
        return _aligned_edges(lo, hi, _nice_width(lo, hi, n))
    if K.is_list(spec):
        a = spec.args
        if len(a) == 2 and a[0] == "Log":
            pos = [v for v in vals if v > 0]
            if not pos:
                raise ValueError(f"{fname}: logarithmic bins need positive data")
            lv = [math.log10(float(v)) for v in pos]
            le = bin_edges(lv, a[1], fname, allow_fun=False)
            return [10.0 ** float(e) for e in le]
        if len(a) == 1 and K.is_list(a[0]):
            edges = [real_number(e, fname, "a bin edge") for e in a[0].args]
            if len(edges) < 2 or any(e2 <= e1 for e1, e2 in zip(edges, edges[1:])):
                raise ValueError(f"{fname}: bin edges must be at least two increasing numbers")
            return edges
        if len(a) == 1:
            w = pos_number(a[0], fname, "the bin width")
            return _aligned_edges(lo, hi, w)
        if len(a) == 3:
            mn = real_number(a[0], fname, "the bin minimum")
            mx = real_number(a[1], fname, "the bin maximum")
            w = pos_number(a[2], fname, "the bin width")
            if not mx > mn:
                raise ValueError(f"{fname}: the bin specification needs min < max")
            k = math.ceil((mx - mn) / w)
            return [mn + w * i for i in range(k + 1)]
        raise ValueError(f"{fname}: {K_input(spec)} is not a valid bin specification")
    if allow_fun and (isinstance(spec, Symbol) or (isinstance(spec, Expr) and spec.head == "Function")):
        r = ev(Expr(spec, data_expr if data_expr is not None else L(*[_out(v) for v in vals])))
        if not K.is_list(r):
            raise ValueError(f"{fname}: the bin function returned {K_input(r)}, not a list of bin edges")
        return bin_edges(vals, L(r), fname, allow_fun=False)
    raise ValueError(f"{fname}: {K_input(spec)} is not a valid bin specification")


def _out(v):
    """Kernel value of a Fraction/float."""
    if isinstance(v, Fraction):
        return v.numerator if v.denominator == 1 else Expr("Rational", v.numerator, v.denominator)
    return float(v)


def _lt(a, b):
    """a < b comparing exact with exact and inexact as floats."""
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a < b
    return float(a) < float(b)


def bin_counts(vals, edges):
    """Counts of the values in the half-open bins [e_i, e_(i+1))."""
    import bisect
    fe = [float(e) for e in edges]
    counts = [0] * (len(edges) - 1)
    for v in vals:
        if _lt(v, edges[0]) or not _lt(v, edges[-1]):
            continue
        if isinstance(v, Fraction) and all(isinstance(e, Fraction) for e in edges):
            i = bisect.bisect_right(edges, v) - 1
        else:
            i = bisect.bisect_right(fe, float(v)) - 1
        counts[min(max(i, 0), len(counts) - 1)] += 1
    return counts


HEIGHTS = ("Count", "CumulativeCount", "SurvivalCount", "Probability", "Intensity", "PDF", "CDF", "SF",
           "ProbabilityDensity")


def heights(counts, widths, ntotal, hspec, fname, bins_expr=None):
    """Bar heights (floats) for a height specification."""
    if hspec is None or is_sym(hspec, "Automatic"):
        hspec = "Count"
    if isinstance(hspec, str):
        if hspec in ("HF", "CHF"):
            raise NotImplementedError(f"{fname}: the height specification {hspec!r} is not supported")
        if hspec not in HEIGHTS:
            raise ValueError(f"{fname}: unknown height specification {hspec!r}")
        cum, acc = [], 0
        for c in counts:
            acc += c
            cum.append(acc)
        surv = [ntotal - (cum[i - 1] if i else 0) for i in range(len(counts))]
        N = float(ntotal) if ntotal else 1.0
        if hspec == "Count":
            return [float(c) for c in counts]
        if hspec == "CumulativeCount":
            return [float(c) for c in cum]
        if hspec == "SurvivalCount":
            return [float(c) for c in surv]
        if hspec == "Probability":
            return [c / N for c in counts]
        if hspec == "Intensity":
            return [c / w for c, w in zip(counts, widths)]
        if hspec in ("PDF", "ProbabilityDensity"):
            return [c / (N * w) for c, w in zip(counts, widths)]
        if hspec == "CDF":
            return [c / N for c in cum]
        return [c / N for c in surv]                     # SF
    if K.is_list(hspec) and len(hspec.args) == 2 and hspec.args[0] == "Log":
        raise NotImplementedError(f"{fname}: logarithmic height specifications are not supported")
    if isinstance(hspec, Symbol) or (isinstance(hspec, Expr) and hspec.head == "Function"):
        r = ev(Expr(hspec, bins_expr, L(*counts)))
        if not K.is_list(r) or len(r.args) != len(counts):
            raise ValueError(f"{fname}: the height function must return one height per bin, not {K_input(r)}")
        out = []
        for h in r.args:
            v = num(h)
            if v is None:
                raise ValueError(f"{fname}: the height function returned {K_input(h)}, not a real number")
            out.append(v)
        return out
    raise ValueError(f"{fname}: {K_input(hspec)} is not a valid height specification")


def bins_expr(edges):
    return L(*[L(_out(a), _out(b)) for a, b in zip(edges, edges[1:])])


# ───────────────────────────── Histogram ─────────────────────────────

_HIST_DEFAULTS = {"ChartStyle": AUTO, "ChartBaseStyle": AUTO, "ChartLayout": AUTO, "BarOrigin": Symbol("Bottom"),
                  "PlotRange": AUTO, "ChartLegends": NONE, "ChartElementFunction": AUTO, "ChartLabels": NONE,
                  "LabelingFunction": AUTO, "PerformanceGoal": AUTO, "PlotTheme": AUTO, "ScalingFunctions": NONE,
                  "ChartElements": AUTO, "ColorFunction": AUTO, "ColorFunctionScaling": True,
                  "PlotInteractivity": AUTO, "Method": AUTO}
_TRIVIAL = {"Automatic", "None", "Inherited"}


def _unsupported(o, fname, allowed):
    for n, v in o.items():
        if n in allowed or n == "PlotRange":
            continue
        if is_sym(v, *_TRIVIAL) or (n == "ColorFunctionScaling" and v is True) or \
                (n == "PerformanceGoal" and v in ("Quality", "Speed")) or (n == "PlotTheme" and v == "Default") or \
                (n == "BarOrigin" and is_sym(v, "Bottom")):
            continue
        raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


def _legend(g, o, styles, fname, labels=None):
    spec = o.get("ChartLegends", NONE)
    if is_sym(spec, "None"):
        return g
    if K.is_list(spec):
        labs = list(spec.args)
    elif is_sym(spec, "Automatic") and labels is not None:
        labs = labels
    else:
        raise NotImplementedError(f"{fname}: ChartLegends -> {K_input(spec)} is not supported; give a list of labels")
    styles = [s if s is not None else Expr("GrayLevel", 0) for s in styles]
    n = min(len(labs), len(styles))
    return Expr("Legended", g, Expr("Placed", Expr("SwatchLegend", L(*styles[:n]), L(*labs[:n])), Symbol("After")))


def _range_2d(o, fname, xr, yr):
    return P2.resolve_plot_range(o["PlotRange"], fname, [xr, yr])


def histogram_1d(fname, data, bspec, hspec, o, gopts, orient="Bottom"):
    sets = datasets_1d(data, fname)
    allv = [v for vs, _ in sets for v in vs]
    if not allv:
        raise ValueError(f"{fname}: the data has no numeric values")
    edges = bin_edges(allv, bspec, fname)
    widths = [float(b) - float(a) for a, b in zip(edges, edges[1:])]
    layout = o.get("ChartLayout", AUTO)
    if not (is_sym(layout, "Automatic") or layout in ("Overlapped", "Stacked")):
        raise NotImplementedError(f"{fname}: ChartLayout -> {K_input(layout)} is not supported")
    stacked = layout == "Stacked"
    styles = chart_styles(o, len(sets), fname, opacity=0.5 if len(sets) > 1 and not stacked else None)
    content = []
    base = [0.0] * len(widths)
    hmax = 0.0
    for k, (vs, st) in enumerate(sets):
        cs = bin_counts(vs, edges)
        hs = heights(cs, widths, len(vs), hspec, fname, bins_expr(edges))
        rects = []
        for i, h in enumerate(hs):
            if h == 0:
                continue
            a, b = fval(edges[i]), fval(edges[i + 1])
            y0 = base[i] if stacked else 0.0
            y1 = y0 + h
            if orient == "Left":
                rects.append(Expr("Rectangle", pt(min(y0, y1), a), pt(max(y0, y1), b)))
            else:
                rects.append(Expr("Rectangle", pt(a, min(y0, y1)), pt(b, max(y0, y1))))
            hmax = max(hmax, y1)
        if stacked:
            base = [b_ + h for b_, h in zip(base, hs)]
        s = styles[k]
        if s is None:
            continue
        if st is not None:
            s = Expr("Directive", *s.args, *_directive_items(st))
        content.append(L(s, *rects))
    xr = (fval(edges[0]), fval(edges[-1]))
    yr = (min(0.0, hmax), max(hmax, 0.0) if hmax else 1.0)
    if orient == "Left":
        r = _range_2d(o, fname, yr, xr)
    else:
        r = _range_2d(o, fname, xr, yr)
    defaults = [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r)),
                ("PlotRangePadding", AUTO)]
    g = finish_graphics(L(*content), gopts, defaults)
    return _legend(g, o, styles, fname)


def _hist_args(args, fname, lo=1, hi=3):
    pos, opts = split_args(args, fname, lo, hi)
    return pos, opts


_HIST_NOTE = ("Bin specifications: Automatic (Freedman-Diaconis width rounded to a nice width), n (the nice width "
              "giving about n bins), {dx} (bins at multiples of dx), {min, max, dx}, {{b1, b2, ...}}, \"Sturges\", "
              "\"Scott\", \"FreedmanDiaconis\" (target width rounded to a nice width), {\"Log\", bspec} (log10 bins) and "
              "a function fb applied to the data list; \"Knuth\" and \"Wand\" raise NotImplementedError. Height "
              "specifications: \"Count\", \"CumulativeCount\", \"SurvivalCount\", \"Probability\", \"Intensity\" "
              "(count/width), \"PDF\" (count/(n width)), \"CDF\", \"SF\" and a function fh[bins, counts]; \"HF\", "
              "\"CHF\" and {\"Log\", h} raise NotImplementedError. Bins are [b_i, b_(i+1)); values outside the bins "
              "are not counted but n is the number of numeric values; missing and non-real values are dropped.")


@fn("Histogram", area=AREA, status="added",
    notes="Histogram[data], [data, bspec], [data, bspec, hspec] and lists/associations of datasets: Graphics[{{style, "
          "Rectangle[{b_i, 0}, {b_(i+1), h_i}], ...}, ...}] (zero bars omitted), one group per dataset (default "
          "ChartLayout \"Overlapped\" with Opacity[0.5]; \"Stacked\" stacks), AspectRatio 1/GoldenRatio, Axes. "
          + _HIST_NOTE + " Options: ChartStyle, ChartBaseStyle, ChartLayout, ChartLegends (labels), BarOrigin "
          "(Bottom, Left), PlotRange and Graphics options; other settings raise NotImplementedError. Symbolic data "
          "stays unevaluated; strings raise TypeError. Check: exact (integer counts; tests compare with BinCounts "
          "and independently computed bins).")
def _histogram(data: ListOf[Real] | ListOf[ListOf[Real]], *rest: Expr | Option) -> Expr:
    fname = "Histogram"
    pos, opts = _hist_args((data,) + rest, fname)
    o, gopts = split_options(fname, opts, _HIST_DEFAULTS)
    _unsupported(o, fname, ("ChartStyle", "ChartBaseStyle", "ChartLayout", "ChartLegends", "BarOrigin"))
    orient = o["BarOrigin"]
    if not is_sym(orient, "Bottom", "Left"):
        raise NotImplementedError(f"{fname}: BarOrigin -> {K_input(orient)} is not supported (Bottom or Left)")
    bspec = pos[1] if len(pos) > 1 else None
    hspec = pos[2] if len(pos) > 2 else None
    return histogram_1d(fname, pos[0], bspec, hspec, o, gopts, orient=orient.name)


# ───────────────────────────── 2-D histograms ─────────────────────────────

def values_2d(d, fname):
    """[(x, y)] of a list of pairs (missing / non-real pairs dropped; symbolic -> Unevaluated)."""
    d, _st, _w = P2._unwrap(d, fname)
    if isinstance(d, Expr) and d.head in ("WeightedData", "Tabular", "Dataset", "TimeSeries", "QuantityArray"):
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of pairs {{x, y}}, not {K_input(d)}")
    out = []
    for p in d.args:
        if is_missing(p):
            continue
        if not K.is_list(p) or len(p.args) != 2:
            if isinstance(p, (int, float, str)) or K.is_list(p):
                raise ValueError(f"{fname}: the data must be a list of pairs {{x, y}}; {K_input(p)} is not a pair")
            if isinstance(p, (Symbol, Expr)):
                raise Unevaluated()
        xs = [P2.scalar(c, fname) for c in p.args]
        if any(v is None for v in xs):
            continue
        out.append(tuple(_frac(c) if isinstance(_frac(c), Fraction) else v for c, v in zip(p.args, xs)))
    if not out:
        raise ValueError(f"{fname}: the data has no numeric pairs")
    return out


def _is_1d_spec(s):
    return (s is None or is_sym(s, "Automatic") or isinstance(s, str) or (K.is_int(s) and not isinstance(s, bool))
            or K.is_list(s) or isinstance(s, Symbol) or (isinstance(s, Expr) and s.head == "Function"))


def split_2d_spec(spec, fname):
    """(xspec, yspec) of a 2-D bin specification."""
    if K.is_list(spec) and len(spec.args) == 2 and spec.args[0] != "Log" and all(_is_1d_spec(a) for a in spec.args) \
            and not all(num(a) is not None and not K.is_int(a) for a in spec.args):
        return spec.args[0], spec.args[1]
    return spec, spec


def heights_2d(C, xe, ye, ntotal, hspec, fname):
    np = _np()
    C = np.array(C, dtype=float)
    wx = np.array([float(b) - float(a) for a, b in zip(xe, xe[1:])])
    wy = np.array([float(b) - float(a) for a, b in zip(ye, ye[1:])])
    area = np.outer(wx, wy)
    N = float(ntotal) or 1.0
    if hspec is None or is_sym(hspec, "Automatic"):
        hspec = "Count"
    if isinstance(hspec, str):
        if hspec == "Count":
            return C
        if hspec == "Probability":
            return C / N
        if hspec == "Intensity":
            return C / area
        if hspec in ("PDF", "ProbabilityDensity"):
            return C / (N * area)
        if hspec in ("CumulativeCount", "CDF"):
            cum = C.cumsum(axis=0).cumsum(axis=1)
            return cum if hspec == "CumulativeCount" else cum / N
        if hspec in ("SurvivalCount", "SF"):
            sv = C[::-1, ::-1].cumsum(axis=0).cumsum(axis=1)[::-1, ::-1]
            return sv if hspec == "SurvivalCount" else sv / N
        if hspec in ("HF", "CHF"):
            raise NotImplementedError(f"{fname}: the height specification {hspec!r} is not supported")
        raise ValueError(f"{fname}: unknown height specification {hspec!r}")
    if K.is_list(hspec) and len(hspec.args) == 2 and hspec.args[0] == "Log":
        raise NotImplementedError(f"{fname}: logarithmic height specifications are not supported")
    if isinstance(hspec, Symbol) or (isinstance(hspec, Expr) and hspec.head == "Function"):
        r = ev(Expr(hspec, bins_expr(xe), bins_expr(ye), L(*[L(*[int(c) for c in row]) for row in C])))
        try:
            H = np.array([[num(v) for v in row.args] for row in r.args], dtype=float)
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f"{fname}: the height function must return a matrix of numbers, not {K_input(r)}")
        if H.shape != C.shape or not np.isfinite(H).all():
            raise ValueError(f"{fname}: the height function must return a {C.shape[0]} x {C.shape[1]} matrix of "
                             "real numbers")
        return H
    raise ValueError(f"{fname}: {K_input(hspec)} is not a valid height specification")


def hist2d(pairs, bspec, hspec, fname, data_expr=None):
    """(x edges, y edges, heights[i][j]) with counts of x in [x_i, x_(i+1)) and y in [y_j, y_(j+1))."""
    sx, sy = split_2d_spec(bspec, fname)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    xe = bin_edges(xs, sx, fname)
    ye = bin_edges(ys, sy, fname)
    import bisect
    fx, fy = [float(e) for e in xe], [float(e) for e in ye]
    C = [[0] * (len(ye) - 1) for _ in range(len(xe) - 1)]
    for x, y in pairs:
        if _lt(x, xe[0]) or not _lt(x, xe[-1]) or _lt(y, ye[0]) or not _lt(y, ye[-1]):
            continue
        i = bisect.bisect_right(fx, float(x)) - 1
        j = bisect.bisect_right(fy, float(y)) - 1
        # exact edges: re-check with exact comparisons
        while i > 0 and _lt(x, xe[i]):
            i -= 1
        while i < len(xe) - 2 and not _lt(x, xe[i + 1]):
            i += 1
        while j > 0 and _lt(y, ye[j]):
            j -= 1
        while j < len(ye) - 2 and not _lt(y, ye[j + 1]):
            j += 1
        C[i][j] += 1
    H = heights_2d(C, xe, ye, len(pairs), hspec, fname)
    return xe, ye, H


_H2_NOTE = (" 2-D bin specifications: one 1-D specification for both coordinates (Automatic, n, {dx}, {min, max, "
            "dx}, {{b1, ...}}, a rule name, {\"Log\", spec}) or {xspec, yspec}; heights \"Count\", \"Probability\", "
            "\"Intensity\" (count/area), \"PDF\", \"CumulativeCount\"/\"CDF\" and \"SurvivalCount\"/\"SF\" (2-D "
            "cumulative sums) or fh[xbins, ybins, counts]; \"HF\", \"CHF\" and log heights raise NotImplementedError. "
            "Bins are half-open; pairs with missing or non-real coordinates are dropped.")

_H2_DEFAULTS = dict(_HIST_DEFAULTS)
_H2_DEFAULTS.update({"ColorFunction": AUTO, "ColorFunctionScaling": True, "PlotLegends": NONE, "Mesh": NONE,
                     "MeshStyle": AUTO, "BubbleSizes": AUTO, "BubbleScale": AUTO, "PlotStyle": AUTO,
                     "ClippingStyle": NONE})


def _cf_values(o, fname, H):
    """Colours rgb(cf(t)) for the heights (scaled to [0, 1] by the maximum unless ColorFunctionScaling False)."""
    np = _np()
    cf = color_function(o.get("ColorFunction", AUTO), fname, lambda t: blend(DENSITY_GRADIENT, t))
    hmin = float(np.min(H[H > 0])) if (H > 0).any() else 0.0
    hmax = float(np.max(H)) if H.size else 1.0
    scale = o.get("ColorFunctionScaling", True) is not False

    def col(h):
        if not scale:
            return rgb(cf(h))
        t = (h - hmin) / (hmax - hmin) if hmax > hmin else 1.0
        return rgb(cf(t))
    return col


def _hist2d_args(args, fname):
    pos, opts = split_args(args, fname, 1, 3)
    o, gopts = split_options(fname, opts, _H2_DEFAULTS)
    return pos, o, gopts


@fn("Histogram3D", area=AREA, status="added",
    notes="Histogram3D[{{x1, y1}, ...}], [data, bspec], [data, bspec, hspec] and lists of datasets: Graphics3D[{{"
          "style, Cuboid[{x_i, y_j, 0}, {x_(i+1), y_(j+1), h_ij}], ...}, ...}] (zero bins omitted), BoxRatios {1, 1, "
          "0.4}, Axes." + _H2_NOTE + " Options: ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and Graphics3D "
          "options; others raise NotImplementedError. Symbolic data stays unevaluated. Check: exact (integer "
          "counts compared with independent counting in tests).")
def _histogram3d(data: MatrixOf[Real] | ListOf[MatrixOf[Real]], *rest: Expr | Option) -> Expr:
    fname = "Histogram3D"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _HIST_DEFAULTS, dim=3)
    _unsupported(o, fname, ("ChartStyle", "ChartBaseStyle", "ChartLegends"))
    du, _, _ = P2._unwrap(pos[0], fname)
    multi = K.is_list(du) and du.args and all(K.is_list(a) and a.args and all(K.is_list(p) for p in a.args)
                                              for a in du.args)
    sets = [values_2d(a, fname) for a in du.args] if multi else [values_2d(pos[0], fname)]
    allp = [p for s in sets for p in s]
    bspec = pos[1] if len(pos) > 1 else None
    hspec = pos[2] if len(pos) > 2 else None
    sx, sy = split_2d_spec(bspec, fname)
    xe = bin_edges([p[0] for p in allp], sx, fname)
    ye = bin_edges([p[1] for p in allp], sy, fname)
    fixed = L(L(L(*[_out(e) for e in xe])), L(L(*[_out(e) for e in ye])))
    styles = chart_styles(o, len(sets), fname, opacity=0.5 if len(sets) > 1 else None)
    content, hmax = [], 0.0
    for s, st in zip(sets, styles):
        _, _, H = hist2d(s, fixed, hspec, fname)
        cubes = []
        for i in range(len(xe) - 1):
            for j in range(len(ye) - 1):
                h = float(H[i][j])
                if h == 0:
                    continue
                hmax = max(hmax, h)
                cubes.append(Expr("Cuboid", pt(xe[i], ye[j], 0.0), pt(xe[i + 1], ye[j + 1], h)))
        if st is not None:
            content.append(L(st, *cubes))
    r = P2.resolve_plot_range(o["PlotRange"], fname, [(fval(xe[0]), fval(xe[-1])), (fval(ye[0]), fval(ye[-1])),
                                                      (0.0, hmax or 1.0)], dims=3)
    g = finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 1, 0.4)),
                                             ("PlotRange", range_expr(r))], dim=3)
    return _legend(g, o, styles, fname)


def density_hist_content(pairs, bspec, hspec, o, fname):
    xe, ye, H = hist2d(pairs, bspec, hspec, fname)
    col = _cf_values(o, fname, H)
    content = []
    for i in range(len(xe) - 1):
        for j in range(len(ye) - 1):
            h = float(H[i][j])
            if h == 0:
                continue
            content.append(L(col(h), Expr("Rectangle", pt(xe[i], ye[j]), pt(xe[i + 1], ye[j + 1]))))
    return content, xe, ye, H


@fn("DensityHistogram", area=AREA, status="added",
    notes="DensityHistogram[{{x1, y1}, ...}], [data, bspec], [data, bspec, hspec]: Graphics[{{colour_ij, "
          "Rectangle[{x_i, y_j}, {x_(i+1), y_(j+1)}]}, ...}] with colour ColorFunction((h - hmin)/(hmax - hmin)) "
          "(hmin the smallest non-zero height; default gradient as DensityPlot, an approximation of Wolfram's); "
          "empty bins are not drawn; Frame, AspectRatio 1." + _H2_NOTE + " Options: ColorFunction (colour head "
          "or pure function), ColorFunctionScaling, PlotRange and Graphics options; others raise "
          "NotImplementedError. Check: exact (counts compared with independent counting in tests).")
def _densityhistogram(data: MatrixOf[Real], *rest: Expr | Option) -> Expr:
    fname = "DensityHistogram"
    pos, o, gopts = _hist2d_args((data,) + rest, fname)
    _unsupported(o, fname, ("ColorFunction", "ColorFunctionScaling"))
    pairs = values_2d(pos[0], fname)
    content, xe, ye, _ = density_hist_content(pairs, pos[1] if len(pos) > 1 else None,
                                              pos[2] if len(pos) > 2 else None, o, fname)
    r = _range_2d(o, fname, (fval(xe[0]), fval(xe[-1])), (fval(ye[0]), fval(ye[-1])))
    return finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr(r)),
                                                ("PlotRangePadding", 0)])


def _bubble_sizes(spec, fname, default=(0.2, 1.0)):
    if spec is None or is_sym(spec, "Automatic"):
        return default
    if K.is_list(spec) and len(spec.args) == 2:
        a = num(spec.args[0])
        b = num(spec.args[1])
        if a is None or b is None or a < 0 or b < 0:
            raise ValueError(f"{fname}: BubbleSizes must be {{smin, smax}} with non-negative numbers")
        return a, b
    raise NotImplementedError(f"{fname}: BubbleSizes -> {K_input(spec)} is not supported; give {{smin, smax}}")


def _bubble_scale(spec, fname, default):
    if spec is None or is_sym(spec, "Automatic"):
        return default
    if spec in ("Diameter", "Area"):
        return spec
    raise NotImplementedError(f"{fname}: BubbleScale -> {K_input(spec)} is not supported (\"Diameter\", \"Area\")")


def bubble_diameters(zs, smin, smax, scale):
    """Diameters as fractions in [smin, smax]: the measure (diameter, or area for "Area") is interpolated linearly
    between the smallest and largest z."""
    zlo, zhi = min(zs), max(zs)
    out = []
    for z in zs:
        t = (z - zlo) / (zhi - zlo) if zhi > zlo else 1.0
        if scale == "Area":
            out.append(math.sqrt(smin ** 2 + t * (smax ** 2 - smin ** 2)))
        else:
            out.append(smin + t * (smax - smin))
    return out


@fn("BubbleHistogram", area=AREA, status="added",
    notes="BubbleHistogram[{{x1, y1}, ...}], [data, bspec], [data, bspec, hspec]: Graphics[{{colour, Disk[centre_ij, "
          "r_ij]}, ...}] with one bubble per non-empty bin at its centre. As documented, BubbleScale \"Diameter\" "
          "(default) makes the diameter proportional to the height scale and BubbleSizes {0.2, 1} (default) maps "
          "the smallest height to 20% and the largest to 100% of the bin size (the smaller bin side); here the "
          "diameter (or area, BubbleScale -> \"Area\") is interpolated linearly between those. Colour "
          "ColorFunction of the scaled height (default gradient as DensityHistogram); Frame, AspectRatio 1."
          + _H2_NOTE + " Options: BubbleSizes, BubbleScale, ColorFunction, ColorFunctionScaling, PlotRange and "
          "Graphics options; others raise NotImplementedError. Check: exact (counts) and reference (bubble "
          "diameters from the documented size rule in tests).")
def _bubblehistogram(data: MatrixOf[Real], *rest: Expr | Option) -> Expr:
    fname = "BubbleHistogram"
    pos, o, gopts = _hist2d_args((data,) + rest, fname)
    _unsupported(o, fname, ("ColorFunction", "ColorFunctionScaling", "BubbleSizes", "BubbleScale"))
    pairs = values_2d(pos[0], fname)
    xe, ye, H = hist2d(pairs, pos[1] if len(pos) > 1 else None, pos[2] if len(pos) > 2 else None, fname)
    smin, smax = _bubble_sizes(o["BubbleSizes"], fname)
    scale = _bubble_scale(o["BubbleScale"], fname, "Diameter")
    col = _cf_values(o, fname, H)
    cells = [(i, j, float(H[i][j])) for i in range(len(xe) - 1) for j in range(len(ye) - 1) if H[i][j] != 0]
    content = []
    if cells:
        ds = bubble_diameters([c[2] for c in cells], smin, smax, scale)
        for (i, j, h), dfrac in zip(cells, ds):
            w = min(fval(xe[i + 1]) - fval(xe[i]), fval(ye[j + 1]) - fval(ye[j]))
            c = ((fval(xe[i]) + fval(xe[i + 1])) / 2, (fval(ye[j]) + fval(ye[j + 1])) / 2)
            content.append(L(col(h), GRAY_EDGE, Expr("Disk", pt(*c), fl(dfrac * w / 2))))
    r = _range_2d(o, fname, (fval(xe[0]), fval(xe[-1])), (fval(ye[0]), fval(ye[-1])))
    return finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr(r)),
                                                ("PlotRangePadding", 0)])


def _matrix_columns(data, fname):
    du, _, _ = P2._unwrap(data, fname)
    if not K.is_list(du) or not du.args:
        if isinstance(du, (Symbol, Expr)) and not K.is_list(du):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a matrix {{{{y11, ..., y1n}}, ...}}, not {K_input(du)}")
    if not all(K.is_list(r) for r in du.args):
        raise ValueError(f"{fname}: the data must be a matrix (a list of rows)")
    n = len(du.args[0].args)
    if n < 2 or any(len(r.args) != n for r in du.args):
        raise ValueError(f"{fname}: every row must have the same number (>= 2) of columns")
    return [[r.args[j] for r in du.args] for j in range(n)]


def pairwise_grid(fname, data, opts, panel, diag):
    """GraphicsGrid of n x n panels: panel {i, j} (i != j) = panel(points {y_kj, y_ki}), diagonal = diag(column i)."""
    o, gopts = split_options(fname, opts, _H2_DEFAULTS)
    _unsupported(o, fname, ("ColorFunction", "ColorFunctionScaling", "ChartStyle"))
    cols = _matrix_columns(data, fname)
    n = len(cols)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                vals = values_of(L(*cols[i]), fname)
                row.append(diag(vals, o))
                continue
            pairs = []
            for a, b in zip(cols[j], cols[i]):
                xa, xb = P2.scalar(a, fname), P2.scalar(b, fname)
                if xa is not None and xb is not None:
                    pairs.append((_frac(a) if isinstance(_frac(a), Fraction) else xa,
                                  _frac(b) if isinstance(_frac(b), Fraction) else xb))
            row.append(panel(pairs, o))
        rows.append(L(*row))
    return Expr("GraphicsGrid", L(*rows), *gopts)


def _diag_hist(fname):
    def diag(vals, o):
        g = histogram_1d(fname, L(*[_out(v) for v in vals]), None, None, o, [], "Bottom")
        return g
    return diag


@fn("PairwiseDensityHistogram", area=AREA, status="added",
    notes="PairwiseDensityHistogram[{{y11, ..., y1n}, ...}]: GraphicsGrid of n x n panels; panel {i, j} (i != j) is "
          "the DensityHistogram (automatic bins) of the points {y_kj, y_ki}, the diagonal panel i the Histogram of "
          "column i. Rows with missing values in a pair are dropped for that panel. Options: ColorFunction, "
          "ColorFunctionScaling, ChartStyle (diagonal histograms) and GraphicsGrid options; others raise "
          "NotImplementedError. Check: exact (panel counts compared with DensityHistogram of the column pairs in "
          "tests).")
def _pairwisedensityhistogram(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "PairwiseDensityHistogram"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)

    def panel(pairs, o):
        content, xe, ye, _ = density_hist_content(pairs, None, None, o, fname)
        return Expr("Graphics", L(*content), Expr("Rule", Symbol("Frame"), True),
                    Expr("Rule", Symbol("AspectRatio"), 1),
                    Expr("Rule", Symbol("PlotRange"), range_expr([(fval(xe[0]), fval(xe[-1])),
                                                                  (fval(ye[0]), fval(ye[-1]))])))
    return pairwise_grid(fname, pos[0], opts_, panel, _diag_hist(fname))


# ───────────────────────────── paired, categorical and date histograms ─────────────────────────────

@fn("PairedHistogram", area=AREA, status="added",
    notes="PairedHistogram[data1, data2], [data1, data2, bspec], [data1, data2, bspec, hspec] and lists of datasets "
          "on each side: back-to-back histograms with the bins along the vertical axis, data1 to the left "
          "(Rectangle[{-h_i, b_i}, {0, b_(i+1)}]) and data2 to the right (Rectangle[{0, b_i}, {h_i, b_(i+1)}]); "
          "automatic bins are computed from all datasets combined (as documented). " + _HIST_NOTE + " Options: "
          "ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and Graphics options; others raise "
          "NotImplementedError. Check: exact (counts compared with Histogram of each side in tests).")
def _pairedhistogram(d1: ListOf[Real], d2: ListOf[Real], *rest: Expr | Option) -> Expr:
    fname = "PairedHistogram"
    pos, opts = split_args((d1, d2) + rest, fname, 2, 4)
    o, gopts = split_options(fname, opts, _HIST_DEFAULTS)
    _unsupported(o, fname, ("ChartStyle", "ChartBaseStyle", "ChartLegends"))
    left = datasets_1d(pos[0], fname)
    right = datasets_1d(pos[1], fname)
    allv = [v for vs, _ in left + right for v in vs]
    edges = bin_edges(allv, pos[2] if len(pos) > 2 else None, fname)
    widths = [float(b) - float(a) for a, b in zip(edges, edges[1:])]
    hspec = pos[3] if len(pos) > 3 else None
    nl, nr = len(left), len(right)
    styles = chart_styles(o, nl + nr, fname, opacity=0.5 if max(nl, nr) > 1 else None)
    content, hmax = [], 0.0
    for side, sets, off in ((-1, left, 0), (1, right, nl)):
        for k0, (vs, st) in enumerate(sets):
            k = off + k0
            hs = heights(bin_counts(vs, edges), widths, len(vs), hspec, fname, bins_expr(edges))
            rs = []
            for i, h in enumerate(hs):
                if h == 0:
                    continue
                hmax = max(hmax, abs(h))
                a, b = fval(edges[i]), fval(edges[i + 1])
                x0, x1 = (min(0.0, side * h), max(0.0, side * h))
                rs.append(Expr("Rectangle", pt(x0, a), pt(x1, b)))
            s = styles[k]
            if s is not None:
                content.append(L(s, *rs))
    r = _range_2d(o, fname, (-(hmax or 1.0), hmax or 1.0), (fval(edges[0]), fval(edges[-1])))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


def _category_items(d, fname):
    d, _st, _w = P2._unwrap(d, fname)
    if isinstance(d, Expr) and d.head == "Association":
        d = L(*[v for _, v in P2._assoc_items(d)])
    if isinstance(d, Expr) and d.head in ("WeightedData", "Tabular", "Dataset", "TimeSeries", "EventSeries"):
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of category values, not {K_input(d)}")
    out = []
    for c in d.args:
        if is_missing(c):
            continue
        if isinstance(c, Expr) and c.head in ("Nominal", "Ordinal"):
            raise NotImplementedError(f"{fname}: {c.head} category values are not supported")
        out.append(c)
    return out


def _cat_key(c):
    return K_input(c)


@fn("CategoricalHistogram", area=AREA, status="added",
    notes="CategoricalHistogram[{c1, c2, ...}], [data, spec], [data, spec, hspec] and lists of datasets: one bar per "
          "category at x = 1, 2, ... (Rectangle[{k - 0.4, 0}, {k + 0.4, h_k}]; several datasets share the slot, "
          "side by side), with Ticks labelling the categories. Categories (any expressions, compared structurally) "
          "are in decreasing order of frequency (ties: first appearance) as documented; spec: Automatic, "
          "\"Decreasing\", \"Increasing\", n (the n most common) or {cat1, cat2, ...} (only these, in this order). "
          "Heights: \"Count\", \"Probability\" (fraction of all values), \"CumulativeCount\", \"CDF\" (in bar "
          "order); other specs, Nominal/Ordinal values, OrdinalScale and property associations raise "
          "NotImplementedError. Missing values are dropped. Options: ChartStyle, ChartBaseStyle, ChartLegends, "
          "PlotRange and Graphics options. Check: exact (counts compared with Tally in tests).")
def _categoricalhistogram(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "CategoricalHistogram"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _HIST_DEFAULTS)
    _unsupported(o, fname, ("ChartStyle", "ChartBaseStyle", "ChartLegends"))
    du, _, _ = P2._unwrap(pos[0], fname)
    multi = K.is_list(du) and du.args and all(K.is_list(a) for a in du.args)
    sets = [_category_items(a, fname) for a in du.args] if multi else [_category_items(pos[0], fname)]
    tally, first, rep = {}, {}, {}
    for s in sets:
        for c in s:
            k = _cat_key(c)
            tally[k] = tally.get(k, 0) + 1
            first.setdefault(k, len(first))
            rep.setdefault(k, c)
    if not tally:
        raise ValueError(f"{fname}: the data has no category values")
    spec = pos[1] if len(pos) > 1 else AUTO
    keys = sorted(tally, key=lambda k: (-tally[k], first[k]))
    if is_sym(spec, "Automatic") or spec == "Decreasing":
        pass
    elif spec == "Increasing":
        keys = sorted(tally, key=lambda k: (tally[k], first[k]))
    elif K.is_int(spec) and not isinstance(spec, bool):
        keys = keys[:pos_int(spec, fname, "the number of categories")]
    elif K.is_list(spec):
        keys = []
        for c in spec.args:
            if is_rule(c):
                raise NotImplementedError(f"{fname}: category labels cat -> lab are not supported")
            keys.append(_cat_key(c))
            rep.setdefault(_cat_key(c), c)
    else:
        raise NotImplementedError(f"{fname}: the category specification {K_input(spec)} is not supported")
    hspec = pos[2] if len(pos) > 2 else "Count"
    if hspec not in ("Count", "Probability", "CumulativeCount", "CDF"):
        raise NotImplementedError(f"{fname}: the height specification {K_input(hspec)} is not supported")
    styles = chart_styles(o, len(sets), fname)
    m = len(sets)
    w = 0.8 / m
    content, hmax = [], 0.0
    for d, (s, st) in enumerate(zip(sets, styles)):
        cnt = {}
        for c in s:
            cnt[_cat_key(c)] = cnt.get(_cat_key(c), 0) + 1
        hs = [float(cnt.get(k, 0)) for k in keys]
        if hspec in ("CumulativeCount", "CDF"):
            acc, cum = 0.0, []
            for h in hs:
                acc += h
                cum.append(acc)
            hs = cum
        if hspec in ("Probability", "CDF"):
            hs = [h / len(s) if s else 0.0 for h in hs]
        rs = []
        for k, h in enumerate(hs):
            if h == 0:
                continue
            x0 = k + 1 - 0.4 + d * w
            rs.append(Expr("Rectangle", pt(x0, 0.0), pt(x0 + w, h)))
            hmax = max(hmax, h)
        if st is not None:
            content.append(L(st, *rs))
    ticks = L(L(*[L(k + 1, rep[key]) for k, key in enumerate(keys)]), AUTO)
    r = _range_2d(o, fname, (0.5, len(keys) + 0.5), (0.0, hmax or 1.0))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("Ticks", ticks),
                                             ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


_DATE_UNITS = ("Year", "Quarter", "Month", "Week", "Day", "Hour", "Minute", "Second")


def date_values(d, fname):
    d, _, _ = P2._unwrap(d, fname)
    if isinstance(d, Expr) and d.head in ("TimeSeries", "EventSeries", "TemporalData", "WeightedData"):
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of dates, not {K_input(d)}")
    out = []
    for x in d.args:
        if is_missing(x):
            continue
        out.append(P2.abs_time(x, fname))
    if not out:
        raise ValueError(f"{fname}: the data has no dates")
    return out


def _to_dt(t):
    import datetime as dt
    return dt.datetime(1900, 1, 1) + dt.timedelta(seconds=t)


def _from_dt(d):
    import datetime as dt
    return (d - dt.datetime(1900, 1, 1)).total_seconds()


def _floor_unit(d, unit, k=1):
    import datetime as dt
    if unit == "Year":
        return dt.datetime(d.year - (d.year % k if k > 1 else 0), 1, 1)
    if unit == "Quarter":
        return dt.datetime(d.year, 3 * ((d.month - 1) // 3) + 1, 1)
    if unit == "Month":
        return dt.datetime(d.year, d.month, 1)
    if unit == "Week":
        day = dt.datetime(d.year, d.month, d.day)
        return day - dt.timedelta(days=day.weekday())
    if unit == "Day":
        return dt.datetime(d.year, d.month, d.day)
    if unit == "Hour":
        return dt.datetime(d.year, d.month, d.day, d.hour)
    if unit == "Minute":
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute)
    return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)


def _add_unit(d, unit, k):
    import datetime as dt
    if unit in ("Year", "Quarter", "Month"):
        months = {"Year": 12, "Quarter": 3, "Month": 1}[unit] * k
        m = d.month - 1 + months
        return d.replace(year=d.year + m // 12, month=m % 12 + 1)
    secs = {"Week": 604800, "Day": 86400, "Hour": 3600, "Minute": 60, "Second": 1}[unit] * k
    return d + dt.timedelta(seconds=secs)


def date_edges(ts, spec, fname):
    """Calendar bin edges (AbsoluteTime) for a date bin specification."""
    lo, hi = min(ts), max(ts)
    if spec is None or is_sym(spec, "Automatic"):
        span = hi - lo
        unit = "Year" if span > 3 * 365.25 * 86400 else "Month" if span > 90 * 86400 else \
            "Day" if span > 3 * 86400 else "Hour" if span > 3 * 3600 else "Minute" if span > 180 else "Second"
        k = 1
    elif isinstance(spec, str):
        if spec not in _DATE_UNITS:
            raise ValueError(f"{fname}: {spec!r} is not a date bin unit ({', '.join(_DATE_UNITS)})")
        unit, k = spec, 1
    elif K.is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str):
        unit = spec.args[0]
        if unit not in _DATE_UNITS:
            raise ValueError(f"{fname}: {unit!r} is not a date bin unit")
        k = pos_int(spec.args[1], fname, "the number of units per bin")
        if unit in ("Quarter", "Week") and k != 1:
            raise NotImplementedError(f"{fname}: multiples of {unit} are not supported")
    else:
        raise NotImplementedError(f"{fname}: the date bin specification {K_input(spec)} is not supported; use a "
                                  "unit such as \"Month\" or {\"Day\", n}")
    d = _floor_unit(_to_dt(lo), unit)
    edges = [_from_dt(d)]
    while edges[-1] <= hi:
        d = _add_unit(d, unit, k)
        edges.append(_from_dt(d))
        if len(edges) > 100000:
            raise ValueError(f"{fname}: too many date bins; use a coarser unit")
    return edges


@fn("DateHistogram", area=AREA, status="added",
    notes="DateHistogram[{date1, date2, ...}], [dates, bspec], [dates, bspec, hspec] and lists of datasets: a "
          "histogram on the AbsoluteTime axis (seconds since 1 Jan 1900, as DateListPlot) with calendar bins: bspec "
          "\"Year\", \"Quarter\", \"Month\", \"Week\" (ISO weeks from Monday), \"Day\", \"Hour\", \"Minute\", "
          "\"Second\" or {unit, n}; Automatic picks the unit from the span (> 3 years: Year, > 90 days: Month, > 3 "
          "days: Day, > 3 hours: Hour, > 3 minutes: Minute, else Second). Dates: DateObject, date lists, date "
          "strings or AbsoluteTime numbers. Heights as Histogram (\"Count\", \"Probability\", \"PDF\" per second, "
          "...). Date ticks at whole calendar units. Options as Histogram. Check: exact (calendar bins and counts "
          "compared with hand-counted dates in tests).")
def _datehistogram(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "DateHistogram"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _HIST_DEFAULTS)
    _unsupported(o, fname, ("ChartStyle", "ChartBaseStyle", "ChartLegends", "ChartLayout"))
    du, _, _ = P2._unwrap(pos[0], fname)
    multi = K.is_list(du) and du.args and all(K.is_list(a) and a.args and all(P2.is_date(x) or is_missing(x)
                                                                              for x in a.args)
                                              and not P2.is_date(a) for a in du.args)
    sets = [date_values(a, fname) for a in du.args] if multi else [date_values(pos[0], fname)]
    edges = date_edges([t for s in sets for t in s], pos[1] if len(pos) > 1 else None, fname)
    data_lists = L(*[L(*s) for s in sets]) if multi else L(*sets[0])
    g = histogram_1d(fname, data_lists, L(L(*edges)), pos[2] if len(pos) > 2 else None, o, gopts)
    ticks = P2.date_ticks(edges[0], edges[-1])
    inner = unwrap_legend(g)
    new = Expr(inner.head, *inner.args, Expr("Rule", Symbol("Ticks"), L(ticks, AUTO)))
    return new if g is inner else Expr("Legended", new, *g.args[1:])


def unwrap_legend(g):
    return g.args[0] if isinstance(g, Expr) and g.head == "Legended" else g


# ───────────────────────────── smooth (kernel density) histograms ─────────────────────────────

SQ2PI = math.sqrt(2 * math.pi)


def _sd(xs):
    np = _np()
    return float(np.std(np.asarray(xs, dtype=float), ddof=1)) if len(xs) > 1 else 0.0


def bandwidth(xs, spec, fname, dim=1):
    """Gaussian kernel bandwidth: Automatic/"Silverman" 0.9 min(sd, IQR/1.34) n^(-1/5) (1-D) or sd n^(-1/6)
    per coordinate (2-D, Scott's rule), "Scott" 1.06 sd n^(-1/5), a number h, or {"Standardized", h} (h sd)."""
    xs = [float(x) for x in xs]
    n = len(xs)
    sd = _sd(xs)
    if spec is None or is_sym(spec, "Automatic") or spec in ("Silverman", "Scott"):
        if n < 2:
            raise ValueError(f"{fname}: an automatic bandwidth needs at least two data points")
        if dim == 2:
            h = sd * n ** (-1 / 6)
        elif spec == "Scott":
            h = 1.06 * sd * n ** (-0.2)
        else:
            iqr = _quantile_lin(xs, 0.75) - _quantile_lin(xs, 0.25)
            a = min(sd, iqr / 1.34) if iqr > 0 else sd
            h = 0.9 * a * n ** (-0.2)
        if not h > 0:
            raise ValueError(f"{fname}: the data has no spread; give an explicit bandwidth")
        return h
    if isinstance(spec, str):
        raise NotImplementedError(f"{fname}: the bandwidth method {spec!r} is not supported (Silverman, Scott, a "
                                  "number or {\"Standardized\", h})")
    if K.is_list(spec) and len(spec.args) == 2 and spec.args[0] == "Standardized":
        return float(pos_number(spec.args[1], fname, "the standardized bandwidth")) * sd
    if K.is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[1], str):
        if spec.args[1] != "Gaussian":
            raise NotImplementedError(f"{fname}: the kernel {spec.args[1]!r} is not supported (only \"Gaussian\")")
        return bandwidth(xs, spec.args[0], fname, dim)
    return float(pos_number(spec, fname, "the bandwidth"))


def _phi(z):
    np = _np()
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / math.sqrt(2)))


def kde_fun(xs, h, dfun, fname):
    """Vectorised f(t) for the Gaussian kernel estimate of the data."""
    np = _np()
    X = np.asarray([float(x) for x in xs])
    n = len(X)
    if dfun is None or is_sym(dfun, "Automatic"):
        dfun = "PDF"
    if dfun not in ("PDF", "Intensity", "CDF", "SF", "HF", "CHF"):
        if isinstance(dfun, str):
            raise ValueError(f"{fname}: unknown distribution function {dfun!r}")
        raise NotImplementedError(f"{fname}: the distribution function {K_input(dfun)} is not supported")

    def pdf(t):
        t = np.asarray(t, dtype=float)
        z = (t[..., None] - X) / h
        return np.exp(-z * z / 2).sum(axis=-1) / (n * h * SQ2PI)

    def cdf(t):
        t = np.asarray(t, dtype=float)
        return _phi((t[..., None] - X) / h).mean(axis=-1)

    def f(t):
        with np.errstate(all="ignore"):
            if dfun == "PDF":
                return pdf(t)
            if dfun == "Intensity":
                return n * pdf(t)
            if dfun == "CDF":
                return cdf(t)
            sf = 1 - cdf(t)
            if dfun == "SF":
                return sf
            if dfun == "HF":
                return np.where(sf > 1e-300, pdf(t) / sf, np.nan)
            return np.where(sf > 1e-300, -np.log(sf), np.nan)
    return f


_SMOOTH_DEFAULTS = {"PlotStyle": AUTO, "PlotRange": AUTO, "PlotLegends": NONE, "Filling": NONE, "PlotPoints": AUTO,
                    "MaxRecursion": AUTO, "PlotTheme": AUTO, "PerformanceGoal": AUTO, "Method": AUTO,
                    "ColorFunction": AUTO, "ColorFunctionScaling": True, "Mesh": NONE, "MeshStyle": AUTO,
                    "PlotHighlighting": AUTO, "ScalingFunctions": NONE, "FillingStyle": AUTO}


def _smooth_range(sets, hs):
    lo = min(min(float(v) for v in vs) - 3 * h for vs, h in zip(sets, hs))
    hi = max(max(float(v) for v in vs) + 3 * h for vs, h in zip(sets, hs))
    return lo, hi


def smooth_curves(fname, sets, bw, dfun):
    hs = [bandwidth(vs, bw, fname) for vs in sets]
    a, b = _smooth_range(sets, hs)
    curves = []
    for vs, h in zip(sets, hs):
        f = kde_fun(vs, h, dfun, fname)
        curves.append(P2.sample_1d(f, a, b, n=60, maxrec=4))
    return curves, (a, b), hs


_SMOOTH_NOTE = (" Gaussian kernel estimate with bandwidth bw: Automatic/\"Silverman\" (0.9 min(sd, IQR/1.34) "
                "n^(-1/5), Silverman's rule; IQR from linearly interpolated quartiles), \"Scott\" (1.06 sd n^(-1/5)), "
                "a number h, {\"Standardized\", h} (h sd) or {bw, \"Gaussian\"}; other kernels and methods raise "
                "NotImplementedError. Distribution functions dfun: \"PDF\" (default), \"Intensity\" (n PDF), \"CDF\", "
                "\"SF\", \"HF\" (PDF/SF), \"CHF\" (-Log[SF]), all computed exactly from the kernel sum. The curve is "
                "sampled adaptively over [min - 3 h, max + 3 h].")


@fn("SmoothHistogram", area=AREA, status="added",
    notes="SmoothHistogram[data], [data, bw], [data, bw, dfun] and lists of datasets: Graphics[{{style, Line[...]}, "
          "...}] with one curve per dataset (default colours as Plot)." + _SMOOTH_NOTE + " Options: PlotStyle, "
          "PlotRange, PlotLegends and Graphics options; others raise NotImplementedError. Symbolic data stays "
          "unevaluated. Check: reference (curve points equal the kernel sum computed independently in tests; the "
          "PDF integrates to 1).")
def _smoothhistogram(data: ListOf[Real] | ListOf[ListOf[Real]], *rest: Expr | Option) -> Expr:
    fname = "SmoothHistogram"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _SMOOTH_DEFAULTS)
    _unsupported(o, fname, ("PlotStyle", "PlotLegends"))
    sets = [vs for vs, _ in datasets_1d(pos[0], fname)]
    curves, (a, b), _ = smooth_curves(fname, sets, pos[1] if len(pos) > 1 else None, pos[2] if len(pos) > 2 else None)
    styles = P2.style_list(o["PlotStyle"], len(curves), fname)
    content, ys = [], []
    for cs, st in zip(curves, styles):
        for c in cs:
            ys += [p[1] for p in c]
        if st is not None and cs:
            content.append(L(st, Expr("Line", L(*[pts(c) for c in cs]))))
    yr = P2.data_range(ys)
    r = _range_2d(o, fname, (a, b), (min(0.0, yr[0]), yr[1]))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])
    return P2.legend_wrap(g, o["PlotLegends"], styles, None, fname)


@fn("SmoothDateHistogram", area=AREA, status="added",
    notes="SmoothDateHistogram[dates], [dates, bw], [dates, bw, dfun]: SmoothHistogram of the AbsoluteTime values "
          "of the dates (seconds since 1 Jan 1900; DateObject, date lists, strings or numbers) with date ticks; "
          "a numeric bandwidth is in seconds." + _SMOOTH_NOTE + " Options as SmoothHistogram. Check: reference "
          "(curve equals the kernel sum over AbsoluteTime values in tests).")
def _smoothdatehistogram(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "SmoothDateHistogram"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _SMOOTH_DEFAULTS)
    _unsupported(o, fname, ("PlotStyle", "PlotLegends"))
    ts = date_values(pos[0], fname)
    curves, (a, b), _ = smooth_curves(fname, [ts], pos[1] if len(pos) > 1 else None,
                                      pos[2] if len(pos) > 2 else None)
    styles = P2.style_list(o["PlotStyle"], 1, fname)
    ys = [p[1] for c in curves[0] for p in c]
    content = [L(styles[0], Expr("Line", L(*[pts(c) for c in curves[0]])))] if styles[0] is not None else []
    yr = P2.data_range(ys)
    r = _range_2d(o, fname, (a, b), (0.0, yr[1]))
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True),
                                                ("Ticks", L(P2.date_ticks(a, b), AUTO)), ("PlotRange", range_expr(r))])


@fn("PairedSmoothHistogram", area=AREA, status="added",
    notes="PairedSmoothHistogram[data1, data2], [data1, data2, bw], [data1, data2, bw, dfun]: back-to-back kernel "
          "density estimates with the values along the vertical axis: data1 to the left (points {-f1(y), y}) and "
          "data2 to the right ({f2(y), y}), each drawn as a filled Polygon closed along the axis plus its Line; "
          "each side has its own automatic bandwidth." + _SMOOTH_NOTE + " Options: PlotStyle, PlotRange and "
          "Graphics options. Check: reference (curve points equal the kernel sums in tests).")
def _pairedsmoothhistogram(d1: ListOf[Real], d2: ListOf[Real], *rest: Expr | Option) -> Expr:
    fname = "PairedSmoothHistogram"
    pos, opts = split_args((d1, d2) + rest, fname, 2, 4)
    o, gopts = split_options(fname, opts, _SMOOTH_DEFAULTS)
    _unsupported(o, fname, ("PlotStyle",))
    s1, s2 = values_of(pos[0], fname), values_of(pos[1], fname)
    curves, (a, b), _ = smooth_curves(fname, [s1, s2], pos[2] if len(pos) > 2 else None,
                                      pos[3] if len(pos) > 3 else None)
    styles = P2.style_list(o["PlotStyle"], 2, fname)
    content, xm = [], 0.0
    for side, cs, st in ((-1, curves[0], styles[0]), (1, curves[1], styles[1])):
        lines = [[(side * y, x) for x, y in c] for c in cs]
        for ln in lines:
            xm = max(xm, max(abs(p[0]) for p in ln))
        if st is None:
            continue
        polys = [Expr("Polygon", pts(ln + [(0.0, ln[-1][1]), (0.0, ln[0][1])])) for ln in lines]
        content.append(L(Expr("Directive", P2.style_color(st), Expr("Opacity", 0.3), Expr("EdgeForm")), *polys))
        content.append(L(st, Expr("Line", L(*[pts(ln) for ln in lines]))))
    r = _range_2d(o, fname, (-(xm or 1.0), xm or 1.0), (a, b))
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])


def kde2_fun(pairs, hx, hy, dfun, fname):
    np = _np()
    X = np.asarray([float(p[0]) for p in pairs])
    Y = np.asarray([float(p[1]) for p in pairs])
    n = len(X)
    if dfun is None or is_sym(dfun, "Automatic"):
        dfun = "PDF"
    if dfun not in ("PDF", "Intensity", "CDF", "SF"):
        if dfun in ("HF", "CHF"):
            raise NotImplementedError(f"{fname}: the distribution function {dfun!r} is not supported in 2-D")
        raise ValueError(f"{fname}: unknown distribution function {K_input(dfun)}")

    def f(u, v):
        u = np.asarray(u, dtype=float)[..., None]
        v = np.asarray(v, dtype=float)[..., None]
        zx, zy = (u - X) / hx, (v - Y) / hy
        if dfun in ("PDF", "Intensity"):
            d = np.exp(-(zx * zx + zy * zy) / 2).sum(axis=-1) / (n * hx * hy * 2 * math.pi)
            return d * n if dfun == "Intensity" else d
        if dfun == "CDF":
            return (_phi(zx) * _phi(zy)).mean(axis=-1)
        return ((1 - _phi(zx)) * (1 - _phi(zy))).mean(axis=-1)
    return f


def _bw2(pairs, bw, fname):
    if K.is_list(bw) and len(bw.args) == 2 and bw.args[0] != "Standardized" and not isinstance(bw.args[1], str):
        return (bandwidth([p[0] for p in pairs], bw.args[0], fname, 2),
                bandwidth([p[1] for p in pairs], bw.args[1], fname, 2))
    return (bandwidth([p[0] for p in pairs], bw, fname, 2), bandwidth([p[1] for p in pairs], bw, fname, 2))


_SMOOTH2_NOTE = (" Product Gaussian kernel estimate with bandwidths per coordinate: Automatic sd n^(-1/6) (Scott's "
                 "rule in 2-D), a number h, {hx, hy} or {\"Standardized\", h}; dfun \"PDF\" (default), \"Intensity\", "
                 "\"CDF\" and \"SF\" (2-D, exact kernel sums); \"HF\"/\"CHF\" raise NotImplementedError. The grid "
                 "covers [min - 3 h, max + 3 h] in each coordinate.")


def _smooth2_grid(pairs, bw, dfun, fname, n):
    hx, hy = _bw2(pairs, bw, fname)
    f = kde2_fun(pairs, hx, hy, dfun, fname)
    xs = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    a, b, c, d = min(xs) - 3 * hx, max(xs) + 3 * hx, min(ys) - 3 * hy, max(ys) + 3 * hy
    Pm, T, _, _ = P2.grid_mesh(a, b, c, d, n, n)
    V = f(Pm[:, 0], Pm[:, 1])
    return Pm, T, V, (a, b), (c, d)


@fn("SmoothDensityHistogram", area=AREA, status="added",
    notes="SmoothDensityHistogram[{{x1, y1}, ...}], [data, bw], [data, bw, dfun]: a density plot (as DensityPlot: "
          "GraphicsComplex of a 41 x 41 grid with VertexColors ColorFunction((f - fmin)/(fmax - fmin))) of the "
          "kernel estimate." + _SMOOTH2_NOTE + " Options: ColorFunction, ColorFunctionScaling, PlotRange and "
          "Graphics options. Check: reference (vertex values equal the kernel sum computed independently in "
          "tests).")
def _smoothdensityhistogram(data: MatrixOf[Real], *rest: Expr | Option) -> Expr:
    fname = "SmoothDensityHistogram"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _SMOOTH_DEFAULTS)
    _unsupported(o, fname, ("ColorFunction", "ColorFunctionScaling"))
    pairs = values_2d(pos[0], fname)
    return smooth_density_graphics(fname, pairs, pos[1] if len(pos) > 1 else None, pos[2] if len(pos) > 2 else None,
                                   o, gopts)


def smooth_density_graphics(fname, pairs, bw, dfun, o, gopts):
    Pm, T, V, xr, yr = _smooth2_grid(pairs, bw, dfun, fname, 41)
    cf = color_function(o.get("ColorFunction", AUTO), fname, lambda t: blend(DENSITY_GRADIENT, t))
    g = P2.density_complex(Pm, V, T, cf, float(V.min()), float(V.max()), o.get("ColorFunctionScaling", True)
                           is not False)
    r = _range_2d(o, fname, xr, yr)
    return finish_graphics(L(g), gopts, [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr(r)),
                                         ("PlotRangePadding", 0)])


@fn("SmoothHistogram3D", area=AREA, status="added",
    notes="SmoothHistogram3D[{{x1, y1}, ...}], [data, bw], [data, bw, dfun]: Graphics3D[{style, GraphicsComplex["
          "{{x, y, f(x, y)}, ...}, {EdgeForm[], Polygon[quads]}]}] over a 41 x 41 grid (layout as Plot3D)."
          + _SMOOTH2_NOTE + " Options: PlotStyle, PlotRange and Graphics3D options. Check: reference (vertex heights "
          "equal the kernel sum in tests; the PDF integrates to about 1).")
def _smoothhistogram3d(data: MatrixOf[Real], *rest: Expr | Option) -> Expr:
    fname = "SmoothHistogram3D"
    pos, opts = split_args((data,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _SMOOTH_DEFAULTS, dim=3)
    _unsupported(o, fname, ("PlotStyle",))
    pairs = values_2d(pos[0], fname)
    np = _np()
    Pm, T, V, xr, yr = _smooth2_grid(pairs, pos[1] if len(pos) > 1 else None, pos[2] if len(pos) > 2 else None,
                                     fname, 41)
    P3 = np.column_stack([Pm, V])
    gc = P2.surface_complex(P3, P2.grid_faces(41, 41))
    st = P2.surface_styles(o["PlotStyle"], 1, fname)[0]
    content = [L(st, gc)] if st is not None and gc is not None else []
    r = P2.resolve_plot_range(o["PlotRange"], fname, [xr, yr, (0.0, float(V.max()))], dims=3)
    return finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 1, 0.4)),
                                                ("PlotRange", range_expr(r))], dim=3)


@fn("PairwiseSmoothDensityHistogram", area=AREA, status="added",
    notes="PairwiseSmoothDensityHistogram[{{y11, ..., y1n}, ...}]: GraphicsGrid of n x n panels; panel {i, j} "
          "(i != j) is the SmoothDensityHistogram of the points {y_kj, y_ki}, the diagonal panel i the "
          "SmoothHistogram of column i (automatic bandwidths)." + _SMOOTH2_NOTE + " Options: ColorFunction, "
          "ColorFunctionScaling and GraphicsGrid options. Check: reference (panels equal SmoothDensityHistogram of "
          "the column pairs in tests).")
def _pairwisesmoothdensityhistogram(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "PairwiseSmoothDensityHistogram"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)

    def panel(pairs, o):
        return smooth_density_graphics(fname, pairs, None, None, o, [])

    def diag(vals, o):
        curves, (a, b), _ = smooth_curves(fname, [vals], None, None)
        ys = [p[1] for c in curves[0] for p in c]
        return Expr("Graphics", L(L(default_color(0), Expr("Line", L(*[pts(c) for c in curves[0]])))),
                    Expr("Rule", Symbol("Frame"), True), Expr("Rule", Symbol("AspectRatio"), 1),
                    Expr("Rule", Symbol("PlotRange"), range_expr([(a, b), (0.0, max(ys))])))
    return pairwise_grid(fname, pos[0], opts_, panel, diag)
