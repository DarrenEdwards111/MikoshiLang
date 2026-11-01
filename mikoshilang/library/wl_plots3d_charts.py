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


def _abs_time(x, fname):
    """Absolute time of a date (errors renamed to fname)."""
    try:
        return P2.abs_time(x, fname)
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        if not msg.startswith(fname + ":"):
            msg = f"{fname}: " + msg.split(": ", 1)[-1]
        raise type(exc)(msg) from None


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
        out.append(_abs_time(x, fname))
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


# ───────────────────────────── gauges ─────────────────────────────

def _gauge_any(x):
    return True


_GAUGE_OPTION_SPECS = [
    ("GaugeFaceElementFunction", "Automatic, a named element function or a function",
     [PO.AUTO, PO._string(), PO._func],
     "option of gauges: how the face is drawn; gauges here support Automatic (other settings raise "
     "NotImplementedError)."),
    ("GaugeFaceStyle", "Automatic, None or a graphics directive", [PO.AUTO_NONE, PO.STYLE],
     "option of gauges: style (colour) of the gauge face."),
    ("GaugeFrameElementFunction", "Automatic, a named element function or a function",
     [PO.AUTO, PO._string(), PO._func],
     "option of gauges: how the frame is drawn; gauges here support Automatic (others raise NotImplementedError)."),
    ("GaugeFrameSize", "Automatic, None, Tiny, Small, Medium, Large or a size s > 0",
     [PO._sym("Automatic", "None", *PO.SIZES), PO.POS],
     "option of gauges: width of the frame as a fraction of the gauge size (Tiny 0.02, Small 0.04, Medium 0.06, "
     "Large 0.1, Automatic 0.04)."),
    ("GaugeFrameStyle", "Automatic, None or a graphics directive", [PO.AUTO_NONE, PO.STYLE],
     "option of gauges: style (colour) of the frame."),
    ("GaugeLabels", "None, Automatic, \"Value\", a label or a list of labels",
     [PO._sym("None", "Automatic", "All", "Full"), PO._string(), PO._list(_gauge_any), PO.LABEL],
     "option of gauges: labels; None (default), Automatic or \"Value\" (the value shown as text), or a label "
     "expression; Placed labels raise NotImplementedError."),
    ("GaugeMarkers", "Automatic, None, a marker name, a graphic or a list of markers",
     [PO.AUTO_NONE, PO._string(), PO._list(_gauge_any), PO._head("Graphics", "Graphics3D", "Image")],
     "option of gauges: shape of the value markers (needles, bars); gauges here support Automatic (others raise "
     "NotImplementedError)."),
    ("GaugeStyle", "Automatic, a graphics directive or a list of directives", [PO.AUTO, PO.STYLE],
     "option of gauges: style of the value markers (one directive per value, cycled)."),
]
for _n, _m, _preds, _note in _GAUGE_OPTION_SPECS:
    PO.OPTION_VALIDATORS[_n] = PO.make_validator(_n, _m, *_preds)
    inert_option(_n, _note + " Invalid settings raise ValueError. Inert as an expression (Name and Name -> value "
                 "evaluate to themselves). Check: reference (Wolfram documentation settings) in tests.", area=AREA)


_GAUGE_DEFAULTS = {"GaugeFaceElementFunction": AUTO, "GaugeFaceStyle": AUTO, "GaugeFrameElementFunction": AUTO,
                   "GaugeFrameSize": AUTO, "GaugeFrameStyle": AUTO, "GaugeLabels": NONE, "GaugeMarkers": AUTO,
                   "GaugeStyle": AUTO, "ScaleDivisions": AUTO, "ScaleOrigin": AUTO, "ScalePadding": AUTO,
                   "ScaleRanges": NONE, "ScaleRangeStyle": AUTO, "TargetUnits": AUTO, "PlotTheme": AUTO,
                   "PerformanceGoal": AUTO, "TicksStyle": AUTO, "TimeZone": AUTO, "ReferenceMarkers": AUTO,
                   "ReferenceMarkerStyle": AUTO, "Method": AUTO}

FACE = Expr("GrayLevel", 0.97)
FRAME = Expr("GrayLevel", 0.6)


def gauge_options(fname, opts, extra=()):
    o, gopts = split_options(fname, opts, _GAUGE_DEFAULTS)
    allowed = {"GaugeFaceStyle", "GaugeFrameSize", "GaugeFrameStyle", "GaugeLabels", "GaugeStyle",
               "ScaleDivisions", "ScaleRanges", "ScaleRangeStyle"} | set(extra)
    for n in ("GaugeFaceElementFunction", "GaugeFrameElementFunction", "GaugeMarkers"):
        if not is_sym(o[n], "Automatic"):
            raise NotImplementedError(f"{fname}: {n} -> {K_input(o[n])} is not supported")
    _unsupported(o, fname, allowed | {"GaugeFaceElementFunction", "GaugeFrameElementFunction", "GaugeMarkers"})
    return o, gopts


def gauge_value(x, lo, hi, fname):
    """The plotted value: clipped to [lo, hi]; missing values show the minimum (as documented)."""
    if is_missing(x):
        return lo
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: the value {K_input(x)} is not a real number")
    if isinstance(x, Expr) and x.head in ("Quantity", "Dynamic"):
        raise NotImplementedError(f"{fname}: {x.head} values are not supported")
    v = num(x)
    if v is None:
        if isinstance(x, complex) or (isinstance(x, Expr) and x.head == "Complex"):
            return lo
        raise Unevaluated()
    return min(max(v, lo), hi)


def gauge_values(x, lo, hi, fname):
    if K.is_list(x):
        if not x.args:
            raise ValueError(f"{fname}: the list of values is empty")
        return [gauge_value(a, lo, hi, fname) for a in x.args]
    return [gauge_value(x, lo, hi, fname)]


def gauge_range(spec, fname, pos=2):
    if spec is None:
        return 0.0, 1.0
    if not K.is_list(spec) or len(spec.args) != 2:
        if isinstance(spec, (Symbol,)) or (isinstance(spec, Expr) and not K.is_list(spec)):
            raise Unevaluated()
        raise ValueError(f"{fname}: argument {pos} must be a range {{min, max}}, not {K_input(spec)}")
    if any(isinstance(a, Expr) and a.head == "Quantity" for a in spec.args):
        raise NotImplementedError(f"{fname}: Quantity ranges are not supported")
    lo = float(real_number(spec.args[0], fname, "the minimum"))
    hi = float(real_number(spec.args[1], fname, "the maximum"))
    if not hi > lo:
        raise ValueError(f"{fname}: the range must have min < max, not {K_input(spec)}")
    return lo, hi


def scale_ticks(lo, hi, spec, fname):
    """(major tick values, minor tick values) of a gauge scale."""
    if is_sym(spec, "None"):
        return [], []
    if is_sym(spec, "Automatic"):
        w = float(_nice_width(lo, hi, 5))
        k0 = math.ceil(lo / w - 1e-9)
        k1 = math.floor(hi / w + 1e-9)
        major = [k * w for k in range(k0, k1 + 1)]
        minor = []
        for a in major[:-1]:
            minor += [a + w * j / 5 for j in range(1, 5)]
        return major, minor
    if K.is_int(spec):
        n = pos_int(spec, fname, "ScaleDivisions")
        return [lo + (hi - lo) * k / n for k in range(n + 1)], []
    if K.is_list(spec) and len(spec.args) == 2:
        n = pos_int(spec.args[0], fname, "ScaleDivisions")
        m = pos_int(spec.args[1], fname, "ScaleDivisions")
        major = [lo + (hi - lo) * k / n for k in range(n + 1)]
        minor = [a + (major[1] - major[0]) * j / m for a in major[:-1] for j in range(1, m)]
        return major, minor
    raise ValueError(f"{fname}: ScaleDivisions -> {K_input(spec)} is not valid")


def _label_num(v):
    r = round(v, 10)
    return int(r) if r == int(r) else r


def _style_of(o, name, default):
    v = o[name]
    if is_sym(v, "Automatic"):
        return default
    if is_sym(v, "None"):
        return None
    G.directive_like(v, "gauge", name)
    return v


def gauge_styles(o, n):
    spec = o["GaugeStyle"]
    user = list(spec.args) if K.is_list(spec) else ([spec] if not is_sym(spec, "Automatic") else [])
    out = []
    for i in range(n):
        u = user[i % len(user)] if user else None
        out.append(Expr("Directive", default_color(i), *([] if u is None else _directive_items(u))))
    return out


def frame_size(o, fname):
    v = o["GaugeFrameSize"]
    if is_sym(v, "None"):
        return 0.0
    if is_sym(v, "Automatic"):
        return 0.04
    sizes = {"Tiny": 0.02, "Small": 0.04, "Medium": 0.06, "Large": 0.1}
    if isinstance(v, Symbol) and v.name in sizes:
        return sizes[v.name]
    return float(pos_number(v, fname, "GaugeFrameSize"))


def gauge_label(o, values, fname):
    v = o["GaugeLabels"]
    if is_sym(v, "None"):
        return None
    if is_sym(v, "Automatic") or v == "Value":
        return ", ".join(str(_label_num(x)) for x in values)
    if isinstance(v, Expr) and v.head == "Placed":
        raise NotImplementedError(f"{fname}: Placed gauge labels are not supported")
    return v


def scale_ranges(o, fname, lo, hi):
    """[(a, b, style)] of ScaleRanges (clipped to the scale)."""
    spec = o["ScaleRanges"]
    if is_sym(spec, "None"):
        return []
    if is_sym(spec, "All"):
        items = [L(lo, hi)]
    elif K.is_list(spec) and len(spec.args) == 2 and all(num(a) is not None for a in spec.args):
        items = [spec]
    elif K.is_list(spec):
        items = list(spec.args)
    else:
        raise ValueError(f"{fname}: ScaleRanges -> {K_input(spec)} is not valid")
    st = o["ScaleRangeStyle"]
    sts = list(st.args) if K.is_list(st) else ([st] if not is_sym(st, "Automatic") else
                                               [Expr("RGBColor", 0.6, 0.8, 0.6), Expr("RGBColor", 1.0, 0.85, 0.4),
                                                Expr("RGBColor", 0.95, 0.5, 0.45)])
    out = []
    for k, it in enumerate(items):
        if not (K.is_list(it) and len(it.args) == 2):
            raise NotImplementedError(f"{fname}: the scale range {K_input(it)} is not supported; give {{vmin, vmax}}")
        a = min(max(float(real_number(it.args[0], fname, "a scale range limit")), lo), hi)
        b = min(max(float(real_number(it.args[1], fname, "a scale range limit")), lo), hi)
        if not b > a:
            raise ValueError(f"{fname}: a scale range needs vmin < vmax")
        out.append((a, b, sts[k % len(sts)]))
    return out


ANG0, ANG1 = 5 * math.pi / 4, -math.pi / 4


def angle_of(v, lo, hi, a0=ANG0, a1=ANG1):
    return a0 + (a1 - a0) * (v - lo) / (hi - lo)


def _gauge_args(args, fname, lo_n=1, hi_n=2):
    pos, opts = split_args(args, fname, lo_n, hi_n)
    return pos, opts


def _scale_origin(o, fname):
    v = o["ScaleOrigin"]
    if is_sym(v, "Automatic"):
        return ANG0, ANG1
    if K.is_list(v) and len(v.args) == 2 and all(num(a) is not None for a in v.args):
        a0, a1 = (float(num(a)) for a in v.args)
        if a0 == a1:
            raise ValueError(f"{fname}: ScaleOrigin angles must differ")
        return a0, a1
    raise NotImplementedError(f"{fname}: ScaleOrigin -> {K_input(v)} is not supported; give {{theta_min, theta_max}}")


def _cs(t):
    return (math.cos(t), math.sin(t))


@fn("AngularGauge", area=AREA, status="added",
    notes="AngularGauge[v], [v, {min, max}] and [{v1, v2, ...}, ...]: a dial Graphics with the face Disk[{0, 0}, 1], "
          "frame Annulus, scale arc Circle[{0, 0}, 0.9, ...], tick Lines and Text labels, and one needle Line[{{0, 0}, "
          "0.8 {Cos[a], Sin[a]}}] per value as the last group, where a = a0 + (a1 - a0) (v - min)/(max - min) with "
          "a0 = 5 Pi/4 and a1 = -Pi/4 (clockwise 270 degree scale; ScaleOrigin -> {a0, a1} changes it). Default "
          "range {0, 1} (as documented); values are clipped to the range and missing values show min (as "
          "documented). Options: GaugeStyle, GaugeFaceStyle, GaugeFrameStyle, GaugeFrameSize, GaugeLabels, "
          "ScaleDivisions (Automatic nice ticks, n or {n, m}), ScaleOrigin, ScaleRanges, ScaleRangeStyle and "
          "Graphics options; element functions, markers, Dynamic and Quantity values raise NotImplementedError. "
          "Symbolic values stay unevaluated. Check: reference (needle angles from the linear scale in tests).")
def _angulargauge(v: Real | ListOf[Real], *rest: ListOf[Real] | Option) -> Expr:
    fname = "AngularGauge"
    pos, opts = _gauge_args((v,) + rest, fname)
    o, gopts = gauge_options(fname, opts, ("ScaleOrigin",))
    lo, hi = gauge_range(pos[1] if len(pos) > 1 else None, fname)
    vals = gauge_values(pos[0], lo, hi, fname)
    a0, a1 = _scale_origin(o, fname)
    content = []
    fs = frame_size(o, fname)
    face = _style_of(o, "GaugeFaceStyle", FACE)
    if face is not None:
        content.append(L(face, Expr("Disk", pt(0, 0), 1.0)))
    frame = _style_of(o, "GaugeFrameStyle", FRAME)
    if frame is not None and fs > 0:
        content.append(L(frame, Expr("Annulus", pt(0, 0), L(1.0, fl(1 + fs)))))
    for a, b, st in scale_ranges(o, fname, lo, hi):
        t0, t1 = sorted((angle_of(a, lo, hi, a0, a1), angle_of(b, lo, hi, a0, a1)))
        content.append(L(st, Expr("Annulus", pt(0, 0), L(0.8, 0.9), L(fl(t0), fl(t1)))))
    t0, t1 = sorted((a0, a1))
    major, minor = scale_ticks(lo, hi, o["ScaleDivisions"], fname)
    sc = [Expr("Circle", pt(0, 0), 0.9, L(fl(t0), fl(t1)))]
    for m in major:
        c, s = _cs(angle_of(m, lo, hi, a0, a1))
        sc.append(Expr("Line", pts([(0.9 * c, 0.9 * s), (0.8 * c, 0.8 * s)])))
        sc.append(Expr("Text", _label_num(m), pt(0.68 * c, 0.68 * s)))
    for m in minor:
        c, s = _cs(angle_of(m, lo, hi, a0, a1))
        sc.append(Expr("Line", pts([(0.9 * c, 0.9 * s), (0.85 * c, 0.85 * s)])))
    content.append(L(Expr("GrayLevel", 0.3), *sc))
    lab = gauge_label(o, vals, fname)
    if lab is not None:
        content.append(Expr("Text", lab, pt(0, -0.45)))
    needles = []
    for x, st in zip(vals, gauge_styles(o, len(vals))):
        c, s = _cs(angle_of(x, lo, hi, a0, a1))
        needles.append(L(st, Expr("AbsoluteThickness", 3), Expr("Line", pts([(0, 0), (0.8 * c, 0.8 * s)]))))
    content.append(L(*needles, L(Expr("GrayLevel", 0.3), Expr("Disk", pt(0, 0), 0.06))))
    r = 1 + fs + 0.02
    return finish_graphics(L(*content), gopts, [("PlotRange", range_expr([(-r, r), (-r, r)])),
                                                ("AspectRatio", 1)])


def _linear_gauge(fname, args, horizontal, thermometer=False):
    pos, opts = _gauge_args(args, fname)
    o, gopts = gauge_options(fname, opts)
    lo, hi = gauge_range(pos[1] if len(pos) > 1 else None, fname)
    vals = gauge_values(pos[0], lo, hi, fname)
    s = hi - lo

    def P(u, y):                     # u across the track (in units of s), y along the scale (data value)
        return pt(y, u * s) if horizontal else pt(u * s, y)

    def R(u0, y0, u1, y1):
        if horizontal:
            return Expr("Rectangle", pt(y0, u0 * s), pt(y1, u1 * s))
        return Expr("Rectangle", pt(u0 * s, y0), pt(u1 * s, y1))
    content = []
    hw = 0.035 if thermometer else 0.05
    face = _style_of(o, "GaugeFaceStyle", FACE)
    frame = _style_of(o, "GaugeFrameStyle", FRAME)
    fs = frame_size(o, fname) * 0.5
    if frame is not None and fs > 0:
        content.append(L(frame, R(-hw - fs, lo - fs * s, hw + fs, hi + fs * s)))
    if face is not None:
        content.append(L(face, R(-hw, lo, hw, hi)))
    for a, b, st in scale_ranges(o, fname, lo, hi):
        content.append(L(st, R(hw, a, hw + 0.04, b)))
    major, minor = scale_ticks(lo, hi, o["ScaleDivisions"], fname)
    sc = []
    for m in major:
        sc.append(Expr("Line", L(P(hw + 0.04, m), P(hw + 0.08, m))))
        sc.append(Expr("Text", _label_num(m), P(hw + 0.16, m)))
    for m in minor:
        sc.append(Expr("Line", L(P(hw + 0.04, m), P(hw + 0.06, m))))
    content.append(L(Expr("GrayLevel", 0.3), *sc))
    lab = gauge_label(o, vals, fname)
    if lab is not None:
        content.append(Expr("Text", lab, P(0, lo - 0.25 * s)))
    styles = gauge_styles(o, len(vals))
    marks = []
    if thermometer:
        st0 = styles[0]
        marks.append(L(st0, Expr("Disk", P(0, lo - 0.07 * s), fl(0.08 * s)),
                       R(-hw * 0.6, lo - 0.05 * s, hw * 0.6, vals[0])))
        for x, st in zip(vals[1:], styles[1:]):
            marks.append(L(st, Expr("Line", L(P(-hw, x), P(hw, x)))))
    else:
        order = sorted(range(len(vals)), key=lambda i: -vals[i])
        for i in order:
            marks.append(L(styles[i], R(-hw * 0.7, lo, hw * 0.7, vals[i])))
    content.append(L(*marks))
    across = (-hw - 0.06, hw + 0.25)
    along = (lo - (0.2 if thermometer else 0.05) * s, hi + 0.05 * s)
    rr = [along, (across[0] * s, across[1] * s)] if horizontal else [(across[0] * s, across[1] * s), along]
    return finish_graphics(L(*content), gopts, [("PlotRange", range_expr(rr)), ("AspectRatio", Symbol("Automatic"))])


_LINEAR_NOTE = ("Default range {0, 1} (as documented); values are clipped to the range and missing values show min. "
                "Options: GaugeStyle, GaugeFaceStyle, GaugeFrameStyle, GaugeFrameSize, GaugeLabels, ScaleDivisions "
                "(Automatic nice ticks, n or {n, m}), ScaleRanges, ScaleRangeStyle and Graphics options; element "
                "functions, markers, Dynamic and Quantity values raise NotImplementedError. Symbolic values stay "
                "unevaluated. Check: reference (bar ends equal the clipped values in tests).")


@fn("VerticalGauge", area=AREA, status="added",
    notes="VerticalGauge[v], [v, {min, max}] and [{v1, ...}, ...]: Graphics in data coordinates with the scale along "
          "y from min to max: track Rectangle[{-0.05 s, min}, {0.05 s, max}] (s = max - min), ticks and labels to the "
          "right, and one bar Rectangle[{-0.035 s, min}, {0.035 s, v}] per value as the last group (largest first). "
          + _LINEAR_NOTE)
def _verticalgauge(v: Real | ListOf[Real], *rest: ListOf[Real] | Option) -> Expr:
    return _linear_gauge("VerticalGauge", (v,) + rest, horizontal=False)


@fn("HorizontalGauge", area=AREA, status="added",
    notes="HorizontalGauge[v], [v, {min, max}] and [{v1, ...}, ...]: as VerticalGauge with the scale along x: bars "
          "Rectangle[{min, -0.035 s}, {v, 0.035 s}] as the last group. " + _LINEAR_NOTE)
def _horizontalgauge(v: Real | ListOf[Real], *rest: ListOf[Real] | Option) -> Expr:
    return _linear_gauge("HorizontalGauge", (v,) + rest, horizontal=True)


@fn("ThermometerGauge", area=AREA, status="added",
    notes="ThermometerGauge[v], [v, {min, max}] and [{v1, ...}, ...]: a vertical thermometer in data coordinates: "
          "tube Rectangle from min to max, bulb Disk below min, and the mercury Rectangle[{-0.021 s, min - 0.05 s}, "
          "{0.021 s, v1}] (s = max - min) as the last group; further values are marker Lines. " + _LINEAR_NOTE)
def _thermometergauge(v: Real | ListOf[Real], *rest: ListOf[Real] | Option) -> Expr:
    return _linear_gauge("ThermometerGauge", (v,) + rest, horizontal=False, thermometer=True)


@fn("BulletGauge", area=AREA, status="added",
    notes="BulletGauge[v, ref, {min, max}], [v, ref, {min, m1, ..., max}], lists of values and of references: a "
          "horizontal bullet chart in data coordinates (s = max - min): qualitative bands Rectangle[{m_k, -0.15 s}, "
          "{m_(k+1), 0.15 s}] in decreasing gray levels, one value bar Rectangle[{min, -0.05 s}, {v, 0.05 s}] per "
          "value and one reference Line[{{r, -0.1 s}, {r, 0.1 s}}] per reference (values and references clipped to "
          "the range, as documented). The division points must increase. Options: GaugeStyle, GaugeLabels, "
          "ScaleDivisions and Graphics options; others raise NotImplementedError. Check: reference (bar ends, "
          "reference positions and band limits in tests).")
def _bulletgauge(v: Real | ListOf[Real], ref: Real | ListOf[Real], rng: ListOf[Real], *opts: Option) -> Expr:
    fname = "BulletGauge"
    pos, opts_ = split_args((v, ref, rng) + opts, fname, 3, 3)
    o, gopts = gauge_options(fname, opts_)
    spec = pos[2]
    if not K.is_list(spec) or len(spec.args) < 2:
        if isinstance(spec, (Symbol, Expr)) and not K.is_list(spec):
            raise Unevaluated()
        raise ValueError(f"{fname}: argument 3 must be {{min, max}} or {{min, m1, ..., max}}")
    marks = [float(real_number(a, fname, "a range point")) for a in spec.args]
    if any(b <= a for a, b in zip(marks, marks[1:])):
        raise ValueError(f"{fname}: the range points must increase")
    lo, hi = marks[0], marks[-1]
    s = hi - lo
    vals = gauge_values(pos[0], lo, hi, fname)
    refs = gauge_values(pos[1], lo, hi, fname)
    content = []
    nb = len(marks) - 1
    for k in range(nb):
        g = 0.6 + 0.3 * (k / max(1, nb - 1) if nb > 1 else 0.5)
        content.append(L(Expr("GrayLevel", fl(g)), Expr("Rectangle", pt(marks[k], -0.15 * s),
                                                          pt(marks[k + 1], 0.15 * s))))
    major, minor = scale_ticks(lo, hi, o["ScaleDivisions"], fname)
    sc = []
    for m in major:
        sc.append(Expr("Line", pts([(m, -0.15 * s), (m, -0.19 * s)])))
        sc.append(Expr("Text", _label_num(m), pt(m, -0.26 * s)))
    content.append(L(Expr("GrayLevel", 0.3), *sc))
    lab = gauge_label(o, vals, fname)
    if lab is not None:
        content.append(Expr("Text", lab, pt(lo - 0.12 * s, 0)))
    styles = gauge_styles(o, len(vals))
    bars_ = [L(st, Expr("Rectangle", pt(lo, -0.05 * s), pt(x, 0.05 * s))) for x, st in
             sorted(zip(vals, styles), key=lambda t: -t[0])]
    reflines = [L(Expr("GrayLevel", 0), Expr("AbsoluteThickness", 3), Expr("Line", pts([(r, -0.1 * s), (r, 0.1 * s)])))
                for r in refs]
    content.append(L(*bars_, *reflines))
    return finish_graphics(L(*content), gopts, [("PlotRange", range_expr([(lo - 0.2 * s, hi + 0.02 * s),
                                                                          (-0.3 * s, 0.17 * s)])),
                                                ("AspectRatio", Symbol("Automatic"))])


def clock_hms(t, fname):
    """(h, m, s) of a ClockGauge time specification."""
    import datetime as dt
    if t is None:
        now = dt.datetime.now()
        return now.hour, now.minute, now.second + now.microsecond / 1e6
    if is_missing(t):
        return 0, 0, 0.0
    if K.is_list(t) and len(t.args) == 3:
        h, m, s = (float(real_number(a, fname, "a time component")) for a in t.args)
        return h, m, s
    if K.is_list(t) and len(t.args) == 6:
        _y, _mo, _d, h, m, s = (float(real_number(a, fname, "a date component")) for a in t.args)
        return h, m, s
    if isinstance(t, Expr) and t.head in ("DateObject", "TimeObject") or isinstance(t, str) or \
            (num(t) is not None and not isinstance(t, bool)):
        if isinstance(t, Expr) and t.head == "DateObject" and len(t.args) > 2:
            raise NotImplementedError(f"{fname}: DateObject time zones are not supported; give {{h, m, s}}")
        r = ev(Expr("DateList", t))
        if not K.is_list(r) or len(r.args) != 6:
            raise ValueError(f"{fname}: {K_input(t)} is not a valid time")
        return float(num(r.args[3])), float(num(r.args[4])), float(num(r.args[5]))
    if isinstance(t, (Symbol, Expr)) and not K.is_list(t):
        raise Unevaluated()
    raise TypeError(f"{fname}: {K_input(t)} is not a time specification")


def clock_angles(h, m, s):
    """Angles (radians, counterclockwise from the x axis) of the hour, minute and second hands."""
    th = math.pi / 2 - 2 * math.pi * ((h % 12) + m / 60 + s / 3600) / 12
    tm = math.pi / 2 - 2 * math.pi * (m + s / 60) / 60
    ts = math.pi / 2 - 2 * math.pi * s / 60
    return th, tm, ts


@fn("ClockGauge", area=AREA, status="added",
    notes="ClockGauge[], ClockGauge[{h, m, s}], [{y, m, d, h, m, s}], [AbsoluteTime], [DateObject or TimeObject] and "
          "[\"string\"] (via DateList): an analog clock Graphics with face Disk[{0, 0}, 1], 12 hour marks with labels "
          "and, as the last group, the hour, minute and second hands Line[{{0, 0}, r {Cos[a], Sin[a]}}] (r = 0.5, "
          "0.75, 0.85) at a = Pi/2 - 2 Pi (h mod 12 + m/60 + s/3600)/12, Pi/2 - 2 Pi (m + s/60)/60 and Pi/2 - 2 Pi "
          "s/60. ClockGauge[] uses the local time of the session; missing values show 0:00:00 (as documented). "
          "DateObject time zones and the TimeZone option raise NotImplementedError. Options: GaugeStyle (hands), "
          "GaugeFaceStyle, GaugeFrameStyle, GaugeFrameSize, GaugeLabels and Graphics options. Check: reference "
          "(hand angles for known times in tests).")
def _clockgauge(*args: ListOf[Real] | Expr | Option) -> Expr:
    fname = "ClockGauge"
    pos, opts = split_args(args, fname, 0, 1)
    o, gopts = gauge_options(fname, opts)
    h, m, s = clock_hms(pos[0] if pos else None, fname)
    content = []
    fs = frame_size(o, fname)
    face = _style_of(o, "GaugeFaceStyle", FACE)
    if face is not None:
        content.append(L(face, Expr("Disk", pt(0, 0), 1.0)))
    frame = _style_of(o, "GaugeFrameStyle", FRAME)
    if frame is not None and fs > 0:
        content.append(L(frame, Expr("Annulus", pt(0, 0), L(1.0, fl(1 + fs)))))
    marks = []
    for k in range(12):
        a = math.pi / 2 - 2 * math.pi * k / 12
        c, sn = _cs(a)
        marks.append(Expr("Line", pts([(0.9 * c, 0.9 * sn), (0.97 * c, 0.97 * sn)])))
        marks.append(Expr("Text", 12 if k == 0 else k, pt(0.78 * c, 0.78 * sn)))
    content.append(L(Expr("GrayLevel", 0.3), *marks))
    lab = gauge_label(o, [], fname)
    if lab is not None and lab != "":
        content.append(Expr("Text", lab, pt(0, -0.45)))
    th, tm, ts = clock_angles(h, m, s)
    sty = gauge_styles(o, 3)
    hands = [L(Expr("GrayLevel", 0.2), Expr("AbsoluteThickness", 4), Expr("Line", pts([(0, 0), (0.5 * math.cos(th),
                                                                                     0.5 * math.sin(th))]))),
             L(Expr("GrayLevel", 0.2), Expr("AbsoluteThickness", 3), Expr("Line", pts([(0, 0), (0.75 * math.cos(tm),
                                                                                     0.75 * math.sin(tm))]))),
             L(sty[0] if not is_sym(o["GaugeStyle"], "Automatic") else Expr("RGBColor", 0.8, 0.1, 0.1),
               Expr("AbsoluteThickness", 1), Expr("Line", pts([(0, 0), (0.85 * math.cos(ts), 0.85 * math.sin(ts))])))]
    content.append(L(*hands))
    r = 1 + fs + 0.02
    return finish_graphics(L(*content), gopts, [("PlotRange", range_expr([(-r, r), (-r, r)])), ("AspectRatio", 1)])


# ───────────────────────────── statistical charts ─────────────────────────────

_CHART_DEFAULTS = {"ChartStyle": AUTO, "ChartBaseStyle": AUTO, "ChartLabels": NONE, "ChartLegends": NONE,
                   "ChartElementFunction": AUTO, "ChartElements": AUTO, "ChartLayout": AUTO, "BarSpacing": AUTO,
                   "BarOrigin": Symbol("Bottom"), "BubbleSizes": AUTO, "BubbleScale": AUTO, "SectorOrigin": AUTO,
                   "SectorSpacing": AUTO, "PlotRange": AUTO, "LabelingFunction": AUTO, "PlotTheme": AUTO,
                   "PerformanceGoal": AUTO, "ColorFunction": AUTO, "ColorFunctionScaling": True, "Method": AUTO,
                   "PlotInteractivity": AUTO, "ScalingFunctions": NONE}


def chart_options(fname, opts, allowed, dim=2):
    o, gopts = split_options(fname, opts, _CHART_DEFAULTS, dim=dim)
    _unsupported(o, fname, set(allowed) | {"ChartStyle", "ChartBaseStyle", "ChartLegends"})
    return o, gopts


def chart_datasets(data, fname, allow_groups=False):
    """[values] for one dataset {y1, ...} or a list of datasets {{...}, {...}}."""
    du, _, _ = P2._unwrap(data, fname)
    if K.is_list(du) and du.args and all(K.is_list(P2._unwrap(a, fname)[0]) for a in du.args):
        if any(K.is_list(P2._unwrap(a, fname)[0]) and any(K.is_list(b) for b in P2._unwrap(a, fname)[0].args)
               for a in du.args) and not allow_groups:
            raise NotImplementedError(f"{fname}: grouped datasets {{{{data1, data2, ...}}, ...}} are not supported")
        return [values_of(a, fname) for a in du.args]
    return [values_of(data, fname)]


def _ticks_labels(o, n, fname):
    lab = o["ChartLabels"]
    if is_sym(lab, "None"):
        return AUTO
    if K.is_list(lab):
        items = list(lab.args)
        return L(L(*[L(k + 1, items[k]) for k in range(min(n, len(items)))]), AUTO)
    raise NotImplementedError(f"{fname}: ChartLabels -> {K_input(lab)} is not supported; give a list of labels")


def quartiles(vals, fname):
    r = ev(Expr("Quartiles", L(*[_out(v) for v in vals])))
    if not K.is_list(r) or len(r.args) != 3:
        raise ArithmeticError(f"{fname}: could not compute the quartiles")
    return [float(num(q)) for q in r.args]


@fn("BoxWhiskerChart", area=AREA, status="added",
    notes="BoxWhiskerChart[data] and [{data1, data2, ...}], [data, bwspec]: one box-and-whisker element per dataset "
          "at x = 1, 2, ...: box Rectangle[{i - 0.3, Q1}, {i + 0.3, Q3}], median Line at Q2 and whisker Lines to the "
          "minimum and maximum, with the quartiles of the Quartiles function (Wolfram's definition, as "
          "documented). bwspec \"Outliers\": whiskers end at the most extreme data within 1.5 IQR of the box and "
          "the points beyond are drawn as Point; \"Mean\": a dashed Line at the mean; other element specifications "
          "(notches, fences styling) and grouped datasets raise NotImplementedError. Options: ChartStyle, "
          "ChartBaseStyle, ChartLabels (tick labels), ChartLegends, PlotRange and Graphics options. Check: "
          "reference (box limits equal Quartiles, whiskers equal Min/Max in tests).")
def _boxwhiskerchart(data: ListOf[Real] | ListOf[ListOf[Real]], *rest: Expr | Option) -> Expr:
    fname = "BoxWhiskerChart"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    o, gopts = chart_options(fname, opts, ("ChartLabels",))
    sets = chart_datasets(pos[0], fname)
    spec = pos[1] if len(pos) > 1 else AUTO
    specs = [spec] if not K.is_list(spec) else list(spec.args)
    outliers = mean = False
    for s in specs:
        if is_sym(s, "Automatic") or s == "Median":
            continue
        if s == "Outliers":
            outliers = True
        elif s == "Mean":
            mean = True
        else:
            raise NotImplementedError(f"{fname}: the box-whisker specification {K_input(s)} is not supported")
    styles = chart_styles(o, len(sets), fname)
    content, ys = [], []
    for i, (vs, st) in enumerate(zip(sets, styles)):
        if not vs:
            continue
        x = i + 1.0
        q1, q2, q3 = quartiles(vs, fname)
        fv = sorted(float(v) for v in vs)
        lo, hi = fv[0], fv[-1]
        out_pts = []
        if outliers:
            iqr = q3 - q1
            inside = [v for v in fv if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
            lo, hi = (min(min(inside), q1), max(max(inside), q3)) if inside else (q1, q3)
            out_pts = [v for v in fv if v < q1 - 1.5 * iqr or v > q3 + 1.5 * iqr]
        ys += fv
        prims = [Expr("Line", pts([(x, q3), (x, hi)])), Expr("Line", pts([(x, q1), (x, lo)])),
                 Expr("Rectangle", pt(x - 0.3, q1), pt(x + 0.3, q3)),
                 L(Expr("GrayLevel", 1), Expr("Line", pts([(x - 0.3, q2), (x + 0.3, q2)])))]
        if mean:
            mu = sum(fv) / len(fv)
            prims.append(L(Expr("GrayLevel", 1), Expr("Dashing", L(0.02)), Expr("Line", pts([(x - 0.3, mu),
                                                                                             (x + 0.3, mu)]))))
        if out_pts:
            prims.append(Expr("Point", pts([(x, v) for v in out_pts])))
        if st is not None:
            content.append(L(st, *prims))
    yr = P2.data_range(ys)
    r = _range_2d(o, fname, (0.5, len(sets) + 0.5), yr)
    tl = _ticks_labels(o, len(sets), fname)
    bottom = tl.args[0] if K.is_list(tl) else NONE
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Frame", True),
                                             ("FrameTicks", L(L(AUTO, NONE), L(bottom, NONE))),
                                             ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


@fn("DistributionChart", area=AREA, status="added",
    notes="DistributionChart[{data1, data2, ...}] and [data, \"SmoothDensity\"]: a violin per dataset at x = 1, 2, "
          "...: the Polygon {i +- 0.4 f(y)/fmax, y} for y over [min, max] of the dataset, where f is the Gaussian "
          "kernel density estimate with Silverman's bandwidth (as SmoothHistogram), plus a median Line. Other "
          "element specifications (\"Density\" histograms, \"Quantile\", glass styles) and grouped datasets raise "
          "NotImplementedError. Options: ChartStyle, ChartBaseStyle, ChartLabels, ChartLegends, PlotRange and "
          "Graphics options. Check: reference (violin half-widths proportional to the kernel density in tests).")
def _distributionchart(data: ListOf[ListOf[Real]], *rest: Expr | Option) -> Expr:
    fname = "DistributionChart"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    o, gopts = chart_options(fname, opts, ("ChartLabels",))
    sets = chart_datasets(pos[0], fname)
    spec = pos[1] if len(pos) > 1 else AUTO
    if not (is_sym(spec, "Automatic") or spec == "SmoothDensity"):
        raise NotImplementedError(f"{fname}: the element specification {K_input(spec)} is not supported "
                                  "(\"SmoothDensity\")")
    np = _np()
    styles = chart_styles(o, len(sets), fname)
    content, ys = [], []
    for i, (vs, st) in enumerate(zip(sets, styles)):
        if len(vs) < 2:
            raise ValueError(f"{fname}: each dataset needs at least two values")
        x = i + 1.0
        h = bandwidth(vs, None, fname)
        f = kde_fun(vs, h, "PDF", fname)
        lo, hi = min(float(v) for v in vs), max(float(v) for v in vs)
        yy = np.linspace(lo, hi, 61)
        dens = f(yy)
        w = 0.4 * dens / dens.max()
        poly = [(x + float(a), float(y)) for a, y in zip(w, yy)] + \
               [(x - float(a), float(y)) for a, y in zip(w[::-1], yy[::-1])]
        med = float(num(ev(Expr("Median", L(*[_out(v) for v in vs])))))
        ys += [lo, hi]
        if st is not None:
            content.append(L(st, Expr("Polygon", pts(poly)),
                             L(Expr("GrayLevel", 1), Expr("Line", pts([(x - 0.1, med), (x + 0.1, med)])))))
    r = _range_2d(o, fname, (0.5, len(sets) + 0.5), P2.data_range(ys))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Frame", True), ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


def tuples_of(d, fname, k, what):
    """[(v1, ..., vk)] of a list of k-tuples (missing tuples dropped)."""
    d, _, _ = P2._unwrap(d, fname)
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of {what}, not {K_input(d)}")
    out = []
    for p in d.args:
        if is_missing(p):
            continue
        p, _, _ = P2._unwrap(p, fname)
        if not K.is_list(p) or len(p.args) != k:
            if isinstance(p, (Symbol,)) or (isinstance(p, Expr) and not K.is_list(p)):
                raise Unevaluated()
            raise ValueError(f"{fname}: the data must be a list of {what}; {K_input(p)} is not")
        vs = [P2.scalar(c, fname) for c in p.args]
        if any(v is None for v in vs):
            continue
        out.append(tuple(vs))
    if not out:
        raise ValueError(f"{fname}: the data has no numeric {what}")
    return out


def tuple_datasets(data, fname, k, what):
    du, _, _ = P2._unwrap(data, fname)
    if K.is_list(du) and du.args and all(K.is_list(a) and a.args and all(K.is_list(b) for b in a.args)
                                         for a in du.args):
        return [tuples_of(a, fname, k, what) for a in du.args]
    return [tuples_of(data, fname, k, what)]


@fn("BubbleChart", area=AREA, status="added",
    notes="BubbleChart[{{x1, y1, z1}, ...}] and lists of datasets: Graphics[{{style, Disk[{x, y}, r], ...}, ...}] "
          "with one group per dataset. Bubble diameters are fractions of the larger coordinate span: BubbleSizes "
          "{smin, smax} (default {0.05, 0.2}) for the smallest and largest z of all datasets, with the area "
          "(BubbleScale \"Area\", default, as documented) or the diameter (\"Diameter\") interpolated linearly in z. "
          "Options: BubbleSizes, BubbleScale, ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and Graphics "
          "options. Check: reference (bubble areas affine in z in tests).")
def _bubblechart(data: MatrixOf[Real] | ListOf[MatrixOf[Real]], *opts: Option) -> Expr:
    fname = "BubbleChart"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("BubbleSizes", "BubbleScale"))
    sets = tuple_datasets(pos[0], fname, 3, "triples {x, y, z}")
    smin, smax = _bubble_sizes(o["BubbleSizes"], fname, (0.05, 0.2))
    scale = _bubble_scale(o["BubbleScale"], fname, "Area")
    allz = [p[2] for s in sets for p in s]
    xr = P2.data_range([p[0] for s in sets for p in s])
    yr = P2.data_range([p[1] for s in sets for p in s])
    span = max(xr[1] - xr[0], yr[1] - yr[0]) or 1.0
    ds = bubble_diameters(allz, smin, smax, scale)
    styles = chart_styles(o, len(sets), fname)
    content, k, rmax = [], 0, 0.0
    for s, st in zip(sets, styles):
        disks = []
        for p in s:
            r = ds[k] * span / 2
            k += 1
            rmax = max(rmax, r)
            disks.append(Expr("Disk", pt(p[0], p[1]), fl(r)))
        if st is not None:
            content.append(L(st, *disks))
    r = _range_2d(o, fname, (xr[0] - rmax, xr[1] + rmax), (yr[0] - rmax, yr[1] + rmax))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


@fn("BubbleChart3D", area=AREA, status="added",
    notes="BubbleChart3D[{{x1, y1, z1, u1}, ...}] and lists of datasets: Graphics3D[{{style, Sphere[{x, y, z}, r], "
          "...}, ...}] with diameters as fractions of the largest coordinate span: BubbleSizes {smin, smax} (default "
          "{0.05, 0.2}) for the smallest and largest u, the volume (BubbleScale \"Volume\", default), area or "
          "diameter interpolated linearly in u. Options: BubbleSizes, BubbleScale, ChartStyle, ChartBaseStyle, "
          "ChartLegends, PlotRange and Graphics3D options. Check: reference (sphere volumes affine in u in tests).")
def _bubblechart3d(data: MatrixOf[Real] | ListOf[MatrixOf[Real]], *opts: Option) -> Expr:
    fname = "BubbleChart3D"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("BubbleSizes", "BubbleScale"), dim=3)
    sets = tuple_datasets(pos[0], fname, 4, "quadruples {x, y, z, u}")
    smin, smax = _bubble_sizes(o["BubbleSizes"], fname, (0.05, 0.2))
    sc = o["BubbleScale"]
    scale = "Volume" if is_sym(sc, "Automatic") or sc == "Volume" else _bubble_scale(sc, fname, "Area")
    us = [p[3] for s in sets for p in s]
    if scale == "Volume":
        ulo, uhi = min(us), max(us)
        ds = [((smin ** 3 + ((u - ulo) / (uhi - ulo) if uhi > ulo else 1.0) * (smax ** 3 - smin ** 3)) ** (1 / 3))
              for u in us]
    else:
        ds = bubble_diameters(us, smin, smax, scale)
    rngs = [P2.data_range([p[c] for s in sets for p in s]) for c in range(3)]
    span = max(b - a for a, b in rngs) or 1.0
    styles = chart_styles(o, len(sets), fname)
    content, k = [], 0
    for s, st in zip(sets, styles):
        sph = []
        for p in s:
            sph.append(Expr("Sphere", pt(*p[:3]), fl(ds[k] * span / 2)))
            k += 1
        if st is not None:
            content.append(L(st, *sph))
    rm = max(ds) * span / 2
    r = P2.resolve_plot_range(o["PlotRange"], fname, [(a - rm, b + rm) for a, b in rngs], dims=3)
    g = finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 1, 1)), ("PlotRange", range_expr(r))],
                        dim=3)
    return _legend(g, o, styles, fname)


def _bar_data(data, fname):
    """[[y_gj]] groups of bar heights (a flat list is one group per value: groups of size 1)."""
    du, _, _ = P2._unwrap(data, fname)
    if K.is_list(du) and du.args and all(K.is_list(a) for a in du.args):
        groups = [values_of(a, fname) for a in du.args]
    else:
        groups = [[v] for v in values_of(data, fname)]
    return groups


@fn("BarChart3D", area=AREA, status="added",
    notes="BarChart3D[{y1, y2, ...}] and [{{y11, y12, ...}, ...}] (groups): Graphics3D with one Cuboid[{p - 0.4, "
          "-0.4, min(0, y)}, {p + 0.4, 0.4, max(0, y)}] per value at positions p = 1, 2, ... along x (groups are "
          "separated by one empty slot) and colours by position within a group (one colour per value for a flat "
          "list). Missing values leave gaps. Options: ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and "
          "Graphics3D options; ChartLayout and other settings raise NotImplementedError. Check: reference (cuboid "
          "heights equal the data in tests).")
def _barchart3d(data: ListOf[Real] | MatrixOf[Real], *opts: Option) -> Expr:
    fname = "BarChart3D"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, (), dim=3)
    groups = _bar_data(pos[0], fname)
    flat = all(len(g) == 1 for g in groups)
    m = max(len(g) for g in groups)
    styles = chart_styles(o, len(groups) if flat else m, fname)
    content, p, zs = [], 0, [0.0]
    for gi, g in enumerate(groups):
        for j, y in enumerate(g):
            p += 1
            y = float(y)
            zs.append(y)
            st = styles[gi] if flat else styles[j]
            if st is not None:
                content.append(L(st, Expr("Cuboid", pt(p - 0.4, -0.4, min(0.0, y)), pt(p + 0.4, 0.4, max(0.0, y)))))
        if not flat:
            p += 1
    r = P2.resolve_plot_range(o["PlotRange"], fname, [(0.5, p + 0.5), (-0.5, 0.5), P2.data_range(zs)], dims=3)
    g = finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 0.3, 0.5)),
                                             ("PlotRange", range_expr(r))], dim=3)
    return _legend(g, o, styles, fname)


def _bar_spacing(o, fname):
    v = o["BarSpacing"]
    if is_sym(v, "Automatic", "None"):
        return 0.0
    x = num(v)
    if x is None or x < 0:
        raise NotImplementedError(f"{fname}: BarSpacing -> {K_input(v)} is not supported; give a number >= 0")
    return x


@fn("RectangleChart", area=AREA, status="added",
    notes="RectangleChart[{{x1, y1}, {x2, y2}, ...}] and lists of datasets: rectangles of width x_i and height y_i "
          "placed side by side from x = 0 (Rectangle[{X_i, 0}, {X_i + x_i, y_i}], X_i the running sum of widths "
          "plus BarSpacing s between rectangles, default 0); widths must be non-negative. Options: BarSpacing (a "
          "number), ChartStyle, ChartBaseStyle, ChartLabels, ChartLegends, PlotRange and Graphics options. Check: "
          "reference (rectangle widths and heights equal the data in tests).")
def _rectanglechart(data: MatrixOf[Real] | ListOf[MatrixOf[Real]], *opts: Option) -> Expr:
    fname = "RectangleChart"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("BarSpacing", "ChartLabels"))
    sets = tuple_datasets(pos[0], fname, 2, "pairs {width, height}")
    gap = _bar_spacing(o, fname)
    n = max(len(s) for s in sets)
    styles = chart_styles(o, n, fname)
    content, X, ys = [], 0.0, [0.0]
    for s in sets:
        for i, (w, h) in enumerate(s):
            if w < 0:
                raise ValueError(f"{fname}: widths must be non-negative, not {w}")
            if styles[i] is not None:
                content.append(L(styles[i], Expr("Rectangle", pt(X, min(0.0, h)), pt(X + w, max(0.0, h)))))
            ys.append(h)
            X += w + gap
        X += gap
    r = _range_2d(o, fname, (0.0, max(X - 2 * gap, 1e-9) if X else 1.0), P2.data_range(ys))
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


@fn("RectangleChart3D", area=AREA, status="added",
    notes="RectangleChart3D[{{x1, y1, z1}, ...}] and lists of datasets: cuboids of width x_i, depth y_i and height "
          "z_i side by side along x from 0 (Cuboid[{X_i, 0, 0}, {X_i + x_i, y_i, z_i}], X_i the running sum of "
          "widths plus BarSpacing). Widths and depths must be non-negative. Options as RectangleChart. Check: "
          "reference (cuboid sizes equal the data in tests).")
def _rectanglechart3d(data: MatrixOf[Real] | ListOf[MatrixOf[Real]], *opts: Option) -> Expr:
    fname = "RectangleChart3D"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("BarSpacing", "ChartLabels"), dim=3)
    sets = tuple_datasets(pos[0], fname, 3, "triples {width, depth, height}")
    gap = _bar_spacing(o, fname)
    n = max(len(s) for s in sets)
    styles = chart_styles(o, n, fname)
    content, X, ys, zs = [], 0.0, [0.0], [0.0]
    for s in sets:
        for i, (w, d, h) in enumerate(s):
            if w < 0 or d < 0:
                raise ValueError(f"{fname}: widths and depths must be non-negative")
            if styles[i] is not None:
                content.append(L(styles[i], Expr("Cuboid", pt(X, 0.0, min(0.0, h)), pt(X + w, d, max(0.0, h)))))
            ys.append(d)
            zs.append(h)
            X += w + gap
        X += gap
    r = P2.resolve_plot_range(o["PlotRange"], fname, [(0.0, X or 1.0), P2.data_range(ys), P2.data_range(zs)], dims=3)
    g = finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", Symbol("Automatic")),
                                             ("PlotRange", range_expr(r))], dim=3)
    return _legend(g, o, styles, fname)


def sector_origin(o, fname):
    """(start angle, direction +1 counterclockwise / -1 clockwise)."""
    v = o["SectorOrigin"]
    if is_sym(v, "Automatic"):
        return 0.0, 1
    names = {"Right": 0.0, "Top": math.pi / 2, "Left": math.pi, "Bottom": -math.pi / 2}
    if isinstance(v, Symbol) and v.name in names:
        return names[v.name], 1
    x = num(v)
    if x is not None:
        return x, 1
    if K.is_list(v) and len(v.args) == 2:
        a, d = v.args
        a0 = names[a.name] if isinstance(a, Symbol) and a.name in names else (0.0 if is_sym(a, "Automatic") else num(a))
        if a0 is None:
            raise ValueError(f"{fname}: SectorOrigin -> {K_input(v)} is not valid")
        if d == "Clockwise" or d == -1:
            return a0, -1
        if d == "Counterclockwise" or d == 1:
            return a0, 1
    raise NotImplementedError(f"{fname}: SectorOrigin -> {K_input(v)} is not supported")


def sector_angles(weights, a0, sense, fname):
    tot = sum(weights)
    if any(w < 0 for w in weights):
        raise ValueError(f"{fname}: sector sizes must be non-negative")
    if not tot > 0:
        raise ValueError(f"{fname}: the sector sizes must have a positive total")
    out, a = [], a0
    for w in weights:
        b = a + sense * 2 * math.pi * w / tot
        out.append((min(a, b), max(a, b)))
        a = b
    return out


@fn("SectorChart", area=AREA, status="added",
    notes="SectorChart[{{a1, r1}, {a2, r2}, ...}] and lists of datasets (concentric rings are not supported: "
          "NotImplementedError): sector i spans the angle 2 Pi a_i/Total[a] with radius r_i, drawn as Disk[{0, 0}, "
          "r_i, {t_i, t_(i+1)}]; sectors start at angle 0 and go counterclockwise (SectorOrigin -> t0, a named side "
          "or {t0, \"Clockwise\"} changes this). Negative sizes or radii raise ValueError. Options: SectorOrigin, "
          "ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and Graphics options. Check: reference (sector angles "
          "proportional to a_i and radii r_i in tests).")
def _sectorchart(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "SectorChart"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("SectorOrigin",))
    sets = tuple_datasets(pos[0], fname, 2, "pairs {angle size, radius}")
    if len(sets) > 1:
        raise NotImplementedError(f"{fname}: several datasets (concentric sector rings) are not supported")
    s = sets[0]
    a0, sense = sector_origin(o, fname)
    angs = sector_angles([p[0] for p in s], a0, sense, fname)
    styles = chart_styles(o, len(s), fname)
    content, rmax = [], 0.0
    for (a, r), (t0, t1), st in zip(s, angs, styles):
        if r < 0:
            raise ValueError(f"{fname}: radii must be non-negative")
        rmax = max(rmax, r)
        if st is not None and t1 > t0:
            content.append(L(st, Expr("Disk", pt(0, 0), fl(r), L(fl(t0), fl(t1)))))
    rr = rmax or 1.0
    g = finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("PlotRange", range_expr(
        _range_2d(o, fname, (-rr, rr), (-rr, rr))))])
    return _legend(g, o, styles, fname)


def wedge_polygons(t0, t1, r, h, rin=0.0):
    """Polygons of a 3-D pie wedge between angles t0 < t1, radius r, from z = 0 to z = h."""
    n = max(2, int(math.ceil((t1 - t0) / (math.pi / 36))) + 1)
    ts = [t0 + (t1 - t0) * k / (n - 1) for k in range(n)]
    arc = [(r * math.cos(t), r * math.sin(t)) for t in ts]
    inner = [(rin * math.cos(t), rin * math.sin(t)) for t in reversed(ts)] if rin > 0 else [(0.0, 0.0)]
    top = [(x, y, h) for x, y in arc + inner]
    bot = [(x, y, 0.0) for x, y in arc + inner]
    polys = [top, bot[::-1]]
    for (x0, y0), (x1, y1) in zip(arc, arc[1:]):
        polys.append([(x0, y0, 0.0), (x1, y1, 0.0), (x1, y1, h), (x0, y0, h)])
    for x, y in (arc[0], arc[-1]):
        cx, cy = inner[-1] if (x, y) == arc[0] else inner[0]
        polys.append([(cx, cy, 0.0), (x, y, 0.0), (x, y, h), (cx, cy, h)])
    return Expr("Polygon", L(*[pts(p) for p in polys]))


@fn("PieChart3D", area=AREA, status="added",
    notes="PieChart3D[{y1, y2, ...}]: Graphics3D with one wedge per value: the solid between angles t_i and t_(i+1) "
          "(sector angle 2 Pi y_i/Total[y], from angle 0 counterclockwise; SectorOrigin as SectorChart), radius 1 "
          "and height 0.2, given as Polygon faces (top, bottom, outer side approximated by facets of at most 5 "
          "degrees, and the two radial sides). Negative values raise ValueError; missing values are dropped; "
          "several datasets (rings) raise NotImplementedError. Options: SectorOrigin, ChartStyle, ChartBaseStyle, "
          "ChartLegends, PlotRange and Graphics3D options. Check: reference (wedge angles and radius in tests).")
def _piechart3d(data: ListOf[Real], *opts: Option) -> Expr:
    fname = "PieChart3D"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("SectorOrigin",), dim=3)
    du, _, _ = P2._unwrap(pos[0], fname)
    if K.is_list(du) and du.args and all(K.is_list(a) for a in du.args):
        raise NotImplementedError(f"{fname}: several datasets (rings) are not supported")
    vs = [float(v) for v in values_of(pos[0], fname)]
    a0, sense = sector_origin(o, fname)
    angs = sector_angles(vs, a0, sense, fname)
    styles = chart_styles(o, len(vs), fname)
    content = [L(st, wedge_polygons(t0, t1, 1.0, 0.2)) for (t0, t1), st in zip(angs, styles)
               if st is not None and t1 > t0]
    r = P2.resolve_plot_range(o["PlotRange"], fname, [(-1.0, 1.0), (-1.0, 1.0), (0.0, 0.2)], dims=3)
    g = finish_graphics(L(*content), gopts, [("Boxed", False), ("BoxRatios", Symbol("Automatic")),
                                             ("PlotRange", range_expr(r))], dim=3)
    return _legend(g, o, styles, fname)


@fn("SectorChart3D", area=AREA, status="added",
    notes="SectorChart3D[{{a1, r1, h1}, ...}]: 3-D sectors with angle 2 Pi a_i/Total[a], radius r_i and height h_i "
          "(Polygon faces as PieChart3D), from angle 0 counterclockwise (SectorOrigin as SectorChart). Negative "
          "sizes, radii or heights raise ValueError; several datasets raise NotImplementedError. Options: "
          "SectorOrigin, ChartStyle, ChartBaseStyle, ChartLegends, PlotRange and Graphics3D options. Check: "
          "reference (wedge angles, radii and heights in tests).")
def _sectorchart3d(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "SectorChart3D"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = chart_options(fname, opts_, ("SectorOrigin",), dim=3)
    sets = tuple_datasets(pos[0], fname, 3, "triples {angle size, radius, height}")
    if len(sets) > 1:
        raise NotImplementedError(f"{fname}: several datasets are not supported")
    s = sets[0]
    a0, sense = sector_origin(o, fname)
    angs = sector_angles([p[0] for p in s], a0, sense, fname)
    styles = chart_styles(o, len(s), fname)
    content, rmax, hmax = [], 0.0, 0.0
    for (a, r, h), (t0, t1), st in zip(s, angs, styles):
        if r < 0 or h < 0:
            raise ValueError(f"{fname}: radii and heights must be non-negative")
        rmax, hmax = max(rmax, r), max(hmax, h)
        if st is not None and t1 > t0:
            content.append(L(st, wedge_polygons(t0, t1, r, h)))
    rr = rmax or 1.0
    rg = P2.resolve_plot_range(o["PlotRange"], fname, [(-rr, rr), (-rr, rr), (0.0, hmax or 1.0)], dims=3)
    g = finish_graphics(L(*content), gopts, [("BoxRatios", Symbol("Automatic")), ("PlotRange", range_expr(rg))],
                        dim=3)
    return _legend(g, o, styles, fname)


# ───────────────────────────── financial charts ─────────────────────────────

UP = Expr("RGBColor", 0.2, 0.6, 0.2)
DOWN = Expr("RGBColor", 0.8, 0.2, 0.2)
_FIN_DEFAULTS = {"TrendStyle": AUTO, "ChartBaseStyle": AUTO, "ChartElementFunction": AUTO, "ChartElements": AUTO,
                 "DateFunction": AUTO, "EventLabels": NONE, "PerformanceGoal": AUTO, "PlotTheme": AUTO,
                 "ScalingFunctions": NONE, "TargetUnits": AUTO, "PlotRange": AUTO, "Method": AUTO,
                 "ChartStyle": AUTO, "PlotInteractivity": AUTO}


def fin_options(fname, opts):
    o, gopts = split_options(fname, opts, _FIN_DEFAULTS)
    _unsupported(o, fname, ("TrendStyle", "ChartBaseStyle"))
    return o, gopts


def trend_styles(o, fname):
    v = o["TrendStyle"]
    base = [] if is_sym(o["ChartBaseStyle"], "Automatic", "None") else _directive_items(o["ChartBaseStyle"])
    if is_sym(v, "Automatic"):
        up, down = UP, DOWN
    elif K.is_list(v) and len(v.args) == 2:
        G.directive_like(v, fname, "TrendStyle")
        up, down = v.args
    elif isinstance(v, str):
        raise NotImplementedError(f"{fname}: the named TrendStyle {v!r} is not supported; give {{up, down}}")
    elif is_sym(v, "None"):
        raise NotImplementedError(f"{fname}: TrendStyle -> None is not supported; give {{up, down}}")
    else:
        G.directive_like(v, fname, "TrendStyle")
        up, down = v, Expr("GrayLevel", 1)
    return (Expr("Directive", *base, *_directive_items(up)), Expr("Directive", *base, *_directive_items(down)))


def dated_values(data, fname, k=1, what="date-price pairs {date, p}"):
    """[(date expr or index, value(s))] of {{date, p}, ...} (k = 1) or {{date, {o, h, l, c(, v)}}, ...}; a flat list
    of prices (k = 1) or of tuples is taken as an ordered sequence without dates."""
    d, _, _ = P2._unwrap(data, fname)
    if isinstance(d, str) or (K.is_list(d) and d.args and isinstance(d.args[0], str)):
        raise NotImplementedError(f"{fname}: FinancialData entities (\"name\") need curated online data and are "
                                  "not supported; give the data")
    if isinstance(d, Expr) and d.head in ("TimeSeries", "EventSeries", "TemporalData", "Tabular"):
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of {what}, not {K_input(d)}")
    out = []
    for i, e in enumerate(d.args):
        e, _, _ = P2._unwrap(e, fname)
        if is_rule(e):
            e = e.args[0]
        if is_missing(e):
            continue
        if k == 1 and not K.is_list(e):
            v = P2.scalar(e, fname)
            if v is not None:
                out.append((None, v))
            continue
        if not K.is_list(e):
            raise ValueError(f"{fname}: {K_input(e)} is not one of {what}")
        if k > 1 and len(e.args) >= 4 and not any(K.is_list(a) for a in e.args):
            vals, date = e, None
        elif len(e.args) == 2:
            date, vals = e.args
            if k > 1 and not K.is_list(vals):
                raise ValueError(f"{fname}: {K_input(e)} is not one of {what}")
            if k == 1 and K.is_list(vals):
                raise ValueError(f"{fname}: {K_input(e)} is not one of {what}")
            if date is not None:
                _abs_time(date, fname)            # validates the date
        else:
            raise ValueError(f"{fname}: {K_input(e)} is not one of {what}")
        if k == 1:
            v = P2.scalar(vals, fname)
            if v is not None:
                out.append((date, v))
        else:
            vs = [P2.scalar(a, fname) for a in vals.args]
            if len(vs) not in (4, 5) or (len(vs) < k):
                raise ValueError(f"{fname}: price data must be {{open, high, low, close}} or with volume")
            if any(v is None for v in vs):
                continue
            out.append((date, vs))
    if not out:
        raise ValueError(f"{fname}: the data has no prices")
    return out


def _date_ticks_seq(dates, n=6):
    idx = [i for i, d in enumerate(dates) if d is not None]
    if not idx:
        return AUTO
    step = max(1, len(dates) // n)
    return L(*[L(i + 1, dates[i]) for i in range(0, len(dates), step) if dates[i] is not None])


def threshold(spec, prices, fname, default):
    """('abs', t) or ('pct', p) of a Kagi/Renko size specification."""
    rng = max(prices) - min(prices)
    if spec is None or is_sym(spec, "Automatic"):
        return "abs", default * rng
    v = num(spec)
    if v is not None and not K.is_list(spec):
        if not v > 0:
            raise ValueError(f"{fname}: the size must be positive")
        return "abs", v * rng
    if K.is_list(spec) and len(spec.args) == 2 and isinstance(spec.args[0], str):
        kind = spec.args[0]
        p = float(pos_number(spec.args[1], fname, "the size"))
        if kind == "Relative":
            return "abs", p * rng
        if kind == "Absolute":
            return "abs", p
        if kind == "Percent":
            return "pct", p / 100
        raise ValueError(f"{fname}: unknown size specification {kind!r} (\"Relative\", \"Absolute\", \"Percent\")")
    raise ValueError(f"{fname}: {K_input(spec)} is not a valid size specification")


def kagi_columns(prices, thr):
    """Kagi columns [(start, end)]: a column continues while prices move in its direction and reverses when the
    price moves against it by more than the threshold (absolute, or a fraction of the column extreme)."""
    kind, t = thr

    def T(e):
        return t * abs(e) if kind == "pct" else t
    p0 = prices[0]
    s, e, d = p0, p0, 0
    cols = []
    for p in prices[1:]:
        if d == 0:
            if abs(p - p0) > T(p0):
                d = 1 if p > p0 else -1
                e = p
        elif d == 1:
            if p > e:
                e = p
            elif e - p > T(e):
                cols.append((s, e))
                s, e, d = e, p, -1
        else:
            if p < e:
                e = p
            elif p - e > T(e):
                cols.append((s, e))
                s, e, d = e, p, 1
    cols.append((s, e))
    return cols


def kagi_segments(cols):
    """[(x0, y0, x1, y1, thick)] line segments (verticals at x = k, horizontal connectors) with the yang/yin state:
    thick after the price rises above the previous column high, thin after it falls below the previous low."""
    segs = []
    thick = cols[0][1] >= cols[0][0]
    for k, (s, e) in enumerate(cols):
        x = k + 1
        if k >= 1:
            segs.append((x - 1, s, x, s, thick))
        prev = cols[k - 1] if k >= 1 else None
        if prev is not None:
            if e > s:
                lvl = max(prev)                     # the previous high (start of the down column)
                if not thick and e > lvl:
                    segs.append((x, s, x, lvl, thick))
                    thick = True
                    segs.append((x, lvl, x, e, thick))
                    continue
            else:
                lvl = min(prev)
                if thick and e < lvl:
                    segs.append((x, s, x, lvl, thick))
                    thick = False
                    segs.append((x, lvl, x, e, thick))
                    continue
        segs.append((x, s, x, e, thick))
    return segs


@fn("KagiChart", area=AREA, status="added",
    notes="KagiChart[{{date1, p1}, ...}] (or a list of prices) and [data, rt]: vertical lines at x = 1, 2, ... (one "
          "column per trend) joined by horizontal connectors; a column continues while the price moves in its "
          "direction and a new column starts when the price reverses by more than the threshold rt (as "
          "documented): {\"Relative\", p} (p times the price range, default p = 0.02), {\"Absolute\", p}, "
          "{\"Percent\", p} (p% of the column extreme) or a number p (= {\"Relative\", p}). Lines are thick "
          "(yang, first TrendStyle, default green) after the price rises above the previous column high and thin "
          "(yin, second style, default red) after it falls below the previous low. Dates are an ordered sequence "
          "(as documented) used as tick labels. FinancialData entity names, EventLabels and TimeSeries data raise "
          "NotImplementedError. Check: reference (columns of a hand-worked example in tests).")
def _kagichart(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "KagiChart"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    o, gopts = fin_options(fname, opts)
    dv = dated_values(pos[0], fname)
    prices = [v for _, v in dv]
    thr = threshold(pos[1] if len(pos) > 1 else None, prices, fname, 0.02)
    cols = kagi_columns(prices, thr)
    segs = kagi_segments(cols)
    up, down = trend_styles(o, fname)
    yang = [Expr("Line", pts([(a, b), (c, d)])) for a, b, c, d, t in segs if t]
    yin = [Expr("Line", pts([(a, b), (c, d)])) for a, b, c, d, t in segs if not t]
    content = []
    if yang:
        content.append(L(up, Expr("AbsoluteThickness", 3), *yang))
    if yin:
        content.append(L(down, Expr("AbsoluteThickness", 1), *yin))
    r = _range_2d({"PlotRange": o["PlotRange"]}, fname, (0.5, len(cols) + 0.5), P2.data_range(prices))
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])


def renko_bricks(prices, b):
    """Renko bricks [(lo, hi, up)]: a brick is added when the price rises above the top of the last brick or falls
    below its bottom by the brick height b (as documented), one brick per full height."""
    lo = hi = prices[0]
    bricks = []
    for p in prices[1:]:
        while p >= hi + b - 1e-12 * max(1.0, abs(hi)):
            bricks.append((hi, hi + b, True))
            lo, hi = hi, hi + b
        while p <= lo - b + 1e-12 * max(1.0, abs(lo)):
            bricks.append((lo - b, lo, False))
            lo, hi = lo - b, lo
    return bricks


@fn("RenkoChart", area=AREA, status="added",
    notes="RenkoChart[{{date1, p1}, ...}] (or a list of prices) and [data, s]: fixed-height bricks Rectangle[{k - "
          "0.45, lo_k}, {k + 0.45, lo_k + b}], one per column k = 1, 2, ...: starting from the first price, a brick is "
          "added when the price rises above the top of the last brick or drops below its bottom by the brick "
          "height b (as documented; a reversal therefore needs a move of 2 b from the last brick's far edge). "
          "Brick height: {\"Relative\", p} (p times the price range, default p = 0.1), {\"Absolute\", p} or a "
          "number p (= {\"Relative\", p}). Up bricks use the first TrendStyle (default green), down bricks the "
          "second (red). FinancialData names, EventLabels and TimeSeries raise NotImplementedError. Check: "
          "reference (bricks of a hand-worked example in tests).")
def _renkochart(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "RenkoChart"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    o, gopts = fin_options(fname, opts)
    prices = [v for _, v in dated_values(pos[0], fname)]
    kind, b = threshold(pos[1] if len(pos) > 1 else None, prices, fname, 0.1)
    if kind != "abs":
        raise NotImplementedError(f"{fname}: percentage brick sizes are not supported")
    if not b > 0:
        raise ValueError(f"{fname}: the prices have no range; give {{\"Absolute\", b}}")
    bricks = renko_bricks(prices, b)
    up, down = trend_styles(o, fname)
    ups = [Expr("Rectangle", pt(k + 1 - 0.45, a), pt(k + 1 + 0.45, c)) for k, (a, c, u) in enumerate(bricks) if u]
    dns = [Expr("Rectangle", pt(k + 1 - 0.45, a), pt(k + 1 + 0.45, c)) for k, (a, c, u) in enumerate(bricks) if not u]
    content = ([L(up, *ups)] if ups else []) + ([L(down, *dns)] if dns else [])
    ys = [a for a, _, _ in bricks] + [c for _, c, _ in bricks] + prices[:1]
    r = _range_2d({"PlotRange": o["PlotRange"]}, fname, (0.5, len(bricks) + 0.5), P2.data_range(ys))
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("PlotRange", range_expr(r))])


def candles(dv, fname, up, down):
    content_up, content_dn, wicks, ys = [], [], [], []
    for i, (_d, (o_, h, l_, c) ) in enumerate([(d, v[:4]) for d, v in dv]):
        if h < max(o_, c) or l_ > min(o_, c):
            raise ValueError(f"{fname}: element {i + 1} has high < max(open, close) or low > min(open, close)")
        x = i + 1.0
        wicks.append(Expr("Line", pts([(x, l_), (x, h)])))
        body = Expr("Rectangle", pt(x - 0.3, min(o_, c)), pt(x + 0.3, max(o_, c)))
        (content_up if c >= o_ else content_dn).append(body)
        ys += [l_, h]
    content = [L(Expr("GrayLevel", 0.3), *wicks)]
    if content_up:
        content.append(L(up, Expr("EdgeForm", Expr("GrayLevel", 0.3)), *content_up))
    if content_dn:
        content.append(L(down, Expr("EdgeForm", Expr("GrayLevel", 0.3)), *content_dn))
    return content, ys


@fn("CandlestickChart", area=AREA, status="added",
    notes="CandlestickChart[{{date1, {o1, h1, l1, c1}}, ...}] or [{{o1, h1, l1, c1}, ...}]: one candlestick per "
          "element at x = 1, 2, ... (dates as an ordered sequence with date tick labels): wick Line[{{x, low}, "
          "{x, high}}] and body Rectangle[{x - 0.3, Min[o, c]}, {x + 0.3, Max[o, c]}], rising candles (c >= o) in "
          "the first TrendStyle (default green), falling ones in the second (red). A high below max(o, c) or low "
          "above min(o, c) raises ValueError; FinancialData names and TimeSeries raise NotImplementedError. "
          "Options: TrendStyle, ChartBaseStyle, PlotRange and Graphics options. Check: reference (wick and body "
          "extents equal the OHLC values in tests).")
def _candlestickchart(data: ListOf[Expr], *opts: Option) -> Expr:
    fname = "CandlestickChart"
    pos, opts_ = split_args((data,) + opts, fname, 1, 1)
    o, gopts = fin_options(fname, opts_)
    dv = dated_values(pos[0], fname, k=4, what="{date, {open, high, low, close}} elements")
    up, down = trend_styles(o, fname)
    content, ys = candles(dv, fname, up, down)
    r = _range_2d({"PlotRange": o["PlotRange"]}, fname, (0.5, len(dv) + 0.5), P2.data_range(ys))
    ticks = _date_ticks_seq([d for d, _ in dv])
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("Ticks", L(ticks, AUTO)),
                                                ("PlotRange", range_expr(r))])


@fn("TradingChart", area=AREA, status="added",
    notes="TradingChart[{{date1, {o1, h1, l1, c1, v1}}, ...}] and [data, {\"Volume\"}]: GraphicsColumn[{candlestick "
          "chart (as CandlestickChart), volume bar chart}] with volume bars Rectangle[{x - 0.3, 0}, {x + 0.3, v}] "
          "(the volume panel is omitted when the data has no volumes). Other indicators (FinancialIndicator "
          "objects, \"SMA\", ...), FinancialData names and interactive controls raise NotImplementedError. "
          "Options: TrendStyle, ChartBaseStyle and GraphicsColumn options. Check: reference (candles and volume "
          "bars equal the data in tests).")
def _tradingchart(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "TradingChart"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    o, gopts = fin_options(fname, opts)
    inds = pos[1] if len(pos) > 1 else L()
    ind_list = list(inds.args) if K.is_list(inds) else [inds]
    for ind in ind_list:
        if ind not in ("Volume", "V"):
            raise NotImplementedError(f"{fname}: the indicator {K_input(ind)} is not supported (only \"Volume\")")
    dv = dated_values(pos[0], fname, k=4, what="{date, {open, high, low, close, volume}} elements")
    up, down = trend_styles(o, fname)
    content, ys = candles(dv, fname, up, down)
    ticks = _date_ticks_seq([d for d, _ in dv])
    xr = (0.5, len(dv) + 0.5)
    panels = [Expr("Graphics", L(*content), Expr("Rule", Symbol("AspectRatio"), Expr("Times", 1 / 2)),
                   Expr("Rule", Symbol("Axes"), True), Expr("Rule", Symbol("Ticks"), L(ticks, AUTO)),
                   Expr("Rule", Symbol("PlotRange"), range_expr([xr, P2.data_range(ys)])))]
    vols = [v[4] for _, v in dv if len(v) == 5]
    if vols and len(vols) == len(dv):
        if any(v < 0 for v in vols):
            raise ValueError(f"{fname}: volumes must be non-negative")
        vb = [Expr("Rectangle", pt(i + 1 - 0.3, 0.0), pt(i + 1 + 0.3, v)) for i, v in enumerate(vols)]
        panels.append(Expr("Graphics", L(L(Expr("GrayLevel", 0.5), *vb)), Expr("Rule", Symbol("AspectRatio"), 0.2),
                           Expr("Rule", Symbol("Axes"), True),
                           Expr("Rule", Symbol("PlotRange"), range_expr([xr, (0.0, max(vols) or 1.0)]))))
    elif any(i in ("Volume", "V") for i in ind_list):
        raise ValueError(f"{fname}: the Volume indicator needs {{open, high, low, close, volume}} data")
    return Expr("GraphicsColumn", L(*panels), *gopts)


# ───────────────────────────── word clouds ─────────────────────────────

def _word_weights(data, fname):
    d, _, _ = P2._unwrap(data, fname)
    if isinstance(d, str):
        raise NotImplementedError(f"{fname}: word clouds of a text string need Wolfram's stopword list and are not "
                                  "supported; give a list of words or {word, weight} pairs")
    if isinstance(d, Expr) and d.head == "Association":
        return [(k, P2.scalar(v, fname)) for k, v in P2._assoc_items(d)]
    if is_rule(d) and K.is_list(d.args[0]) and K.is_list(d.args[1]):
        ws, ss = d.args[0].args, d.args[1].args
        if len(ws) != len(ss):
            raise ValueError(f"{fname}: the weight and word lists must have the same length")
        return [(s, P2.scalar(w, fname)) for w, s in zip(ws, ss)]
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of words, not {K_input(d)}")
    items = list(d.args)
    if items and all(is_rule(e) for e in items):
        return [(e.args[1], P2.scalar(e.args[0], fname)) for e in items]
    if items and all(K.is_list(e) and len(e.args) == 2 and num(e.args[1]) is not None for e in items):
        return [(e.args[0], P2.scalar(e.args[1], fname)) for e in items]
    counts, order = {}, []
    for e in items:
        if is_missing(e):
            continue
        k = K_input(e)
        if k not in counts:
            counts[k] = [e, 0]
            order.append(k)
        counts[k][1] += 1
    return [(counts[k][0], float(counts[k][1])) for k in order]


def wordcloud_layout(words, fmax=40.0):
    """[(word, font size, x, y, w, h)]: sizes proportional to the weights (largest fmax), boxes 0.6 size per
    character wide and size high, placed in decreasing weight along an Archimedean spiral at the first position
    where the box overlaps no earlier box."""
    wmax = max(w for _, w in words)
    order = sorted(range(len(words)), key=lambda i: -words[i][1])
    placed = []
    for i in order:
        s, w = words[i]
        fs = fmax * w / wmax
        text = s if isinstance(s, str) else K_input(s)
        bw, bh = 0.6 * fs * max(1, len(text)), fs
        t = 0.0
        while True:
            r = 0.8 * t
            x, y = r * math.cos(t), r * math.sin(t) * 0.6
            ok = all(abs(x - px) * 2 >= bw + pw - 1e-9 or abs(y - py) * 2 >= bh + ph - 1e-9
                     for _, _, px, py, pw, ph in placed)
            if ok:
                placed.append((s, fs, x, y, bw, bh))
                break
            t += 0.05
    return placed


@fn("WordCloud", area=AREA, status="added",
    notes="WordCloud[{s1, s2, ...}] (weights = multiplicities), [{w1 -> s1, ...}], [<|s1 -> w1, ...|>], [{w1, ...} -> "
          "{s1, ...}] and [{{s1, w1}, ...}]: Graphics[{{colour, Text[Style[s, FontSize -> f], {x, y}]}, ...}] with "
          "the font size f proportional to the weight (largest 40; \"linear dimensions proportional to the weight\", "
          "as documented). Layout (Wolfram's is randomised): words in decreasing weight on an Archimedean spiral, "
          "each at the first position where its box (0.6 f per character by f) overlaps no earlier box; the "
          "layout is verified to be overlap-free. Weights must be positive. Text strings (stopword removal), "
          "shape masks, rotations and WordOrientation/WordSpacings settings raise NotImplementedError. Check: "
          "self-check (non-overlap of the boxes, raises ArithmeticError) and reference (sizes proportional to "
          "weights in tests).")
def _wordcloud(data: ListOf[Expr] | Assoc | Rule, *rest: Expr | Option) -> Expr:
    fname = "WordCloud"
    pos, opts = split_args((data,) + rest, fname, 1, 2)
    if len(pos) > 1:
        raise NotImplementedError(f"{fname}: shape masks are not supported")
    o, gopts = split_options(fname, opts, {"WordOrientation": AUTO, "WordSpacings": AUTO, "ScalingFunctions": NONE,
                                           "WordSelectionFunction": AUTO, "ColorFunction": AUTO,
                                           "ColorFunctionScaling": True, "PerformanceGoal": AUTO,
                                           "MaxItems": AUTO, "FontFamily": AUTO, "PlotTheme": AUTO, "Method": AUTO})
    _unsupported(o, fname, ())
    words = [(s, w) for s, w in _word_weights(pos[0], fname) if w is not None]
    if not words:
        raise ValueError(f"{fname}: there are no words with weights")
    if any(not w > 0 for _, w in words):
        raise ValueError(f"{fname}: weights must be positive")
    lay = wordcloud_layout(words)
    for a in range(len(lay)):
        for b in range(a):
            _, _, x1, y1, w1, h1 = lay[a]
            _, _, x2, y2, w2, h2 = lay[b]
            if abs(x1 - x2) * 2 < w1 + w2 - 1e-6 and abs(y1 - y2) * 2 < h1 + h2 - 1e-6:
                raise ArithmeticError(f"{fname}: result failed verification (overlapping words)")
    content = [L(default_color(k), Expr("Text", Expr("Style", s, Expr("Rule", Symbol("FontSize"), fl(f))), pt(x, y)))
               for k, (s, f, x, y, _w, _h) in enumerate(lay)]
    xs = [x - w / 2 for _, _, x, _, w, _ in lay] + [x + w / 2 for _, _, x, _, w, _ in lay]
    ys = [y - h / 2 for _, _, _, y, _, h in lay] + [y + h / 2 for _, _, _, y, _, h in lay]
    return finish_graphics(L(*content), gopts, [("PlotRange", range_expr([(min(xs), max(xs)), (min(ys), max(ys))])),
                                                ("AspectRatio", Symbol("Automatic"))])


# ───────────────────────────── image plots ─────────────────────────────

def image_pixels(img, fname):
    """(numpy array h x w x c of floats, colour space) of an Image."""
    np = _np()
    if not (isinstance(img, Expr) and img.head == "Image"):
        if isinstance(img, (Symbol,)) or (isinstance(img, Expr) and not K.is_list(img)):
            raise Unevaluated()
        raise TypeError(f"{fname}: argument 1 must be an Image, not {K_input(img)}")
    if img.head == "Image3D":
        raise NotImplementedError(f"{fname}: Image3D is not supported")
    data = ev(Expr("ImageData", img))
    cs = ev(Expr("ImageColorSpace", img))
    try:
        A = np.array(K.to_python(data) if hasattr(K, "to_python") else _nested(data), dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{fname}: could not read the image data")
    if A.ndim == 2:
        A = A[:, :, None]
    return A, cs


def _nested(x):
    if K.is_list(x):
        return [_nested(a) for a in x.args]
    v = num(x)
    return v if v is not None else float("nan")


_IMG_DEFAULTS = {"PlotStyle": AUTO, "PlotRange": AUTO, "PlotLegends": NONE, "PlotTheme": AUTO, "Joined": AUTO,
                 "InterpolationOrder": AUTO, "Method": AUTO, "MaxPlotPoints": AUTO, "AppearanceElements": AUTO}


@fn("ImageHistogram", area=AREA, status="added",
    notes="ImageHistogram[image], [image, bspec], [image, bspec, range]: per channel (gray, or red/green/blue for RGB "
          "and \"RGBA\" without alpha), the pixel counts in n equal bins over the range, drawn as Rectangle bars "
          "{colour, Opacity[0.5], Rectangle[{b_i, 0}, {b_(i+1), count_i}], ...} (zero bins omitted); the last bin "
          "is closed so that the maximum level counts (as Wolfram's level histograms). bspec: Automatic (128 bins; "
          "2 for \"Bit\" images, as documented), n, or \"Byte\" (256 levels after rounding to bytes); range: "
          "Automatic ({0, 1}), {min, max}, {All, max}, {min, All} or All (the data range). Other colour spaces and "
          "Image3D raise NotImplementedError. Check: exact (counts compared with BinCounts of ImageData in tests).")
def _imagehistogram(img: Expr, *rest: Expr | Option) -> Expr:
    fname = "ImageHistogram"
    pos, opts = split_args((img,) + rest, fname, 1, 3)
    o, gopts = split_options(fname, opts, _IMG_DEFAULTS)
    _unsupported(o, fname, ())
    np = _np()
    A, cs = image_pixels(pos[0], fname)
    nch = A.shape[2]
    if cs not in ("Grayscale", "RGB", "Automatic", Symbol("Automatic")) and not is_sym(cs, "Automatic"):
        raise NotImplementedError(f"{fname}: the colour space {K_input(cs)} is not supported")
    if nch == 4:
        A, nch = A[:, :, :3], 3
    if nch not in (1, 3):
        raise NotImplementedError(f"{fname}: images with {nch} channels are not supported")
    itype = ev(Expr("ImageType", pos[0]))
    bspec = pos[1] if len(pos) > 1 else AUTO
    rspec = pos[2] if len(pos) > 2 else AUTO
    if is_sym(rspec, "Automatic"):
        lo, hi = 0.0, 1.0
    elif is_sym(rspec, "All"):
        lo, hi = float(np.nanmin(A)), float(np.nanmax(A))
    elif K.is_list(rspec) and len(rspec.args) == 2:
        a, b = rspec.args
        lo = float(np.nanmin(A)) if is_sym(a, "All") else float(real_number(a, fname, "the range minimum"))
        hi = float(np.nanmax(A)) if is_sym(b, "All") else float(real_number(b, fname, "the range maximum"))
    else:
        raise ValueError(f"{fname}: {K_input(rspec)} is not a valid range")
    if not hi > lo:
        raise ValueError(f"{fname}: the range needs min < max")
    vals = A
    if bspec == "Byte":
        vals = np.round(np.clip((A - lo) / (hi - lo), 0, 1) * 255) / 255 * (hi - lo) + lo
        n = 256
        edges = np.linspace(lo - 0.5 * (hi - lo) / 255, hi + 0.5 * (hi - lo) / 255, 257)
    else:
        if is_sym(bspec, "Automatic"):
            n = 2 if itype == "Bit" else 128
        elif is_sym(bspec, "All"):
            raise NotImplementedError(f"{fname}: the bin specification All is not supported")
        else:
            n = pos_int(bspec, fname, "the number of bins")
        edges = np.linspace(lo, hi, n + 1)
    colors = [Expr("GrayLevel", 0.4)] if nch == 1 else [Expr("RGBColor", 1, 0, 0), Expr("RGBColor", 0, 0.7, 0),
                                                         Expr("RGBColor", 0, 0, 1)]
    content, cmax = [], 0
    for c in range(nch):
        v = vals[:, :, c].ravel()
        v = v[np.isfinite(v)]
        v = v[(v >= edges[0]) & (v <= edges[-1])]
        idx = np.minimum(np.searchsorted(edges, v, side="right") - 1, n - 1)
        counts = np.bincount(idx, minlength=n)
        cmax = max(cmax, int(counts.max()) if counts.size else 0)
        rs = [Expr("Rectangle", pt(edges[i], 0.0), pt(edges[i + 1], float(counts[i]))) for i in range(n) if counts[i]]
        content.append(L(colors[c], Expr("Opacity", 0.5), Expr("EdgeForm"), *rs))
    return finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Frame", True),
                                                ("PlotRange", range_expr([(float(edges[0]), float(edges[-1])),
                                                                          (0.0, float(cmax or 1))]))])


KR, KG, KB = 0.2126, 0.7152, 0.0722         # Y of the sRGB/BT.709 primaries in XYZ


@fn("ImageVectorscopePlot", area=AREA, status="added",
    notes="ImageVectorscopePlot[image]: the chrominance {Cb, Cr} of every pixel as Point[{{Cb, Cr}, ...}, "
          "VertexColors -> pixel colours], with Y = 0.2126 R + 0.7152 G + 0.0722 B (the Y components of the XYZ "
          "representation of the sRGB primaries, as documented), Cb = (B - Y)/(2 (1 - 0.0722)) and Cr = (R - Y)/(2 "
          "(1 - 0.2126)); grayscale images give the origin. More than MaxPlotPoints pixels (default 40000) are "
          "subsampled on a regular stride. AspectRatio 1, PlotRange {{-0.5, 0.5}, {-0.5, 0.5}}. Other colour spaces "
          "raise NotImplementedError. Check: reference (Cb, Cr of pure colours in tests: red {-0.1146, 0.5}).")
def _imagevectorscopeplot(img: Expr, *opts: Option) -> Expr:
    fname = "ImageVectorscopePlot"
    pos, opts_ = split_args((img,) + opts, fname, 1, 1)
    o, gopts = split_options(fname, opts_, _IMG_DEFAULTS)
    _unsupported(o, fname, ("MaxPlotPoints",))
    np = _np()
    A, cs = image_pixels(pos[0], fname)
    if A.shape[2] == 1:
        A = np.repeat(A, 3, axis=2)
    if A.shape[2] == 4:
        A = A[:, :, :3]
    if A.shape[2] != 3:
        raise NotImplementedError(f"{fname}: images with {A.shape[2]} channels are not supported")
    if isinstance(cs, str) and cs not in ("RGB", "Grayscale"):
        raise NotImplementedError(f"{fname}: the colour space {cs!r} is not supported")
    mp = o["MaxPlotPoints"]
    m = 40000 if is_sym(mp, "Automatic") else pos_int(mp, fname, "MaxPlotPoints")
    px = A.reshape(-1, 3)
    if len(px) > m:
        px = px[:: int(math.ceil(len(px) / m))]
    R, Gc, B = px[:, 0], px[:, 1], px[:, 2]
    Y = KR * R + KG * Gc + KB * B
    Cb = (B - Y) / (2 * (1 - KB))
    Cr = (R - Y) / (2 * (1 - KR))
    pts_ = pts(list(zip(Cb.tolist(), Cr.tolist())))
    cols = L(*[Expr("RGBColor", *[fl(min(max(c, 0.0), 1.0)) for c in p]) for p in px.tolist()])
    content = [Expr("Point", pts_, Expr("Rule", Symbol("VertexColors"), cols))]
    return finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Axes", True),
                                                ("PlotRange", range_expr([(-0.5, 0.5), (-0.5, 0.5)]))])


@fn("PairedBarChart", area=AREA, status="added",
    notes="PairedBarChart[{y1, y2, ...}, {z1, z2, ...}]: back-to-back horizontal bars, item i at height i "
          "(bottom to top), y_i to the left (Rectangle[{-y_i, i - 0.4}, {0, i + 0.4}]) and z_i to the right "
          "(Rectangle[{0, i - 0.4}, {z_i, i + 0.4}]); negative values point the other way, zero bars are omitted. "
          "The two lists may differ in length. Grouped data {{...}, ...} on a side raises NotImplementedError. "
          "Options: ChartStyle (one style per side, or per bar with {{...}, {...}}), ChartBaseStyle, ChartLabels "
          "(labels of the items), ChartLegends, PlotRange and Graphics options. Check: reference (bar extents "
          "equal the data in tests).")
def _pairedbarchart(d1: ListOf[Real], d2: ListOf[Real], *opts: Option) -> Expr:
    fname = "PairedBarChart"
    pos, opts_ = split_args((d1, d2) + opts, fname, 2, 2)
    o, gopts = chart_options(fname, opts_, ("ChartLabels",))
    sides = []
    for d in pos:
        du, _, _ = P2._unwrap(d, fname)
        if K.is_list(du) and any(K.is_list(P2._unwrap(a, fname)[0]) for a in du.args):
            raise NotImplementedError(f"{fname}: grouped data on a side is not supported")
        sides.append(values_of(d, fname))
    styles = chart_styles(o, 2, fname)
    content, hmax = [], 0.0
    for side, vs, st in ((-1, sides[0], styles[0]), (1, sides[1], styles[1])):
        rs = []
        for i, v in enumerate(vs):
            v = float(v)
            if v == 0:
                continue
            hmax = max(hmax, abs(v))
            x0, x1 = sorted((0.0, side * v))
            rs.append(Expr("Rectangle", pt(x0, i + 1 - 0.4), pt(x1, i + 1 + 0.4)))
        if st is not None and rs:
            content.append(L(st, *rs))
    n = max(len(sides[0]), len(sides[1]), 1)
    r = _range_2d(o, fname, (-(hmax or 1.0), hmax or 1.0), (0.5, n + 0.5))
    lab = o["ChartLabels"]
    ticks = AUTO
    if K.is_list(lab):
        ticks = L(*[L(k + 1, lab.args[k]) for k in range(min(n, len(lab.args)))])
    elif not is_sym(lab, "None"):
        raise NotImplementedError(f"{fname}: ChartLabels -> {K_input(lab)} is not supported; give a list of labels")
    g = finish_graphics(L(*content), gopts, [("AspectRatio", GOLDEN), ("Axes", True), ("Ticks", L(AUTO, ticks)),
                                             ("PlotRange", range_expr(r))])
    return _legend(g, o, styles, fname)


# Every symbol of the batch is implemented (some partially; see the notes of each head).
SKIPPED: dict = {}
