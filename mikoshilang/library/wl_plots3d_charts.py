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
