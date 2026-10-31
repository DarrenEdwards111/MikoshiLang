"""Wolfram batch W16: 2-D function, list, log, date and statistical plots (plus the 3-D list/function plots that
the batch plan assigns here).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Plots evaluate to inert ``Graphics`` / ``Graphics3D`` objects (W14 graphics core), built from the primitives
``Line``, ``Point``, ``Polygon``, ``Rectangle``, ``Raster``, ``Text`` and ``GraphicsComplex``, with the options
validated by W15 (``wl_plot_options``). The *data* of a plot (the sampled points, the polygons of a region, the cell
colours of an array plot) is what the tests compare, not pixels.

Output layout (stable, documented; Wolfram's own output wraps the same data in ``Annotation``/``GraphicsComplex``
forms that differ between versions):

* 2-D plots: ``Graphics[{fill_1, ..., {style_1, prims_1}, {style_2, prims_2}, ...}, opts]``; ``opts`` are the
  user's ``Graphics`` options first, then ``AspectRatio``, ``Axes``, ``PlotRange -> {{xmin, xmax}, {ymin, ymax}}``
  (always explicit numbers), ``PlotRangeClipping`` and ``PlotRangePadding``.
* ``PlotLegends`` gives ``Legended[graphics, Placed[LineLegend[styles, labels], After]]`` (``PointLegend`` for
  point plots).
* Log scales store the logarithm (natural log, as Wolfram) of the coordinate and give tick labels in the original
  scale (``Ticks``/``FrameTicks``).

Sampling of functions: ``PlotPoints`` initial equally spaced samples (default 50), then ``MaxRecursion`` rounds of
adaptive bisection (default 6) where the curve is not straight to within 1/1000 of the plotted range. A jump that
survives 40 further bisections is a discontinuity (``Exclusions -> Automatic``) and splits the curve; non-real or
undefined values split it too. Expressions are compiled with SymPy/NumPy and cross-checked against the kernel's
own ``N`` at sample points; heads that cannot be compiled are evaluated point by point by the kernel.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, S
from .wl_graphics_color import real_value, parse_color, K_input
from . import wl_graphics_core as G
from .wl_graphics_core import is_sym, is_rule, opt_name, split_trailing_options
from . import wl_plot_options as PO

AREA = "plotting: 2-D and list plots (W16)"

# ColorData[97] (Wolfram's "DefaultPlotColors"), first ten entries.
DEFAULT_COLORS = ((0.368417, 0.506779, 0.709798), (0.880722, 0.611041, 0.142051), (0.560181, 0.691569, 0.194885),
                  (0.922526, 0.385626, 0.209179), (0.528488, 0.470624, 0.701351), (0.772079, 0.431554, 0.102387),
                  (0.363898, 0.618501, 0.782349), (1.0, 0.75, 0.0), (0.647624, 0.37816, 0.614037),
                  (0.571589, 0.586483, 0.0))
POINT_SIZE = 0.012833333333333334     # Wolfram's default PointSize in ListPlot
LINE_THICKNESS = 2                    # AbsoluteThickness of default curves (Wolfram 13+)
GOLDEN = Expr("Power", Symbol("GoldenRatio"), -1)


def _np():
    import numpy as np
    return np


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x)


def rgb(c):
    return Expr("RGBColor", *[float(v) for v in c])


def default_color(i):
    return rgb(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])


def fl(x):
    """Plain float for output coordinates (ints stay floats so the output is uniformly inexact)."""
    return float(x)


def pt(*cs):
    return L(*[fl(c) for c in cs])


def pts(seq):
    return L(*[pt(*p) for p in seq])


# ───────────────────────────── numbers and arguments ─────────────────────────────

def num(x):
    """Float of a real numeric value, else None (never raises)."""
    if isinstance(x, bool) or isinstance(x, str):
        return None
    if isinstance(x, int):
        return float(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, complex):
        return None
    try:
        v = real_value(x, "", "x")
    except (Unevaluated, TypeError, ValueError, ArithmeticError):
        return None
    return v if math.isfinite(v) else None


def real_arg(x, fname, what):
    """A finite real (float); TypeError for strings/lists/booleans, Unevaluated for symbols, ValueError for complex
    or infinite values."""
    if isinstance(x, (bool, str)) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    if isinstance(x, complex):
        raise ValueError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    v = real_value(x, fname, what)
    if not math.isfinite(v):
        raise ValueError(f"{fname}: {what} must be finite")
    return v


def is_missing(x):
    return is_sym(x, "None", "Null", "Indeterminate") or (isinstance(x, Expr) and x.head == "Missing")


def is_symbolic(x):
    """A user symbol or a non-numeric compound expression (stays unevaluated as plot data)."""
    if isinstance(x, Symbol):
        return not is_missing(x) and x.name not in ("True", "False")
    return (isinstance(x, Expr) and not K.is_list(x) and x.head not in
            ("Rule", "RuleDelayed", "Association", "Missing", "Complex", "DirectedInfinity") and num(x) is None)


def var_symbol(x, fname, what="variable"):
    if isinstance(x, Symbol) and not x.name.startswith("$"):
        return x
    raise ValueError(f"{fname}: {K_input(x)} is not a valid {what}")


def iterator(spec, fname, pos=2):
    """{x, xmin, xmax} -> (x, a, b) with a < b (floats); held input."""
    if isinstance(spec, Expr) and spec.head == "Element":
        raise NotImplementedError(f"{fname}: the region form x \u2208 reg is not supported; use {{x, xmin, xmax}}")
    if not K.is_list(spec):
        raise ValueError(f"{fname}: argument {pos} must be an iterator {{x, xmin, xmax}}, not {K_input(spec)}")
    if len(spec.args) != 3:
        raise ValueError(f"{fname}: argument {pos} must be an iterator {{x, xmin, xmax}}, not {K_input(spec)}")
    v = var_symbol(spec.args[0], fname)
    a = real_arg(ev(spec.args[1]), fname, "the iterator minimum")
    b = real_arg(ev(spec.args[2]), fname, "the iterator maximum")
    if not a < b:
        raise ValueError(f"{fname}: the iterator range must have xmin < xmax, not {K_input(spec)}")
    return v, a, b


def hold_args(args):
    """Evaluate held trailing options; return (positional held args, evaluated option rules)."""
    args = list(args)
    opts = []
    while args:
        a = args[-1]
        if isinstance(a, Expr) and a.head == "Evaluate" and len(a.args) == 1:
            a = ev(a.args[0])
        if is_rule(a) and isinstance(a.args[0], (Symbol, str)):
            opts.insert(0, Expr(a.head, a.args[0], ev(a.args[1])))
            args.pop()
        elif K.is_list(a) and a.args and all(is_rule(r) for r in a.args) and len(args) > 2:
            opts[0:0] = [Expr(r.head, r.args[0], ev(r.args[1])) for r in a.args]
            args.pop()
        else:
            break
    return args, opts


# ───────────────────────────── options ─────────────────────────────

AUTO = Symbol("Automatic")
NONE = Symbol("None")


def _graphics_option_names(dim):
    return set(G.GRAPHICS_OPTIONS[dim])


def split_options(fname, opts, defaults, dim=2):
    """Split option rules into (plot options {name: value} with defaults filled, graphics options [rules]).

    * names in ``defaults`` are this plot's own options (validated with W15's validators);
    * container options of Graphics/Graphics3D (AspectRatio, Axes, Frame, PlotLabel, ...) are passed to the result;
    * other known option names (any W15 validator) set to a non-trivial value raise NotImplementedError, and the
      trivial settings Automatic/None/Inherited are accepted (they mean the default);
    * unknown names raise ValueError."""
    mine = {}
    gopts = []
    gnames = _graphics_option_names(dim)
    seen = set()
    for r in opts:
        n = opt_name(r)
        if n is None:
            raise ValueError(f"{fname}: {K_input(r)} is not an option rule Name -> value")
        if n in seen:
            continue
        seen.add(n)
        val = r.args[1]
        if n in defaults:
            v = PO.OPTION_VALIDATORS.get(n)
            if v is not None and r.head == "Rule":
                v(val, fname)
            mine[n] = val
        elif n in gnames:
            gopts.append(Expr("Rule", Symbol(n), val))
        elif n in PO.OPTION_VALIDATORS or n in _EXTRA_KNOWN:
            if is_sym(val, "Automatic", "None", "Inherited") or (n in _BOOL_DEFAULT_FALSE and val is False):
                continue
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(val)} is not supported")
        else:
            raise ValueError(f"{fname}: {n} is not a known option of {fname}")
    out = dict(defaults)
    out.update(mine)
    if dim == 2:
        gopts = [g for g in gopts if opt_name(g) != "PlotRange"]
    else:
        gopts = [g for g in gopts if opt_name(g) != "PlotRange"]
    return out, gopts


_EXTRA_KNOWN = {"PlotTheme", "PerformanceGoal", "WorkingPrecision", "Method", "EvaluationMonitor",
                "PlotHighlighting", "PlotInteractivity", "TargetUnits", "IntervalMarkers", "IntervalMarkersStyle",
                "LabelingFunction", "LabelingSize", "PlotFit", "PlotFitElements", "PlotLayout", "PlotLabels",
                "RegionFunction", "ClippingStyle", "ColorFunction", "ColorFunctionScaling", "Mesh", "MeshFunctions",
                "MeshShading", "MeshStyle", "ScalingFunctions", "PlotMarkers", "Exclusions", "ExclusionsStyle",
                "PlotRangeClipping", "Filling", "FillingStyle", "Joined", "InterpolationOrder", "DataRange",
                "MaxRecursion", "PlotPoints", "PlotStyle", "PlotLegends", "BoundaryStyle", "ContourLabels",
                "ContourShading", "ContourStyle", "Contours", "LightingAngle", "ColorRules", "ReferenceLineStyle",
                "Frame", "FrameTicks", "DateFunction", "DateTicksFormat", "TargetUnits", "ImageMargins",
                "TextureCoordinateFunction", "TextureCoordinateScaling", "Lighting", "PlotRangePadding",
                "NormalsFunction", "VertexColors", "BoxRatios", "ViewPoint", "Boxed", "AxesEdge",
                "MaxPlotPoints", "ExtentSize", "ExtentMarkers", "ExtentElementFunction", "Sampling",
                "DisplayFunction", "SphericalRegion", "Ticks", "AxesLabel", "Axes", "OpacityFunction",
                "OpacityFunctionScaling"}
_BOOL_DEFAULT_FALSE = {"Joined", "Frame"}


def plot_range_option(gopts, mine):
    return mine.get("PlotRange", AUTO)


def finish_graphics(content, gopts, defaults, dim=2):
    """Graphics[content, user graphics options first, then our defaults for names the user did not set]."""
    have = {opt_name(r) for r in gopts}
    rules = list(gopts)
    for name, val in defaults:
        if name not in have:
            rules.append(Expr("Rule", Symbol(name), val))
    head = "Graphics" if dim == 2 else "Graphics3D"
    return Expr(head, content, *rules)


# ───────────────────────────── styles ─────────────────────────────

def style_list(spec, n, fname, kind="line"):
    """Directive per curve/dataset: Wolfram's default colours and thickness/point size, overridden by PlotStyle."""
    base = []
    for i in range(n):
        if kind == "point":
            base.append([default_color(i), Expr("PointSize", POINT_SIZE), Expr("AbsoluteThickness", LINE_THICKNESS)])
        else:
            base.append([default_color(i), Expr("AbsoluteThickness", LINE_THICKNESS)])
    if spec is None or is_sym(spec, "Automatic"):
        user = [None] * n
    elif K.is_list(spec):
        items = list(spec.args)
        if not items:
            user = [None] * n
        else:
            user = [items[i % len(items)] for i in range(n)]
    else:
        user = [spec] * n
    out = []
    for b, u in zip(base, user):
        if u is None or is_sym(u, "Automatic"):
            out.append(Expr("Directive", *b))
        elif is_sym(u, "None"):
            out.append(None)
        else:
            G.directive_like(u, fname, "PlotStyle")
            extra = list(u.args) if isinstance(u, Expr) and u.head == "Directive" else [u]
            out.append(Expr("Directive", *b, *extra))
    return out


def style_color(d):
    """Last colour in a directive (for fillings), default black."""
    col = None
    if d is not None:
        for a in d.args:
            if parse_color(a) is not None:
                col = a
    return col if col is not None else Expr("GrayLevel", 0)


# ───────────────────────────── function compilation ─────────────────────────────

def _real_or_nan(v):
    if isinstance(v, bool):
        return math.nan
    if isinstance(v, (int, float)):
        f = float(v)
        return f if math.isfinite(f) else math.nan
    if isinstance(v, complex):
        return v.real if abs(v.imag) <= 1e-12 * max(1.0, abs(v.real)) else math.nan
    x = num(v)
    return math.nan if x is None else x


class Compiled:
    """A plotted expression in variables ``vars``: ``__call__(*arrays) -> float array`` (NaN where the value is not
    real or not defined). ``slow`` is True when the kernel evaluates point by point."""

    def __init__(self, expr, vars_, fname):
        self.fname = fname
        self.names = [v.name for v in vars_]
        self.dummies = [Symbol(f"plot$w16${i}") for i in range(len(vars_))]
        held = K.substitute(expr, {v.name: d for v, d in zip(vars_, self.dummies)})
        self.held = held
        try:
            self.sym = ev(held)
        except (TypeError, ValueError, NotImplementedError, ArithmeticError, ZeroDivisionError):
            self.sym = None          # evaluate per point
        self.fast = None
        if self.sym is not None:
            self._build_fast()

    def _build_fast(self):
        np = _np()
        try:
            e = S(self.sym)
        except Exception:
            return
        if not isinstance(e, sp.Basic):
            return
        dsyms = [sp.Symbol(d.name) for d in self.dummies]
        if e.free_symbols - set(dsyms):
            return
        if any(isinstance(a, sp.core.function.AppliedUndef) for a in sp.preorder_traversal(e)):
            return
        try:
            lam = sp.lambdify(dsyms, e, modules=["numpy", "scipy"] if _have_scipy() else ["numpy"])
        except Exception:
            return

        def f(*arrs):
            with np.errstate(all="ignore"):
                try:
                    out = lam(*[np.asarray(a, dtype=complex) for a in arrs])
                    out = np.broadcast_to(np.asarray(out, dtype=complex), np.shape(arrs[0]))
                    re, im = out.real.astype(float), out.imag
                    bad = ~np.isfinite(re) | ~np.isfinite(im) | (np.abs(im) > 1e-12 * np.maximum(1.0, np.abs(re)))
                except (TypeError, ValueError, AttributeError):
                    out = lam(*[np.asarray(a, dtype=float) for a in arrs])
                    re = np.array(np.broadcast_to(np.asarray(out, dtype=float), np.shape(arrs[0])), dtype=float)
                    bad = ~np.isfinite(re)
            re = np.array(re, dtype=float)
            re[bad] = np.nan
            return re
        try:
            probe = [np.array([0.3713, 1.1937, -0.7071]) for _ in dsyms]
            f(*probe)
        except Exception:
            return
        self.fast = f

    def point(self, *vals):
        """Kernel evaluation N[expr /. vars -> vals] (the reference used for the cross-check)."""
        b = {d.name: float(v) for d, v in zip(self.dummies, vals)}
        e = K.substitute(self.held if self.sym is None else self.sym, b)
        try:
            r = ev(Expr("N", e))
        except (TypeError, ValueError, NotImplementedError, ArithmeticError, ZeroDivisionError, OverflowError):
            return math.nan
        if isinstance(r, (Symbol, Expr)) and not isinstance(r, bool) and num(r) is None:
            if isinstance(r, Expr) and r.head in ("Complex", "DirectedInfinity") or is_sym(r, "Indeterminate",
                                                                                          "ComplexInfinity"):
                return math.nan
            if _free_of_dummies(r, self.dummies):
                raise Unevaluated()          # a symbolic parameter: Plot[a x, {x, 0, 1}] stays unevaluated
            return math.nan
        return _real_or_nan(r)

    def __call__(self, *arrs):
        np = _np()
        arrs = [np.asarray(a, dtype=float) for a in arrs]
        if self.fast is not None:
            return self.fast(*arrs)
        flat = [a.ravel() for a in arrs]
        out = np.array([self.point(*vals) for vals in zip(*flat)], dtype=float)
        return out.reshape(arrs[0].shape)

    def cross_check(self, samples):
        """Compare the compiled function with the kernel at a few sample points (self-check). On a mismatch the
        compiled code is discarded and the kernel evaluates every point (never a silently different curve)."""
        if self.fast is None:
            return
        np = _np()
        for vals in samples:
            a = self.fast(*[np.array([v]) for v in vals])[0]
            b = self.point(*vals)
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isnan(a) != math.isnan(b) or abs(a - b) > 1e-7 * max(1.0, abs(b)):
                self.fast = None
                return


def _free_of_dummies(r, dummies):
    names = {d.name for d in dummies}

    def walk(x):
        if isinstance(x, Symbol):
            return x.name not in names
        if isinstance(x, Expr):
            return all(walk(a) for a in x.args) and (isinstance(x.head, str) or walk(x.head))
        return True
    return walk(r)


def _have_scipy():
    try:
        import scipy.special  # noqa: F401
        return True
    except ImportError:
        return False


def symbolic_check(c, vars_, fname):
    """Unevaluated for a plotted expression with free parameters (Plot[a x, {x, 0, 1}]): the kernel cannot give it
    numeric values at any sample point."""
    if c.sym is None:
        return
    try:
        e = S(c.sym)
    except Exception:
        return
    if isinstance(e, sp.Basic):
        names = {d.name for d in c.dummies}
        extra = {s for s in e.free_symbols if s.name not in names}
        if extra:
            raise Unevaluated()


# ───────────────────────────── 1-D adaptive sampling ─────────────────────────────

def _scale_of(ys):
    np = _np()
    f = ys[np.isfinite(ys)]
    if f.size == 0:
        return 1.0
    s = float(np.max(f) - np.min(f))
    return s if s > 0 else max(1.0, abs(float(f[0])))


def _robust_scale(xs, ys):
    lo, hi = weighted_range(xs, ys)
    s = hi - lo
    return s if s > 0 else max(1.0, abs(lo))


def weighted_range(xs, ys, qlo=0.05, qhi=0.95, k=2.0):
    """Automatic range of sampled values: the [5%, 95%] quantiles weighted by x spacing, widened by twice their
    spread and limited to the data (clips the spikes next to poles, keeps every bounded curve whole)."""
    np = _np()
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    ok = np.isfinite(ys)
    if not ok.any():
        return 0.0, 1.0
    if xs.size < 3:
        return float(np.min(ys[ok])), float(np.max(ys[ok]))
    w = np.zeros_like(xs)
    dx = np.diff(xs)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    w = np.where(ok, np.abs(w), 0.0)
    y = ys[ok]
    w = w[ok]
    if w.sum() <= 0:
        return float(y.min()), float(y.max())
    order = np.argsort(y)
    y, w = y[order], w[order]
    cw = np.cumsum(w) / w.sum()
    a = float(y[min(np.searchsorted(cw, qlo), y.size - 1)])
    b = float(y[min(np.searchsorted(cw, qhi), y.size - 1)])
    sp_ = b - a
    lo = max(float(y[0]), a - k * sp_)
    hi = min(float(y[-1]), b + k * sp_)
    if sp_ == 0:
        lo, hi = float(y[0]), float(y[-1])
    return lo, hi


def sample_1d(f, a, b, n=50, maxrec=6, jumps=True, exclusions=(), tol=1e-3):
    """Adaptive samples of f on [a, b] -> list of polylines [[(x, y), ...], ...] (split at undefined values,
    detected jumps and explicit exclusions)."""
    np = _np()
    n = max(int(n), 2)
    xs = np.linspace(a, b, n)
    ys = f(xs)
    for _ in range(int(maxrec)):
        if xs.size > 20000:
            break
        mids = (xs[:-1] + xs[1:]) / 2
        ym = f(mids)
        y0, y1 = ys[:-1], ys[1:]
        fin0, fin1, finm = np.isfinite(y0), np.isfinite(y1), np.isfinite(ym)
        scale = _robust_scale(xs, ys)
        with np.errstate(all="ignore"):
            bend = fin0 & fin1 & finm & (np.abs(ym - (y0 + y1) / 2) > tol * scale)
        mixed = (fin0 != fin1) | (fin0 != finm)
        need = bend | mixed
        if not need.any():
            break
        xs = np.concatenate([xs, mids[need]])
        ys = np.concatenate([ys, ym[need]])
        o = np.argsort(xs, kind="stable")
        xs, ys = xs[o], ys[o]
    breaks = set()
    extra_x, extra_y = [], []
    # boundaries of undefined regions: bisect to the last defined point
    fin = np.isfinite(ys)
    for i in np.nonzero(fin[:-1] != fin[1:])[0][:400]:
        lo, hi = float(xs[i]), float(xs[i + 1])
        left_ok = bool(fin[i])
        for _ in range(30):
            m = (lo + hi) / 2
            fm = f(np.array([m]))[0]
            if np.isfinite(fm) == left_ok:
                lo = m
            else:
                hi = m
        x_ok = lo if left_ok else hi
        y_ok = f(np.array([x_ok]))[0]
        if np.isfinite(y_ok):
            extra_x.append(x_ok)
            extra_y.append(y_ok)
    if jumps:
        scale = _robust_scale(xs, ys)
        dy = np.abs(np.diff(ys))
        cand = np.nonzero(np.isfinite(dy) & (dy > 0.02 * scale))[0]
        if cand.size > 300:
            cand = cand[np.argsort(-dy[cand])[:300]]
        for i in cand:
            lo, hi = float(xs[i]), float(xs[i + 1])
            flo, fhi = float(ys[i]), float(ys[i + 1])
            d0 = abs(fhi - flo)
            keep = None
            for step in range(40):
                m = (lo + hi) / 2
                fm = float(f(np.array([m]))[0])
                if not math.isfinite(fm):
                    break
                if abs(fm - flo) >= abs(fhi - fm):
                    hi, fhi = m, fm
                else:
                    lo, flo = m, fm
                if step == 11:
                    keep = (lo, flo, hi, fhi)
                if abs(fhi - flo) < 0.25 * d0:
                    break
            else:
                breaks.add((keep[0], keep[2]))
                extra_x += [keep[0], keep[2]]
                extra_y += [keep[1], keep[3]]
    if extra_x:
        xs = np.concatenate([xs, extra_x])
        ys = np.concatenate([ys, extra_y])
        o = np.argsort(xs, kind="stable")
        xs, ys = xs[o], ys[o]
        keep = np.concatenate([[True], np.diff(xs) > 0])
        xs, ys = xs[keep], ys[keep]
    # explicit exclusions
    excl = sorted(e for e in exclusions if a < e < b)
    lines, cur = [], []
    bl = sorted(breaks)
    for j in range(xs.size):
        x, y = float(xs[j]), float(ys[j])
        if cur and j > 0:
            px = float(xs[j - 1])
            if any(px <= p0 and x >= p1 for p0, p1 in bl) or any(px < e < x for e in excl):
                lines.append(cur)
                cur = []
        if any(abs(x - e) <= 1e-12 * max(1.0, abs(e)) for e in excl) or not math.isfinite(y):
            if cur:
                lines.append(cur)
            cur = []
            continue
        cur.append((x, y))
    if cur:
        lines.append(cur)
    return [ln for ln in lines if ln]


# ───────────────────────────── plot ranges ─────────────────────────────

def _pair(v, fname):
    if K.is_list(v) and len(v.args) == 2:
        return v.args
    raise ValueError(f"{fname}: PlotRange {K_input(v)} is not a valid range")


def _one_range(spec, data, auto, full, fname):
    """Resolve one coordinate's PlotRange setting -> (lo, hi) floats."""
    if spec is None or is_sym(spec, "Automatic"):
        r = auto
    elif is_sym(spec, "All"):
        r = data
    elif is_sym(spec, "Full"):
        r = full if full is not None else data
    elif K.is_list(spec):
        lo_s, hi_s = _pair(spec, fname)
        base = auto
        lo = base[0] if is_sym(lo_s, "Automatic", "All", "Full") else real_arg(lo_s, fname, "PlotRange minimum")
        hi = base[1] if is_sym(hi_s, "Automatic", "All", "Full") else real_arg(hi_s, fname, "PlotRange maximum")
        r = (lo, hi)
    else:
        s = real_arg(spec, fname, "PlotRange")
        r = (-abs(s), abs(s))
    lo, hi = float(r[0]), float(r[1])
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        d = 1.0 if lo == 0 else abs(lo) * 0.1
        lo, hi = lo - d, hi + d
    return lo, hi


def resolve_plot_range(spec, fname, data, auto=None, full=None, dims=2):
    """PlotRange setting -> explicit {{xmin, xmax}, {ymin, ymax}(, {zmin, zmax})}. ``data``/``auto``/``full`` are
    per-coordinate (lo, hi) tuples (``None`` entries fall back to ``data``)."""
    auto = auto or [None] * dims
    full = full or [None] * dims
    auto = [a if a is not None else d for a, d in zip(auto, data)]
    if spec is None or is_sym(spec, "Automatic", "All", "Full") or not K.is_list(spec):
        if spec is not None and not is_sym(spec, "Automatic", "All", "Full"):
            # a single number or {min, max}: the last (dependent) coordinate
            per = [None] * (dims - 1) + [spec]
        else:
            per = [spec] * dims
    elif len(spec.args) == dims and all(K.is_list(a) or is_sym(a, "Automatic", "All", "Full") for a in spec.args):
        per = list(spec.args)
    elif len(spec.args) == 2:
        per = [None] * (dims - 1) + [spec]
    else:
        raise ValueError(f"{fname}: PlotRange {K_input(spec)} is not a valid range")
    out = []
    for i in range(dims):
        out.append(_one_range(per[i], data[i], auto[i], full[i], fname))
    return out


def range_expr(r):
    return L(*[L(fl(lo), fl(hi)) for lo, hi in r])


def data_range(seq):
    vals = [v for v in seq if v is not None and math.isfinite(v)]
    if not vals:
        return (0.0, 1.0)
    return (min(vals), max(vals))


# ───────────────────────────── legends ─────────────────────────────

def legend_wrap(g, spec, styles, auto_labels, fname, kind="line"):
    """PlotLegends -> Legended[g, Placed[LineLegend[styles, labels], After]]."""
    if spec is None or is_sym(spec, "None"):
        return g
    head = "PointLegend" if kind == "point" else ("SwatchLegend" if kind == "swatch" else "LineLegend")
    styles = [s if s is not None else Expr("GrayLevel", 0) for s in styles]
    pos = Symbol("After")
    if isinstance(spec, Expr) and spec.head == "Placed" and spec.args:
        pos = spec.args[1] if len(spec.args) > 1 else pos
        spec = spec.args[0]
    if isinstance(spec, Expr) and spec.head in ("LineLegend", "PointLegend", "SwatchLegend", "BarLegend"):
        if spec.args and K.is_list(spec.args[0]) or spec.head == "BarLegend":
            leg = spec
        else:
            leg = Expr(spec.head, L(*styles), *spec.args)
        return Expr("Legended", g, Expr("Placed", leg, pos))
    if is_sym(spec, "Automatic") or spec == "Expressions":
        labels = auto_labels
    elif K.is_list(spec):
        labels = list(spec.args)
    elif isinstance(spec, str) and spec != "Expressions":
        labels = [spec] + [""] * (len(styles) - 1) if len(styles) > 1 else [spec]
    else:
        labels = [spec]
    if labels is None:
        raise NotImplementedError(f"{fname}: PlotLegends -> Automatic needs labelled data here; give explicit labels")
    n = min(len(labels), len(styles))
    return Expr("Legended", g, Expr("Placed", Expr(head, L(*styles[:n]), L(*labels[:n])), pos))


# ───────────────────────────── filling ─────────────────────────────

def _fill_level(t, prange, fname):
    if is_sym(t, "Axis"):
        return 0.0
    if is_sym(t, "Bottom"):
        return prange[1][0]
    if is_sym(t, "Top"):
        return prange[1][1]
    v = num(t)
    if v is not None:
        return v
    raise NotImplementedError(f"{fname}: the filling target {K_input(t)} is not supported")


def fill_polys(lines, level):
    out = []
    for ln in lines:
        if len(ln) < 2:
            continue
        poly = [(x, y) for x, y in ln] + [(ln[-1][0], level), (ln[0][0], level)]
        out.append(Expr("Polygon", pts(poly)))
    return out


def fill_between(lines_a, lines_b, fname, evaluator=None):
    """Polygons between two curves sampled on the same x values (or re-sampled through ``evaluator``)."""
    np = _np()
    out = []
    xa = sorted({p[0] for ln in lines_a for p in ln})
    for ln in lines_a:
        if len(ln) < 2:
            continue
        xs = np.array([p[0] for p in ln])
        if evaluator is not None:
            yb = evaluator(xs)
        else:
            bmap = {p[0]: p[1] for l2 in lines_b for p in l2}
            if not all(x in bmap for x in xs):
                raise NotImplementedError(f"{fname}: filling between datasets with different x values is not "
                                          "supported")
            yb = np.array([bmap[x] for x in xs])
        seg = []
        for (x, y), y2 in zip(ln, yb):
            if math.isfinite(y2):
                seg.append((x, y, float(y2)))
            elif seg:
                out.append(seg)
                seg = []
        if seg:
            out.append(seg)
    del xa
    return [Expr("Polygon", pts([(x, y) for x, y, _ in s] + [(x, y2) for x, _, y2 in reversed(s)]))
            for s in out if len(s) >= 2]


def filling_content(spec, fstyle, curves, styles, prange, fname, evaluators=None):
    """Filling option -> list of {style, Polygon...} groups; ``curves`` is a list of polylines per curve."""
    if spec is None or is_sym(spec, "None"):
        return []
    n = len(curves)
    pairs = []
    if K.is_list(spec) and all(is_rule(r) for r in spec.args):
        for r in spec.args:
            i = r.args[0]
            if not (isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n):
                raise ValueError(f"{fname}: Filling {K_input(r)} refers to no curve")
            tgt, sty = r.args[1], None
            if K.is_list(tgt) and len(tgt.args) == 2 and K.is_list(tgt.args[0]):
                tgt, sty = tgt.args
            pairs.append((i - 1, tgt, sty))
    else:
        pairs = [(i, spec, None) for i in range(n)]
    out = []
    for i, tgt, sty in pairs:
        col = style_color(styles[i])
        st = sty if sty is not None else (fstyle if fstyle is not None and not is_sym(fstyle, "Automatic")
                                          else Expr("Directive", Expr("Opacity", 0.2), col))
        if K.is_list(st):
            raise NotImplementedError(f"{fname}: separate above/below filling styles are not supported")
        if K.is_list(tgt) and len(tgt.args) == 1:
            j = tgt.args[0]
            if not (isinstance(j, int) and not isinstance(j, bool) and 1 <= j <= n):
                raise ValueError(f"{fname}: Filling target {K_input(tgt)} refers to no curve")
            ev_j = evaluators[j - 1] if evaluators else None
            polys = fill_between(curves[i], curves[j - 1], fname, ev_j)
        else:
            polys = fill_polys(curves[i], _fill_level(tgt, prange, fname))
        if polys:
            out.append(L(st, *polys))
    return out


# ───────────────────────────── Plot family ─────────────────────────────

_PLOT_DEFAULTS = {"PlotRange": L(Symbol("Full"), AUTO), "PlotStyle": AUTO, "PlotPoints": AUTO, "MaxRecursion": AUTO,
                  "Exclusions": AUTO, "ExclusionsStyle": NONE, "Filling": NONE, "FillingStyle": AUTO,
                  "PlotLegends": NONE, "WorkingPrecision": Symbol("MachinePrecision"), "ClippingStyle": NONE,
                  "PerformanceGoal": AUTO, "PlotTheme": AUTO, "Mesh": NONE, "ColorFunction": AUTO,
                  "ScalingFunctions": NONE, "PlotHighlighting": AUTO, "Method": AUTO, "EvaluationMonitor": NONE,
                  "RegionFunction": AUTO}


def _int_opt(v, default, fname, name, lo):
    if is_sym(v, "Automatic"):
        return default
    if isinstance(v, int) and not isinstance(v, bool) and v >= lo:
        return v
    if K.is_list(v):
        raise NotImplementedError(f"{fname}: {name} -> {K_input(v)} is not supported for this plot")
    raise ValueError(f"{fname}: {name} must be an integer >= {lo} or Automatic, not {K_input(v)}")


def _check_unsupported(o, fname, names):
    for n in names:
        v = o.get(n)
        if v is None:
            continue
        if n in ("Mesh", "ExclusionsStyle", "ClippingStyle", "EvaluationMonitor") and is_sym(v, "None", "Automatic"):
            continue
        if n in ("ColorFunction", "PlotHighlighting", "Method", "RegionFunction", "PerformanceGoal") and \
                is_sym(v, "Automatic", "None"):
            continue
        if n == "PerformanceGoal" and v in ("Quality", "Speed"):
            continue
        if n == "PlotTheme" and (is_sym(v, "Automatic") or v == "Default"):
            continue
        if n == "WorkingPrecision" and is_sym(v, "MachinePrecision"):
            continue
        if n == "ScalingFunctions" and is_sym(v, "None", "Automatic"):
            continue
        if n == "RegionFunction" and isinstance(v, Expr) and v.head == "Function" and len(v.args) == 1 and \
                v.args[0] is True:
            continue
        raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


def _exclusion_points(spec, var, a, b, fname):
    if spec is None or is_sym(spec, "Automatic", "None"):
        return []
    items = spec.args if K.is_list(spec) else (spec,)
    out = []
    for e in items:
        if isinstance(e, Expr) and e.head == "Equal" and len(e.args) == 2:
            lhs, rhs = e.args
            v = num(rhs) if isinstance(lhs, Symbol) and lhs.name == var.name else None
            if v is None:
                try:
                    sv = sp.Symbol(var.name)
                    eq = S(Expr("Plus", lhs, Expr("Times", -1, rhs)))
                    sols = sp.solveset(eq.subs(sp.Symbol(var.name), sv), sv, sp.Interval(a, b))
                except Exception:
                    sols = None
                if isinstance(sols, sp.FiniteSet) and all(s.is_real for s in sols):
                    out += [float(s) for s in sols]
                    continue
                raise NotImplementedError(f"{fname}: the exclusion {K_input(e)} is not supported (give x == c)")
            out.append(v)
        else:
            v = num(e)
            if v is None:
                raise NotImplementedError(f"{fname}: the exclusion {K_input(e)} is not supported")
            out.append(v)
    return out


def _curves_of(f, var):
    """Held f -> list of held expressions (an explicit list, or an expression evaluating to a list)."""
    if isinstance(f, Expr) and f.head == "Evaluate" and len(f.args) == 1:
        f = ev(f.args[0])
    if K.is_list(f):
        return list(f.args), True
    return [f], False


def _ticks_for_log(lo, hi, base=10.0):
    """Tick marks {Log[v], v} for powers of ten (and 2, 5 multiples when the range is short) inside exp([lo, hi])."""
    out = []
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return Symbol("Automatic")
    k0, k1 = math.floor(lo / math.log(10)), math.ceil(hi / math.log(10))
    mults = (1, 2, 5) if k1 - k0 <= 3 else (1,)
    for k in range(int(k0), int(k1) + 1):
        for m in mults:
            v = m * 10.0 ** k
            lv = math.log(v)
            if lo - 1e-12 <= lv <= hi + 1e-12:
                lab = (int(v) if v >= 1 and v == int(v) else v)
                out.append(L(fl(lv), lab))
    return L(*out) if out else Symbol("Automatic")


def function_plot(fname, args, logx=False, logy=False):
    pos, opts = hold_args(args)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected {fname}[f, {{x, xmin, xmax}}, options], got {len(pos)} positional "
                         "argument(s)")
    f, spec = pos
    var, a, b = iterator(spec, fname)
    if logx and a <= 0:
        raise ValueError(f"{fname}: the x range must be positive for a logarithmic x axis")
    o, gopts = split_options(fname, opts, _PLOT_DEFAULTS)
    _check_unsupported(o, fname, ("ColorFunction", "Mesh", "ClippingStyle", "ExclusionsStyle", "WorkingPrecision",
                                  "PlotTheme", "PlotHighlighting", "Method", "EvaluationMonitor", "RegionFunction",
                                  "PerformanceGoal"))
    sf = o["ScalingFunctions"]
    if not is_sym(sf, "None", "Automatic"):
        if logx or logy:
            raise NotImplementedError(f"{fname}: ScalingFunctions is not supported in {fname}")
        if sf == "Log" or (K.is_list(sf) and len(sf.args) == 2 and is_sym(sf.args[0], "None") and sf.args[1] == "Log"):
            logy = True
        elif K.is_list(sf) and len(sf.args) == 2 and sf.args[0] == "Log" and sf.args[1] in ("Log",):
            logx = logy = True
        elif K.is_list(sf) and len(sf.args) == 2 and sf.args[0] == "Log" and is_sym(sf.args[1], "None"):
            logx = True
        else:
            raise NotImplementedError(f"{fname}: ScalingFunctions -> {K_input(sf)} is not supported")
        if logx and a <= 0:
            raise ValueError(f"{fname}: the x range must be positive for a logarithmic x scale")
    n = _int_opt(o["PlotPoints"], 50, fname, "PlotPoints", 2)
    maxrec = _int_opt(o["MaxRecursion"], 6, fname, "MaxRecursion", 0)
    maxrec = min(maxrec, 15)
    held, explicit_list = _curves_of(f, var)
    compiled = []
    for h in held:
        c = Compiled(h, [var], fname)
        symbolic_check(c, [var], fname)
        compiled.append(c)
    if not explicit_list and len(compiled) == 1 and compiled[0].sym is not None and K.is_list(compiled[0].sym):
        held = list(K.substitute(compiled[0].sym, {compiled[0].dummies[0].name: var}).args)
        compiled = [Compiled(h, [var], fname) for h in held]
        for c in compiled:
            symbolic_check(c, [var], fname)
    np = _np()
    excl = _exclusion_points(o["Exclusions"], var, a, b, fname)
    jumps = not is_sym(o["Exclusions"], "None")
    curves, evaluators = [], []
    ta, tb = (math.log(a), math.log(b)) if logx else (a, b)
    for c in compiled:
        probe = [(a + (b - a) * t,) for t in (0.1234, 0.5017, 0.8761)]
        c.cross_check(probe)

        def g(t, c=c):
            x = np.exp(t) if logx else t
            y = c(x)
            if logy:
                with np.errstate(all="ignore"):
                    y = np.where(y > 0, np.log(np.where(y > 0, y, 1.0)), np.nan)
            return y
        tex = [math.log(e) for e in excl if e > 0] if logx else excl
        lines = sample_1d(g, ta, tb, n, maxrec, jumps, tex)
        lines = [ln for ln in lines if len(ln) >= 2]
        curves.append(lines)
        evaluators.append(g)
    allx = [p[0] for cv in curves for ln in cv for p in ln]
    ally = [p[1] for cv in curves for ln in cv for p in ln]
    xs_all = np.array([p[0] for cv in curves for ln in cv for p in ln])
    ys_all = np.array([p[1] for cv in curves for ln in cv for p in ln])
    auto_y = None
    if ys_all.size:
        o_ = np.argsort(xs_all, kind="stable")
        auto_y = None
        lo_hi = [weighted_range(np.array([p[0] for ln in cv for p in ln]), np.array([p[1] for ln in cv for p in ln]))
                 for cv in curves if any(cv)]
        auto_y = (min(r[0] for r in lo_hi), max(r[1] for r in lo_hi))
        del o_
    xr = (ta, tb)
    prange = resolve_plot_range(o["PlotRange"], fname, [xr, data_range(ally)], auto=[xr, auto_y], full=[xr, None])
    del allx
    styles = style_list(o["PlotStyle"], len(curves), fname)
    content = filling_content(o["Filling"], o["FillingStyle"], curves, styles, prange, fname, evaluators)
    for st, lines in zip(styles, curves):
        if st is None:
            continue
        content.append(L(st, *[Expr("Line", pts(ln)) for ln in lines]))
    defaults = [("AspectRatio", GOLDEN), ("Axes", L(True, True)), ("PlotRange", range_expr(prange)),
                ("PlotRangeClipping", True),
                ("PlotRangePadding", L(L(Expr("Scaled", 0.02), Expr("Scaled", 0.02)),
                                       L(Expr("Scaled", 0.05), Expr("Scaled", 0.05))))]
    if logx or logy:
        defaults.insert(2, ("Ticks", L(_ticks_for_log(*prange[0]) if logx else AUTO,
                                       _ticks_for_log(*prange[1]) if logy else AUTO)))
    gr = finish_graphics(L(*content), gopts, defaults)
    labels = [Expr("HoldForm", h) for h in held]
    return legend_wrap(gr, o["PlotLegends"], styles, labels, fname)


_PLOT_NOTE = ("Sampling: PlotPoints (default 50) equally spaced points, then MaxRecursion (default 6, at most 15) "
              "rounds of adaptive bisection; curves split where the value is not real, at detected jumps "
              "(Exclusions -> Automatic; None joins across) and at Exclusions -> {x1, ...} / {x == c, ...}. "
              "Options: PlotRange (Full/All/Automatic/{ymin, ymax}/s/{{xmin, xmax}, {ymin, ymax}}), PlotStyle, "
              "PlotPoints, MaxRecursion, Exclusions, Filling (Axis/Bottom/Top/c/{i -> j-spec}), FillingStyle, "
              "PlotLegends (labels, Automatic, Placed, LineLegend), ScalingFunctions (None, \"Log\" axes), and all "
              "Graphics options (passed to the result). Mesh, ColorFunction, PlotTheme other than the default, "
              "WorkingPrecision other than MachinePrecision, ExclusionsStyle and ClippingStyle settings raise "
              "NotImplementedError. A plotted expression with free parameters stays unevaluated (Wolfram draws an "
              "empty plot); a non-numeric range raises. Result: Graphics[{{Directive[colour, AbsoluteThickness[2]], "
              "Line[...], ...}, ...}, opts] with an explicit PlotRange.")

K.BUILTIN_ATTRIBUTES["Plot"] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["LogPlot"] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["LogLinearPlot"] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["LogLogPlot"] = {"HoldAll", "Protected"}


@fn("Plot", area=AREA, status="added", hold=True,
    notes="Plot[f, {x, xmin, xmax}] and Plot[{f1, f2, ...}, {x, xmin, xmax}] (HoldAll; x is localised): "
          "Graphics of the curves y = f(x). " + _PLOT_NOTE + " Check: numeric cross-check (compiled code compared "
          "with the kernel's N at sample points; mismatch falls back to kernel evaluation) and reference (tests "
          "check every sampled point against f).")
def _plot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    return function_plot("Plot", (f, spec) + opts)


@fn("LogPlot", area=AREA, status="added", hold=True,
    notes="LogPlot[f, {x, xmin, xmax}] (lists of f too): Plot with a logarithmic y axis; points store {x, Log[y]} "
          "(natural log, as Wolfram) and Ticks label powers of 10 in the original scale; non-positive values are "
          "omitted (they split the curve). " + _PLOT_NOTE + " Check: numeric cross-check and reference (sampled "
          "points satisfy Log[f(x)]).")
def _logplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    return function_plot("LogPlot", (f, spec) + opts, logy=True)


@fn("LogLinearPlot", area=AREA, status="added", hold=True,
    notes="LogLinearPlot[f, {x, xmin, xmax}] with 0 < xmin: logarithmic x axis (points {Log[x], f(x)}, sampled "
          "evenly in Log[x]); xmin <= 0 raises ValueError. " + _PLOT_NOTE + " Check: numeric cross-check and "
          "reference (sampled points satisfy y = f(Exp[t])).")
def _loglinearplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    return function_plot("LogLinearPlot", (f, spec) + opts, logx=True)


@fn("LogLogPlot", area=AREA, status="added", hold=True,
    notes="LogLogPlot[f, {x, xmin, xmax}] with 0 < xmin: logarithmic x and y axes (points {Log[x], Log[f(x)]}); "
          "non-positive values are omitted. " + _PLOT_NOTE + " Check: numeric cross-check and reference (a power "
          "law x^k plots as a straight line of slope k).")
def _loglogplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    return function_plot("LogLogPlot", (f, spec) + opts, logx=True, logy=True)


# ───────────────────────────── list data ─────────────────────────────

_DATA_WRAPPERS = {"Tooltip", "Annotation", "Hyperlink", "StatusArea", "PopupWindow", "Button", "EventHandler",
                  "Mouseover"}
_UNSUPPORTED_WRAPPERS = {"Labeled", "Callout", "Legended", "Highlighted"}
_UNSUPPORTED_DATA = {"Quantity", "Around", "Interval", "TimeSeries", "EventSeries", "TemporalData", "WeightedData",
                     "EventData", "SparseArray", "QuantityArray", "Tabular", "Dataset", "Rule"}


def scalar(x, fname, what="data value"):
    """float, None (missing), Unevaluated (symbolic), or raise (strings, unsupported value forms)."""
    if is_missing(x):
        return None
    if isinstance(x, bool):
        raise TypeError(f"{fname}: {what} {K_input(x)} is not a real number")
    v = num(x)
    if v is not None:
        return v
    if isinstance(x, (int, float)):
        return None                                    # inf/nan values are missing
    if isinstance(x, str):
        raise TypeError(f"{fname}: {what} {K_input(x)} is not a real number")
    if isinstance(x, complex) or (isinstance(x, Expr) and x.head == "Complex"):
        return None                                    # non-real values are not plotted, as in Wolfram
    if isinstance(x, Expr) and x.head in _UNSUPPORTED_DATA:
        raise NotImplementedError(f"{fname}: {x.head} data values are not supported")
    if isinstance(x, Expr) and x.head == "DirectedInfinity":
        return None
    if K.is_list(x):
        raise ValueError(f"{fname}: {K_input(x)} is not a valid {what}")
    try:
        c = ev(Expr("N", x))
        if isinstance(c, complex) or (isinstance(c, Expr) and c.head == "Complex"):
            return None
    except Exception:
        pass
    raise Unevaluated()


def _is_scalar_like(x):
    return not K.is_list(x) and not (isinstance(x, Expr) and x.head in _DATA_WRAPPERS | _UNSUPPORTED_WRAPPERS
                                     | {"Style", "Association"})


def _is_pair(x, k=2):
    return K.is_list(x) and len(x.args) == k and all(_is_scalar_like(a) for a in x.args)


class Dataset:
    def __init__(self, points, style=None, wrappers=(), labels=None):
        self.points = points          # list of tuples (floats) or None for a gap
        self.style = style
        self.wrappers = list(wrappers)
        self.labels = labels


def _unwrap(d, fname):
    style, wrappers = None, []
    while isinstance(d, Expr) and isinstance(d.head, str) and (d.head == "Style" or d.head in _DATA_WRAPPERS
                                                                or d.head in _UNSUPPORTED_WRAPPERS):
        if d.head in _UNSUPPORTED_WRAPPERS:
            raise NotImplementedError(f"{fname}: the data wrapper {d.head} is not supported")
        if not d.args:
            raise ValueError(f"{fname}: empty {d.head} wrapper")
        if d.head == "Style":
            rest = d.args[1:]
            style = rest[0] if len(rest) == 1 else Expr("Directive", *rest)
        else:
            wrappers.append((d.head, d.args[1:]))
        d = d.args[0]
    return d, style, wrappers


def _assoc_items(a):
    return [(r.args[0], r.args[1]) for r in a.args if is_rule(r)]


def one_dataset(d, fname, k=2, data_range=None, xform=None):
    """A single dataset -> Dataset of k-tuples (y lists get x = 1, 2, ... or DataRange)."""
    d, style, wrappers = _unwrap(d, fname)
    if isinstance(d, Expr) and d.head == "Association":
        items = _assoc_items(d)
        if items and all(num(kk) is not None for kk, _ in items) and k == 2:
            d = L(*[L(kk, v) for kk, v in items])
        else:
            d = L(*[v for _, v in items])
    if isinstance(d, Expr) and d.head in _UNSUPPORTED_DATA:
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, Symbol) or isinstance(d, Expr):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list, not {K_input(d)}")
    items = list(d.args)
    pts_ = []
    if items and all(_is_pair(i, k) or is_missing(i) for i in items) and any(_is_pair(i, k) for i in items) and k > 1:
        for i in items:
            if is_missing(i):
                pts_.append(None)
                continue
            i, _, _ = _unwrap(i, fname)
            cs = [scalar(c, fname) for c in i.args]
            pts_.append(None if any(c is None for c in cs) else tuple(cs))
    else:
        ys = []
        for i in items:
            i, _, _ = _unwrap(i, fname)
            if K.is_list(i):
                raise ValueError(f"{fname}: the data {K_input(d)} mixes values and lists")
            ys.append(scalar(i, fname))
        n = len(ys)
        if k == 2:
            if data_range is not None:
                a, b = data_range
                xs = [a + (b - a) * j / (n - 1) if n > 1 else a for j in range(n)]
            else:
                xs = [float(j + 1) for j in range(n)]
            if xform is not None:
                xs = xform(n)
            pts_ = [None if y is None else (x, y) for x, y in zip(xs, ys)]
        elif k == 1:
            pts_ = [None if y is None else (y,) for y in ys]
        else:
            raise ValueError(f"{fname}: expected a list of {k}-tuples, not {K_input(d)}")
    return Dataset(pts_, style, wrappers)


def datasets(data, fname, k=2, data_range=None, xform=None, multi=False):
    """Data argument -> list of Dataset (a single y list / point list, or a list of datasets; ``multi``: the
    elements are always datasets, as in StackedListPlot)."""
    data, style, wrappers = _unwrap(data, fname)
    if isinstance(data, Expr) and data.head == "Association":
        items = _assoc_items(data)
        if items and all(K.is_list(v) or isinstance(v, Expr) and v.head in ("Association", "Style")
                         for _, v in items):
            out = [one_dataset(v, fname, k, data_range, xform) for _, v in items]
            for (kk, _), ds in zip(items, out):
                ds.labels = kk
            return out
        return [one_dataset(data, fname, k, data_range, xform)]
    if not K.is_list(data):
        if isinstance(data, Expr) and data.head in _UNSUPPORTED_DATA:
            raise NotImplementedError(f"{fname}: {data.head} data is not supported")
        if isinstance(data, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list, not {K_input(data)}")
    items = list(data.args)
    multi = multi or items and all(
        (K.is_list(_unwrap(i, fname)[0]) or (isinstance(_unwrap(i, fname)[0], Expr) and
                                             _unwrap(i, fname)[0].head == "Association"))
        for i in items) and not all(_is_pair(i, k) or is_missing(i) for i in items)
    if multi:
        out = [one_dataset(i, fname, k, data_range, xform) for i in items]
        return out
    ds = one_dataset(data, fname, k, data_range, xform)
    ds.style = ds.style or style
    ds.wrappers = wrappers + ds.wrappers
    return [ds]


def _data_range_opt(v, fname):
    if v is None or is_sym(v, "Automatic", "All"):
        return None
    if K.is_list(v) and len(v.args) == 2 and all(num(a) is not None for a in v.args):
        return (num(v.args[0]), num(v.args[1]))
    raise NotImplementedError(f"{fname}: DataRange -> {K_input(v)} is not supported")


def runs(points):
    """Consecutive non-missing runs of a point list."""
    out, cur = [], []
    for p in points:
        if p is None:
            if cur:
                out.append(cur)
            cur = []
        else:
            cur.append(p)
    if cur:
        out.append(cur)
    return out


def step_points(run, step, fname):
    """Horizontal steps through the points of one run (ListStepPlot: Right, Left or Center)."""
    if len(run) == 1:
        x, y = run[0]
        return [(x - 0.5, y), (x + 0.5, y)] if step == "Center" else ([(x, y), (x + 1, y)] if step == "Right"
                                                                          else [(x - 1, y), (x, y)])
    xs = [p[0] for p in run]
    ys = [p[1] for p in run]
    out = []
    n = len(run)
    if step == "Right":
        for i in range(n):
            x1 = xs[i + 1] if i + 1 < n else xs[i] + (xs[i] - xs[i - 1])
            out += [(xs[i], ys[i]), (x1, ys[i])]
    elif step == "Left":
        for i in range(n):
            x0 = xs[i - 1] if i > 0 else xs[0] - (xs[1] - xs[0])
            out += [(x0, ys[i]), (xs[i], ys[i])]
    else:
        for i in range(n):
            x0 = (xs[i - 1] + xs[i]) / 2 if i > 0 else xs[0] - (xs[1] - xs[0]) / 2
            x1 = (xs[i] + xs[i + 1]) / 2 if i + 1 < n else xs[i] + (xs[i] - xs[i - 1]) / 2
            out += [(x0, ys[i]), (x1, ys[i])]
    return out


def wrap_prims(prims, ds):
    out = L(*prims)
    for h, extra in reversed(ds.wrappers):
        out = Expr(h, out, *extra)
    return out


_LIST_DEFAULTS = {"PlotRange": AUTO, "PlotStyle": AUTO, "Joined": AUTO, "Filling": NONE, "FillingStyle": AUTO,
                  "PlotLegends": NONE, "DataRange": AUTO, "PlotMarkers": NONE, "InterpolationOrder": AUTO,
                  "ScalingFunctions": NONE, "ClippingStyle": NONE, "PlotTheme": AUTO, "Mesh": NONE,
                  "ColorFunction": AUTO, "PlotHighlighting": AUTO, "Method": AUTO, "PerformanceGoal": AUTO,
                  "IntervalMarkers": AUTO, "PlotLabels": NONE, "PlotLayout": AUTO, "PlotFit": NONE,
                  "LabelingFunction": AUTO}


def _check_list_unsupported(o, fname):
    for n in ("ClippingStyle", "Mesh", "PlotLabels", "PlotFit"):
        if not is_sym(o.get(n, NONE), "None", "Automatic"):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    for n in ("ColorFunction", "PlotHighlighting", "Method", "IntervalMarkers", "LabelingFunction"):
        if not is_sym(o.get(n, AUTO), "Automatic", "None"):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    pg = o.get("PerformanceGoal", AUTO)
    if not (is_sym(pg, "Automatic") or pg in ("Quality", "Speed")):
        raise NotImplementedError(f"{fname}: PerformanceGoal -> {K_input(pg)} is not supported")
    th = o.get("PlotTheme", AUTO)
    if not (is_sym(th, "Automatic") or th == "Default"):
        raise NotImplementedError(f"{fname}: PlotTheme -> {K_input(th)} is not supported")
    pl = o.get("PlotLayout", AUTO)
    if not (is_sym(pl, "Automatic") or pl == "Overlaid"):
        raise NotImplementedError(f"{fname}: PlotLayout -> {K_input(pl)} is not supported")
    pm = o.get("PlotMarkers", NONE)
    if not is_sym(pm, "None", "Automatic") or is_sym(pm, "Automatic"):
        if not is_sym(pm, "None"):
            raise NotImplementedError(f"{fname}: PlotMarkers -> {K_input(pm)} is not supported (points are drawn "
                                      "with Point)")


def _log_axis(v, fname, what):
    if v is None:
        return None
    if v <= 0:
        return None
    return math.log(v)


def list_plot(fname, data, opts, joined_default=False, step=None, logx=False, logy=False, polar=False,
              defaults_extra=None, xform=None, dates=False, frame=False, stacked=False):
    defaults = dict(_LIST_DEFAULTS)
    if defaults_extra:
        defaults.update(defaults_extra)
    o, gopts = split_options(fname, opts, defaults)
    _check_list_unsupported(o, fname)
    sf = o["ScalingFunctions"]
    if not is_sym(sf, "None", "Automatic"):
        if logx or logy or polar or dates:
            raise NotImplementedError(f"{fname}: ScalingFunctions is not supported in {fname}")
        if sf == "Log" or (K.is_list(sf) and len(sf.args) == 2 and is_sym(sf.args[0], "None") and sf.args[1] == "Log"):
            logy = True
        elif K.is_list(sf) and len(sf.args) == 2 and sf.args[0] == "Log" and sf.args[1] == "Log":
            logx = logy = True
        elif K.is_list(sf) and len(sf.args) == 2 and sf.args[0] == "Log" and is_sym(sf.args[1], "None"):
            logx = True
        else:
            raise NotImplementedError(f"{fname}: ScalingFunctions -> {K_input(sf)} is not supported")
    dr = _data_range_opt(o["DataRange"], fname)
    if dates:
        dss = date_datasets(data, fname, xform, multi=stacked)
    else:
        dss = datasets(data, fname, 2, dr, multi=stacked)
    if polar:
        for ds in dss:
            new = []
            raw = ds.points
            if dr is None and all(p is None or p[0] == float(j + 1) for j, p in enumerate(raw)) and \
                    getattr(ds, "_ylist", True) and not _pairs_given(data, fname):
                n = len(raw)
                raw = [None if p is None else (2 * math.pi * j / n, p[1]) for j, p in enumerate(raw)]
            for p in raw:
                if p is None:
                    new.append(None)
                else:
                    th, r = p
                    new.append((r * math.cos(th), r * math.sin(th)))
            ds.points = new
    io = o["InterpolationOrder"]
    if not is_sym(io, "Automatic", "None") and io not in (0, 1):
        raise NotImplementedError(f"{fname}: InterpolationOrder -> {K_input(io)} is not supported (0 or 1)")
    if stacked:
        dss = stack_datasets(dss, fname)
    # log transforms
    if logx or logy:
        for ds in dss:
            new = []
            for p in ds.points:
                if p is None:
                    new.append(None)
                    continue
                x, y = p
                x2 = _log_axis(x, fname, "x") if logx else x
                y2 = _log_axis(y, fname, "y") if logy else y
                new.append(None if x2 is None or y2 is None else (x2, y2))
            ds.points = new
    joined = o["Joined"]
    if is_sym(joined, "Automatic"):
        jl = [joined_default] * len(dss)
    elif joined is True or joined is False:
        jl = [joined] * len(dss)
    elif K.is_list(joined) and all(j is True or j is False for j in joined.args) and joined.args:
        jl = [joined.args[i % len(joined.args)] for i in range(len(dss))]
    else:
        raise ValueError(f"{fname}: Joined must be True, False or a list of them, not {K_input(joined)}")
    if step is None and io == 0:
        step_kind = "Right"
    else:
        step_kind = step
    kinds = ["line" if j or step_kind else "point" for j in jl]
    styles = []
    for i, ds in enumerate(dss):
        st = style_list(o["PlotStyle"], len(dss), fname, kinds[i])[i]
        if ds.style is not None and st is not None:
            st = Expr("Directive", *st.args, *(ds.style.args if isinstance(ds.style, Expr) and
                                               ds.style.head == "Directive" else [ds.style]))
        styles.append(st)
    curves = []
    for ds, j in zip(dss, jl):
        rs = runs(ds.points)
        if step_kind and (j or step is not None):
            curves.append([step_points(r, step_kind, fname) for r in rs])
        else:
            curves.append(rs)
    allp = [p for cv in curves for r in cv for p in r]
    xr = data_range([p[0] for p in allp])
    yr = data_range([p[1] for p in allp])
    prange = resolve_plot_range(o["PlotRange"], fname, [xr, yr], auto=[xr, yr], full=[xr, yr])
    content = []
    fill = o["Filling"]
    if not is_sym(fill, "None"):
        if all(jl) or step_kind:
            content += filling_content(fill, o["FillingStyle"], curves, styles, prange, fname)
        else:
            content += stems(fill, o["FillingStyle"], curves, styles, prange, fname)
    for ds, st, j, cv in zip(dss, styles, jl, curves):
        if st is None:
            continue
        if j or (step_kind and step is not None):
            prims = [Expr("Line", pts(r)) for r in cv if len(r) >= 2] + \
                    [Expr("Point", pts(r)) for r in cv if len(r) == 1]
            if step is not None and not j:
                prims = [Expr("Line", pts(r[k:k + 2])) for r in cv for k in range(0, len(r), 2)]
        else:
            flat = [p for r in cv for p in r]
            prims = [Expr("Point", pts(flat))] if flat else []
        content.append(L(st, wrap_prims(prims, ds)) if ds.wrappers else L(st, *prims))
    defaults_g = [("AspectRatio", Symbol("Automatic") if polar else GOLDEN),
                  ("Axes", L(not frame, not frame) if not frame else L(False, False)),
                  ("PlotRange", range_expr(prange)), ("PlotRangeClipping", True),
                  ("PlotRangePadding", L(L(Expr("Scaled", 0.02), Expr("Scaled", 0.02)),
                                         L(Expr("Scaled", 0.05), Expr("Scaled", 0.05))))]
    if frame:
        defaults_g.insert(2, ("Frame", True))
    if logx or logy:
        defaults_g.insert(2, ("Ticks", L(_ticks_for_log(*prange[0]) if logx else AUTO,
                                         _ticks_for_log(*prange[1]) if logy else AUTO)))
    if dates:
        defaults_g.insert(2, ("FrameTicks", L(L(AUTO, AUTO), L(date_ticks(*prange[0]), AUTO))))
    gr = finish_graphics(L(*content), gopts, defaults_g)
    labels = [ds.labels for ds in dss]
    auto_labels = labels if all(lb is not None for lb in labels) else \
        [f"{i + 1}" for i in range(len(dss))] if len(dss) > 1 else None
    kind = "line" if any(jl) or step_kind else "point"
    return legend_wrap(gr, o["PlotLegends"], styles, auto_labels, fname, kind)


def _pairs_given(data, fname):
    d, _, _ = _unwrap(data, fname)
    if not K.is_list(d) or not d.args:
        return False
    if all(_is_pair(i) or is_missing(i) for i in d.args):
        return True
    first = _unwrap(d.args[0], fname)[0]
    return K.is_list(first) and first.args and all(_is_pair(i) or is_missing(i) for i in first.args)


def stems(spec, fstyle, curves, styles, prange, fname):
    """Filling for point plots: vertical stems from the filling level to each point (Wolfram's default)."""
    out = []
    n = len(curves)
    if K.is_list(spec) and all(is_rule(r) for r in spec.args):
        pairs = []
        for r in spec.args:
            i = r.args[0]
            if not (isinstance(i, int) and not isinstance(i, bool) and 1 <= i <= n):
                raise ValueError(f"{fname}: Filling {K_input(r)} refers to no dataset")
            pairs.append((i - 1, r.args[1]))
    else:
        pairs = [(i, spec) for i in range(n)]
    for i, tgt in pairs:
        if K.is_list(tgt):
            raise NotImplementedError(f"{fname}: filling between point datasets is not supported")
        lev = _fill_level(tgt, prange, fname)
        col = style_color(styles[i])
        st = fstyle if fstyle is not None and not is_sym(fstyle, "Automatic") else \
            Expr("Directive", Expr("Opacity", 0.3), col)
        lines = [Expr("Line", pts([(x, lev), (x, y)])) for r in curves[i] for (x, y) in r]
        if lines:
            out.append(L(st, *lines))
    return out


def stack_datasets(dss, fname):
    """StackedListPlot accumulation: dataset i becomes the running sum of datasets 1..i at each x (missing = 0);
    positive and negative parts stack separately."""
    xs = sorted({p[0] for ds in dss for p in ds.points if p is not None})
    pos_acc = {x: 0.0 for x in xs}
    neg_acc = {x: 0.0 for x in xs}
    out = []
    for ds in dss:
        vals = {}
        for p in ds.points:
            if p is not None:
                if p[0] in vals:
                    raise ValueError(f"{fname}: dataset has repeated x value {p[0]}")
                vals[p[0]] = p[1]
        new = []
        for x in xs:
            v = vals.get(x, 0.0)
            if v >= 0:
                pos_acc[x] += v
                new.append((x, pos_acc[x]))
            else:
                neg_acc[x] += v
                new.append((x, neg_acc[x]))
        out.append(Dataset(new, ds.style, ds.wrappers, ds.labels))
    return out


# ───────────────────────────── dates ─────────────────────────────

_EPOCH_ORDINAL = 693596          # date(1900, 1, 1).toordinal()
_UNITS = ("Year", "Month", "Day", "Hour", "Minute", "Second")


def is_date(x):
    if isinstance(x, Expr) and x.head in ("DateObject", "TimeObject"):
        return True
    if isinstance(x, str):
        return True
    if K.is_list(x) and 1 <= len(x.args) <= 6 and all(num(a) is not None for a in x.args) and \
            isinstance(x.args[0], int) and not isinstance(x.args[0], bool):
        return True
    return False


def abs_time(x, fname):
    """AbsoluteTime of a date (DateObject, TimeObject, string, date list, or a number = absolute time)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if not is_date(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {K_input(x)} is not a date")
    r = ev(Expr("AbsoluteTime", x))
    v = num(r)
    if v is None:
        raise ValueError(f"{fname}: {K_input(x)} is not a valid date")
    return v


def date_list(x, fname):
    r = ev(Expr("DateList", x))
    if not K.is_list(r) or len(r.args) != 6:
        raise ValueError(f"{fname}: {K_input(x)} is not a valid date")
    return [num(a) for a in r.args]


def _unit_of(x):
    if isinstance(x, Expr) and x.head == "DateObject" and len(x.args) > 1 and isinstance(x.args[1], str):
        return x.args[1] if x.args[1] in _UNITS else "Second"
    if K.is_list(x):
        return _UNITS[min(len(x.args), 6) - 1]
    return "Day"


def _add_units(dl, unit, k):
    """Date list + k units (calendar arithmetic for years and months) -> absolute time."""
    import datetime as dt
    y, mo, d, h, mi, s = dl
    y, mo, d, h, mi = int(y), int(mo), int(d), int(h), int(mi)
    if unit in ("Year", "Quarter", "Month"):
        months = k * {"Year": 12, "Quarter": 3, "Month": 1}[unit]
        t = (y * 12 + (mo - 1)) + months
        y, mo = divmod(t, 12)
        mo += 1
        import calendar
        d = min(d, calendar.monthrange(y, mo)[1])
        base = dt.datetime(y, mo, d, h, mi)
        extra = 0.0
    else:
        base = dt.datetime(y, mo, d, h, mi)
        extra = {"Week": 7 * 86400, "Day": 86400, "Hour": 3600, "Minute": 60, "Second": 1,
                 "Millisecond": 0.001}[unit] * k
    days = base.toordinal() - _EPOCH_ORDINAL
    return days * 86400.0 + base.hour * 3600 + base.minute * 60 + float(s) + extra


def date_sequence(spec, n, fname):
    """Absolute times of n regular values for a DateListPlot datespec."""
    if K.is_list(spec) and len(spec.args) in (2, 3) and not is_date(spec):
        a = spec.args
        if len(a) == 2:
            t0, t1 = abs_time(a[0], fname), abs_time(a[1], fname)
            return [t0 + (t1 - t0) * j / (n - 1) if n > 1 else t0 for j in range(n)]
        unit = a[2]
        if not (isinstance(unit, str) and unit in _UNITS + ("Quarter", "Week", "Millisecond")):
            raise NotImplementedError(f"{fname}: the date increment {K_input(unit)} is not supported")
        if is_sym(a[1], "Automatic"):
            dl = date_list(a[0], fname)
            return [_add_units(dl, unit, j) - _add_units(dl, unit, 0) + abs_time(a[0], fname) for j in range(n)] \
                if unit not in ("Year", "Quarter", "Month") else \
                [abs_time(a[0], fname) + _add_units(dl, unit, j) - _add_units(dl, unit, 0) for j in range(n)]
        if is_sym(a[0], "Automatic"):
            dl = date_list(a[1], fname)
            t_end = abs_time(a[1], fname)
            return [t_end + _add_units(dl, unit, j - (n - 1)) - _add_units(dl, unit, 0) for j in range(n)]
        raise ValueError(f"{fname}: the datespec {K_input(spec)} is not valid")
    if not is_date(spec):
        if isinstance(spec, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {K_input(spec)} is not a date specification")
    unit = _unit_of(spec)
    dl = date_list(spec, fname)
    t0 = abs_time(spec, fname)
    return [t0 + _add_units(dl, unit, j) - _add_units(dl, unit, 0) for j in range(n)]


def _date_pair(i):
    return K.is_list(i) and len(i.args) == 2 and is_date(i.args[0]) and _is_scalar_like(i.args[1]) and \
        not (isinstance(i.args[0], int) and isinstance(i.args[1], int) and False)


def _one_date_dataset(d, fname, xs=None):
    d, style, wrappers = _unwrap(d, fname)
    if isinstance(d, Expr) and d.head == "Association":
        items = _assoc_items(d)
        if items and all(is_date(k) or num(k) is not None for k, _ in items):
            d = L(*[L(k, v) for k, v in items])
        else:
            d = L(*[v for _, v in items])
    if isinstance(d, Expr) and d.head in _UNSUPPORTED_DATA:
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list, not {K_input(d)}")
    items = list(d.args)
    if items and all(_date_pair(i) or is_missing(i) for i in items):
        p = []
        for i in items:
            if is_missing(i):
                p.append(None)
                continue
            y = scalar(i.args[1], fname)
            p.append(None if y is None else (abs_time(i.args[0], fname), y))
        good = sorted([q for q in p if q is not None], key=lambda q: q[0])
        if all(q is not None for q in p):
            p = good                                   # chronological order
        return Dataset(p, style, wrappers)
    ys = [scalar(_unwrap(i, fname)[0], fname) for i in items]
    if xs is None:
        raise ValueError(f"{fname}: regular data {{y1, y2, ...}} needs a date specification as the second argument")
    tt = xs(len(ys))
    return Dataset([None if y is None else (t, y) for t, y in zip(tt, ys)], style, wrappers)


def date_datasets(data, fname, xs=None, multi=False):
    data, style, wrappers = _unwrap(data, fname)
    if not K.is_list(data):
        if isinstance(data, Expr) and data.head == "Association":
            return [_one_date_dataset(data, fname, xs)]
        if isinstance(data, Expr) and data.head in _UNSUPPORTED_DATA:
            raise NotImplementedError(f"{fname}: {data.head} data is not supported")
        if isinstance(data, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list, not {K_input(data)}")
    items = list(data.args)
    single = not multi and all(_date_pair(i) or is_missing(i) or _is_scalar_like(_unwrap(i, fname)[0]) for i in items)
    if single:
        ds = _one_date_dataset(data, fname, xs)
        ds.style = style
        ds.wrappers = wrappers
        return [ds]
    return [_one_date_dataset(i, fname, xs) for i in items]


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def date_ticks(t0, t1):
    """Ticks {{AbsoluteTime, label}, ...} at whole years, months, days or hours across [t0, t1]."""
    import datetime as dt
    if not (math.isfinite(t0) and math.isfinite(t1)) or t1 < t0:
        return AUTO
    span = t1 - t0
    start = dt.datetime(1900, 1, 1) + dt.timedelta(seconds=t0)
    out = []

    def at(d):
        return (d - dt.datetime(1900, 1, 1)).total_seconds()
    if span > 3 * 365 * 86400:
        step = max(1, int(span / (365.2425 * 86400) / 6) or 1)
        y = start.year + (1 if (start.month, start.day, start.hour) != (1, 1, 0) else 0)
        while True:
            d = dt.datetime(y, 1, 1)
            if at(d) > t1:
                break
            out.append(L(fl(at(d)), str(y)))
            y += step
    elif span > 80 * 86400:
        step = max(1, int(span / (30.44 * 86400) / 6) or 1)
        y, m = start.year, start.month + (1 if (start.day, start.hour) != (1, 0) else 0)
        while True:
            y2, m2 = y + (m - 1) // 12, (m - 1) % 12 + 1
            d = dt.datetime(y2, m2, 1)
            if at(d) > t1:
                break
            out.append(L(fl(at(d)), f"{_MONTHS[m2 - 1]} {y2}"))
            m += step
    elif span >= 1.5 * 86400:
        step = max(1, int(span / 86400 / 7) or 1)
        d = dt.datetime(start.year, start.month, start.day) + (dt.timedelta(days=1) if start.hour or start.minute
                                                                 or start.second else dt.timedelta(0))
        while at(d) <= t1:
            out.append(L(fl(at(d)), f"{_MONTHS[d.month - 1]} {d.day}"))
            d += dt.timedelta(days=step)
    else:
        step = max(1, int(span / 3600 / 6) or 1)
        d = dt.datetime(start.year, start.month, start.day, start.hour) + dt.timedelta(hours=1 if (
            start.minute or start.second) else 0)
        while at(d) <= t1:
            out.append(L(fl(at(d)), f"{d.hour:02d}:00"))
            d += dt.timedelta(hours=step)
    return L(*out) if out else AUTO


# ───────────────────────────── list plot heads ─────────────────────────────

def _list_args(args, fname, lo=1, hi=1):
    pos, opts = split_trailing_options(args)
    if not lo <= len(pos) <= hi:
        want = str(lo) if lo == hi else f"{lo} or {hi}"
        raise ValueError(f"{fname}: expected {want} positional argument(s), got {len(pos)}")
    return pos, opts


_LIST_NOTE = ("Data: {y1, ...} (x = 1, 2, ... or DataRange), {{x1, y1}, ...}, several datasets {data1, ...}, "
              "associations, Style[data, s] and the Tooltip/Annotation/Hyperlink/StatusArea/Button/EventHandler/"
              "PopupWindow wrappers; None and Missing[...] values are gaps; a symbolic value leaves the call "
              "unevaluated (Wolfram drops it), strings raise TypeError, Quantity/Around/Interval/TimeSeries data "
              "and the Labeled/Callout/Legended/Highlighted wrappers raise NotImplementedError. Options: PlotRange, "
              "PlotStyle, Joined, Filling (stems for points, polygons for lines), FillingStyle, PlotLegends, "
              "DataRange, InterpolationOrder (0 or 1), ScalingFunctions (None/\"Log\") and all Graphics options; "
              "PlotMarkers other than None, Mesh, ColorFunction, PlotTheme, PlotLabels, PlotFit and PlotLayout "
              "settings raise NotImplementedError.")


@fn("ListPlot", area=AREA, status="added",
    notes="ListPlot[data]: Graphics of the points ({Directive[colour, PointSize[0.0128], AbsoluteThickness[2]], "
          "Point[{...}]} per dataset; Joined -> True draws lines). " + _LIST_NOTE + " Check: reference (Wolfram "
          "documentation forms; tests compare the plotted coordinates with the data).")
def _listplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListPlot")
    return list_plot("ListPlot", pos[0], o)


@fn("ListLinePlot", area=AREA, status="added",
    notes="ListLinePlot[data]: lines through the points of each dataset (split at missing values; "
          "InterpolationOrder -> 0 gives steps). " + _LIST_NOTE + " Check: reference (plotted vertices equal the "
          "data).")
def _listlineplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListLinePlot")
    return list_plot("ListLinePlot", pos[0], o, joined_default=True)


def _step_arg(pos, fname):
    if len(pos) < 2:
        return "Right"
    s = pos[1]
    if is_sym(s, "Right", "Left", "Center"):
        return s.name
    if isinstance(s, Symbol):
        raise ValueError(f"{fname}: the step specification must be Right, Left or Center, not {K_input(s)}")
    raise TypeError(f"{fname}: the step specification must be Right, Left or Center, not {K_input(s)}")


@fn("ListStepPlot", area=AREA, status="added",
    notes="ListStepPlot[data] and ListStepPlot[data, Right | Left | Center] (default Right): each point is a "
          "horizontal step; the last (Right) or first (Left) step repeats the neighbouring spacing; Joined -> False "
          "draws the steps without the vertical risers. " + _LIST_NOTE + " Check: reference (step vertices computed "
          "independently in tests).")
def _liststepplot(data: Expr, *rest: Expr | Option) -> Expr:
    pos, o = _list_args((data,) + rest, "ListStepPlot", 1, 2)
    return list_plot("ListStepPlot", pos[0], o, joined_default=True, step=_step_arg(pos, "ListStepPlot"),
                     defaults_extra={"Joined": True})


@fn("ListLogPlot", area=AREA, status="added",
    notes="ListLogPlot[data]: ListPlot with a logarithmic y axis (points {x, Log[y]}, ticks at powers of 10; "
          "non-positive y values are omitted). " + _LIST_NOTE + " Check: reference (Log of the data in tests).")
def _listlogplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListLogPlot")
    return list_plot("ListLogPlot", pos[0], o, logy=True)


@fn("ListLogLinearPlot", area=AREA, status="added",
    notes="ListLogLinearPlot[data]: logarithmic x axis (points {Log[x], y}; non-positive x omitted). " + _LIST_NOTE +
          " Check: reference (Log of the x data in tests).")
def _listloglinearplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListLogLinearPlot")
    return list_plot("ListLogLinearPlot", pos[0], o, logx=True)


@fn("ListLogLogPlot", area=AREA, status="added",
    notes="ListLogLogPlot[data]: logarithmic x and y axes (points {Log[x], Log[y]}; non-positive values omitted). " +
          _LIST_NOTE + " Check: reference (power-law data lies on a straight line of the exponent's slope).")
def _listloglogplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListLogLogPlot")
    return list_plot("ListLogLogPlot", pos[0], o, logx=True, logy=True)


@fn("ListPolarPlot", area=AREA, status="added",
    notes="ListPolarPlot[{r1, ..., rn}] (r_i at angle 2 Pi (i - 1)/n, as documented) and ListPolarPlot[{{t1, r1}, "
          "...}] (angle in radians): points {r Cos[t], r Sin[t]}; AspectRatio -> Automatic. " + _LIST_NOTE +
          " Check: reference (Cartesian coordinates of the polar data in tests; every radius is preserved).")
def _listpolarplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "ListPolarPlot")
    return list_plot("ListPolarPlot", pos[0], o, polar=True)


def _stacked_defaults(n):
    rules = [Expr("Rule", 1, Symbol("Axis"))] + [Expr("Rule", i, L(i - 1)) for i in range(2, n + 1)]
    return L(*rules)


@fn("StackedListPlot", area=AREA, status="added",
    notes="StackedListPlot[{data1, data2, ...}]: curve i is the accumulation of data1 ... data_i at each x (missing "
          "values count as 0; negative values stack downwards separately), joined, with Filling {1 -> Axis, "
          "i -> {i - 1}} by default. Data forms and options as ListLinePlot. " + _LIST_NOTE + " Check: reference "
          "(the running sums computed in tests).")
def _stackedlistplot(data: Expr, *opts: Option) -> Expr:
    pos, o = _list_args((data,) + opts, "StackedListPlot")
    n = max(1, len(datasets(pos[0], "StackedListPlot", multi=True)))
    have = {opt_name(r) for r in o}
    extra = [] if "Filling" in have else [Expr("Rule", Symbol("Filling"), _stacked_defaults(n))]
    return list_plot("StackedListPlot", pos[0], list(o) + extra, joined_default=True, stacked=True)


def _date_args(args, fname):
    pos, opts = split_trailing_options(args)
    if not 1 <= len(pos) <= 2:
        raise ValueError(f"{fname}: expected {fname}[data] or {fname}[values, datespec]")
    xs = None
    if len(pos) == 2:
        spec = pos[1]
        xs = (lambda n, spec=spec: date_sequence(spec, n, fname))
    return pos[0], opts, xs


_DATE_NOTE = ("Dates: DateObject, date strings, date lists {y, m, d, ...} and absolute times; the x coordinate is "
              "AbsoluteTime (seconds since 1 Jan 1900, as Wolfram); {{date, y}, ...} data is sorted "
              "chronologically; regular data {y1, ...} needs a datespec: start (step = the smallest unit given), "
              "{start, end}, {start, Automatic, \"Day\" | \"Month\" | ...} or {Automatic, end, step} (calendar "
              "arithmetic for months and years). Frame -> True with date labels at whole years, months, days or "
              "hours (FrameTicks). TimeSeries/EventSeries input raises NotImplementedError. ")


@fn("DateListPlot", area=AREA, status="added",
    notes="DateListPlot[{{date1, y1}, ...}], DateListPlot[{y1, ...}, datespec], DateListPlot[{data1, ...}]: joined "
          "curves against dates. " + _DATE_NOTE + _LIST_NOTE + " Check: reference (AbsoluteTime of each date and "
          "calendar increments checked in tests).")
def _datelistplot(data: Expr, *rest: Expr | Option) -> Expr:
    d, o, xs = _date_args((data,) + rest, "DateListPlot")
    return list_plot("DateListPlot", d, o, joined_default=True, dates=True, xform=xs, frame=True)


@fn("DateListLogPlot", area=AREA, status="added",
    notes="DateListLogPlot[data] / DateListLogPlot[values, datespec]: DateListPlot with a logarithmic y axis "
          "(points {AbsoluteTime, Log[y]}; non-positive values omitted). " + _DATE_NOTE + _LIST_NOTE +
          " Check: reference.")
def _datelistlogplot(data: Expr, *rest: Expr | Option) -> Expr:
    d, o, xs = _date_args((data,) + rest, "DateListLogPlot")
    return list_plot("DateListLogPlot", d, o, joined_default=True, dates=True, xform=xs, frame=True, logy=True)


@fn("DateListStepPlot", area=AREA, status="added",
    notes="DateListStepPlot[data] / DateListStepPlot[values, datespec]: steps (Right) against dates. " + _DATE_NOTE +
          _LIST_NOTE + " Check: reference.")
def _dateliststepplot(data: Expr, *rest: Expr | Option) -> Expr:
    d, o, xs = _date_args((data,) + rest, "DateListStepPlot")
    return list_plot("DateListStepPlot", d, o, joined_default=True, dates=True, xform=xs, frame=True, step="Right",
                     defaults_extra={"Joined": True})


@fn("StackedDateListPlot", area=AREA, status="added",
    notes="StackedDateListPlot[{data1, data2, ...}]: DateListPlot with the datasets accumulated as in "
          "StackedListPlot (filled between consecutive curves). " + _DATE_NOTE + _LIST_NOTE + " Check: reference "
          "(running sums at each date).")
def _stackeddatelistplot(data: Expr, *rest: Expr | Option) -> Expr:
    d, o, xs = _date_args((data,) + rest, "StackedDateListPlot")
    n = max(1, len(date_datasets(d, "StackedDateListPlot", xs, multi=True)))
    have = {opt_name(r) for r in o}
    extra = [] if "Filling" in have else [Expr("Rule", Symbol("Filling"), _stacked_defaults(n))]
    return list_plot("StackedDateListPlot", d, list(o) + extra, joined_default=True, dates=True, xform=xs,
                     frame=True, stacked=True)
