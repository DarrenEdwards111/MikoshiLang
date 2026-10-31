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
