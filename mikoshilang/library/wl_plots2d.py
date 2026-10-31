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


def split_options(fname, opts, defaults, dim=2, own_validation=()):
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
            if v is not None and r.head == "Rule" and n not in own_validation:
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


# ───────────────────────────── DiscretePlot ─────────────────────────────

def discrete_values(spec, fname):
    """Standard iterator {n, nmax} / {n, nmin, nmax} / {n, nmin, nmax, dn} / {n, {n1, ...}} -> (var, [values])."""
    if not K.is_list(spec) or len(spec.args) < 2 or len(spec.args) > 4:
        raise ValueError(f"{fname}: the iterator must be {{n, nmax}}, {{n, nmin, nmax}}, {{n, nmin, nmax, dn}} or "
                         f"{{n, {{n1, ...}}}}, not {K_input(spec)}")
    v = var_symbol(spec.args[0], fname)
    a = [ev(x) for x in spec.args[1:]]
    if len(a) == 1 and K.is_list(a[0]):
        vals = [real_arg(x, fname, "iterator value") for x in a[0].args]
        return v, vals, list(a[0].args)
    if len(a) == 1:
        lo, hi, d = 1, a[0], 1
    elif len(a) == 2:
        lo, hi, d = a[0], a[1], 1
    else:
        lo, hi, d = a
    lo_f, hi_f = real_arg(lo, fname, "iterator minimum"), real_arg(hi, fname, "iterator maximum")
    d_f = real_arg(d, fname, "iterator step")
    if d_f == 0:
        raise ValueError(f"{fname}: the iterator step must be non-zero")
    cnt = math.floor((hi_f - lo_f) / d_f + 1e-12) + 1
    if cnt > 100000:
        raise ValueError(f"{fname}: the iterator has {cnt} values (at most 100000 are plotted)")
    exact = all(isinstance(x, int) and not isinstance(x, bool) for x in (lo, d))
    vals = [lo_f + k * d_f for k in range(max(cnt, 0))]
    raw = [lo + k * d for k in range(max(cnt, 0))] if exact else [fl(x) for x in vals]
    return v, vals, raw


_DISCRETE_DEFAULTS = {"PlotRange": AUTO, "PlotStyle": AUTO, "Filling": Symbol("Axis"), "FillingStyle": AUTO,
                      "PlotLegends": NONE, "Joined": False, "ExtentSize": NONE, "PlotMarkers": NONE,
                      "ClippingStyle": NONE, "PlotTheme": AUTO, "Mesh": NONE, "ColorFunction": AUTO,
                      "PlotHighlighting": AUTO, "Method": AUTO, "PerformanceGoal": AUTO, "ScalingFunctions": NONE,
                      "ExtentMarkers": NONE, "ExtentElementFunction": AUTO, "PlotLabels": NONE,
                      "LabelingFunction": AUTO, "PlotLayout": AUTO, "PlotFit": NONE, "IntervalMarkers": AUTO}

K.BUILTIN_ATTRIBUTES["DiscretePlot"] = {"HoldAll", "Protected"}


def _kernel_values(held, var, raw, fname):
    """Exact kernel values f(n) for the iterator values (DiscretePlot evaluates f at the actual n)."""
    out = []
    d = Symbol("plot$w16$n")
    h = K.substitute(held, {var.name: d})
    for r in raw:
        y = ev(K.substitute(h, {d.name: r}))
        v = num(y)
        if v is None:
            if isinstance(y, Symbol) and not is_missing(y) or (isinstance(y, Expr) and _has_user_symbol(y)):
                raise Unevaluated()
            out.append(None)
        else:
            out.append(v)
    return out


def _has_user_symbol(e):
    if isinstance(e, Symbol):
        return e.name not in ("Pi", "E", "I", "Infinity", "ComplexInfinity", "Indeterminate", "GoldenRatio",
                              "EulerGamma", "Catalan", "Degree", "Khinchin", "Glaisher", "True", "False", "None")
    if isinstance(e, Expr):
        if e.head in ("Missing", "Complex", "DirectedInfinity"):
            return False
        return any(_has_user_symbol(a) for a in e.args)
    return False


@fn("DiscretePlot", area=AREA, status="added", hold=True,
    notes="DiscretePlot[f, {n, nmax}], {n, nmin, nmax}, {n, nmin, nmax, dn}, {n, {n1, ...}} and lists {f1, ...} "
          "(HoldAll; n is localised): points {n, f(n)} with stems to the axis (Filling -> Axis by default; "
          "Filling -> None, Bottom, Top, c, {i -> ...}); f is evaluated by the kernel at each exact n; non-real or "
          "undefined values are omitted; a value with free symbols leaves the call unevaluated. Joined -> True "
          "draws lines. ExtentSize other than None, PlotMarkers, Mesh, ColorFunction and PlotTheme settings raise "
          "NotImplementedError. At most 100000 values. Check: reference (the plotted points equal the kernel's "
          "f(n); tests compare them with independently computed sequences).")
def _discreteplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "DiscretePlot"
    pos, o_ = hold_args((f, spec) + opts)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected DiscretePlot[f, {{n, nmin, nmax}}, options]")
    f, spec = pos
    var, vals, raw = discrete_values(spec, fname)
    held, _ = _curves_of(f, var)
    o, gopts = split_options(fname, o_, _DISCRETE_DEFAULTS)
    _check_list_unsupported(o, fname)
    if not is_sym(o["ExtentSize"], "None"):
        raise NotImplementedError(f"{fname}: ExtentSize -> {K_input(o['ExtentSize'])} is not supported")
    if not is_sym(o["ScalingFunctions"], "None", "Automatic"):
        raise NotImplementedError(f"{fname}: ScalingFunctions is not supported")
    curves = []
    for h in held:
        ys = _kernel_values(h, var, raw, fname)
        curves.append([[(x, y)] for x, y in zip(vals, ys) if y is not None])
    joined = o["Joined"] is True
    if joined:
        curves = [[[p[0] for p in cv]] if cv else [] for cv in curves]
    allp = [p for cv in curves for r in cv for p in r]
    xr = data_range([p[0] for p in allp])
    yr = data_range([p[1] for p in allp] + [0.0])
    prange = resolve_plot_range(o["PlotRange"], fname, [xr, yr], auto=[xr, yr], full=[xr, yr])
    styles = style_list(o["PlotStyle"], len(curves), fname, "line" if joined else "point")
    content = []
    fill = o["Filling"]
    if not is_sym(fill, "None"):
        if joined:
            content += filling_content(fill, o["FillingStyle"], curves, styles, prange, fname)
        else:
            content += stems(fill, o["FillingStyle"], curves, styles, prange, fname)
    for st, cv in zip(styles, curves):
        if st is None:
            continue
        if joined:
            content.append(L(st, *[Expr("Line", pts(r)) for r in cv if len(r) >= 2]))
        else:
            flat = [p for r in cv for p in r]
            content.append(L(st, Expr("Point", pts(flat))) if flat else L(st))
    defaults = [("AspectRatio", GOLDEN), ("Axes", L(True, True)), ("PlotRange", range_expr(prange)),
                ("PlotRangeClipping", True),
                ("PlotRangePadding", L(L(Expr("Scaled", 0.02), Expr("Scaled", 0.02)),
                                       L(Expr("Scaled", 0.05), Expr("Scaled", 0.05))))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, [Expr("HoldForm", h) for h in held], fname,
                       "line" if joined else "point")


# ───────────────────────────── colours for array/density plots ─────────────────────────────

def blend(stops, t):
    """Piecewise-linear colour gradient through RGB stops (t in [0, 1])."""
    t = 0.0 if not math.isfinite(t) else min(1.0, max(0.0, t))
    n = len(stops) - 1
    k = min(int(t * n), n - 1)
    u = t * n - k
    a, b = stops[k], stops[k + 1]
    return tuple(a[i] + (b[i] - a[i]) * u for i in range(3))


# Approximations of Wolfram's default gradients (the exact Wolfram colour tables are not published).
DENSITY_GRADIENT = ((0.26, 0.15, 0.44), (0.27, 0.42, 0.71), (0.45, 0.69, 0.72), (0.86, 0.86, 0.51),
                    (0.96, 0.75, 0.37), (0.94, 0.49, 0.24))
MATRIX_NEG = ((0.95, 0.95, 0.95), (0.26, 0.47, 0.75), (0.07, 0.17, 0.45))
MATRIX_POS = ((0.95, 0.95, 0.95), (0.93, 0.58, 0.29), (0.70, 0.13, 0.11))
RELIEF_GRADIENT = ((0.20, 0.38, 0.28), (0.53, 0.66, 0.40), (0.85, 0.80, 0.55), (0.70, 0.55, 0.40),
                   (0.95, 0.95, 0.95))


def color_function(spec, fname, default):
    """ColorFunction setting -> Python f(t) -> (r, g, b). Automatic: ``default`` (a callable); a colour scheme
    head such as Hue/GrayLevel or a pure function is applied by the kernel to each value."""
    if spec is None or is_sym(spec, "Automatic"):
        return default
    if isinstance(spec, str):
        raise NotImplementedError(f"{fname}: the named colour scheme {spec!r} is not supported; use a function")
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and spec.head == "Function"):
        cache = {}

        def f(t):
            key = round(t, 6)
            if key not in cache:
                c = ev(Expr(spec, float(t)))
                if parse_color(c) is None:
                    raise ValueError(f"{fname}: ColorFunction gave {K_input(c)}, which is not a colour")
                r, g, b, _ = _rgba(c)
                cache[key] = (r, g, b)
            return cache[key]
        return f
    raise ValueError(f"{fname}: ColorFunction must be a function or Automatic, not {K_input(spec)}")


def _rgba(c):
    from .wl_graphics_color import to_rgba
    r = to_rgba(c)
    return tuple(float(v) for v in r)


def _raster(rows_rgb, rect):
    """Raster of RGB cells; ``rows_rgb`` is bottom row first."""
    return Expr("Raster", L(*[L(*[L(*[fl(round(v, 6)) for v in c]) for c in row]) for row in rows_rgb]),
                L(L(fl(rect[0][0]), fl(rect[0][1])), L(fl(rect[1][0]), fl(rect[1][1]))))


def matrix_cells(m, fname, allow_colors=True):
    """Array argument -> rows of cells: float, None (background) or an (r, g, b) colour; ragged rows padded."""
    if isinstance(m, Expr) and m.head == "SparseArray":
        m = ev(Expr("Normal", m))
    if not K.is_list(m):
        if isinstance(m, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the argument must be an array (a list of lists), not {K_input(m)}")
    rows = list(m.args)
    if not rows:
        raise ValueError(f"{fname}: the array is empty")
    if not all(K.is_list(r) for r in rows):
        if all(not K.is_list(r) for r in rows):
            rows = [m]                                    # a vector is one row
        else:
            raise ValueError(f"{fname}: the argument must be an array (a list of lists)")
    out = []
    for r in rows:
        row = []
        for c in r.args:
            if is_missing(c):
                row.append(None)
            elif allow_colors and parse_color(c) is not None:
                row.append(_rgba(c)[:3])
            elif K.is_list(c):
                raise ValueError(f"{fname}: array elements must be numbers, not {K_input(c)}")
            else:
                v = scalar(c, fname, "array element")
                row.append(v)
        out.append(row)
    width = max(len(r) for r in out)
    if width == 0:
        raise ValueError(f"{fname}: the array is empty")
    return [r + [None] * (width - len(r)) for r in out]


_ARRAY_DEFAULTS = {"PlotRange": AUTO, "ColorFunction": AUTO, "ColorFunctionScaling": True, "ColorRules": AUTO,
                   "DataRange": AUTO, "DataReversed": False, "Mesh": False, "MeshStyle": AUTO, "PlotLegends": NONE,
                   "ClippingStyle": AUTO, "MaxPlotPoints": AUTO, "PlotTheme": AUTO, "TargetUnits": AUTO,
                   "Method": AUTO, "PerformanceGoal": AUTO}


def _color_rules(spec, fname):
    if spec is None or is_sym(spec, "Automatic", "None"):
        return []
    if not K.is_list(spec) or not all(is_rule(r) for r in spec.args):
        raise NotImplementedError(f"{fname}: ColorRules -> {K_input(spec)} is not supported (give {{v -> colour, ...}})")
    out = []
    for r in spec.args:
        v = num(r.args[0])
        if v is None:
            raise NotImplementedError(f"{fname}: ColorRules patterns other than numbers are not supported")
        if parse_color(r.args[1]) is None:
            raise ValueError(f"{fname}: ColorRules {K_input(r)} does not give a colour")
        out.append((v, _rgba(r.args[1])[:3]))
    return out


def _array_rect(dr, nrows, ncols, fname):
    if dr is None or is_sym(dr, "All", "Automatic"):
        return ((0.0, 0.0), (float(ncols), float(nrows)))
    if K.is_list(dr) and len(dr.args) == 2 and all(K.is_list(a) and len(a.args) == 2 for a in dr.args):
        (x0, x1), (y0, y1) = [[real_arg(v, fname, "DataRange") for v in a.args] for a in dr.args]
        return ((x0, y0), (x1, y1))
    raise ValueError(f"{fname}: DataRange must be All or {{{{xmin, xmax}}, {{ymin, ymax}}}}, not {K_input(dr)}")


def array_plot(fname, m, opts, kind):
    o, gopts = split_options(fname, opts, _ARRAY_DEFAULTS, own_validation=("Mesh",))
    for n in ("MaxPlotPoints", "PlotTheme", "TargetUnits", "Method", "PerformanceGoal", "ClippingStyle",
              "MeshStyle"):
        v = o[n]
        if not (is_sym(v, "Automatic", "None", "Infinity") or (n == "PlotTheme" and v == "Default") or
                (n == "PerformanceGoal" and v in ("Quality", "Speed"))):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")
    if o["Mesh"] not in (False, True) and not is_sym(o["Mesh"], "None", "All", "Automatic"):
        raise NotImplementedError(f"{fname}: Mesh -> {K_input(o['Mesh'])} is not supported")
    if not is_sym(o["PlotLegends"], "None"):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    cells = matrix_cells(m, fname)
    if o["DataReversed"] is True:
        cells = cells[::-1]
    elif o["DataReversed"] is not False:
        raise NotImplementedError(f"{fname}: DataReversed -> {K_input(o['DataReversed'])} is not supported")
    nrows, ncols = len(cells), len(cells[0])
    vals = [v for r in cells for v in r if isinstance(v, float)]
    rules = _color_rules(o["ColorRules"], fname)
    pr = o["PlotRange"]
    scale = o["ColorFunctionScaling"]
    if scale not in (True, False):
        raise ValueError(f"{fname}: ColorFunctionScaling must be True or False")
    background = (1.0, 1.0, 1.0)
    explicit = None
    if not is_sym(pr, "Automatic", "All", "Full"):
        if K.is_list(pr) and len(pr.args) == 2 and all(num(a) is not None for a in pr.args):
            explicit = (num(pr.args[0]), num(pr.args[1]))
        elif num(pr) is not None and kind == "matrix":
            r = abs(num(pr))
            explicit = (-r, r)
        else:
            raise NotImplementedError(f"{fname}: PlotRange -> {K_input(pr)} is not supported (give {{min, max}})")
    if kind == "array":
        if explicit is None:
            mx = max([abs(v) for v in vals] or [1.0]) or 1.0

            def tscale(v):
                return abs(v) / mx
        else:
            lo, hi = explicit

            def tscale(v):
                return (v - lo) / (hi - lo) if hi > lo else 0.0
        if is_sym(o["ColorFunction"], "Automatic"):
            def cf(t):
                g = 1.0 - min(1.0, max(0.0, t))
                return (g, g, g)
        else:
            cf = color_function(o["ColorFunction"], fname, None)
            if explicit is None:
                lo_, hi_ = (min(vals), max(vals)) if vals else (0.0, 1.0)

                def tscale(v, lo_=lo_, hi_=hi_):
                    return (v - lo_) / (hi_ - lo_) if hi_ > lo_ else 0.0
    else:
        mx = max([abs(v) for v in vals] or [1.0]) or 1.0
        if explicit is not None:
            lo, hi = explicit
            mx = max(abs(lo), abs(hi)) or 1.0

        def tscale(v):
            return 0.5 + 0.5 * max(-1.0, min(1.0, v / mx))
        if is_sym(o["ColorFunction"], "Automatic"):
            def cf(t):
                return blend(MATRIX_NEG, (0.5 - t) * 2) if t < 0.5 else blend(MATRIX_POS, (t - 0.5) * 2)
        else:
            cf = color_function(o["ColorFunction"], fname, None)
    if scale is False:
        def tscale(v):
            return v

    def cell_color(v):
        if v is None:
            return background
        if isinstance(v, tuple):
            return v
        for rv, col in rules:
            if abs(rv - v) <= 1e-12 * max(1.0, abs(v)):
                return col
        if explicit is not None and not (explicit[0] <= v <= explicit[1]):
            return background if kind == "array" else cf(tscale(v))
        return cf(tscale(v))
    rows_rgb = [[cell_color(v) for v in r] for r in cells][::-1]     # bottom row first
    rect = _array_rect(o["DataRange"], nrows, ncols, fname)
    content = [_raster(rows_rgb, rect)]
    if o["Mesh"] is True or is_sym(o["Mesh"], "All"):
        (x0, y0), (x1, y1) = rect
        lines = [Expr("Line", pts([(x0 + (x1 - x0) * j / ncols, y0), (x0 + (x1 - x0) * j / ncols, y1)]))
                 for j in range(ncols + 1)]
        lines += [Expr("Line", pts([(x0, y0 + (y1 - y0) * i / nrows), (x1, y0 + (y1 - y0) * i / nrows)]))
                  for i in range(nrows + 1)]
        content.append(L(Expr("GrayLevel", 0.618034), *lines))
    (x0, y0), (x1, y1) = rect
    defaults = [("AspectRatio", Symbol("Automatic")), ("Frame", kind == "matrix"),
                ("PlotRange", range_expr([(x0, x1), (y0, y1)])), ("PlotRangePadding", 0)]
    if kind == "matrix":
        rt = [L(fl(y1 - (i - 0.5) * (y1 - y0) / nrows), i) for i in _tick_indices(nrows)]
        ct = [L(fl(x0 + (j - 0.5) * (x1 - x0) / ncols), j) for j in _tick_indices(ncols)]
        defaults.append(("FrameTicks", L(L(L(*rt), L(*rt)), L(L(*ct), L(*ct)))))
    return finish_graphics(L(*content), gopts, defaults)


def _tick_indices(n):
    if n <= 10:
        return list(range(1, n + 1))
    step = max(1, round(n / 5))
    return sorted({1, n} | set(range(step, n, step)))


@fn("ArrayPlot", area=AREA, status="added",
    notes="ArrayPlot[array]: Graphics[Raster[RGB cells, {{0, 0}, {ncols, nrows}}]] with row 1 at the top; by "
          "default 0 is white and the largest |value| black (GrayLevel 1 - |v|/max|v|, so 0/1 arrays are white/"
          "black); with PlotRange -> {min, max} min is white, max black and values outside are background; "
          "colour directives in the array are used as given, None is background (white); ragged rows are padded. "
          "Options: ColorFunction (a colour head or pure function of the scaled value), ColorFunctionScaling, "
          "ColorRules {v -> colour}, DataRange, DataReversed, Mesh -> True and all Graphics options; other "
          "settings raise NotImplementedError. Symbolic entries leave the call unevaluated. Check: reference "
          "(cell colours of 0/1 and graded arrays against the documented rule).")
def _arrayplot(m: MatrixOf[Real] | ListOf[Real], *opts: Option) -> Expr:
    pos, o = _list_args((m,) + opts, "ArrayPlot")
    return array_plot("ArrayPlot", pos[0], o, "array")


@fn("MatrixPlot", area=AREA, status="added",
    notes="MatrixPlot[m]: like ArrayPlot with a framed plot and row/column index ticks; zero is (near) white, "
          "negative values bluish and positive reddish, scaled by value/max|v| (Wolfram also mixes in the rank of "
          "each value; the colours here are an approximation of Wolfram's gradient); PlotRange -> r clips to "
          "[-r, r]. Large matrices are not downsampled (MaxPlotPoints other than Automatic/Infinity raises). "
          "Options as ArrayPlot. Check: reference (signs and ordering of the cell colours in tests).")
def _matrixplot(m: MatrixOf[Real], *opts: Option) -> Expr:
    pos, o = _list_args((m,) + opts, "MatrixPlot")
    return array_plot("MatrixPlot", pos[0], o, "matrix")


_RELIEF_DEFAULTS = {"PlotRange": AUTO, "ColorFunction": AUTO, "ColorFunctionScaling": True, "DataRange": AUTO,
                    "DataReversed": False, "LightingAngle": AUTO, "Mesh": False, "MeshStyle": AUTO,
                    "PlotLegends": NONE, "ClippingStyle": NONE, "MaxPlotPoints": AUTO, "Method": AUTO,
                    "PerformanceGoal": AUTO, "PlotTheme": AUTO, "BoxRatios": AUTO}


@fn("ReliefPlot", area=AREA, status="added",
    notes="ReliefPlot[array]: shaded relief of a rectangular array of heights, rows going up the page (row 1 at "
          "the bottom, unlike ArrayPlot), cell (i, j) centred at {j, i} (DataRange {{xmin, xmax}, {ymin, ymax}} "
          "rescales); colour = height gradient (low green to high white; approximates Wolfram's) times Lambert "
          "shading from a light at LightingAngle (default 3 Pi/4 from the upper left, 45 degrees elevation); "
          "non-numeric cells are holes (background). Options: ColorFunction, ColorFunctionScaling, DataRange, "
          "DataReversed, LightingAngle -> theta or {theta, phi} and Graphics options. Check: reference (a flat "
          "array is uniformly shaded; brighter colours on slopes facing the light, in tests).")
def _reliefplot(m: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "ReliefPlot"
    pos, o_ = _list_args((m,) + opts, fname)
    o, gopts = split_options(fname, o_, _RELIEF_DEFAULTS, own_validation=("Mesh",))
    for n in ("MaxPlotPoints", "Method", "PerformanceGoal", "PlotTheme", "ClippingStyle", "MeshStyle", "BoxRatios",
              "PlotRange"):
        v = o[n]
        if not (is_sym(v, "Automatic", "None", "All", "Infinity") or (n == "PlotTheme" and v == "Default") or
                (n == "PerformanceGoal" and v in ("Quality", "Speed"))):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")
    if o["Mesh"] is not False and not is_sym(o["Mesh"], "None"):
        raise NotImplementedError(f"{fname}: Mesh -> {K_input(o['Mesh'])} is not supported")
    if not is_sym(o["PlotLegends"], "None"):
        raise NotImplementedError(f"{fname}: PlotLegends is not supported")
    cells = matrix_cells(pos[0], fname, allow_colors=False)
    if len({len(r) for r in cells}) != 1 or any(len(r) == 0 for r in cells):
        raise ValueError(f"{fname}: the array must be rectangular")
    if o["DataReversed"] is True:
        cells = cells[::-1]
    np = _np()
    Z = np.array([[np.nan if v is None else v for v in r] for r in cells], dtype=float)
    la = o["LightingAngle"]
    if is_sym(la, "Automatic"):
        az, el = 3 * math.pi / 4, math.pi / 4
    elif num(la) is not None:
        az, el = num(la), math.pi / 4
    elif K.is_list(la) and len(la.args) == 2 and all(num(a) is not None for a in la.args):
        az, el = num(la.args[0]), num(la.args[1])
    else:
        raise ValueError(f"{fname}: LightingAngle must be Automatic, an angle or {{theta, phi}}")
    fin = np.isfinite(Z)
    if not fin.any():
        raise ValueError(f"{fname}: the array has no numeric heights")
    lo, hi = float(np.nanmin(Z)), float(np.nanmax(Z))
    span = hi - lo if hi > lo else 1.0
    Zf = np.where(fin, Z, lo)
    nr, nc = Z.shape
    dzdx = np.gradient(Zf, axis=1) if nc > 1 else np.zeros_like(Zf)
    dzdy = np.gradient(Zf, axis=0) if nr > 1 else np.zeros_like(Zf)
    rel = max(nr, nc) / span
    nx, ny, nz = -dzdx * rel, -dzdy * rel, np.ones_like(Zf)
    nn = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    lx, ly, lz = math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)
    shade = np.clip((nx * lx + ny * ly + nz * lz) / nn / lz, 0.0, 1.6)
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(RELIEF_GRADIENT, t))
    scale = o["ColorFunctionScaling"]
    rows = []
    for i in range(nr):                      # row 1 at the bottom: Raster's first row is the bottom row
        row = []
        for j in range(nc):
            if not fin[i, j]:
                row.append((1.0, 1.0, 1.0))
                continue
            t = (Z[i, j] - lo) / span if scale is not False else Z[i, j]
            c = cf(float(t))
            s = float(shade[i, j])
            row.append(tuple(min(1.0, max(0.0, ch * (0.55 + 0.45 * s))) for ch in c))
        rows.append(row)
    dr = o["DataRange"]
    if is_sym(dr, "Automatic", "All"):
        rect = ((0.5, 0.5), (nc + 0.5, nr + 0.5))
    else:
        (x0, y0), (x1, y1) = _array_rect(dr, nr, nc, fname)
        dx, dy = (x1 - x0) / max(nc - 1, 1) / 2, (y1 - y0) / max(nr - 1, 1) / 2
        rect = ((x0 - dx, y0 - dy), (x1 + dx, y1 + dy))
    (x0, y0), (x1, y1) = rect
    defaults = [("AspectRatio", Symbol("Automatic")), ("Frame", True), ("FrameTicks", NONE),
                ("PlotRange", range_expr([(x0, x1), (y0, y1)])), ("PlotRangePadding", 0)]
    return finish_graphics(L(_raster(rows, rect)), gopts, defaults)


# ───────────────────────────── triangle meshes (density, contour and region plots) ─────────────────────────────

def grid_mesh(a, b, c, d, nx, ny):
    """Vertices of an nx x ny grid over [a, b] x [c, d] and its triangles (two per cell)."""
    np = _np()
    xs = np.linspace(a, b, nx)
    ys = np.linspace(c, d, ny)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel()])
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            k = j * nx + i
            tris.append((k, k + 1, k + nx + 1))
            tris.append((k, k + nx + 1, k + nx))
    return P, np.array(tris, dtype=int), xs, ys


def clip_poly(poly, lo=None, hi=None):
    """Clip a polygon [(x, y, v), ...] (v linear along edges) to lo <= v <= hi (Sutherland-Hodgman)."""
    def clip(poly, inside, cross):
        out = []
        n = len(poly)
        for i in range(n):
            p, q = poly[i], poly[(i + 1) % n]
            ip, iq = inside(p), inside(q)
            if ip:
                out.append(p)
            if ip != iq:
                out.append(cross(p, q))
        return out

    def cross_at(level):
        def f(p, q):
            t = (level - p[-1]) / (q[-1] - p[-1]) if q[-1] != p[-1] else 0.5
            return tuple(p[k] + t * (q[k] - p[k]) for k in range(len(p) - 1)) + (level,)
        return f
    if lo is not None:
        poly = clip(poly, lambda p: p[-1] >= lo, cross_at(lo))
    if hi is not None and poly:
        poly = clip(poly, lambda p: p[-1] <= hi, cross_at(hi))
    return poly


def iso_segment(tri, level):
    """The segment where the linear interpolant of a triangle [(x, y, v)] * 3 equals ``level`` (or None)."""
    pts_ = []
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]
        if (p[-1] - level) * (q[-1] - level) < 0:
            t = (level - p[-1]) / (q[-1] - p[-1])
            pts_.append(tuple(p[k] + t * (q[k] - p[k]) for k in range(len(p) - 1)))
    return pts_ if len(pts_) == 2 else None


def bands(P, V, T, levels):
    """Polygons of the mesh for each band [levels[k], levels[k+1]] (with -inf/+inf at the ends)."""
    np = _np()
    edges = [-math.inf] + list(levels) + [math.inf]
    out = [[] for _ in range(len(edges) - 1)]
    for t in T:
        vs = V[t]
        if not np.isfinite(vs).all():
            continue
        tri = [(P[k][0], P[k][1], V[k]) for k in t]
        lo_v, hi_v = float(vs.min()), float(vs.max())
        for k in range(len(edges) - 1):
            lo, hi = edges[k], edges[k + 1]
            if hi_v < lo or lo_v > hi:
                continue
            if lo <= lo_v and hi_v <= hi:
                out[k].append([(p[0], p[1]) for p in tri])
            else:
                c = clip_poly(tri, None if lo == -math.inf else lo, None if hi == math.inf else hi)
                if len(c) >= 3:
                    out[k].append([(p[0], p[1]) for p in c])
    return out


def iso_lines(P, V, T, level):
    np = _np()
    segs = []
    for t in T:
        vs = V[t]
        if not np.isfinite(vs).all():
            continue
        if vs.min() < level < vs.max():
            s = iso_segment([(P[k][0], P[k][1], V[k]) for k in t], level)
            if s:
                segs.append(s)
    return chain_segments(segs)


def chain_segments(segs, ndigits=9):
    """Join segments that share end points into polylines."""
    def key(p):
        return tuple(round(c, ndigits) for c in p)
    from collections import defaultdict
    adj = defaultdict(list)
    for i, (p, q) in enumerate(segs):
        adj[key(p)].append(i)
        adj[key(q)].append(i)
    used = [False] * len(segs)
    lines = []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        line = [segs[i][0], segs[i][1]]
        for end in (1, 0):
            while True:
                tip = line[-1] if end == 1 else line[0]
                nxt = None
                for j in adj[key(tip)]:
                    if not used[j]:
                        nxt = j
                        break
                if nxt is None:
                    break
                used[nxt] = True
                p, q = segs[nxt]
                new = q if key(p) == key(tip) else p
                if end == 1:
                    line.append(new)
                else:
                    line.insert(0, new)
        lines.append(line)
    return lines


def multi_polygon(polys):
    if len(polys) == 1:
        return Expr("Polygon", pts(polys[0]))
    return Expr("Polygon", L(*[pts(p) for p in polys]))


def contour_levels(spec, lo, hi, fname):
    """Contours setting -> list of levels (Automatic: FindDivisions-style nice values strictly inside the range)."""
    if spec is None or is_sym(spec, "Automatic"):
        ts = PO.tick_positions(lo, hi, 10) if hi > lo else []
        return [t for t in ts if lo < t < hi] or ([(lo + hi) / 2] if hi > lo else [])
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec < 0:
            raise ValueError(f"{fname}: the number of contours must be non-negative")
        return [lo + (hi - lo) * (k + 1) / (spec + 1) for k in range(spec)]
    if K.is_list(spec):
        out = []
        for c in spec.args:
            if K.is_list(c) and c.args:
                c = c.args[0]
            v = num(c)
            if v is None:
                raise ValueError(f"{fname}: contour level {K_input(c)} is not a real number")
            out.append(v)
        return sorted(out)
    raise NotImplementedError(f"{fname}: Contours -> {K_input(spec)} is not supported")


def density_complex(P, V, T, cf, vlo, vhi, scale=True):
    """GraphicsComplex with the mesh triangles and per-vertex colours cf((v - vlo)/(vhi - vlo))."""
    np = _np()
    ok = np.isfinite(V)
    tris = [t for t in T if ok[t].all()]
    span = (vhi - vlo) if vhi > vlo else 1.0
    cols = [rgb(cf((float(v) - vlo) / span if scale else float(v))) if math.isfinite(v) else rgb((1.0, 1.0, 1.0))
            for v in V]
    poly = Expr("Polygon", L(*[L(*[int(k) + 1 for k in t]) for t in tris]))
    return Expr("GraphicsComplex", pts([tuple(p) for p in P]), L(Expr("EdgeForm"), poly),
                Expr("Rule", Symbol("VertexColors"), L(*cols)))


def two_iterators(pos, fname):
    if len(pos) != 3:
        raise ValueError(f"{fname}: expected {fname}[f, {{x, xmin, xmax}}, {{y, ymin, ymax}}, options]")
    f, sx, sy = pos
    x, a, b = iterator(sx, fname, 2)
    y, c, d = iterator(sy, fname, 3)
    if x.name == y.name:
        raise ValueError(f"{fname}: the two iterator variables must differ")
    return f, x, a, b, y, c, d


_FIELD_DEFAULTS = {"PlotRange": AUTO, "PlotPoints": AUTO, "MaxRecursion": AUTO, "ColorFunction": AUTO,
                   "ColorFunctionScaling": True, "PlotLegends": NONE, "Mesh": NONE, "MeshStyle": AUTO,
                   "MeshFunctions": AUTO, "Exclusions": AUTO, "ExclusionsStyle": NONE, "RegionFunction": AUTO,
                   "PlotTheme": AUTO, "PerformanceGoal": AUTO, "WorkingPrecision": Symbol("MachinePrecision"),
                   "EvaluationMonitor": NONE, "Method": AUTO, "ClippingStyle": NONE, "ScalingFunctions": NONE,
                   "BoundaryStyle": AUTO, "PlotStyle": AUTO, "Contours": AUTO, "ContourStyle": AUTO,
                   "ContourShading": AUTO, "ContourLabels": AUTO, "InterpolationOrder": AUTO,
                   "DataRange": AUTO, "MaxPlotPoints": AUTO, "PlotRangePadding": AUTO, "PlotHighlighting": AUTO,
                   "TextureCoordinateFunction": AUTO, "TextureCoordinateScaling": AUTO}


def _check_field_unsupported(o, fname, allowed):
    for n, v in o.items():
        if n in allowed or n == "PlotRange":
            continue
        if is_sym(v, "Automatic", "None", "Infinity") or (n == "WorkingPrecision" and is_sym(v, "MachinePrecision")) \
                or (n == "PlotTheme" and v == "Default") or (n == "PerformanceGoal" and v in ("Quality", "Speed")) \
                or (n == "ColorFunctionScaling" and v is True):
            continue
        raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


def _grid_n(o, fname, default):
    n = _int_opt(o["PlotPoints"], default, fname, "PlotPoints", 2)
    rec = _int_opt(o["MaxRecursion"], 1, fname, "MaxRecursion", 0)
    n = (n - 1) * (2 ** min(rec, 2)) + 1
    return min(n, 401)


@fn("DensityPlot", area=AREA, status="added", hold=True,
    notes="DensityPlot[f, {x, xmin, xmax}, {y, ymin, ymax}] (HoldAll): Graphics[GraphicsComplex[grid points, "
          "{EdgeForm[], Polygon[triangles]}, VertexColors -> colours]] with colours ColorFunction((f - fmin)/"
          "(fmax - fmin)) (default gradient dark blue - teal - yellow - orange, an approximation of Wolfram's); "
          "grid of (PlotPoints - 1) 2^min(MaxRecursion, 2) + 1 points per side (default PlotPoints 25, "
          "MaxRecursion 1: 49 x 49); undefined/non-real values leave holes. Options: PlotPoints, MaxRecursion, "
          "ColorFunction (colour head or pure function), ColorFunctionScaling, PlotRange (clips the colour "
          "scale) and Graphics options; PlotLegends, Mesh, RegionFunction and other settings raise "
          "NotImplementedError. Free parameters leave the call unevaluated. Check: numeric cross-check (compiled "
          "vs kernel N) and reference (tests compare vertex values with f).")
def _densityplot(f: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "DensityPlot"
    pos, o_ = hold_args((f, sx, sy) + opts)
    f, x, a, b, y, c, d = two_iterators(pos, fname)
    o, gopts = split_options(fname, o_, _FIELD_DEFAULTS)
    _check_field_unsupported(o, fname, ("PlotPoints", "MaxRecursion", "ColorFunction", "ColorFunctionScaling"))
    n = _grid_n(o, fname, 25)
    comp = Compiled(f, [x, y], fname)
    symbolic_check(comp, [x, y], fname)
    comp.cross_check([(a + (b - a) * 0.31, c + (d - c) * 0.67), (a + (b - a) * 0.77, c + (d - c) * 0.21)])
    P, T, _, _ = grid_mesh(a, b, c, d, n, n)
    V = comp(P[:, 0], P[:, 1])
    fin = V[_np().isfinite(V)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    pr = o["PlotRange"]
    if not is_sym(pr, "Automatic", "All", "Full"):
        rr = resolve_plot_range(pr, fname, [(a, b), (c, d), (vlo, vhi)], dims=3)
        vlo, vhi = rr[2]
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    g = density_complex(P, V, T, cf, vlo, vhi, o["ColorFunctionScaling"] is not False)
    defaults = [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr([(a, b), (c, d)])),
                ("PlotRangePadding", 0)]
    return finish_graphics(L(g), gopts, defaults)


def _level_function(pred, fname):
    """Inequality predicate -> held expression g with g >= 0 exactly where pred holds (strictness ignored, as the
    boundary has no area), or None if pred is not a combination of inequalities."""
    if isinstance(pred, Expr):
        h = pred.head
        if h in ("Greater", "GreaterEqual", "Less", "LessEqual") and len(pred.args) >= 2:
            parts = []
            for p_, q_ in zip(pred.args, pred.args[1:]):
                if h in ("Greater", "GreaterEqual"):
                    parts.append(Expr("Plus", p_, Expr("Times", -1, q_)))
                else:
                    parts.append(Expr("Plus", q_, Expr("Times", -1, p_)))
            return parts[0] if len(parts) == 1 else Expr("Min", *parts)
        if h == "Inequality":
            raise NotImplementedError(f"{fname}: Inequality[...] predicates are not supported; use a < x < b")
        if h == "And":
            gs = [_level_function(a, fname) for a in pred.args]
            return None if any(g is None for g in gs) else Expr("Min", *gs)
        if h == "Or":
            gs = [_level_function(a, fname) for a in pred.args]
            return None if any(g is None for g in gs) else Expr("Max", *gs)
        if h == "Not" and len(pred.args) == 1:
            g = _level_function(pred.args[0], fname)
            return None if g is None else Expr("Times", -1, g)
    return None


_REGION_DEFAULTS = dict(_FIELD_DEFAULTS)


@fn("RegionPlot", area=AREA, status="added", hold=True,
    notes="RegionPlot[pred, {x, xmin, xmax}, {y, ymin, ymax}] and lists of predicates (HoldAll): filled polygons "
          "of the region where pred is True. Predicates made of <, <=, >, >= (chains too), And, Or and Not are "
          "turned into a level function g (min/max of lhs - rhs) whose zero set is interpolated linearly on a "
          "grid of triangles (default 81 x 81: PlotPoints 41, MaxRecursion 1), so boundaries are accurate to the "
          "grid; other Boolean predicates are sampled as True/False with the boundary at the cell midpoints. "
          "Output: {Directive[Opacity[0.3], colour], Polygon[{...}], boundary Line} per predicate. Options: "
          "PlotPoints, MaxRecursion, PlotStyle, BoundaryStyle (None or a directive), PlotLegends and Graphics "
          "options. Equations (==) and other settings raise NotImplementedError; free parameters leave the call "
          "unevaluated. Check: numeric cross-check and reference (tests check polygon vertices satisfy pred and "
          "the filled area against the exact area).")
def _regionplot(pred: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "RegionPlot"
    pos, o_ = hold_args((pred, sx, sy) + opts)
    pred, x, a, b, y, c, d = two_iterators(pos, fname)
    o, gopts = split_options(fname, o_, _REGION_DEFAULTS)
    _check_field_unsupported(o, fname, ("PlotPoints", "MaxRecursion", "PlotStyle", "BoundaryStyle", "PlotLegends"))
    n = _grid_n(o, fname, 41)
    preds, _ = _curves_of(pred, x)
    styles = style_list(o["PlotStyle"], len(preds), fname)
    content = []
    np = _np()
    P, T, _, _ = grid_mesh(a, b, c, d, n, n)
    for i, pr in enumerate(preds):
        ev_pred = pr
        if isinstance(pr, Expr) and pr.head == "Equal":
            raise NotImplementedError(f"{fname}: equations have no area; use ContourPlot for curves")
        g = _level_function(ev_pred, fname)
        if g is None:
            g = Expr("Plus", Expr("Times", 2, Expr("Boole", ev_pred)), -1)
        comp = Compiled(g, [x, y], fname)
        symbolic_check(comp, [x, y], fname)
        comp.cross_check([(a + (b - a) * 0.31, c + (d - c) * 0.67), (a + (b - a) * 0.77, c + (d - c) * 0.21)])
        V = comp(P[:, 0], P[:, 1])
        if not np.isfinite(V).any():
            v0 = comp.point(a + (b - a) * 0.37, c + (d - c) * 0.41)
            if math.isnan(v0):
                pass
        polys = region_polygons(P, V, T, n, n)
        col = style_color(styles[i]) if styles[i] is not None else None
        if styles[i] is None:
            continue
        fill_style = Expr("Directive", Expr("Opacity", 0.3), col) if is_sym(o["PlotStyle"], "Automatic") else \
            styles[i]
        grp = [fill_style, Expr("EdgeForm")]
        if polys:
            grp.append(multi_polygon(polys))
        bs = o["BoundaryStyle"]
        if not is_sym(bs, "None"):
            lines = iso_lines(P, V, T, 0.0)
            if lines:
                bstyle = col if is_sym(bs, "Automatic") else bs
                if not is_sym(bs, "Automatic"):
                    G.directive_like(bs, fname, "BoundaryStyle")
                grp.append(L(bstyle, Expr("Line", L(*[pts(ln) for ln in lines]))))
        content.append(L(*grp))
    defaults = [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr([(a, b), (c, d)])),
                ("PlotRangePadding", 0)]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], [Expr("Directive", Expr("Opacity", 0.3), style_color(s)) for s in styles],
                       [Expr("HoldForm", p) for p in preds], fname, "swatch")


def region_polygons(P, V, T, nx, ny):
    """Polygons of {v >= 0}: whole grid cells merged into row runs, boundary cells clipped per triangle."""
    np = _np()
    out = []
    Vg = V.reshape(ny, nx)
    inside_v = np.isfinite(Vg) & (Vg >= 0)
    cell_in = inside_v[:-1, :-1] & inside_v[:-1, 1:] & inside_v[1:, :-1] & inside_v[1:, 1:]
    xs = P[:nx, 0]
    ys = P[::nx, 1]
    for j in range(ny - 1):
        i = 0
        while i < nx - 1:
            if cell_in[j, i]:
                k = i
                while k + 1 < nx - 1 and cell_in[j, k + 1]:
                    k += 1
                out.append([(xs[i], ys[j]), (xs[k + 1], ys[j]), (xs[k + 1], ys[j + 1]), (xs[i], ys[j + 1])])
                i = k + 1
            else:
                i += 1
    for idx, t in enumerate(T):
        j, i = divmod(idx // 2, nx - 1)
        if cell_in[j, i]:
            continue
        vs = V[t]
        if not np.isfinite(vs).all() or vs.max() < 0:
            continue
        c = clip_poly([(P[k][0], P[k][1], V[k]) for k in t], 0.0, None)
        if len(c) >= 3:
            out.append([(p[0], p[1]) for p in c])
    return out


# ───────────────────────────── list density / contour plots ─────────────────────────────

def field_data(data, fname, dr):
    """Array {{z11, ...}, ...} (z_ij at x = j, y = i, or DataRange) or {{x, y, z}, ...} -> (P, V, T, ranges)."""
    np = _np()
    data, _, _ = _unwrap(data, fname)
    if not K.is_list(data) or not data.args:
        if isinstance(data, (Symbol, Expr)) and not K.is_list(data):
            raise Unevaluated()
        raise ValueError(f"{fname}: the data must be a non-empty array or a list of {{x, y, z}} triples")
    rows = data.args
    if all(_is_pair(r, 3) for r in rows) and len(rows) > 3:
        pts3 = []
        for r in rows:
            cs = [scalar(c, fname) for c in r.args]
            if cs[0] is None or cs[1] is None:
                continue
            pts3.append((cs[0], cs[1], math.nan if cs[2] is None else cs[2]))
        if len(pts3) < 3:
            raise ValueError(f"{fname}: at least 3 points {{x, y, z}} are needed")
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            raise NotImplementedError(f"{fname}: irregular {{x, y, z}} data needs scipy (pip install scipy)")
        P = np.array([(p[0], p[1]) for p in pts3], dtype=float)
        V = np.array([p[2] for p in pts3], dtype=float)
        try:
            T = Delaunay(P).simplices
        except Exception as exc:
            raise ValueError(f"{fname}: the points cannot be triangulated ({exc.__class__.__name__})")
        xr = (float(P[:, 0].min()), float(P[:, 0].max()))
        yr = (float(P[:, 1].min()), float(P[:, 1].max()))
        return P, V, np.array(T, dtype=int), xr, yr
    cells = matrix_cells(data, fname, allow_colors=False)
    if len({len(r) for r in cells}) != 1:
        raise ValueError(f"{fname}: the array must be rectangular")
    ny, nx = len(cells), len(cells[0])
    if nx < 2 or ny < 2:
        raise ValueError(f"{fname}: the array must be at least 2 x 2")
    if dr is None or is_sym(dr, "Automatic", "All"):
        a, b, c, d = 1.0, float(nx), 1.0, float(ny)
    elif K.is_list(dr) and len(dr.args) == 2 and all(K.is_list(q) and len(q.args) == 2 for q in dr.args):
        (a, b), (c, d) = [[real_arg(v, fname, "DataRange") for v in q.args] for q in dr.args]
    else:
        raise ValueError(f"{fname}: DataRange must be Automatic or {{{{xmin, xmax}}, {{ymin, ymax}}}}")
    P, T, _, _ = grid_mesh(a, b, c, d, nx, ny)
    V = np.array([math.nan if v is None else v for r in cells for v in r], dtype=float)
    return P, V, T, (a, b), (c, d)


@fn("ListDensityPlot", area=AREA, status="added",
    notes="ListDensityPlot[array] (z_ij at {j, i}, row 1 at the bottom, or DataRange {{xmin, xmax}, {ymin, "
          "ymax}}) and ListDensityPlot[{{x, y, z}, ...}] (Delaunay triangulation; needs scipy; an n x 3 list is read as "
          "triples when n > 3, as Wolfram's documented forms are ambiguous for 3 x 3 arrays): "
          "Graphics[GraphicsComplex[points, {EdgeForm[], Polygon[triangles]}, VertexColors -> colours]], colours "
          "interpolated linearly between data points (default gradient as DensityPlot). Missing values leave "
          "holes. Options: ColorFunction, ColorFunctionScaling, DataRange, PlotRange (colour clipping) and "
          "Graphics options; others raise NotImplementedError. Check: reference (vertex coordinates and colour "
          "ordering against the data in tests).")
def _listdensityplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "ListDensityPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, _FIELD_DEFAULTS)
    _check_field_unsupported(o, fname, ("ColorFunction", "ColorFunctionScaling", "DataRange", "InterpolationOrder"))
    io = o["InterpolationOrder"]
    if not (is_sym(io, "Automatic") or io == 1):
        raise NotImplementedError(f"{fname}: InterpolationOrder -> {K_input(io)} is not supported (1)")
    P, V, T, xr, yr = field_data(pos[0], fname, o["DataRange"])
    np = _np()
    fin = V[np.isfinite(V)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    pr = o["PlotRange"]
    if not is_sym(pr, "Automatic", "All", "Full"):
        vlo, vhi = resolve_plot_range(pr, fname, [xr, yr, (vlo, vhi)], dims=3)[2]
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    g = density_complex(P, V, T, cf, vlo, vhi, o["ColorFunctionScaling"] is not False)
    defaults = [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr([xr, yr])), ("PlotRangePadding", 0)]
    return finish_graphics(L(g), gopts, defaults)


def _contour_style(spec, k, fname):
    if spec is None or is_sym(spec, "Automatic"):
        return Expr("Directive", Expr("Opacity", 0.5), Expr("GrayLevel", 0))
    if is_sym(spec, "None"):
        return None
    if K.is_list(spec):
        if not spec.args:
            return None
        s = spec.args[k % len(spec.args)]
        return None if is_sym(s, "None") else s
    G.directive_like(spec, fname, "ContourStyle")
    return spec


@fn("ListContourPlot", area=AREA, status="added",
    notes="ListContourPlot[array] / ListContourPlot[{{x, y, z}, ...}] (data forms as ListDensityPlot): filled "
          "bands between contour levels plus the contour lines, from linear interpolation on the data triangles. "
          "Contours: Automatic (FindDivisions-style nice levels inside the data range), n (n equally spaced "
          "levels), or {c1, ...}. Output: {{EdgeForm[], colour_k, Polygon[{...}]} per band, {ContourStyle, "
          "Line[...]} per level}. Options: Contours, ContourShading (Automatic gradient, None, or a list of "
          "colours), ContourStyle (Automatic, None, directive or list), ColorFunction, DataRange and Graphics "
          "options; others raise NotImplementedError. Check: reference (every contour vertex interpolates to its "
          "level; tests compare levels and vertices with the data).")
def _listcontourplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "ListContourPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, _FIELD_DEFAULTS)
    _check_field_unsupported(o, fname, ("Contours", "ContourShading", "ContourStyle", "ColorFunction",
                                        "ColorFunctionScaling", "DataRange", "InterpolationOrder"))
    P, V, T, xr, yr = field_data(pos[0], fname, o["DataRange"])
    return contour_graphics(fname, P, V, T, xr, yr, o, gopts)


def contour_graphics(fname, P, V, T, xr, yr, o, gopts):
    np = _np()
    fin = V[np.isfinite(V)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    levels = contour_levels(o["Contours"], vlo, vhi, fname)
    content = []
    shading = o["ContourShading"]
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    if not is_sym(shading, "None") and shading is not False:
        bs = bands(P, V, T, levels)
        nb = len(bs)
        for k, polys in enumerate(bs):
            if not polys:
                continue
            if K.is_list(shading) and shading.args:
                col = shading.args[k % len(shading.args)]
                if is_sym(col, "None"):
                    continue
            elif is_sym(shading, "Automatic") or shading is True:
                lo = vlo if k == 0 else levels[k - 1]
                hi = vhi if k == nb - 1 else levels[k]
                mid = ((lo + hi) / 2 - vlo) / (vhi - vlo) if vhi > vlo else 0.5
                col = rgb(cf(mid))
            else:
                raise NotImplementedError(f"{fname}: ContourShading -> {K_input(shading)} is not supported")
            content.append(L(Expr("EdgeForm"), col, multi_polygon(polys)))
    for k, c in enumerate(levels):
        st = _contour_style(o["ContourStyle"], k, fname)
        if st is None:
            continue
        lines = iso_lines(P, V, T, c)
        if lines:
            content.append(L(st, Expr("Line", L(*[pts(ln) for ln in lines]))))
    defaults = [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr([xr, yr])), ("PlotRangePadding", 0)]
    return finish_graphics(L(*content), gopts, defaults)


# ───────────────────────────── statistical plots ─────────────────────────────

def is_dist(x):
    return isinstance(x, Expr) and isinstance(x.head, str) and x.head.endswith("Distribution") and \
        x.head not in ("EmpiricalDistribution",)


def sample_values(d, fname):
    """Sorted finite floats of a dataset (missing values dropped); symbolic entries -> Unevaluated."""
    d, _, _ = _unwrap(d, fname)
    if isinstance(d, Expr) and d.head == "Association":
        d = L(*[v for _, v in _assoc_items(d)])
    if isinstance(d, Expr) and d.head in _UNSUPPORTED_DATA:
        raise NotImplementedError(f"{fname}: {d.head} data is not supported")
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of real numbers, not {K_input(d)}")
    vals = []
    for x in d.args:
        if K.is_list(x):
            raise ValueError(f"{fname}: the data must be a flat list of real numbers")
        v = scalar(x, fname)
        if v is not None:
            vals.append(v)
    if not vals:
        raise ValueError(f"{fname}: the data has no numeric values")
    return sorted(vals)


def _num_or_raise(v, fname, what):
    x = num(v)
    if x is None:
        if isinstance(v, Expr) and v.head == "DirectedInfinity" or is_sym(v, "Infinity"):
            return math.inf
        raise NotImplementedError(f"{fname}: could not evaluate the {what} numerically ({K_input(v)})")
    return x


def dist_quantiles(dist, ps, fname):
    return [_num_or_raise(ev(Expr("N", Expr("InverseCDF", dist, float(p)))), fname, "quantile") for p in ps]


def dist_cdf(dist, xs, fname):
    return [_num_or_raise(ev(Expr("N", Expr("CDF", dist, float(x)))), fname, "CDF") for x in xs]


def _has_free_symbols(e):
    return _has_user_symbol(e)


def resolve_reference(ref, data_vals, fname):
    """A reference distribution: numeric parameters as given, symbolic parameters estimated from the data
    (EstimatedDistribution, as Wolfram documents)."""
    if not is_dist(ref):
        raise TypeError(f"{fname}: the reference must be a distribution or a dataset, not {K_input(ref)}")
    if _has_free_symbols(ref):
        if data_vals is None:
            raise Unevaluated()
        est = ev(Expr("EstimatedDistribution", L(*data_vals), ref))
        if not is_dist(est) or _has_free_symbols(est):
            raise NotImplementedError(f"{fname}: could not estimate {K_input(ref)} from the data")
        return est
    return ref


def normal_fit(vals):
    """NormalDistribution[mean, ML standard deviation] (EstimatedDistribution's maximum-likelihood estimate)."""
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / n)
    if s == 0:
        s = 1.0
    return Expr("NormalDistribution", fl(m), fl(s))


def empirical_quantile(sorted_vals, q):
    """Wolfram's default Quantile[data, q] = x_(Ceiling[n q]) (x_(1) for q = 0)."""
    n = len(sorted_vals)
    k = max(1, min(n, math.ceil(n * q - 1e-12)))
    return sorted_vals[k - 1]


_STAT_DEFAULTS = {"PlotRange": AUTO, "PlotStyle": AUTO, "Joined": AUTO, "Filling": NONE, "FillingStyle": AUTO,
                  "PlotLegends": NONE, "PlotMarkers": NONE, "ReferenceLineStyle": AUTO, "ScalingFunctions": NONE,
                  "ClippingStyle": AUTO, "PlotTheme": AUTO, "Mesh": NONE, "ColorFunction": AUTO,
                  "PlotHighlighting": AUTO, "Method": AUTO, "PerformanceGoal": AUTO,
                  "WorkingPrecision": Symbol("MachinePrecision")}


def _stat_check(o, fname):
    for n in ("ScalingFunctions", "ClippingStyle", "Mesh", "ColorFunction", "PlotHighlighting", "Method",
              "PerformanceGoal", "PlotTheme", "PlotMarkers", "WorkingPrecision", "Filling"):
        v = o[n]
        if is_sym(v, "Automatic", "None", "MachinePrecision") or (n == "PlotTheme" and v == "Default") or \
                (n == "PerformanceGoal" and v in ("Quality", "Speed")):
            continue
        raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


def _dotted(st):
    return Expr("Directive", *(st.args if st is not None else [Expr("GrayLevel", 0)]),
                Expr("Dashing", L(0.005, 0.01)))


def _stat_inputs(args, fname):
    pos, opts = split_trailing_options(args)
    if not 1 <= len(pos) <= 2:
        raise ValueError(f"{fname}: expected {fname}[data] or {fname}[data, reference]")
    d = pos[0]
    du, _, _ = _unwrap(d, fname)
    if is_dist(du):
        items = [du]
    elif K.is_list(du) and du.args and all(is_dist(_unwrap(x, fname)[0]) or K.is_list(_unwrap(x, fname)[0])
                                           for x in du.args):
        items = [_unwrap(x, fname)[0] for x in du.args]
    else:
        items = [du]
    return items, (pos[1] if len(pos) == 2 else None), opts


def _stat_graphics(fname, curves, joined, styles, o, gopts, ref_lines, labels, prange_data, extra_defaults=()):
    content = []
    rstyle = o["ReferenceLineStyle"]
    for i, (seg, st) in enumerate(zip(ref_lines, styles)):
        if seg is None or is_sym(rstyle, "None") or st is None:
            continue
        rs = _dotted(st) if is_sym(rstyle, "Automatic") else rstyle
        if not is_sym(rstyle, "Automatic"):
            G.directive_like(rstyle, fname, "ReferenceLineStyle")
        content.append(L(rs, Expr("Line", pts(seg))))
    for cv, j, st in zip(curves, joined, styles):
        if st is None or not cv:
            continue
        content.append(L(st, Expr("Line", pts(cv)) if j else Expr("Point", pts(cv))))
    allp = [p for cv in curves for p in cv]
    xr = prange_data[0] or data_range([p[0] for p in allp])
    yr = prange_data[1] or data_range([p[1] for p in allp])
    prange = resolve_plot_range(o["PlotRange"], fname, [xr, yr], auto=[xr, yr])
    defaults = [("AspectRatio", GOLDEN), ("Frame", True), ("Axes", L(False, False)), *extra_defaults,
                ("PlotRange", range_expr(prange)), ("PlotRangePadding", L(L(Expr("Scaled", 0.02), Expr("Scaled", 0.02)),
                                                                          L(Expr("Scaled", 0.05),
                                                                            Expr("Scaled", 0.05))))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, labels, fname, "point")


def _joined(o, n, defaults, fname):
    j = o["Joined"]
    if is_sym(j, "Automatic"):
        return defaults
    if j is True or j is False:
        return [j] * n
    raise ValueError(f"{fname}: Joined must be True, False or Automatic")


_QP_PS = [0.005 + 0.99 * k / 120 for k in range(121)]


@fn("QuantilePlot", area=AREA, status="added",
    notes="QuantilePlot[data], QuantilePlot[dist], QuantilePlot[data, ref] and QuantilePlot[{data1, ...}, ref]: "
          "Q-Q plot, points {Quantile[ref, p_i], x_(i)} with Hazen plotting positions p_i = (i - 1/2)/n (Wolfram "
          "does not document its positions); the default reference is the maximum-likelihood NormalDistribution "
          "of each dataset (EstimatedDistribution); a reference with symbolic parameters is estimated from the "
          "data; a reference dataset uses Wolfram's default Quantile (x_(Ceiling[n p])); distributions are drawn "
          "as joined curves (quantiles at p from 0.005 to 0.995) against the normal with the same mean and "
          "standard deviation; reference line through the first and third quartiles (ReferenceLineStyle, "
          "Automatic = dotted plot colour, None). Frame -> True. Options: PlotStyle, Joined, PlotLegends, "
          "PlotRange, ReferenceLineStyle and Graphics options; others raise NotImplementedError. Check: "
          "reference (quantiles from InverseCDF; data from a normal sample lie near the line in tests).")
def _quantileplot(data: ListOf[Real] | Expr, *rest: Expr | Option) -> Expr:
    fname = "QuantilePlot"
    items, ref, opts = _stat_inputs((data,) + rest, fname)
    o, gopts = split_options(fname, opts, _STAT_DEFAULTS)
    _stat_check(o, fname)
    curves, refl, is_d = [], [], []
    for it in items:
        if is_dist(it):
            if _has_free_symbols(it):
                raise Unevaluated()
            ps = _QP_PS
            ys = dist_quantiles(it, ps, fname)
            if ref is None:
                m = _num_or_raise(ev(Expr("N", Expr("Mean", it))), fname, "mean")
                s = _num_or_raise(ev(Expr("N", Expr("StandardDeviation", it))), fname, "standard deviation")
                if not (math.isfinite(m) and math.isfinite(s)) or s <= 0:
                    raise NotImplementedError(f"{fname}: {K_input(it)} has no finite mean and variance; give a "
                                              "reference distribution")
                rdist = Expr("NormalDistribution", m, s)
                xs = dist_quantiles(rdist, ps, fname)
            elif is_dist(ref):
                xs = dist_quantiles(resolve_reference(ref, None, fname), ps, fname)
            else:
                rv = sample_values(ref, fname)
                xs = [empirical_quantile(rv, p) for p in ps]
            curve = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
            q1, q3 = (dist_quantiles(it, [0.25, 0.75], fname))
            is_d.append(True)
        else:
            vals = sample_values(it, fname)
            n = len(vals)
            ps = [(i - 0.5) / n for i in range(1, n + 1)]
            if ref is None:
                rdist = normal_fit(vals)
                xs = dist_quantiles(rdist, ps, fname)
                rq = dist_quantiles(rdist, [0.25, 0.75], fname)
            elif is_dist(_unwrap(ref, fname)[0]):
                rdist = resolve_reference(_unwrap(ref, fname)[0], vals, fname)
                xs = dist_quantiles(rdist, ps, fname)
                rq = dist_quantiles(rdist, [0.25, 0.75], fname)
            else:
                rv = sample_values(ref, fname)
                xs = [empirical_quantile(rv, p) for p in ps]
                rq = [empirical_quantile(rv, 0.25), empirical_quantile(rv, 0.75)]
            curve = [(x, y) for x, y in zip(xs, vals) if math.isfinite(x)]
            q1, q3 = empirical_quantile(vals, 0.25), empirical_quantile(vals, 0.75)
            is_d.append(False)
        curves.append(curve)
        xs_c = [p[0] for p in curve]
        if is_d[-1]:
            refl.append(None if not curve else _line_through(curve, q1, q3, it, ref, fname))
        else:
            if rq[1] != rq[0]:
                slope = (q3 - q1) / (rq[1] - rq[0])
                x0, x1 = min(xs_c), max(xs_c)
                refl.append([(x0, q1 + slope * (x0 - rq[0])), (x1, q1 + slope * (x1 - rq[0]))])
            else:
                refl.append(None)
    styles = style_list(o["PlotStyle"], len(curves), fname, "point")
    joined = _joined(o, len(curves), is_d, fname)
    labels = [f"{i + 1}" for i in range(len(curves))] if len(curves) > 1 else None
    return _stat_graphics(fname, curves, joined, styles, o, gopts, refl, labels, (None, None))


def _line_through(curve, q1, q3, dist, ref, fname):
    """Reference line for a distribution curve: through the points at p = 1/4 and 3/4."""
    xs = [p[0] for p in curve]
    if ref is None:
        m = _num_or_raise(ev(Expr("N", Expr("Mean", dist))), fname, "mean")
        s = _num_or_raise(ev(Expr("N", Expr("StandardDeviation", dist))), fname, "standard deviation")
        r1, r3 = dist_quantiles(Expr("NormalDistribution", m, s), [0.25, 0.75], fname)
    elif is_dist(ref):
        r1, r3 = dist_quantiles(resolve_reference(ref, None, fname), [0.25, 0.75], fname)
    else:
        rv = sample_values(ref, fname)
        r1, r3 = empirical_quantile(rv, 0.25), empirical_quantile(rv, 0.75)
    if r3 == r1:
        return None
    slope = (q3 - q1) / (r3 - r1)
    x0, x1 = min(xs), max(xs)
    return [(x0, q1 + slope * (x0 - r1)), (x1, q1 + slope * (x1 - r1))]


@fn("ProbabilityPlot", area=AREA, status="added",
    notes="ProbabilityPlot[data], ProbabilityPlot[dist], ProbabilityPlot[data, ref], ProbabilityPlot[{data1, ...}, "
          "ref]: P-P plot, points {CDF[ref, x_(i)], i/n} (empirical CDF of the data against the reference CDF); "
          "the default reference is the maximum-likelihood NormalDistribution; symbolic reference parameters "
          "are estimated; a reference dataset uses its empirical CDF; a distribution is drawn as the joined "
          "curve {CDF[ref, x], CDF[dist, x]} over its 0.5% - 99.5% quantile range; diagonal reference line "
          "(ReferenceLineStyle). PlotRange {{0, 1}, {0, 1}}. Options as QuantilePlot. Check: reference (CDF "
          "values from the kernel; a dataset compared with its own distribution lies near the diagonal).")
def _probabilityplot(data: ListOf[Real] | Expr, *rest: Expr | Option) -> Expr:
    fname = "ProbabilityPlot"
    items, ref, opts = _stat_inputs((data,) + rest, fname)
    o, gopts = split_options(fname, opts, _STAT_DEFAULTS)
    _stat_check(o, fname)
    curves, is_d = [], []
    for it in items:
        if is_dist(it):
            if _has_free_symbols(it):
                raise Unevaluated()
            xs = dist_quantiles(it, _QP_PS, fname)
            xs = [x for x in xs if math.isfinite(x)]
            ys = dist_cdf(it, xs, fname)
            if ref is None:
                m = _num_or_raise(ev(Expr("N", Expr("Mean", it))), fname, "mean")
                s = _num_or_raise(ev(Expr("N", Expr("StandardDeviation", it))), fname, "standard deviation")
                if not (math.isfinite(m) and math.isfinite(s)) or s <= 0:
                    raise NotImplementedError(f"{fname}: {K_input(it)} has no finite mean and variance; give a "
                                              "reference distribution")
                rx = dist_cdf(Expr("NormalDistribution", m, s), xs, fname)
            elif is_dist(ref):
                rx = dist_cdf(resolve_reference(ref, None, fname), xs, fname)
            else:
                rv = sample_values(ref, fname)
                rx = [sum(1 for v in rv if v <= x) / len(rv) for x in xs]
            curves.append(list(zip(rx, ys)))
            is_d.append(True)
        else:
            vals = sample_values(it, fname)
            n = len(vals)
            if ref is None:
                rx = dist_cdf(normal_fit(vals), vals, fname)
            elif is_dist(_unwrap(ref, fname)[0]):
                rx = dist_cdf(resolve_reference(_unwrap(ref, fname)[0], vals, fname), vals, fname)
            else:
                rv = sample_values(ref, fname)
                rx = [sum(1 for v in rv if v <= x) / len(rv) for x in vals]
            curves.append([(x, (i + 1) / n) for i, x in enumerate(rx)])
            is_d.append(False)
    styles = style_list(o["PlotStyle"], len(curves), fname, "point")
    joined = _joined(o, len(curves), is_d, fname)
    refl = [[(0.0, 0.0), (1.0, 1.0)]] + [None] * (len(curves) - 1)
    labels = [f"{i + 1}" for i in range(len(curves))] if len(curves) > 1 else None
    return _stat_graphics(fname, curves, joined, styles, o, gopts, refl, labels, ((0.0, 1.0), (0.0, 1.0)))


def _erfcinv(y):
    from mpmath import erfinv
    return float(erfinv(1 - y))


_PSP = {
    "Normal": (lambda x: x, lambda y: 1 - math.sqrt(2) * _erfcinv(2 * y)),
    "LogNormal": (lambda x: math.log(x) if x > 0 else math.nan, lambda y: 1 - math.sqrt(2) * _erfcinv(2 * y)),
    "Exponential": (lambda x: x, lambda y: -math.log(1 - y)),
    "Weibull": (lambda x: math.log(x) if x > 0 else math.nan, lambda y: math.log(-math.log(1 - y))),
    "Gumbel": (lambda x: x, lambda y: math.log(-math.log(1 - y))),
    "Frechet": (lambda x: math.log(x) if x > 0 else math.nan, lambda y: 1 - math.log(-math.log(y))),
    "Rayleigh": (lambda x: x, lambda y: math.sqrt(math.sqrt(2) * (-math.log(1 - y)))),
}


@fn("ProbabilityScalePlot", area=AREA, status="added",
    notes="ProbabilityScalePlot[data], ProbabilityScalePlot[data, \"dist\"] and ProbabilityScalePlot[{data1, ...}, "
          "\"dist\"] for \"Normal\" (default), \"LogNormal\", \"Exponential\", \"Weibull\", \"Gumbel\", "
          "\"Frechet\", \"Rayleigh\": points {g(x_(i)), h(y_i)} with the transformations g, h from the Wolfram "
          "documentation (e.g. Normal: g = x, h = 1 - Sqrt[2] InverseErfc[2 y]; Weibull: g = Log[x], "
          "h = Log[-Log[1 - y]]) and median ranks y_i = (i - 0.3)/(n + 0.4) (Benard's approximation; Wolfram's "
          "constants are not documented); y ticks are labelled with probabilities. Points that the scale cannot "
          "show (x <= 0 on log scales) are omitted. Distribution objects as data, ScalingFunctions and other "
          "settings raise NotImplementedError. Check: reference (exact Weibull/exponential quantiles map to a "
          "straight line in tests).")
def _probabilityscaleplot(data: ListOf[Real], *rest: String | Option) -> Expr:
    fname = "ProbabilityScalePlot"
    pos, opts = split_trailing_options((data,) + rest)
    if not 1 <= len(pos) <= 2:
        raise ValueError(f"{fname}: expected {fname}[data] or {fname}[data, \"dist\"]")
    name = "Normal"
    if len(pos) == 2:
        name = pos[1]
        if not isinstance(name, str):
            if is_dist(name):
                raise NotImplementedError(f"{fname}: distribution objects as the scale are not supported; give a "
                                          "name such as \"Weibull\"")
            raise TypeError(f"{fname}: the scale must be a string such as \"Weibull\", not {K_input(name)}")
        if name not in _PSP:
            raise ValueError(f"{fname}: unknown scale {name!r}; use one of {', '.join(sorted(_PSP))}")
    g, h = _PSP[name]
    o, gopts = split_options(fname, opts, _STAT_DEFAULTS)
    _stat_check(o, fname)
    du, _, _ = _unwrap(pos[0], fname)
    if is_dist(du):
        raise NotImplementedError(f"{fname}: distributions as data are not supported")
    sets = list(du.args) if K.is_list(du) and du.args and all(K.is_list(_unwrap(x, fname)[0]) for x in du.args) \
        else [du]
    curves = []
    for s in sets:
        vals = sample_values(s, fname)
        n = len(vals)
        cv = []
        for i, x in enumerate(vals, 1):
            y = (i - 0.3) / (n + 0.4)
            gx, hy = g(x), h(y)
            if math.isfinite(gx) and math.isfinite(hy):
                cv.append((gx, hy))
        curves.append(cv)
    styles = style_list(o["PlotStyle"], len(curves), fname, "point")
    ticks = []
    for p in (0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999):
        try:
            v = h(p)
        except (ValueError, ZeroDivisionError):
            continue
        if math.isfinite(v):
            ticks.append(L(fl(v), p))
    ft = L(L(L(*ticks), Symbol("Automatic")), L(Symbol("Automatic"), Symbol("Automatic")))
    labels = [f"{i + 1}" for i in range(len(curves))] if len(curves) > 1 else None
    return _stat_graphics(fname, curves, [False] * len(curves), styles, o, gopts, [None] * len(curves), labels,
                          (None, None), extra_defaults=(("FrameTicks", ft),))


# ───────────────────────────── pairwise plots ─────────────────────────────

def _columns(data, fname):
    data, _, _ = _unwrap(data, fname)
    if not K.is_list(data) or not data.args:
        if isinstance(data, (Symbol, Expr)) and not K.is_list(data):
            raise Unevaluated()
        raise ValueError(f"{fname}: the data must be a non-empty matrix {{{{y11, ..., y1n}}, ...}}")
    rows = [r for r in data.args]
    if not all(K.is_list(r) for r in rows):
        raise ValueError(f"{fname}: the data must be a matrix (a list of rows)")
    n = len(rows[0].args)
    if n < 2 or any(len(r.args) != n for r in rows):
        raise ValueError(f"{fname}: every row must have the same number (>= 2) of columns")
    cols = [[scalar(r.args[j], fname) for r in rows] for j in range(n)]
    return cols


def _headers(spec, n, fname):
    if spec is None or is_sym(spec, "Automatic", "None"):
        return [None] * n if is_sym(spec, "None") else [f"y{j + 1}" for j in range(n)]
    if spec == "Indexed":
        return [str(j + 1) for j in range(n)]
    if K.is_list(spec) and len(spec.args) == n:
        return list(spec.args)
    raise ValueError(f"{fname}: Headers must be Automatic, None, \"Indexed\" or a list of {n} labels")


_PAIR_DEFAULTS = {"PlotStyle": AUTO, "Headers": AUTO, "PlotLegends": NONE, "PlotTheme": AUTO, "Method": AUTO,
                  "PlotHighlighting": AUTO, "HeaderStyle": NONE, "HeaderAlignment": AUTO,
                  "HeaderBackground": AUTO, "HeaderDisplayFunction": AUTO, "PlotLayout": AUTO,
                  "PlotMarkers": NONE, "GridLines": AUTO, "GridLinesStyle": AUTO, "Frame": AUTO, "FrameTicks": AUTO,
                  "AspectRatio": 1}
_EXTRA_KNOWN |= {"Headers", "HeaderStyle", "HeaderAlignment", "HeaderBackground", "HeaderDisplayFunction"}


def pairwise(fname, args, panel):
    pos, opts = split_trailing_options(args)
    if len(pos) != 1:
        raise ValueError(f"{fname}: expected {fname}[matrix]")
    o, gopts = split_options(fname, opts, _PAIR_DEFAULTS)
    for n in ("PlotLegends", "PlotTheme", "Method", "PlotHighlighting", "HeaderStyle", "HeaderAlignment",
              "HeaderBackground", "HeaderDisplayFunction", "PlotLayout", "PlotMarkers", "GridLines",
              "GridLinesStyle", "Frame", "FrameTicks"):
        v = o[n]
        if not (is_sym(v, "Automatic", "None") or (n == "PlotTheme" and v == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")
    du, _, _ = _unwrap(pos[0], fname)
    groups = [du]
    if K.is_list(du) and du.args and all(K.is_list(r) and r.args and all(K.is_list(c) for c in r.args)
                                         for r in du.args):
        groups = list(du.args)                         # several datasets (each a matrix)
    colsets = [_columns(g, fname) for g in groups]
    n = len(colsets[0])
    if any(len(c) != n for c in colsets):
        raise ValueError(f"{fname}: all datasets must have the same number of columns")
    heads = _headers(o["Headers"], n, fname)
    styles = style_list(o["PlotStyle"], len(colsets), fname, "point")
    ar = o["AspectRatio"]
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                content = [Expr("Text", heads[i], L(0.5, 0.5))] if heads[i] is not None else []
                row.append(Expr("Graphics", L(*content), Expr("Rule", Symbol("PlotRange"), L(L(0, 1), L(0, 1))),
                                Expr("Rule", Symbol("AspectRatio"), ar), Expr("Rule", Symbol("Frame"), True),
                                Expr("Rule", Symbol("FrameTicks"), NONE)))
                continue
            content = []
            allp = []
            for cols, st in zip(colsets, styles):
                p = panel(cols[i], cols[j])
                allp += p
                if p and st is not None:
                    content.append(L(st, Expr("Point", pts(p))))
            xr = data_range([q[0] for q in allp])
            yr = data_range([q[1] for q in allp])
            pr = resolve_plot_range(AUTO, fname, [xr, yr])
            row.append(Expr("Graphics", L(*content), Expr("Rule", Symbol("Frame"), True),
                            Expr("Rule", Symbol("AspectRatio"), ar),
                            Expr("Rule", Symbol("PlotRange"), range_expr(pr)),
                            Expr("Rule", Symbol("PlotRangePadding"), Expr("Scaled", 0.05))))
        rows.append(L(*row))
    return Expr("GraphicsGrid", L(*rows), *gopts)


def _pair_points(yi, yj):
    return [(b, a) for a, b in zip(yi, yj) if a is not None and b is not None]


def _qq_points(yi, yj):
    a = sorted(v for v in yi if v is not None)
    b = sorted(v for v in yj if v is not None)
    m = min(len(a), len(b))
    if len(a) != len(b):
        a = [empirical_quantile(a, (k + 0.5) / m) for k in range(m)] if len(a) > m else a
        b = [empirical_quantile(b, (k + 0.5) / m) for k in range(m)] if len(b) > m else b
    return [(b[k], a[k]) for k in range(m)]


def _pp_points(yi, yj):
    a = sorted(v for v in yi if v is not None)
    b = sorted(v for v in yj if v is not None)
    if not a or not b:
        return []
    m = len(b)
    import bisect
    return [((k + 1) / m, bisect.bisect_right(a, b[k]) / len(a)) for k in range(m)]


_PAIR_NOTE = (" Output: GraphicsGrid of n x n panels (Graphics with Frame -> True); the diagonal panels show the "
              "column headers (Headers: Automatic \"y1\", ..., \"Indexed\", None or a list). Several datasets "
              "{m1, m2, ...} are drawn in each panel with the default colours. Missing/non-numeric values are "
              "dropped; symbolic data leaves the call unevaluated. Options: PlotStyle, Headers, AspectRatio and "
              "GraphicsGrid options; others raise NotImplementedError.")


@fn("PairwiseListPlot", area=AREA, status="added",
    notes="PairwiseListPlot[{{y11, ..., y1n}, ...}]: scatter-plot matrix; panel {i, j} shows the points {y_kj, "
          "y_ki} (as documented)." + _PAIR_NOTE + " Check: reference (panel points equal the column pairs).")
def _pairwiselistplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    return pairwise("PairwiseListPlot", (data,) + opts, _pair_points)


@fn("PairwiseQuantilePlot", area=AREA, status="added",
    notes="PairwiseQuantilePlot[matrix]: panel {i, j} is the Q-Q plot of column i against column j, points "
          "{k-th smallest y_j, k-th smallest y_i} (as documented; unequal counts after dropping missing values "
          "use Hazen positions)." + _PAIR_NOTE + " Check: reference (sorted columns in tests).")
def _pairwisequantileplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    return pairwise("PairwiseQuantilePlot", (data,) + opts, _qq_points)


@fn("PairwiseProbabilityPlot", area=AREA, status="added",
    notes="PairwiseProbabilityPlot[matrix]: panel {i, j} is the P-P plot of column i against reference column j, "
          "points {k/m, fraction of y_i <= the k-th smallest y_j} (the empirical CDF of y_i at the order "
          "statistics of y_j)." + _PAIR_NOTE + " Check: reference (empirical CDFs computed in tests; a column "
          "against itself lies on the diagonal).")
def _pairwiseprobabilityplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    return pairwise("PairwiseProbabilityPlot", (data,) + opts, _pp_points)


# ───────────────────────────── number line, parallel/radial axes, ternary ─────────────────────────────

def _sympy_set(pred, var, fname):
    """Solution set (SymPy) of an inequality/equation predicate in one real variable."""
    vs = sp.Symbol(var.name, real=True)
    plain = sp.Symbol(var.name)

    def walk(p):
        if isinstance(p, Expr) and p.head == "And":
            out = sp.S.Reals
            for a in p.args:
                out = out.intersect(walk(a))
            return out
        if isinstance(p, Expr) and p.head == "Or":
            out = sp.S.EmptySet
            for a in p.args:
                out = out.union(walk(a))
            return out
        if isinstance(p, Expr) and p.head == "Not" and len(p.args) == 1:
            return sp.S.Reals - walk(p.args[0])
        if p is True:
            return sp.S.Reals
        if p is False:
            return sp.S.EmptySet
        if isinstance(p, Expr) and p.head in ("Less", "LessEqual", "Greater", "GreaterEqual", "Equal", "Unequal"):
            try:
                if len(p.args) != 2:
                    raise ValueError
                a_, b_ = S(p.args[0]), S(p.args[1])
                rel = {"Less": sp.Lt, "LessEqual": sp.Le, "Greater": sp.Gt, "GreaterEqual": sp.Ge,
                       "Equal": sp.Eq, "Unequal": sp.Ne}[p.head](a_, b_)
            except Exception:
                raise NotImplementedError(f"{fname}: could not convert {K_input(p)} to an inequality")
            rel = rel.subs(plain, vs)
            if rel.free_symbols - {vs}:
                raise Unevaluated()
            try:
                return sp.solveset(rel, vs, sp.S.Reals)
            except Exception:
                raise NotImplementedError(f"{fname}: could not solve {K_input(p)}")
        if isinstance(p, Expr) and p.head == "Element" and len(p.args) == 2 and is_sym(p.args[1], "Reals"):
            return sp.S.Reals
        raise NotImplementedError(f"{fname}: the predicate {K_input(p)} is not an inequality in {var.name}")
    return walk(pred)


def _set_pieces(st, fname):
    """SymPy set -> ([(lo, hi, lo_closed, hi_closed)], [points])."""
    pieces, points = [], []
    parts = st.args if isinstance(st, sp.Union) else [st]
    for p in parts:
        if p is sp.S.EmptySet:
            continue
        if isinstance(p, sp.FiniteSet):
            for v in p:
                if not v.is_real:
                    continue
                points.append(float(v))
        elif isinstance(p, sp.Interval):
            lo = -math.inf if p.start is sp.S.NegativeInfinity else float(p.start)
            hi = math.inf if p.end is sp.S.Infinity else float(p.end)
            pieces.append((lo, hi, not p.left_open and math.isfinite(lo), not p.right_open and math.isfinite(hi)))
        elif p is sp.S.Reals:
            pieces.append((-math.inf, math.inf, False, False))
        else:
            raise NotImplementedError(f"{fname}: the solution set {p} cannot be drawn")
    return pieces, points


def _numberline_spec(spec, var, fname):
    """One spec -> (pieces, points)."""
    if isinstance(spec, Expr) and spec.head == "Interval":
        pieces = []
        for iv in spec.args:
            if not (K.is_list(iv) and len(iv.args) == 2):
                raise ValueError(f"{fname}: malformed Interval {K_input(spec)}")
            lo, hi = [_num_or_raise(v, fname, "interval end") if not is_sym(v, "Infinity") else math.inf
                      for v in iv.args]
            if isinstance(iv.args[0], Expr) and iv.args[0].head == "DirectedInfinity":
                lo = -math.inf
            pieces.append((lo, hi, math.isfinite(lo), math.isfinite(hi)))
        return pieces, []
    if K.is_list(spec):
        pieces, points = [], []
        for s in spec.args:
            a, b = _numberline_spec(s, var, fname)
            pieces += a
            points += b
        return pieces, points
    v = num(spec)
    if v is not None:
        return [], [v]
    if is_missing(spec):
        return [], []
    if isinstance(spec, Expr) and isinstance(spec.head, str) and spec.head in (
            "Less", "LessEqual", "Greater", "GreaterEqual", "Equal", "Unequal", "And", "Or", "Not", "Element"):
        if var is None:
            raise ValueError(f"{fname}: a predicate needs the variable as the second argument")
        return _set_pieces(_sympy_set(spec, var, fname), fname)
    if isinstance(spec, str):
        raise TypeError(f"{fname}: {K_input(spec)} is not a number, interval or predicate")
    raise Unevaluated()


_NUMBERLINE_DEFAULTS = {"PlotStyle": AUTO, "PlotLegends": NONE, "Spacings": AUTO, "PlotTheme": AUTO,
                        "IntervalMarkers": AUTO, "IntervalMarkersStyle": AUTO, "PlotRange": AUTO}
_EXTRA_KNOWN |= {"Spacings"}


@fn("NumberLinePlot", area=AREA, status="added",
    notes="NumberLinePlot[{v1, ...}], NumberLinePlot[Interval[...]], NumberLinePlot[pred, x], NumberLinePlot[pred, "
          "{x, xmin, xmax}] and NumberLinePlot[{spec1, spec2, ...}, ...]: values as points, intervals and the "
          "solution sets of predicates (inequalities/equations combined with And, Or, Not; solved exactly with "
          "SymPy over the reals) as segments at heights 1, 2, ... (Spacings -> s, {s1, ...} or None); closed ends "
          "are filled points, open ends hollow, infinite ends arrows to the plot edge. Axes -> {True, False}. "
          "Predicates with other free symbols leave the call unevaluated; non-inequality predicates raise "
          "NotImplementedError. Check: exact (SymPy solveset) and reference (interval ends in tests).")
def _numberlineplot(spec: Expr, *rest: Expr | Option) -> Expr:
    fname = "NumberLinePlot"
    pos, opts = split_trailing_options((spec,) + rest)
    if not 1 <= len(pos) <= 2:
        raise ValueError(f"{fname}: expected {fname}[spec] or {fname}[pred, x]")
    o, gopts = split_options(fname, opts, _NUMBERLINE_DEFAULTS)
    for n in ("PlotTheme", "IntervalMarkers", "IntervalMarkersStyle"):
        if not (is_sym(o[n], "Automatic", "None") or (n == "PlotTheme" and o[n] == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    var, xr_explicit = None, None
    if len(pos) == 2:
        it = pos[1]
        if isinstance(it, Symbol):
            var = it
        elif K.is_list(it) and len(it.args) == 3 and isinstance(it.args[0], Symbol):
            var = it.args[0]
            xr_explicit = (real_arg(it.args[1], fname, "xmin"), real_arg(it.args[2], fname, "xmax"))
            if not xr_explicit[0] < xr_explicit[1]:
                raise ValueError(f"{fname}: the range must have xmin < xmax")
        else:
            raise ValueError(f"{fname}: the second argument must be a variable or {{x, xmin, xmax}}")
    s = pos[0]
    if K.is_list(s) and s.args and all(num(a) is not None or is_missing(a) for a in s.args):
        specs = [s]
    elif K.is_list(s):
        specs = list(s.args)
    else:
        specs = [s]
    sets = [_numberline_spec(sp_, var, fname) for sp_ in specs]
    finite = [v for pcs, ps in sets for v in ps] + [e for pcs, _ in sets for p in pcs for e in p[:2]
                                                   if math.isfinite(e)]
    if xr_explicit:
        x0, x1 = xr_explicit
    elif finite:
        lo, hi = min(finite), max(finite)
        pad = (hi - lo) * 0.1 if hi > lo else 1.0
        x0, x1 = lo - pad, hi + pad
    else:
        x0, x1 = -1.0, 1.0
    sp_opt = o["Spacings"]
    n = len(sets)
    if is_sym(sp_opt, "Automatic"):
        heights = [float(i + 1) for i in range(n)]
    elif is_sym(sp_opt, "None"):
        heights = [1.0] * n
    elif num(sp_opt) is not None:
        heights = [num(sp_opt) * (i + 1) for i in range(n)]
    elif K.is_list(sp_opt) and all(num(a) is not None for a in sp_opt.args) and sp_opt.args:
        acc, heights = 0.0, []
        for i in range(n):
            acc += num(sp_opt.args[min(i, len(sp_opt.args) - 1)])
            heights.append(acc)
    else:
        raise ValueError(f"{fname}: Spacings must be Automatic, None, a number or a list of numbers")
    styles = style_list(o["PlotStyle"], n, fname, "point")
    content = []
    for (pcs, ps), h, st in zip(sets, heights, styles):
        if st is None:
            continue
        grp = [st]
        closed, opened = [], []
        for lo, hi, lc, hc in pcs:
            a, b = max(lo, x0), min(hi, x1)
            if not math.isfinite(lo) and not math.isfinite(hi):
                grp.append(Expr("Arrow", pts([(0.5 * (x0 + x1), h), (x0, h)])))
                grp.append(Expr("Arrow", pts([(0.5 * (x0 + x1), h), (x1, h)])))
                continue
            if not math.isfinite(lo):
                grp.append(Expr("Arrow", pts([(b, h), (x0, h)])))
            elif not math.isfinite(hi):
                grp.append(Expr("Arrow", pts([(a, h), (x1, h)])))
            else:
                grp.append(Expr("Line", pts([(lo, h), (hi, h)])))
            if math.isfinite(lo):
                (closed if lc else opened).append((lo, h))
            if math.isfinite(hi):
                (closed if hc else opened).append((hi, h))
        closed += [(v, h) for v in ps]
        if closed:
            grp.append(Expr("Point", pts(closed)))
        if opened:
            grp.append(Expr("Point", pts(opened)))
            grp.append(L(Expr("GrayLevel", 1), Expr("PointSize", POINT_SIZE * 0.6), Expr("Point", pts(opened))))
        content.append(L(*grp))
    ymax = max(heights) if heights else 1.0
    defaults = [("AspectRatio", Symbol("Automatic") if False else 0.15 * max(1, n)), ("Axes", L(True, False)),
                ("AxesOrigin", L(x0, 0)),
                ("PlotRange", range_expr([(x0, x1), (0.0, ymax + 0.5)])), ("PlotRangePadding", L(Expr("Scaled", 0.02),
                                                                                                Expr("Scaled", 0.1)))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname, "point")


def _rows_datasets(data, fname):
    """ParallelAxisPlot/RadialAxisPlot data -> list of datasets, each a list of rows (lists of floats/None)."""
    d, _, _ = _unwrap(data, fname)
    if isinstance(d, Expr) and d.head == "Association":
        d = L(*[v for _, v in _assoc_items(d)])
    if not K.is_list(d) or not d.args:
        if isinstance(d, (Symbol, Expr)) and not K.is_list(d):
            raise Unevaluated()
        raise ValueError(f"{fname}: the data must be a non-empty list")

    def row(r):
        r, _, _ = _unwrap(r, fname)
        if isinstance(r, Expr) and r.head == "Association":
            r = L(*[v for _, v in _assoc_items(r)])
        if not K.is_list(r):
            raise ValueError(f"{fname}: {K_input(r)} is not a list of values")
        return [scalar(v, fname) for v in r.args]
    items = d.args
    if all(not K.is_list(_unwrap(i, fname)[0]) for i in items):
        return [[row(d)]]
    if all(K.is_list(_unwrap(i, fname)[0]) and _unwrap(i, fname)[0].args and
           all(not K.is_list(v) for v in _unwrap(i, fname)[0].args) for i in items):
        return [[row(i) for i in items]]
    return [[row(r) for r in _unwrap(i, fname)[0].args] for i in items]


_AXISPLOT_DEFAULTS = {"PlotStyle": AUTO, "PlotLegends": NONE, "PlotTheme": AUTO, "Method": AUTO,
                      "PlotHighlighting": AUTO, "PlotRange": AUTO, "Filling": AUTO}


@fn("ParallelAxisPlot", area=AREA, status="added",
    notes="ParallelAxisPlot[{{y11, ..., y1n}, ...}] and ParallelAxisPlot[{data1, ...}]: each point is a line "
          "through its coordinates on n vertical axes at x = 1, ..., n, each axis scaled from its minimum (0) to "
          "its maximum (1) over all data (a constant axis maps to 1/2); missing values break the line; axes are "
          "drawn as gray lines labelled with their minimum and maximum. Options: PlotStyle, PlotLegends and "
          "Graphics options; others raise NotImplementedError. Check: reference (scaled coordinates computed "
          "independently in tests).")
def _parallelaxisplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "ParallelAxisPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, _AXISPLOT_DEFAULTS)
    _axisplot_check(o, fname)
    dss = _rows_datasets(pos[0], fname)
    n = max(len(r) for ds in dss for r in ds)
    if n < 2:
        raise ValueError(f"{fname}: points need at least 2 coordinates")
    rng = []
    for j in range(n):
        vals = [r[j] for ds in dss for r in ds if j < len(r) and r[j] is not None]
        rng.append((min(vals), max(vals)) if vals else (0.0, 1.0))

    def sc(v, j):
        lo, hi = rng[j]
        return 0.5 if hi == lo else (v - lo) / (hi - lo)
    styles = style_list(o["PlotStyle"], len(dss), fname)
    content = [L(Expr("GrayLevel", 0.6), *[Expr("Line", pts([(j + 1, 0), (j + 1, 1)])) for j in range(n)],
                 *[Expr("Text", _num_label(rng[j][0]), pt(j + 1, -0.06)) for j in range(n)],
                 *[Expr("Text", _num_label(rng[j][1]), pt(j + 1, 1.06)) for j in range(n)])]
    for ds, st in zip(dss, styles):
        if st is None:
            continue
        lines = []
        for r in ds:
            seq = [None if j >= len(r) or r[j] is None else (j + 1, sc(r[j], j)) for j in range(n)]
            lines += [ln for ln in runs(seq) if len(ln) >= 2]
        content.append(L(st, *[Expr("Line", pts(ln)) for ln in lines]))
    defaults = [("AspectRatio", GOLDEN), ("Axes", False), ("PlotRange", range_expr([(1.0, float(n)), (-0.1, 1.1)])),
                ("PlotRangePadding", L(Expr("Scaled", 0.05), 0))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname)


def _num_label(v):
    return int(v) if float(v).is_integer() else round(v, 6)


def _axisplot_check(o, fname):
    for n in ("PlotTheme", "Method", "PlotHighlighting", "PlotRange", "Filling"):
        v = o[n]
        if not (is_sym(v, "Automatic", "None") or (n == "PlotTheme" and v == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


@fn("RadialAxisPlot", area=AREA, status="added",
    notes="RadialAxisPlot[{y1, ..., yn}] and RadialAxisPlot[{data1, ...}] (n >= 3 values per shape): axis j points "
          "at angle Pi/2 - 2 Pi (j - 1)/n (first axis up, then clockwise), each axis scaled from min(0, axis "
          "minimum) at the centre to the axis maximum over all data at radius 1; each shape is a closed line with "
          "a translucent filling (Filling -> None removes it); missing values are skipped. Wolfram's own axis "
          "orientation and scaling are not documented. Options: PlotStyle, PlotLegends, Filling and Graphics "
          "options. Check: reference (vertex radii equal the scaled values in tests).")
def _radialaxisplot(data: ListOf[Real] | MatrixOf[Real], *opts: Option) -> Expr:
    fname = "RadialAxisPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, _AXISPLOT_DEFAULTS)
    fill = o["Filling"]
    o2 = dict(o)
    o2["Filling"] = AUTO
    _axisplot_check(o2, fname)
    dss = _rows_datasets(pos[0], fname)
    shapes = [r for ds in dss for r in ds]
    n = max(len(r) for r in shapes)
    if n < 3:
        raise ValueError(f"{fname}: each shape needs at least 3 values")
    rng = []
    for j in range(n):
        vals = [r[j] for r in shapes if j < len(r) and r[j] is not None]
        lo = min([0.0] + vals)
        hi = max(vals) if vals else 1.0
        rng.append((lo, hi if hi > lo else lo + 1.0))
    ang = [math.pi / 2 - 2 * math.pi * j / n for j in range(n)]
    styles = style_list(o["PlotStyle"], len(shapes), fname)
    content = [L(Expr("GrayLevel", 0.6), *[Expr("Line", pts([(0, 0), (math.cos(a), math.sin(a))])) for a in ang])]
    for r, st in zip(shapes, styles):
        if st is None:
            continue
        vs = []
        for j in range(n):
            if j < len(r) and r[j] is not None:
                lo, hi = rng[j]
                rad = (r[j] - lo) / (hi - lo)
                vs.append((rad * math.cos(ang[j]), rad * math.sin(ang[j])))
        if len(vs) < 2:
            continue
        grp = [st]
        if not is_sym(fill, "None"):
            grp.append(L(Expr("Opacity", 0.2), style_color(st), Expr("EdgeForm"), Expr("Polygon", pts(vs))))
        grp.append(Expr("Line", pts(vs + [vs[0]])))
        content.append(L(*grp))
    defaults = [("AspectRatio", 1), ("Axes", False), ("PlotRange", range_expr([(-1.1, 1.1), (-1.1, 1.1)])),
                ("PlotRangePadding", 0)]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname)


@fn("TernaryListPlot", area=AREA, status="added",
    notes="TernaryListPlot[{{u1, v1, w1}, ...}] and TernaryListPlot[{data1, ...}]: points normalised by t = u + v "
          "+ w and placed at {v/t + w/(2 t), Sqrt[3]/2 w/t} inside the equilateral triangle with vertices {0, 0} "
          "(u), {1, 0} (v), {1/2, Sqrt[3]/2} (w), as documented; points with negative or non-real coordinates "
          "(or t = 0) are missing. The triangle is drawn in gray. Options: PlotStyle, PlotLegends, Joined and "
          "Graphics options. Check: reference (barycentric-to-Cartesian map in tests; pure components land on "
          "the vertices).")
def _ternarylistplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "TernaryListPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, {"PlotStyle": AUTO, "PlotLegends": NONE, "Joined": False,
                                         "PlotTheme": AUTO, "PlotMarkers": NONE, "PlotHighlighting": AUTO})
    for n in ("PlotTheme", "PlotMarkers", "PlotHighlighting"):
        if not (is_sym(o[n], "Automatic", "None") or (n == "PlotTheme" and o[n] == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    d, _, _ = _unwrap(pos[0], fname)
    if not K.is_list(d) or not d.args:
        if isinstance(d, (Symbol, Expr)) and not K.is_list(d):
            raise Unevaluated()
        raise ValueError(f"{fname}: the data must be a non-empty list of {{u, v, w}} triples")
    sets = [d] if all(_is_pair(i, 3) or is_missing(i) for i in d.args) else list(d.args)
    curves = []
    for s in sets:
        s, _, _ = _unwrap(s, fname)
        if not K.is_list(s) or not all(_is_pair(i, 3) or is_missing(i) for i in s.args):
            raise ValueError(f"{fname}: each dataset must be a list of {{u, v, w}} triples")
        cv = []
        for i in s.args:
            if is_missing(i):
                continue
            u, v, w = [scalar(c, fname) for c in i.args]
            if None in (u, v, w) or min(u, v, w) < 0 or u + v + w <= 0:
                continue
            t = u + v + w
            cv.append((v / t + w / (2 * t), math.sqrt(3) / 2 * w / t))
        curves.append(cv)
    joined = o["Joined"] is True
    styles = style_list(o["PlotStyle"], len(curves), fname, "line" if joined else "point")
    tri = [(0, 0), (1, 0), (0.5, math.sqrt(3) / 2), (0, 0)]
    content = [L(Expr("GrayLevel", 0.6), Expr("Line", pts(tri)))]
    for cv, st in zip(curves, styles):
        if st is not None and cv:
            content.append(L(st, Expr("Line" if joined else "Point", pts(cv))))
    defaults = [("AspectRatio", Symbol("Automatic")), ("Axes", False),
                ("PlotRange", range_expr([(0.0, 1.0), (0.0, math.sqrt(3) / 2)])),
                ("PlotRangePadding", Expr("Scaled", 0.05))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname, "line" if joined else "point")


# ───────────────────────────── timeline, curve path, plot grid ─────────────────────────────

def _timeline_event(e, fname):
    """An event -> ('point', t) or ('interval', t0, t1)."""
    e, _, _ = _unwrap(e, fname)
    if isinstance(e, Expr) and e.head == "DateInterval":
        iv = e.args[0] if e.args else None
        if K.is_list(iv) and iv.args and K.is_list(iv.args[0]) and len(iv.args[0].args) == 2 and \
                is_date(iv.args[0].args[0]):
            iv = iv.args[0]
        if not (K.is_list(iv) and len(iv.args) == 2):
            raise ValueError(f"{fname}: malformed DateInterval {K_input(e)}")
        return ("interval", abs_time(iv.args[0], fname), abs_time(iv.args[1], fname))
    if isinstance(e, Expr) and e.head in ("Entity", "EntityClass", "TimeSeries", "EventSeries"):
        raise NotImplementedError(f"{fname}: {e.head} events are not supported")
    return ("point", abs_time(e, fname))


@fn("TimelinePlot", area=AREA, status="added",
    notes="TimelinePlot[{date1, ...}], TimelinePlot[{event1, ...}] (dates or DateInterval[{d1, d2}]), "
          "TimelinePlot[<|label -> event, ...|>] and TimelinePlot[{data1, ...}]: dataset i is drawn at height i "
          "(points for dates, thick segments for intervals, association keys as labels above the events) on a "
          "horizontal date axis (x = AbsoluteTime; Ticks at whole years, months, days or hours). Entity, "
          "TimeSeries and EventSeries events raise NotImplementedError; symbolic dates leave the call unevaluated. "
          "Options: PlotStyle, PlotLegends and Graphics options. Check: reference (AbsoluteTime of each event in "
          "tests).")
def _timelineplot(data: ListOf[Expr] | Assoc, *opts: Option) -> Expr:
    fname = "TimelinePlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, {"PlotStyle": AUTO, "PlotLegends": NONE, "PlotTheme": AUTO,
                                         "PlotHighlighting": AUTO, "PlotLayout": AUTO})
    for n in ("PlotTheme", "PlotHighlighting", "PlotLayout"):
        if not (is_sym(o[n], "Automatic", "None") or (n == "PlotTheme" and o[n] == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    d, _, _ = _unwrap(pos[0], fname)

    def events(x):
        x, _, _ = _unwrap(x, fname)
        if isinstance(x, Expr) and x.head == "Association":
            return [(k, _timeline_event(v, fname)) for k, v in _assoc_items(x)]
        if K.is_list(x) and not is_date(x):
            return [(None, _timeline_event(v, fname)) for v in x.args]
        return [(None, _timeline_event(x, fname))]
    if K.is_list(d) and d.args and all((K.is_list(_unwrap(i, fname)[0]) and not is_date(_unwrap(i, fname)[0])) or
                                       (isinstance(_unwrap(i, fname)[0], Expr) and
                                        _unwrap(i, fname)[0].head == "Association") for i in d.args):
        sets = [events(i) for i in d.args]
    elif K.is_list(d) or (isinstance(d, Expr) and d.head == "Association"):
        sets = [events(d)]
    else:
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of dates or events")
    styles = style_list(o["PlotStyle"], len(sets), fname, "point")
    content = []
    ts = []
    for i, (evs, st) in enumerate(zip(sets, styles)):
        h = float(i + 1)
        if st is None:
            continue
        points, segs, texts = [], [], []
        for lab, e in evs:
            if e[0] == "point":
                points.append((e[1], h))
                ts.append(e[1])
                xpos = e[1]
            else:
                segs.append([(e[1], h), (e[2], h)])
                ts += [e[1], e[2]]
                xpos = (e[1] + e[2]) / 2
            if lab is not None:
                texts.append(Expr("Text", lab, pt(xpos, h), L(0, -1.5)))
        grp = [st]
        if segs:
            grp.append(L(Expr("AbsoluteThickness", 6), *[Expr("Line", pts(s)) for s in segs]))
        if points:
            grp.append(Expr("Point", pts(points)))
        grp += texts
        content.append(L(*grp))
    if not ts:
        raise ValueError(f"{fname}: there are no events to plot")
    t0, t1 = min(ts), max(ts)
    if t1 == t0:
        t0, t1 = t0 - 86400, t1 + 86400
    defaults = [("AspectRatio", 0.25 * max(1, len(sets))), ("Axes", L(True, False)),
                ("AxesOrigin", L(t0, 0)), ("Ticks", L(date_ticks(t0, t1), NONE)),
                ("PlotRange", range_expr([(t0, t1), (0.0, len(sets) + 1.0)])),
                ("PlotRangePadding", L(Expr("Scaled", 0.05), 0))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname, "point")


def curve_paths(points):
    """Smooth paths through unordered points: chains grown from an end point by nearest neighbours, broken when
    the next point is farther than 3 times the median nearest-neighbour distance; a chain whose ends are that
    close is closed."""
    np = _np()
    P = np.array(points, dtype=float)
    n = len(P)
    if n < 2:
        return [list(map(tuple, P))]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    nn = D.min(axis=1)
    thr = 3.0 * float(np.median(nn)) if np.median(nn) > 0 else float(np.max(nn[np.isfinite(nn)]) or 1.0)
    unused = set(range(n))
    paths = []
    while unused:
        # start at an end point: the unused point whose two nearest unused neighbours are most one-sided
        idx = sorted(unused)
        best, best_score = idx[0], -2.0
        for i in idx:
            others = [j for j in idx if j != i]
            if len(others) < 2:
                break
            o = sorted(others, key=lambda j: D[i, j])[:2]
            v1, v2 = P[o[0]] - P[i], P[o[1]] - P[i]
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-300))
            if c > best_score:
                best, best_score = i, c
        path = [best]
        unused.discard(best)
        while True:
            cur = path[-1]
            cand = [j for j in unused if D[cur, j] <= thr]
            if not cand:
                break
            nxt = min(cand, key=lambda j: D[cur, j])
            path.append(nxt)
            unused.discard(nxt)
        if len(path) > 3 and D[path[0], path[-1]] <= thr:
            path.append(path[0])
        paths.append([tuple(P[k]) for k in path])
    return paths


@fn("ListCurvePathPlot", area=AREA, status="added",
    notes="ListCurvePathPlot[{{x1, y1}, ...}] (or several datasets): lines through the points ordered as smooth "
          "paths independently of the input order: chains start at an end point and follow nearest neighbours, "
          "break at gaps larger than 3 times the median neighbour spacing (giving disconnected curves) and close "
          "loops. Wolfram's own reconstruction method is not documented; well-sampled curves are reconstructed "
          "exactly in vertex order, ambiguous clouds may join differently. InterpolationOrder other than None/1, "
          "Mesh, ColorFunction and other settings raise NotImplementedError. Check: reference (shuffled samples "
          "of a circle and of a parabola are reconnected in curve order in tests).")
def _listcurvepathplot(data: MatrixOf[Real], *opts: Option) -> Expr:
    fname = "ListCurvePathPlot"
    pos, o_ = _list_args((data,) + opts, fname)
    o, gopts = split_options(fname, o_, {"PlotStyle": AUTO, "PlotRange": AUTO, "InterpolationOrder": NONE,
                                         "PlotTheme": AUTO, "Mesh": NONE, "ColorFunction": AUTO,
                                         "MaxPlotPoints": AUTO, "Method": AUTO, "ScalingFunctions": NONE,
                                         "PlotLegends": NONE})
    for n in ("PlotTheme", "Mesh", "ColorFunction", "MaxPlotPoints", "Method", "ScalingFunctions"):
        if not (is_sym(o[n], "Automatic", "None", "Infinity") or (n == "PlotTheme" and o[n] == "Default")):
            raise NotImplementedError(f"{fname}: the option {n} -> {K_input(o[n])} is not supported")
    if not (is_sym(o["InterpolationOrder"], "None", "Automatic") or o["InterpolationOrder"] == 1):
        raise NotImplementedError(f"{fname}: InterpolationOrder -> {K_input(o['InterpolationOrder'])} is not "
                                  "supported")
    dss = datasets(pos[0], fname)
    curves = []
    for ds in dss:
        ptsl = [p for p in ds.points if p is not None]
        uniq = list(dict.fromkeys(ptsl))
        curves.append(curve_paths(uniq) if uniq else [])
    allp = [p for cv in curves for ln in cv for p in ln]
    xr, yr = data_range([p[0] for p in allp]), data_range([p[1] for p in allp])
    prange = resolve_plot_range(o["PlotRange"], fname, [xr, yr])
    styles = style_list(o["PlotStyle"], len(curves), fname)
    content = [L(st, *[Expr("Line", pts(ln)) for ln in cv if len(ln) >= 2]) for cv, st in zip(curves, styles)
               if st is not None]
    defaults = [("AspectRatio", GOLDEN), ("Axes", L(True, True)), ("PlotRange", range_expr(prange)),
                ("PlotRangePadding", L(L(Expr("Scaled", 0.02), Expr("Scaled", 0.02)),
                                       L(Expr("Scaled", 0.05), Expr("Scaled", 0.05))))]
    gr = finish_graphics(L(*content), gopts, defaults)
    return legend_wrap(gr, o["PlotLegends"], styles, None, fname)


@fn("PlotGrid", area=AREA, status="added",
    notes="PlotGrid[{{g11, g12, ...}, ...}] (and PlotGrid[{g1, g2, ...}] for one row): Graphics[{Inset[g_ij, "
          "{j - 1/2, nrows - i + 1/2}, Center, {1, 1}], ...}] placing each plot in a unit cell (row 1 at the "
          "top), PlotRange {{0, ncols}, {0, nrows}}; None leaves a cell empty; entries must be Graphics (or "
          "Legended graphics). Wolfram's merging of shared frame ticks and labels between neighbouring panels is "
          "not done (each panel keeps its own frame). Check: reference (cell positions in tests).")
def _plotgrid(grid: MatrixOf[Expr] | ListOf[Expr], *opts: Option) -> Expr:
    fname = "PlotGrid"
    pos, o_ = _list_args((grid,) + opts, fname)
    o, gopts = split_options(fname, o_, {"Spacings": AUTO, "PlotLayout": AUTO})
    if not (is_sym(o["Spacings"], "Automatic", "None") or o["Spacings"] == 0):
        raise NotImplementedError(f"{fname}: Spacings -> {K_input(o['Spacings'])} is not supported")
    if not is_sym(o["PlotLayout"], "Automatic"):
        raise NotImplementedError(f"{fname}: PlotLayout is not supported")
    g = pos[0]
    if not K.is_list(g) or not g.args:
        if isinstance(g, (Symbol, Expr)) and not K.is_list(g):
            raise Unevaluated()
        raise ValueError(f"{fname}: the argument must be a list of rows of plots")
    rows = [list(r.args) for r in g.args] if all(K.is_list(r) for r in g.args) else [list(g.args)]
    nr, nc = len(rows), max(len(r) for r in rows)
    content = []
    for i, r in enumerate(rows):
        for j, cell in enumerate(r):
            if is_sym(cell, "None") or cell is None:
                continue
            inner = cell.args[0] if isinstance(cell, Expr) and cell.head == "Legended" and cell.args else cell
            if not (isinstance(inner, Expr) and inner.head in ("Graphics", "Graphics3D", "GraphicsGrid")):
                if isinstance(cell, (Symbol, Expr)) and not K.is_list(cell) and not isinstance(cell, str):
                    raise Unevaluated()
                raise TypeError(f"{fname}: {K_input(cell)} is not a plot")
            content.append(Expr("Inset", cell, pt(j + 0.5, nr - i - 0.5), Symbol("Center"), L(1, 1)))
    defaults = [("AspectRatio", fl(nr) / nc), ("PlotRange", range_expr([(0.0, float(nc)), (0.0, float(nr))])),
                ("PlotRangePadding", 0)]
    return finish_graphics(L(*content), gopts, defaults)
