"""Wolfram batch W20 (part 2): vector, stream, displacement, slice, line-integral-convolution, complex-function
and revolution plots.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). Plots evaluate to inert ``Graphics`` / ``Graphics3D`` objects (W14 graphics core). Tests
compare the *data* (arrow positions and directions, streamline tangents, sampled values, colours) with the
field, never pixels. Documented layout choices (Wolfram's own sampling and placement are not documented):

* vector plots: one arrow per cell centre of an n x n (3-D: n x n x n) grid, centred at the sample point, length
  0.9 cell size x |v|/max|v|, coloured by |v| (default gradient) - ``{colour, Arrow[{tail, head}]}``;
* stream plots: streamlines integrated with fixed-step RK4 along the unit field from seeds on a regular grid in
  both directions, ending at the boundary, at critical points, at sharp turns or where another streamline already
  passes (occupancy grid); every streamline is verified to be tangent to the field (ArithmeticError otherwise);
* density backgrounds as ``DensityPlot`` (W16): ``GraphicsComplex[..., VertexColors -> ...]``.

Grid list data ``{{v11, v12, ...}, ...}`` puts v_ij at {j, i} (row 1 at the bottom), as ListDensityPlot, unless
DataRange is given; 3-D arrays put v_kji at {i, j, k}. Irregular data needs scipy (NotImplementedError without).
"""
from __future__ import annotations

import cmath
import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, S
from .wl_graphics_color import K_input
from . import wl_graphics_core as G
from .wl_graphics_core import is_sym, is_rule
from . import wl_plots2d as P2
from .wl_plots2d import (ev, rgb, default_color, fl, pt, pts, num, real_arg, split_options, finish_graphics,
                         range_expr, AUTO, NONE, blend, DENSITY_GRADIENT, color_function, Compiled, symbolic_check,
                         hold_args, iterator, grid_mesh, density_complex)

AREA = "plotting: vector, stream, slice and complex plots (W20)"


def _np():
    import numpy as np
    return np


def _scipy(fname, what):
    try:
        import scipy.interpolate  # noqa: F401
        import scipy.spatial  # noqa: F401
        return scipy
    except ImportError:
        raise NotImplementedError(f"{fname}: {what} needs scipy (pip install scipy)")


_VEC_OPTS = {"VectorPoints": AUTO, "VectorScaling": AUTO, "VectorSizes": AUTO, "VectorStyle": AUTO,
             "VectorColorFunction": AUTO, "VectorColorFunctionScaling": True, "VectorMarkers": AUTO,
             "VectorRange": AUTO, "VectorAspectRatio": AUTO, "StreamPoints": AUTO, "StreamScale": AUTO,
             "StreamStyle": AUTO, "StreamColorFunction": AUTO, "StreamColorFunctionScaling": True,
             "StreamMarkers": AUTO, "RegionBoundaryStyle": AUTO, "RegionFillingStyle": AUTO,
             "LineIntegralConvolutionScale": AUTO, "LightingAngle": AUTO}
_DEFAULTS = dict(P2._FIELD_DEFAULTS)
_DEFAULTS.update(_VEC_OPTS)
_TRIVIAL = {"Automatic", "None", "Inherited"}


def _unsupported(o, fname, allowed):
    for n, v in o.items():
        if n in allowed or n == "PlotRange":
            continue
        if is_sym(v, *_TRIVIAL) or v is True and n.endswith("Scaling") \
                or (n == "WorkingPrecision" and is_sym(v, "MachinePrecision")) \
                or (n == "PlotTheme" and v == "Default") or (n == "PerformanceGoal" and v in ("Quality", "Speed")):
            continue
        raise NotImplementedError(f"{fname}: the option {n} -> {K_input(v)} is not supported")


def options(fname, opts, allowed, dim=2):
    o, gopts = split_options(fname, opts, _DEFAULTS, dim=dim)
    _unsupported(o, fname, set(allowed))
    return o, gopts


def int_option(v, default, fname, name, lo=1):
    if is_sym(v, "Automatic"):
        return default
    if isinstance(v, int) and not isinstance(v, bool) and v >= lo:
        return v
    if isinstance(v, int) and not isinstance(v, bool):
        raise ValueError(f"{fname}: {name} must be an integer >= {lo}")
    raise NotImplementedError(f"{fname}: {name} -> {K_input(v)} is not supported (give an integer)")


# ───────────────────────────── fields of functions ─────────────────────────────

def field_spec(f, k, fname):
    """Held {v1, ..., vk} or {{v1, ..., vk}, s} -> (component exprs, scalar expr or None)."""
    if isinstance(f, Expr) and f.head == "Evaluate" and len(f.args) == 1:
        f = ev(f.args[0])
    if not K.is_list(f):
        g = ev(f)
        if K.is_list(g):
            f = g
        elif isinstance(g, (Symbol, Expr)):
            raise Unevaluated()
        else:
            raise TypeError(f"{fname}: the field must be a list of {k} components, not {K_input(g)}")
    a = list(f.args)
    if len(a) == k and not any(K.is_list(c) for c in a):
        return a, None
    if len(a) == 2 and K.is_list(a[0]) and len(a[0].args) == k and not K.is_list(a[1]):
        return list(a[0].args), a[1]
    if a and all(K.is_list(c) for c in a):
        raise NotImplementedError(f"{fname}: several vector fields in one plot are not supported")
    raise ValueError(f"{fname}: the field must be a list of {k} components {{v1, ..., v{k}}}")


def compile_all(exprs, vars_, ranges, fname):
    probe = [tuple(lo + (hi - lo) * t for (lo, hi), t in zip(ranges, ts)) for ts in ((0.31, 0.67, 0.53),
                                                                                    (0.77, 0.21, 0.38))]
    out = []
    for e in exprs:
        c = Compiled(e, vars_, fname)
        symbolic_check(c, vars_, fname)
        c.cross_check(probe)
        out.append(c)
    return out


def vector_function(comps):
    """Vectorised F(*coordinate arrays) -> (m, k) array (NaN where undefined)."""
    np = _np()

    def F(*X):
        X = [np.asarray(x, dtype=float) for x in X]
        return np.stack([np.broadcast_to(c(*X), X[0].shape) for c in comps], axis=-1)
    return F


def iterators(pos, k, fname, first=2):
    its = [iterator(s, fname, first + i) for i, s in enumerate(pos)]
    if len(its) != k:
        raise ValueError(f"{fname}: expected {k} iterators {{x, xmin, xmax}}")
    if len({v.name for v, _, _ in its}) != k:
        raise ValueError(f"{fname}: the iterator variables must differ")
    return [v for v, _, _ in its], [(a, b) for _, a, b in its]


def cell_centres(ranges, n):
    np = _np()
    axes = [lo + (np.arange(n) + 0.5) * (hi - lo) / n for lo, hi in ranges]
    mesh = np.meshgrid(*axes, indexing="xy") if len(ranges) == 2 else np.meshgrid(*axes, indexing="ij")
    P = np.column_stack([m.ravel() for m in mesh])
    return P, min((hi - lo) / n for lo, hi in ranges)


def magnitude_colors(mags, o, fname, key="VectorColorFunction", default=None):
    """Colours of |v| (scaled to [0, 1] over the plot), None entries mean the plain style."""
    np = _np()
    spec = o.get(key, AUTO)
    if is_sym(spec, "None"):
        return [None] * len(mags)
    fin = mags[np.isfinite(mags)]
    lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    cf = color_function(spec, fname, default or (lambda t: blend(DENSITY_GRADIENT, t)))
    scale = o.get(key + "Scaling", True) is not False
    out = []
    for m in mags:
        t = (float(m) - lo) / (hi - lo) if (scale and hi > lo) else (0.5 if scale else float(m))
        out.append(rgb(cf(t)))
    return out


def arrows(P, V, cell, o, fname, default_style=None):
    """Arrow groups centred at P with lengths 0.9 cell |v|/max|v| (zero and undefined vectors omitted)."""
    np = _np()
    mags = np.linalg.norm(V, axis=1)
    ok = np.isfinite(mags) & (mags > 0)
    if not ok.any():
        return [], 0.0
    mmax = float(mags[ok].max())
    style = o.get("VectorStyle", AUTO)
    if not is_sym(style, "Automatic"):
        G.directive_like(style, fname, "VectorStyle")
    if is_sym(o.get("VectorColorFunction", AUTO), "Automatic") and default_style is not None:
        cols = [default_style] * len(mags)
    else:
        cols = magnitude_colors(np.where(ok, mags, 0.0), o, fname)
    out = []
    for i in np.nonzero(ok)[0]:
        u = V[i] / mags[i] * (0.9 * cell * mags[i] / mmax) / 2
        a, b = P[i] - u, P[i] + u
        grp = [c for c in (cols[i], None if is_sym(style, "Automatic") else style) if c is not None]
        out.append(L(*grp, Expr("Arrow", pts([tuple(a), tuple(b)]))))
    return out, mmax


# ───────────────────────────── streamlines ─────────────────────────────

def streamlines(F, ranges, seeds_n, fname, occ_n=None, max_steps=None):
    """Streamlines [[point, ...], ...] of the field F (vectorised on (m, d) arrays of points)."""
    np = _np()
    d = len(ranges)
    lo = np.array([a for a, _ in ranges], dtype=float)
    hi = np.array([b for _, b in ranges], dtype=float)
    span = hi - lo
    occ_n = occ_n or (30 if d == 2 else 16)
    h = float(span.min()) / (occ_n * 3)
    max_steps = max_steps or int(6 * occ_n * 3 * float(span.max() / span.min()))
    occ = np.zeros((occ_n,) * d, dtype=bool)

    def cell(p):
        c = np.floor((p - lo) / span * occ_n).astype(int)
        return tuple(np.clip(c, 0, occ_n - 1))

    def inside(p):
        return bool(np.all(p >= lo - 1e-12) and np.all(p <= hi + 1e-12))

    def unit(p):
        v = F(*[np.array([c]) for c in p])[0]
        m = float(np.linalg.norm(v))
        if not math.isfinite(m) or m == 0:
            return None, 0.0
        return v / m, m

    probe = cell_centres(ranges, 12)[0]
    mags = np.linalg.norm(F(*probe.T), axis=1)
    vmax = float(np.nanmax(mags)) if np.isfinite(mags).any() else 0.0
    if not vmax > 0:
        return []
    seeds, _ = cell_centres(ranges, seeds_n)
    lines = []
    for s in seeds:
        if occ[cell(s)]:
            continue
        halves = []
        own = set()
        for sign in (1.0, -1.0):
            p = s.copy()
            path = [p.copy()]
            prev_u = None
            for _ in range(max_steps):
                k1, m = unit(p)
                if k1 is None or m < 1e-9 * vmax:
                    break
                k1 = sign * k1
                k2, _ = unit(p + h / 2 * k1)
                if k2 is None:
                    break
                k2 = sign * k2
                k3, _ = unit(p + h / 2 * k2)
                if k3 is None:
                    break
                k3 = sign * k3
                k4, _ = unit(p + h * k3)
                if k4 is None:
                    break
                k4 = sign * k4
                step = (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if prev_u is not None and float(np.dot(step, prev_u)) < math.cos(math.radians(30)) * \
                        float(np.linalg.norm(step)):
                    break                                   # sharp turn: near a critical point
                q = p + h * step
                if not inside(q):
                    break
                c = cell(q)
                if occ[c] and c not in own:
                    break
                own.add(c)
                path.append(q.copy())
                prev_u = step / max(float(np.linalg.norm(step)), 1e-300)
                p = q
            halves.append(path)
        line = halves[1][::-1] + halves[0][1:]
        if len(line) >= 3:
            for c in own | {cell(s)}:
                occ[c] = True
            lines.append(line)
    check_tangent(lines, F, fname)
    return lines


def check_tangent(lines, F, fname):
    """Self-check: every streamline chord is parallel to the field at its midpoint."""
    np = _np()
    for ln in lines:
        A = np.array(ln)
        mids = (A[1:] + A[:-1]) / 2
        ch = A[1:] - A[:-1]
        V = F(*mids.T)
        nv = np.linalg.norm(V, axis=1)
        nc = np.linalg.norm(ch, axis=1)
        ok = (nv > 0) & (nc > 0) & np.isfinite(nv)
        cos = np.abs(np.sum(ch[ok] * V[ok], axis=1)) / (nv[ok] * nc[ok])
        if cos.size and float(cos.min()) < 0.97:
            raise ArithmeticError(f"{fname}: result failed verification (streamline not tangent to the field)")


def stream_groups(lines, F, o, fname, default_style=None):
    np = _np()
    if not lines:
        return []
    mags = np.array([float(np.nanmean(np.linalg.norm(F(*np.array(ln).T), axis=1))) for ln in lines])
    style = o.get("StreamStyle", AUTO)
    if not is_sym(style, "Automatic"):
        G.directive_like(style, fname, "StreamStyle")
    if is_sym(o.get("StreamColorFunction", AUTO), "Automatic") and default_style is not None:
        cols = [default_style] * len(lines)
    else:
        cols = magnitude_colors(mags, o, fname, "StreamColorFunction")
    out = []
    for ln, c in zip(lines, cols):
        grp = [x for x in (c, None if is_sym(style, "Automatic") else style) if x is not None]
        out.append(L(*grp, Expr("Arrow", pts([tuple(p) for p in ln]))))
    return out


def _stream_n(o, fname):
    return int_option(o["StreamPoints"], 10, fname, "StreamPoints", 2)


def _vec_n(o, fname, default):
    return int_option(o["VectorPoints"], default, fname, "VectorPoints", 2)


def background(Fs, ranges, o, fname):
    """Density background of the scalar function Fs (vectorised) as DensityPlot."""
    np = _np()
    n = P2._grid_n(o, fname, 25)
    (a, b), (c, d) = ranges
    P, T, _, _ = grid_mesh(a, b, c, d, n, n)
    V = np.asarray(Fs(P[:, 0], P[:, 1]), dtype=float)
    fin = V[np.isfinite(V)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    return density_complex(P, V, T, cf, vlo, vhi, o["ColorFunctionScaling"] is not False)


def span(r):
    """A non-degenerate (lo, hi): equal ends are widened by 10% of |lo| (or 1)."""
    lo, hi = float(r[0]), float(r[1])
    if hi > lo:
        return (lo, hi)
    d = 0.1 * abs(lo) or 1.0
    return (lo - d, hi + d)


def finish2(content, gopts, ranges, o, fname):
    r = [span(x) for x in ranges]
    pr = o["PlotRange"]
    if not is_sym(pr, "Automatic", "All", "Full"):
        r = P2.resolve_plot_range(pr, fname, r)
    return finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Frame", True), ("PlotRange", range_expr(r)),
                                                ("PlotRangePadding", Expr("Scaled", 0.02))])


def finish3(content, gopts, ranges, o, fname):
    r = [span(x) for x in ranges]
    pr = o["PlotRange"]
    if not is_sym(pr, "Automatic", "All", "Full"):
        r = P2.resolve_plot_range(pr, fname, r, dims=3)
    return finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 1, 1)),
                                                ("PlotRange", range_expr(r))], dim=3)


def function_field(args, k, fname, allowed, dim=None):
    """Common front end of the function forms: (F, scalar Fs or None, ranges, o, gopts)."""
    pos, o_ = hold_args(args)
    if len(pos) != k + 1:
        raise ValueError(f"{fname}: expected {fname}[{{v1, ..., v{k}}}, " + ", ".join(["{x, xmin, xmax}"] * k) + "]")
    comps, s = field_spec(pos[0], k, fname)
    vars_, ranges = iterators(pos[1:], k, fname)
    o, gopts = options(fname, o_, allowed, dim=dim or k)
    cs = compile_all(comps, vars_, ranges, fname)
    F = vector_function(cs)
    Fs = None
    if s is not None:
        Fs = compile_all([s], vars_, ranges, fname)[0]
    return F, Fs, ranges, o, gopts


_VEC_NOTE = ("Arrows: one per cell centre of a VectorPoints x VectorPoints grid (default {n}), centred at the "
             "sample point with length 0.9 cell x |v|/max|v|, coloured by |v| (VectorColorFunction: Automatic, None "
             "or a function; VectorStyle adds a directive); zero and undefined vectors are omitted. ")
_STREAM_NOTE = ("Streamlines: fixed-step RK4 along the unit field from the cell centres of a StreamPoints x "
                "StreamPoints grid (default 10), in both directions, stopping at the boundary, critical points, "
                "sharp turns (> 30 degrees per step) or an occupancy cell (1/30 of the range) already crossed by "
                "another streamline; each is Arrow[{points}] coloured by its mean |v| (StreamColorFunction, "
                "StreamStyle). Every streamline is verified to be tangent to the field (ArithmeticError otherwise). ")


@fn("StreamPlot", area=AREA, status="added", hold=True,
    notes="StreamPlot[{vx, vy}, {x, xmin, xmax}, {y, ymin, ymax}] (HoldAll): Graphics of streamlines. " + _STREAM_NOTE
          + "Options: StreamPoints, StreamStyle, StreamColorFunction(Scaling), PlotRange and Graphics options; "
          "VectorPoints, regions, several fields and other settings raise NotImplementedError; free parameters leave "
          "it unevaluated. Check: self-check (streamline tangency, raises ArithmeticError) and reference (tests).")
def _streamplot(f: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "StreamPlot"
    F, Fs, ranges, o, gopts = function_field((f, sx, sy) + opts, 2, fname,
                                             ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                              "StreamColorFunctionScaling"))
    if Fs is not None:
        raise NotImplementedError(f"{fname}: a scalar field {{{{vx, vy}}, s}} needs StreamDensityPlot")
    lines = streamlines(lambda *X: F(*X), ranges, _stream_n(o, fname), fname)
    return finish2(stream_groups(lines, F, o, fname), gopts, ranges, o, fname)


@fn("StreamDensityPlot", area=AREA, status="added", hold=True,
    notes="StreamDensityPlot[{vx, vy}, {x, ...}, {y, ...}] and [{{vx, vy}, s}, ...] (HoldAll): a density "
          "background of |v| (or the scalar field s), as DensityPlot (ColorFunction, PlotPoints, MaxRecursion), "
          "with black streamlines on top. " + _STREAM_NOTE + "Other settings raise NotImplementedError. Check: "
          "self-check (streamline tangency) and reference (background values equal |v| in tests).")
def _streamdensityplot(f: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "StreamDensityPlot"
    F, Fs, ranges, o, gopts = function_field((f, sx, sy) + opts, 2, fname,
                                             ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                              "StreamColorFunctionScaling", "ColorFunction", "ColorFunctionScaling",
                                              "PlotPoints", "MaxRecursion"))
    np = _np()
    Fb = Fs if Fs is not None else (lambda x, y: np.linalg.norm(F(x, y), axis=-1))
    bg = background(Fb, ranges, o, fname)
    lines = streamlines(F, ranges, _stream_n(o, fname), fname)
    return finish2([bg] + stream_groups(lines, F, o, fname, Expr("GrayLevel", 0)), gopts, ranges, o, fname)


@fn("VectorDensityPlot", area=AREA, status="added", hold=True,
    notes="VectorDensityPlot[{vx, vy}, {x, ...}, {y, ...}] and [{{vx, vy}, s}, ...] (HoldAll): a density background "
          "of |v| (or s) as DensityPlot with black arrows on top. " + _VEC_NOTE.format(n=15) + "Options: "
          "VectorPoints, VectorStyle, VectorColorFunction, ColorFunction, PlotPoints, MaxRecursion, PlotRange and "
          "Graphics options; others raise NotImplementedError. Check: reference (arrow directions and background "
          "values compared with the field in tests).")
def _vectordensityplot(f: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "VectorDensityPlot"
    F, Fs, ranges, o, gopts = function_field((f, sx, sy) + opts, 2, fname,
                                             ("VectorPoints", "VectorStyle", "VectorColorFunction",
                                              "VectorColorFunctionScaling", "ColorFunction", "ColorFunctionScaling",
                                              "PlotPoints", "MaxRecursion"))
    np = _np()
    Fb = Fs if Fs is not None else (lambda x, y: np.linalg.norm(F(x, y), axis=-1))
    bg = background(Fb, ranges, o, fname)
    P, cell = cell_centres(ranges, _vec_n(o, fname, 15))
    ar, _ = arrows(P, F(*P.T), cell, o, fname, Expr("GrayLevel", 0))
    return finish2([bg] + ar, gopts, ranges, o, fname)


@fn("VectorPlot3D", area=AREA, status="added", hold=True,
    notes="VectorPlot3D[{vx, vy, vz}, {x, ...}, {y, ...}, {z, ...}] (HoldAll): Graphics3D of arrows. "
          + _VEC_NOTE.format(n=7).replace("VectorPoints x VectorPoints", "n x n x n") + "Options: VectorPoints, "
          "VectorStyle, VectorColorFunction(Scaling), PlotRange and Graphics3D options; others raise "
          "NotImplementedError. Check: reference (arrow directions equal the field in tests).")
def _vectorplot3d(f: Held, sx: Held, sy: Held, sz: Held, *opts: Held | Option) -> Expr:
    fname = "VectorPlot3D"
    F, Fs, ranges, o, gopts = function_field((f, sx, sy, sz) + opts, 3, fname,
                                             ("VectorPoints", "VectorStyle", "VectorColorFunction",
                                              "VectorColorFunctionScaling"))
    if Fs is not None:
        raise NotImplementedError(f"{fname}: scalar fields {{{{vx, vy, vz}}, s}} are not supported")
    P, cell = cell_centres(ranges, _vec_n(o, fname, 7))
    ar, _ = arrows(P, F(*P.T), cell, o, fname)
    return finish3(ar, gopts, ranges, o, fname)


@fn("StreamPlot3D", area=AREA, status="added", hold=True,
    notes="StreamPlot3D[{vx, vy, vz}, {x, ...}, {y, ...}, {z, ...}] (HoldAll): Graphics3D of streamlines. "
          + _STREAM_NOTE.replace("StreamPoints x StreamPoints grid (default 10)", "n x n x n grid (StreamPoints, "
                                 "default 5)").replace("1/30", "1/16") + "Other settings raise NotImplementedError. "
          "Check: self-check (streamline tangency, raises ArithmeticError).")
def _streamplot3d(f: Held, sx: Held, sy: Held, sz: Held, *opts: Held | Option) -> Expr:
    fname = "StreamPlot3D"
    F, Fs, ranges, o, gopts = function_field((f, sx, sy, sz) + opts, 3, fname,
                                             ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                              "StreamColorFunctionScaling"))
    if Fs is not None:
        raise NotImplementedError(f"{fname}: scalar fields are not supported")
    lines = streamlines(F, ranges, int_option(o["StreamPoints"], 5, fname, "StreamPoints", 2), fname)
    return finish3(stream_groups(lines, F, o, fname), gopts, ranges, o, fname)


# ───────────────────────────── list vector data ─────────────────────────────

def _vec(e, k):
    """Float tuple of a k-vector expression, None for missing/undefined."""
    if not (K.is_list(e) and len(e.args) == k):
        return False
    vs = [num(c) for c in e.args]
    return None if any(v is None for v in vs) else tuple(vs)


def _data_range(dr, k, shape, fname):
    if dr is None or is_sym(dr, "Automatic", "All"):
        return [(1.0, float(n)) for n in shape]
    if K.is_list(dr) and len(dr.args) == k and all(K.is_list(q) and len(q.args) == 2 for q in dr.args):
        out = [tuple(real_arg(v, fname, "DataRange") for v in q.args) for q in dr.args]
        if any(not a < b for a, b in out):
            raise ValueError(f"{fname}: DataRange needs min < max")
        return out
    raise ValueError(f"{fname}: DataRange must be Automatic or a list of {k} {{min, max}} pairs")


class VectorData:
    """List vector data: a regular grid (axes, values V[..., k] indexed [y, x] or [z, y, x]) or scattered points."""

    def __init__(self, data, k, fname, dr=None, scalar=False):
        np = _np()
        self.k, self.fname = k, fname
        d, _, _ = P2._unwrap(data, fname)
        if not K.is_list(d) or not d.args:
            if isinstance(d, (Symbol, Expr)) and not K.is_list(d):
                raise Unevaluated()
            raise TypeError(f"{fname}: the data must be a non-empty list, not {K_input(d)}")
        w = 1 if scalar else k
        rows = list(d.args)
        pairs = all(K.is_list(r) and len(r.args) == 2 and _vec(r.args[0], k) is not False for r in rows) and \
            (scalar or all(_vec(r.args[1], k) is not False for r in rows))
        if pairs and not scalar and k == 2 and len(rows) == 2:
            pairs = False          # {{v11, v12}, {v21, v22}}: a 2 x 2 array (ambiguous with two {point, vector})
        if pairs:
            P, V = [], []
            for r in rows:
                p = _vec(r.args[0], k)
                v = (num(r.args[1]),) if scalar else _vec(r.args[1], k)
                if p is None:
                    continue
                P.append(p)
                V.append(tuple(math.nan for _ in range(w)) if v is None or None in v else v)
            if len(P) < 1:
                raise ValueError(f"{fname}: the data has no points")
            self.grid = False
            self.P = np.array(P, dtype=float)
            self.V = np.array(V, dtype=float)
            self.ranges = [P2.data_range(list(self.P[:, j])) for j in range(k)]
            return
        if scalar and all(K.is_list(r) and len(r.args) == k + 1 and all(not K.is_list(c) for c in r.args)
                          for r in rows) and len(rows) > k + 1:
            P = [tuple(num(c) for c in r.args) for r in rows]
            P = [p for p in P if None not in p[:k]]
            self.grid = False
            self.P = np.array([p[:k] for p in P], dtype=float)
            self.V = np.array([[math.nan if p[k] is None else p[k]] for p in P], dtype=float)
            self.ranges = [P2.data_range(list(self.P[:, j])) for j in range(k)]
            return
        arr = self._array(d, k, w)
        if arr.ndim != k + 1:
            raise ValueError(f"{fname}: the data must be a {k}-dimensional array of " +
                             ("values" if scalar else f"{k}-vectors") + " or a list of {point, value} pairs")
        shape = arr.shape[:k][::-1]                    # (nx, ny[, nz])
        if any(n < 2 for n in shape):
            raise ValueError(f"{fname}: the array must have at least 2 points along each axis")
        self.grid = True
        self.ranges = _data_range(dr, k, shape, fname)
        self.axes = [np.linspace(a, b, n) for (a, b), n in zip(self.ranges, shape)]
        self.A = arr
        mesh = np.meshgrid(*self.axes, indexing="xy") if k == 2 else \
            [m.transpose(2, 1, 0) for m in np.meshgrid(*self.axes, indexing="ij")]
        self.P = np.column_stack([m.ravel() for m in mesh])
        self.V = arr.reshape(-1, w)

    def _array(self, d, k, w):
        np = _np()
        fname = self.fname

        def conv(e, depth):
            if depth == k:
                if w == 1:
                    if K.is_list(e):
                        raise ValueError(f"{fname}: the array entries must be numbers")
                    v = num(e)
                    return [math.nan if v is None else v]
                v = _vec(e, k)
                if v is False:
                    raise ValueError(f"{fname}: the array entries must be {k}-vectors, not {K_input(e)}")
                return [math.nan] * k if v is None else list(v)
            if not K.is_list(e):
                raise ValueError(f"{fname}: the data must be a rectangular array")
            return [conv(a, depth + 1) for a in e.args]
        try:
            return np.array(conv(d, 0), dtype=float)
        except ValueError as exc:
            if "rectangular" in str(exc) or fname in str(exc):
                raise
            raise ValueError(f"{fname}: the data must be a rectangular array")

    def interpolator(self):
        """Vectorised F(*coords) -> (m, w) values: multilinear on grids, linear (scipy) on scattered points."""
        np = _np()
        if self.grid:
            axes, A, k = self.axes, self.A, self.k

            def F(*X):
                X = [np.asarray(x, dtype=float).ravel() for x in X]
                idx, fr, ok = [], [], np.ones(X[0].shape, dtype=bool)
                for ax, x in zip(axes, X):
                    t = (x - ax[0]) / (ax[-1] - ax[0]) * (len(ax) - 1)
                    ok &= (t >= -1e-9) & (t <= len(ax) - 1 + 1e-9)
                    i = np.clip(np.floor(t).astype(int), 0, len(ax) - 2)
                    idx.append(i)
                    fr.append(np.clip(t - i, 0.0, 1.0))
                out = 0.0
                for corner in range(2 ** k):
                    wgt = 1.0
                    ix = []
                    for j in range(k):
                        bit = (corner >> j) & 1
                        wgt = wgt * (fr[j] if bit else 1 - fr[j])
                        ix.append(idx[j] + bit)
                    out = out + wgt[:, None] * A[tuple(ix[::-1])]
                out = np.where(ok[:, None], out, np.nan)
                return out
            return F
        _scipy(self.fname, "interpolating scattered data")
        from scipy.interpolate import LinearNDInterpolator
        if len(self.P) < self.k + 1:
            raise ValueError(f"{self.fname}: at least {self.k + 1} points are needed")
        try:
            itp = LinearNDInterpolator(self.P, self.V)
        except Exception as exc:
            raise ValueError(f"{self.fname}: the points cannot be triangulated ({exc.__class__.__name__})")

        def F(*X):
            X = np.column_stack([np.asarray(x, dtype=float).ravel() for x in X])
            return np.asarray(itp(X), dtype=float).reshape(len(X), -1)
        return F

    def cell(self):
        if self.grid:
            return min((ax[-1] - ax[0]) / (len(ax) - 1) for ax in self.axes)
        np = _np()
        vol = float(np.prod([max(b - a, 1e-300) for a, b in self.ranges]))
        return (vol / max(len(self.P), 1)) ** (1 / self.k)


def list_args(args, fname, n_pos=1):
    pos, opts = G.split_trailing_options(args)
    if len(pos) != n_pos:
        raise ValueError(f"{fname}: expected {n_pos} positional argument(s), got {len(pos)}")
    return list(pos), list(opts)


def _thin(P, V, nmax, k):
    """At most about nmax^k data arrows (regular stride on the data order)."""
    if len(P) <= nmax ** k:
        return P, V
    s = int(math.ceil(len(P) / nmax ** k))
    return P[::s], V[::s]


def list_vector(fname, data, opts, k, allowed, extra=None):
    o, gopts = options(fname, opts, set(allowed) | {"DataRange"}, dim=k)
    vd = VectorData(data, k, fname, o["DataRange"])
    return vd, o, gopts


_LIST_NOTE = ("Data: an array {{v11, v12, ...}, ...} (v_ij at {j, i}, row 1 at the bottom, or DataRange) or "
              "{{{x1, y1}, {vx1, vy1}}, ...}; missing vectors are skipped. ")


@fn("ListVectorPlot", area=AREA, status="added",
    notes="ListVectorPlot[data]: Graphics of arrows at the data points (at most 30 x 30, taken on a regular "
          "stride), centred, length 0.9 spacing x |v|/max|v| (spacing = grid step, or sqrt(area/n) for scattered "
          "points), coloured by |v|. " + _LIST_NOTE + "Options: VectorStyle, VectorColorFunction(Scaling), "
          "DataRange, PlotRange and Graphics options; others raise NotImplementedError. Check: reference (arrow "
          "midpoints and directions equal the data in tests).")
def _listvectorplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListVectorPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 2, ("VectorStyle", "VectorColorFunction",
                                                          "VectorColorFunctionScaling"))
    P, V = _thin(vd.P, vd.V, 30, 2)
    ar, _ = arrows(P, V, vd.cell(), o, fname)
    return finish2(ar, gopts, vd.ranges, o, fname)


@fn("ListVectorPlot3D", area=AREA, status="added",
    notes="ListVectorPlot3D[data]: Graphics3D of arrows at the data points (at most 10 x 10 x 10 on a regular "
          "stride), as ListVectorPlot. Data: an array v_kji at {i, j, k} (or DataRange) or {{{x, y, z}, {vx, vy, "
          "vz}}, ...}. Options: VectorStyle, VectorColorFunction(Scaling), DataRange, PlotRange and Graphics3D "
          "options. Check: reference (arrow midpoints and directions equal the data in tests).")
def _listvectorplot3d(data: ListOf[Expr], *opts: Option) -> Expr:
    fname = "ListVectorPlot3D"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 3, ("VectorStyle", "VectorColorFunction",
                                                          "VectorColorFunctionScaling"))
    P, V = _thin(vd.P, vd.V, 10, 3)
    ar, _ = arrows(P, V, vd.cell(), o, fname)
    return finish3(ar, gopts, vd.ranges, o, fname)


def _list_background(vd, F, o, fname):
    np = _np()
    if vd.grid:
        (a, b), (c, d) = vd.ranges
        nx, ny = len(vd.axes[0]), len(vd.axes[1])
        P, T, _, _ = grid_mesh(a, b, c, d, nx, ny)
        Vv = np.linalg.norm(vd.V, axis=1)
    else:
        _scipy(fname, "scattered data")
        from scipy.spatial import Delaunay
        P = vd.P
        T = Delaunay(P).simplices
        Vv = np.linalg.norm(vd.V, axis=1)
    fin = Vv[np.isfinite(Vv)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    return density_complex(P, Vv, T, cf, vlo, vhi, o["ColorFunctionScaling"] is not False)


@fn("ListVectorDensityPlot", area=AREA, status="added",
    notes="ListVectorDensityPlot[data]: a density background of |v| interpolated linearly between the data points "
          "(grid triangles, or a Delaunay triangulation of scattered points - needs scipy) with black arrows at the "
          "data points as ListVectorPlot. " + _LIST_NOTE + "Options: ColorFunction, ColorFunctionScaling, "
          "VectorStyle, VectorColorFunction, DataRange, PlotRange and Graphics options. Check: reference (vertex "
          "values equal |v| of the data in tests).")
def _listvectordensityplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListVectorDensityPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 2, ("VectorStyle", "VectorColorFunction",
                                                          "VectorColorFunctionScaling", "ColorFunction",
                                                          "ColorFunctionScaling"))
    bg = _list_background(vd, None, o, fname)
    P, V = _thin(vd.P, vd.V, 30, 2)
    ar, _ = arrows(P, V, vd.cell(), o, fname, Expr("GrayLevel", 0))
    return finish2([bg] + ar, gopts, vd.ranges, o, fname)


@fn("ListStreamPlot", area=AREA, status="added",
    notes="ListStreamPlot[data]: streamlines of the field interpolated from the data (bilinear on grids; linear on "
          "a Delaunay triangulation of scattered points - needs scipy). " + _LIST_NOTE + _STREAM_NOTE + "Options: "
          "StreamPoints, StreamStyle, StreamColorFunction(Scaling), DataRange, PlotRange and Graphics options. "
          "Check: self-check (streamline tangency to the interpolated field, raises ArithmeticError).")
def _liststreamplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListStreamPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 2, ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                                          "StreamColorFunctionScaling"))
    F = vd.interpolator()
    lines = streamlines(F, vd.ranges, _stream_n(o, fname), fname)
    return finish2(stream_groups(lines, F, o, fname), gopts, vd.ranges, o, fname)


@fn("ListStreamDensityPlot", area=AREA, status="added",
    notes="ListStreamDensityPlot[data]: the |v| density background of ListVectorDensityPlot with black streamlines "
          "of ListStreamPlot on top. " + _LIST_NOTE + "Options: ColorFunction, ColorFunctionScaling, StreamPoints, "
          "StreamStyle, StreamColorFunction, DataRange, PlotRange and Graphics options. Check: self-check "
          "(streamline tangency) and reference (background values in tests).")
def _liststreamdensityplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListStreamDensityPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 2, ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                                          "StreamColorFunctionScaling", "ColorFunction",
                                                          "ColorFunctionScaling"))
    bg = _list_background(vd, None, o, fname)
    F = vd.interpolator()
    lines = streamlines(F, vd.ranges, _stream_n(o, fname), fname)
    return finish2([bg] + stream_groups(lines, F, o, fname, Expr("GrayLevel", 0)), gopts, vd.ranges, o, fname)


@fn("ListStreamPlot3D", area=AREA, status="added",
    notes="ListStreamPlot3D[data]: 3-D streamlines of the field interpolated from the data (trilinear on arrays "
          "v_kji at {i, j, k}; linear on a Delaunay tetrahedralisation of {{{x, y, z}, {vx, vy, vz}}, ...} - needs "
          "scipy), seeds on a StreamPoints^3 grid (default 5), as StreamPlot3D. Options: StreamPoints, StreamStyle, "
          "StreamColorFunction(Scaling), DataRange, PlotRange and Graphics3D options. Check: self-check (streamline "
          "tangency, raises ArithmeticError).")
def _liststreamplot3d(data: ListOf[Expr], *opts: Option) -> Expr:
    fname = "ListStreamPlot3D"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 3, ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                                          "StreamColorFunctionScaling"))
    F = vd.interpolator()
    lines = streamlines(F, vd.ranges, int_option(o["StreamPoints"], 5, fname, "StreamPoints", 2), fname)
    return finish3(stream_groups(lines, F, o, fname), gopts, vd.ranges, o, fname)


# ───────────────────────────── displacement plots ─────────────────────────────

DISPLACEMENT_FRACTION = 0.1


def displace(P, V, ranges):
    """P + s V with s chosen so that the largest displacement is 10% of the region's diagonal."""
    np = _np()
    mags = np.linalg.norm(V, axis=1)
    fin = mags[np.isfinite(mags)]
    m = float(fin.max()) if fin.size else 0.0
    diag = math.sqrt(sum((b - a) ** 2 for a, b in ranges))
    s = DISPLACEMENT_FRACTION * diag / m if m > 0 else 0.0
    return P + s * np.nan_to_num(V), mags, s


def displaced_complex(Q, mags, faces, o, fname):
    np = _np()
    ok = np.isfinite(mags)
    cols = magnitude_colors(np.where(ok, mags, 0.0), {"VectorColorFunction": o["ColorFunction"],
                                                      "VectorColorFunctionScaling": o["ColorFunctionScaling"]}, fname)
    fs = [f for f in faces if all(ok[k] for k in f)]
    return Expr("GraphicsComplex", pts([tuple(q) for q in Q]),
                L(Expr("EdgeForm"), Expr("Polygon", L(*[L(*[int(k) + 1 for k in f]) for f in fs]))),
                Expr("Rule", Symbol("VertexColors"), L(*[c if c is not None else Expr("GrayLevel", 0.6)
                                                          for c in cols])))


def _region_iterators(pos, k, fname):
    """{x, ...}, {y, ...} iterators, or {x, y} ∈ Rectangle/Cuboid."""
    if len(pos) == 1 and isinstance(pos[0], Expr) and pos[0].head == "Element":
        vs, reg = pos[0].args
        reg = ev(reg)
        if not K.is_list(vs) or len(vs.args) != k:
            raise ValueError(f"{fname}: the region form needs {{x, y{', z' if k == 3 else ''}}} \u2208 reg")
        if isinstance(reg, Expr) and reg.head in ("Rectangle", "Cuboid") and len(reg.args) == 2:
            lo, hi = reg.args
            r = [(real_arg(a, fname, "the region"), real_arg(b, fname, "the region")) for a, b in zip(lo.args,
                                                                                                     hi.args)]
            if len(r) == k and all(a < b for a, b in r):
                return [P2.var_symbol(v, fname) for v in vs.args], r
        raise NotImplementedError(f"{fname}: regions other than Rectangle/Cuboid are not supported")
    return iterators(pos, k, fname)


@fn("VectorDisplacementPlot", area=AREA, status="added", hold=True,
    notes="VectorDisplacementPlot[{vx, vy}, {x, ...}, {y, ...}] and [{vx, vy}, {x, y} \u2208 Rectangle[...]] (HoldAll): "
          "the region's grid (PlotPoints^2, default 25 x 25 refined by MaxRecursion as DensityPlot) displaced to "
          "p + s v(p), with s such that the largest displacement is 10% of the region's diagonal (Wolfram's "
          "automatic scaling is not documented), as GraphicsComplex[displaced points, {EdgeForm[], "
          "Polygon[cells]}, VertexColors -> colours by |v|]. Other regions, {{vx, vy}, s} fields and other settings "
          "raise NotImplementedError. Check: reference (displaced vertices equal p + s v(p) in tests).")
def _vectordisplacementplot(f: Held, *rest: Held | Option) -> Expr:
    fname = "VectorDisplacementPlot"
    pos, o_ = hold_args((f,) + rest)
    comps, s = field_spec(pos[0], 2, fname)
    if s is not None:
        raise NotImplementedError(f"{fname}: scalar styling fields are not supported")
    vars_, ranges = _region_iterators(pos[1:], 2, fname)
    o, gopts = options(fname, o_, ("PlotPoints", "MaxRecursion", "ColorFunction", "ColorFunctionScaling"))
    F = vector_function(compile_all(comps, vars_, ranges, fname))
    n = P2._grid_n(o, fname, 25)
    (a, b), (c, d) = ranges
    P, _, _, _ = grid_mesh(a, b, c, d, n, n)
    Q, mags, _ = displace(P, F(P[:, 0], P[:, 1]), ranges)
    faces = [(j * n + i, j * n + i + 1, (j + 1) * n + i + 1, (j + 1) * n + i) for j in range(n - 1)
             for i in range(n - 1)]
    g = displaced_complex(Q, mags, faces, o, fname)
    np = _np()
    fin = Q[np.isfinite(Q).all(axis=1)]
    r = [P2.data_range(list(fin[:, 0])), P2.data_range(list(fin[:, 1]))]
    return finish_graphics(L(g), gopts, [("AspectRatio", Symbol("Automatic")), ("Frame", True),
                                         ("PlotRange", range_expr([span(x) for x in r]))])


def box_faces(n):
    """Boundary quads of an n x n x n grid of the unit cube: (points (m, 3) in [0, 1], faces)."""
    np = _np()
    t = np.linspace(0, 1, n)
    U, V = np.meshgrid(t, t, indexing="xy")
    u, v = U.ravel(), V.ravel()
    P, faces = [], []
    for axis in range(3):
        for val in (0.0, 1.0):
            base = len(P)
            cols = [u, v]
            pts3 = np.zeros((n * n, 3))
            others = [j for j in range(3) if j != axis]
            pts3[:, axis] = val
            pts3[:, others[0]], pts3[:, others[1]] = cols
            P.extend(pts3.tolist())
            faces += [(base + j * n + i, base + j * n + i + 1, base + (j + 1) * n + i + 1, base + (j + 1) * n + i)
                      for j in range(n - 1) for i in range(n - 1)]
    return np.array(P), faces


@fn("VectorDisplacementPlot3D", area=AREA, status="added", hold=True,
    notes="VectorDisplacementPlot3D[{vx, vy, vz}, {x, ...}, {y, ...}, {z, ...}] and [..., {x, y, z} \u2208 Cuboid[...]] "
          "(HoldAll): the boundary of the box (n x n quads per face, default n = 15) displaced to p + s v(p) with "
          "the largest displacement 10% of the diagonal, as Graphics3D[GraphicsComplex[points, {EdgeForm[], "
          "Polygon[...]}, VertexColors -> colours by |v|]]. Other regions and settings raise NotImplementedError. "
          "Check: reference (displaced vertices equal p + s v(p) in tests).")
def _vectordisplacementplot3d(f: Held, *rest: Held | Option) -> Expr:
    fname = "VectorDisplacementPlot3D"
    pos, o_ = hold_args((f,) + rest)
    comps, s = field_spec(pos[0], 3, fname)
    if s is not None:
        raise NotImplementedError(f"{fname}: scalar styling fields are not supported")
    vars_, ranges = _region_iterators(pos[1:], 3, fname)
    o, gopts = options(fname, o_, ("PlotPoints", "ColorFunction", "ColorFunctionScaling"), dim=3)
    np = _np()
    F = vector_function(compile_all(comps, vars_, ranges, fname))
    n = int_option(o["PlotPoints"], 15, fname, "PlotPoints", 2)
    U, faces = box_faces(n)
    lo = np.array([a for a, _ in ranges])
    sp_ = np.array([b - a for a, b in ranges])
    P = lo + U * sp_
    Q, mags, _ = displace(P, F(*P.T), ranges)
    g = displaced_complex(Q, mags, faces, o, fname)
    fin = Q[np.isfinite(Q).all(axis=1)]
    return finish3([g], gopts, [P2.data_range(list(fin[:, j])) for j in range(3)], o, fname)


@fn("ListVectorDisplacementPlot", area=AREA, status="added",
    notes="ListVectorDisplacementPlot[data]: the data points displaced to p + s v with the largest displacement "
          "10% of the diagonal, as GraphicsComplex[displaced points, Polygon[cells], VertexColors -> colours by "
          "|v|] - grid cells for arrays (v_ij at {j, i} or DataRange), a Delaunay triangulation of the original "
          "points for {{{x, y}, {vx, vy}}, ...} (needs scipy). Options: ColorFunction, ColorFunctionScaling, "
          "DataRange and Graphics options. Check: reference (displaced vertices in tests).")
def _listvectordisplacementplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListVectorDisplacementPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 2, ("ColorFunction", "ColorFunctionScaling"))
    np = _np()
    if vd.grid:
        nx, ny = len(vd.axes[0]), len(vd.axes[1])
        faces = [(j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i) for j in range(ny - 1)
                 for i in range(nx - 1)]
    else:
        _scipy(fname, "scattered data")
        from scipy.spatial import Delaunay
        if len(vd.P) < 3:
            raise ValueError(f"{fname}: at least 3 points are needed")
        faces = [tuple(t) for t in Delaunay(vd.P).simplices]
    Q, mags, _ = displace(vd.P, vd.V, vd.ranges)
    g = displaced_complex(Q, mags, faces, o, fname)
    return finish_graphics(L(g), gopts, [("AspectRatio", Symbol("Automatic")), ("Frame", True),
                                         ("PlotRange", range_expr([span(P2.data_range(list(Q[:, j])))
                                                                   for j in range(2)]))])


@fn("ListVectorDisplacementPlot3D", area=AREA, status="added",
    notes="ListVectorDisplacementPlot3D[data]: the boundary of the data displaced to p + s v (largest displacement "
          "10% of the diagonal): the boundary faces of an array v_kji at {i, j, k}, or the convex hull of "
          "{{{x, y, z}, {vx, vy, vz}}, ...} (needs scipy), as Graphics3D[GraphicsComplex[..., VertexColors -> "
          "colours by |v|]]. Options: ColorFunction, ColorFunctionScaling, DataRange and Graphics3D options. "
          "Check: reference (displaced vertices in tests).")
def _listvectordisplacementplot3d(data: ListOf[Expr], *opts: Option) -> Expr:
    fname = "ListVectorDisplacementPlot3D"
    pos, opts_ = list_args((data,) + opts, fname)
    vd, o, gopts = list_vector(fname, pos[0], opts_, 3, ("ColorFunction", "ColorFunctionScaling"))
    np = _np()
    if vd.grid:
        nx, ny, nz = (len(a) for a in vd.axes)

        def idx(i, j, k):
            return (k * ny + j) * nx + i
        faces = []
        for k in (0, nz - 1):
            faces += [(idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k))
                      for j in range(ny - 1) for i in range(nx - 1)]
        for j in (0, ny - 1):
            faces += [(idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j, k + 1), idx(i, j, k + 1))
                      for k in range(nz - 1) for i in range(nx - 1)]
        for i in (0, nx - 1):
            faces += [(idx(i, j, k), idx(i, j + 1, k), idx(i, j + 1, k + 1), idx(i, j, k + 1))
                      for k in range(nz - 1) for j in range(ny - 1)]
    else:
        _scipy(fname, "scattered data")
        from scipy.spatial import ConvexHull
        try:
            faces = [tuple(t) for t in ConvexHull(vd.P).simplices]
        except Exception as exc:
            raise ValueError(f"{fname}: the points have no 3-D hull ({exc.__class__.__name__})")
    Q, mags, _ = displace(vd.P, vd.V, vd.ranges)
    g = displaced_complex(Q, mags, faces, o, fname)
    return finish3([g], gopts, [P2.data_range(list(Q[:, j])) for j in range(3)], o, fname)


# ───────────────────────────── slice plots ─────────────────────────────

def slice_planes(surf, vars_, ranges, fname):
    """[(axis, value)] coordinate planes of a slice-surface specification."""
    mids = [(a + b) / 2 for a, b in ranges]
    if surf is None or is_sym(surf, "Automatic") or surf == "CenterPlanes":
        return [(0, mids[0]), (1, mids[1]), (2, mids[2])]
    if surf == "BackPlanes":
        return [(0, ranges[0][0]), (1, ranges[1][1]), (2, ranges[2][0])]
    names = ("XStackedPlanes", "YStackedPlanes", "ZStackedPlanes")
    if isinstance(surf, str) and surf in names:
        surf = L(surf, 3)
    if K.is_list(surf) and len(surf.args) == 2 and surf.args[0] in names:
        ax = names.index(surf.args[0])
        a, b = ranges[ax]
        spec = surf.args[1]
        if K.is_list(spec):
            vals = [real_arg(v, fname, "a plane position") for v in spec.args]
        else:
            n = int_option(spec, 3, fname, "the number of planes", 1)
            vals = [a + (b - a) * (i + 1) / (n + 1) for i in range(n)]
        for v in vals:
            if not a - 1e-12 <= v <= b + 1e-12:
                raise ValueError(f"{fname}: the plane position {v} is outside the plot range")
        return [(ax, v) for v in vals]
    if isinstance(surf, Expr) and surf.head == "Equal" and len(surf.args) == 2:
        lhs, rhs = surf.args
        names_ = [v.name for v in vars_]
        if isinstance(lhs, Symbol) and lhs.name in names_ and num(ev(rhs)) is not None:
            ax = names_.index(lhs.name)
            v = num(ev(rhs))
            a, b = ranges[ax]
            if not a - 1e-12 <= v <= b + 1e-12:
                raise ValueError(f"{fname}: the plane {K_input(surf)} is outside the plot range")
            return [(ax, v)]
        raise NotImplementedError(f"{fname}: only coordinate planes var == c are supported as equations")
    if K.is_list(surf) and surf.args and not (len(surf.args) == 2 and surf.args[0] in names):
        out = []
        for s_ in surf.args:
            out += slice_planes(s_, vars_, ranges, fname)
        return out
    if isinstance(surf, str):
        raise NotImplementedError(f"{fname}: the slice surface {surf!r} is not supported (coordinate planes only)")
    raise NotImplementedError(f"{fname}: the slice surface {K_input(surf)} is not supported (coordinate planes only)")


def plane_grid(ax, val, ranges, n):
    """(3-D points, 2-D parameter points, triangles) of an n x n grid on the plane coordinate[ax] = val."""
    np = _np()
    others = [j for j in range(3) if j != ax]
    (a, b), (c, d) = ranges[others[0]], ranges[others[1]]
    P2d, T, _, _ = grid_mesh(a, b, c, d, n, n)
    P3 = np.zeros((len(P2d), 3))
    P3[:, ax] = val
    P3[:, others[0]], P3[:, others[1]] = P2d[:, 0], P2d[:, 1]
    return P3, P2d, T, others


def _slice_args(args, fname, k_field):
    pos, o_ = hold_args(args)
    if len(pos) == 4:
        f, surf, its = pos[0], None, pos[1:]
    elif len(pos) == 5:
        f, surf, its = pos[0], pos[1], pos[2:]
        surf = ev(surf) if not (isinstance(surf, Expr) and surf.head == "Equal") else surf
    else:
        raise ValueError(f"{fname}: expected {fname}[f, surf, {{x, ...}}, {{y, ...}}, {{z, ...}}]")
    vars_, ranges = iterators(its, 3, fname, 3 if surf is not None else 2)
    return f, surf, vars_, ranges, o_


def slice_density(planes, Fs, ranges, n, o, fname):
    np = _np()
    parts = [plane_grid(ax, v, ranges, n) for ax, v in planes]
    vals = [np.asarray(Fs(*P3.T), dtype=float).ravel() for P3, _, _, _ in parts]
    allv = np.concatenate(vals)
    fin = allv[np.isfinite(allv)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    out = []
    for (P3, _, T, _), V in zip(parts, vals):
        g = density_complex(P3, V, T, cf, vlo, vhi, o["ColorFunctionScaling"] is not False)
        out.append(g)
    return out, (vlo, vhi)


def slice_contours(planes, Fs, ranges, n, o, fname):
    np = _np()
    parts = [plane_grid(ax, v, ranges, n) for ax, v in planes]
    vals = [np.asarray(Fs(*P3.T), dtype=float).ravel() for P3, _, _, _ in parts]
    allv = np.concatenate(vals)
    fin = allv[np.isfinite(allv)]
    vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
    levels = P2.contour_levels(o["Contours"], vlo, vhi, fname)
    cf = color_function(o["ColorFunction"], fname, lambda t: blend(DENSITY_GRADIENT, t))
    out = []
    edges = [vlo] + list(levels) + [vhi]
    for (P3, P2d, T, others), V, (ax, val) in zip(parts, vals, planes):
        bands = P2.bands(P2d, V, T, levels)
        nb = len(bands)
        for kb, polys in enumerate(bands):
            if not polys:
                continue
            col = rgb(cf((kb + 0.5) / nb if nb else 0.5))
            out.append(L(col, Expr("EdgeForm"), *[Expr("Polygon", pts([_lift(p, ax, val, others) for p in poly]))
                                                  for poly in polys]))
        lines = []
        for lev in levels:
            for ln in P2.iso_lines(P2d, V, T, lev):
                lines.append(pts([_lift(p, ax, val, others) for p in ln]))
        if lines:
            out.append(L(Expr("GrayLevel", 0.3), Expr("Line", L(*lines))))
    return out, levels, edges


def _lift(p, ax, val, others):
    q = [0.0, 0.0, 0.0]
    q[ax] = val
    q[others[0]], q[others[1]] = float(p[0]), float(p[1])
    return tuple(q)


def slice_vectors(planes, F, ranges, n, o, fname):
    np = _np()
    Ps = []
    for ax, v in planes:
        others = [j for j in range(3) if j != ax]
        P2d, cell = cell_centres([ranges[others[0]], ranges[others[1]]], n)
        P3 = np.zeros((len(P2d), 3))
        P3[:, ax] = v
        P3[:, others[0]], P3[:, others[1]] = P2d[:, 0], P2d[:, 1]
        Ps.append((P3, cell))
    P = np.vstack([p for p, _ in Ps])
    cell = min(c for _, c in Ps)
    ar, _ = arrows(P, np.asarray(F(*P.T), dtype=float).reshape(len(P), 3), cell, o, fname)
    return ar


_SLICE_NOTE = ("Slice surfaces: Automatic/\"CenterPlanes\" (x, y, z planes through the centre), \"BackPlanes\" "
               "(x = xmin, y = ymax, z = zmin), \"XStackedPlanes\" etc. (3 planes, {\"XStackedPlanes\", n} n equally "
               "spaced interior planes, {\"XStackedPlanes\", {x1, ...}} given positions), x == c, and lists of "
               "these; spheres, cut boxes, implicit surfaces and regions raise NotImplementedError. ")


@fn("SliceDensityPlot3D", area=AREA, status="added", hold=True,
    notes="SliceDensityPlot3D[f, surf, {x, ...}, {y, ...}, {z, ...}] and [f, {x, ...}, ...] (HoldAll): one "
          "GraphicsComplex per plane (n x n grid, default PlotPoints 25 refined by MaxRecursion 1 to 49) with "
          "VertexColors from ColorFunction((f - fmin)/(fmax - fmin)) over all planes (DensityPlot's gradient). "
          + _SLICE_NOTE + "Options: PlotPoints, MaxRecursion, ColorFunction, ColorFunctionScaling, PlotRange and "
          "Graphics3D options. Check: numeric cross-check (compiled vs kernel N) and reference (vertex values in "
          "tests).")
def _slicedensityplot3d(f: Held, *rest: Held | Option) -> Expr:
    fname = "SliceDensityPlot3D"
    f, surf, vars_, ranges, o_ = _slice_args((f,) + rest, fname, 1)
    o, gopts = options(fname, o_, ("PlotPoints", "MaxRecursion", "ColorFunction", "ColorFunctionScaling"), dim=3)
    Fs = compile_all([f], vars_, ranges, fname)[0]
    planes = slice_planes(surf, vars_, ranges, fname)
    content, _ = slice_density(planes, Fs, ranges, P2._grid_n(o, fname, 25), o, fname)
    return finish3(content, gopts, ranges, o, fname)


@fn("SliceContourPlot3D", area=AREA, status="added", hold=True,
    notes="SliceContourPlot3D[f, surf, {x, ...}, {y, ...}, {z, ...}] and [f, {x, ...}, ...] (HoldAll): filled "
          "contour bands (Polygons coloured by band, as ContourPlot) and gray contour Lines on each plane, levels "
          "from Contours (Automatic: as ContourPlot; n; or a list) over the value range of all planes. "
          + _SLICE_NOTE + "Options: Contours, PlotPoints, MaxRecursion, ColorFunction, PlotRange and Graphics3D "
          "options. Check: reference (contour line vertices have f = level in tests).")
def _slicecontourplot3d(f: Held, *rest: Held | Option) -> Expr:
    fname = "SliceContourPlot3D"
    f, surf, vars_, ranges, o_ = _slice_args((f,) + rest, fname, 1)
    o, gopts = options(fname, o_, ("PlotPoints", "MaxRecursion", "ColorFunction", "ColorFunctionScaling",
                                   "Contours"), dim=3)
    Fs = compile_all([f], vars_, ranges, fname)[0]
    planes = slice_planes(surf, vars_, ranges, fname)
    content, _, _ = slice_contours(planes, Fs, ranges, P2._grid_n(o, fname, 25), o, fname)
    return finish3(content, gopts, ranges, o, fname)


@fn("SliceVectorPlot3D", area=AREA, status="added", hold=True,
    notes="SliceVectorPlot3D[{vx, vy, vz}, surf, {x, ...}, {y, ...}, {z, ...}] and without surf (HoldAll): 3-D "
          "arrows of the field at the cell centres of a VectorPoints x VectorPoints grid (default 7) on each "
          "plane, centred, length 0.9 cell |v|/max|v|, coloured by magnitude (as documented). " + _SLICE_NOTE +
          "Options: VectorPoints, VectorStyle, VectorColorFunction(Scaling), PlotRange and Graphics3D options. "
          "Check: reference (arrow directions equal the field in tests).")
def _slicevectorplot3d(f: Held, *rest: Held | Option) -> Expr:
    fname = "SliceVectorPlot3D"
    f, surf, vars_, ranges, o_ = _slice_args((f,) + rest, fname, 3)
    comps, s = field_spec(f, 3, fname)
    if s is not None:
        raise NotImplementedError(f"{fname}: scalar fields are not supported")
    o, gopts = options(fname, o_, ("VectorPoints", "VectorStyle", "VectorColorFunction",
                                   "VectorColorFunctionScaling"), dim=3)
    F = vector_function(compile_all(comps, vars_, ranges, fname))
    planes = slice_planes(surf, vars_, ranges, fname)
    return finish3(slice_vectors(planes, F, ranges, _vec_n(o, fname, 7), o, fname), gopts, ranges, o, fname)


def _list_slice(args, fname, scalar):
    pos, opts = G.split_trailing_options(args)
    if len(pos) not in (1, 2):
        raise ValueError(f"{fname}: expected {fname}[data] or {fname}[data, surf]")
    return pos[0], (pos[1] if len(pos) > 1 else None), opts


_X3 = [Symbol("plot$w20$x"), Symbol("plot$w20$y"), Symbol("plot$w20$z")]
_LIST3_NOTE = ("Data: an array f_kji at {i, j, k} (or DataRange {{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}), "
               "interpolated trilinearly, or {{x, y, z, f}, ...} interpolated linearly on a Delaunay "
               "tetrahedralisation (needs scipy). ")


@fn("ListSliceDensityPlot3D", area=AREA, status="added",
    notes="ListSliceDensityPlot3D[data] and [data, surf]: as SliceDensityPlot3D on the interpolated data. "
          + _LIST3_NOTE + _SLICE_NOTE + "Options: PlotPoints, ColorFunction, ColorFunctionScaling, DataRange and "
          "Graphics3D options. Check: reference (vertex values equal the data at grid points in tests).")
def _listslicedensityplot3d(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "ListSliceDensityPlot3D"
    d, surf, opts = _list_slice((data,) + rest, fname, True)
    o, gopts = options(fname, opts, ("PlotPoints", "ColorFunction", "ColorFunctionScaling", "DataRange"), dim=3)
    vd = VectorData(d, 3, fname, o["DataRange"], scalar=True)
    F = vd.interpolator()
    planes = slice_planes(surf, _X3, vd.ranges, fname)
    content, _ = slice_density(planes, lambda *X: F(*X)[:, 0], vd.ranges,
                               int_option(o["PlotPoints"], 25, fname, "PlotPoints", 2), o, fname)
    return finish3(content, gopts, vd.ranges, o, fname)


@fn("ListSliceContourPlot3D", area=AREA, status="added",
    notes="ListSliceContourPlot3D[data] and [data, surf]: as SliceContourPlot3D on the interpolated data. "
          + _LIST3_NOTE + _SLICE_NOTE + "Options: Contours, PlotPoints, ColorFunction, DataRange and Graphics3D "
          "options. Check: reference (contour vertices have the interpolated value = level in tests).")
def _listslicecontourplot3d(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "ListSliceContourPlot3D"
    d, surf, opts = _list_slice((data,) + rest, fname, True)
    o, gopts = options(fname, opts, ("PlotPoints", "ColorFunction", "ColorFunctionScaling", "DataRange",
                                     "Contours"), dim=3)
    vd = VectorData(d, 3, fname, o["DataRange"], scalar=True)
    F = vd.interpolator()
    planes = slice_planes(surf, _X3, vd.ranges, fname)
    content, _, _ = slice_contours(planes, lambda *X: F(*X)[:, 0], vd.ranges,
                                   int_option(o["PlotPoints"], 25, fname, "PlotPoints", 2), o, fname)
    return finish3(content, gopts, vd.ranges, o, fname)


@fn("ListSliceVectorPlot3D", area=AREA, status="added",
    notes="ListSliceVectorPlot3D[data] and [data, surf]: as SliceVectorPlot3D on the interpolated vector data "
          "(arrays v_kji at {i, j, k} interpolated trilinearly, or {{{x, y, z}, {vx, vy, vz}}, ...} linearly - "
          "needs scipy). " + _SLICE_NOTE + "Options: VectorPoints, VectorStyle, VectorColorFunction, DataRange "
          "and Graphics3D options. Check: reference (arrow directions equal the data in tests).")
def _listslicevectorplot3d(data: ListOf[Expr], *rest: Expr | Option) -> Expr:
    fname = "ListSliceVectorPlot3D"
    d, surf, opts = _list_slice((data,) + rest, fname, False)
    o, gopts = options(fname, opts, ("VectorPoints", "VectorStyle", "VectorColorFunction",
                                     "VectorColorFunctionScaling", "DataRange"), dim=3)
    vd = VectorData(d, 3, fname, o["DataRange"])
    F = vd.interpolator()
    planes = slice_planes(surf, _X3, vd.ranges, fname)
    return finish3(slice_vectors(planes, F, vd.ranges, _vec_n(o, fname, 7), o, fname), gopts, vd.ranges, o, fname)


# ───────────────────────────── line integral convolution ─────────────────────────────

LIC_SEED = 20


def lic(F, ranges, n, length):
    """Gray intensities (n x n, row 0 at the bottom): the mean of white noise (numpy default_rng(LIC_SEED) uniform
    [0, 1)) over the pixel and `length` unit-pixel Euler steps each way along the unit field, stopping at the
    border or where the field vanishes."""
    np = _np()
    (a, b), (c, d) = ranges
    noise = np.random.default_rng(LIC_SEED).random((n, n))
    xs = a + (np.arange(n) + 0.5) * (b - a) / n
    ys = c + (np.arange(n) + 0.5) * (d - c) / n
    X, Y = np.meshgrid(xs, ys)
    acc = noise.copy().ravel()
    cnt = np.ones(n * n)
    for sign in (1.0, -1.0):
        px = (np.arange(n * n) % n + 0.5).astype(float)
        py = (np.arange(n * n) // n + 0.5).astype(float)
        alive = np.ones(n * n, dtype=bool)
        for _ in range(length):
            wx = a + px * (b - a) / n
            wy = c + py * (d - c) / n
            V = F(wx, wy)
            vx = V[:, 0] / ((b - a) / n)
            vy = V[:, 1] / ((d - c) / n)
            m = np.hypot(vx, vy)
            alive &= np.isfinite(m) & (m > 0)
            m = np.where(alive, m, 1.0)
            px = px + sign * np.where(alive, vx / m, 0.0)
            py = py + sign * np.where(alive, vy / m, 0.0)
            alive &= (px >= 0) & (px < n) & (py >= 0) & (py < n)
            ix = np.clip(np.floor(px).astype(int), 0, n - 1)
            iy = np.clip(np.floor(py).astype(int), 0, n - 1)
            acc += np.where(alive, noise[iy, ix], 0.0)
            cnt += alive
    return (acc / cnt).reshape(n, n), X, Y


def lic_graphics(F, ranges, o, gopts, fname):
    np = _np()
    n = int_option(o["RasterSize"], 128, fname, "RasterSize", 4)
    ln = int_option(o["LineIntegralConvolutionScale"], 10, fname, "LineIntegralConvolutionScale", 1)
    I, X, Y = lic(F, ranges, n, ln)
    if is_sym(o["ColorFunction"], "Automatic"):
        rows = L(*[L(*[fl(float(v)) for v in row]) for row in I])
    else:
        V = F(X.ravel(), Y.ravel())
        mags = np.linalg.norm(V, axis=1)
        cols = magnitude_colors(mags, {"VectorColorFunction": o["ColorFunction"],
                                       "VectorColorFunctionScaling": o["ColorFunctionScaling"]}, fname)
        rgbs = []
        for k, cexpr in enumerate(cols):
            cc = [float(num(t)) for t in cexpr.args[:3]]
            g = float(I.ravel()[k])
            rgbs.append(L(*[fl(t * g) for t in cc]))
        rows = L(*[L(*rgbs[r * n:(r + 1) * n]) for r in range(n)])
    (a, b), (c, d) = ranges
    ras = Expr("Raster", rows, L(L(fl(a), fl(c)), L(fl(b), fl(d))))
    return finish_graphics(L(ras), gopts, [("AspectRatio", 1), ("Frame", True),
                                           ("PlotRange", range_expr(list(ranges))), ("PlotRangePadding", 0)])


_LIC_DEFAULTS = dict(_DEFAULTS)
_LIC_DEFAULTS["RasterSize"] = AUTO
_LIC_NOTE = ("Raster[intensities, {{xmin, ymin}, {xmax, ymax}}] of RasterSize x RasterSize pixels (default 128; row 1 "
             "at the bottom): each pixel is the mean of white noise (numpy default_rng(20), uniform) over the pixel "
             "and LineIntegralConvolutionScale (default 10) unit-pixel steps both ways along the unit field (Euler "
             "steps, stopping at the border or where the field vanishes) - the documented line integral convolution "
             "of white noise; Wolfram's noise and kernel are not documented. Gray by default; a ColorFunction "
             "colours |v| and is modulated by the intensity. ")


@fn("LineIntegralConvolutionPlot", area=AREA, status="added", hold=True,
    notes="LineIntegralConvolutionPlot[{vx, vy}, {x, ...}, {y, ...}] (HoldAll): " + _LIC_NOTE + "The {{vx, vy}, "
          "image} form and other settings raise NotImplementedError. Check: reference (a constant field gives the "
          "running means of the noise along rows in tests).")
def _lineintegralconvolutionplot(f: Held, sx: Held, sy: Held, *opts: Held | Option) -> Expr:
    fname = "LineIntegralConvolutionPlot"
    pos, o_ = hold_args((f, sx, sy) + opts)
    if len(pos) != 3:
        raise ValueError(f"{fname}: expected {fname}[{{vx, vy}}, {{x, xmin, xmax}}, {{y, ymin, ymax}}]")
    fv = pos[0]
    if K.is_list(fv) and len(fv.args) == 2 and K.is_list(fv.args[0]):
        raise NotImplementedError(f"{fname}: convolving a given image is not supported (white noise only)")
    comps, _ = field_spec(fv, 2, fname)
    vars_, ranges = iterators(pos[1:], 2, fname)
    o, gopts = split_options(fname, o_, _LIC_DEFAULTS)
    _unsupported(o, fname, ("RasterSize", "LineIntegralConvolutionScale", "ColorFunction", "ColorFunctionScaling"))
    F = vector_function(compile_all(comps, vars_, ranges, fname))
    return lic_graphics(F, ranges, o, gopts, fname)


@fn("ListLineIntegralConvolutionPlot", area=AREA, status="added",
    notes="ListLineIntegralConvolutionPlot[array] (vectors v_ij at {j, i}, or DataRange) and "
          "[{{{x, y}, {vx, vy}}, ...}] (needs scipy): the field interpolated bilinearly (linearly) and drawn as "
          "LineIntegralConvolutionPlot: " + _LIC_NOTE + "Check: reference (constant data gives the running means "
          "of the noise in tests).")
def _listlineintegralconvolutionplot(data: ListOf[Expr] | MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ListLineIntegralConvolutionPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    o, gopts = split_options(fname, opts_, _LIC_DEFAULTS)
    _unsupported(o, fname, ("RasterSize", "LineIntegralConvolutionScale", "ColorFunction", "ColorFunctionScaling",
                            "DataRange"))
    d = pos[0]
    if K.is_list(d) and len(d.args) == 2 and isinstance(d.args[1], Expr) and d.args[1].head == "Image":
        raise NotImplementedError(f"{fname}: convolving a given image is not supported (white noise only)")
    vd = VectorData(d, 2, fname, o["DataRange"])
    return lic_graphics(vd.interpolator(), vd.ranges, o, gopts, fname)


# ───────────────────────────── complex functions ─────────────────────────────

class ComplexFunction:
    """f(z) on complex arrays (NaN where undefined): sympy-compiled when possible, cross-checked against the
    kernel, else evaluated by the kernel point by point."""

    def __init__(self, f, z, fname):
        self.fname = fname
        self.d = Symbol("plot$w20$z")
        self.held = K.substitute(f, {z.name: self.d})
        try:
            self.sym = ev(self.held)
        except (TypeError, ValueError, NotImplementedError, ArithmeticError, ZeroDivisionError):
            self.sym = None
        self.fast = None
        if self.sym is not None:
            try:
                e = S(self.sym)
            except Exception:
                e = None
            if isinstance(e, sp.Basic):
                extra = {s for s in e.free_symbols if s.name != self.d.name}
                if extra:
                    raise Unevaluated()
                if not any(isinstance(a, sp.core.function.AppliedUndef) for a in sp.preorder_traversal(e)):
                    try:
                        self.fast = sp.lambdify([sp.Symbol(self.d.name)], e, modules=["numpy"])
                        self(_np().array([0.37 + 0.21j]))
                    except Exception:
                        self.fast = None

    def point(self, w):
        e = K.substitute(self.held if self.sym is None else self.sym, {self.d.name: complex(w)})
        try:
            r = ev(Expr("N", e))
        except (TypeError, ValueError, NotImplementedError, ArithmeticError, ZeroDivisionError, OverflowError):
            return complex(math.nan, math.nan)
        if isinstance(r, bool):
            return complex(math.nan, math.nan)
        if isinstance(r, (int, float, complex)):
            c = complex(r)
            return c if cmath.isfinite(c) else complex(math.nan, math.nan)
        v = num(r)
        if v is not None:
            return complex(v)
        if isinstance(r, Expr) and P2._free_of_dummies(r, [self.d]) and not is_sym(r, "Indeterminate",
                                                                                   "ComplexInfinity"):
            if not (isinstance(r, Expr) and r.head in ("DirectedInfinity",)):
                raise Unevaluated()
        return complex(math.nan, math.nan)

    def __call__(self, W):
        np = _np()
        W = np.asarray(W, dtype=complex)
        if self.fast is not None:
            with np.errstate(all="ignore"):
                out = np.broadcast_to(np.asarray(self.fast(W), dtype=complex), W.shape).copy()
            out[~np.isfinite(out)] = complex(math.nan, math.nan)
            return out
        return np.array([self.point(w) for w in W.ravel()], dtype=complex).reshape(W.shape)

    def cross_check(self, samples):
        if self.fast is None:
            return
        np = _np()
        for w in samples:
            a = complex(self(np.array([w]))[0])
            b = self.point(w)
            if cmath.isnan(a) and cmath.isnan(b):
                continue
            if cmath.isnan(a) != cmath.isnan(b) or abs(a - b) > 1e-7 * max(1.0, abs(b)):
                self.fast = None
                return


def complex_value(x, fname, what):
    r = ev(Expr("N", x))
    if isinstance(r, (int, float, complex)) and not isinstance(r, bool):
        c = complex(r)
        if cmath.isfinite(c):
            return c
    v = num(r)
    if v is not None:
        return complex(v)
    if isinstance(r, (Symbol, Expr)) and not K.is_list(r):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be a complex number, not {K_input(x)}")


def complex_iterator(spec, fname, pos=2):
    """{z, zmin, zmax} or {z, n} (= {z, -n - n I, n + n I}, as documented) -> (z, (xmin, xmax), (ymin, ymax))."""
    if isinstance(spec, Expr) and spec.head == "Element":
        raise NotImplementedError(f"{fname}: the region form z \u2208 reg is not supported")
    if not K.is_list(spec) or len(spec.args) not in (2, 3):
        raise ValueError(f"{fname}: argument {pos} must be {{z, zmin, zmax}} or {{z, n}}, not {K_input(spec)}")
    z = P2.var_symbol(spec.args[0], fname)
    if len(spec.args) == 2:
        n = real_arg(ev(spec.args[1]), fname, "n")
        if not n > 0:
            raise ValueError(f"{fname}: n must be positive")
        a, b = complex(-n, -n), complex(n, n)
    else:
        a = complex_value(spec.args[1], fname, "zmin")
        b = complex_value(spec.args[2], fname, "zmax")
    xr = (min(a.real, b.real), max(a.real, b.real))
    yr = (min(a.imag, b.imag), max(a.imag, b.imag))
    if not (xr[0] < xr[1] and yr[0] < yr[1]):
        raise ValueError(f"{fname}: the corners zmin and zmax must span a rectangle")
    return z, xr, yr


def _complex_fn(f, z, xr, yr, fname):
    cf = ComplexFunction(f, z, fname)
    cf.cross_check([complex(xr[0] + 0.31 * (xr[1] - xr[0]), yr[0] + 0.67 * (yr[1] - yr[0])),
                    complex(xr[0] + 0.77 * (xr[1] - xr[0]), yr[0] + 0.21 * (yr[1] - yr[0]))])
    return cf


def arg_color(w):
    """RGBColor of Hue[Arg[w]/(2 Pi) mod 1] (white where w is undefined)."""
    import colorsys
    if cmath.isnan(w):
        return Expr("RGBColor", 1.0, 1.0, 1.0)
    h = (cmath.phase(w) / (2 * math.pi)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return Expr("RGBColor", fl(r), fl(g), fl(b))


def _complex_grid(cf, xr, yr, n):
    np = _np()
    P, T, _, _ = grid_mesh(xr[0], xr[1], yr[0], yr[1], n, n)
    W = cf(P[:, 0] + 1j * P[:, 1])
    return P, T, W


def _colored_complex(P, T, W):
    np = _np()
    ok = np.isfinite(W)
    tris = [t for t in T if ok[t].all()]
    return Expr("GraphicsComplex", pts([tuple(p) for p in P]),
                L(Expr("EdgeForm"), Expr("Polygon", L(*[L(*[int(k) + 1 for k in t]) for t in tris]))),
                Expr("Rule", Symbol("VertexColors"), L(*[arg_color(complex(w)) for w in W])))


def _cplx_args(args, fname, allowed, dim=2):
    pos, o_ = hold_args(args)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected {fname}[f, {{z, zmin, zmax}}]")
    z, xr, yr = complex_iterator(pos[1], fname)
    o, gopts = options(fname, o_, allowed, dim=dim)
    return pos[0], z, xr, yr, o, gopts


_HUE_NOTE = ("Colours: Hue[Arg[f]/(2 Pi) mod 1] (a cyclic colour function of Arg[f], as documented; Wolfram's "
             "default additionally shades by |f|, which is not documented); undefined values leave holes. ")


@fn("ComplexPlot", area=AREA, status="added", hold=True,
    notes="ComplexPlot[f, {z, zmin, zmax}] and [f, {z, n}] (= corners -n - n I and n + n I, as documented) "
          "(HoldAll): Graphics[GraphicsComplex[grid points {Re z, Im z}, {EdgeForm[], Polygon[triangles]}, "
          "VertexColors -> colours]] on a grid of (PlotPoints - 1) 2^MaxRecursion + 1 points per side (default "
          "PlotPoints 41, MaxRecursion 1: 81 x 81). " + _HUE_NOTE + "Options: PlotPoints, MaxRecursion and "
          "Graphics options; ColorFunction settings and others raise NotImplementedError. Check: numeric "
          "cross-check (compiled vs kernel N) and reference (vertex hues equal Arg[f] in tests).")
def _complexplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ComplexPlot"
    f, z, xr, yr, o, gopts = _cplx_args((f, spec) + opts, fname, ("PlotPoints", "MaxRecursion"))
    cf = _complex_fn(f, z, xr, yr, fname)
    P, T, W = _complex_grid(cf, xr, yr, P2._grid_n(o, fname, 41))
    return finish_graphics(L(_colored_complex(P, T, W)), gopts,
                           [("AspectRatio", Symbol("Automatic")), ("Frame", True),
                            ("PlotRange", range_expr([xr, yr])), ("PlotRangePadding", 0)])


@fn("ComplexPlot3D", area=AREA, status="added", hold=True,
    notes="ComplexPlot3D[f, {z, zmin, zmax}] and [f, {z, n}] (HoldAll): the surface {Re z, Im z, Abs[f]} as "
          "Graphics3D[GraphicsComplex[points, {EdgeForm[], Polygon[triangles]}, VertexColors -> colours]] on a "
          "41 x 41 grid by default (PlotPoints 21, MaxRecursion 1). " + _HUE_NOTE + "The height range is the "
          "range of the finite |f| values unless PlotRange is given. Options: PlotPoints, MaxRecursion, PlotRange "
          "and Graphics3D options. Check: numeric cross-check and reference (heights equal Abs[f] in tests).")
def _complexplot3d(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ComplexPlot3D"
    f, z, xr, yr, o, gopts = _cplx_args((f, spec) + opts, fname, ("PlotPoints", "MaxRecursion"), dim=3)
    np = _np()
    cf = _complex_fn(f, z, xr, yr, fname)
    P, T, W = _complex_grid(cf, xr, yr, P2._grid_n(o, fname, 21))
    H = np.abs(W)
    P3 = np.column_stack([P, np.where(np.isfinite(H), H, np.nan)])
    ok = np.isfinite(H)
    tris = [t for t in T if ok[t].all()]
    cols = [arg_color(complex(w)) for w in W]
    pts3 = pts([(p[0], p[1], p[2] if math.isfinite(p[2]) else 0.0) for p in P3])
    g = Expr("GraphicsComplex", pts3, L(Expr("EdgeForm"), Expr("Polygon", L(*[L(*[int(k) + 1 for k in t])
                                                                            for t in tris]))),
             Expr("Rule", Symbol("VertexColors"), L(*cols)))
    zr = P2.data_range(list(H[ok])) if ok.any() else (0.0, 1.0)
    return finish3([g], gopts, [xr, yr, zr], o, fname)


def _rewrite(exc, inner, fname):
    msg = str(exc)
    if msg.startswith(inner + ":"):
        msg = fname + msg[len(inner):]
    elif not msg.startswith(fname + ":"):
        msg = f"{fname}: {msg}"
    return type(exc)(msg)


def _delegate(head, fname, expr):
    try:
        r = ev(expr)
    except Unevaluated:
        raise
    except (TypeError, ValueError, NotImplementedError, ArithmeticError) as exc:
        raise _rewrite(exc, head, fname) from None
    if isinstance(r, Expr) and r.head == head:
        raise Unevaluated()
    return r


_RE, _IM = Symbol("plot$w20$re"), Symbol("plot$w20$im")


def _xy_form(f, spec, opts, fname, head):
    pos, o_ = hold_args((f, spec) + opts)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected {fname}[f, {{z, zmin, zmax}}]")
    z, xr, yr = complex_iterator(pos[1], fname)
    g = K.substitute(pos[0], {z.name: Expr("Plus", _RE, Expr("Times", Symbol("I"), _IM))})
    return _delegate(head, fname, Expr(head, g, L(_RE, xr[0], xr[1]), L(_IM, yr[0], yr[1]), *o_))


def real_grid(cf, P):
    np = _np()
    W = cf(P[:, 0] + 1j * P[:, 1])
    re = W.real.copy()
    bad = ~np.isfinite(W) | (np.abs(W.imag) > 1e-12 * np.maximum(1.0, np.abs(re)))
    re[bad] = np.nan
    return re


@fn("ComplexContourPlot", area=AREA, status="added", hold=True,
    notes="ComplexContourPlot[f, {z, zmin, zmax}] and [f, {z, n}] (HoldAll): the filled contour plot of the "
          "real-valued f over z = x + I y, as ContourPlot (W16 layout: band Polygons coloured by the band "
          "midpoint, contour Lines; levels from Contours); [f == g, ...] gives the curves f = g and [{f1, f2, ...}, "
          "...] / [{f1 == g1, ...}, ...] contour lines of each (as documented), one default colour per item. Grid "
          "of (PlotPoints - 1) 2^MaxRecursion + 1 points per side (default 41, 1: 81 x 81); non-real values leave "
          "holes. Options: Contours, ContourShading, ContourStyle, ColorFunction, PlotPoints, MaxRecursion and "
          "Graphics options. Check: numeric cross-check and reference (contour vertices satisfy f = level in "
          "tests).")
def _complexcontourplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ComplexContourPlot"
    f, z, xr, yr, o, gopts = _cplx_args((f, spec) + opts, fname, ("PlotPoints", "MaxRecursion", "Contours",
                                                                   "ContourShading", "ContourStyle",
                                                                   "ColorFunction", "ColorFunctionScaling"))
    np = _np()
    n = P2._grid_n(o, fname, 41)
    P, T, _, _ = grid_mesh(xr[0], xr[1], yr[0], yr[1], n, n)
    items, is_list = P2._curves_of(f, None)
    if not is_list and not (isinstance(f, Expr) and f.head == "Equal"):
        V = real_grid(_complex_fn(f, z, xr, yr, fname), P)
        return P2.contour_graphics(fname, P, V, T, xr, yr, o, gopts)
    content = []
    for i, g in enumerate(items):
        eq = isinstance(g, Expr) and g.head == "Equal" and len(g.args) == 2
        h = Expr("Plus", g.args[0], Expr("Times", -1, g.args[1])) if eq else g
        V = real_grid(_complex_fn(h, z, xr, yr, fname), P)
        fin = V[np.isfinite(V)]
        if eq:
            levels = [0.0]
        else:
            vlo, vhi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
            levels = P2.contour_levels(o["Contours"], vlo, vhi, fname)
        lines = [pts(ln) for lev in levels for ln in P2.iso_lines(P, V, T, lev)]
        if lines:
            content.append(L(default_color(i), Expr("AbsoluteThickness", 1.6), Expr("Line", L(*lines))))
    return finish_graphics(L(*content), gopts, [("AspectRatio", 1), ("Frame", True),
                                                ("PlotRange", range_expr([xr, yr])), ("PlotRangePadding", 0)])


@fn("ComplexRegionPlot", area=AREA, status="added", hold=True,
    notes="ComplexRegionPlot[pred, {z, zmin, zmax}] and [pred, {z, n}] (HoldAll): exactly RegionPlot[pred /. z -> "
          "x + I y, {x, Re zmin, Re zmax}, {y, Im zmin, Im zmax}, opts] with RegionPlot's options and layout. "
          "Check: inherits RegionPlot's checks; reference (polygon vertices satisfy pred in tests).")
def _complexregionplot(pred: Held, spec: Held, *opts: Held | Option) -> Expr:
    return _xy_form(pred, spec, opts, "ComplexRegionPlot", "RegionPlot")


def _complex_field(f, spec, opts, fname, allowed):
    f, z, xr, yr, o, gopts = _cplx_args((f, spec) + opts, fname, allowed)
    np = _np()
    cf = _complex_fn(f, z, xr, yr, fname)

    def F(x, y):
        w = cf(np.asarray(x, dtype=float) + 1j * np.asarray(y, dtype=float))
        return np.stack([w.real, w.imag], axis=-1)
    return F, [xr, yr], o, gopts


@fn("ComplexVectorPlot", area=AREA, status="added", hold=True,
    notes="ComplexVectorPlot[f, {z, zmin, zmax}] and [f, {z, n}] (HoldAll): arrows of the field {Re f, Im f} at "
          "z = x + I y (as documented). " + _VEC_NOTE.format(n=15) + "Options: VectorPoints, VectorStyle, "
          "VectorColorFunction(Scaling), PlotRange and Graphics options. Check: numeric cross-check and reference "
          "(arrow directions equal {Re f, Im f} in tests).")
def _complexvectorplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ComplexVectorPlot"
    F, ranges, o, gopts = _complex_field(f, spec, opts, fname, ("VectorPoints", "VectorStyle", "VectorColorFunction",
                                                                "VectorColorFunctionScaling"))
    P, cell = cell_centres(ranges, _vec_n(o, fname, 15))
    ar, _ = arrows(P, F(P[:, 0], P[:, 1]), cell, o, fname)
    return finish2(ar, gopts, ranges, o, fname)


@fn("ComplexStreamPlot", area=AREA, status="added", hold=True,
    notes="ComplexStreamPlot[f, {z, zmin, zmax}] and [f, {z, n}] (HoldAll): streamlines of the field {Re f, Im f}. "
          + _STREAM_NOTE + "Options: StreamPoints, StreamStyle, StreamColorFunction(Scaling), PlotRange and "
          "Graphics options. Check: self-check (streamline tangency, raises ArithmeticError).")
def _complexstreamplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ComplexStreamPlot"
    F, ranges, o, gopts = _complex_field(f, spec, opts, fname, ("StreamPoints", "StreamStyle", "StreamColorFunction",
                                                                "StreamColorFunctionScaling"))
    lines = streamlines(F, ranges, _stream_n(o, fname), fname)
    return finish2(stream_groups(lines, F, o, fname), gopts, ranges, o, fname)


def _complex_items(d, fname):
    """[[w, ...]] datasets of complex numbers (missing values skipped)."""
    d, _, _ = P2._unwrap(d, fname)
    if not K.is_list(d):
        if isinstance(d, (Symbol, Expr)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a list of complex numbers, not {K_input(d)}")
    if d.args and all(K.is_list(a) for a in d.args):
        return [_complex_items(a, fname)[0] for a in d.args]
    out = []
    for e in d.args:
        if P2.is_missing(e):
            continue
        r = ev(Expr("N", e))
        if isinstance(r, (int, float, complex)) and not isinstance(r, bool):
            out.append(complex(r))
        elif num(r) is not None:
            out.append(complex(num(r)))
        elif isinstance(r, (Symbol, Expr)) and not K.is_list(r):
            raise Unevaluated()
        else:
            raise TypeError(f"{fname}: {K_input(e)} is not a complex number")
    return [out]


@fn("ComplexListPlot", area=AREA, status="added",
    notes="ComplexListPlot[{z1, z2, ...}] and [{data1, data2, ...}]: exactly ListPlot[{{Re z1, Im z1}, ...}, opts] "
          "(points in the complex plane, as documented) with AspectRatio -> Automatic unless given; ListPlot's "
          "options and layout; missing values are skipped and symbolic entries leave it unevaluated. Check: "
          "reference (point coordinates equal {Re z, Im z} in tests).")
def _complexlistplot(data: ListOf[Expr], *opts: Option) -> Expr:
    fname = "ComplexListPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    sets = _complex_items(pos[0], fname)
    pdata = [L(*[L(fl(w.real), fl(w.imag)) for w in s]) for s in sets]
    arg = pdata[0] if len(sets) == 1 and not (K.is_list(pos[0]) and pos[0].args and K.is_list(pos[0].args[0])) \
        else L(*pdata)
    names = {G.opt_name(r) for r in opts_}
    extra = [] if "AspectRatio" in names else [Expr("Rule", Symbol("AspectRatio"), Symbol("Automatic"))]
    return _delegate("ListPlot", fname, Expr("ListPlot", arg, *opts_, *extra))


@fn("ReImPlot", area=AREA, status="added", hold=True,
    notes="ReImPlot[f, {x, xmin, xmax}] and [{f1, f2, ...}, ...] (HoldAll): exactly Plot[{Re[f], Im[f]}, {x, xmin, "
          "xmax}, opts] (Re[f1], Im[f1], Re[f2], ... for lists), with Plot's sampling, options and layout. Check: "
          "inherits Plot's checks; reference (curve values equal Re/Im of f in tests).")
def _reimplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "ReImPlot"
    pos, o_ = hold_args((f, spec) + opts)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected {fname}[f, {{x, xmin, xmax}}]")
    fs, _ = P2._curves_of(pos[0], None)
    curves = []
    for g in fs:
        curves += [Expr("Re", g), Expr("Im", g)]
    return _delegate("Plot", fname, Expr("Plot", L(*curves), pos[1], *o_))


@fn("AbsArgPlot", area=AREA, status="added", hold=True,
    notes="AbsArgPlot[f, {x, xmin, xmax}] (HoldAll): the curve |f(x)| (adaptive sampling as Plot: 50 initial "
          "points, MaxRecursion 6, split where undefined) as Line[{{x, Abs[f]}, ...}, VertexColors -> colours] "
          "with colours Hue[Arg[f]/(2 Pi) mod 1] at each sample (the documented colouring by Arg). Lists of "
          "functions and other settings raise NotImplementedError. Check: numeric cross-check and reference "
          "(vertices equal Abs[f], colours Arg[f] in tests).")
def _absargplot(f: Held, spec: Held, *opts: Held | Option) -> Expr:
    fname = "AbsArgPlot"
    pos, o_ = hold_args((f, spec) + opts)
    if len(pos) != 2:
        raise ValueError(f"{fname}: expected {fname}[f, {{x, xmin, xmax}}]")
    if K.is_list(pos[0]):
        raise NotImplementedError(f"{fname}: lists of functions are not supported")
    x, a, b = iterator(pos[1], fname)
    o, gopts = options(fname, o_, ("PlotPoints", "MaxRecursion"))
    np = _np()
    cf = _complex_fn(pos[0], x, (a, b), (0.0, 1.0), fname)
    cf.cross_check([complex(a + 0.31 * (b - a)), complex(a + 0.77 * (b - a))])

    def absf(xs):
        return np.abs(cf(np.asarray(xs, dtype=float) + 0j))
    n = int_option(o["PlotPoints"], 50, fname, "PlotPoints", 2)
    rec = int_option(o["MaxRecursion"], 6, fname, "MaxRecursion", 0) if not is_sym(o["MaxRecursion"], "Automatic") \
        else 6
    lines = P2.sample_1d(absf, a, b, n, rec)
    content = []
    ys = []
    for ln in lines:
        xs_ = np.array([p[0] for p in ln])
        cols = [arg_color(complex(w)) for w in cf(xs_ + 0j)]
        ys += [p[1] for p in ln]
        content.append(Expr("Line", pts(ln), Expr("Rule", Symbol("VertexColors"), L(*cols))))
    yr = P2.data_range(ys) if ys else (0.0, 1.0)
    return finish_graphics(L(Expr("AbsoluteThickness", 3), *content), gopts,
                           [("AspectRatio", P2.GOLDEN), ("Axes", True), ("PlotRange", range_expr([(a, b), span(yr)]))])


@fn("ComplexArrayPlot", area=AREA, status="added",
    notes="ComplexArrayPlot[array]: Graphics[Raster[colours, {{0, 0}, {ncols, nrows}}]] with row 1 of the array at "
          "the top (as ArrayPlot), colours Hue[Arg[z]/(2 Pi) mod 1] (black for 0, white for missing or non-numeric "
          "entries); Wolfram's additional |z| shading is not documented and not reproduced. Options: Graphics "
          "options; ColorFunction and others raise NotImplementedError. Check: reference (cell colours equal the "
          "hue of Arg[z] in tests).")
def _complexarrayplot(data: MatrixOf[Expr], *opts: Option) -> Expr:
    fname = "ComplexArrayPlot"
    pos, opts_ = list_args((data,) + opts, fname)
    o, gopts = options(fname, opts_, ())
    d, _, _ = P2._unwrap(pos[0], fname)
    if not K.is_list(d) or not d.args or not all(K.is_list(r) for r in d.args):
        if isinstance(d, (Symbol, Expr)) and not K.is_list(d):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a matrix of complex numbers")
    if len({len(r.args) for r in d.args}) != 1 or not d.args[0].args:
        raise ValueError(f"{fname}: the array must be rectangular and non-empty")
    rows = []
    for r in d.args:
        row = []
        for e in r.args:
            if P2.is_missing(e):
                row.append(L(1.0, 1.0, 1.0))
                continue
            rr = ev(Expr("N", e))
            if isinstance(rr, (int, float, complex)) and not isinstance(rr, bool):
                w = complex(rr)
            elif num(rr) is not None:
                w = complex(num(rr))
            elif isinstance(rr, (Symbol, Expr)) and not K.is_list(rr):
                raise Unevaluated()
            else:
                raise TypeError(f"{fname}: {K_input(e)} is not a complex number")
            c = Expr("RGBColor", 0.0, 0.0, 0.0) if w == 0 else arg_color(w)
            row.append(L(*c.args))
        rows.append(L(*row))
    nr, nc = len(rows), len(rows[0].args)
    ras = Expr("Raster", L(*rows[::-1]), L(L(0, 0), L(nc, nr)))
    return finish_graphics(L(ras), gopts, [("AspectRatio", Symbol("Automatic")), ("Frame", True),
                                           ("PlotRange", range_expr([(0.0, float(nc)), (0.0, float(nr))])),
                                           ("PlotRangePadding", 0)])


# ───────────────────────────── surfaces of revolution ─────────────────────────────

_THETA = Symbol("plot$w20$theta")


@fn("RevolutionPlot3D", area=AREA, status="added", hold=True,
    notes="RevolutionPlot3D[fz, {t, tmin, tmax}] (= the curve {t, 0, fz} rotated about the z axis, as "
          "documented), [{fx, fz}, ...], [{fx, fy, fz}, ...], [{{f}, {g}, ...}, ...] (several surfaces) and with "
          "{theta, thetamin, thetamax} (default 0 to 2 Pi, counterclockwise from the x axis): exactly "
          "ParametricPlot3D[{fx Cos[theta] - fy Sin[theta], fx Sin[theta] + fy Cos[theta], fz}, {t, ...}, {theta, "
          "...}, opts] with ParametricPlot3D's sampling, options and layout. RevolutionAxis other than the z axis "
          "raises NotImplementedError. Check: inherits ParametricPlot3D's checks; reference (vertices lie at "
          "radius t and height fz in tests).")
def _revolutionplot3d(f: Held, st: Held, *rest: Held | Option) -> Expr:
    fname = "RevolutionPlot3D"
    pos, o_ = hold_args((f, st) + rest)
    if len(pos) not in (2, 3):
        raise ValueError(f"{fname}: expected {fname}[f, {{t, tmin, tmax}}] or with {{theta, min, max}}")
    t, _, _ = iterator(pos[1], fname)
    if len(pos) == 3:
        th, _, _ = iterator(pos[2], fname, 3)
        th_spec = pos[2]
    else:
        th, th_spec = _THETA, L(_THETA, 0, Expr("Times", 2, Symbol("Pi")))
    opts = []
    for r in o_:
        if G.opt_name(r) == "RevolutionAxis":
            v = ev(r.args[1])
            if not (is_sym(v, "Automatic") or (K.is_list(v) and [num(c) for c in v.args] == [0.0, 0.0, 1.0])):
                raise NotImplementedError(f"{fname}: RevolutionAxis -> {K_input(v)} is not supported (z axis only)")
            continue
        opts.append(r)
    fs = pos[0]
    if isinstance(fs, Expr) and fs.head == "Evaluate" and len(fs.args) == 1:
        fs = ev(fs.args[0])

    def surface(g):
        if K.is_list(g):
            if len(g.args) == 2:
                fx, fy, fz = g.args[0], 0, g.args[1]
            elif len(g.args) == 3:
                fx, fy, fz = g.args
            else:
                raise ValueError(f"{fname}: give fz, {{fx, fz}} or {{fx, fy, fz}}")
        else:
            fx, fy, fz = t, 0, g
        c, s = Expr("Cos", th), Expr("Sin", th)
        return L(Expr("Plus", Expr("Times", fx, c), Expr("Times", -1, fy, s)),
                 Expr("Plus", Expr("Times", fx, s), Expr("Times", fy, c)), fz)
    if K.is_list(fs) and fs.args and all(K.is_list(g) and len(g.args) == 1 for g in fs.args):
        surf = L(*[surface(g.args[0]) for g in fs.args])
    else:
        surf = surface(fs)
    return _delegate("ParametricPlot3D", fname, Expr("ParametricPlot3D", surf, pos[1], th_spec, *opts))


SKIPPED: dict = {}
