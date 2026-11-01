"""Plotting.  Each function returns a matplotlib Figure (plotly Figure for
Interactive3D, matplotlib FuncAnimation for AnimatePlot) without calling
plt.show(); the Jupyter kernel renders returned figures.  Legacy behaviour:
Plot3D/ContourPlot/ParametricPlot/PolarPlot/BarChart/PieChart/ScatterPlot3D/
ImplicitPlot/VectorPlot were shadowed by inert descriptors that returned the
input unevaluated, Plot2D printed a warning and drew an empty plot when the
expression failed to evaluate, and the real functions blocked in plt.show()."""
import sympy as sp

from ..expr import Expr, Symbol
from .core import Unevaluated, fn, error, record, S, R, L, rule, list_arg, num_list, number, is_list, variable, sym

AREA = "plots"


def _need(mod):
    try:
        return __import__(mod)
    except ImportError:
        raise NotImplementedError(f"plotting requires {mod} (pip install mikoshilang[plotting])")


def _range(r, fname):
    items = list_arg(r, fname, "range {x, min, max}")
    if len(items) != 3:
        raise ValueError(f"{fname}: range must be {{x, min, max}}")
    v = variable(items[0], fname)
    a, b = float(number(items[1], fname, "range min")), float(number(items[2], fname, "range max"))
    if not a < b:
        raise ValueError(f"{fname}: range min must be less than max")
    return v, a, b


def _viz(name, nranges, notes, kwargs=None):
    @fn(name, area=AREA, status="fixed", notes=notes)
    def impl(*args):
        _need("matplotlib")
        _need("numpy")
        if len(args) != 1 + nranges and name != "Heatmap":
            raise ValueError(f"{name} called with {len(args)} argument(s); expected {1 + nranges}")
        for r in args[1:]:
            if name != "AnimatePlot":
                _range(r, name)
        from .. import visualization as V
        try:
            return getattr(V, name)(*args, show=False, **(kwargs or {}))
        except ImportError as exc:
            raise NotImplementedError(f"{name}: {exc}")
    return impl


_viz("Plot2D", 1, "Plot2D[f or {f1, ...}, {x, a, b}] line plot; raises when f cannot be evaluated (legacy drew an empty plot)"
                  ". Check: reference (differential cases in tests/library).")
_viz("Plot3D", 2, "surface plot of f(x, y); previously shadowed by an inert descriptor"
                  ". Check: reference (differential cases in tests/library).")
_viz("ContourPlot", 2, "contour plot of f(x, y); previously shadowed by an inert descriptor"
                       ". Check: reference (differential cases in tests/library).")
_viz("VectorFieldPlot", 2, "quiver plot of {fx, fy}"
                           ". Check: reference (differential cases in tests/library).")
_viz("ParametricPlot", 1, "ParametricPlot[{x(t), y(t)[, z(t)]}, {t, a, b}]; previously shadowed by an inert descriptor"
                          ". Check: reference (differential cases in tests/library).")
_viz("PolarPlot", 1, "PolarPlot[r(θ), {θ, a, b}]; previously shadowed by an inert descriptor"
                     ". Check: reference (differential cases in tests/library).")
_viz("Interactive3D", 2, "plotly surface (requires plotly)"
                         ". Check: reference (differential cases in tests/library).")


@fn("AnimatePlot", area=AREA, status="fixed",
    notes="AnimatePlot[f(x, p), x, {p, a, b}] matplotlib animation over p with x in [-10, 10]"
          ". Check: reference (differential cases in tests/library).")
def _anim(e, x, pr):
    _need("matplotlib")
    _need("numpy")
    variable(x, "AnimatePlot")
    _range(pr, "AnimatePlot")
    from .. import visualization as V
    return V.AnimatePlot(e, x, pr, show=False)


@fn("Heatmap", area=AREA, status="fixed", notes="image of a numeric matrix (rectangular, real entries)"
                                                ". Check: reference (differential cases in tests/library).")
def _heat(m):
    rows = list_arg(m, "Heatmap", "matrix", 1)
    if not all(is_list(r) for r in rows) or len({len(r.args) for r in rows}) != 1:
        raise ValueError("Heatmap: argument must be a rectangular numeric matrix")
    for r in rows:
        num_list(r, "Heatmap", "matrix row")
    _need("matplotlib")
    _need("numpy")
    from .. import visualization as V
    return V.Heatmap(m, show=False)


def _plt():
    _need("matplotlib")
    import matplotlib.pyplot as plt
    return plt


def _labels_values(data, fname):
    items = list_arg(data, fname, "data", 1)
    if all(isinstance(i, Expr) and i.head == "Rule" for i in items):
        labels = [str(i.args[0]) for i in items]
        vals = [float(number(i.args[1], fname, "value")) for i in items]
    else:
        labels = [str(k + 1) for k in range(len(items))]
        vals = [float(v) for v in num_list(data, fname, "data")]
    return labels, vals


@fn("BarChart", area=AREA, status="fixed",
    notes="bar chart of {v1, ...} or {label -> v, ...}; previously an inert descriptor"
          ". Check: reference (differential cases in tests/library).")
def _bar(data):
    labels, vals = _labels_values(data, "BarChart")
    plt = _plt()
    fig, ax = plt.subplots()
    ax.bar(labels, vals)
    return fig


@fn("PieChart", area=AREA, status="fixed",
    notes="pie chart of nonnegative values {v1, ...} or {label -> v, ...}; previously an inert descriptor"
          ". Check: reference (differential cases in tests/library).")
def _pie(data):
    labels, vals = _labels_values(data, "PieChart")
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError("PieChart: values must be nonnegative and not all zero")
    plt = _plt()
    fig, ax = plt.subplots()
    ax.pie(vals, labels=labels)
    return fig


@fn("ScatterPlot3D", area=AREA, status="fixed",
    notes="ScatterPlot3D[xs, ys, zs] with equal-length coordinate lists; previously an inert descriptor"
          ". Check: reference (differential cases in tests/library).")
def _sc3(xs, ys, zs):
    cs = [[float(v) for v in num_list(c, "ScatterPlot3D", "coordinates")] for c in (xs, ys, zs)]
    if len({len(c) for c in cs}) != 1:
        raise ValueError("ScatterPlot3D: coordinate lists must have equal length")
    plt = _plt()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*cs)
    return fig


def _grid(fname, xr, yr, n=200):
    np = _need("numpy")
    x, a, b = _range(xr, fname)
    y, c, d = _range(yr, fname)
    X, Y = np.meshgrid(np.linspace(a, b, n), np.linspace(c, d, n))
    return np, x, y, X, Y


def _eval2(e, x, y, fname, np, X, Y):
    f = sym(e, fname)
    if f.free_symbols - {x, y}:
        raise ValueError(f"{fname}: expression has free symbols other than {x}, {y}")
    try:
        return np.broadcast_to(np.asarray(sp.lambdify((x, y), f, "numpy")(X, Y), dtype=float), X.shape)
    except Exception as exc:
        raise ValueError(f"{fname}: could not evaluate {e} on the plot range: {exc}") from exc


@fn("ImplicitPlot", area=AREA, status="fixed",
    notes="ImplicitPlot[lhs == rhs, {x, a, b}, {y, c, d}] zero contour of lhs - rhs; previously an inert descriptor"
          ". Check: reference (differential cases in tests/library).")
def _implicit(eq, xr, yr):
    if isinstance(eq, Expr) and eq.head == "Equal" and len(eq.args) == 2:
        e = Expr("Plus", eq.args[0], Expr("Times", -1, eq.args[1]))
    else:
        e = eq
    np, x, y, X, Y = _grid("ImplicitPlot", xr, yr)
    Z = _eval2(e, x, y, "ImplicitPlot", np, X, Y)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.contour(X, Y, Z, levels=[0])
    return fig


@fn("VectorPlot", area=AREA, status="fixed",
    notes="VectorPlot[{fx, fy}, {x, a, b}, {y, c, d}] quiver plot (Wolfram name); previously an inert descriptor"
          ". Check: reference (differential cases in tests/library).")
def _vplot(fs, xr, yr):
    comps = list_arg(fs, "VectorPlot", "vector field", 2)
    if len(comps) != 2:
        raise ValueError("VectorPlot: field must be {fx, fy}")
    np, x, y, X, Y = _grid("VectorPlot", xr, yr, 20)
    U = _eval2(comps[0], x, y, "VectorPlot", np, X, Y)
    W = _eval2(comps[1], x, y, "VectorPlot", np, X, Y)
    plt = _plt()
    fig, ax = plt.subplots()
    ax.quiver(X, Y, U, W)
    return fig
