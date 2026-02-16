"""Plotting functions for MikoshiLang — wraps matplotlib."""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
from .expr import Expr, Symbol
from .math_ops import to_sympy
import sympy as sp


def _expr_to_callable(expr, var: Symbol):
    """Convert a MikoshiLang expression to a numpy-callable function."""
    sexpr = to_sympy(expr)
    svar = sp.Symbol(var.name)
    fn = sp.lambdify(svar, sexpr, modules=["numpy"])
    return fn


def Plot(expr, spec: Expr, **kwargs) -> Any:
    """Plot[expr, {var, min, max}] — 2D function plot.
    
    Returns a matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (isinstance(spec, Expr) and spec.head == "List" and len(spec.args) == 3):
        raise ValueError("Plot spec must be {var, min, max}")

    var, xmin, xmax = spec.args
    if isinstance(var, Symbol):
        pass
    else:
        var = Symbol(str(var))

    xmin = float(xmin) if not isinstance(xmin, float) else xmin
    xmax = float(xmax) if not isinstance(xmax, float) else xmax

    xs = np.linspace(xmin, xmax, 500)

    fig, ax = plt.subplots()

    # Handle list of expressions
    exprs = []
    if isinstance(expr, Expr) and expr.head == "List":
        exprs = list(expr.args)
    else:
        exprs = [expr]

    for e in exprs:
        fn = _expr_to_callable(e, var)
        try:
            ys = fn(xs)
            ax.plot(xs, ys, **kwargs)
        except Exception:
            ys = np.array([fn(x) for x in xs])
            ax.plot(xs, ys, **kwargs)

    ax.set_xlabel(str(var))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def ListPlot(data: Expr, **kwargs) -> Any:
    """ListPlot[{y1, y2, ...}] or ListPlot[{{x1,y1}, {x2,y2}, ...}] — scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (isinstance(data, Expr) and data.head == "List"):
        raise ValueError("ListPlot expects a List")

    fig, ax = plt.subplots()

    if data.args and isinstance(data.args[0], Expr) and data.args[0].head == "List":
        xs = [float(p.args[0]) for p in data.args]
        ys = [float(p.args[1]) for p in data.args]
    else:
        xs = list(range(1, len(data.args) + 1))
        ys = [float(y) for y in data.args]

    ax.scatter(xs, ys, **kwargs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def ListLinePlot(data: Expr, **kwargs) -> Any:
    """ListLinePlot[{y1, y2, ...}] — line plot of data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not (isinstance(data, Expr) and data.head == "List"):
        raise ValueError("ListLinePlot expects a List")

    fig, ax = plt.subplots()

    if data.args and isinstance(data.args[0], Expr) and data.args[0].head == "List":
        xs = [float(p.args[0]) for p in data.args]
        ys = [float(p.args[1]) for p in data.args]
    else:
        xs = list(range(1, len(data.args) + 1))
        ys = [float(y) for y in data.args]

    ax.plot(xs, ys, **kwargs)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
