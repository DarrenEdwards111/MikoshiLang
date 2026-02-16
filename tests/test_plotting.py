"""Tests for plotting functions."""

import pytest

matplotlib = pytest.importorskip("matplotlib", reason="matplotlib not compatible in this environment")

from mikoshilang.expr import Expr, Symbol
from mikoshilang.plotting import Plot, ListPlot, ListLinePlot


@pytest.fixture(autouse=True)
def _mpl_backend():
    matplotlib.use("Agg")


def test_plot_returns_figure():
    import matplotlib.figure
    x = Symbol("x")
    spec = Expr("List", x, -3.14, 3.14)
    fig = Plot(Expr("Sin", x), spec)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_multiple():
    import matplotlib.figure
    x = Symbol("x")
    spec = Expr("List", x, -3.14, 3.14)
    exprs = Expr("List", Expr("Sin", x), Expr("Cos", x))
    fig = Plot(exprs, spec)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_list_plot_returns_figure():
    import matplotlib.figure
    data = Expr("List", 1, 4, 9, 16)
    fig = ListPlot(data)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_list_plot_xy():
    import matplotlib.figure
    data = Expr("List",
                Expr("List", 1, 2),
                Expr("List", 2, 4),
                Expr("List", 3, 6))
    fig = ListPlot(data)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_list_line_plot_returns_figure():
    import matplotlib.figure
    data = Expr("List", 1, 4, 9, 16)
    fig = ListLinePlot(data)
    assert isinstance(fig, matplotlib.figure.Figure)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_invalid_spec():
    x = Symbol("x")
    with pytest.raises(ValueError):
        Plot(Expr("Sin", x), Expr("List", x, -1))
