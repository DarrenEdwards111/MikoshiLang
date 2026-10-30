"""Plot and machine-learning heads. With the optional dependency missing every head must raise
NotImplementedError (never return an inert placeholder); with it installed the returned figure or model
output is checked against values computed directly."""
import importlib.util
import math

import pytest

from _harness import Raises, Pred, check, ids

HAVE_MPL = importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("numpy") is not None
HAVE_SK = importlib.util.find_spec("sklearn") is not None
HAVE_PLOTLY = importlib.util.find_spec("plotly") is not None

if HAVE_MPL:
    import matplotlib
    matplotlib.use("Agg")


def fig(pred=lambda f: True, what="a matplotlib Figure"):
    def f(r):
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        ok = isinstance(r, Figure) and bool(r.axes) and pred(r)
        plt.close("all")
        return ok
    return Pred(f, what)


def _line_matches(fn):
    def p(f):
        ln = f.axes[0].get_lines()[0]
        return all(abs(y - fn(x)) < 1e-9 for x, y in zip(ln.get_xdata(), ln.get_ydata()))
    return p


def _bars(vals):
    return lambda f: [round(b.get_height(), 12) for b in f.axes[0].patches] == vals


PLOT_OK = [
    ("Plot2D[Sin[x], {x, 0, 1}]", fig(_line_matches(math.sin), "line of sin")),
    ("Plot2D[x^2, {x, -1, 1}]", fig(_line_matches(lambda x: x * x), "line of x^2")),
    ("Plot2D[3, {x, 0, 1}]", fig(_line_matches(lambda x: 3.0), "constant line")),
    ("Plot2D[Sin[x], {x, 1, 0}]", Raises(ValueError, "less than")), ("Plot2D[y z, {x, 0, 1}]", Raises(ValueError)),
    ("Plot3D[x y, {x, 0, 1}, {y, 0, 1}]", fig()), ("Plot3D[1, {x, 0, 1}, {y, 0, 1}]", fig()),
    ("Plot3D[x, {x, 0, 1}, {y, 0, 1}]", fig()), ("Plot3D[x, {x, 0, 1}]", Raises()),
    ("ContourPlot[x^2 + y^2, {x, -1, 1}, {y, -1, 1}]", fig()), ("ContourPlot[x - y, {x, 0, 1}, {y, 0, 1}]", fig()),
    ("ContourPlot[Sin[x] Cos[y], {x, 0, 3}, {y, 0, 3}]", fig()), ("ContourPlot[x, {x, 1}, {y, 0, 1}]", Raises()),
    ("VectorFieldPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}]", fig()), ("VectorFieldPlot[{1, 0}, {x, 0, 1}, {y, 0, 1}]", fig()),
    ("VectorFieldPlot[{x, y}, {x, -1, 1}, {y, -1, 1}]", fig()), ("VectorFieldPlot[{x, y}, {x, 0, 1}]", Raises()),
    ("ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2 Pi}]",
     fig(lambda f: all(abs(a * a + b * b - 1) < 1e-9 for a, b in f.axes[0].get_lines()[0].get_xydata()), "unit circle")),
    ("ParametricPlot[{t, t^2}, {t, 0, 1}]", fig()), ("ParametricPlot[{t, 1}, {t, 0, 1}]", fig()),
    ("ParametricPlot[{t, t}, {t, 2, 1}]", Raises()),
    ("PolarPlot[1, {t, 0, 2 Pi}]", fig()), ("PolarPlot[t, {t, 0, Pi}]", fig()), ("PolarPlot[Cos[t], {t, 0, Pi}]", fig()),
    ("PolarPlot[1, {t, 0}]", Raises()),
    ("AnimatePlot[Sin[a x], x, {a, 1, 2}]", Pred(lambda r: r is not None, "animation")),
    ("AnimatePlot[a x, x, {a, 0, 1}]", Pred(lambda r: r is not None, "animation")),
    ("AnimatePlot[x + a, x, {a, 0, 2}]", Pred(lambda r: r is not None, "animation")),
    ("AnimatePlot[x, x, {a, 2, 1}]", Raises()),
    ("Heatmap[{{1, 2}, {3, 4}}]", fig(lambda f: f.axes[0].images[0].get_array().tolist() == [[1, 2], [3, 4]], "image data")),
    ("Heatmap[{{0}}]", fig()), ("Heatmap[{{1, 2}, {3}}]", Raises(ValueError, "rectangular")), ("Heatmap[{{x}}]", "Heatmap[{{x}}]"),
    ("BarChart[{1, 2, 3}]", fig(_bars([1, 2, 3]), "bar heights")), ("BarChart[{a -> 2, b -> 5}]", fig(_bars([2, 5]), "bar heights")),
    ("BarChart[{1/2}]", fig(_bars([0.5]), "bar heights")), ("BarChart[{x}]", "BarChart[{x}]"),
    ("PieChart[{1, 1, 2}]", fig(lambda f: len(f.axes[0].patches) == 3, "3 wedges")), ("PieChart[{a -> 1, b -> 3}]", fig()),
    ("PieChart[{5}]", fig()), ("PieChart[{-1, 2}]", Raises(ValueError, "nonnegative")), ("PieChart[{0, 0}]", Raises(ValueError)),
    ("ScatterPlot3D[{1, 2}, {3, 4}, {5, 6}]", fig()), ("ScatterPlot3D[{1}, {1}, {1}]", fig()),
    ("ScatterPlot3D[{0, 1, 2}, {0, 1, 2}, {0, 1, 4}]", fig()), ("ScatterPlot3D[{1, 2}, {3}, {5, 6}]", Raises(ValueError, "equal length")),
    ("ImplicitPlot[x^2 + y^2 == 1, {x, -2, 2}, {y, -2, 2}]", fig()), ("ImplicitPlot[x == y, {x, -1, 1}, {y, -1, 1}]", fig()),
    ("ImplicitPlot[x^2 - y, {x, -1, 1}, {y, -1, 1}]", fig()), ("ImplicitPlot[x == z, {x, -1, 1}, {y, -1, 1}]", Raises(ValueError, "free symbols")),
    ("VectorPlot[{-y, x}, {x, -1, 1}, {y, -1, 1}]", fig()), ("VectorPlot[{1, 1}, {x, 0, 1}, {y, 0, 1}]", fig()),
    ("VectorPlot[{x, -y}, {x, -1, 1}, {y, -1, 1}]", fig()), ("VectorPlot[{x}, {x, 0, 1}, {y, 0, 1}]", Raises(ValueError)),
]
PLOT_HEADS = sorted({c[0].split("[")[0] for c in PLOT_OK})

if HAVE_PLOTLY:
    PLOT_OK += [("Interactive3D[x y, {x, 0, 1}, {y, 0, 1}]", Pred(lambda r: r is not None, "plotly figure")),
                ("Interactive3D[1, {x, 0, 1}, {y, 0, 1}]", Pred(lambda r: r is not None, "plotly figure")),
                ("Interactive3D[x, {x, 0, 1}, {y, 1, 0}]", Raises())]
else:
    PLOT_OK += [("Interactive3D[x y, {x, 0, 1}, {y, 0, 1}]", Raises(NotImplementedError))] * 3

if not HAVE_MPL:
    # every plot head must raise a clear error rather than return a placeholder
    PLOT_OK = [(src, Raises(NotImplementedError)) for src, exp in PLOT_OK if not isinstance(exp, (Raises, str))]


def _rule(r, key):
    for x in r.args:
        if str(x.args[0]) == key:
            return x.args[1]
    raise KeyError(key)


def _flt(m):
    return [[float(v) for v in row.args] for row in m.args] if hasattr(m.args[0], "args") and m.args[0].args else [float(v) for v in m.args]


ML = [
    ("KMeans[{{0, 0}, {0, 1}, {10, 10}, {10, 11}}, 2]",
     Pred(lambda r: sorted(map(tuple, _flt(_rule(r, "centers")))) == [(0.0, 0.5), (10.0, 10.5)], "centres")),
    ("KMeans[{{0}, {1}, {2}}, 1]", Pred(lambda r: _flt(_rule(r, "centers")) == [[1.0]], "mean")),
    ("KMeans[{{0, 0}, {0, 1}, {10, 10}, {10, 11}}, 2]",
     Pred(lambda r: len(set(_flt(_rule(r, "labels"))[:2])) == 1 and _flt(_rule(r, "labels"))[0] != _flt(_rule(r, "labels"))[2], "labels")),
    ("KMeans[{{0}, {1}}, 3]", Raises(ValueError, "exceeds")), ("KMeans[{{0}, {1, 2}}, 1]", Raises(ValueError, "same length")),
    ("PCA[{{1, 2}, {2, 4}, {3, 6}}, 1]", Pred(lambda r: abs(_flt(_rule(r, "explainedVariance"))[0] - 1) < 1e-12, "rank one")),
    ("PCA[{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}, 2]", Pred(lambda r: all(abs(v - 0.5) < 1e-12 for v in _flt(_rule(r, "explainedVariance"))), "isotropic")),
    ("PCA[{{1, 2}, {2, 4}, {3, 6}}, 1]",
     Pred(lambda r: all(abs(abs(a[0]) - abs(b)) < 1e-9 for a, b in zip(_flt(_rule(r, "transformed")), [-math.sqrt(5), 0, math.sqrt(5)])), "scores")),
    ("PCA[{{1, 2}}, 3]", Raises(ValueError)),
    ("LogisticRegressionTrain[{{0}, {1}, {2}, {3}}, {0, 0, 1, 1}]", Pred(lambda r: float(_rule(r, "accuracy")) == 1.0, "separable")),
    ("LogisticRegressionTrain[{{0}, {1}, {2}, {3}}, {0, 0, 1, 1}]",
     Pred(lambda r: abs(-float(_rule(r, "intercept")) / _flt(_rule(r, "coefficients"))[0] - 1.5) < 1e-3, "boundary at 1.5")),
    ("LogisticRegressionTrain[{{0}, {1}, {2}, {3}}, {1, 1, 0, 0}]", Pred(lambda r: _flt(_rule(r, "coefficients"))[0] < 0, "negative slope")),
    ("LogisticRegressionTrain[{{0}, {1}, {2}}, {0, 1, 2}]", Raises(ValueError, "binary")),
    ("LogisticRegressionTrain[{{0}, {1}}, {0}]", Raises(ValueError, "labels")),
    ("DecisionTreeTrain[{{0}, {1}, {2}, {3}}, {0, 0, 1, 1}]", "{accuracy -> 1.0, depth -> 1}"),
    ("DecisionTreeTrain[{{0}, {1}, {2}, {3}}, {0, 1, 0, 1}]", "{accuracy -> 1.0, depth -> 3}"),
    ("DecisionTreeTrain[{{0}, {1}, {2}, {3}}, {0, 1, 0, 1}, 1]", Pred(lambda r: float(_rule(r, "accuracy")) == 0.75, "depth-1 stump")),
    ("DecisionTreeTrain[{{0}, {1}}, {0, 0}]", Raises(ValueError, "two classes")),
]
if not HAVE_SK:
    ML = [(f"{h}[{{{{0}}, {{1}}}}, {a}]", Raises(NotImplementedError, "scikit-learn"))
          for h, a in [("KMeans", "1"), ("PCA", "1"), ("LogisticRegressionTrain", "{0, 1}"), ("DecisionTreeTrain", "{0, 1}")]
          for _ in range(3)]

CASES = PLOT_OK + ML


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_plots_ml(case):
    check(case)
